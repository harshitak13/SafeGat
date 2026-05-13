"""
envs/grid_env_wrapper.py

Multi-junction SUMO environment wrapper.

make_grid_env() creates one TSCEnvironment per junction (via make_single_env_fn),
then exposes a unified step / reset interface that returns:

    obs    : np.ndarray  shape (NUM_NODES, OBS_DIM)   float32
    rewards: np.ndarray  shape (NUM_NODES,)             float32
    done   : bool
    infos  : list[dict]  length NUM_NODES

Observation per junction (OBS_DIM = 8):
    [0]   current phase index  (0–3), normalised by 3
    [1]   phase elapsed ratio  = elapsed_seconds / max_phase_duration
    [2–5] movement occupancies for the 4 incoming directions  (0.0–1.0)
    [6]   mean queue length, normalised by 100 m
    [7]   rescue/emergency vehicle present  (0 or 1)

Info dict per junction:
    movement_occ        : dict  edge_id -> occupancy float
    jam_length_meters   : dict  edge_id -> jam length float
    rescue_movement_ids : list  of edge IDs with emergency vehicles
    information_missing : bool
    missing_id          : list  of edge IDs with missing sensor data

Source: iLLM-TSC2 (envs/grid_env_wrapper.py), lightly cleaned.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from tshub.tsc.tsc_environment import TSCEnvironment
from network.net_config import (
    CONTROLLED_TLS, TLS_INCOMING_EDGES, NUM_NODES, NUM_ACTIONS
)

# ── Normalisation constants ────────────────────────────────────────────────────
_MAX_PHASE_DUR  = 42.0   # longest green phase duration (seconds)
_MAX_QUEUE_M    = 100.0  # normalisation cap for jam length (metres)
_NUM_DIRECTIONS = 4      # incoming edges sampled per junction


# ── Internal helpers ───────────────────────────────────────────────────────────

def _safe_get(d: dict, key, default=0.0):
    v = d.get(key)
    return default if v is None else v


def _build_obs_for_junction(
    tls_id: str,
    raw_obs: dict,
    raw_info: dict,
) -> np.ndarray:
    """
    Build an (OBS_DIM=8,) feature vector for one junction from the raw
    observation and info dicts returned by TSCEnvironment.step() / .reset().
    """
    incoming = TLS_INCOMING_EDGES[tls_id]

    phase_idx     = float(_safe_get(raw_obs, "current_phase", 0))
    phase_norm    = phase_idx / max(NUM_ACTIONS - 1, 1)

    elapsed       = float(_safe_get(raw_obs, "phase_elapsed_time", 0.0))
    elapsed_ratio = min(elapsed / _MAX_PHASE_DUR, 1.0)

    movement_occ = raw_info.get("movement_occ", {})
    occ_vals: List[float] = []
    for edge in incoming[:_NUM_DIRECTIONS]:
        occ_vals.append(float(_safe_get(movement_occ, edge, 0.5)))
    while len(occ_vals) < _NUM_DIRECTIONS:
        occ_vals.append(0.5)

    jam_dict      = raw_info.get("jam_length_meters", {})
    jam_vals      = [float(v) for v in jam_dict.values()] if jam_dict else [0.0]
    mean_jam_norm = min(float(np.mean(jam_vals)) / _MAX_QUEUE_M, 1.0)

    rescue_ids    = raw_info.get("rescue_movement_ids", [])
    rescue_flag   = 1.0 if rescue_ids else 0.0

    return np.array(
        [phase_norm, elapsed_ratio] + occ_vals + [mean_jam_norm, rescue_flag],
        dtype=np.float32,
    )


def _reward_from_info(raw_info: dict) -> float:
    """
    Reward = -(mean normalised queue length).
    Ranges in [-1, 0]; higher (closer to 0) is better.
    """
    jam_dict = raw_info.get("jam_length_meters", {})
    if not jam_dict:
        return 0.0
    mean_jam = float(np.mean([float(v) for v in jam_dict.values()]))
    return -min(mean_jam / _MAX_QUEUE_M, 1.0)


# ── GridEnv ────────────────────────────────────────────────────────────────────

class GridEnv:
    """
    Thin wrapper around N independent TSCEnvironment instances.

    Interface::

        obs  = env.reset()                    -> np.ndarray (N, OBS_DIM)
        obs, rewards, done, infos = env.step(actions)
                                               actions: np.ndarray (N,) int
        env.close()
    """

    def __init__(
        self,
        single_envs: List[TSCEnvironment],
        tls_ids: List[str],
        obs_dim: int = 8,
    ):
        assert len(single_envs) == len(tls_ids), "One env per TLS required"
        self.envs       = single_envs
        self.tls_ids    = tls_ids
        self.obs_dim    = obs_dim
        self._raw_infos = [{} for _ in tls_ids]
        self._dones     = [False] * len(tls_ids)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        obs_matrix  = np.zeros((len(self.tls_ids), self.obs_dim), dtype=np.float32)
        self._dones = [False] * len(self.tls_ids)
        for i, (tls, env) in enumerate(zip(self.tls_ids, self.envs)):
            try:
                raw = env.reset()
                if isinstance(raw, tuple):
                    raw_obs  = raw[0]
                    raw_info = raw[1] if len(raw) > 1 else {}
                else:
                    raw_obs, raw_info = raw, {}
                raw_info = raw_info if isinstance(raw_info, dict) else {}
                self._raw_infos[i] = raw_info
                obs_matrix[i] = _build_obs_for_junction(tls, raw_obs, raw_info)
            except Exception as exc:
                logger.warning(f"[GridEnv.reset] {tls}: {exc}")
                obs_matrix[i] = np.zeros(self.obs_dim, dtype=np.float32)
        return obs_matrix

    # ── step ──────────────────────────────────────────────────────────────────
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, List[dict]]:
        """
        actions : (N,) int array — one phase index per junction.

        Returns
        -------
        obs     : (N, obs_dim) float32
        rewards : (N,) float32
        done    : bool — True when ALL envs are done
        infos   : list of N info dicts
        """
        obs_matrix = np.zeros((len(self.tls_ids), self.obs_dim), dtype=np.float32)
        rewards    = np.zeros(len(self.tls_ids), dtype=np.float32)
        infos      = [{} for _ in self.tls_ids]

        for i, (tls, env) in enumerate(zip(self.tls_ids, self.envs)):
            if self._dones[i]:
                obs_matrix[i] = np.zeros(self.obs_dim, dtype=np.float32)
                continue
            try:
                result = env.step(int(actions[i]))

                if len(result) == 4:
                    raw_obs, reward, done, raw_info = result
                elif len(result) == 5:
                    raw_obs, reward, terminated, truncated, raw_info = result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected step return length {len(result)}")

                raw_info = raw_info if isinstance(raw_info, dict) else {}
                raw_info.setdefault("movement_occ",        {})
                raw_info.setdefault("jam_length_meters",   {})
                raw_info.setdefault("rescue_movement_ids", [])
                raw_info.setdefault("information_missing", False)
                raw_info.setdefault("missing_id",          [])

                self._raw_infos[i] = raw_info
                self._dones[i]     = bool(done)
                obs_matrix[i]      = _build_obs_for_junction(tls, raw_obs, raw_info)
                rewards[i]         = (float(reward)
                                      if reward is not None
                                      else _reward_from_info(raw_info))
                infos[i]           = raw_info

            except Exception as exc:
                logger.warning(f"[GridEnv.step] {tls}: {exc}")
                obs_matrix[i]  = np.zeros(self.obs_dim, dtype=np.float32)
                rewards[i]     = 0.0
                self._dones[i] = True

        done_global = all(self._dones)
        return obs_matrix, rewards, done_global, infos

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self):
        for tls, env in zip(self.tls_ids, self.envs):
            try:
                env.close()
            except Exception as exc:
                logger.warning(f"[GridEnv.close] {tls}: {exc}")


# ── Factory ────────────────────────────────────────────────────────────────────

def make_grid_env(
    make_single_env_fn: Callable,
    tls_ids: List[str],
    sumo_cfg: str,
    num_seconds: int = 3600,
    use_gui: bool = False,
    log_file: str = "./log/",
    obs_dim: int = 8,
    trip_info: Optional[str] = None,
) -> GridEnv:
    """
    Create a GridEnv wrapping one TSCEnvironment per junction.

    Parameters
    ----------
    make_single_env_fn : callable — typically ``make_env`` from utils/make_tsc_env.py
    tls_ids            : ordered list of junction IDs (must match CONTROLLED_TLS)
    sumo_cfg           : path to the .sumocfg file
    num_seconds        : episode length in simulation seconds
    use_gui            : open sumo-gui if True
    log_file           : directory for SUMO stdout logs
    obs_dim            : feature vector size per junction (default 8)
    trip_info          : optional XML path for SUMO tripinfo output
    """
    os.makedirs(log_file, exist_ok=True)
    single_envs = []

    for idx, tls_id in enumerate(tls_ids):
        this_trip = trip_info if (trip_info and idx == len(tls_ids) - 1) else None
        env = make_single_env_fn(
            tls_id      = tls_id,
            sumo_cfg    = sumo_cfg,
            num_seconds = num_seconds,
            use_gui     = use_gui and (idx == 0),
            log_file    = log_file,
            obs_dim     = obs_dim,
            trip_info   = this_trip,
        )
        single_envs.append(env)
        logger.debug(f"Created env for {tls_id}")

    logger.info(f"GridEnv ready — {len(single_envs)} junctions")
    return GridEnv(single_envs=single_envs, tls_ids=tls_ids, obs_dim=obs_dim)
