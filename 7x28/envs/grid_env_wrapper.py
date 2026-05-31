"""
envs/grid_env_wrapper.py

Multi-junction SUMO environment wrapper for the 7x28 network.

KEY CHANGE vs. original
------------------------
The original code spawned one SUMO process per junction (196 processes),
which caused the system to hang. This rewrite uses a SINGLE shared SUMO
process (SharedSUMOConnection) and drives all 196 junctions through one
TraCI connection.

make_grid_env() now builds a GridEnv backed by one SharedSUMOConnection.

Interface (unchanged from caller's perspective):
    obs               = env.reset()              -> np.ndarray (N, OBS_DIM)
    obs, rew, done, i = env.step(actions)        -> actions: np.ndarray (N,) int
    env.close()

Observation per junction (OBS_DIM = 8):
    [0]   current phase index  (0-3), normalised by 3
    [1]   phase elapsed ratio  = elapsed_seconds / max_phase_duration
    [2-5] movement occupancies for up to 4 incoming directions  (0.0-1.0)
    [6]   mean queue length, normalised by 100 m
    [7]   rescue/emergency vehicle present  (0 or 1)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from network.net_config import (
    CONTROLLED_TLS, TLS_INCOMING_EDGES, NUM_NODES, NUM_ACTIONS,
)
from utils.make_tsc_env import SharedSUMOConnection

# ── Normalisation constants ────────────────────────────────────────────────────
_MAX_PHASE_DUR  = 42.0   # longest green phase duration (seconds)
_MAX_QUEUE_M    = 100.0  # normalisation cap for jam length (metres)
_NUM_DIRECTIONS = 4      # incoming edges sampled per junction


def _is_closed_connection(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection closed" in text or "connection already closed" in text


# ── Observation builder ────────────────────────────────────────────────────────

def _build_obs_for_junction(
    tls_id:   str,
    raw_info: dict,
) -> np.ndarray:
    """
    Build an (OBS_DIM=8,) feature vector for one junction from the raw
    observation dict returned by SharedSUMOConnection.get_obs().
    """
    incoming = TLS_INCOMING_EDGES.get(tls_id, [])

    phase_idx  = float(raw_info.get("current_phase", 0))
    phase_norm = phase_idx / max(NUM_ACTIONS - 1, 1)

    elapsed       = float(raw_info.get("phase_elapsed_time", 0.0))
    elapsed_ratio = min(elapsed / _MAX_PHASE_DUR, 1.0)

    movement_occ = raw_info.get("movement_occ", {})
    occ_vals: List[float] = []
    for edge in incoming[:_NUM_DIRECTIONS]:
        occ_vals.append(float(movement_occ.get(edge, 0.5)))
    while len(occ_vals) < _NUM_DIRECTIONS:
        occ_vals.append(0.5)

    jam_dict      = raw_info.get("jam_length_meters", {})
    jam_vals      = [float(v) for v in jam_dict.values()] if jam_dict else [0.0]
    mean_jam_norm = min(float(np.mean(jam_vals)) / _MAX_QUEUE_M, 1.0)

    rescue_ids  = raw_info.get("rescue_movement_ids", [])
    rescue_flag = 1.0 if rescue_ids else 0.0

    return np.array(
        [phase_norm, elapsed_ratio] + occ_vals + [mean_jam_norm, rescue_flag],
        dtype=np.float32,
    )


# ── GridEnv ────────────────────────────────────────────────────────────────────

class GridEnv:
    """
    Unified environment for all 196 junctions backed by ONE SUMO process.

    Parameters
    ----------
    sumo_conn : SharedSUMOConnection — already started
    tls_ids   : ordered list of junction IDs
    obs_dim   : feature vector length per junction (default 8)
    """

    def __init__(
        self,
        sumo_conn: SharedSUMOConnection,
        tls_ids:   List[str],
        obs_dim:   int = 8,
    ):
        self.conn    = sumo_conn
        self.tls_ids = tls_ids
        self.obs_dim = obs_dim
        self.n       = len(tls_ids)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """Reload the SUMO simulation and return initial observations."""
        self.conn.reset()
        # Advance one step so TraCI getters return valid values
        self.conn.step_sim()
        return self._collect_obs()

    # ── step ──────────────────────────────────────────────────────────────────
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, List[dict]]:
        """
        Apply actions to all junctions, advance simulation by 1 step,
        collect observations and rewards.

        Parameters
        ----------
        actions : (N,) int array — phase index per junction

        Returns
        -------
        obs     : (N, obs_dim) float32
        rewards : (N,) float32
        done    : bool
        infos   : list of N dicts (raw TraCI obs per junction)
        """
        # 1. Set phases for all junctions
        for i, tls_id in enumerate(self.tls_ids):
            try:
                self.conn.set_phase(tls_id, int(actions[i]))
            except Exception as exc:
                if _is_closed_connection(exc):
                    logger.warning(f"[GridEnv.step] SUMO connection closed while setting {tls_id}: {exc}")
                    return self._terminal_step()
                raise

        # 2. Advance simulation by one second
        try:
            self.conn.step_sim()
        except Exception as exc:
            if _is_closed_connection(exc):
                logger.warning(f"[GridEnv.step] SUMO connection closed during simulationStep: {exc}")
                return self._terminal_step()
            raise

        # 3. Collect observations and rewards
        try:
            obs     = self._collect_obs(raise_on_closed=True)
            rewards = np.array(
                [self.conn.get_reward(tls_id) for tls_id in self.tls_ids],
                dtype=np.float32,
            )
            done  = self.conn.is_done
            infos = [self.conn.get_obs(tls_id) for tls_id in self.tls_ids]
        except Exception as exc:
            if _is_closed_connection(exc):
                logger.warning(f"[GridEnv.step] SUMO connection closed while collecting step data: {exc}")
                return self._terminal_step()
            raise

        return obs, rewards, done, infos

    def _terminal_step(self) -> Tuple[np.ndarray, np.ndarray, bool, List[dict]]:
        obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n, dtype=np.float32)
        infos = [{} for _ in self.tls_ids]
        return obs, rewards, True, infos

    def _collect_obs(self, raise_on_closed: bool = False) -> np.ndarray:
        obs_matrix = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        for i, tls_id in enumerate(self.tls_ids):
            try:
                raw = self.conn.get_obs(tls_id)
                obs_matrix[i] = _build_obs_for_junction(tls_id, raw)
            except Exception as exc:
                if raise_on_closed and _is_closed_connection(exc):
                    raise
                logger.warning(f"[GridEnv._collect_obs] {tls_id}: {exc}")
                obs_matrix[i] = np.zeros(self.obs_dim, dtype=np.float32)
        return obs_matrix

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self):
        self.conn.close()


# ── Factory ────────────────────────────────────────────────────────────────────

def make_grid_env(
    make_single_env_fn: Callable,   # kept for API compatibility, not used
    tls_ids:     List[str],
    sumo_cfg:    str,
    num_seconds: int  = 1800,
    use_gui:     bool = False,
    log_file:    str  = "./log/",
    obs_dim:     int  = 8,
    trip_info:   Optional[str] = None,
) -> GridEnv:
    """
    Create a GridEnv backed by a single shared SUMO process.

    Parameters
    ----------
    make_single_env_fn : ignored (kept for API compatibility with train.py)
    tls_ids            : ordered list of junction IDs
    sumo_cfg           : path to the .sumocfg file
    num_seconds        : episode length in simulation seconds
    use_gui            : open sumo-gui if True
    log_file           : directory for SUMO logs
    obs_dim            : feature vector size per junction (default 8)
    trip_info          : unused (kept for API compatibility)
    """
    os.makedirs(log_file, exist_ok=True)

    logger.info(
        f"Starting single shared SUMO process for {len(tls_ids)} junctions..."
    )

    conn = SharedSUMOConnection(
        sumo_cfg    = sumo_cfg,
        num_seconds = num_seconds,
        use_gui     = use_gui,
        log_file    = log_file,
        port        = int(os.environ["SAFEGAT_SUMO_PORT"]) if os.environ.get("SAFEGAT_SUMO_PORT") else None,
    )
    conn.start()   # launches SUMO + opens TraCI — fast, single process

    missing = sorted(set(tls_ids) - conn.traffic_light_ids())
    if missing:
        raise RuntimeError(
            f"{len(missing)} configured TLS IDs are not present in the loaded SUMO network. "
            f"Sample: {missing[:10]}. Check SUMO_CFG and CONTROLLED_TLS."
        )

    # Log a sample of junction IDs (not all 196)
    logger.debug(
        f"Controlled TLS sample: {tls_ids[:5]}...{tls_ids[-5:]}"
    )
    logger.info(f"GridEnv ready — {len(tls_ids)} junctions via single SUMO process")

    return GridEnv(sumo_conn=conn, tls_ids=tls_ids, obs_dim=obs_dim)
