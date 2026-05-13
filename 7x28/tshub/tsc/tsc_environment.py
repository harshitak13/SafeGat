"""
tshub/tsc/tsc_environment.py

Compatibility shim — tshub v1.3 has no tsc submodule.
Wraps tshub_env.TshubEnv + tshub.traffic_light to replicate the
TSCEnvironment interface expected by make_tsc_env.py and grid_env_wrapper.py.

Interface:
    env = TSCEnvironment(sumo_cfg, net_file, num_seconds, tls_id,
                         tls_action_type, use_gui, trip_info=None)
    obs_or_tuple = env.reset()     # returns (obs_dict, info_dict) or obs_dict
    result       = env.step(action) # returns (obs, reward, done, info)
                                    # or (obs, reward, terminated, truncated, info)
    env.close()
"""

import os
import traci
import sumolib
from loguru import logger
from typing import Optional, Dict, Any, Tuple


class TSCEnvironment:
    """
    Single-junction traffic signal control environment.
    Wraps a raw TraCI/SUMO connection directly since tshub v1.3
    does not expose a tsc submodule.
    """

    def __init__(
        self,
        sumo_cfg: str,
        net_file: str,
        num_seconds: int = 3600,
        tls_id: str = "J1",
        tls_action_type: str = "choose_next_phase",
        use_gui: bool = False,
        trip_info: Optional[str] = None,
        delta_time: int = 5,          # seconds per step
        yellow_time: int = 3,
        min_green: int = 5,
        max_green: int = 42,
        port_offset: int = 0,
    ):
        self.sumo_cfg        = sumo_cfg
        self.net_file        = net_file
        self.num_seconds     = num_seconds
        self.tls_id          = tls_id
        self.tls_action_type = tls_action_type
        self.use_gui         = use_gui
        self.trip_info       = trip_info
        self.delta_time      = delta_time
        self.yellow_time     = yellow_time
        self.min_green       = min_green
        self.max_green       = max_green

        self._sumo_binary    = "sumo-gui" if use_gui else "sumo"
        self._traci_conn     = None
        self._sim_step       = 0
        self._current_phase  = 0
        self._phase_elapsed  = 0.0
        self._num_phases     = 4       # phases 0-3

        # Derive a unique label so multiple envs can coexist
        self._label = f"tsc_{tls_id}"

        # Load net to find incoming edges for this TLS
        self._incoming_edges: list = []
        self._load_net_info()

    # ── net info ─────────────────────────────────────────────────────────────
    def _load_net_info(self):
        try:
            net = sumolib.net.readNet(self.net_file, withInternal=False)
            node = net.getNode(self.tls_id)
            if node:
                self._incoming_edges = [e.getID() for e in node.getIncoming()]
        except Exception as e:
            logger.warning(f"[TSCEnvironment-{self.tls_id}] net read error: {e}")

    # ── SUMO launch ──────────────────────────────────────────────────────────
    def _start_sumo(self):
        cmd = [
            self._sumo_binary,
            "-c", self.sumo_cfg,
            "--no-warnings",
            "--no-step-log",
            "--time-to-teleport", "-1",
        ]
        if self.trip_info:
            cmd += ["--tripinfo-output", self.trip_info]

        # Use a unique label to allow multiple parallel connections
        try:
            traci.start(cmd, label=self._label)
            self._traci_conn = traci.getConnection(self._label)
        except traci.exceptions.TraCIException:
            # Label already exists — close old then restart
            try:
                traci.getConnection(self._label).close()
            except Exception:
                pass
            traci.start(cmd, label=self._label)
            self._traci_conn = traci.getConnection(self._label)

    # ── observation helpers ───────────────────────────────────────────────────
    def _get_obs(self) -> Dict[str, Any]:
        try:
            phase   = self._traci_conn.trafficlight.getPhase(self.tls_id)
            elapsed = self._traci_conn.trafficlight.getPhaseDuration(self.tls_id)
        except Exception:
            phase, elapsed = 0, 0.0
        self._current_phase = phase
        self._phase_elapsed = elapsed
        return {
            "current_phase":     phase,
            "phase_elapsed_time": elapsed,
        }

    def _get_info(self) -> Dict[str, Any]:
        movement_occ      = {}
        jam_length_meters = {}
        rescue_ids        = []
        missing_ids       = []
        info_missing      = False

        for edge_id in self._incoming_edges:
            try:
                lanes = [
                    f"{edge_id}_{i}"
                    for i in range(self._traci_conn.edge.getLaneNumber(edge_id))
                ]
                occ  = max(
                    self._traci_conn.lane.getLastStepOccupancy(l) for l in lanes
                ) if lanes else 0.0
                jam  = max(
                    self._traci_conn.lane.getLastStepHaltingNumber(l) * 7.5
                    for l in lanes
                ) if lanes else 0.0
                movement_occ[edge_id]      = float(occ)
                jam_length_meters[edge_id] = float(jam)
            except Exception:
                movement_occ[edge_id]      = 0.5
                jam_length_meters[edge_id] = 0.0
                missing_ids.append(edge_id)
                info_missing = True

        return {
            "movement_occ":        movement_occ,
            "jam_length_meters":   jam_length_meters,
            "rescue_movement_ids": rescue_ids,
            "information_missing": info_missing,
            "missing_id":          missing_ids,
        }

    def _compute_reward(self, info: dict) -> float:
        jam_dict = info.get("jam_length_meters", {})
        if not jam_dict:
            return 0.0
        import numpy as np
        mean_jam = float(np.mean(list(jam_dict.values())))
        return -min(mean_jam / 100.0, 1.0)

    # ── action application ────────────────────────────────────────────────────
    def _apply_action(self, action: int):
        """Set the traffic light to the given phase index."""
        try:
            num_phases = len(
                self._traci_conn.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
            )
            phase = int(action) % num_phases
            self._traci_conn.trafficlight.setPhase(self.tls_id, phase)
        except Exception as e:
            logger.warning(f"[TSCEnvironment-{self.tls_id}] set phase error: {e}")

    # ── public API ────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[Dict, Dict]:
        if self._traci_conn is not None:
            try:
                self._traci_conn.close()
            except Exception:
                pass
            self._traci_conn = None

        self._start_sumo()
        self._sim_step = 0

        # Advance one step to get valid sensor readings
        self._traci_conn.simulationStep()
        self._sim_step += 1

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self._apply_action(action)

        for _ in range(self.delta_time):
            self._traci_conn.simulationStep()
            self._sim_step += 1

        obs    = self._get_obs()
        info   = self._get_info()
        reward = self._compute_reward(info)
        done   = self._sim_step >= self.num_seconds

        return obs, reward, done, info

    def close(self):
        if self._traci_conn is not None:
            try:
                self._traci_conn.close()
            except Exception:
                pass
            self._traci_conn = None
