"""
utils/make_tsc_env.py

Provides a SharedSUMOConnection — a single SUMO process shared across ALL
196 junctions via one TraCI connection.

PERFORMANCE FIX (critical)
---------------------------
The original _get_incoming_lanes() called traci.trafficlight.getControlledLinks()
on EVERY step for EVERY junction. With 196 junctions this meant ~196 TraCI
round-trips per step just for lane lookups — the main bottleneck causing
~2 minutes per 300 steps.

Fix: lanes are fetched ONCE after connection and cached in self._lane_cache.
All subsequent calls to _get_incoming_lanes() are O(1) dict lookups.
get_reward() also uses the cache directly.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, List, Optional

import traci
from loguru import logger


# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_PORT    = 8813
_TRACI_TIMEOUT   = 60      # seconds to wait for SUMO to accept TraCI connection
_STEP_LENGTH     = 1.0     # simulation step in seconds


class SharedSUMOConnection:
    """
    Manages a single SUMO process shared by all 196 junctions.

    Parameters
    ----------
    sumo_cfg    : absolute path to the .sumocfg file
    num_seconds : episode duration in seconds
    use_gui     : launch sumo-gui instead of headless sumo
    log_file    : directory for SUMO stdout/stderr logs
    port        : TraCI port (default 8813; change if port is in use)
    """

    def __init__(
        self,
        sumo_cfg:    str,
        num_seconds: int  = 1800,
        use_gui:     bool = False,
        log_file:    str  = "./log/",
        port:        int  = _DEFAULT_PORT,
    ):
        self.sumo_cfg    = sumo_cfg
        self.num_seconds = num_seconds
        self.use_gui     = use_gui
        self.log_file    = log_file
        self.port        = port
        self._proc: Optional[subprocess.Popen] = None
        self._connected  = False
        self._first_reset = True   # skip traci.load() on first reset (sim just started)
        # KEY FIX: cache tls_id -> [lane, ...] after first connection
        self._lane_cache: Dict[str, List[str]] = {}

    # ── Launch / connect ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start SUMO and open the TraCI connection. Call once before reset()."""
        os.makedirs(self.log_file, exist_ok=True)
        self._launch_sumo()
        self._connect_traci()
        self._build_lane_cache()   # fetch all lane lists once, cache them

    def _launch_sumo(self) -> None:
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",            self.sumo_cfg,
            "--remote-port", str(self.port),
            "--step-length", str(_STEP_LENGTH),
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",          # disable teleporting
            "--collision.action", "warn",
        ]
        log_out = open(os.path.join(self.log_file, "sumo_stdout.log"), "w")
        log_err = open(os.path.join(self.log_file, "sumo_stderr.log"), "w")
        self._proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err)
        logger.info(f"SUMO launched (PID {self._proc.pid}) on port {self.port}")

    def _connect_traci(self) -> None:
        deadline = time.time() + _TRACI_TIMEOUT
        last_err = None
        while time.time() < deadline:
            try:
                traci.init(port=self.port, numRetries=0)
                self._connected = True
                logger.info(f"TraCI connected on port {self.port}")
                return
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        # If we get here, connection failed — kill SUMO and raise
        self._kill_proc()
        raise RuntimeError(
            f"Could not connect to SUMO on port {self.port} "
            f"after {_TRACI_TIMEOUT}s. Last error: {last_err}"
        )

    def _build_lane_cache(self) -> None:
        """
        Fetch controlled lanes for every TLS once and store in self._lane_cache.
        Called once after TraCI connects — eliminates per-step TraCI round-trips.
        """
        tls_ids = traci.trafficlight.getIDList()
        for tls_id in tls_ids:
            self._lane_cache[tls_id] = self._fetch_incoming_lanes(tls_id)
        logger.info(
            f"Lane cache built for {len(self._lane_cache)} traffic lights "
            f"(total lanes: {sum(len(v) for v in self._lane_cache.values())})"
        )

    def _fetch_incoming_lanes(self, tls_id: str) -> List[str]:
        """Fetch incoming lanes from TraCI (called once per TLS at startup)."""
        try:
            links = traci.trafficlight.getControlledLinks(tls_id)
            lanes = []
            seen  = set()
            for link_group in links:
                for link in link_group:
                    lane = link[0]   # incoming lane is index 0
                    if lane not in seen:
                        seen.add(lane)
                        lanes.append(lane)
            return lanes
        except traci.TraCIException:
            return []

    # ── Episode control ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reload the simulation to time=0.
        Skips traci.load() on the very first call (SUMO just launched).
        Lane cache is preserved across resets (topology doesn't change).
        """
        if not self._connected:
            self.start()
            return

        if self._first_reset:
            self._first_reset = False
            logger.debug("First reset — skipping traci.load() (sim already fresh)")
            return

        traci.load([
            "-c",            self.sumo_cfg,
            "--step-length", str(_STEP_LENGTH),
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
        ])
        logger.debug("SUMO simulation reloaded via traci.load()")

    def step_sim(self) -> None:
        """Advance the simulation by one step (1 second)."""
        traci.simulationStep()

    @property
    def sim_time(self) -> float:
        return traci.simulation.getTime()

    @property
    def is_done(self) -> bool:
        return self.sim_time >= self.num_seconds

    # ── Per-junction observation ───────────────────────────────────────────────

    def get_obs(self, tls_id: str) -> dict:
        """
        Return a raw observation dict for one junction using TraCI getters.
        Uses cached lane lists — no getControlledLinks() call per step.

        Keys
        ----
        current_phase       : int
        phase_elapsed_time  : float  (seconds)
        movement_occ        : dict   lane_id -> occupancy (0-1)
        jam_length_meters   : dict   lane_id -> jam length (m)
        rescue_movement_ids : list   (always empty — requires extra sensor)
        """
        try:
            phase   = traci.trafficlight.getPhase(tls_id)
            elapsed = traci.trafficlight.getPhaseDuration(tls_id) \
                      - traci.trafficlight.getNextSwitch(tls_id) \
                      + traci.simulation.getTime()
            elapsed = max(0.0, float(elapsed))
        except traci.TraCIException:
            phase, elapsed = 0, 0.0

        # O(1) cache lookup instead of TraCI round-trip
        controlled_lanes = self._get_incoming_lanes(tls_id)

        movement_occ      = {}
        jam_length_meters = {}
        for lane in controlled_lanes:
            try:
                movement_occ[lane]      = traci.lane.getLastStepOccupancy(lane)
                jam_length_meters[lane] = traci.lane.getLastStepHaltingNumber(lane) * 7.5
            except traci.TraCIException:
                movement_occ[lane]      = 0.0
                jam_length_meters[lane] = 0.0

        return {
            "current_phase":       phase,
            "phase_elapsed_time":  elapsed,
            "movement_occ":        movement_occ,
            "jam_length_meters":   jam_length_meters,
            "rescue_movement_ids": [],
            "information_missing": False,
            "missing_id":          [],
        }

    def _get_incoming_lanes(self, tls_id: str) -> List[str]:
        """Return cached incoming lanes for this TLS (O(1) lookup)."""
        return self._lane_cache.get(tls_id, [])

    # ── Phase control ──────────────────────────────────────────────────────────

    def set_phase(self, tls_id: str, phase_idx: int) -> None:
        """Set the traffic light phase for one junction."""
        try:
            traci.trafficlight.setPhase(tls_id, int(phase_idx))
        except traci.TraCIException as exc:
            logger.warning(f"set_phase({tls_id}, {phase_idx}): {exc}")

    # ── Reward ─────────────────────────────────────────────────────────────────

    def get_reward(self, tls_id: str) -> float:
        """
        Reward = -(mean normalised queue length) for incoming lanes.
        Range [-1, 0]; closer to 0 is better.
        Uses cached lane list — no extra TraCI calls for lane lookup.
        """
        lanes = self._get_incoming_lanes(tls_id)   # O(1) cache lookup
        if not lanes:
            return 0.0
        jams = []
        for lane in lanes:
            try:
                jams.append(traci.lane.getLastStepHaltingNumber(lane) * 7.5)
            except traci.TraCIException:
                jams.append(0.0)
        mean_jam = float(sum(jams) / len(jams)) if jams else 0.0
        return -min(mean_jam / 100.0, 1.0)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._connected:
            try:
                traci.close()
            except Exception:
                pass
            self._connected = False
        self._kill_proc()

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            logger.info("SUMO process terminated.")


def make_env(
    tls_id:      str,
    sumo_cfg:    str,
    num_seconds: int  = 1800,
    use_gui:     bool = False,
    log_file:    str  = "./log/",
    obs_dim:     int  = 8,
    trip_info:   str  = None,
) -> None:
    """
    Stub kept for API compatibility with make_grid_env().
    The real work is now done by SharedSUMOConnection inside GridEnv.
    Returns None — make_grid_env no longer uses per-junction env objects.
    """
    return None
