"""Live CityFlow environment wrapper for SafeGAT.

This module adapts CityFlow roadnet/flow datasets to the same simple API used
by the SUMO wrappers in ``4x4`` and ``7x28``:

    obs = env.reset()
    next_obs, rewards, done, infos = env.step(actions)

The wrapper builds graph metadata directly from roadnet JSON, controls traffic
lights through ``cityflow.Engine.set_tl_phase``, and derives compact 8-D node
observations from live lane counts and waiting counts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


OBS_DIM = 8


@dataclass
class RoadApproach:
    road_id: str
    lane_ids: List[str]
    capacity: float


@dataclass
class CityFlowNetwork:
    controlled_tls: List[str]
    neighbors: Dict[str, List[str]]
    incoming: Dict[str, List[RoadApproach]]
    legal_actions: Dict[str, List[int]]
    phase_counts: Dict[str, int]
    edge_index: List[Tuple[int, int]]
    num_actions: int


def _road_length(road: Dict[str, Any]) -> float:
    points = road.get("points") or []
    if len(points) < 2:
        return 300.0
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    return max(total, 1.0)


def _lane_ids(road: Dict[str, Any]) -> List[str]:
    road_id = road["id"]
    lanes = road.get("lanes") or [{}]
    return [f"{road_id}_{idx}" for idx in range(len(lanes))]


def load_cityflow_network(roadnet_path: Path) -> CityFlowNetwork:
    data = json.loads(roadnet_path.read_text(encoding="utf-8"))
    intersections = data.get("intersections", [])
    roads = data.get("roads", [])

    controlled = [item["id"] for item in intersections if not item.get("virtual", False)]
    controlled_set = set(controlled)
    index = {tls_id: idx for idx, tls_id in enumerate(controlled)}

    road_by_id = {road["id"]: road for road in roads}
    incoming: Dict[str, List[RoadApproach]] = {tls_id: [] for tls_id in controlled}
    neighbors: Dict[str, set[str]] = {tls_id: set() for tls_id in controlled}

    for road in roads:
        road_id = road.get("id")
        start = road.get("startIntersection")
        end = road.get("endIntersection")
        if not road_id:
            continue
        if end in controlled_set:
            lane_ids = _lane_ids(road)
            length = _road_length(road)
            # Approximate jam capacity using 7.5 m per vehicle.
            capacity = max(1.0, len(lane_ids) * length / 7.5)
            incoming[end].append(RoadApproach(road_id, lane_ids, capacity))
        if start in controlled_set and end in controlled_set:
            neighbors[start].add(end)
            neighbors[end].add(start)

    legal_actions: Dict[str, List[int]] = {}
    phase_counts: Dict[str, int] = {}
    max_action = 0
    for item in intersections:
        tls_id = item.get("id")
        if tls_id not in controlled_set:
            continue
        phases = item.get("trafficLight", {}).get("lightphases", [])
        phase_counts[tls_id] = max(1, len(phases))
        legal = [
            idx
            for idx, phase in enumerate(phases)
            if phase.get("availableRoadLinks")
        ]
        if not legal:
            legal = list(range(phase_counts[tls_id]))
        legal_actions[tls_id] = legal
        max_action = max(max_action, max(legal))

    edge_index: List[Tuple[int, int]] = []
    for src, nbs in neighbors.items():
        for dst in sorted(nbs):
            edge_index.append((index[src], index[dst]))

    if not edge_index:
        edge_index = [(idx, idx) for idx in range(len(controlled))]

    return CityFlowNetwork(
        controlled_tls=controlled,
        neighbors={key: sorted(value) for key, value in neighbors.items()},
        incoming=incoming,
        legal_actions=legal_actions,
        phase_counts=phase_counts,
        edge_index=edge_index,
        num_actions=max_action + 1,
    )


def edge_index_tensor(network: CityFlowNetwork, device: str = "cpu") -> torch.Tensor:
    edges = network.edge_index or [(idx, idx) for idx in range(len(network.controlled_tls))]
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def write_graph_metadata(network: CityFlowNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_nodes": len(network.controlled_tls),
        "num_edges": len(network.edge_index),
        "num_actions": network.num_actions,
        "controlled_tls": network.controlled_tls,
        "neighbors": network.neighbors,
        "legal_actions": network.legal_actions,
        "edge_index": network.edge_index,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CityFlowSignalEnv:
    """Live CityFlow traffic-signal control environment."""

    def __init__(
        self,
        dataset_dir: Path | str,
        roadnet: str,
        flow: str,
        output_dir: Path | str,
        episode_seconds: int = 1800,
        action_interval: int = 5,
        seed: int = 42,
        save_replay: bool = False,
        thread_num: int = 1,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.roadnet = roadnet
        self.flow = flow
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_seconds = int(episode_seconds)
        self.action_interval = max(1, int(action_interval))
        self.seed = int(seed)
        self.save_replay = bool(save_replay)
        self.thread_num = int(thread_num)

        self.network = load_cityflow_network(self.dataset_dir / self.roadnet)
        self.controlled_tls = self.network.controlled_tls
        self.num_nodes = len(self.controlled_tls)
        self.num_actions = self.network.num_actions
        self.step_count = 0
        self.elapsed_seconds = 0
        self._episode_idx = 0
        self._engine = None
        self._phases = {tls_id: self.network.legal_actions[tls_id][0] for tls_id in self.controlled_tls}
        self._runtimes = {tls_id: 0 for tls_id in self.controlled_tls}

    def reset(self) -> np.ndarray:
        self.close()
        self.step_count = 0
        self.elapsed_seconds = 0
        self._episode_idx += 1
        self._phases = {tls_id: self.network.legal_actions[tls_id][0] for tls_id in self.controlled_tls}
        self._runtimes = {tls_id: 0 for tls_id in self.controlled_tls}
        self._engine = self._make_engine()
        for tls_id, phase in self._phases.items():
            self._set_phase(tls_id, phase)
        return self._observe()

    def step(self, actions: Iterable[int]):
        if self._engine is None:
            raise RuntimeError("Call reset() before step().")

        safe_actions = self.legalize_actions(actions)
        for tls_id, action in zip(self.controlled_tls, safe_actions):
            self._set_phase(tls_id, int(action))

        for _ in range(self.action_interval):
            self._engine.next_step()
            self.elapsed_seconds += 1

        next_obs = self._observe()
        rewards = self._reward(next_obs)
        done = self.elapsed_seconds >= self.episode_seconds
        infos = self._infos(next_obs)
        self.step_count += 1
        return next_obs, rewards, done, infos

    def legalize_actions(self, actions: Iterable[int]) -> np.ndarray:
        """Return actions repaired to each intersection's legal phase set."""
        return self._repair_actions(actions)

    def close(self) -> None:
        self._engine = None

    @property
    def phases(self) -> Dict[str, int]:
        return dict(self._phases)

    @property
    def runtimes(self) -> Dict[str, int]:
        return dict(self._runtimes)

    def neighbor_summary(self, node_idx: int, rl_actions: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
        tls_id = self.controlled_tls[node_idx]
        tls_index = {tls: idx for idx, tls in enumerate(self.controlled_tls)}
        summary = {}
        for nb in self.network.neighbors.get(tls_id, []):
            nb_idx = tls_index.get(nb)
            if nb_idx is None:
                continue
            summary[nb] = {
                "mean_occ": round(float(obs[nb_idx, 2:6].mean()), 4),
                "rl_action": int(rl_actions[nb_idx]),
            }
        return summary

    def _make_engine(self):
        try:
            import cityflow
        except ImportError as exc:
            raise RuntimeError(
                "The CityFlow Python package is required for live simulation. "
                "Install CityFlow in this environment, then rerun the command."
            ) from exc

        cfg_path = self.output_dir / f"cityflow_config_ep{self._episode_idx}.json"
        replay_prefix = f"replay_ep{self._episode_idx}"
        cfg = {
            "interval": 1.0,
            "seed": self.seed + self._episode_idx,
            "dir": str(self.dataset_dir).replace("\\", "/") + "/",
            "roadnetFile": self.roadnet,
            "flowFile": self.flow,
            "rlTrafficLight": True,
            "saveReplay": self.save_replay,
            "roadnetLogFile": f"{replay_prefix}_roadnet.json",
            "replayLogFile": f"{replay_prefix}.txt",
        }
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        try:
            return cityflow.Engine(str(cfg_path), self.thread_num)
        except TypeError:
            return cityflow.Engine(str(cfg_path))

    def _set_phase(self, tls_id: str, action: int) -> None:
        action = self._legal_action(tls_id, action)
        old_phase = self._phases.get(tls_id)
        self._engine.set_tl_phase(tls_id, int(action))
        self._phases[tls_id] = int(action)
        self._runtimes[tls_id] = self._runtimes.get(tls_id, 0) + 1 if old_phase == action else 1

    def _repair_actions(self, actions: Iterable[int]) -> np.ndarray:
        arr = np.asarray(list(actions), dtype=int)
        if arr.size != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} actions, got {arr.size}")
        return np.asarray(
            [self._legal_action(tls_id, int(action)) for tls_id, action in zip(self.controlled_tls, arr)],
            dtype=np.int64,
        )

    def _legal_action(self, tls_id: str, action: int) -> int:
        legal = self.network.legal_actions[tls_id]
        if action in legal:
            return int(action)
        current = self._phases.get(tls_id)
        if current in legal:
            return int(current)
        return int(legal[0])

    def _lane_counts(self) -> tuple[Dict[str, float], Dict[str, float]]:
        count = self._engine.get_lane_vehicle_count()
        try:
            waiting = self._engine.get_lane_waiting_vehicle_count()
        except AttributeError:
            waiting = {}
        return count or {}, waiting or {}

    def _observe(self) -> np.ndarray:
        lane_count, lane_wait = self._lane_counts()
        rows = []
        max_phase_denom = max(1, self.num_actions - 1)
        for tls_id in self.controlled_tls:
            approaches = self.network.incoming.get(tls_id, [])
            occ_bins = np.zeros(4, dtype=np.float32)
            total_count = 0.0
            total_wait = 0.0
            total_capacity = 0.0
            for idx, approach in enumerate(approaches):
                approach_count = sum(float(lane_count.get(lane_id, 0.0)) for lane_id in approach.lane_ids)
                approach_wait = sum(float(lane_wait.get(lane_id, 0.0)) for lane_id in approach.lane_ids)
                capacity = max(1.0, approach.capacity)
                occ_bins[idx % 4] = max(occ_bins[idx % 4], min(1.0, approach_count / capacity))
                total_count += approach_count
                total_wait += approach_wait
                total_capacity += capacity
            total_capacity = max(1.0, total_capacity)
            phase = self._phases.get(tls_id, self.network.legal_actions[tls_id][0])
            rows.append([
                float(phase) / max_phase_denom,
                min(1.0, len(approaches) / 4.0),
                *occ_bins.tolist(),
                min(1.0, total_count / total_capacity),
                min(1.0, total_wait / total_capacity),
            ])
        return np.asarray(rows, dtype=np.float32)

    def _reward(self, obs: np.ndarray) -> np.ndarray:
        queue = obs[:, 6]
        waiting = obs[:, 7]
        mean_occ = obs[:, 2:6].mean(axis=1)
        return -(queue + 0.5 * waiting + 0.25 * mean_occ).astype(np.float32)

    def _infos(self, obs: np.ndarray) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for idx, tls_id in enumerate(self.controlled_tls):
            infos.append({
                "phase_runtime": int(self._runtimes.get(tls_id, 0)),
                "current_phase": int(self._phases.get(tls_id, 0)),
                "movement_occ": {f"approach_{i}": float(obs[idx, 2 + i]) for i in range(4)},
                "queue": float(obs[idx, 6]),
                "waiting": float(obs[idx, 7]),
                "sim_time_s": int(self.elapsed_seconds),
                "information_missing": False,
            })
        return infos
