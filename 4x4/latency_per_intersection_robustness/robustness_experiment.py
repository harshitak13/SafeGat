"""
robustness_experiment.py
========================
Injects sensor noise and traffic spikes mid-simulation and compares:
    - SafeGAT-iLLM  (GAT-DQN + LLM oversight)
    - Pure RL        (GAT-DQN only, no LLM)

For each regime the script runs three conditions back-to-back:
    baseline   — no perturbation
    noise      — Gaussian sensor noise injected from step INJECT_AT
    spike      — occupancy spiked to 1.0 on random junctions from step INJECT_AT

Output
------
    data/output/robustness/
        results.json           — raw per-step reward arrays for all runs
        robustness_summary.csv — per-condition mean/std reward + recovery delta
        robustness_plot.png    — side-by-side reward curves with injection marker

Run
---
    python experiments/robustness_experiment.py

Requires the same environment as run_safegat.py.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Project root is one directory up from this file ──────────────────────────
import sys
_HERE = Path(__file__).resolve().parent
# Auto-detect project root: works if script is in root OR in experiments/ subfolder
if (_HERE / 'run_safegat.py').exists():
    _ROOT = _HERE
elif (_HERE.parent / 'run_safegat.py').exists():
    _ROOT = _HERE.parent
else:
    _ROOT = _HERE
sys.path.insert(0, str(_ROOT))

from network.net_config    import CONTROLLED_TLS, NUM_ACTIONS, NUM_NODES
from network.graph_builder import EDGE_INDEX

from envs.grid_env_wrapper    import make_grid_env
from training.gat_dqn_trainer import FastGATDQNTrainer

from llm.action_refiner        import SafeGATRefiner
from llm.decision_logger       import DecisionLogger
from llm.intervention_gate     import InterventionGate
from llm.llm_gateway           import LLMGateway
from llm.safety_shield         import SafetyShield
from llm.scenario_detector     import ScenarioDetector
from llm.traffic_prompt_builder import TrafficPromptBuilder
from llm.types                 import RLDecisionInfo

from utils.make_tsc_env import make_env
from utils.readConfig   import read_config
from utils.margin       import compute_q_margins, select_uncertain_nodes

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_PATH    = _ROOT / "log"
MODEL_PATH  = _ROOT / "models" / "gat_dqn_final.pt"
OUT_DIR     = _ROOT / "data" / "output" / "robustness"
SUMO_CFG    = _ROOT / "network" / "4x4.sumocfg"
CONFIG_FILE = _ROOT / "configs" / "config.yaml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
OBS_DIM            = 8
HIDDEN_DIM         = 64
GAT_HEADS          = 4
SIM_SECONDS        = 600          # shorter episode for quick comparison
INJECT_AT          = 200          # step at which perturbation begins
NOISE_STD          = 0.25         # Gaussian std for sensor noise
SPIKE_FRACTION     = 0.5          # fraction of junctions to spike
Q_MARGIN_TAU       = 0.05
LLM_BUDGET         = 600
MAX_NODES_PER_STEP = 2
MIN_GREEN_STEPS    = 3

RANDOM_SEED        = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CONDITIONS = ["baseline", "noise", "spike"]
AGENTS     = ["safegat", "pure_rl"]


# ── Perturbation helpers ──────────────────────────────────────────────────────

def inject_noise(obs: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    """Add Gaussian noise to observation, clip to [0,1]."""
    noisy = obs + np.random.normal(0, std, obs.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def inject_spike(obs: np.ndarray, fraction: float = SPIKE_FRACTION) -> np.ndarray:
    """Set occupancy channels (2-5) and queue (6) to 1.0 on random junctions."""
    spiked = obs.copy()
    n_spike = max(1, int(NUM_NODES * fraction))
    targets = np.random.choice(NUM_NODES, n_spike, replace=False)
    for i in targets:
        spiked[i, 2:7] = 1.0   # occupancies + queue channel
    return spiked


def perturb(obs: np.ndarray, condition: str, step: int) -> np.ndarray:
    if step < INJECT_AT or condition == "baseline":
        return obs
    if condition == "noise":
        return inject_noise(obs)
    if condition == "spike":
        return inject_spike(obs)
    return obs


# ── Model factory ─────────────────────────────────────────────────────────────

def build_trainer() -> FastGATDQNTrainer:
    trainer = FastGATDQNTrainer(
        node_feature_dim    = OBS_DIM,
        num_nodes           = NUM_NODES,
        num_actions         = NUM_ACTIONS,
        hidden_dim          = HIDDEN_DIM,
        gat_heads           = GAT_HEADS,
        epsilon_start       = 0.0,   # fully greedy at inference
        epsilon_end         = 0.0,
        epsilon_decay_steps = 1,
    )
    trainer.edge_index = EDGE_INDEX.to(trainer.device)
    trainer.load(str(MODEL_PATH))
    trainer.online_net.eval()
    return trainer


def build_refiner(config: dict) -> Optional[SafeGATRefiner]:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        api_key  = config["OPENAI_API_KEY"]
        model    = config["OPENAI_API_MODEL"]
        base_url = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        chat     = ChatOpenAI(
            model=model, temperature=0.0,
            openai_api_key=api_key, openai_api_base=base_url,
            request_timeout=20,
        )
        backend = lambda p: chat.invoke([HumanMessage(content=p)]).content
    except Exception as e:
        print(f"[WARNING] LLM backend unavailable ({e}); SafeGAT will use mock responses.")
        backend = lambda p: json.dumps({"action": 0, "confidence": 0.5, "reasoning": "mock"})

    return SafeGATRefiner(
        detector       = ScenarioDetector(queue_spike_threshold=0.85,
                                          zero_fraction_corruption_threshold=0.90),
        gate           = InterventionGate(confidence_threshold=Q_MARGIN_TAU,
                                          intervention_budget=MAX_NODES_PER_STEP),
        prompt_builder = TrafficPromptBuilder(),
        llm_gateway    = LLMGateway(backend=backend, min_call_interval_s=4.0,
                                    max_backoff_retries=3, backoff_wait_s=20.0),
        safety_shield  = SafetyShield(min_green_hold=MIN_GREEN_STEPS),
        decision_logger= DecisionLogger(
            str(OUT_DIR / "llm_decisions_robustness.jsonl")),
    )


# ── Single-run episode ─────────────────────────────────────────────────────────

def run_episode(
    agent:     str,
    condition: str,
    config:    dict,
) -> Dict:
    """
    Run one episode and return per-step reward + metadata dict.
    """
    print(f"\n[RUN]  agent={agent:<10}  condition={condition:<10}", flush=True)

    trainer = build_trainer()
    refiner = build_refiner(config) if agent == "safegat" else None

    env = make_grid_env(
        make_single_env_fn = make_env,
        tls_ids            = CONTROLLED_TLS,
        sumo_cfg           = str(SUMO_CFG),
        num_seconds        = SIM_SECONDS,
        use_gui            = False,
        log_file           = str(LOG_PATH),
        obs_dim            = OBS_DIM,
    )

    obs   = env.reset()
    done  = False
    step  = 0

    phase_runtime = np.zeros(NUM_NODES, dtype=int)
    last_phase    = np.full(NUM_NODES, -1, dtype=int)
    infos         = [{} for _ in range(NUM_NODES)]

    step_rewards:    List[float] = []
    llm_calls_log:   List[int]   = []
    override_log:    List[int]   = []
    perturbation_on: List[bool]  = []
    llm_calls_total  = 0
    overrides_total  = 0
    budget_remaining = LLM_BUDGET

    while not done:
        # Apply perturbation before the agent sees the observation
        obs_in = perturb(obs, condition, step)
        perturbation_on.append(step >= INJECT_AT and condition != "baseline")

        rl_actions, q_values, attn_np = trainer.select_actions(obs_in)
        margins      = compute_q_margins(q_values)
        final_actions = rl_actions.copy()

        if agent == "safegat" and refiner is not None and budget_remaining > 0:
            anomaly_flags = np.array([
                bool(refiner.detector.detect(obs_in[i], infos[i])["tags"])
                for i in range(NUM_NODES)
            ])
            uncertain_nodes = select_uncertain_nodes(margins, anomaly_flags, Q_MARGIN_TAU)

            if uncertain_nodes:
                to_review = uncertain_nodes[:min(MAX_NODES_PER_STEP, budget_remaining)]
                budget_remaining -= len(to_review)

                for node_idx in to_review:
                    tls_id = CONTROLLED_TLS[node_idx]
                    rl_info = RLDecisionInfo(
                        intersection_id   = tls_id,
                        observation       = obs_in[node_idx].tolist(),
                        phase             = int(last_phase[node_idx]) if last_phase[node_idx] >= 0
                                            else int(round(float(obs_in[node_idx, 0]) * 3)),
                        rl_action         = int(rl_actions[node_idx]),
                        action_scores     = q_values[node_idx].tolist(),
                        confidence_margin = float(margins[node_idx]),
                        legal_actions     = list(range(NUM_ACTIONS)),
                        neighbor_summary  = {},
                        anomaly_tags      = [],
                        metadata          = {
                            "phase_runtime":     int(phase_runtime[node_idx]),
                            "emergency_vehicle": bool(obs_in[node_idx, 7] > 0.5),
                        },
                    )
                    try:
                        result = refiner.refine(rl_info)
                        llm_calls_total += 1
                        if result.final_action != int(rl_actions[node_idx]):
                            overrides_total += 1
                        final_actions[node_idx] = result.final_action
                    except Exception as exc:
                        print(f"  refine error @ {tls_id}: {exc!r}", flush=True)

        obs, rewards, done, infos = env.step(final_actions)

        # Update phase tracking
        for i in range(NUM_NODES):
            cur = int(round(float(obs[i, 0]) * 3))
            if cur == last_phase[i]:
                phase_runtime[i] += 1
            else:
                last_phase[i]    = cur
                phase_runtime[i] = 1

        step_rewards.append(float(rewards.mean()))
        llm_calls_log.append(llm_calls_total)
        override_log.append(overrides_total)
        step += 1

        if step % 50 == 0:
            print(f"  step={step:>4}  mean_rew={rewards.mean():.3f}  "
                  f"llm_calls={llm_calls_total}  overrides={overrides_total}",
                  flush=True)

    env.close()

    return {
        "agent":          agent,
        "condition":      condition,
        "step_rewards":   step_rewards,
        "llm_calls_log":  llm_calls_log,
        "override_log":   override_log,
        "perturbation_on": perturbation_on,
        "total_reward":   float(np.sum(step_rewards)),
        "mean_reward":    float(np.mean(step_rewards)),
        "post_inject_mean_reward": float(np.mean(step_rewards[INJECT_AT:])) if len(step_rewards) > INJECT_AT else 0.0,
        "llm_calls_total": llm_calls_total,
        "overrides_total": overrides_total,
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def compute_recovery_metric(rewards: List[float], inject_at: int, window: int = 50) -> float:
    """
    Recovery delta = mean reward in first `window` steps after injection
    minus mean reward in last `window` steps before injection.
    Positive = got better after injection (less negative reward).
    """
    if len(rewards) <= inject_at:
        return 0.0
    pre  = rewards[max(0, inject_at - window):inject_at]
    post = rewards[inject_at:inject_at + window]
    return float(np.mean(post)) - float(np.mean(pre))


def save_summary_csv(all_results: List[Dict], path: Path):
    import csv
    rows = []
    for r in all_results:
        recovery = compute_recovery_metric(r["step_rewards"], INJECT_AT)
        rows.append({
            "agent":                r["agent"],
            "condition":            r["condition"],
            "total_reward":         round(r["total_reward"], 3),
            "mean_reward":          round(r["mean_reward"], 4),
            "post_inject_mean_rew": round(r["post_inject_mean_reward"], 4),
            "recovery_delta":       round(recovery, 4),
            "llm_calls":            r["llm_calls_total"],
            "overrides":            r["overrides_total"],
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {path}")


def plot_results(all_results: List[Dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[SKIP] matplotlib not available — skipping plot.")
        return

    agent_color = {"safegat": "#2196F3", "pure_rl": "#F44336"}
    cond_ls     = {"baseline": "-", "noise": "--", "spike": ":"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        for r in [x for x in all_results if x["condition"] == cond]:
            agent  = r["agent"]
            rews   = r["step_rewards"]
            # smooth with rolling mean
            rews_s = np.convolve(rews, np.ones(20) / 20, mode="same")
            ax.plot(rews_s, color=agent_color[agent], linestyle=cond_ls[cond],
                    linewidth=1.5, label=agent)
        ax.axvline(INJECT_AT, color="gray", linestyle="--", linewidth=1,
                   label=f"Inject @ {INJECT_AT}")
        ax.set_title(cond.capitalize())
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean reward (smoothed)" if ax == axes[0] else "")
        ax.grid(alpha=0.3)

    # Shared legend
    handles = [
        mpatches.Patch(color=agent_color["safegat"], label="SafeGAT"),
        mpatches.Patch(color=agent_color["pure_rl"],  label="Pure RL"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Robustness: SafeGAT vs Pure RL under Perturbation", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = read_config(str(CONFIG_FILE))
    all_results: List[Dict] = []

    for condition in CONDITIONS:
        for agent in AGENTS:
            result = run_episode(agent=agent, condition=condition, config=config)
            all_results.append(result)
            # checkpoint after each run
            tmp = OUT_DIR / "results_partial.json"
            with open(tmp, "w") as f:
                json.dump(all_results, f, indent=2)

    # Save full results
    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] {results_path}")

    # CSV summary
    save_summary_csv(all_results, OUT_DIR / "robustness_summary.csv")

    # Plot
    plot_results(all_results, OUT_DIR / "robustness_plot.png")

    # Print table
    print("\n=== Robustness Summary ===")
    print(f"{'Agent':<12} {'Condition':<12} {'Mean Rew':>9} {'Post-Inject':>12} {'Recovery Δ':>11} {'LLM Calls':>10}")
    print("-" * 70)
    for r in all_results:
        rec = compute_recovery_metric(r["step_rewards"], INJECT_AT)
        print(f"{r['agent']:<12} {r['condition']:<12} "
              f"{r['mean_reward']:>9.4f} {r['post_inject_mean_reward']:>12.4f} "
              f"{rec:>11.4f} {r['llm_calls_total']:>10}")


if __name__ == "__main__":
    main()
