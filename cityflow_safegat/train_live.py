"""Train SafeGAT GAT-DQN on live CityFlow datasets."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from loguru import logger

from cityflow_safegat.live_env import CityFlowSignalEnv, edge_index_tensor, write_graph_metadata


OBS_DIM = 8
HIDDEN_DIM = 64


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _import_trainer(impl_dir: Path):
    sys.path.insert(0, str(impl_dir))
    from training.gat_dqn_trainer import FastGATDQNTrainer

    return FastGATDQNTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--impl-dir", required=True, help="SafeGAT implementation folder, e.g. 4x4 or 7x28")
    parser.add_argument("--output-dir", default="safegat_cityflow_live_training")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=int, default=1800)
    parser.add_argument("--action-interval", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--store-every", type=int, default=None)
    parser.add_argument("--update-every", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    impl_dir = Path(args.impl_dir).resolve()
    output_dir = dataset_dir / args.output_dir
    model_dir = dataset_dir / args.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    FastGATDQNTrainer = _import_trainer(impl_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = device == "cuda"

    args.episodes = _env_int("SAFEGAT_TRAIN_EPISODES", args.episodes)
    args.episode_seconds = _env_int("SAFEGAT_EPISODE_SECONDS", args.episode_seconds)
    args.action_interval = _env_int("SAFEGAT_ACTION_INTERVAL", args.action_interval)
    args.batch_size = _env_int("SAFEGAT_BATCH_SIZE", args.batch_size if has_cuda else min(args.batch_size, 16))
    args.warmup_steps = _env_int("SAFEGAT_WARMUP_STEPS", args.warmup_steps if has_cuda else min(args.warmup_steps, 300))
    args.store_every = _env_int("SAFEGAT_STORE_EVERY", args.store_every or max(1, args.action_interval))
    args.update_every = _env_int("SAFEGAT_UPDATE_EVERY", args.update_every or (1 if has_cuda else 20))
    args.progress_every = _env_int("SAFEGAT_PROGRESS_EVERY", args.progress_every)

    env = CityFlowSignalEnv(
        dataset_dir=dataset_dir,
        roadnet=args.roadnet,
        flow=args.flow,
        output_dir=output_dir,
        episode_seconds=args.episode_seconds,
        action_interval=args.action_interval,
        seed=args.seed,
        thread_num=args.thread_num,
    )
    atexit.register(env.close)
    edge_index = edge_index_tensor(env.network, device=device)
    write_graph_metadata(env.network, output_dir / "cityflow_graph.json")

    trainer = FastGATDQNTrainer(
        node_feature_dim=OBS_DIM,
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        hidden_dim=HIDDEN_DIM,
        gat_heads=args.gat_heads,
        top_k=args.top_k,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        warmup_steps=args.warmup_steps,
        buffer_capacity=args.buffer_capacity,
        device=device,
    )
    trainer.edge_index = edge_index

    logger.info(
        f"CityFlow SafeGAT training start | dataset={dataset_dir} | nodes={env.num_nodes} "
        f"| actions={env.num_actions} | episodes={args.episodes} | episode_seconds={args.episode_seconds} "
        f"| action_interval={args.action_interval} | batch={args.batch_size} | warmup={args.warmup_steps} "
        f"| store_every={args.store_every} | update_every={args.update_every} | device={device}"
    )

    curve = []
    for episode in range(1, args.episodes + 1):
        logger.info(f"Episode {episode}/{args.episodes} start | eps={trainer.epsilon:.4f}")
        obs = env.reset()
        done = False
        ep_reward = np.zeros(env.num_nodes, dtype=np.float32)
        losses = []
        steps = 0
        loss = None

        while not done:
            actions, _q_values, attn = trainer.select_actions(obs)
            safe_actions = env.legalize_actions(actions)
            next_obs, rewards, done, _infos = env.step(safe_actions)
            if steps % args.store_every == 0 or done:
                trainer.store_transition(
                    obs=obs,
                    actions=safe_actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=np.full(env.num_nodes, float(done), dtype=np.float32),
                    attn_weights=attn,
                    is_llm=False,
                )
            if steps % args.update_every == 0:
                loss = trainer.update()
                if loss is not None:
                    losses.append(float(loss))
            ep_reward += rewards
            obs = next_obs
            steps += 1
            if args.progress_every > 0 and steps % args.progress_every == 0:
                loss_str = "warmup" if loss is None else f"{loss:.4f}"
                logger.info(
                    f"  ep={episode:>3} step={steps:>4} | sim_s={env.elapsed_seconds:>5} "
                    f"| mean_rew={rewards.mean():.4f} | loss={loss_str} | eps={trainer.epsilon:.4f}"
                )

        row = {
            "episode": episode,
            "total_reward": float(ep_reward.sum()),
            "mean_reward": float(ep_reward.mean()),
            "steps": steps,
            "epsilon": float(trainer.epsilon),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "env_steps": int(trainer.total_steps),
        }
        curve.append(row)
        logger.info(
            f"Episode {episode:>3}/{args.episodes} | reward={row['total_reward']:.2f} "
            f"| mean={row['mean_reward']:.4f} | steps={steps} | eps={trainer.epsilon:.4f}"
        )

        if episode % args.checkpoint_freq == 0:
            ckpt = model_dir / f"gat_dqn_ep{episode}.pt"
            trainer.save(str(ckpt))
            logger.info(f"Checkpoint saved -> {ckpt}")

        (output_dir / "training_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

    final_path = model_dir / "gat_dqn_final.pt"
    trainer.save(str(final_path))
    env.close()
    logger.info(f"Training complete. Final model saved -> {final_path}")


if __name__ == "__main__":
    main()
