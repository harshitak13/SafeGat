"""
training/gat_dqn_trainer.py

FastGATDQNTrainer — vectorised batched DQN training over a graph.

Key design choices
------------------
1. Vectorised batch update:
   The naive approach fires one full graph forward pass per sample in the
   batch (64 separate calls).  Here we tile the graph B times using tiled
   edge-index offsets so a SINGLE forward pass processes all B graphs at
   once, reducing update time by ~30–50×.

2. Epsilon-greedy action selection with a shared online network.

3. Target network with periodic hard updates (every ``target_update_freq``
   gradient steps).

4. Gradient clipping to prevent exploding gradients.

5. Model save / load with full training-state resumption.

Sources
-------
- iLLM-TSC2 (train_grid.py FastGATDQNTrainer + training/gat_dqn_trainer.py)
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.gat_network import GATQNetwork


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Each transition stores:
        obs          (num_nodes, obs_dim)
        actions      (num_nodes,)
        rewards      (num_nodes,)
        next_obs     (num_nodes, obs_dim)
        dones        (num_nodes,)  — float, 1.0 if terminal
        attn_weights raw attention array (stored for logging; not used in update)
    """

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        attn_weights,
    ):
        self.buf.append((obs, actions, rewards, next_obs, dones, attn_weights))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones, attn = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),   # (B, N, obs_dim)
            np.array(actions,  dtype=np.int64),      # (B, N)
            np.array(rewards,  dtype=np.float32),    # (B, N)
            np.array(next_obs, dtype=np.float32),    # (B, N, obs_dim)
            np.array(dones,    dtype=np.float32),    # (B, N)
            attn,
        )

    def __len__(self) -> int:
        return len(self.buf)


# ── Trainer ────────────────────────────────────────────────────────────────────

class FastGATDQNTrainer:
    """
    DQN trainer with vectorised batched graph updates.

    Usage::

        trainer = FastGATDQNTrainer(node_feature_dim=8, num_nodes=12, num_actions=4)
        trainer.edge_index = EDGE_INDEX.to(trainer.device)

        # Inside training loop:
        actions, q_vals, attn = trainer.select_actions(obs)
        ...
        trainer.store_transition(obs, actions, rewards, next_obs, dones, attn)
        loss = trainer.update()

    Parameters
    ----------
    node_feature_dim    : int   — obs dimension per junction (default 8)
    num_nodes           : int   — number of controlled junctions (default 12)
    num_actions         : int   — number of discrete phase choices (default 4)
    hidden_dim          : int   — network hidden dimension (default 64)
    gat_heads           : int   — GAT multi-head attention count (default 4)
    lr                  : float — Adam learning rate (default 1e-3)
    gamma               : float — discount factor (default 0.95)
    epsilon_start       : float — initial exploration rate (default 1.0)
    epsilon_end         : float — minimum exploration rate (default 0.05)
    epsilon_decay_steps : int   — linear decay steps to epsilon_end (default 25 000)
    batch_size          : int   — replay batch size (default 64)
    target_update_freq  : int   — gradient steps between target net syncs (default 500)
    warmup_steps        : int   — minimum buffer size before updates start (default 500)
    buffer_capacity     : int   — replay buffer size (default 50 000)
    grad_clip           : float — max gradient norm (default 10.0)
    device              : str   — "cpu" or "cuda" (default "cpu")
    """

    def __init__(
        self,
        node_feature_dim: int,
        num_nodes: int = 12,
        num_actions: int = 4,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 25_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        warmup_steps: int = 500,
        buffer_capacity: int = 50_000,
        grad_clip: float = 10.0,
        device: str = "cpu",
    ):
        self.num_nodes   = num_nodes
        self.num_actions = num_actions
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps = warmup_steps
        self.grad_clip   = grad_clip
        self.device      = device

        # Epsilon-greedy schedule
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.total_steps   = 0
        self.updates_done  = 0

        # Networks
        self.online_net = GATQNetwork(
            node_feature_dim = node_feature_dim,
            hidden_dim       = hidden_dim,
            num_actions      = num_actions,
            gat_heads        = gat_heads,
        ).to(device)
        self.target_net = GATQNetwork(
            node_feature_dim = node_feature_dim,
            hidden_dim       = hidden_dim,
            num_actions      = num_actions,
            gat_heads        = gat_heads,
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        # Must be set externally before use:  trainer.edge_index = EDGE_INDEX.to(device)
        self.edge_index: Optional[torch.Tensor] = None

    # ── Private: tiled edge_index for batched graph ────────────────────────────

    def _batch_edge_index(self, B: int) -> torch.Tensor:
        """
        Tile edge_index B times with per-graph node offsets so all B graphs
        remain independent in a single batched forward pass.

        Returns shape (2, B * E).
        """
        N = self.num_nodes
        E = self.edge_index.shape[1]
        offsets = torch.arange(B, device=self.device).repeat_interleave(E) * N
        ei = self.edge_index.repeat(1, B) + offsets.unsqueeze(0)
        return ei

    def _batch_forward(
        self, net: nn.Module, obs_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        obs_batch : (B, num_nodes, obs_dim)
        Returns   : (B, num_nodes, num_actions)
        """
        B  = obs_batch.shape[0]
        x  = obs_batch.reshape(B * self.num_nodes, -1)
        ei = self._batch_edge_index(B)
        q_vals, _ = net(x, ei)
        return q_vals.reshape(B, self.num_nodes, self.num_actions)

    # ── Action selection ───────────────────────────────────────────────────────

    def select_actions(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        obs : (num_nodes, obs_dim) float32

        Returns
        -------
        actions      : (num_nodes,) int
        q_values     : (num_nodes, num_actions) float
        attn_weights : (E,) float — per-edge attention from last GAT layer
        """
        assert self.edge_index is not None, (
            "Set trainer.edge_index = EDGE_INDEX.to(device) before calling select_actions"
        )
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_vals, attn = self.online_net(x, self.edge_index)
        self.online_net.train()

        q_np    = q_vals.cpu().numpy()
        attn_np = attn.cpu().numpy().squeeze(-1)

        actions = np.array([
            random.randrange(self.num_actions)
            if random.random() < self.epsilon
            else int(np.argmax(q_np[i]))
            for i in range(self.num_nodes)
        ])

        self.total_steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return actions, q_np, attn_np

    # ── Replay buffer helpers ──────────────────────────────────────────────────

    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        attn_weights,
    ):
        """Push one transition into the replay buffer."""
        self.buffer.push(obs, actions, rewards, next_obs, dones, attn_weights)

    # ── Vectorised batch update ────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one DQN gradient step.

        Returns the scalar loss, or None if the buffer is still warming up.
        """
        if len(self.buffer) < self.warmup_steps:
            return None

        obs, actions, rewards, next_obs, dones, _ = self.buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        act_t      = torch.tensor(actions,  dtype=torch.long,    device=self.device)
        rew_t      = torch.tensor(rewards,  dtype=torch.float32, device=self.device)
        done_t     = torch.tensor(dones,    dtype=torch.float32, device=self.device)

        # Single batched forward pass for both online and target nets
        q_vals  = self._batch_forward(self.online_net, obs_t)       # (B, N, A)
        with torch.no_grad():
            q_next     = self._batch_forward(self.target_net, next_obs_t)
            max_q_next = q_next.max(dim=2).values                   # (B, N)

        # Q(s, a) for taken actions
        q_taken = q_vals.gather(2, act_t.unsqueeze(2)).squeeze(2)   # (B, N)

        # Bellman targets
        targets = rew_t + self.gamma * max_q_next * (1.0 - done_t) # (B, N)

        loss = nn.functional.mse_loss(q_taken, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates_done += 1
        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.detach())

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save full training state (networks + optimizer + schedule)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "total_steps":  self.total_steps,
            "updates_done": self.updates_done,
        }, path)

    def load(self, path: str):
        """Resume training state from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon      = ckpt.get("epsilon",      self.epsilon_end)
        self.total_steps  = ckpt.get("total_steps",  0)
        self.updates_done = ckpt.get("updates_done", 0)
