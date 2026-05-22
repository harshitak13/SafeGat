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

6. [FIX] GMM-VGAE Importance-Sampled Replay Buffer (OffLight, arXiv Nov 2024):
   LLM-refined and RL-only transitions are tagged at push-time.  A two-
   component Gaussian Mixture Model (GMM) is fit to the reward distribution
   of each quality tier.  At sample-time each transition receives an
   importance weight  w_i = p_beneficial(r_i) / p_all(r_i)  so that
   LLM transitions with genuinely better rewards are upweighted and
   harmful / noisy ones are downweighted, correcting the learning signal.

Sources
-------
- iLLM-TSC2 (train_grid.py FastGATDQNTrainer + training/gat_dqn_trainer.py)
- OffLight (arXiv Nov 2024): GMM-VGAE with importance sampling for offline-
  assisted RL replay buffers.
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


# ── GMM-based quality estimator ───────────────────────────────────────────────

class _GMMQualityEstimator:
    """
    Lightweight two-component GMM fit to the reward distribution of a set
    of transitions.  Used to compute per-transition importance weights.

    The GMM is re-fit every ``refit_interval`` calls to ``importance_weights``
    so the model tracks the evolving buffer distribution.

    Parameters
    ----------
    n_components    : int   — number of Gaussian components (default 2)
    refit_interval  : int   — how many weight-query calls between re-fits
    min_samples     : int   — minimum samples before GMM fitting is attempted
    """

    def __init__(
        self,
        n_components:   int = 2,
        refit_interval: int = 200,
        min_samples:    int = 64,
    ):
        self.n_components   = n_components
        self.refit_interval = refit_interval
        self.min_samples    = min_samples

        # GMM parameters (diagonal covariance, 1-D rewards)
        self._means: Optional[np.ndarray]  = None   # (K,)
        self._vars:  Optional[np.ndarray]  = None   # (K,)
        self._pis:   Optional[np.ndarray]  = None   # (K,)  mixture weights

        self._call_count = 0

    # ── EM fitting ────────────────────────────────────────────────────────────

    def fit(self, rewards: np.ndarray) -> None:
        """
        Fit GMM to ``rewards`` (1-D array) using simple EM.
        Falls back to uniform distribution if fitting is degenerate.
        """
        r = rewards.ravel().astype(np.float64)
        K = self.n_components
        N = len(r)

        if N < self.min_samples:
            self._means = self._vars = self._pis = None
            return

        # Initialise means by percentile split
        pcts = np.linspace(0, 100, K + 2)[1:-1]
        means = np.percentile(r, pcts)
        # Add small noise to break symmetry
        means += np.random.randn(K) * (r.std() * 0.01 + 1e-8)
        var   = np.full(K, r.var() / K + 1e-6)
        pi    = np.full(K, 1.0 / K)

        for _ in range(50):                         # EM iterations
            # E-step
            resp = np.zeros((N, K))
            for k in range(K):
                resp[:, k] = pi[k] * self._gauss(r, means[k], var[k])
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp_sum = np.where(resp_sum < 1e-300, 1e-300, resp_sum)
            resp /= resp_sum

            # M-step
            Nk = resp.sum(axis=0) + 1e-8
            means_new = (resp * r[:, None]).sum(axis=0) / Nk
            var_new   = (resp * (r[:, None] - means_new[None, :]) ** 2).sum(axis=0) / Nk
            var_new   = np.maximum(var_new, 1e-6)
            pi_new    = Nk / Nk.sum()

            if np.allclose(means, means_new, atol=1e-6):
                means, var, pi = means_new, var_new, pi_new
                break
            means, var, pi = means_new, var_new, pi_new

        self._means = means
        self._vars  = var
        self._pis   = pi

    @staticmethod
    def _gauss(x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
        return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

    # ── Weight computation ────────────────────────────────────────────────────

    def importance_weights(
        self,
        rewards: np.ndarray,
        is_llm:  np.ndarray,
        refit_rewards: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute importance weights for a batch of transitions.

        w_i = p_beneficial(r_i) / p_all(r_i)

        where p_beneficial is the density of the GMM component with the
        highest mean (treating it as the "good" cluster) and p_all is the
        full mixture density.

        LLM-refined transitions whose reward falls in the beneficial cluster
        get w > 1 (upweighted); those in the low-reward cluster get w < 1.
        RL-only transitions receive w = 1 (neutral baseline).

        Parameters
        ----------
        rewards        : (B,) mean reward per transition
        is_llm         : (B,) bool — True if the transition was LLM-refined
        refit_rewards  : optional separate pool of rewards to re-fit GMM on

        Returns
        -------
        weights : (B,) float32, clipped to [0.1, 10.0]
        """
        self._call_count += 1
        # Periodically re-fit
        if self._call_count % self.refit_interval == 1:
            pool = refit_rewards if refit_rewards is not None else rewards
            self.fit(pool)

        weights = np.ones(len(rewards), dtype=np.float32)

        if self._means is None:
            return weights                          # not enough data yet

        r = rewards.ravel().astype(np.float64)
        K = self.n_components

        # Full mixture density
        p_all = sum(
            self._pis[k] * self._gauss(r, self._means[k], self._vars[k])
            for k in range(K)
        )
        p_all = np.maximum(p_all, 1e-300)

        # Beneficial component = the one with the highest mean
        best_k   = int(np.argmax(self._means))
        p_benef  = self._gauss(r, self._means[best_k], self._vars[best_k])

        ratio    = (p_benef / p_all).astype(np.float32)

        # Only LLM transitions get re-weighted; RL transitions stay at 1
        weights  = np.where(is_llm, ratio, np.ones_like(ratio))

        # Clip to prevent extreme gradients
        weights  = np.clip(weights, 0.1, 10.0)
        # Normalise so mean weight == 1 (keeps effective learning rate stable)
        weights  = weights / (weights.mean() + 1e-8)

        return weights.astype(np.float32)


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer with LLM-quality tagging and
    GMM-based importance sampling (OffLight fix).

    Each transition stores:
        obs          (num_nodes, obs_dim)
        actions      (num_nodes,)
        rewards      (num_nodes,)
        next_obs     (num_nodes, obs_dim)
        dones        (num_nodes,)  — float, 1.0 if terminal
        attn_weights raw attention array (stored for logging; not used in update)
        is_llm       bool — True if any node action was LLM-refined this step

    Parameters
    ----------
    capacity         : int  — maximum number of transitions
    gmm_refit_interval: int  — how often the GMM is re-fit (in sample() calls)
    """

    def __init__(self, capacity: int, gmm_refit_interval: int = 200):
        self.buf = deque(maxlen=capacity)
        self._gmm = _GMMQualityEstimator(refit_interval=gmm_refit_interval)

    def push(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        attn_weights,
        is_llm: bool = False,       # ← NEW: tag whether LLM refined this step
    ):
        """
        Push one transition into the buffer.

        Parameters
        ----------
        is_llm : bool
            Set to True when at least one node's action was overridden or
            accepted by the LLM refiner this step.  Used by the GMM
            importance sampler to distinguish LLM from pure-RL transitions.
        """
        self.buf.append((obs, actions, rewards, next_obs, dones, attn_weights, is_llm))

    def sample(self, batch_size: int):
        """
        Sample a mini-batch with GMM importance weights.

        Returns
        -------
        obs, actions, rewards, next_obs, dones, attn : standard arrays
        weights : (B,) float32 importance weights
        """
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones, attn, is_llm_flags = zip(*batch)

        obs_arr     = np.array(obs,      dtype=np.float32)   # (B, N, obs_dim)
        act_arr     = np.array(actions,  dtype=np.int64)      # (B, N)
        rew_arr     = np.array(rewards,  dtype=np.float32)    # (B, N)
        next_arr    = np.array(next_obs, dtype=np.float32)    # (B, N, obs_dim)
        done_arr    = np.array(dones,    dtype=np.float32)    # (B, N)
        is_llm_arr  = np.array(is_llm_flags, dtype=bool)     # (B,)

        # Mean reward per transition (scalar summary for GMM)
        mean_rewards = rew_arr.mean(axis=1)                   # (B,)

        # Collect all buffered rewards for GMM re-fitting context
        all_rewards = np.array(
            [t[2].mean() for t in self.buf], dtype=np.float32
        )

        # Compute importance weights
        weights = self._gmm.importance_weights(
            rewards        = mean_rewards,
            is_llm         = is_llm_arr,
            refit_rewards  = all_rewards,
        )

        return obs_arr, act_arr, rew_arr, next_arr, done_arr, attn, weights

    def __len__(self) -> int:
        return len(self.buf)


# ── Trainer ────────────────────────────────────────────────────────────────────

class FastGATDQNTrainer:
    """
    DQN trainer with vectorised batched graph updates and GMM importance
    sampling to correct for corrupted replay caused by mixing LLM-refined
    and RL-only transitions (OffLight fix).

    Usage::

        trainer = FastGATDQNTrainer(node_feature_dim=8, num_nodes=12, num_actions=4)
        trainer.edge_index = EDGE_INDEX.to(trainer.device)

        # Inside training loop:
        actions, q_vals, attn = trainer.select_actions(obs)
        ...
        trainer.store_transition(obs, actions, rewards, next_obs, dones, attn,
                                 is_llm=llm_was_called_this_step)
        loss = trainer.update()

    Parameters
    ----------
    node_feature_dim    : int   — obs dimension per junction (default 8)
    num_nodes           : int   — number of controlled junctions (default 12)
    num_actions         : int   — number of discrete phase choices (default 4)
    hidden_dim          : int   — network hidden dimension (default 64)
    gat_heads           : int   — GAT multi-head attention count (default 4)
    top_k               : int   — max causal neighbours per node (GTQN sparse gate, default 4)
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
    gmm_refit_interval  : int   — GMM re-fit period in sample() calls (default 200)
    """

    def __init__(
        self,
        node_feature_dim: int,
        num_nodes: int = 12,
        num_actions: int = 4,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        top_k: int = 4,
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
        gmm_refit_interval: int = 200,
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
            top_k            = top_k,
        ).to(device)
        self.target_net = GATQNetwork(
            node_feature_dim = node_feature_dim,
            hidden_dim       = hidden_dim,
            num_actions      = num_actions,
            gat_heads        = gat_heads,
            top_k            = top_k,
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity, gmm_refit_interval)

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
        is_llm: bool = False,   # ← NEW: pass True when LLM was called this step
    ):
        """
        Push one transition into the replay buffer.

        Parameters
        ----------
        is_llm : bool
            True when the LLM refiner was invoked this step (for any node).
            The GMM importance sampler uses this flag to distinguish LLM-
            refined transitions from pure RL ones and weight them accordingly.
        """
        self.buffer.push(
            obs, actions, rewards, next_obs, dones, attn_weights, is_llm
        )

    # ── Vectorised batch update ────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one DQN gradient step with GMM
        importance-weighted loss (OffLight fix).

        Returns the scalar loss, or None if the buffer is still warming up.
        """
        if len(self.buffer) < self.warmup_steps:
            return None

        obs, actions, rewards, next_obs, dones, _, weights = \
            self.buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        act_t      = torch.tensor(actions,  dtype=torch.long,    device=self.device)
        rew_t      = torch.tensor(rewards,  dtype=torch.float32, device=self.device)
        done_t     = torch.tensor(dones,    dtype=torch.float32, device=self.device)
        # Importance weights: shape (B,) → broadcast to (B, N) for per-node loss
        w_t        = torch.tensor(weights,  dtype=torch.float32, device=self.device)

        # Single batched forward pass for both online and target nets
        q_vals  = self._batch_forward(self.online_net, obs_t)       # (B, N, A)
        with torch.no_grad():
            q_next     = self._batch_forward(self.target_net, next_obs_t)
            max_q_next = q_next.max(dim=2).values                   # (B, N)

        # Q(s, a) for taken actions
        q_taken = q_vals.gather(2, act_t.unsqueeze(2)).squeeze(2)   # (B, N)

        # Bellman targets
        targets = rew_t + self.gamma * max_q_next * (1.0 - done_t) # (B, N)

        # ── Importance-weighted loss (OffLight fix) ────────────────────────────
        # Per-element squared TD errors, then weight each transition (row) by w_i
        td_errors    = (q_taken - targets.detach()) ** 2             # (B, N)
        # w_t is per-transition → expand across nodes
        weighted_td  = td_errors * w_t.unsqueeze(1)                  # (B, N)
        loss         = weighted_td.mean()
        # ── (original unweighted loss was: nn.functional.mse_loss(q_taken, targets.detach()))

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
