"""
training/gat_network.py

Graph Attention Q-Network for multi-junction traffic signal control.

Architecture
------------
    node_encoder  (Linear + ReLU)
        ↓
    SparsePeerGating-1  (Stage-1 score gate + Stage-2 causal message passing)
        ↓
    SparsePeerGating-2  (single output head, returns causal edge weights)
        ↓
    Q-head        (Linear + ReLU + Linear) → per-node Q-values

Fix: Attention Diffusion at Scale (GTQN, Scientific Reports 2026)
------------------------------------------------------------------
The original SafeGAT used dense softmax GATConv layers which spread
attention uniformly across ALL neighbours at scale, polluting Q-values
with irrelevant spatial correlations.

This module replaces those layers with GTQN's two-stage sparse peer gating:

  Stage 1 — Score gate:
      For every candidate edge (i→j), a learned scorer f_gate produces a
      scalar logit.  A hard top-K (or threshold) mask is applied PER NODE
      so only the K most causally-relevant neighbours survive.  All other
      edges are zeroed out before message passing, preventing irrelevant
      spatial correlations from polluting Q-values.

  Stage 2 — Causal message passing:
      Only the gated (sparse) edge set is used for aggregation.  Attention
      weights over the surviving edges are computed with a standard dot-
      product attention and softmax, so the weights remain interpretable and
      comparable with the original pipeline's LLM explainability context.

Causal edge weights from layer-2 are returned alongside Q-values so the
LLM pipeline can use them as explainability context.

Sources
-------
- GTQN (Scientific Reports 2026) — two-stage sparse peer gating
- iLLM-TSC2 (training/gat_network.py) — original SafeGAT architecture
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


# ── Two-stage Sparse Peer Gating layer ────────────────────────────────────────

class SparsePeerGating(nn.Module):
    """
    GTQN-style two-stage sparse peer gating layer.

    Stage 1 — Gating:
        A learned MLP scorer assigns each candidate edge (i→j) a scalar
        logit from the concatenation [h_i ‖ h_j].  Per-node top-K selection
        keeps only the K edges with highest scores.  Edges outside the top-K
        are masked to −∞ before softmax, producing exactly-zero attention on
        irrelevant neighbours.

    Stage 2 — Causal message passing:
        Dot-product attention weights are computed over the surviving sparse
        edge set.  The weighted sum of neighbour embeddings is the output.

    Parameters
    ----------
    in_channels  : int   — input feature dimension per node
    out_channels : int   — output feature dimension per node
    heads        : int   — number of parallel attention heads
    top_k        : int   — maximum causal neighbours per node (Stage-1 budget)
    dropout      : float — dropout on attention weights (training only)
    concat       : bool  — if True, concatenate head outputs; else average
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        heads:        int   = 4,
        top_k:        int   = 4,
        dropout:      float = 0.1,
        concat:       bool  = True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.top_k        = top_k
        self.dropout      = dropout
        self.concat       = concat

        # Stage 1: per-edge scorer (shared across heads for efficiency)
        # Input: [h_i ‖ h_j]  →  scalar logit
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )

        # Stage 2: per-head value projection (source nodes)
        self.W_val = nn.Linear(in_channels, out_channels * heads, bias=False)

        # Stage 2: per-head attention query/key projections
        self.W_q = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels * heads, bias=False)

        self.bias = nn.Parameter(torch.zeros(
            out_channels * heads if concat else out_channels
        ))

        self._scale = out_channels ** -0.5

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,  # (N, in_channels)
        edge_index: torch.Tensor,  # (2, E)
        return_attention_weights: bool = False,
    ):
        """
        Parameters
        ----------
        x                        : (N, in_channels)
        edge_index               : (2, E) — source, target indices
        return_attention_weights : if True, also return (edge_index, attn)

        Returns
        -------
        out  : (N, out_channels*heads) if concat else (N, out_channels)
        [optionally] (edge_index_sparse, attn_weights)
        """
        N = x.size(0)
        src, dst = edge_index  # both (E,)
        E = src.size(0)

        # ── Stage 1: Score gate ───────────────────────────────────────────────
        # Concatenate source and destination embeddings for every candidate edge
        edge_feats  = torch.cat([x[src], x[dst]], dim=-1)      # (E, 2*in)
        gate_logits = self.edge_scorer(edge_feats).squeeze(-1)  # (E,)

        # Per-node top-K selection: zero-out all but the top-K incoming edges
        # We operate on *destination* nodes (each node selects its best sources)
        mask = torch.full((E,), float('-inf'), device=x.device)

        # Group edges by destination node and pick top-K per group
        # Using scatter: for each dst, find top-K src logits
        for node_id in range(N):
            node_edges = (dst == node_id).nonzero(as_tuple=True)[0]
            if node_edges.numel() == 0:
                continue
            k = min(self.top_k, node_edges.numel())
            node_logits = gate_logits[node_edges]
            topk_local  = torch.topk(node_logits, k=k, largest=True).indices
            selected    = node_edges[topk_local]
            mask[selected] = gate_logits[selected]  # restore logit for survivors

        # mask is -inf for pruned edges, logit value for survivors
        # We'll use it as an additive mask on attention logits in Stage 2

        # ── Stage 2: Causal message passing ──────────────────────────────────
        H = self.heads
        D = self.out_channels

        # Project to Q, K, V per head
        q = self.W_q(x[dst]).view(-1, H, D)   # (E, H, D) — destination queries
        k = self.W_k(x[src]).view(-1, H, D)   # (E, H, D) — source keys
        v = self.W_val(x[src]).view(-1, H, D) # (E, H, D) — source values

        # Dot-product attention scores (scale by sqrt(D))
        attn = (q * k).sum(dim=-1) * self._scale  # (E, H)

        # Add Stage-1 gate mask (pruned edges → -inf → zero softmax weight)
        attn = attn + mask.unsqueeze(-1)           # (E, H)

        # Softmax per (destination node, head) over surviving edges only
        # pyg_softmax: softmax over edges grouped by index vector
        attn = pyg_softmax(attn, dst, num_nodes=N) # (E, H)

        # Dropout on attention weights during training
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Weighted sum of values → aggregate to destination nodes
        # attn: (E, H), v: (E, H, D) → weighted: (E, H, D)
        weighted = attn.unsqueeze(-1) * v   # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device)
        dst_expanded = dst.view(-1, 1, 1).expand(-1, H, D)
        out.scatter_add_(0, dst_expanded, weighted)  # (N, H, D)

        if self.concat:
            out = out.reshape(N, H * D)
        else:
            out = out.mean(dim=1)               # (N, D)

        out = out + self.bias

        if return_attention_weights:
            # Return mean attention across heads for interpretability
            attn_mean = attn.mean(dim=-1, keepdim=True)  # (E, 1)
            return out, (edge_index, attn_mean)

        return out


# ── GATQNetwork using SparsePeerGating ────────────────────────────────────────

class GATQNetwork(nn.Module):
    """
    Shared GAT Q-network for all controlled TLS in the grid.

    Replaces dense softmax GATConv with GTQN two-stage sparse peer gating
    to prevent attention diffusion at scale (GTQN, Scientific Reports 2026).

    Parameters
    ----------
    node_feature_dim : int   — flat obs dimension per junction (default 8)
    hidden_dim       : int   — internal embedding dimension (default 64)
    num_actions      : int   — number of discrete phases (default 4)
    gat_heads        : int   — multi-head attention in first layer (default 4)
    dropout          : float — dropout probability in gating layers (default 0.1)
    top_k            : int   — max causal neighbours per node per layer (default 4)
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim:       int   = 64,
        num_actions:      int   = 4,
        gat_heads:        int   = 4,
        dropout:          float = 0.1,
        top_k:            int   = 4,
    ):
        super().__init__()
        self.num_actions = num_actions

        # Node feature encoder (unchanged from original SafeGAT)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # Layer 1: multi-head, concat → hidden_dim * gat_heads output
        # Replaces: GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=True)
        self.gat1 = SparsePeerGating(
            in_channels  = hidden_dim,
            out_channels = hidden_dim,
            heads        = gat_heads,
            top_k        = top_k,
            dropout      = dropout,
            concat       = True,
        )

        # Layer 2: single head, no concat → hidden_dim; returns causal weights
        # Replaces: GATConv(hidden_dim*gat_heads, hidden_dim, heads=1, concat=False)
        self.gat2 = SparsePeerGating(
            in_channels  = hidden_dim * gat_heads,
            out_channels = hidden_dim,
            heads        = 1,
            top_k        = top_k,
            dropout      = dropout,
            concat       = False,
        )

        # Per-node Q-value head (unchanged)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x:          torch.Tensor,  # (num_nodes, node_feature_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        q_values     : Tensor (num_nodes, num_actions)
        attn_weights : Tensor (E, 1)  — per-edge causal weight from layer 2
        """
        h = self.node_encoder(x)

        # Stage-1 sparse gating: only genuinely causal neighbours survive
        h = F.elu(self.gat1(h, edge_index))

        # Stage-2 causal message passing with interpretable edge weights
        h, (_, attn) = self.gat2(h, edge_index, return_attention_weights=True)
        h = F.elu(h)

        self._last_attn_weights = attn.detach()
        return self.q_head(h), attn

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return causal attention weights from the last forward pass."""
        return self._last_attn_weights
