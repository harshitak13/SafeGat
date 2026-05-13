"""
training/gat_network.py

Graph Attention Q-Network for multi-junction traffic signal control.

Architecture
------------
    node_encoder  (Linear + ReLU)
        ↓
    GATConv-1     (4-head, concat=True)   → captures local neighbourhood
        ↓
    GATConv-2     (1-head, concat=False)  → returns interpretable attention weights
        ↓
    Q-head        (Linear + ReLU + Linear) → per-node Q-values

Attention weights from gat2 are returned alongside Q-values so the LLM
pipeline can use them as explainability context ("which neighbours most
influenced each junction's decision").

Source: iLLM-TSC2 (training/gat_network.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATQNetwork(nn.Module):
    """
    Shared GAT Q-network for all controlled TLS in the grid.

    Parameters
    ----------
    node_feature_dim : int   — flat obs dimension per junction (default 8)
    hidden_dim       : int   — internal embedding dimension (default 64)
    num_actions      : int   — number of discrete phases (default 4)
    gat_heads        : int   — multi-head attention in first GAT layer (default 4)
    dropout          : float — dropout probability in GAT layers (default 0.1)
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_actions: int = 4,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_actions = num_actions

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # GAT layer 1: multi-head, concat → hidden_dim * gat_heads output
        self.gat1 = GATConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim,
            heads        = gat_heads,
            dropout      = dropout,
            concat       = True,
        )

        # GAT layer 2: single head, no concat → hidden_dim; returns attention
        self.gat2 = GATConv(
            in_channels  = hidden_dim * gat_heads,
            out_channels = hidden_dim,
            heads        = 1,
            dropout      = dropout,
            concat       = False,
        )

        # Per-node Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,           # (num_nodes, node_feature_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        q_values     : Tensor (num_nodes, num_actions)
        attn_weights : Tensor (E, 1)  — per-edge attention from gat2
        """
        h = self.node_encoder(x)
        h = F.elu(self.gat1(h, edge_index))
        h, (_, attn) = self.gat2(h, edge_index, return_attention_weights=True)
        h = F.elu(h)
        self._last_attn_weights = attn.detach()
        return self.q_head(h), attn

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass."""
        return self._last_attn_weights
