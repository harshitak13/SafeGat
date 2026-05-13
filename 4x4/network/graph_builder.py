"""
Graph topology builder using the EXACT junction IDs and edges from 4x4.net.xml.

Controlled junctions (12 total, NOT 16):
  Row 0: J1  J6  J11 J16
  Row 1: J2  J7  J12 J17
  Row 2: J3  J8  J13 J18

Edge index encodes bidirectional N<->S and E<->W adjacency between these 12 nodes.
"""
import torch
import numpy as np
from typing import Dict, List

# net_config lives in the same package (network/)
from network.net_config import (
    CONTROLLED_TLS, TLS_INDEX, TLS_GRID_POS,
    TLS_NEIGHBOR_MAP, NUM_NODES, GRID_ROWS, GRID_COLS
)


def build_exact_edge_index() -> torch.Tensor:
    """
    Build COO edge_index for the 3x4 grid of 12 controlled junctions.
    Edges are bidirectional between each adjacent N-S and E-W pair.

    Returns:
        edge_index : LongTensor shape (2, num_edges)
    """
    edges = []
    for tls, neighbors in TLS_NEIGHBOR_MAP.items():
        src = TLS_INDEX[tls]
        for nb in neighbors:
            dst = TLS_INDEX[nb]
            edges.append([src, dst])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index  # (2, E)


def get_neighbor_tls_ids(tls_id: str) -> List[str]:
    """Return list of neighbouring junction IDs for a given TLS."""
    return TLS_NEIGHBOR_MAP.get(tls_id, [])


def attn_weights_to_dict(
    attn_edge_index: np.ndarray,   # (2, E)
    attn_weights: np.ndarray,      # (E,)
) -> Dict[int, Dict[int, float]]:
    """
    Map raw attention weight arrays -> nested dict:
        node_idx -> {neighbour_idx: weight}
    """
    result: Dict[int, Dict[int, float]] = {n: {} for n in range(NUM_NODES)}
    for (src, dst), w in zip(attn_edge_index.T, attn_weights):
        result[int(src)][int(dst)] = float(w)
    return result


def node_attn_by_tls_id(
    attn_edge_index: np.ndarray,
    attn_weights: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Same as attn_weights_to_dict but keyed by TLS string IDs.
    Returns: tls_id -> {neighbour_tls_id: attention_weight}
    """
    idx_to_tls = {i: tls for tls, i in TLS_INDEX.items()}
    raw = attn_weights_to_dict(attn_edge_index, attn_weights)
    result = {}
    for node_idx, nb_dict in raw.items():
        tls = idx_to_tls[node_idx]
        result[tls] = {idx_to_tls[nb]: w for nb, w in nb_dict.items()}
    return result


# Pre-built edge_index (constant — reuse to avoid repeated allocation)
EDGE_INDEX: torch.Tensor = build_exact_edge_index()
EDGE_INDEX_NP: np.ndarray = EDGE_INDEX.numpy()
