"""
AlphaFold3 triangle self-attention on SO(3)-equivariant edge (pair) features.

The pair representation of AlphaFold is a dense N x N grid; here the pairs are the
directed edges of a batch of molecular graphs. A triangle (i, j, k) only involves
neighbours of its anchor i, so the edges are grouped by anchor atom and the
attention over the third atom runs on (N, M, M) blocks, M the largest neighbour
count, instead of a padded (B, N, N, N) grid: memory grows with N * M^2, linearly
in the atom count at a fixed density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from fairchem.core.models.uma.nn.layer_norm import get_normalization_layer
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear


@dataclass(frozen=True)
class AnchoredPairLayout:
    """The directed edges of a batch grouped by their anchor (source) atom.

    Edge e = (i, j) sits in a slot of anchor i; the slots of an anchor are its edges
    in target order, padded to M, the largest out-degree of the batch. Build it on
    ``edge_index.flip(0)`` to anchor on the targets instead (the ending-node variant).

    Attributes:
        slot_edge: (N, M) the edge in each slot of each anchor, -1 for padding.
        pair_edge: (N, M, M) the edge from the target of slot s to the target of
            slot t, -1 where there is none.
        edge_slot: (E,) flat position anchor * M + slot of each edge.
    """

    slot_edge: torch.Tensor
    pair_edge: torch.Tensor
    edge_slot: torch.Tensor

    @classmethod
    def from_graph(
        cls, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> AnchoredPairLayout:
        source, target = edge_index[0], edge_index[1]
        if not torch.equal(batch[source], batch[target]):
            raise ValueError("Edges must not connect atoms of different systems.")
        num_atoms, num_edges = batch.shape[0], source.shape[0]
        key = source * num_atoms + target
        sorted_key, order = torch.sort(key)
        if bool((sorted_key[1:] == sorted_key[:-1]).any()):
            raise ValueError(
                "Triangle attention needs at most one edge per directed atom pair; "
                "periodic images are not supported."
            )
        degree = torch.bincount(source, minlength=num_atoms)
        first = torch.cumsum(degree, 0) - degree
        rank = torch.empty_like(source)
        rank[order] = (
            torch.arange(num_edges, device=source.device) - first[source[order]]
        )
        num_slots = int(degree.max())

        slot_edge = source.new_full((num_atoms, num_slots), -1)
        slot_edge[source, rank] = torch.arange(num_edges, device=source.device)
        neighbour = source.new_full((num_atoms, num_slots), -1)
        neighbour[source, rank] = target

        # Look the (target of s, target of t) pairs up among the sorted edge keys.
        # A padded slot's -1 could alias a real key, so it is masked explicitly.
        query = neighbour.unsqueeze(2) * num_atoms + neighbour.unsqueeze(1)
        found_at = torch.searchsorted(sorted_key, query).clamp(max=num_edges - 1)
        valid = (neighbour.unsqueeze(2) >= 0) & (neighbour.unsqueeze(1) >= 0)
        found = valid & (sorted_key[found_at] == query)
        pair_edge = torch.where(found, order[found_at], -1)
        return cls(slot_edge, pair_edge, source * num_slots + rank)


def _gather_or_zero(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """values[index], zero where index is -1."""
    return torch.cat([values.new_zeros(1, *values.shape[1:]), values])[index + 1]


class EquivariantTriangleAttention(nn.Module):
    """Triangle self-attention around the starting node (AlphaFold3 Algorithm 14).

    Edge (i, j) attends over the edges (i, k) whose triangle closes, i.e. (j, k) is
    an edge too: query from (i, j), keys and values from (i, k), and a per-head bias
    from (j, k). On a layout of the flipped edges it becomes the ending-node variant
    (Algorithm 15): (i, j) attends over (k, j) with the bias from (k, i).

    Logits and gates are computed from the invariant l=0 channels and the values are
    the full spherical features, so the update is exactly SO(3)-equivariant. Edges
    without a closing triangle get a zero update. As in AlphaFold the output
    projection starts at zero, so the residual block starts as the identity.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        num_heads: int,
        key_channels: int,
        value_channels: int,
        norm_type: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.norm = get_normalization_layer(norm_type, lmax, channels)
        self.to_qk = nn.Linear(channels, 2 * num_heads * key_channels, bias=False)
        self.to_bias = nn.Linear(channels, num_heads, bias=False)
        self.to_gate = nn.Linear(channels, num_heads * value_channels)
        self.to_value = SO3_Linear(channels, num_heads * value_channels, lmax)
        self.to_out = SO3_Linear(num_heads * value_channels, channels, lmax)
        nn.init.zeros_(self.to_out.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, layout: AnchoredPairLayout) -> torch.Tensor:
        """(E, (lmax+1)^2, C) edge features in the global frame -> residual update."""
        num_edges, num_coefficients, _ = x.shape
        x = self.norm(x)
        scalars = x[:, 0, :]
        q, k = (
            self.to_qk(scalars)
            .view(num_edges, 2, self.num_heads, self.key_channels)
            .unbind(1)
        )
        v = self.to_value(x).view(
            num_edges, num_coefficients, self.num_heads, self.value_channels
        )
        v = v.transpose(1, 2).reshape(num_edges, self.num_heads, -1)

        # logits[a, h, s, t] = q_(a,s) . k_(a,t) / sqrt(c) + bias_(s,t), per anchor a
        q_slots = _gather_or_zero(q, layout.slot_edge).permute(0, 2, 1, 3)
        k_slots = _gather_or_zero(k, layout.slot_edge).permute(0, 2, 3, 1)
        logits = torch.matmul(q_slots, k_slots) / math.sqrt(self.key_channels)
        bias = _gather_or_zero(self.to_bias(scalars), layout.pair_edge)
        logits = logits + bias.permute(0, 3, 1, 2)
        # (i, k) and (j, k) both edges; (j, j) never is one, so k = j drops out
        closes = (layout.pair_edge >= 0).unsqueeze(1)
        logits = logits.float().masked_fill(~closes, torch.finfo(torch.float32).min)
        weights = self.dropout(torch.softmax(logits, dim=-1) * closes)

        v_slots = _gather_or_zero(v, layout.slot_edge).permute(0, 2, 1, 3)
        out = torch.matmul(weights.to(v.dtype), v_slots).transpose(1, 2)
        out = out.reshape(-1, self.num_heads, out.shape[-1])[layout.edge_slot]
        out = out.view(num_edges, self.num_heads, num_coefficients, self.value_channels)
        out = out.transpose(1, 2).reshape(num_edges, num_coefficients, -1)
        out = out * torch.sigmoid(self.to_gate(scalars)).unsqueeze(1)
        return self.to_out(out)
