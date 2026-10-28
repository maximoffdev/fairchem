"""
AlphaFold3 triangle self-attention on SO(3)-equivariant edge (pair) features.

The pair representation of AlphaFold is a dense N x N grid; here the pairs are the
directed edges of a batch of molecular graphs, laid out on a padded per-system grid
so the attention over the third atom runs as batched matmuls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from fairchem.core.models.uma.nn.layer_norm import get_normalization_layer
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear


@dataclass(frozen=True)
class DensePairLayout:
    """Positions of the directed edges of a batch on a padded (B, N, N) pair grid.

    Attributes:
        system: (E,) system of each edge.
        source: (E,) source atom of each edge, indexed within its system.
        target: (E,) target atom of each edge, indexed within its system.
        exists: (B, N, N) True where [b, source, target] is an edge.
    """

    system: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    exists: torch.Tensor

    @classmethod
    def from_graph(
        cls, edge_index: torch.Tensor, batch: torch.Tensor, natoms: torch.Tensor
    ) -> DensePairLayout:
        system = batch[edge_index[0]]
        if not torch.equal(system, batch[edge_index[1]]):
            raise ValueError("Edges must not connect atoms of different systems.")
        node_ptr = torch.cumsum(natoms, dim=0) - natoms
        source = edge_index[0] - node_ptr[system]
        target = edge_index[1] - node_ptr[system]
        num_atoms = int(natoms.max())
        exists = torch.zeros(
            (natoms.numel(), num_atoms, num_atoms),
            dtype=torch.bool,
            device=edge_index.device,
        )
        exists[system, source, target] = True
        if int(exists.sum()) != edge_index.shape[1]:
            raise ValueError(
                "Triangle attention needs at most one edge per directed atom pair; "
                "periodic images are not supported."
            )
        return cls(system, source, target, exists)

    def transpose(self) -> DensePairLayout:
        """Swap source and target: pair (i, j) sits where (j, i) was."""
        return DensePairLayout(
            self.system, self.target, self.source, self.exists.transpose(1, 2)
        )

    def densify(self, x: torch.Tensor) -> torch.Tensor:
        """(E, ...) per-edge values -> (B, N, N, ...), zero where there is no edge."""
        dense = x.new_zeros(*self.exists.shape, *x.shape[1:])
        return dense.index_put((self.system, self.source, self.target), x)

    def gather(self, dense: torch.Tensor) -> torch.Tensor:
        """(B, N, N, ...) -> (E, ...) values at the edges."""
        return dense[self.system, self.source, self.target]


class EquivariantTriangleAttention(nn.Module):
    """Triangle self-attention around the starting node (AlphaFold3 Algorithm 14).

    Edge (i, j) attends over the edges (i, k) whose triangle closes, i.e. (j, k) is
    an edge too: query from (i, j), keys and values from (i, k), and a per-head bias
    from (j, k). Applied to ``layout.transpose()`` it becomes the ending-node variant
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

    def forward(self, x: torch.Tensor, layout: DensePairLayout) -> torch.Tensor:
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

        # logits[b, h, i, j, k] = q_ij . k_ik / sqrt(c) + bias_jk
        logits = torch.einsum(
            "bijhc,bikhc->bhijk", layout.densify(q), layout.densify(k)
        ) / math.sqrt(self.key_channels)
        bias = layout.densify(self.to_bias(scalars)).permute(0, 3, 1, 2)
        logits = logits + bias.unsqueeze(2)
        # (i, k) and (j, k) both edges
        closes = (layout.exists.unsqueeze(2) & layout.exists.unsqueeze(1)).unsqueeze(1)
        logits = logits.float().masked_fill(~closes, torch.finfo(torch.float32).min)
        weights = self.dropout(torch.softmax(logits, dim=-1) * closes)

        out = torch.einsum("bhijk,bikhd->bijhd", weights.to(v.dtype), layout.densify(v))
        out = layout.gather(out).view(
            num_edges, self.num_heads, num_coefficients, self.value_channels
        )
        out = out.transpose(1, 2).reshape(num_edges, num_coefficients, -1)
        out = out * torch.sigmoid(self.to_gate(scalars)).unsqueeze(1)
        return self.to_out(out)
