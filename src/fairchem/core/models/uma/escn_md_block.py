"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function
from typing_extensions import Literal

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    SeparableS2Activation_M,
)
from fairchem.core.models.uma.nn.layer_norm import (
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.models.uma.nn.radial import PolynomialEnvelope
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid


def set_mole_ac_start_index(module: nn.Module, index: int) -> None:
    for submodule in module.modules():
        if isinstance(submodule, MOLE):
            submodule.global_mole_tensors.ac_start_idx = index


def edge_envelope(edge_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    """(E,) smooth cutoff weight of each edge, 1 at r=0 and 0 at the cutoff."""
    return PolynomialEnvelope(exponent=5)(edge_distance / cutoff)


def smooth_degree(
    edge_distance: torch.Tensor,
    target_index: torch.Tensor,
    num_nodes: int,
    cutoff: float,
) -> torch.Tensor:
    """(N,) neighbour count sum_j envelope(r_ij / cutoff) of each target node.

    Smooth in the positions (each neighbour fades in at the cutoff), so normalizing
    by it keeps energies continuous when atoms cross the cutoff.
    """
    env = edge_envelope(edge_distance, cutoff)
    return torch.zeros(num_nodes, dtype=env.dtype, device=env.device).index_add(
        0, target_index, env
    )


class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        cutoff: float,
        # Enables activation checkpointing of edges in
        # activation_checkpoint_chunk_size size edge blocks
        activation_checkpoint_chunk_size: int | None,
        act_type: Literal["gate", "s2"] = "gate",
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            # NOTE: this is the only place where the SO3 grid of the edges (lmax/mmax) is used
            self.act = SeparableS2Activation_M(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        node_offset: int = 0,
        x_source_nodes: torch.Tensor | None = None,
    ):
        """``x_source_nodes``: features read on the neighbour (source) side of each
        edge instead of ``x``; the strictly local mode passes fixed species features."""
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                x_source_nodes=x_source_nodes,
            )
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner_and_M_mapping.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        # when chunking, we need to keep track of the start index of the chunk and give this information
        # to the mole layers
        ac_mole_start_idx = 0
        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                torch.utils.checkpoint.checkpoint(
                    self.forward_chunk,
                    x,
                    x_edge_partitions[idx],
                    edge_distance_parititons[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    node_offset,
                    ac_mole_start_idx,
                    x_source_nodes,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
        return torch.stack(new_embeddings).sum(axis=0)

    def forward_chunk(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        node_offset: int = 0,
        ac_mole_start_idx: int = 0,
        x_source_nodes: torch.Tensor | None = None,
    ):
        # here we need to update the ac_start_idx of the mole layers under here for this chunking to
        # work properly with MoLE together
        set_mole_ac_start_index(self, ac_mole_start_idx)

        source_nodes = x if x_source_nodes is None else x_source_nodes
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(x, dim=0)
            source_full = (
                x_full
                if x_source_nodes is None
                else gp_utils.gather_from_model_parallel_region_sum_grad(
                    source_nodes, dim=0
                )
            )
            x_source = source_full[edge_index[0]]
            x_target = x_full[edge_index[1]]
        else:
            x_source = source_nodes[edge_index[0]]
            x_target = x[edge_index[1]]

        x_message = torch.cat((x_source, x_target), dim=2)

        with record_function("SO2Conv"):
            # Rotate the irreps to align with the edge
            x_message = torch.bmm(wigner_and_M_mapping, x_message)

            # SO2 convolution
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)

            # M-prime...
            x_message = self.act(x_0_gating, x_message)

            x_message = self.so2_conv_2(x_message, x_edge)

            # envelope
            dist_scaled = edge_distance / self.cutoff
            env = self.envelope(dist_scaled)
            x_message = x_message * env.view(-1, 1, 1)

            # Rotate back the irreps
            x_message = torch.bmm(wigner_and_M_mapping_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
        # reset ac start index
        set_mole_ac_start_index(self, 0)
        return new_embedding


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        x = self.so3_linear_2(x)
        return x


class GridAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def forward(self, x):
        # Project to grid
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        # Perform point-wise operations
        x_grid = self.grid_mlp(x_grid)
        # Project back to spherical harmonic coefficients
        x = self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)
        return x


class eSCNMD_Block(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        if ff_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif ff_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        sys_node_embedding=None,
        node_offset: int = 0,
        x_source: torch.Tensor | None = None,
        node_scale: torch.Tensor | None = None,
    ):
        """``x_source``: fixed neighbour-side features (strictly local mode), normalized
        like ``x``. ``node_scale``: (N,) factor on each node's summed messages, e.g.
        1/sqrt(degree); None keeps the plain sum."""
        x_res = x
        x = self.norm_1(x)

        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        x_source_nodes = None
        if x_source is not None:
            x_source_nodes = self.norm_1(x_source)
            if sys_node_embedding is not None:
                x_source_nodes[:, 0, :] = x_source_nodes[:, 0, :] + sys_node_embedding

        with record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                x_source_nodes=x_source_nodes,
            )
            if node_scale is not None:
                x = x * node_scale.view(-1, 1, 1).to(x.dtype)
            x = x + x_res

        x_res = x
        x = self.norm_2(x)

        with record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res

        return x
