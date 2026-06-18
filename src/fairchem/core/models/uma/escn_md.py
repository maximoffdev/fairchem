"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch_geometric.utils
from omegaconf import DictConfig, ListConfig
from torch.distributed.nn.functional import all_reduce as all_reduce_with_grad
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.common.quaternion.quaternion_wigner_utils import (
    create_wigner_data_module,
)
from fairchem.core.models.uma.common.quaternion.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
)
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from fairchem.core.models.uma.nn.embedding import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.execution_backends import (
    get_execution_backend,
)
from fairchem.core.models.uma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole_utils import MOLEInterface
from fairchem.core.models.uma.nn.radial import (
    GaussianSmearing,
    PolynomialEnvelope,
    RadialMLP,
)
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
from fairchem.core.models.uma.nn.so2_tp import SO2_Convolution_TensorProduct
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    compute_hessian,
    get_l_component_range,
    reduce_node_to_system,
)
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum
from fairchem.core.units.mlip_unit.api.inference import (
    validate_uma_atoms_data,
)
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task

from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE = 1024 * 128
AUTO_EDGE_CHUNK_FRACTION = 0.05


@dataclass
class GradRegressConfig:
    """
    Configuration for gradient-based computation of forces and stress.
    """

    direct_forces: bool = False
    direct_stress: bool = False
    forces: bool = False
    stress: bool = False
    hessian: bool = False
    hessian_vmap: bool = True


def add_n_empty_edges(
    graph_dict: dict, edges_to_add: int, cutoff: float, node_offset: int = 0
):
    graph_dict["edge_index"] = torch.cat(
        (
            graph_dict["edge_index"].new_ones(2, edges_to_add) * node_offset,
            graph_dict["edge_index"],
        ),
        dim=1,
    )

    self_edge_distance_vec = graph_dict["edge_distance_vec"].new_ones(1, 3) + cutoff
    graph_dict["edge_distance_vec"] = torch.cat(
        (
            self_edge_distance_vec.expand(edges_to_add, 3),
            graph_dict["edge_distance_vec"],
        ),
        dim=0,
    )

    edge_distance = torch.linalg.norm(self_edge_distance_vec, dim=-1, keepdim=False)
    graph_dict["edge_distance"] = torch.cat(
        (edge_distance.expand(edges_to_add), graph_dict["edge_distance"]), dim=0
    )


def validate_contiguous_channels(channels: list[int], name: str) -> tuple[int, int]:
    """Validate channels are contiguous, return (start, end) slice indices.

    Args:
        channels: List of channel indices to validate
        name: Name of the channel list for error messages

    Returns:
        Tuple of (start_idx, end_idx) for slicing. Returns (0, 0) if channels is empty.

    Raises:
        ValueError: If channels are not contiguous
    """
    if not channels:
        return 0, 0
    sorted_channels = sorted(channels)
    expected = list(range(sorted_channels[0], sorted_channels[-1] + 1))
    if sorted_channels != expected:
        raise ValueError(f"{name} must be contiguous (e.g., [0, 1, 2]). Got {channels}")
    return sorted_channels[0], sorted_channels[-1] + 1


def balance_channels_batched(
    emb: torch.Tensor,
    target: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    target_offset: float = 0.0,
) -> torch.Tensor:
    """Balance a contiguous range of channels to target sum per system.

    This batched version processes all channels in a contiguous range in a single
    call, which is more efficient than processing each channel individually.

    Args:
        emb: Node embeddings of shape [num_atoms, sph_features, channels]
        target: Target sum per system of shape [num_systems]
        natoms: Number of atoms per system of shape [num_systems]
        batch: Batch indices mapping atoms to systems of shape [num_atoms]
        start_idx: Start index of channel range (inclusive)
        end_idx: End index of channel range (exclusive)
        target_offset: Offset to subtract from target (e.g., 1.0 for spin)

    Returns:
        Modified embeddings with the specified channel range balanced to sum to target.

    Supports graph parallel (GP) mode using torch.distributed.nn.functional.all_reduce
    which provides correct gradients in both forward and backward passes.
    """
    out_emb = emb.clone()
    num_systems = len(natoms)
    n_channels = end_idx - start_idx

    # Batched extraction: [num_atoms, n_channels]
    channels_to_balance = emb[:, 0, start_idx:end_idx]

    # Batched sum: [num_systems, n_channels]
    system_sums = torch.zeros(
        num_systems, n_channels, device=emb.device, dtype=emb.dtype
    )
    system_sums.index_add_(0, batch, channels_to_balance)

    # Reduce partial sums across all graph parallel ranks
    if gp_utils.initialized():
        system_sums = all_reduce_with_grad(system_sums, group=gp_utils.get_gp_group())

    # Batched correction: broadcast target to all channels
    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    corrections = (system_sums - target_sums) / natoms.unsqueeze(1)

    out_emb[:, 0, start_idx:end_idx] = channels_to_balance - corrections[batch]
    return out_emb


def resolve_dataset_mapping(
    deprecated_list: list[str] | None,
    dataset_mapping: dict[str, str] | None,
    deprecated_param_name: str = "dataset_list",
) -> dict[str, str]:
    """
    Validate and resolve dataset mapping from either a deprecated list or a mapping dict.

    Args:
        deprecated_list: Deprecated list of dataset names. If provided, it is
            converted to a mapping where each name maps to itself.
        dataset_mapping: Mapping from the config dataset name to desired dataset name for embeddings and heads.
            Allows multiple subsets to share the same dataset embedding and/or output head by mapping
            them to the same identifier.
        deprecated_param_name: Name of the deprecated parameter, used in
            warning/error messages.

    Returns:
        The resolved dataset mapping dict.

    Raises:
        ValueError: If both or neither arguments are provided, if the mapping
            is not a non-empty dict, or if mapping values are not a subset of
            mapping keys.
    """
    if deprecated_list is not None and dataset_mapping is not None:
        msg = (
            f"Both '{deprecated_param_name}' (={deprecated_list}) and "
            f"'dataset_mapping' (={dataset_mapping}) have been provided. "
            f"Please provide 'dataset_mapping' only in the config as '{deprecated_param_name}' is deprecated."
        )
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is None and dataset_mapping is None:
        msg = "'dataset_mapping' must be provided in the config to use dataset embeddings."
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is not None:
        if not isinstance(deprecated_list, (list, ListConfig)):
            msg = f"If '{deprecated_param_name}' is provided in the config, it must be a list of dataset names. Got: {deprecated_list!r}"
            logging.error(msg, stack_info=True)
            raise ValueError(msg)
        dataset_mapping = {name: name for name in deprecated_list}
        logging.warning(
            f"If '{deprecated_param_name}' is provided in the config, the code assumes that each dataset "
            f"maps to itself. Please use 'dataset_mapping' as '{deprecated_param_name}' "
            "is deprecated and will be removed in the future."
        )
    if not isinstance(dataset_mapping, (dict, DictConfig)) or not dataset_mapping:
        msg = f"'dataset_mapping' must be a non-empty dictionary, got: {dataset_mapping!r}"
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if not set(dataset_mapping.values()) <= set(dataset_mapping.keys()):
        missing = set(dataset_mapping.values()) - set(dataset_mapping.keys())
        msg = (
            f"dataset_mapping values {missing} are not present in "
            f"dataset_mapping keys {set(dataset_mapping.keys())}. "
            f"Values must be a subset of keys. Full mapping provided: {dataset_mapping}"
        )
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    return dataset_mapping


@registry.register_model("escnmd_backbone")
class eSCNMDBackbone(nn.Module, MOLEInterface):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,  # NOTE not used
        # NOTE: graph construction related, to remove
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,  # deprecated
        use_pbc_single: bool = True,  # deprecated
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: Literal["gaussian"] = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        direct_stress: bool = False,
        regress_stress: bool = False,
        regress_hessian: bool = False,
        hessian_vmap: bool = True,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: (
            list[str] | None
        ) = None,  # deprecated, use dataset_mapping instead
        dataset_mapping: (
            dict[str, str] | None
        ) = None,  # mapping from config dataset name to dataset embedding name e.g. {"omol": "omol", "oc20": "oc20", "oc20_subset": "oc20"}, this allows multiple subsets to use the same dataset embedding.
        use_dataset_embedding: bool = True,
        use_cuda_graph_wigner: bool = False,
        use_quaternion_wigner: bool = True,
        radius_pbc_version: int = 2,
        always_use_pbc: bool = True,
        output_edge_features: bool = False,
        charge_balanced_channels: list[int] | None = None,
        spin_balanced_channels: list[int] | None = None,
        edge_chunk_size: int = 1,
        execution_mode: str = "general",
    ) -> None:
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples
        self.output_edge_features = output_edge_features
        # set this True if we want to ALWAYS use pbc for internal graph gen
        # despite what's in the input data this only affects when otf_graph is True
        # in this mode, the user must be responsible for providing a large vaccum box
        # for aperiodic systems
        self.always_use_pbc = always_use_pbc

        # energy conservation related
        self.regress_config = GradRegressConfig(
            direct_forces=direct_forces,
            forces=regress_forces,
            stress=regress_stress,
            direct_stress=direct_stress,
            hessian=regress_hessian,
            hessian_vmap=hessian_vmap,
        )

        # which channels to balance - validate contiguity and store slice indices
        charge_channels = (
            list(charge_balanced_channels) if charge_balanced_channels else []
        )
        spin_channels = list(spin_balanced_channels) if spin_balanced_channels else []

        self.charge_channel_start, self.charge_channel_end = (
            validate_contiguous_channels(charge_channels, "charge_balanced_channels")
        )
        self.spin_channel_start, self.spin_channel_end = validate_contiguous_channels(
            spin_channels, "spin_balanced_channels"
        )

        # NOTE: graph construction related, to remove, except for cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.use_quaternion_wigner = use_quaternion_wigner
        self.enforce_max_neighbors_strictly = False

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            # The size of edge blocks to use in activation checkpointing
            activation_checkpoint_chunk_size = (
                ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE
            )
        self.edge_chunk_size = edge_chunk_size

        self.backend = get_execution_backend(execution_mode)

        # related to charge spin dataset system embedding
        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_mapping = dataset_mapping
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        if self.use_dataset_embedding:
            self.dataset_mapping = resolve_dataset_mapping(
                self.dataset_list, dataset_mapping, "dataset_list"
            )
        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])

        # Precompute Wigner coefficients for quaternion path (like Jd for Euler path)
        if self.use_quaternion_wigner:
            # lmin=5 because l=0,1,2,3,4 use custom kernels in the hybrid method
            self.wigner_data = create_wigner_data_module(lmax=self.lmax, lmin=5)

        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        # charge / spin embedding
        self.charge_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "charge",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.spin_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "spin",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )

        # dataset embedding
        if self.use_dataset_embedding:
            self.dataset_embedding = DatasetEmbedding(
                self.sphere_channels,
                enable_grad=self.dataset_emb_grad,
                dataset_mapping=self.dataset_mapping,
            )
            # mix charge, spin, dataset embeddings
            self.mix_csd = nn.Linear(3 * self.sphere_channels, self.sphere_channels)
        else:
            # mix charge, spin
            self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,  # NOTE: sqrt avg degree
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=self.backend,
        )

        self.envelope = PolynomialEnvelope(exponent=5)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSCNMD_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.ff_type,
                activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
                backend=self.backend,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    def balance_channels(
        self,
        x_message_prime: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        natoms: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.charge_channel_end > self.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms=natoms,
                batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end,
                target_offset=0.0,
            )
        if self.spin_channel_end > self.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms=natoms,
                batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end,
                target_offset=1.0,
            )
        return x_message_prime

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_quaternion_wigner:
            with record_function("obtain rotmat wigner quaternion"):
                wigner, wigner_inv = axis_angle_wigner_hybrid(
                    edge_distance_vecs,
                    self.lmax,
                    coeffs=self.wigner_data.coeffs,
                    U_blocks=self.wigner_data.U_blocks,
                    custom_kernels=self.wigner_data.custom_kernels,
                )
        else:
            Jd_buffers = [
                getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
                for l in range(self.lmax + 1)
            ]

            with record_function("obtain rotmat wigner original"):
                euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
                wigner = eulers_to_wigner(
                    euler_angles,
                    0,
                    self.lmax,
                    Jd_buffers,
                )
                wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # Both axis_angle_wigner_hybrid and eulers_to_wigner return contiguous D
        # (created via torch.zeros + slice assignment)
        # wigner_inv is made contiguous by .transpose().contiguous() above
        return wigner, wigner_inv

    def csd_embedding(self, charge, spin, dataset):
        with record_function("charge spin dataset embeddings"):
            # Add charge, spin, and dataset embeddings
            chg_emb = self.charge_embedding(charge)
            spin_emb = self.spin_embedding(spin)
            if self.use_dataset_embedding:
                assert dataset is not None
                dataset_emb = self.dataset_embedding(dataset)
                return torch.nn.SiLU()(
                    self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
                )
            return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def _generate_graph(self, data_dict):
        data_dict["gp_node_offset"] = 0
        node_partition = None
        if gp_utils.initialized():
            # create the partitions
            atomic_numbers_full = data_dict["atomic_numbers_full"]
            node_partition = torch.tensor_split(
                torch.arange(
                    len(atomic_numbers_full), device=atomic_numbers_full.device
                ),
                gp_utils.get_gp_world_size(),
            )[gp_utils.get_gp_rank()]
            assert (
                node_partition.numel() > 0
            ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"

        if self.otf_graph:
            pbc = None
            if self.always_use_pbc:
                pbc = torch.ones(len(data_dict), 3, dtype=torch.bool)
            else:
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            # for v2 graph gen we used to pass node_partition as part of the data_dict directly to radius_pbc to allow it generate partial graphs
            # to make it more general to accomodate v3, we scrapped and instead have generate_graph handle the partitioning after the graph has been generated
            graph_dict = generate_graph(
                data_dict,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
                node_partition=node_partition,
            )
        else:
            # this assume edge_index is provided
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"

            # Compute shifts from cell offsets
            if len(data_dict["natoms"]) == 1:
                # Single system: use matmul (compile-friendly, no data-dependent ops)
                shifts = data_dict["cell_offsets"].to(
                    data_dict["cell"].dtype
                ) @ data_dict["cell"].squeeze(0)
            else:
                # Batched: need repeat_interleave for variable edges per system
                cell_per_edge = data_dict["cell"].repeat_interleave(
                    data_dict["nedges"], dim=0
                )
                shifts = torch.einsum(
                    "ij,ijk->ik",
                    data_dict["cell_offsets"].to(cell_per_edge.dtype),
                    cell_per_edge,
                )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )  # [n_edges, 3]
            # pylint: disable=E1102
            edge_distance = torch.linalg.norm(
                edge_distance_vec, dim=-1, keepdim=False
            )  # [n_edges, 1]

            graph_dict = {
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
            }

        if gp_utils.initialized():
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                node_partition
            ]
            data_dict["batch"] = data_dict["batch_full"][node_partition]
            data_dict["gp_node_offset"] = node_partition.min().item()

        if graph_dict["edge_index"].shape[1] == 0:
            add_n_empty_edges(graph_dict, 1, self.cutoff, data_dict["gp_node_offset"])

        return graph_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict: AtomicData) -> dict[str, torch.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )

        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )

        # Enable gradients for autograd-based force/stress computation.
        # Must be set before graph generation so the computation graph
        # tracks positions and cell through edge distance calculations.
        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        with record_function("obtain wigner"):
            wigner, wigner_inv = self._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"],
            )
            coefficient_index = (
                self.coefficient_index if self.mmax != self.lmax else None
            )
            wigner, wigner_inv = self.backend.prepare_wigner(
                wigner,
                wigner_inv,
                self.mappingReduced,
                coefficient_index,
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        ###
        # Hook to allow MOLE
        ###
        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()

        # edge degree embedding
        with record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding),
                dim=1,
            )

            # Pre-fuse envelope into wigner_inv
            wigner_inv_envelope = wigner_inv * edge_envelope

            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_inv_envelope,
                data_dict["gp_node_offset"],
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        # Get edge embeddings for each layer
        # General backend: raw x_edge (rad_func computed inside SO2_Convolution)
        # Fast backends: precomputed radials
        with record_function("layer_radial_emb"):
            x_edge_per_layer = self.backend.get_layer_radial_emb(x_edge, self)

        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge_per_layer[i],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    node_offset=data_dict["gp_node_offset"],
                )
                # balance any channels requested
                x_message = self.balance_channels(
                    x_message,
                    charge=data_dict["charge"],
                    spin=data_dict["spin"],
                    natoms=data_dict["natoms"],
                    batch=data_dict["batch"],
                )

        # Final layer norm
        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
        }

        # Optionally include edge features for edge-level prediction heads
        if self.output_edge_features:
            out["edge_embedding"] = x_edge
            out["edge_index"] = graph_dict["edge_index"]
            out["edge_distance_vec"] = graph_dict["edge_distance_vec"]

        return out

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    @classmethod
    def build_inference_settings(cls, settings: InferenceSettings) -> dict:
        """Build backbone config overrides from inference settings."""
        overrides = {}

        # Always disable PBC wrapping for inference
        overrides["always_use_pbc"] = False

        if settings.activation_checkpointing is not None:
            overrides["activation_checkpointing"] = settings.activation_checkpointing
        if settings.edge_chunk_size is not None:
            overrides["edge_chunk_size"] = settings.edge_chunk_size
        if settings.external_graph_gen is not None:
            overrides["otf_graph"] = not settings.external_graph_gen
        if settings.internal_graph_gen_version is not None:
            overrides["radius_pbc_version"] = settings.internal_graph_gen_version
        if settings.use_quaternion_wigner is not None:
            overrides["use_quaternion_wigner"] = settings.use_quaternion_wigner
        if settings.execution_mode is not None:
            overrides["execution_mode"] = settings.execution_mode

        return overrides

    def get_default_untrained_tasks(
        self,
        checkpoint_tasks: dict[str, Task],
        inference_settings: InferenceSettings,
    ) -> list[Task]:
        """
        Return default untrained tasks for eSCNMDBackbone.

        For this backbone, we add stress tasks for all energy datasets
        that don't already have stress (either trained or explicitly requested).
        Stress can be computed via autograd from energy predictions.

        Returns empty list if the model uses direct forces, since autograd-based
        stress computation requires energy-conserving force computation.
        """
        # Direct force models can't compute stress via autograd
        if self.regress_config.direct_forces:
            return []

        tasks = []

        # Find datasets with energy but no stress
        energy_datasets = set()
        stress_datasets = set()
        energy_task_by_dataset = {}

        for task in checkpoint_tasks.values():
            if task.property == "energy":
                for dataset in task.datasets:
                    energy_datasets.add(dataset)
                    energy_task_by_dataset[dataset] = task
            elif task.property == "stress":
                stress_datasets.update(task.datasets)

        # Also exclude datasets already in predict_untrained_stress
        stress_datasets.update(inference_settings.predict_untrained_stress)

        # Create stress tasks for missing datasets
        missing_stress_datasets = energy_datasets - stress_datasets

        for dataset in missing_stress_datasets:
            energy_task = energy_task_by_dataset[dataset]
            # Infer task name prefix from energy task naming convention
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            tasks.append(
                Task(
                    name=f"{task_prefix}stress",
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(
                        dim=[1, 9], dtype=inference_settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )

        return tasks

    def validate_tasks(self, dataset_to_tasks: dict[str, list]) -> None:
        """
        Validate that task datasets are compatible with this backbone.
        """
        if self.use_dataset_embedding:
            assert set(dataset_to_tasks.keys()).issubset(
                set(self.dataset_mapping.keys())
            ), "Datasets in tasks is not a strict subset of datasets in backbone."

    def prepare_for_inference(self, data: AtomicData, settings: InferenceSettings):
        """
        Prepare model for inference. Called once on first prediction.
        """
        self._inference_settings = settings
        self.backend.validate(self.lmax, self.mmax, settings)
        self.backend.prepare_model_for_inference(self)
        return self

    def on_predict_check(self, data: AtomicData) -> None:
        """
        Called before each prediction.
        """

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        UMA-specific validation: handle charge/spin for OMOL task.

        Uses the shared validation logic from the api.inference module.
        """
        validate_uma_atoms_data(atoms, task_name)


class MLP_EFS_Head(nn.Module, HeadInterface):
    """MLP head for predicting energy, forces, and stress using autograd derivatives.

    This head computes forces and stress by taking gradients of the energy with respect to
    atomic positions and cell displacement.
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: str | None = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()

        self.reduce = reduce
        self.prefix = prefix
        self.wrap_property = wrap_property

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        backbone.energy_block = None
        backbone.force_block = None
        self.regress_config = backbone.regress_config

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        energy_key = f"{self.prefix}_energy" if self.prefix else "energy"
        forces_key = f"{self.prefix}_forces" if self.prefix else "forces"
        stress_key = f"{self.prefix}_stress" if self.prefix else "stress"
        hessian_key = f"{self.prefix}_hessian" if self.prefix else "hessian"

        outputs = {}

        # Use shared energy computation from parent class
        energy, energy_part = compute_energy(
            emb,
            self.energy_block,
            data["batch"],
            len(data["natoms"]),
            natoms=data["natoms"],
            reduce=self.reduce,
        )

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = emb["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )

        # Determine if we need create_graph for higher-order derivatives
        # Hessian computation requires second derivatives, so we need create_graph=True
        create_graph = self.training or self.regress_config.hessian

        if self.regress_config.stress and not self.regress_config.direct_stress:
            forces, stress = compute_forces_and_stress(
                energy_part,
                data["pos"],
                data["cell"],
                batch=data["batch_full"],  # use batch_full to work with GP reduction
                training=create_graph,
            )
            # TODO should we assume gradient forces always when stress is requested?
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
        elif self.regress_config.forces and not self.regress_config.direct_forces:
            forces = compute_forces(energy_part, data["pos"], training=self.training)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        else:
            forces = None

        if self.regress_config.hessian:
            if forces is None:
                raise ValueError(
                    "Hessian computation requires forces. "
                    "Please enable regress_forces or regress_stress."
                )
            if data["natoms"].numel() != 1:
                raise ValueError(
                    f"Hessian computation requires exactly 1 system in batch, "
                    f"found {data['natoms'].numel()}"
                )

            hessian = compute_hessian(
                forces,
                data["pos"],
                vmap=self.regress_config.hessian_vmap,
                training=create_graph,
            )
            outputs[hessian_key] = (
                {"hessian": hessian} if self.wrap_property else hessian
            )

        return outputs


# Deprecate this head in favor of MLP_EFS_Head with a regress_config.forces=False and regress_config.stress=False.
class MLP_Energy_Head(MLP_EFS_Head):
    """MLP head for predicting energy."""

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: str | None = None,
        wrap_property: bool = False,
    ) -> None:
        super().__init__(backbone, reduce, prefix, wrap_property)
        assert (
            backbone.regress_config.forces is False
            and backbone.regress_config.stress is False
        ) or (
            backbone.regress_config.direct_forces is True
            or backbone.regress_config.direct_stress is True
        ), (
            "regress_forces and regress_stress must be False or direct_forces must be True to use an MLP_Energy_Head. "
            "Use an MLP_EFS_Head if you want to predict gradient forces and stress."
        )


class Linear_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        energy, _ = compute_energy(
            emb,
            self.energy_block,
            data_dict["batch"],
            len(data_dict["natoms"]),
            natoms=data_dict["natoms"],
            reduce=self.reduce,
        )
        return {"energy": energy}


class IQA_Energy_Head(nn.Module, HeadInterface):
    """
    Advanced per-atom head: predict e_iqa_a using L=0 features AND norms of L>0 features.
    Includes Dropout, LayerNorm, and Residual connections.
    """

    def __init__(self, backbone: eSCNMDBackbone, dropout: float = 0.0) -> None:
        super().__init__()
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax

        # We will concatenate L=0 features with the norms of L=1..Lmax features.
        # Input dim = C * (Lmax + 1)
        input_dim = self.sphere_channels * (self.lmax + 1)

        # Projection from combined features to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_channels), nn.SiLU(), nn.Dropout(dropout)
        )

        # Residual Block 1
        self.res1 = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # Final output
        self.final = nn.Sequential(
            nn.LayerNorm(self.hidden_channels), nn.Linear(self.hidden_channels, 1)
        )

    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # emb["node_embedding"]: (N, (Lmax+1)^2, C)
        node_emb = emb["node_embedding"]

        # 1. Extract L=0 (Scalar) -> (N, C)
        scalars = node_emb.narrow(1, 0, 1).squeeze(1)

        features = [scalars]

        # 2. Extract Norms of L>0 -> (N, C)
        current_idx = 1
        for l in range(1, self.lmax + 1):
            length = 2 * l + 1
            vec = node_emb.narrow(1, current_idx, length)  # (N, 2l+1, C)
            vec_norm = vec.norm(dim=1)
            features.append(vec_norm)
            current_idx += length

        # 3. Concatenate all invariants
        x = torch.cat(features, dim=-1)  # (N, C * (Lmax+1))

        # 4. MLP with Residuals
        x = self.proj(x)
        x = x + self.res1(x)
        pred = self.final(x).squeeze(-1)

        if gp_utils.initialized():
            pred = gp_utils.gather_from_model_parallel_region(pred, dim=0)

        return {"pred": pred}


class IQA_Edge_Head(nn.Module, HeadInterface):
    """
    Edge-level head: predict pairwise IQA interactions using edge features.
    Operates on edge embeddings to predict per-edge scalar values like E_IQA_Inter(A,B)/2.
    """

    def __init__(self, backbone: eSCNMDBackbone, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_channels = backbone.edge_channels
        self.hidden_channels = backbone.hidden_channels

        # Edge features come from backbone: [distance_embedding + source_emb + target_emb]
        # This is already computed as x_edge in the backbone forward pass
        # x_edge has shape (num_edges, edge_channels_list[0])
        edge_input_dim = backbone.edge_channels_list[
            0
        ]  # distance_basis + 2*edge_channels

        # Projection from edge features to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual Block 1
        self.res1 = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # Residual Block 2
        self.res2 = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # Final output
        self.final = nn.Sequential(
            nn.LayerNorm(self.hidden_channels), nn.Linear(self.hidden_channels, 1)
        )

    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # emb["edge_embedding"]: (num_edges, edge_input_dim)
        if "edge_embedding" not in emb:
            raise ValueError(
                "IQA_Edge_Head requires 'edge_embedding' in emb dict. "
                "Set backbone output_edge_features=True"
            )

        edge_features = emb["edge_embedding"]

        # MLP with Residuals
        x = self.proj(edge_features)
        x = x + self.res1(x)
        x = x + self.res2(x)
        e = self.final(x).squeeze(-1)  # (num_edges,)

        return {"pred": e}


class IQA_Node_Head2(nn.Module, HeadInterface):
    """
    Advanced per-atom head: predict per-atom properties using L=0 features
    AND norms of L>0 features. Includes Dropout, LayerNorm, and Residual connections.
    """

    def __init__(self, backbone: eSCNMDBackbone, dropout: float = 0.1) -> None:
        super().__init__()
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax

        # We will concatenate L=0 features with the norms of L=1..Lmax features.
        # Input dim = C * (Lmax + 1)
        input_dim = self.sphere_channels * (self.lmax + 1)

        # Projection from combined features to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_channels), nn.SiLU(), nn.Dropout(dropout)
        )

        # Residual Block 1
        self.res1 = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # Final output
        self.final = nn.Sequential(
            nn.LayerNorm(self.hidden_channels), nn.Linear(self.hidden_channels, 1)
        )

    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # emb["node_embedding"]: (N, (Lmax+1)^2, C)
        node_emb = emb["node_embedding"]

        # 1. Extract L=0 (Scalar) -> (N, C)
        scalars = node_emb.narrow(1, 0, 1).squeeze(1)

        features = [scalars]

        # 2. Extract Norms of L>0 -> (N, C)
        current_idx = 1
        for l in range(1, self.lmax + 1):
            length = 2 * l + 1
            vec = node_emb.narrow(1, current_idx, length)  # (N, 2l+1, C)
            vec_norm = vec.norm(dim=1)
            features.append(vec_norm)
            current_idx += length

        # 3. Concatenate all invariants
        x = torch.cat(features, dim=-1)  # (N, C * (Lmax+1))

        # 4. MLP with Residuals
        x = self.proj(x)
        x = x + self.res1(x)
        pred = self.final(x).squeeze(-1)

        if gp_utils.initialized():
            pred = gp_utils.gather_from_model_parallel_region(pred, dim=0)

        return {"pred": pred}


class IQA_Edge_Head2(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone) -> None:
        super().__init__()
        # Edge embeddings are 1D features, not spherical harmonics
        # So we use a simple linear layer instead of SO3_Linear
        self.linear = nn.Linear(backbone.edge_channels_list[0], 1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]):
        if "edge_embedding" not in emb:
            raise ValueError(
                "IQA_Edge_Head2 requires 'edge_embedding' in emb dict. "
                "Set backbone output_edge_features=True"
            )

        edge_emb = emb["edge_embedding"]  # (num_edges, edge_input_dim)

        # Simple linear projection to scalar
        pred = self.linear(edge_emb).squeeze(-1)  # (num_edges,)

        if gp_utils.initialized():
            pred = gp_utils.gather_from_model_parallel_region(pred, dim=0)

        return {"pred": pred}


class IQA_Edge_Head_Equivariant(nn.Module, HeadInterface):
    def __init__(self, backbone, lmax=1):
        super().__init__()
        self.sphere_channels = backbone.sphere_channels
        self.backbone_lmax = backbone.lmax

        # Calculate edge embedding dimension: distance_basis + source_embedding + target_embedding
        num_distance_basis = backbone.num_distance_basis
        self.edge_embedding_dim = num_distance_basis + 2 * backbone.edge_channels

        # SO3_Linear: Process aggregated node features equivariantly
        self.so3_layer = SO3_Linear(
            in_features=self.sphere_channels,
            out_features=self.sphere_channels,
            lmax=self.backbone_lmax,
        )

        # Modulation network: use invariant edge embeddings
        self.edge_modulation = nn.Sequential(
            nn.Linear(self.edge_embedding_dim, self.sphere_channels),
            nn.SiLU(),
            nn.Linear(self.sphere_channels, self.sphere_channels),
        )

        # Final linear projection to scalar
        self.linear = nn.Linear(self.sphere_channels, 1)

    def forward(self, data, emb):
        edge_index = emb["edge_index"]
        node_emb = emb["node_embedding"]  # (num_atoms, (lmax+1)^2, sphere_channels)
        edge_embedding = emb[
            "edge_embedding"
        ]  # (num_edges, edge_embedding_dim) - invariant

        # Get source/target node embeddings
        src_emb = node_emb[edge_index[0]]  # (num_edges, (lmax+1)^2, sphere_channels)
        tgt_emb = node_emb[edge_index[1]]  # (num_edges, (lmax+1)^2, sphere_channels)

        # Combine source and target: sum preserves equivariance
        edge_feat = src_emb + tgt_emb  # (num_edges, (lmax+1)^2, sphere_channels)

        # Apply SO3_Linear to transform equivariantly
        x = self.so3_layer(edge_feat)

        # Extract only L=0 component (invariant scalar at index 0)
        x = x.narrow(1, 0, 1)  # (num_edges, 1, sphere_channels)

        # Modulate by edge embeddings (distance + element chemistry)
        modulation = self.edge_modulation(
            edge_embedding
        )  # (num_edges, sphere_channels)
        x = x * modulation.unsqueeze(1)  # (num_edges, 1, sphere_channels)

        # Final projection to scalar
        pred = self.linear(x.squeeze(1)).squeeze(-1)  # (num_edges,)
        return {"pred": pred}


class SO2EquivariantGraphAttentionNodeEdgePrediction(nn.Module, HeadInterface):
    """SO(2)-equivariant graph attention head for joint node/edge prediction.

    **EBDM-ORIGIN REFERENCE:**
    This class is ported from experimental/EBDM/models/transformer_block.py::SO2EquivariantGraphAttentionNodeEdgePrediction.
    The architecture and hyperparameter flags match the EBDM reference exactly.

    **FAIRCHEM ADAPTATIONS:**
    1. Constructor takes backbone object (not separate params) to extract config.
    2. Forward signature uses fairchem's HeadInterface contract: forward(data, emb) with emb as dict.
    3. Output is task-keyed dict (not tuple) for MLIP loss routing.
    4. Uses raw PyTorch tensors instead of SO3_Embedding object wrapper.
    5. Rotation performed via backbone._get_rotmat_and_wigner() instead of object methods.
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        sphere_channels: int | None = None,
        hidden_channels: int | None = None,
        num_heads: int = 8,
        attn_alpha_channels: int | None = None,
        attn_value_channels: int | None = None,
        output_channels_edges: int = 1,
        output_channels_nodes: int = 1,
        out_degree: int = 0,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        SO3_rotation=None,
        mappingReduced: CoefficientMapping | None = None,
        SO3_grid: nn.ModuleDict | None = None,
        max_num_elements: int | None = None,
        edge_channels_list: list[int] | None = None,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        activation: str = "scaled_silu",
        use_tp_reparam: bool = False,
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
        edge_prediction: bool = True,
        node_prediction: bool = True,
        edge_task_name: str = "iqa_inter_ab",
        node_task_name: str = "iqa_intra_a",
        extra_node_task_name: str | None = None,
    ) -> None:
        super().__init__()

        # === EBDM-origin: Parameter initialization ===
        self.backbone = backbone
        self.sphere_channels = (
            sphere_channels if sphere_channels is not None else backbone.sphere_channels
        )
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else backbone.hidden_channels
        )
        self.num_heads = num_heads
        self.output_channels_edges = output_channels_edges
        self.output_channels_nodes = output_channels_nodes

        default_lmax = backbone.lmax
        default_mmax = backbone.mmax
        self.lmax_list = lmax_list if lmax_list is not None else [default_lmax]
        self.mmax_list = mmax_list if mmax_list is not None else [default_mmax]
        self.num_resolutions = len(self.lmax_list)

        self.edge_prediction = edge_prediction
        self.node_prediction = node_prediction
        self.edge_task_name = edge_task_name
        self.node_task_name = node_task_name
        self.extra_node_task_name = extra_node_task_name
        self.out_degree = out_degree

        # === EBDM-origin: Save rotation/mapping references ===
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = (
            mappingReduced if mappingReduced is not None else backbone.mappingReduced
        )
        self.SO3_grid = SO3_grid if SO3_grid is not None else backbone.SO3_grid
        self.max_num_elements = (
            max_num_elements
            if max_num_elements is not None
            else backbone.max_num_elements
        )

        # === EBDM-origin: Feature flags ===
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.activation = activation
        self.use_tp_reparam = use_tp_reparam
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act
        if self.use_s2_act_attn and (self.use_gate_act or self.use_sep_s2_act):
            raise NotImplementedError(
                "use_s2_act_attn=True currently supports only non-gated S2 activation "
                "in this fairchem tensor implementation."
            )

        # === FAIRCHEM ADAPTATION: Edge channel setup ===
        # In fairchem, edge_channels_list comes from backbone config.
        # We extract first element (distance channels) and optionally append atom embeddings.
        if edge_channels_list is None:
            edge_channels_list = copy.deepcopy(backbone.edge_channels_list)
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.edge_distance_channels = backbone.num_distance_basis

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_distance_channels + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding = None
            self.target_embedding = None
            self.edge_channels_list[0] = self.edge_distance_channels

        self.edge_embedding_dim = self.edge_channels_list[0]

        # === EBDM-origin: Attention channel sizing ===
        self.attn_alpha_channels = (
            attn_alpha_channels
            if attn_alpha_channels is not None
            else (self.hidden_channels // self.num_heads)
        )
        self.attn_value_channels = (
            attn_value_channels
            if attn_value_channels is not None
            else (self.hidden_channels // self.num_heads)
        )

        # === EBDM-origin: Output irrep extraction ===
        self.num_irreps_passed = 0
        for l in range(self.out_degree):
            self.num_irreps_passed += 2 * l + 1

        # === EBDM-origin: Attention mechanism setup ===
        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        elif self.use_attn_renorm:
            self.alpha_norm = nn.LayerNorm(self.attn_alpha_channels)
        else:
            self.alpha_norm = nn.Identity()
        if not self.use_s2_act_attn:
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop > 0.0 else None

        # === EBDM-origin: SO(2) convolution setup ===
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels += max(self.lmax_list) * self.hidden_channels
            elif self.use_sep_s2_act:
                extra_m0_output_channels += self.hidden_channels

        # === EBDM-origin: m-share-rad radial weighting path ===
        self.rad_func = None
        if self.use_m_share_rad:
            m_share_edge_channels = copy.deepcopy(self.edge_channels_list)
            m_share_edge_channels.append(
                2 * self.sphere_channels * (max(self.lmax_list) + 1)
            )
            self.rad_func = RadialMLP(m_share_edge_channels)
            self.register_buffer(
                "expand_index",
                self.mappingReduced.l_harmonic.clone(),
                persistent=False,
            )

        # === FAIRCHEM ADAPTATION: SO2 convolution class selection ===
        # In EBDM, class is passed as parameter. Here we select based on use_tp_reparam flag.
        # FAIRCHEM NOTE: SO2_Convolution_TensorProduct takes scalars lmax/mmax, not lists.
        so2_convolution_class = (
            SO2_Convolution_TensorProduct if self.use_tp_reparam else SO2_Convolution
        )

        self.so2_conv_1 = so2_convolution_class(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list[
                0
            ],  # FAIRCHEM ADAPTATION: Scalar, not list (single resolution)
            self.mmax_list[
                0
            ],  # FAIRCHEM ADAPTATION: Scalar, not list (single resolution)
            self.mappingReduced,
            internal_weights=self.use_m_share_rad,
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,
        )

        # === EBDM-origin: Activation layer selection ===
        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        elif self.use_sep_s2_act:
            self.s2_act = SeparableS2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                SO3_grid=self.SO3_grid,
            )
        else:
            self.s2_act = S2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                SO3_grid=self.SO3_grid,
            )

        self.so2_conv_2 = so2_convolution_class(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list[0],  # FAIRCHEM ADAPTATION: Scalar
            self.mmax_list[0],  # FAIRCHEM ADAPTATION: Scalar
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=self.num_heads if self.use_s2_act_attn else None,
        )

        # === EBDM-origin: Output projection layers ===
        # FAIRCHEM ADAPTATION: Use SO3_Linear instead of SO3_LinearV2 (fairchem's version).
        proj_hidden = self.num_heads * max(self.attn_value_channels // 2, 1)
        if self.edge_prediction and self.output_channels_edges > 0:
            self.proj_edges_1 = SO3_Linear(
                self.num_heads * self.attn_value_channels,
                proj_hidden,
                lmax=self.lmax_list[0],
            )
            self.proj_edges_2 = SO3_Linear(
                proj_hidden,
                self.output_channels_edges,
                lmax=self.lmax_list[0],
            )
        if self.node_prediction and self.output_channels_nodes > 0:
            self.proj_nodes_1 = SO3_Linear(
                self.num_heads * self.attn_value_channels,
                proj_hidden,
                lmax=self.lmax_list[0],
            )
            self.proj_nodes_2 = SO3_Linear(
                proj_hidden,
                self.output_channels_nodes,
                lmax=self.lmax_list[0],
            )
            if self.extra_node_task_name is not None:
                self.proj_nodes_2_extra = SO3_Linear(
                    proj_hidden,
                    self.output_channels_nodes,
                    lmax=self.lmax_list[0],
                )
            else:
                self.proj_nodes_2_extra = None

    @staticmethod
    def _squeeze_scalar_output(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1 and x.shape[2] == 1:
            return x.squeeze(1).squeeze(-1)
        if x.shape[2] == 1:
            return x.squeeze(-1)
        return x

    def forward(self, data, emb):
        """Forward pass for SO(2)-equivariant graph attention.

        **FAIRCHEM ADAPTATION:** forward signature differs from EBDM.
        - EBDM: forward(x: SO3_Embedding, atomic_numbers, edge_distance, edge_index)
        - FAIRCHEM: forward(data: dict, emb: dict) via HeadInterface contract
            - data: Contains atomic_numbers, pos, batch, etc. from Data object
            - emb: Dict with node_embedding [N, (lmax+1)², C], edge_embedding [E, C_edge],
                     edge_index [2, E], edge_distance_vec [E, 3]

        Returns:
            Dict[str, Dict[str, Tensor]]: Task-keyed predictions
            - {edge_task_name: {"edge_pred": [E, 1]}} if edge_prediction=True
            - {node_task_name: {"node_pred": [N, 1]}} if node_prediction=True
        """
        # === FAIRCHEM ADAPTATION: Extract emb dict components ===
        # In EBDM, inputs are separate parameters; here they come bundled in emb dict
        # from the backbone's forward output.
        if "edge_embedding" not in emb:
            raise ValueError(
                "SO2EquivariantGraphAttentionNodeEdgePrediction requires "
                "'edge_embedding' in emb. Set backbone output_edge_features=True."
            )

        edge_index = emb["edge_index"]
        node_embedding = emb["node_embedding"]
        edge_embedding = emb["edge_embedding"]
        edge_distance_vec = emb["edge_distance_vec"]

        num_edges = edge_index.shape[1]
        num_nodes = node_embedding.shape[0]

        # === EBDM-origin: Build edge messages ===
        # Concatenate source and target node embeddings along feature dimension.
        # This creates per-edge feature tensors for SO(2) convolution.
        x_source = node_embedding[edge_index[0]]
        x_target = node_embedding[edge_index[1]]
        x_message_data = torch.cat((x_source, x_target), dim=2)

        # === EBDM-origin: Build edge features ===
        # Distance basis + optional atom identity embeddings.
        # FAIRCHEM ADAPTATION: Extract distance channels from precomputed edge_embedding
        # instead of receiving edge_distance as parameter. Backbone handles distance
        # basis expansion.
        x_edge = edge_embedding.narrow(1, 0, self.edge_distance_channels)
        if self.use_atom_edge_embedding:
            source_element = data["atomic_numbers"][edge_index[0]]
            target_element = data["atomic_numbers"][edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((x_edge, source_embedding, target_embedding), dim=1)

        # === EBDM-origin: Rotation to edge-aligned frame ===
        # Apply Wigner D rotation matrices to rotate node embeddings into edge-local frame
        # where SO(2) equivariance is natural (edge aligned with z-axis).
        # FAIRCHEM ADAPTATION: Fetch Wigner matrices from backbone using edge_distance_vec
        # instead of using SO3_Embedding._rotate() method. Backbone computes rotations
        # via Euler angles and Wigner D matrix generation.
        wigner, wigner_inv = self.backbone._get_rotmat_and_wigner(
            edge_distance_vec,
        )
        # FAIRCHEM ADAPTATION: Use torch.bmm instead of SO3_Embedding._rotate() method
        x_message = torch.bmm(wigner, x_message_data)

        # === EBDM-origin: m-share-rad radial weighting (optional) ===
        # If use_m_share_rad=True, apply per-l radial weights to break redundancy.
        # Each l gets shared radial weights across all m ∈ [-l, l] for that l.
        # FAIRCHEM ADAPTATION: Use index_select on expand_index (l_harmonic mapping)
        # to replicate per-l scalars across m channels. EBDM uses similar pattern
        # internally.
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.view(
                -1,
                (max(self.lmax_list) + 1),
                2 * self.sphere_channels,
            )
            x_edge_weight = torch.index_select(
                x_edge_weight,
                dim=1,
                index=self.expand_index,
            )
            x_message = x_message * x_edge_weight

        # === EBDM-origin: First SO(2) convolution ===
        # Message features through SO(2) tensor-product convolution. If not using
        # s2_act_attn, also outputs m=0 coefficients for attention weight computation.
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # === EBDM-origin: Activation between SO(2) convolutions ===
        # Three options: GateActivation, SeparableS2Activation, or standard S2Activation.
        # All are SO(3) equivariant nonlinearities preserving spherical harmonic structure.
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_s2_act_attn:
            x_message = self.s2_act(x_message)
        elif self.use_gate_act:
            x_0_gating = x_0_extra.narrow(
                1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels
            )
            x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)
            x_message = self.gate_act(x_0_gating, x_message)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )
                x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)
                x_message = self.s2_act(x_0_gating, x_message)
            else:
                x_0_alpha = x_0_extra
                x_message = self.s2_act(x_message)

        # === EBDM-origin: Second SO(2) convolution ===
        # Project to num_heads * attn_value_channels for attention computation.
        # Note: so2_conv_2 uses internal_weights=True, so it doesn't need edge features.
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message)
        else:
            x_message = self.so2_conv_2(x_message)

        # === EBDM-origin: Attention weight computation ===
        # Compute per-head scalar attention weights from m=0 (scalar) coefficients.
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik,ik->bi", x_0_alpha, self.alpha_dot)
        # EBDM-origin: Softmax per target node (edge_index[1] groups edges by target).
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = alpha.view(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # === EBDM-origin: Attention weights x message vectors ===
        # Reshape to separate num_heads dimension, apply attention, reshape back.
        attn = x_message.view(
            num_edges,
            x_message.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        x_message = attn.view(
            num_edges,
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )

        # === EBDM-origin: Rotation back to global frame ===
        # Apply inverse Wigner matrix to map from edge-aligned back to global frame.
        x_message = torch.bmm(wigner_inv, x_message)

        # === FAIRCHEM ADAPTATION: Task-keyed output dict ===
        # EBDM returns (out_embedding_nodes, out_embedding_edges) tuple.
        # FAIRCHEM returns {task_name: {pred_key: tensor}} dict to support
        # multi-task loss routing in MLIP framework.
        output = {}

        # === EBDM-origin: Edge prediction (if enabled) ===
        if self.edge_prediction and self.output_channels_edges > 0:
            out_embedding_edges = self.proj_edges_2(self.proj_edges_1(x_message))
            out_embedding_edges = out_embedding_edges.narrow(
                1, self.num_irreps_passed, 2 * self.out_degree + 1
            )
            edge_pred = self._squeeze_scalar_output(out_embedding_edges)
            if gp_utils.initialized():
                edge_pred = gp_utils.gather_from_model_parallel_region(edge_pred, dim=0)
            # FAIRCHEM ADAPTATION: Nested dict with edge_task_name key
            output[self.edge_task_name] = {"edge_pred": edge_pred}

        # === EBDM-origin: Aggregate edge messages to nodes ===
        # Sum all incoming edge messages for each target node. This is the key
        # neighbor-aggregation step in graph neural networks.
        # FAIRCHEM ADAPTATION: Use torch_geometric.utils.scatter instead of
        # SO3_Embedding._reduce_edge() method. Scatter sums edge messages [E, F]
        # by target node index [E] to produce node messages [N, F].
        x_nodes = torch_geometric.utils.scatter(
            x_message,
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        # === EBDM-origin: Node prediction (if enabled) ===
        if self.node_prediction and self.output_channels_nodes > 0:
            proj_nodes = self.proj_nodes_1(x_nodes)
            out_embedding_nodes = self.proj_nodes_2(proj_nodes)
            out_embedding_nodes = out_embedding_nodes.narrow(
                1, self.num_irreps_passed, 2 * self.out_degree + 1
            )
            node_pred = self._squeeze_scalar_output(out_embedding_nodes)
            if gp_utils.initialized():
                node_pred = gp_utils.gather_from_model_parallel_region(node_pred, dim=0)
            # FAIRCHEM ADAPTATION: Nested dict with node_task_name key
            output[self.node_task_name] = {"node_pred": node_pred}
            if self.proj_nodes_2_extra is not None:
                out_embedding_nodes_extra = self.proj_nodes_2_extra(proj_nodes)
                out_embedding_nodes_extra = out_embedding_nodes_extra.narrow(
                    1, self.num_irreps_passed, 2 * self.out_degree + 1
                )
                node_pred_extra = self._squeeze_scalar_output(out_embedding_nodes_extra)
                if gp_utils.initialized():
                    node_pred_extra = gp_utils.gather_from_model_parallel_region(
                        node_pred_extra, dim=0
                    )
                output[self.extra_node_task_name] = {"node_pred": node_pred_extra}

        return output


class IQA_Edge_Head_Equiformer(SO2EquivariantGraphAttentionNodeEdgePrediction):
    """Backward-compatible alias for existing configs."""


class IQA_Components_EFS_Head(nn.Module, HeadInterface):
    """Predict IQA components and forces from component-summed energy."""

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        node_task_name: str = "iqa_intra_a",
        inter_task_name: str = "iqa_inter_a",
        edge_task_name: str = "iqa_inter_ab",
        forces_direct_task_name: str = "iqa_forces_direct",
        forces_grad_task_name: str = "iqa_forces_grad",
        include_inter_a: bool = False,
        edge_energy_mode: str = "half",
        intra_rmsd: float = 1.0,
        inter_a_rmsd: float = 1.0,
        inter_ab_rmsd: float = 1.0,
    ) -> None:
        super().__init__()
        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces
        self.node_task_name = node_task_name
        self.inter_task_name = inter_task_name
        self.edge_task_name = edge_task_name
        self.forces_direct_task_name = forces_direct_task_name
        self.forces_grad_task_name = forces_grad_task_name
        self.include_inter_a = include_inter_a
        self.edge_energy_mode = edge_energy_mode
        self.intra_rmsd = intra_rmsd
        self.inter_a_rmsd = inter_a_rmsd
        self.inter_ab_rmsd = inter_ab_rmsd

        self.edge_head = IQA_Edge_Head_Equiformer(
            backbone,
            edge_prediction=True,
            node_prediction=True,
            edge_task_name=edge_task_name,
            node_task_name=node_task_name,
            extra_node_task_name=inter_task_name if include_inter_a else None,
        )
        self.force_head = Linear_Force_Head(backbone)

        if self.edge_energy_mode not in {"half", "directed", "upper_triangle"}:
            raise ValueError(
                "edge_energy_mode must be one of: half, directed, upper_triangle"
            )

    @staticmethod
    def _sum_nodes(
        values: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        energy_part = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        energy_part.index_add_(0, batch, values)
        return energy_part

    def _sum_edges(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        nedges: torch.Tensor,
    ) -> torch.Tensor:
        edge_batch = torch.repeat_interleave(
            torch.arange(nedges.shape[0], device=nedges.device),
            nedges,
        )

        if self.edge_energy_mode == "upper_triangle":
            mask = edge_index[0] < edge_index[1]
            values = values[mask]
            edge_batch = edge_batch[mask]

        energy_part = torch.zeros(
            nedges.shape[0], device=values.device, dtype=values.dtype
        )
        energy_part.index_add_(0, edge_batch, values)

        if self.edge_energy_mode == "half":
            energy_part = energy_part * 0.5

        return energy_part

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}

        inter_a_pred = None
        edge_outputs = self.edge_head(data, emb)
        intra_pred = edge_outputs[self.node_task_name]["node_pred"]
        outputs[self.node_task_name] = {"node_pred": intra_pred}
        if self.edge_task_name in edge_outputs:
            edge_pred = edge_outputs[self.edge_task_name]["edge_pred"]
            outputs[self.edge_task_name] = {"edge_pred": edge_pred}
        else:
            edge_pred = None

        if self.include_inter_a and self.inter_task_name in edge_outputs:
            inter_a_pred = edge_outputs[self.inter_task_name]["node_pred"]
            outputs[self.inter_task_name] = {"node_pred": inter_a_pred}

        if data["pos"].requires_grad is False:
            data["pos"].requires_grad = True

        energy_nodes = intra_pred * self.intra_rmsd
        if inter_a_pred is not None:
            energy_nodes = energy_nodes + inter_a_pred * self.inter_a_rmsd

        energy_part = self._sum_nodes(energy_nodes, data["batch"], len(data["natoms"]))
        if inter_a_pred is None and edge_pred is not None:
            energy_part = energy_part + self._sum_edges(
                edge_pred * self.inter_ab_rmsd,
                emb["edge_index"],
                data["nedges"],
            )

        if gp_utils.initialized():
            energy_part = gp_utils.reduce_from_model_parallel_region(energy_part)

        forces_grad = -torch.autograd.grad(
            energy_part.sum(),
            data["pos"],
            create_graph=self.training,
        )[0]
        if gp_utils.initialized():
            forces_grad = gp_utils.reduce_from_model_parallel_region(forces_grad)

        forces_direct = self.force_head(data, emb)["forces"]
        outputs[self.forces_direct_task_name] = {"forces": forces_direct}
        outputs[self.forces_grad_task_name] = {"forces": forces_grad}

        return outputs


class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone) -> None:
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]):
        # SO3_Linear with lmax=1 requires both L=0 and L=1 as input
        l0_l1_embedding = get_l_component_range(emb["node_embedding"], l_min=0, l_max=1)
        forces_output = self.linear(l0_l1_embedding)

        # Extract L=1 (vector) component from the output
        forces = get_l_component_range(forces_output, l_min=1, l_max=1)
        forces = forces.view(-1, 3).contiguous()

        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(
                forces, data_dict["atomic_numbers_full"].shape[0]
            )

        return {"forces": forces}


def compose_tensor(
    trace: torch.Tensor,
    l2_symmetric: torch.Tensor,
) -> torch.Tensor:
    """Re-compose a tensor from its decomposition

    Args:
        trace: a tensor with scalar part of the decomposition of r2 tensors in the batch
        l2_symmetric: tensor with the symmetric/traceless part of decomposition

    Returns:
        tensor: rank 2 tensor
    """

    if trace.shape[1] != 1:
        raise ValueError("batch of traces must be shape (batch size, 1)")

    if l2_symmetric.shape[1] != 5:
        raise ValueError("batch of l2_symmetric tensors must be shape (batch size, 5)")

    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError(
            "Shape missmatch between trace and l2_symmetric parts. The first dimension is the batch dimension"
        )

    batch_size = trace.shape[0]
    decomposed_preds = torch.zeros(
        batch_size, irreps_sum(2), device=trace.device
    )  # rank 2
    decomposed_preds[:, : irreps_sum(0)] = trace
    decomposed_preds[:, irreps_sum(1) : irreps_sum(2)] = l2_symmetric

    r2_tensor = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(2, device=trace.device),
        decomposed_preds,
    )
    return r2_tensor


class MLP_Stress_Head(nn.Module, HeadInterface):
    """MLP head for predicting the stress tensor.

    Predicts the isotropic (L=0) and anisotropic (L=2) parts of the stress tensor
    separately to ensure symmetry, then recomposes back to the full stress tensor.
    """

    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "mean") -> None:
        super().__init__()
        self.reduce = reduce
        assert reduce in ["sum", "mean"]
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.scalar_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self.l2_linear = SO3_Linear(backbone.sphere_channels, 1, lmax=2)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        num_systems = len(data_dict["natoms"])
        batch = data_dict["batch"]

        # Compute isotropic (L=0) part of stress using MLP on scalar embedding
        scalar_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=0
        ).squeeze(1)
        node_scalar = self.scalar_block(scalar_embedding).view(-1)
        iso_stress, _ = reduce_node_to_system(node_scalar, batch, num_systems)

        if self.reduce == "mean":
            iso_stress = iso_stress / data_dict["natoms"]

        # Compute anisotropic (L=2) part of stress using SO3_Linear
        l0l1l2_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=2
        )
        l2_output = self.l2_linear(l0l1l2_embedding)

        node_l2 = (
            get_l_component_range(l2_output, l_min=2, l_max=2).view(-1, 5).contiguous()
        )
        aniso_stress, _ = reduce_node_to_system(node_l2, batch, num_systems)

        if self.reduce == "mean":
            aniso_stress = aniso_stress / data_dict["natoms"].unsqueeze(1)

        # Recompose the full stress tensor from isotropic and anisotropic parts
        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)

        return {"stress": stress}
