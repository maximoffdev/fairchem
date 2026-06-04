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
from typing import TYPE_CHECKING, Literal, Dict

import torch
import torch.nn as nn
from torch.profiler import record_function

import torch_geometric.utils

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import get_device_for_local_rank
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.rotation_cuda_graph import RotMatWignerCudaGraph
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.embedding_dev import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole_utils import MOLEInterface
from fairchem.core.models.uma.nn.radial import GaussianSmearing, RadialMLP
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
from fairchem.core.models.uma.nn.so2_tp import SO2_Convolution_TensorProduct
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum

from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


ESCNMD_DEFAULT_EDGE_CHUNK_SIZE = 1024 * 128


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
        regress_stress: bool = False,
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
        dataset_list: list[str] | None = None,
        use_dataset_embedding: bool = True,
        use_cuda_graph_wigner: bool = False,
        radius_pbc_version: int = 1,
        always_use_pbc: bool = True,
        output_edge_features: bool = False,
        term_element_refs: dict[str, list[float]] | None = None,
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
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress

        # NOTE: graph construction related, to remove, except for cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.enforce_max_neighbors_strictly = False

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            # The size of edge blocks to use in activation checkpointing
            activation_checkpoint_chunk_size = ESCNMD_DEFAULT_EDGE_CHUNK_SIZE

        # related to charge spin dataset system embedding
        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        if self.use_dataset_embedding:
            assert (
                self.dataset_list
            ), "the dataset list is empty, please add it to the model backbone config"
        self.use_cuda_graph_wigner = use_cuda_graph_wigner

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
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
                grad=self.dataset_emb_grad,
                dataset_list=self.dataset_list,
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
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,  # NOTE: sqrt avg degree
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
        )

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
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

        self.rot_mat_wigner_cuda = None  # lazily initialize this
        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor, use_cuda_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        if use_cuda_graph:
            if self.rot_mat_wigner_cuda is None:
                self.rot_mat_wigner_cuda = RotMatWignerCudaGraph()
            with record_function("obtain rotmat wigner cudagraph"):
                wigner, wigner_inv = self.rot_mat_wigner_cuda.get_rotmat_and_wigner(
                    edge_distance_vecs, Jd_buffers
                )
        else:
            with record_function("obtain rotmat wigner original"):
                euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
                wigner = eulers_to_wigner(
                    euler_angles,
                    0,
                    self.lmax,
                    Jd_buffers,
                )
                wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
        )
        return wigner_and_M_mapping, wigner_and_M_mapping_inv

    def _get_displacement_and_cell(
        self, data_dict: AtomicData
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ###############################################################
        # gradient-based forces/stress
        ###############################################################
        displacement = None
        orig_cell = None
        if self.regress_stress and not self.direct_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=data_dict["pos"].dtype,
                device=data_dict["pos"].device,
            )
            num_batch = len(data_dict["natoms"])
            displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            if data_dict["pos"].requires_grad is False:
                data_dict["pos"].requires_grad = True
            data_dict["pos_original"] = data_dict["pos"]
            data_dict["pos"] = data_dict["pos"] + torch.bmm(
                data_dict["pos"].unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data_dict["batch"]),
            ).squeeze(-2)

            orig_cell = data_dict["cell"]
            data_dict["cell"] = data_dict["cell"] + torch.bmm(
                data_dict["cell"], symmetric_displacement
            )

        if (
            not self.regress_stress
            and self.regress_forces
            and not self.direct_forces
            and data_dict["pos"].requires_grad is False
        ):
            data_dict["pos"].requires_grad = True
        return displacement, orig_cell

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
        if self.otf_graph:
            pbc = None
            if self.always_use_pbc:
                pbc = torch.ones(len(data_dict), 3, dtype=torch.bool)
            else:
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            assert (
                pbc.all() or (~pbc).all()
            ), "We can only accept pbc that is all true or all false"
            logging.debug(f"Using radius graph gen version {self.radius_pbc_version}")
            graph_dict = generate_graph(
                data_dict,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
            )
        else:
            # this assume edge_index is provided
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"
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
                "node_offset": 0,
            }

        if gp_utils.initialized():
            graph_dict = self._init_gp_partitions(
                graph_dict, data_dict["atomic_numbers_full"]
            )
            # create partial atomic numbers and batch tensors for GP
            node_partition = graph_dict["node_partition"]
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                node_partition
            ]
            data_dict["batch"] = data_dict["batch_full"][node_partition]
        else:
            graph_dict["node_offset"] = 0
            graph_dict["edge_distance_vec_full"] = graph_dict["edge_distance_vec"]
            graph_dict["edge_distance_full"] = graph_dict["edge_distance"]
            graph_dict["edge_index_full"] = graph_dict["edge_index"]

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

        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = self._get_displacement_and_cell(data_dict)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        if graph_dict["edge_index"].numel() == 0:
            raise ValueError(
                f"No edges found in input system, this means either you have a single atom in the system or the atoms are farther apart than the radius cutoff of the model of {self.cutoff} Angstroms. We don't know how to handle this case. Check the positions of system: {data_dict['pos']}"
            )

        with record_function("obtain wigner"):
            (wigner_and_M_mapping_full, wigner_and_M_mapping_inv_full) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec_full"],
                    use_cuda_graph=self.use_cuda_graph_wigner
                    and "cuda" in get_device_for_local_rank()
                    and not self.training,
                )
            )
            if gp_utils.initialized():
                wigner_and_M_mapping = wigner_and_M_mapping_full[
                    graph_dict["edge_partition"]
                ]
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full[
                    graph_dict["edge_partition"]
                ]
            else:
                wigner_and_M_mapping = wigner_and_M_mapping_full
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full

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
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv,
                graph_dict["node_offset"],
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner_and_M_mapping,
                    wigner_and_M_mapping_inv,
                    sys_node_embedding=sys_node_embedding,
                    node_offset=graph_dict["node_offset"],
                )

        # Final layer norm
        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
            "batch": data_dict["batch"],
        }
        
        # Optionally include edge features for edge-level prediction heads
        if self.output_edge_features:
            out["edge_embedding"] = x_edge
            out["edge_index"] = graph_dict["edge_index"]
            out["edge_distance_vec"] = graph_dict["edge_distance_vec"]

        return out

    def _init_gp_partitions(self, graph_dict, atomic_numbers_full):
        """Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        edge_index = graph_dict["edge_index"]
        edge_distance = graph_dict["edge_distance"]
        edge_distance_vec_full = graph_dict["edge_distance_vec"]

        node_partition = torch.tensor_split(
            torch.arange(len(atomic_numbers_full)).to(atomic_numbers_full.device),
            gp_utils.get_gp_world_size(),
        )[gp_utils.get_gp_rank()]

        assert (
            node_partition.numel() > 0
        ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]

        # full versions of data
        graph_dict["edge_distance_vec_full"] = edge_distance_vec_full
        graph_dict["edge_distance_full"] = edge_distance
        graph_dict["edge_index_full"] = edge_index
        graph_dict["edge_partition"] = edge_partition
        graph_dict["node_partition"] = node_partition

        # gp versions of data
        graph_dict["edge_index"] = edge_index[:, edge_partition]
        graph_dict["edge_distance"] = edge_distance[edge_partition]
        graph_dict["edge_distance_vec"] = edge_distance_vec_full[edge_partition]
        graph_dict["node_offset"] = node_partition.min().item()

        return graph_dict

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


class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone: eSCNMDBackbone,
        prefix: str | None = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces
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

        # TODO: this is not very clean, bug-prone.
        # but is currently necessary for finetuning pretrained models that did not have
        # the direct_forces flag set to False
        backbone.direct_forces = False
        assert (
            not backbone.direct_forces
        ), "EFS head is only used for gradient-based forces/stress."

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}
        _input = emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)
        node_energy = _output.view(-1, 1, 1)
        energy_part = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy_part.index_add_(0, data["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        embeddings = emb["node_embedding"].detach()
        if gp_utils.initialized():
            embeddings = gp_utils.gather_from_model_parallel_region(embeddings, dim=0)

        outputs["embeddings"] = (
            {"embeddings": embeddings} if self.wrap_property else embeddings
        )

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy_part.sum()],
                [data["pos_original"], emb["displacement"]],
                create_graph=self.training,
            )
            if gp_utils.initialized():
                grads = (
                    gp_utils.reduce_from_model_parallel_region(grads[0]),
                    gp_utils.reduce_from_model_parallel_region(grads[1]),
                )

            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            stress = stress.view(
                -1, 9
            )  # NOTE to work better with current Multi-task trainer
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy_part.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            if gp_utils.initialized():
                forces = gp_utils.reduce_from_model_parallel_region(forces)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        return outputs


class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


class Linear_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


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
            nn.Linear(input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
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
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, 1)
        )

    def forward(self, data: AtomicData, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # emb["node_embedding"]: (N, (Lmax+1)^2, C)
        node_emb = emb["node_embedding"]
        
        # 1. Extract L=0 (Scalar) -> (N, C)
        scalars = node_emb.narrow(1, 0, 1).squeeze(1)
        
        features = [scalars]
        
        # 2. Extract Norms of L>0 -> (N, C)
        current_idx = 1
        for l in range(1, self.lmax + 1):
            length = 2 * l + 1
            vec = node_emb.narrow(1, current_idx, length) # (N, 2l+1, C)
            vec_norm = vec.norm(dim=1) 
            features.append(vec_norm)
            current_idx += length
            
        # 3. Concatenate all invariants
        x = torch.cat(features, dim=-1) # (N, C * (Lmax+1))
        
        # 4. MLP with Residuals
        x = self.proj(x)
        x = x + self.res1(x)
        pred = self.final(x).squeeze(-1)

        if gp_utils.initialized():
            pred = gp_utils.gather_from_model_parallel_region(pred, dim=0)

        return {"pred": pred}


class IQA_IntraDecompositionHead(nn.Module, HeadInterface):
    """Predict decomposed intra-atomic IQA terms from the last node embedding."""

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        dropout: float = 0.0,
        term_names: list[str] | None = None,
        aggregate_name: str = "iqa_intra_a",
        predict_aggregate: bool = True,
    ) -> None:
        super().__init__()
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax
        self.term_names = term_names or [
            "iqa_kinetic",
            "iqa_vne",
            "iqa_vee",
            "iqa_vee_c",
            "iqa_vee_x",
        ]
        self.aggregate_name = aggregate_name
        self.predict_aggregate = predict_aggregate

        if self.aggregate_name in self.term_names:
            raise ValueError("aggregate_name must not overlap with term_names")

        input_dim = self.sphere_channels

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.term_heads = nn.ModuleDict(
            {
                term_name: nn.Sequential(
                    nn.LayerNorm(self.hidden_channels),
                    nn.Linear(self.hidden_channels, 1),
                )
                for term_name in self.term_names
            }
        )

    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        node_emb = emb["node_embedding"]

        x = node_emb.narrow(1, 0, 1).squeeze(1)
        x = self.proj(x)

        outputs: dict[str, torch.Tensor] = {}
        component_preds: list[torch.Tensor] = []

        for term_name in self.term_names:
            pred = self.term_heads[term_name](x).squeeze(-1)
            if gp_utils.initialized():
                pred = gp_utils.gather_from_model_parallel_region(pred, dim=0)
            outputs[term_name] = {"pred": pred}
            component_preds.append(pred)

        if self.predict_aggregate:
            aggregate_pred = torch.stack(component_preds, dim=0).sum(dim=0)
            outputs[self.aggregate_name] = {"pred": aggregate_pred}

        return outputs


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
        edge_input_dim = backbone.edge_channels_list[0]  # distance_basis + 2*edge_channels
        
        # Projection from edge features to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
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
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, 1)
        )
    
    def forward(self, data: AtomicData, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # emb["edge_embedding"]: (num_edges, edge_input_dim)
        if "edge_embedding" not in emb:
            raise ValueError("IQA_Edge_Head requires 'edge_embedding' in emb dict. "
                           "Set backbone output_edge_features=True")
        
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
            nn.Linear(input_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
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
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, 1)
        )

    def forward(self, data: AtomicData, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # emb["node_embedding"]: (N, (Lmax+1)^2, C)
        node_emb = emb["node_embedding"]
        
        # 1. Extract L=0 (Scalar) -> (N, C)
        scalars = node_emb.narrow(1, 0, 1).squeeze(1)
        
        features = [scalars]
        
        # 2. Extract Norms of L>0 -> (N, C)
        current_idx = 1
        for l in range(1, self.lmax + 1):
            length = 2 * l + 1
            vec = node_emb.narrow(1, current_idx, length) # (N, 2l+1, C)
            vec_norm = vec.norm(dim=1) 
            features.append(vec_norm)
            current_idx += length
            
        # 3. Concatenate all invariants
        x = torch.cat(features, dim=-1) # (N, C * (Lmax+1))
        
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
            raise ValueError("IQA_Edge_Head2 requires 'edge_embedding' in emb dict. "
                           "Set backbone output_edge_features=True")
        
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
            lmax=self.backbone_lmax
        )

        # Modulation network: use invariant edge embeddings
        self.edge_modulation = nn.Sequential(
            nn.Linear(self.edge_embedding_dim, self.sphere_channels),
            nn.SiLU(),
            nn.Linear(self.sphere_channels, self.sphere_channels)
        )

        # Final linear projection to scalar
        self.linear = nn.Linear(self.sphere_channels, 1)

    def forward(self, data, emb):
        edge_index = emb["edge_index"]
        node_emb = emb["node_embedding"]  # (num_atoms, (lmax+1)^2, sphere_channels)
        edge_embedding = emb["edge_embedding"]  # (num_edges, edge_embedding_dim) - invariant

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
        modulation = self.edge_modulation(edge_embedding)  # (num_edges, sphere_channels)
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
            self.lmax_list[0],  # FAIRCHEM ADAPTATION: Scalar, not list (single resolution)
            self.mmax_list[0],  # FAIRCHEM ADAPTATION: Scalar, not list (single resolution)
            self.mappingReduced,
            internal_weights=(False if not self.use_m_share_rad else True),
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
            use_cuda_graph=self.backbone.use_cuda_graph_wigner
            and "cuda" in get_device_for_local_rank()
            and not self.training,
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
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

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

        # === EBDM-origin: Attention weights × message vectors ===
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


class IQA_IntraSO2MultiHead(SO2EquivariantGraphAttentionNodeEdgePrediction):
    """SO2 attention head that emits multiple node predictions (one per subterm).

    This subclass reuses most of the SO2 attention machinery but replaces the
    single `proj_nodes_2` with a `ModuleDict` of per-term `SO3_Linear` heads
    so each subterm gets its own final projection (as requested: different
    `proj_nodes_2` for each sub term).
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        term_names: list[str],
        aggregate_name: str = "iqa_intra_a",
        predict_aggregate: bool = False,
        edge_prediction: bool = False,
        edge_task_name: str = "iqa_inter_ab",
        sphere_channels: int | None = None,
        hidden_channels: int | None = None,
        num_heads: int = 8,
        attn_alpha_channels: int | None = None,
        attn_value_channels: int | None = None,
        out_degree: int = 0,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        term_normalizers: dict[str, dict[str, float]] | None = None,
        term_element_refs: dict[str, list[float]] | None = None,
        **kwargs,
    ) -> None:
        # Create parent with node_prediction enabled but single-node proj_nodes_2
        super().__init__(
            backbone=backbone,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels_edges=1 if edge_prediction else 0,
            output_channels_nodes=1,
            out_degree=out_degree,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            edge_prediction=edge_prediction,
            edge_task_name=edge_task_name,
            **kwargs,
        )

        self.term_names = list(term_names)
        self.aggregate_name = aggregate_name
        self.predict_aggregate = predict_aggregate

        if self.aggregate_name in self.term_names:
            raise ValueError("aggregate_name must not overlap with term_names")
        self._use_term_normalizers = False
        if term_normalizers is not None:
            missing_terms = [term for term in self.term_names if term not in term_normalizers]
            extra_terms = [term for term in term_normalizers.keys() if term not in self.term_names]
            if missing_terms or extra_terms:
                raise ValueError(
                    f"term_normalizers mismatch. Missing={missing_terms}, extra={extra_terms}"
                )
            for term in self.term_names:
                term_norm = term_normalizers[term]
                if "mean" not in term_norm or "rmsd" not in term_norm:
                    raise ValueError(
                        f"term_normalizers[{term}] must include mean and rmsd"
                    )
                self.register_buffer(
                    f"term_mean_{term}",
                    torch.tensor(term_norm["mean"], dtype=torch.get_default_dtype()),
                )
                self.register_buffer(
                    f"term_rmsd_{term}",
                    torch.tensor(term_norm["rmsd"], dtype=torch.get_default_dtype()),
                )
            self._use_term_normalizers = True

        self._use_term_element_refs = False
        if term_element_refs is not None:
            if term_normalizers is None:
                raise ValueError("term_element_refs requires term_normalizers to denormalize terms")
            missing_terms = [term for term in self.term_names if term not in term_element_refs]
            extra_terms = [term for term in term_element_refs.keys() if term not in self.term_names]
            if missing_terms or extra_terms:
                raise ValueError(
                    f"term_element_refs mismatch. Missing={missing_terms}, extra={extra_terms}"
                )
            for term in self.term_names:
                term_refs = term_element_refs[term]
                self.register_buffer(
                    f"term_elem_refs_{term}",
                    torch.tensor(term_refs, dtype=torch.get_default_dtype()),
                )
            self._use_term_element_refs = True

        # compute proj_hidden same as in parent
        proj_hidden = self.num_heads * max(self.attn_value_channels // 2, 1)

        # replace parent proj_nodes_2 with per-term modules
        # keep proj_nodes_1 shared
        self.proj_nodes_2_terms = nn.ModuleDict()
        for term in self.term_names:
            self.proj_nodes_2_terms[term] = SO3_Linear(
                proj_hidden, 1, lmax=self.lmax_list[0]
            )

        # disable the single proj_nodes_2 to avoid accidental use
        self.proj_nodes_2 = None

    def forward(self, data, emb):
        # reuse parent's forward up to x_nodes creation
        # We'll copy-paste the necessary upstream logic to obtain `x_nodes`.
        # For brevity and safety we call the parent forward but intercept the
        # intermediate `x_nodes` by re-running the parts that compute it.

        # Recompute rotations and message passing exactly as parent does
        edge_index = emb["edge_index"]
        node_embedding = emb["node_embedding"]
        edge_embedding = emb["edge_embedding"]
        edge_distance_vec = emb["edge_distance_vec"]

        num_edges = edge_index.shape[1]
        num_nodes = node_embedding.shape[0]

        x_source = node_embedding[edge_index[0]]
        x_target = node_embedding[edge_index[1]]
        x_message_data = torch.cat((x_source, x_target), dim=2)

        x_edge = edge_embedding.narrow(1, 0, self.edge_distance_channels)
        if self.use_atom_edge_embedding:
            source_element = data["atomic_numbers"][edge_index[0]]
            target_element = data["atomic_numbers"][edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((x_edge, source_embedding, target_embedding), dim=1)

        wigner, wigner_inv = self.backbone._get_rotmat_and_wigner(
            edge_distance_vec,
            use_cuda_graph=self.backbone.use_cuda_graph_wigner
            and "cuda" in get_device_for_local_rank()
            and not self.training,
        )
        x_message = torch.bmm(wigner, x_message_data)

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

        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

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

        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik,ik->bi", x_0_alpha, self.alpha_dot)

        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = alpha.view(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

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

        x_message = torch.bmm(wigner_inv, x_message)

        outputs: dict[str, dict] = {}

        if self.edge_prediction and self.output_channels_edges > 0:
            out_embedding_edges = self.proj_edges_2(self.proj_edges_1(x_message))
            out_embedding_edges = out_embedding_edges.narrow(
                1, self.num_irreps_passed, 2 * self.out_degree + 1
            )
            edge_pred = self._squeeze_scalar_output(out_embedding_edges)
            if gp_utils.initialized():
                edge_pred = gp_utils.gather_from_model_parallel_region(edge_pred, dim=0)
            outputs[self.edge_task_name] = {"edge_pred": edge_pred}

        # aggregate to nodes
        x_nodes = torch_geometric.utils.scatter(
            x_message,
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        # per-term projections
        proj_nodes = self.proj_nodes_1(x_nodes)
        component_preds: list[torch.Tensor] = []
        for term in self.term_names:
            out_embedding_nodes_term = self.proj_nodes_2_terms[term](proj_nodes)
            out_embedding_nodes_term = out_embedding_nodes_term.narrow(
                1, self.num_irreps_passed, 2 * self.out_degree + 1
            )
            node_pred = self._squeeze_scalar_output(out_embedding_nodes_term)
            if gp_utils.initialized():
                node_pred = gp_utils.gather_from_model_parallel_region(node_pred, dim=0)
            outputs[term] = {"node_pred": node_pred}
            component_preds.append(node_pred)

        if self.predict_aggregate:
            if self._use_term_normalizers:
                aggregate_pred = None
                for term, node_pred in zip(self.term_names, component_preds):
                    term_mean = getattr(self, f"term_mean_{term}")
                    term_rmsd = getattr(self, f"term_rmsd_{term}")
                    term_physical = node_pred * term_rmsd + term_mean
                    if self._use_term_element_refs:
                        term_refs = getattr(self, f"term_elem_refs_{term}")
                        term_refs = term_refs[data["atomic_numbers"]].to(term_physical.dtype)
                        if term_physical.ndim == 2 and term_physical.shape[1] == 1:
                            term_refs = term_refs.view(-1, 1)
                        term_physical = term_physical + term_refs
                    aggregate_pred = (
                        term_physical
                        if aggregate_pred is None
                        else aggregate_pred + term_physical
                    )
            else:
                aggregate_pred = torch.stack(component_preds, dim=0).sum(dim=0)
            outputs[self.aggregate_name] = {"node_pred": aggregate_pred}

        return outputs


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
    def _sum_nodes(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        energy_part = torch.zeros(
            num_graphs, device=values.device, dtype=values.dtype
        )
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
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
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
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "mean") -> None:
        super().__init__()
        """
        predict the isotropic and anisotropic parts of the stress tensor
        to ensure symmetry and then recompose back to the full stress tensor
        """
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
        node_scalar = self.scalar_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        iso_stress = torch.zeros(
            len(data_dict["natoms"]),
            device=node_scalar.device,
            dtype=node_scalar.dtype,
        )
        iso_stress.index_add_(0, data_dict["batch"], node_scalar.view(-1))

        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")
            # iso_stress = gp_utils.reduce_from_model_parallel_region(iso_stress)

        if self.reduce == "mean":
            iso_stress /= data_dict["natoms"]

        node_l2 = self.l2_linear(emb["node_embedding"].narrow(1, 0, 9))
        node_l2 = node_l2.narrow(1, 4, 5)
        node_l2 = node_l2.view(-1, 5).contiguous()

        aniso_stress = torch.zeros(
            (len(data_dict["natoms"]), 5),
            device=node_l2.device,
            dtype=node_l2.dtype,
        )
        aniso_stress.index_add_(0, data_dict["batch"], node_l2)
        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")
            # aniso_stress = gp_utils.reduce_from_model_parallel_region(aniso_stress)

        if self.reduce == "mean":
            aniso_stress /= data_dict["natoms"].unsqueeze(1)

        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)

        return {"stress": stress}
