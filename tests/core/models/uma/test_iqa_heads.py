"""IQA heads and backbone options: exact A<->B edge symmetries, total-charge
conservation, fp32 outputs, and the locality options for transfer to larger systems."""

from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.build import molecule as get_molecule
from e3nn.o3 import rand_matrix

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.common.rotation import eulers_to_wigner
from fairchem.core.models.uma.escn_md import (
    IQA_IntraSO2MultiHead,
    IQA_MultiTaskSO2Head,
    IQA_Node_Head2,
    SO2EquivariantGraphAttentionNodeEdgePrediction,
    _shift_to_total_charge,
    eSCNMDBackbone,
)
from fairchem.core.models.uma.escn_md_block import smooth_degree
from fairchem.core.models.uma.nn.activation import SeparableS2Activation
from fairchem.core.modules.edge_matching import match_edges_by_node_pairs

BACKBONE_KWARGS = {
    "max_num_elements": 100,
    "sphere_channels": 8,
    "lmax": 2,
    "mmax": 2,
    "otf_graph": True,
    "edge_channels": 8,
    "num_distance_basis": 8,
    "num_layers": 2,
    "hidden_channels": 8,
    "dataset_list": ["iqa_pkl"],
    "always_use_pbc": False,
    "output_edge_features": True,
}


def _batch(rotation: torch.Tensor | None = None) -> AtomicData:
    data = []
    for name in ("H2O", "NH3"):
        atoms = get_molecule(name)
        if rotation is not None:
            atoms.positions = atoms.positions @ rotation.numpy().T
        data.append(
            AtomicData.from_ase(
                input_atoms=atoms,
                task_name="iqa_pkl",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
        )
    return atomicdata_list_to_batch(data)


@pytest.fixture()
def backbone_and_emb() -> tuple[eSCNMDBackbone, AtomicData, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(**BACKBONE_KWARGS)
    batch = _batch()
    with torch.no_grad():
        emb = backbone(batch)
    return backbone, batch, emb


def _reverse(emb: dict[str, torch.Tensor]) -> torch.Tensor:
    edge_index = emb["edge_index"]
    rev = match_edges_by_node_pairs(
        edge_index.flip(0), edge_index, emb["node_embedding"].shape[0]
    )
    assert bool((rev >= 0).all())
    return rev


def _drop_edges(
    emb: dict[str, torch.Tensor], drop: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Emulate a max_neighbors-truncated graph by removing some directed edges."""
    keep = torch.ones(emb["edge_index"].shape[1], dtype=torch.bool)
    keep[drop] = False
    return {
        **emb,
        "edge_index": emb["edge_index"][:, keep],
        "edge_distance_vec": emb["edge_distance_vec"][keep],
        "edge_embedding": emb["edge_embedding"][keep],
    }


@pytest.mark.parametrize("share_trunk", [True, False])
def test_multitask_head_edge_symmetries(backbone_and_emb, share_trunk: bool):
    backbone, batch, emb = backbone_and_emb
    head = IQA_MultiTaskSO2Head(
        backbone,
        node_task_names=["iqa_kinetic"],
        edge_task_names=["iqa_vne_ab", "iqa_vee_ab"],
        share_trunk=share_trunk,
        symmetric_edge_task_names=["iqa_vee_ab"],
        reversed_edge_tasks={"iqa_ven_ab": "iqa_vne_ab"},
    )
    with torch.no_grad():
        out = head(batch, emb)
    rev = _reverse(emb)
    vee = out["iqa_vee_ab"]["edge_pred"]
    vne = out["iqa_vne_ab"]["edge_pred"]
    ven = out["iqa_ven_ab"]["edge_pred"]
    assert torch.equal(vee, vee[rev])
    assert torch.equal(ven, vne[rev])
    assert not torch.allclose(vne, vne[rev])  # V_ne itself stays directed
    assert torch.equal(out["iqa_ven_ab"]["edge_index"], emb["edge_index"])


def test_reversed_messages_lookup_matches_flipped_trunk(backbone_and_emb):
    """On a symmetric graph, looking up the reverse edge's message agrees with running
    the trunk on the flipped graph. Each trunk call draws a random roll of the edge
    frames (init_edge_rot_euler_angles) and the S2 activation is only approximately
    equivariant to it, so agreement is up to the trunk's own call-to-call noise."""
    backbone, batch, emb = backbone_and_emb
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(backbone)
    flipped_emb = {
        **emb,
        "edge_index": emb["edge_index"].flip(0),
        "edge_distance_vec": -emb["edge_distance_vec"],
    }
    with torch.no_grad():
        x = head.compute_edge_messages(batch, emb)
        looked_up = head.reversed_edge_messages(batch, emb, x)
        flipped = head.compute_edge_messages(batch, flipped_emb)
        flipped_again = head.compute_edge_messages(batch, flipped_emb)
    call_noise = (flipped - flipped_again).abs().max()
    assert (looked_up - flipped).abs().max() < 3 * call_noise
    assert call_noise < 0.1 * flipped.abs().max()


def test_symmetric_edge_head_on_truncated_graph(backbone_and_emb):
    """Edges without their reverse in the graph still get a prediction; pairs present
    in both directions stay exactly symmetric."""
    backbone, batch, emb = backbone_and_emb
    dropped = _drop_edges(emb, torch.tensor([0, 5]))
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, node_prediction=False, symmetric_edge_prediction=True
    )
    with torch.no_grad():
        pred = head(batch, dropped)["iqa_inter_ab"]["edge_pred"]
    edge_index = dropped["edge_index"]
    rev = match_edges_by_node_pairs(
        edge_index.flip(0), edge_index, dropped["node_embedding"].shape[0]
    )
    paired = rev >= 0
    assert int((~paired).sum()) == 2
    assert torch.isfinite(pred).all()
    assert torch.equal(pred[paired], pred[rev[paired]])


def test_edge_symmetry_validation(backbone_and_emb):
    backbone, _, _ = backbone_and_emb
    with pytest.raises(ValueError, match="not in edge_task_names"):
        IQA_MultiTaskSO2Head(
            backbone, edge_task_names=["iqa_vne_ab"], symmetric_edge_task_names=["x"]
        )
    with pytest.raises(ValueError, match="must not also be predicted"):
        IQA_MultiTaskSO2Head(
            backbone,
            edge_task_names=["iqa_vne_ab", "iqa_ven_ab"],
            reversed_edge_tasks={"iqa_ven_ab": "iqa_vne_ab"},
        )
    with pytest.raises(ValueError, match="is symmetric"):
        IQA_MultiTaskSO2Head(
            backbone,
            edge_task_names=["iqa_vee_ab"],
            symmetric_edge_task_names=["iqa_vee_ab"],
            reversed_edge_tasks={"x": "iqa_vee_ab"},
        )
    with pytest.raises(ValueError, match="its own reverse"):
        SO2EquivariantGraphAttentionNodeEdgePrediction(
            backbone, symmetric_edge_prediction=True, reversed_edge_task_name="x"
        )


def _charged_batch() -> AtomicData:
    data = []
    for name, charge in (("H2O", 0), ("NH3", 1)):
        atoms = get_molecule(name)
        atoms.info["charge"] = charge
        data.append(
            AtomicData.from_ase(
                input_atoms=atoms,
                task_name="iqa_pkl",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
        )
    return atomicdata_list_to_batch(data)


def _molecule_sums(q: torch.Tensor, batch: AtomicData) -> torch.Tensor:
    return torch.zeros(len(batch.natoms), dtype=q.dtype).index_add(0, batch.batch, q)


def test_shift_to_total_charge_is_uniform_per_molecule():
    batch = _charged_batch()
    q = torch.randn(batch.pos.shape[0], dtype=torch.float64)
    shifted = _shift_to_total_charge(q, batch)
    torch.testing.assert_close(
        _molecule_sums(shifted, batch), batch.charge.to(torch.float64)
    )
    delta = shifted - q
    for i in range(len(batch.natoms)):
        d = delta[batch.batch == i]
        torch.testing.assert_close(d, d[:1].expand_as(d))


def test_charge_head_conserves_total_charge_in_fp32():
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(**BACKBONE_KWARGS)
    batch = _charged_batch()
    head = IQA_Node_Head2(backbone, dropout=0.0, conserve_total_charge=True)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        q = head(batch, backbone(batch))["pred"]
    assert q.dtype == torch.float32
    torch.testing.assert_close(
        _molecule_sums(q, batch), batch.charge.to(torch.float32), atol=1e-5, rtol=0
    )


@pytest.fixture()
def no_frame_roll(monkeypatch):
    """Fix the random roll of the edge frames, so outputs on different graphs compare."""
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)


def _chain(x_last: float) -> AtomicData:
    """H atoms at x = 0, 2 and x_last (A): with a 3 A cutoff atom 0 only sees atom 1."""
    atoms = Atoms("HHH", positions=[[0.0, 0, 0], [2.0, 0, 0], [x_last, 0, 0]])
    return atomicdata_list_to_batch(
        [
            AtomicData.from_ase(
                atoms,
                task_name="iqa_pkl",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
        ]
    )


@pytest.mark.parametrize("strictly_local", [True, False])
def test_strictly_local_node_sees_one_cutoff(no_frame_roll, strictly_local: bool):
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(
        **{**BACKBONE_KWARGS, "cutoff": 3.0}, strictly_local=strictly_local
    )
    with torch.no_grad():
        near = backbone(_chain(4.0))["node_embedding"][0]
        moved = backbone(_chain(4.4))["node_embedding"][0]  # atom 2 stays > 3 A from 0
    assert torch.equal(near, moved) == strictly_local


@pytest.mark.parametrize("gating", ["sigmoid", "softmax"])
def test_sigmoid_gated_edges_ignore_other_neighbours(
    backbone_and_emb, no_frame_roll, gating
):
    backbone, batch, emb = backbone_and_emb
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, node_prediction=False, attention_gating=gating
    )
    dropped = _drop_edges(emb, torch.tensor([0]))
    with torch.no_grad():
        full = head(batch, emb)["iqa_inter_ab"]["edge_pred"]
        less = head(batch, dropped)["iqa_inter_ab"]["edge_pred"]
    # the other edges into the dropped edge's target atom
    target = emb["edge_index"][1, 0]
    others = (emb["edge_index"][1] == target).nonzero().view(-1)[1:]
    same = torch.allclose(full[others], less[others - 1], atol=1e-6)
    assert same == (gating == "sigmoid")


def test_sqrt_degree_backbone_is_continuous_across_the_cutoff(no_frame_roll):
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(
        **{**BACKBONE_KWARGS, "cutoff": 3.0},
        aggregation_norm="sqrt_degree",
        degree_embedding=True,
        strictly_local=True,
    )
    with torch.no_grad():
        inside = backbone(_chain(2.0 + 2.999))["node_embedding"][1]
        outside = backbone(_chain(2.0 + 3.001))["node_embedding"][1]
    torch.testing.assert_close(inside, outside, atol=1e-4, rtol=0)


def test_smooth_degree_counts_neighbours_with_the_envelope():
    distance = torch.tensor([0.0, 1.5, 2.999, 3.0])
    degree = smooth_degree(distance, torch.tensor([0, 0, 1, 1]), 2, cutoff=3.0)
    assert degree[0] > 1.0
    assert degree[1] < 1e-6


def test_locality_option_validation(backbone_and_emb):
    backbone, _, _ = backbone_and_emb
    with pytest.raises(ValueError, match="aggregation_norm"):
        eSCNMDBackbone(**BACKBONE_KWARGS, aggregation_norm="mean")
    with pytest.raises(ValueError, match="attention_gating"):
        SO2EquivariantGraphAttentionNodeEdgePrediction(
            backbone, attention_gating="relu"
        )


# -------- Edge-frame activations and the random roll of the edge frame --------


def _roll_matrices(backbone: eSCNMDBackbone, gamma: float, n: int) -> torch.Tensor:
    """Rotation by gamma about the edge axis in the m-ordered truncated basis (fp64)."""
    lmax, mmax = backbone.lmax, backbone.mmax
    jd = [getattr(backbone, f"Jd_{l}").double() for l in range(lmax + 1)]
    angles = torch.full((n,), gamma, dtype=torch.float64)
    zeros = torch.zeros_like(angles)
    wigner = eulers_to_wigner((angles, zeros, zeros), 0, lmax, jd)
    idx = backbone.SO3_grid["lmax_lmax"].mapping.coefficient_idx(lmax, mmax)
    wigner = wigner[:, idx][:, :, idx]
    to_m = backbone.mappingReduced.to_m.double()
    return to_m @ wigner @ to_m.T


def _scaled(emb: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    """Untrained heads have tiny activations, where SiLU is nearly quadratic and the
    grid quadrature happens to be exact; scale up to the O(1) regime of trained nets."""
    return {**emb, "node_embedding": emb["node_embedding"] * scale}


def _fixed_roll(monkeypatch, turns: float) -> None:
    monkeypatch.setattr(torch, "rand_like", lambda x, **kw: torch.full_like(x, turns))


@pytest.mark.parametrize(
    "activation_type, grid_resolution, tol",
    [
        ("so2_gate", None, 1e-12),
        ("gate", None, 1e-12),
        ("sep_s2", None, 0.3),  # quadrature aliasing: approximate only
        ("sep_s2", 32, 1e-3),
    ],
)
def test_edge_frame_activation_roll_equivariance(
    backbone_and_emb, activation_type: str, grid_resolution: int | None, tol: float
):
    backbone, _, _ = backbone_and_emb
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, activation_type=activation_type, s2_grid_resolution=grid_resolution
    )
    act = head.act.double()
    n, num_coefficients = 32, backbone.mappingReduced.to_m.shape[0]
    torch.manual_seed(0)
    x = 3.0 * torch.randn(
        n, num_coefficients, head.hidden_channels, dtype=torch.float64
    )
    num_gates = head.hidden_channels * (
        backbone.lmax if activation_type == "gate" else 1
    )
    gating = torch.randn(n, num_gates, dtype=torch.float64)
    roll = _roll_matrices(backbone, 1.3, n)
    with torch.no_grad():
        reference = act(gating, x)
        err = (act(gating, torch.bmm(roll, x)) - torch.bmm(roll, reference)).abs().max()
    assert err / reference.abs().max() < tol


@pytest.mark.parametrize("activation_type", ["so2_gate", "gate"])
def test_exact_head_predictions_do_not_depend_on_the_frame_roll(
    backbone_and_emb, monkeypatch, activation_type: str
):
    backbone, batch, emb = backbone_and_emb
    emb = _scaled(emb, 30.0)
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, activation_type=activation_type
    )
    preds = []
    for turns in (0.0, 0.3, 0.6):
        _fixed_roll(monkeypatch, turns)
        with torch.no_grad():
            out = head(batch, emb)
        preds.append(
            (out["iqa_inter_ab"]["edge_pred"], out["iqa_intra_a"]["node_pred"])
        )
    for edge, node in preds[1:]:
        torch.testing.assert_close(edge, preds[0][0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(node, preds[0][1], atol=1e-5, rtol=1e-5)


def test_grid_activation_head_depends_on_the_frame_roll(backbone_and_emb, monkeypatch):
    """Documents why the exact activation is the default: the grid activation's
    output moves with the arbitrary roll, by far more than the tolerance above."""
    backbone, batch, emb = backbone_and_emb
    emb = _scaled(emb, 100.0)
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, activation_type="sep_s2"
    )
    preds = []
    for turns in (0.0, 0.3):
        _fixed_roll(monkeypatch, turns)
        with torch.no_grad():
            preds.append(head(batch, emb)["iqa_inter_ab"]["edge_pred"])
    assert (preds[0] - preds[1]).abs().max() > 1e-4


def test_exact_head_is_rotation_invariant(monkeypatch):
    _fixed_roll(monkeypatch, 0.0)
    torch.manual_seed(0)
    # The default grid feed-forward of the backbone is only approximately
    # rotation-equivariant; the IQA configs use the exact spectral one.
    backbone = eSCNMDBackbone(**BACKBONE_KWARGS, ff_type="spectral")
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, activation_type="so2_gate"
    )
    torch.manual_seed(1)
    rotation = rand_matrix()
    batch, rotated = _batch(), _batch(rotation)
    with torch.no_grad():
        out = head(batch, _scaled(backbone(batch), 30.0))
        out_rot = head(rotated, _scaled(backbone(rotated), 30.0))
    perm = match_edges_by_node_pairs(
        out["iqa_inter_ab"]["edge_index"],
        out_rot["iqa_inter_ab"]["edge_index"],
        batch.num_nodes,
    )
    assert bool((perm >= 0).all())
    torch.testing.assert_close(
        out["iqa_inter_ab"]["edge_pred"],
        out_rot["iqa_inter_ab"]["edge_pred"][perm],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        out["iqa_intra_a"]["node_pred"],
        out_rot["iqa_intra_a"]["node_pred"],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "activation_type, depends", [("so2_gate", True), ("sep_s2", True), ("gate", False)]
)
def test_m_nonzero_edge_frame_features_reach_scalar_outputs(
    backbone_and_emb, no_frame_roll, activation_type: str, depends: bool
):
    """Scalar outputs only read m=0 features after the second SO(2) convolution, so a
    gate activation makes every m!=0 input dead; the other activations couple them in."""
    backbone, batch, emb = backbone_and_emb
    emb = _scaled(emb, 30.0)
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(
        backbone, activation_type=activation_type
    )

    def zero_m_nonzero(module, args):
        gating, x = args
        x = x.clone()
        x[:, backbone.lmax + 1 :] = 0
        return gating, x

    with torch.no_grad():
        full = head(batch, emb)["iqa_inter_ab"]["edge_pred"]
        handle = head.act.register_forward_pre_hook(zero_m_nonzero)
        zeroed = head(batch, emb)["iqa_inter_ab"]["edge_pred"]
        handle.remove()
    assert (not torch.equal(full, zeroed)) == depends


def test_default_activation_keeps_pre_fix_checkpoints_loadable(backbone_and_emb):
    """Checkpoints trained before the activation fix store neither activation weights
    nor an activation_type, so the default must stay the parameter-free legacy one."""
    backbone, _, _ = backbone_and_emb
    head = SO2EquivariantGraphAttentionNodeEdgePrediction(backbone)
    assert isinstance(head.act, SeparableS2Activation)
    assert not [key for key in head.state_dict() if key.startswith("act.")]


def test_activation_type_validation(backbone_and_emb):
    backbone, _, _ = backbone_and_emb
    with pytest.raises(ValueError, match="activation_type"):
        SO2EquivariantGraphAttentionNodeEdgePrediction(backbone, activation_type="relu")
    with pytest.raises(ValueError, match="s2_grid_resolution"):
        SO2EquivariantGraphAttentionNodeEdgePrediction(
            backbone, activation_type="gate", s2_grid_resolution=16
        )
    with pytest.raises(NotImplementedError, match="use_s2_act_attn"):
        SO2EquivariantGraphAttentionNodeEdgePrediction(backbone, use_s2_act_attn=True)


def test_intra_multi_term_head_shares_the_trunk(backbone_and_emb, no_frame_roll):
    backbone, batch, emb = backbone_and_emb
    head = IQA_IntraSO2MultiHead(
        backbone, term_names=["iqa_kinetic", "iqa_vne"], predict_aggregate=True
    )
    with torch.no_grad():
        out = head(batch, emb)
    assert set(out) == {"iqa_kinetic", "iqa_vne", "iqa_intra_a"}
    for value in out.values():
        assert value["node_pred"].shape == (batch.num_nodes,)
        assert value["node_pred"].dtype == torch.float32
    torch.testing.assert_close(
        out["iqa_intra_a"]["node_pred"],
        out["iqa_kinetic"]["node_pred"] + out["iqa_vne"]["node_pred"],
    )
