"""IQA heads: exact A<->B edge symmetries, total-charge conservation, fp32 outputs."""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule as get_molecule

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import (
    IQA_MultiTaskSO2Head,
    IQA_Node_Head2,
    SO2EquivariantGraphAttentionNodeEdgePrediction,
    _shift_to_total_charge,
    eSCNMDBackbone,
)
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


def _batch() -> AtomicData:
    return atomicdata_list_to_batch(
        [
            AtomicData.from_ase(
                input_atoms=get_molecule(name),
                task_name="iqa_pkl",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
            for name in ("H2O", "NH3")
        ]
    )


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
