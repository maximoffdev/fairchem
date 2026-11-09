"""Tests for the IQA point-charge references and the fp32 output projections."""

from __future__ import annotations

import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.iqa_pkl_dataset import Ht_to_eV, angstrom_to_bohr
from fairchem.core.models.uma.escn_md import (
    IQA_Vnn_Analytic_Head,
    _project_fp32,
)
from fairchem.core.modules.iqa_coulomb import point_charge_pair_energy
from fairchem.core.modules.normalization.element_references import (
    AtomElementReferences,
    NeutralPointChargeEdgeReferences,
    PredictedChargeEdgeReferences,
)


@pytest.fixture()
def water() -> AtomicData:
    atoms = Atoms(
        "OHH", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    )
    data = AtomicData.from_ase(atoms)
    # complete directed graph without self-loops, the IQA pkl layout
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    data.edge_index = torch.tensor(pairs).T
    data.pbc = torch.zeros(1, 3, dtype=torch.bool)
    return data


def edge_distances(data: AtomicData) -> torch.Tensor:
    src, dst = data.edge_index
    pos = data.pos.double()
    return torch.linalg.norm(pos[src] - pos[dst], dim=-1)


def test_point_charge_pair_energy_matches_coulomb():
    r_ang = torch.tensor([1.0, 2.5], dtype=torch.float64)
    q1, q2 = torch.tensor([8, 1]), torch.tensor([1, 6])
    expected = Ht_to_eV(0.5 * q1 * q2 / angstrom_to_bohr(r_ang))
    torch.testing.assert_close(point_charge_pair_energy(q1, q2, r_ang), expected)
    hartree = point_charge_pair_energy(
        q1, q2, r_ang, pair_scale=1.0, output_in_ev=False
    )
    torch.testing.assert_close(hartree, q1 * q2 / angstrom_to_bohr(r_ang))


@pytest.mark.parametrize("sign", [-1, 1])
def test_neutral_point_charge_baseline(water: AtomicData, sign: int):
    refs = NeutralPointChargeEdgeReferences(sign)
    z = water.atomic_numbers
    src, dst = water.edge_index
    expected = sign * point_charge_pair_energy(z[src], z[dst], edge_distances(water))
    torch.testing.assert_close(refs.baseline(water, water.edge_index, {}), expected)


def test_neutral_point_charge_roundtrip(water: AtomicData):
    refs = NeutralPointChargeEdgeReferences(-1)
    labels = torch.randn(water.edge_index.shape[1]) * 50
    residual = refs.apply_refs(water, labels, water.edge_index, {})
    assert residual.dtype == labels.dtype
    torch.testing.assert_close(
        refs.undo_refs(water, residual, water.edge_index, {}), labels
    )


def test_neutral_point_charge_follows_given_edge_layout(water: AtomicData):
    """The baseline is laid out on the edge_index it is given (e.g. an otf graph)."""
    refs = NeutralPointChargeEdgeReferences(1)
    perm = torch.randperm(water.edge_index.shape[1])
    torch.testing.assert_close(
        refs.baseline(water, water.edge_index[:, perm], {}),
        refs.baseline(water, water.edge_index, {})[perm],
    )


def test_neutral_point_charge_baselines_cancel_in_inter_energy(water: AtomicData):
    """V_ne + V_en + V_ee baselines plus the exact V_nn sum to zero per edge."""
    ne = NeutralPointChargeEdgeReferences(-1).baseline(water, water.edge_index, {})
    ee = NeutralPointChargeEdgeReferences(1).baseline(water, water.edge_index, {})
    z = water.atomic_numbers
    src, dst = water.edge_index
    nn = point_charge_pair_energy(z[src], z[dst], edge_distances(water))
    torch.testing.assert_close(ne + ne + ee + nn, torch.zeros_like(nn))


def test_neutral_point_charge_rejects_pbc(water: AtomicData):
    water.pbc = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="aperiodic"):
        NeutralPointChargeEdgeReferences(-1).baseline(water, water.edge_index, {})


def test_neutral_point_charge_rejects_bad_sign():
    with pytest.raises(ValueError, match="sign"):
        NeutralPointChargeEdgeReferences(2)


def test_atom_element_references_round_only_the_result(water: AtomicData):
    """Refs of O(1e4) eV are subtracted in float64, so the float32 residual equals
    the exactly computed difference rounded once."""
    refs_table = torch.zeros(119, dtype=torch.float64)
    refs_table[1], refs_table[8] = -13.6056980659, -2041.38731245
    refs = AtomElementReferences(refs_table)
    water.atomic_numbers_full = water.atomic_numbers
    labels = torch.tensor([-2045.123, -14.21, -13.99], dtype=torch.float32)
    residual = refs.apply_refs(water, labels)
    exact = (labels.double() - refs_table[water.atomic_numbers]).float()
    assert torch.equal(residual, exact)
    torch.testing.assert_close(refs.undo_refs(water, residual), labels)


def test_vnn_head_matches_point_charge_energy(water: AtomicData):
    head = IQA_Vnn_Analytic_Head(backbone=None)
    src, dst = water.edge_index
    emb = {
        "edge_index": water.edge_index,
        "edge_distance_vec": water.pos[src] - water.pos[dst],
    }
    vnn = head(water, emb)["iqa_vnn_ab"]["edge_pred"]
    z = water.atomic_numbers
    expected = point_charge_pair_energy(z[src], z[dst], edge_distances(water)).float()
    torch.testing.assert_close(vnn, expected)


def test_project_fp32_escapes_bf16_autocast():
    layer = torch.nn.Linear(16, 1)
    x = torch.randn(4, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert layer(x).dtype == torch.bfloat16
        out = _project_fp32(x.bfloat16(), layer)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, layer(x.bfloat16().float()))


def test_predicted_charge_baseline_uses_model_charges(water: AtomicData):
    q = torch.tensor([-0.8, 0.4, 0.4], requires_grad=True)
    predictions = {"iqa_charge": {"pred": q}}
    refs = PredictedChargeEdgeReferences("iqa_charge")
    src, dst = water.edge_index
    expected = point_charge_pair_energy(
        q.detach().double()[src], q.detach().double()[dst], edge_distances(water)
    )
    baseline = refs.baseline(water, water.edge_index, predictions)
    torch.testing.assert_close(baseline, expected)
    assert not baseline.requires_grad  # detached by default


def test_predicted_charge_baseline_can_train_the_charges(water: AtomicData):
    q = torch.tensor([-0.8, 0.4, 0.4], requires_grad=True)
    refs = PredictedChargeEdgeReferences("iqa_charge", detach_charges=False)
    labels = torch.zeros(water.edge_index.shape[1])
    residual = refs.apply_refs(
        water, labels, water.edge_index, {"iqa_charge": {"pred": q}}
    )
    residual.abs().sum().backward()
    assert q.grad is not None
    assert bool(q.grad.abs().sum() > 0)


def test_predicted_charge_baseline_needs_the_charge_prediction(water: AtomicData):
    refs = PredictedChargeEdgeReferences("iqa_charge")
    with pytest.raises(KeyError, match="iqa_charge"):
        refs.baseline(water, water.edge_index, {"iqa_inter_ab": {}})
