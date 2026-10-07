from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from fairchem.core.datasets.iqa_pkl_dataset import IQAPKLDataset, label_level

if TYPE_CHECKING:
    from pathlib import Path


def _write_pkl(root: Path, atomic_charges: list[float]) -> None:
    sample = {
        "pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.8, 0.0, 0.0]]),
        "atomic_numbers": np.array([8, 1, 1]),
        "edge_index": np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        "q_total": 0,
        "q(A)": np.array(atomic_charges),
    }
    with open(root / "sample.pkl", "wb") as f:
        pickle.dump(sample, f)


def test_atomic_charge_key_sets_atomic_charges_and_total_charge(tmp_path: Path) -> None:
    _write_pkl(tmp_path, [-0.2, 0.61, 0.591])  # sums to ~+1 with integration error
    data = IQAPKLDataset(src=str(tmp_path), atomic_charge_key="q(A)")[0]
    torch.testing.assert_close(
        data.atomic_charges, torch.tensor([-0.2, 0.61, 0.591], dtype=data.pos.dtype)
    )
    assert data.charge.tolist() == [1]


def test_charge_key_is_default_and_has_no_atomic_charges(tmp_path: Path) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.4])
    data = IQAPKLDataset(src=str(tmp_path))[0]
    assert data.charge.tolist() == [0]
    assert "atomic_charges" not in data


def test_charge_keys_are_mutually_exclusive(tmp_path: Path) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.4])
    with pytest.raises(ValueError, match="mutually exclusive"):
        IQAPKLDataset(src=str(tmp_path), charge_key="q_total", atomic_charge_key="q(A)")


def test_missing_atomic_charge_key_raises(tmp_path: Path) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.4])
    with pytest.raises(KeyError, match="q_A"):
        IQAPKLDataset(src=str(tmp_path), atomic_charge_key="q_A")[0]


def test_non_integer_atomic_charge_sum_raises(tmp_path: Path) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.7])
    with pytest.raises(ValueError, match="integer total charge"):
        IQAPKLDataset(src=str(tmp_path), atomic_charge_key="q(A)")[0]


def test_sigma_atomic_charge_adds_noise_preserving_total_charge(
    tmp_path: Path,
) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.4])
    torch.manual_seed(0)
    dataset = IQAPKLDataset(
        src=str(tmp_path), atomic_charge_key="q(A)", sigma_atomic_charge=0.05
    )
    clean = torch.tensor([-0.8, 0.4, 0.4])
    q1, q2 = dataset[0].atomic_charges, dataset[0].atomic_charges
    assert not torch.allclose(q1, clean)
    assert not torch.allclose(q1, q2)
    torch.testing.assert_close(q1.sum(), clean.sum())
    assert dataset[0].charge.tolist() == [0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_atomic_charge": -0.1, "atomic_charge_key": "q(A)"}, ">= 0"),
        ({"sigma_atomic_charge": 0.1}, "requires atomic_charge_key"),
    ],
)
def test_invalid_sigma_atomic_charge_raises(
    tmp_path: Path, kwargs: dict[str, float | str], match: str
) -> None:
    _write_pkl(tmp_path, [-0.8, 0.4, 0.4])
    with pytest.raises(ValueError, match=match):
        IQAPKLDataset(src=str(tmp_path), **kwargs)


LAGRANGIAN_KEY_MAPPING = {
    "energy": "e_total",
    "iqa_kinetic": "T(A)",
    "iqa_forces_direct": "Fn(A,SumB)",
    "iqa_vnn_ab": "Vnn(A,B)/2",
}


def _write_labeled_pkl(root: Path, lagrangian: list[float] | None) -> None:
    # Same water-like graph as _write_pkl: edges 0->1, 1->0, 0->2, 2->0.
    sample: dict[str, object] = {
        "pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.8, 0.0, 0.0]]),
        "atomic_numbers": np.array([8, 1, 1]),
        "edge_index": np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        "q_total": 0,
        "e_total": -76.0,
        "T(A)": np.array([[75.0], [0.5], [0.5]]),
        "Fn(A,SumB)": np.ones((3, 3)),
        "Vnn(A,B)/2": np.array([[1.0], [1.0], [2.0], [2.0]]),
    }
    if lagrangian is not None:
        sample["L(A)"] = np.array(lagrangian).reshape(-1, 1)
    with open(root / "sample.pkl", "wb") as f:
        pickle.dump(sample, f)


def test_lagrangian_cutoff_masks_atoms_and_their_edges(tmp_path: Path) -> None:
    _write_labeled_pkl(tmp_path, [1e-5, -2e-4, -8e-4])  # only atom 2 exceeds 5e-4
    data = IQAPKLDataset(
        src=str(tmp_path),
        key_mapping=LAGRANGIAN_KEY_MAPPING,
        lagrangian_cutoff=5e-4,
    )[0]
    assert torch.isfinite(data.energy).all()
    assert torch.isfinite(data.iqa_kinetic).tolist() == [True, True, False]
    assert torch.isfinite(data.iqa_forces_direct).all(dim=1).tolist() == [
        True,
        True,
        False,
    ]
    assert not torch.isfinite(data.iqa_forces_direct[2]).any()
    assert torch.isfinite(data.iqa_vnn_ab).tolist() == [True, True, False, False]


def test_lagrangian_cutoff_none_keeps_all_labels(tmp_path: Path) -> None:
    _write_labeled_pkl(tmp_path, [1e-5, -2e-4, -8e-4])
    data = IQAPKLDataset(src=str(tmp_path), key_mapping=LAGRANGIAN_KEY_MAPPING)[0]
    assert torch.isfinite(data.iqa_kinetic).all()
    assert torch.isfinite(data.iqa_vnn_ab).all()


def test_lagrangian_cutoff_without_lagrangian_raises(tmp_path: Path) -> None:
    _write_labeled_pkl(tmp_path, None)
    dataset = IQAPKLDataset(
        src=str(tmp_path),
        key_mapping=LAGRANGIAN_KEY_MAPPING,
        lagrangian_cutoff=5e-4,
    )
    with pytest.raises(KeyError, match=r"L\(A\)"):
        dataset[0]


def test_non_positive_lagrangian_cutoff_raises(tmp_path: Path) -> None:
    _write_labeled_pkl(tmp_path, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="lagrangian_cutoff"):
        IQAPKLDataset(src=str(tmp_path), lagrangian_cutoff=0.0)


@pytest.mark.parametrize(
    ("in_key", "level"),
    [
        ("e_total", "system"),
        ("T(A)", "atom"),
        ("Vne(A,A)", "atom"),
        ("Fn(A,SumB)", "atom"),
        ("|Mu(A)|", "atom"),
        ("E_IQA_Inter(A,B)/2", "edge"),
    ],
)
def test_label_level(in_key: str, level: str) -> None:
    assert label_level(in_key) == level
