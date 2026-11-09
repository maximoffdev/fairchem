from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.derived_metrics import ProductMAE, SystemSumMAE
from fairchem.core.units.mlip_unit.mlip_unit import (
    OutputSpec,
    Task,
    get_task_predictions,
)


def _molecule(natoms: int, energy: float, seed: int) -> AtomicData:
    """A molecule on its complete directed edge graph, with random atom/edge labels."""
    gen = torch.Generator().manual_seed(seed)
    u, v = zip(*[(i, j) for i in range(natoms) for j in range(natoms) if i != j])
    edge_index = torch.tensor([u, v])
    num_edges = edge_index.shape[1]
    data = AtomicData(
        pos=torch.rand(natoms, 3, generator=gen),
        atomic_numbers=torch.full((natoms,), 6),
        cell=torch.zeros(1, 3, 3),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
        natoms=torch.tensor([natoms]),
        edge_index=edge_index,
        cell_offsets=torch.zeros(num_edges, 3),
        nedges=torch.tensor([num_edges]),
        charge=torch.zeros(1, dtype=torch.long),
        spin=torch.zeros(1, dtype=torch.long),
        fixed=torch.zeros(natoms, dtype=torch.long),
        tags=torch.zeros(natoms, dtype=torch.long),
        energy=torch.tensor([energy]),
        forces=None,
        stress=None,
        batch=None,
        sid=str(seed),
        dataset="iqa_pkl",
    )
    data.subterm = torch.randn(natoms, generator=gen)
    data.subterm_ab = torch.randn(num_edges, generator=gen)
    return data


def _task(name: str, level: str, mean: float, rmsd: float) -> Task:
    return Task(
        name=name,
        level=level,
        property="node_pred" if level == "atom" else "edge_pred",
        out_spec=OutputSpec(dim=[1], dtype="float32"),
        normalizer=Normalizer(mean=mean, rmsd=rmsd),
        datasets=["iqa_pkl"],
    )


@pytest.fixture()
def batch() -> AtomicData:
    return atomicdata_list_to_batch([_molecule(3, -10.0, 0), _molecule(4, -20.0, 1)])


def test_total_energy_sums_atom_and_edge_predictions_per_system(batch):
    tasks = [_task("intra", "atom", 1.0, 2.0), _task("inter", "edge", -0.5, 3.0)]
    normed_intra = torch.randn(7)
    normed_inter = torch.randn(18)  # 6 + 12 edges, the dataset's graph
    predictions = {
        "intra": {"node_pred": normed_intra},
        "inter": {"edge_pred": normed_inter},
    }
    metric = SystemSumMAE(
        name="total_energy", target="energy", tasks=["intra", "inter"]
    )
    result = metric(batch, get_task_predictions(tasks, predictions, batch))

    intra = normed_intra * 2.0 + 1.0
    inter = normed_inter * 3.0 - 0.5
    total = torch.stack(
        [intra[:3].sum() + inter[:6].sum(), intra[3:].sum() + inter[6:].sum()]
    )
    expected = (total - torch.tensor([-10.0, -20.0])).abs()
    assert result.numel == 2
    assert result.total == pytest.approx(expected.sum().item(), rel=1e-5)


def test_product_on_otf_graph_compares_against_realigned_labels(batch):
    tasks = [_task("frac_ab", "edge", 0.0, 1.0), _task("scale_ab", "edge", 0.0, 1.0)]
    # a permuted subset of the dataset's edges, spanning both systems
    subset = torch.tensor([10, 2, 7, 15, 0])
    edge_index_model = batch.edge_index[:, subset]
    frac, scale = torch.randn(5), torch.randn(5)
    predictions = {
        "frac_ab": {"edge_pred": frac, "edge_index": edge_index_model},
        "scale_ab": {"edge_pred": scale, "edge_index": edge_index_model},
    }
    batch.subterm_ab[subset[1]] = float("nan")  # masked label, skipped
    metric = ProductMAE(name="subterm_ab", target="subterm_ab", tasks=list(predictions))
    result = metric(batch, get_task_predictions(tasks, predictions, batch))

    keep = torch.tensor([0, 2, 3, 4])
    expected = (frac * scale - batch.subterm_ab[subset])[keep].abs()
    assert result.numel == 4
    assert result.total == pytest.approx(expected.sum().item(), rel=1e-5)


def test_total_energy_counts_otf_edges_toward_their_own_system(batch):
    tasks = [_task("inter", "edge", 0.0, 1.0)]
    # system 1's edges first: the otf graph need not be grouped by system
    subset = torch.tensor([8, 6, 1])
    values = torch.tensor([1.0, 2.0, 4.0])
    predictions = {
        "inter": {"edge_pred": values, "edge_index": batch.edge_index[:, subset]}
    }
    metric = SystemSumMAE(name="total_energy", target="energy", tasks=["inter"])
    result = metric(batch, get_task_predictions(tasks, predictions, batch))

    assert result.total == pytest.approx(abs(4.0 + 10.0) + abs(3.0 + 20.0))


def test_check_task_levels_rejects_missing_and_mixed_tasks():
    levels = {"intra": "atom", "inter": "edge", "energy": "system"}
    with pytest.raises(ValueError, match="not configured"):
        SystemSumMAE(name="e", target="energy", tasks=["intra", "x"]).check_task_levels(
            levels
        )
    with pytest.raises(ValueError, match="atom and edge tasks only"):
        SystemSumMAE(name="e", target="energy", tasks=["energy"]).check_task_levels(
            levels
        )
    with pytest.raises(ValueError, match="different levels"):
        ProductMAE(
            name="p", target="subterm", tasks=["intra", "inter"]
        ).check_task_levels(levels)
