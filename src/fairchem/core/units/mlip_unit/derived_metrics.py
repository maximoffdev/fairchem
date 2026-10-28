"""
Validation metrics on quantities assembled from the predictions of several tasks and
compared against a batch label that need not be a task itself, e.g. the molecular
total energy sum_A E_intra(A) + sum_(A,B) E_inter(A,B)/2 against e_total, or the
subterm T(A) = f_T(A) * E_intra(A) of a model that predicts the fraction f_T.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fairchem.core.units.mlip_unit._metrics import Metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.modules.edge_matching import EdgeAlignment


@dataclass
class TaskPrediction:
    """A task's prediction in physical units (denormalized, references added back),
    one scalar per atom or per edge of the graph the model ran on."""

    value: torch.Tensor  # (rows,)
    system_index: torch.Tensor  # (rows,) system of each row
    # edge tasks on an otf graph: re-indexes the dataset's edge labels onto the rows
    edge_alignment: EdgeAlignment | None

    def label(self, batch: AtomicData, key: str) -> torch.Tensor:
        """The batch label `key` laid out on this prediction's rows, as (rows,)."""
        label = batch[key]
        if self.edge_alignment is not None:
            label = self.edge_alignment.apply(label)
        return _rows(label, self.value.numel(), key)


def _rows(tensor: torch.Tensor, rows: int, name: str) -> torch.Tensor:
    if tensor.numel() != rows:
        raise ValueError(
            f"'{name}' has shape {tuple(tensor.shape)}, expected one scalar per row "
            f"({rows})"
        )
    return tensor.reshape(rows)


def _mae(pred: torch.Tensor, target: torch.Tensor) -> Metrics:
    error = (pred.double() - target.double()).abs()
    return Metrics(
        metric=error.mean().item() if error.numel() else 0.0,
        total=error.sum().item(),
        numel=error.numel(),
    )


class DerivedMAE(ABC):
    """MAE of the batch label `target` against a quantity built from the predictions
    of `tasks`. Logged as val/mae/<name>."""

    def __init__(self, name: str, target: str, tasks: Sequence[str]) -> None:
        if not tasks:
            raise ValueError(f"Derived metric '{name}' needs at least one task")
        self.name = name
        self.target = target
        self.tasks = list(tasks)

    def check_task_levels(self, levels: Mapping[str, str]) -> None:
        """Raise unless every task exists and is predicted per atom or per edge."""
        missing = [task for task in self.tasks if task not in levels]
        if missing:
            raise ValueError(
                f"Derived metric '{self.name}' uses tasks {missing} that are not "
                f"configured (have {sorted(levels)})"
            )
        unsupported = [t for t in self.tasks if levels[t] not in ("atom", "edge")]
        if unsupported:
            raise ValueError(
                f"Derived metric '{self.name}' supports atom and edge tasks only, got "
                f"{ {t: levels[t] for t in unsupported} }"
            )

    @abstractmethod
    def __call__(
        self, batch: AtomicData, predictions: Mapping[str, TaskPrediction]
    ) -> Metrics:
        """Absolute errors of one batch, to be accumulated over the eval epoch."""


class SystemSumMAE(DerivedMAE):
    """Per-system label against the sum of the atom and edge predictions of `tasks`
    over each system's atoms and model-graph edges. Pairs the model graph leaves out
    (beyond the otf cutoff) count as zero."""

    def __call__(
        self, batch: AtomicData, predictions: Mapping[str, TaskPrediction]
    ) -> Metrics:
        num_systems = batch.natoms.numel()
        total = torch.zeros(num_systems, dtype=torch.float64, device=batch.pos.device)
        for task in self.tasks:
            prediction = predictions[task]
            total.index_add_(0, prediction.system_index, prediction.value.double())
        target = _rows(batch[self.target], num_systems, self.target)
        valid = torch.isfinite(target)
        return _mae(total[valid], target[valid])


class ProductMAE(DerivedMAE):
    """Atom or edge label against the product of the predictions of `tasks`, which
    must share one level (and graph). Rows whose label is not finite are skipped."""

    def check_task_levels(self, levels: Mapping[str, str]) -> None:
        super().check_task_levels(levels)
        task_levels = {levels[task] for task in self.tasks}
        if len(task_levels) != 1:
            raise ValueError(
                f"Derived metric '{self.name}' multiplies tasks of different levels: "
                f"{ {task: levels[task] for task in self.tasks} }"
            )

    def __call__(
        self, batch: AtomicData, predictions: Mapping[str, TaskPrediction]
    ) -> Metrics:
        first, *rest = (predictions[task] for task in self.tasks)
        product = first.value.double()
        for factor in rest:
            if factor.edge_alignment is not first.edge_alignment:
                raise ValueError(
                    f"Derived metric '{self.name}': tasks {self.tasks} were predicted "
                    f"on different edge graphs"
                )
            product = product * _rows(factor.value, product.numel(), self.name)
        target = first.label(batch, self.target)
        valid = torch.isfinite(target)
        return _mae(product[valid], target[valid])
