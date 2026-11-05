"""Exact sum rules for predicted IQA subterm fractions.

A model that predicts each IQA subterm as a fraction of a reference energy rebuilds
the subterms as fraction x reference. The fractions are predicted independently, so
the rebuilt subterms do not add up to the predicted energy by themselves. The closure
adjusts the fractions so that they satisfy their sum rule exactly:

    intra:  sum_X f_X(A)   = 1                            (f_X = X(A,A) / E_intra(A))
    pair:   sum_X g_X(A,B) = E_inter(A,B) / V_nn(A,B) - 1   (g_X = X(A,B) / V_nn(A,B))

Then sum_X f_X(A) E_intra(A) = E_intra(A) and V_nn + sum_X g_X V_nn = E_inter for the
predicted energies.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import torch
from torch import nn
from typing_extensions import NotRequired

ClosureMode = Literal["proportional", "error_weighted"]


class TaskScaleConfig(TypedDict):
    """The task's normalizer (mean, rmsd) and, for atom tasks, its per-element
    references: physical value = rmsd * prediction + mean + element_references[Z]."""

    mean: float
    rmsd: float
    element_references: NotRequired[list[float]]


class FractionClosureConfig(TypedDict):
    """``mode`` -- how the residual of the sum rule is spread over the fractions:
    ``"proportional"`` rescales all fractions of an atom/pair by one common factor;
    ``"error_weighted"`` shifts fraction X by w_X * residual with w_X = rmsd_X^2 /
    sum rmsd^2, the smallest correction in the standardized units the model
    predicts in. ``scales`` holds a ``TaskScaleConfig`` for every task named below."""

    mode: ClosureMode
    scales: dict[str, TaskScaleConfig]
    intra_fractions: NotRequired[list[str]]
    pair_energy: NotRequired[str]
    pair_fractions: NotRequired[list[str]]


class TaskScale(nn.Module):
    """Maps a head's standardized output to physical units and back, as the task's
    ``Normalizer`` and ``AtomElementReferences`` do in the loss and at inference."""

    def __init__(self, config: TaskScaleConfig) -> None:
        super().__init__()
        unknown = set(config) - {"mean", "rmsd", "element_references"}
        if unknown:
            raise ValueError(f"Unknown task scale keys {sorted(unknown)}")
        if float(config["rmsd"]) <= 0:
            raise ValueError(f"rmsd must be > 0, got {config['rmsd']}")
        self.mean = float(config["mean"])
        self.rmsd = float(config["rmsd"])
        # Not persistent: the values come from the config, and checkpoints of heads
        # without a closure load with it and vice versa.
        refs = config.get("element_references")
        self.element_references: torch.Tensor | None
        if refs is None:
            self.element_references = None
        else:
            self.register_buffer(
                "element_references",
                torch.tensor([float(r) for r in refs], dtype=torch.float64),
                persistent=False,
            )

    def _references(
        self, prediction: torch.Tensor, atomic_numbers: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.element_references is None:
            return None
        if atomic_numbers is None:
            raise ValueError(
                "A task scale with element references needs atomic numbers"
            )
        return self.element_references[atomic_numbers].view_as(prediction)

    def to_physical(
        self, prediction: torch.Tensor, atomic_numbers: torch.Tensor | None
    ) -> torch.Tensor:
        value = prediction.double() * self.rmsd + self.mean
        refs = self._references(prediction, atomic_numbers)
        return value if refs is None else value + refs

    def to_standardized(
        self,
        value: torch.Tensor,
        atomic_numbers: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        refs = self._references(value, atomic_numbers)
        if refs is not None:
            value = value - refs
        return ((value - self.mean) / self.rmsd).to(dtype)


class FractionClosure(nn.Module):
    """Adjusts one group of fractions (all of an atom's, or all of a pair's) so that
    they sum to a given target per row. Works in float64 on physical fractions and
    returns standardized predictions again."""

    def __init__(
        self, mode: ClosureMode, scales: dict[str, TaskScaleConfig], tasks: list[str]
    ) -> None:
        super().__init__()
        if mode not in ("proportional", "error_weighted"):
            raise ValueError(f"Unknown fraction closure mode {mode}")
        if len(tasks) < 2:
            raise ValueError(f"A fraction closure needs >= 2 fractions, got {tasks}")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Duplicate fractions in {tasks}")
        missing = [task for task in tasks if task not in scales]
        if missing:
            raise ValueError(f"No task scale for fractions {missing}")
        self.mode = mode
        self.tasks = list(tasks)
        self.scales = nn.ModuleDict({task: TaskScale(scales[task]) for task in tasks})
        variances = torch.tensor(
            [self.scales[task].rmsd ** 2 for task in tasks], dtype=torch.float64
        )
        self.register_buffer("weights", variances / variances.sum(), persistent=False)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target_sum: torch.Tensor,
        atomic_numbers: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Standardized fraction predictions -> standardized closed ones."""
        physical = torch.stack(
            [
                self.scales[task].to_physical(predictions[task], atomic_numbers)
                for task in self.tasks
            ]
        )
        total = physical.sum(dim=0)
        target_sum = target_sum.double().view_as(total)
        if self.mode == "proportional":
            closed = physical * (target_sum / total)
        else:
            weights = self.weights.view(-1, *([1] * total.ndim))
            closed = physical + weights * (target_sum - total)
        return {
            task: self.scales[task].to_standardized(
                closed[i], atomic_numbers, predictions[task].dtype
            )
            for i, task in enumerate(self.tasks)
        }
