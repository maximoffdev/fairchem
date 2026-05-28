"""Utilities for predicting IQA components from PKL datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.iqa_pkl_dataset import IQAPKLDataset
from fairchem.core.datasets.atomic_data import AtomicData


def _get_task_name(predict_unit: Any, task_name: str | None) -> str:
    datasets = list(predict_unit.dataset_to_tasks.keys())
    if task_name is not None:
        if task_name not in datasets:
            raise ValueError(
                f"Unknown task_name={task_name!r}. Available datasets: {datasets}"
            )
        return task_name
    if len(datasets) == 1:
        return datasets[0]
    raise ValueError(
        "Multiple datasets found in checkpoint. Please pass task_name. "
        f"Available datasets: {datasets}"
    )


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().view(-1).tolist()


def _get_backbone(predict_unit: Any) -> Any | None:
    model = getattr(predict_unit, "model", None)
    if model is None:
        return None
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "backbone", None)


def _get_graph_params(
    predict_unit: Any, radius: float | None, max_neigh: int | None
) -> tuple[float, int]:
    backbone = _get_backbone(predict_unit)
    if radius is None:
        radius = getattr(backbone, "cutoff", None) if backbone is not None else None
    if max_neigh is None:
        max_neigh = (
            getattr(backbone, "max_neighbors", None) if backbone is not None else None
        )
    if radius is None:
        radius = 6.0
    if max_neigh is None:
        max_neigh = 50
    return radius, max_neigh


def _resolve_indices(
    dataset: IQAPKLDataset, input_path: Path, max_items: int | None
) -> list[int]:
    if input_path.is_file():
        target = str(input_path.resolve())
        matches = [
            i
            for i, path in enumerate(dataset.file_paths)
            if str(Path(path).resolve()) == target
        ]
        if not matches:
            raise ValueError(f"PKL file {input_path} not found under dataset root.")
        indices = matches
    else:
        indices = list(range(len(dataset)))

    if max_items is not None:
        indices = indices[:max_items]
    return indices


def predict_iqa_pkl(
    predict_unit: Any,
    input_path: str | Path,
    task_name: str | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """
    Predict IQA atom- and edge-level energies for PKL inputs.

    Args:
        input_path: Directory with .pkl files or a single .pkl file.
        task_name: Dataset/task name from the checkpoint (e.g., iqa_pkl).
        max_items: Optional cap on number of structures to process.

    Returns:
        A dictionary ready to be serialized as JSON.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    dataset_root = path if path.is_dir() else path.parent
    task_name = _get_task_name(predict_unit, task_name)

    dataset = IQAPKLDataset(src=str(dataset_root), name=task_name)
    indices = _resolve_indices(dataset, path, max_items)

    results = {
        "task_name": task_name,
        "input_path": str(path),
        "structures": [],
    }

    for idx in indices:
        data = dataset[idx]
        batch = data_list_collater([data])
        pred = predict_unit.predict(batch)

        predictions: dict[str, list[float]] = {}
        for task in predict_unit.dataset_to_tasks[task_name]:
            if task.property in pred:
                predictions[task.name] = _to_list(pred[task.property])

        entry = {
            "index": idx,
            "sid": getattr(data, "sid", str(idx)),
            "source_path": dataset.file_paths[idx],
            "natoms": int(data.natoms.item()),
            "nedges": int(data.nedges.item()),
            "atomic_numbers": data.atomic_numbers.detach().cpu().tolist(),
            "edge_index": data.edge_index.detach().cpu().t().tolist(),
            "predictions": predictions,
        }
        results["structures"].append(entry)

    return results


class IQACalculator(Calculator):
    """ASE calculator that exposes IQA atom- and edge-level outputs."""

    def __init__(
        self,
        predict_unit: Any,
        task_name: str | None = None,
        radius: float | None = None,
        max_neigh: int | None = None,
        molecule_cell_size: float | None = 10.0,
    ) -> None:
        super().__init__()

        self.predictor = predict_unit
        self.task_name = _get_task_name(predict_unit, task_name)
        self.radius, self.max_neigh = _get_graph_params(
            predict_unit, radius, max_neigh
        )
        self.molecule_cell_size = molecule_cell_size

        self.implemented_properties = [
            task.name for task in predict_unit.dataset_to_tasks[self.task_name]
        ]

    def calculate(self, atoms, properties, system_changes) -> None:
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        Calculator.calculate(self, atoms, properties, system_changes)

        data = AtomicData.from_ase(
            atoms,
            r_edges=True,
            radius=self.radius,
            max_neigh=self.max_neigh,
            molecule_cell_size=self.molecule_cell_size,
            r_data_keys=["spin", "charge"],
            task_name=self.task_name,
        )
        batch = data_list_collater([data])
        pred = self.predictor.predict(batch)

        self.results = {}
        for task in self.predictor.dataset_to_tasks[self.task_name]:
            if task.property in pred:
                self.results[task.name] = pred[task.property].detach().cpu().numpy()

        self.results["edge_index"] = data.edge_index.detach().cpu().t().numpy()
        self.results["atomic_numbers"] = data.atomic_numbers.detach().cpu().numpy()
        self.results["natoms"] = np.array([int(data.natoms.item())])
        self.results["nedges"] = np.array([int(data.nedges.item())])
