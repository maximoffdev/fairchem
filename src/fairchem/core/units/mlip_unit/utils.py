"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig

from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.modules.normalization.element_references import (
    AtomElementReferences,
    ElementReferences,
    PointChargeEdgeReferences,
    TaskReferences,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def predicted_edge_index(
    task_output: dict[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor | None:
    """The graph an edge head predicted on, if it reported one."""
    if not isinstance(task_output, dict):
        return None
    return task_output.get("edge_index")


def apply_task_references(
    task: Task,
    batch: AtomicData,
    tensor: torch.Tensor,
    predictions: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Subtract the task's references from a (model-graph aligned) label tensor."""
    refs = task.element_references
    if refs is None:
        return tensor
    if isinstance(refs, PointChargeEdgeReferences):
        edge_index = _references_edge_index(batch, predictions[task.name])
        return refs.apply_refs(batch, tensor, edge_index, predictions)
    return _node_or_system_references(refs).apply_refs(batch, tensor)


def undo_task_references(
    task: Task,
    batch: AtomicData,
    tensor: torch.Tensor,
    predictions: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Add the task's references back onto a (denormalized) prediction tensor."""
    refs = task.element_references
    if refs is None:
        return tensor
    if isinstance(refs, PointChargeEdgeReferences):
        edge_index = _references_edge_index(batch, predictions[task.name])
        return refs.undo_refs(batch, tensor, edge_index, predictions)
    return _node_or_system_references(refs).undo_refs(batch, tensor)


def _node_or_system_references(
    refs: TaskReferences,
) -> ElementReferences | AtomElementReferences:
    if not isinstance(refs, (ElementReferences, AtomElementReferences)):
        raise TypeError(f"Unsupported task references type: {type(refs).__name__}")
    return refs


def _references_edge_index(
    batch: AtomicData, task_output: dict[str, torch.Tensor] | torch.Tensor
) -> torch.Tensor:
    """Edge layout of an edge task's predictions: the model graph if the head
    reported one (labels are re-indexed onto it), the dataset's otherwise."""
    edge_index = predicted_edge_index(task_output)
    return batch.edge_index if edge_index is None else edge_index


def load_inference_model(
    checkpoint_location: str,
    overrides: dict | None = None,
    use_ema: bool = False,
    return_checkpoint: bool = True,
) -> tuple[torch.nn.Module, MLIPInferenceCheckpoint] | torch.nn.Module:
    checkpoint: MLIPInferenceCheckpoint = torch.load(
        checkpoint_location, map_location="cpu", weights_only=False
    )

    if overrides is not None:
        checkpoint.model_config = update_configs(checkpoint.model_config, overrides)

    model = hydra.utils.instantiate(checkpoint.model_config)
    if use_ema:
        model = torch.optim.swa_utils.AveragedModel(model)
        model_dict = model.state_dict()
        ema_state_dict = checkpoint.ema_state_dict

        n_averaged = ema_state_dict["n_averaged"]
        del model_dict["n_averaged"]
        del ema_state_dict["n_averaged"]

        matched_dict = match_state_dict(model_dict, ema_state_dict)

        matched_dict["n_averaged"] = n_averaged

        load_state_dict(model, matched_dict, strict=True)
    else:
        load_state_dict(model, checkpoint.model_state_dict, strict=True)

    return (model, checkpoint) if return_checkpoint is True else model


def load_tasks(checkpoint_location: str) -> list[Task]:
    """
    Load tasks from a checkpoint file.

    Args:
        checkpoint_location (str): Path to the checkpoint file.

    Returns:
        list[Task]: A list of instantiated Task objects from the checkpoint's tasks_config.
    """
    checkpoint: MLIPInferenceCheckpoint = torch.load(
        checkpoint_location, map_location="cpu", weights_only=False
    )
    return [
        hydra.utils.instantiate(task_config) for task_config in checkpoint.tasks_config
    ]


@contextmanager
def tf32_context_manager():
    # Store the original settings
    original_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_float32_matmul_precision = torch.get_float32_matmul_precision()
    try:
        # Set the desired settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        # Revert to the original settings
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn
        torch.set_float32_matmul_precision(original_float32_matmul_precision)


def update_configs(original_config, new_config):
    updated_config = deepcopy(original_config)
    for k, v in new_config.items():
        is_dict_config = (isinstance(v, (dict, DictConfig))) and (
            isinstance(updated_config[k], (dict, DictConfig))
        )
        if is_dict_config and k in updated_config:
            updated_config[k] = update_configs(updated_config[k], v)
        else:
            updated_config[k] = v
    return updated_config
