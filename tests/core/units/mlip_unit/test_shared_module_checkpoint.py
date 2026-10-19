from __future__ import annotations

import re
import socket
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from fairchem.core.units.mlip_unit.mlip_unit import prefix_shared_module_keys

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class HeadHoldingBackbone(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.out = torch.nn.Linear(4, 1)


class ModelWithSharedBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.output_heads = torch.nn.ModuleDict(
            {"head": HeadHoldingBackbone(self.backbone)}
        )


@pytest.fixture()
def single_rank_process_group(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _ddp_and_ema() -> tuple[DistributedDataParallel, AveragedModel]:
    ddp = DistributedDataParallel(ModelWithSharedBackbone())
    return ddp, AveragedModel(ddp, multi_avg_fn=get_ema_multi_avg_fn(0.999))


def _save(path: Path) -> tuple[DistributedDataParallel, AveragedModel]:
    ddp, ema = _ddp_and_ema()
    with torch.no_grad():
        for param in ema.parameters():
            param.add_(1.0)
    dcp.save(
        {"model": get_model_state_dict(ddp), "ema": get_model_state_dict(ema)},
        checkpoint_id=str(path),
    )
    return ddp, ema


def _load_saved_state(
    path: Path, ddp: DistributedDataParallel, ema: AveragedModel
) -> dict[str, dict[str, torch.Tensor]]:
    state = {"model": get_model_state_dict(ddp), "ema": get_model_state_dict(ema)}
    dcp.load(state, checkpoint_id=str(path))
    return state


@pytest.mark.usefixtures("single_rank_process_group")
def test_shared_module_checkpoint_loads_strictly_after_prefixing(
    tmp_path: Path,
) -> None:
    saved_ddp, saved_ema = _save(tmp_path)
    ddp, ema = _ddp_and_ema()
    state = _load_saved_state(tmp_path, ddp, ema)

    prefix_shared_module_keys(ddp, state["model"])
    set_model_state_dict(ddp, state["model"])
    prefix_shared_module_keys(ema, state["ema"])
    set_model_state_dict(ema, state["ema"])

    for saved, loaded in ((saved_ddp, ddp), (saved_ema, ema)):
        for (name, expected), actual in zip(
            saved.state_dict().items(), loaded.state_dict().values(), strict=True
        ):
            assert torch.equal(expected, actual), name
    head_backbone = ddp.module.output_heads["head"].backbone
    assert head_backbone is ddp.module.backbone


@pytest.mark.usefixtures("single_rank_process_group")
def test_shared_module_checkpoint_fails_without_prefixing(tmp_path: Path) -> None:
    # Documents the torch bug the prefixing works around. If this starts passing,
    # torch renames shared-module keys itself and prefix_shared_module_keys must go.
    _save(tmp_path)
    ddp, ema = _ddp_and_ema()
    state = _load_saved_state(tmp_path, ddp, ema)
    with pytest.raises(
        RuntimeError, match=re.escape("output_heads.head.backbone.weight")
    ):
        set_model_state_dict(ddp, state["model"])
