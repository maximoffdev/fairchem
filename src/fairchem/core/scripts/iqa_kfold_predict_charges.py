"""Predict q(A) with the k fold models of a k-fold charge run.

* train: every molecule is predicted by the one fold model that did not see it
  (out-of-fold), so its charge errors match those on unseen compositions;
* other splits (val, test): mean over all fold models.

Molecules are predicted on each model's own on-the-fly graph, as in training. The
output ``.pt`` maps ``split -> {pkl basename -> (N,) charges in e}``.

Example
-------
    python -m fairchem.core.scripts.iqa_kfold_predict_charges \\
        --data-root .../sumformulasplit --fold-file .../kfold5_sumformula.csv \\
        --checkpoints fold0/inference_ckpt.pt ... fold4/inference_ckpt.pt \\
        --out .../sumformulasplit/kfold5_predicted_charges.pt
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import TYPE_CHECKING

import torch

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.iqa_pkl_dataset import IQAPKLDataset
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

CHARGE_TASK = "iqa_charge"


def predict_charges(
    predict_unit: MLIPPredictUnit, dataset: IQAPKLDataset, batch_size: int
) -> dict[str, torch.Tensor]:
    """pkl basename -> (N,) predicted charges."""
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(dataset), batch_size):
        idx = range(start, min(start + batch_size, len(dataset)))
        batch = data_list_collater([dataset[i] for i in idx])
        # per-task outputs; the public predict() concatenates tasks sharing a property
        q = type(predict_unit).predict.__wrapped__(predict_unit, batch)[CHARGE_TASK]
        q = q.detach().cpu().reshape(-1)
        for i, sample_idx in enumerate(idx):
            name = os.path.basename(dataset.file_paths[sample_idx])
            out[name] = q[batch.batch.cpu() == i].clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="dir with the split subdirs")
    ap.add_argument("--fold-file", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="in fold order")
    ap.add_argument("--ensemble-splits", nargs="*", default=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.fold_file) as f:
        folds = sorted({int(row["fold"]) for row in csv.DictReader(f)})
    if folds != list(range(len(args.checkpoints))):
        raise ValueError(
            f"{args.fold_file} has folds {folds} but {len(args.checkpoints)} "
            "checkpoints were given; pass one checkpoint per fold, in fold order"
        )

    settings = InferenceSettings()
    settings.external_graph_gen = False  # the model's own otf graph, as in training
    units = [
        load_predict_unit(ckpt, inference_settings=settings, device=args.device)
        for ckpt in args.checkpoints
    ]

    result: dict[str, dict[str, torch.Tensor]] = {"train": {}}
    train_dir = os.path.join(args.data_root, "train")
    for fold, unit in zip(folds, units):
        heldout = IQAPKLDataset(
            src=train_dir, fold_file=args.fold_file, include_folds=[fold]
        )
        result["train"].update(predict_charges(unit, heldout, args.batch_size))
        print(f"fold {fold}: {len(heldout)} out-of-fold train molecules")
    for split in args.ensemble_splits:
        dataset = IQAPKLDataset(src=os.path.join(args.data_root, split))
        per_model = [predict_charges(u, dataset, args.batch_size) for u in units]
        result[split] = {
            name: torch.stack([p[name] for p in per_model]).mean(0)
            for name in per_model[0]
        }
        print(f"{split}: {len(dataset)} molecules, mean of {len(units)} fold models")
    torch.save(result, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
