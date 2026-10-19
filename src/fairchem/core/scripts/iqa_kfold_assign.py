"""Assign the pkls of an IQA split to k folds, grouped by sum formula.

The train/val/test split of the IQA data holds out whole sum formulas (see
``IQA_Project/split_sumformula.py``), so a fold must do the same: otherwise conformers
of one composition leak between folds and the out-of-fold error of a model is far
more optimistic than its error on new molecules. Groups are shuffled with a fixed seed
and assigned largest first to the fold with the fewest molecules, which balances the
folds to within one group.

Writes a CSV with columns ``file,fold,formula`` that ``IQAPKLDataset(fold_file=...,
include_folds=... | exclude_folds=...)`` reads.

Example
-------
    python -m fairchem.core.scripts.iqa_kfold_assign --data-dir .../sumformulasplit/train \\
        --k 5 --out .../sumformulasplit/kfold5_sumformula.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor


def sum_formula(path: str) -> str:
    with open(path, "rb") as f:
        z = pickle.load(f)["atomic_numbers"].view(-1).long().tolist()
    return "_".join(f"{el}:{n}" for el, n in sorted(collections.Counter(z).items()))


def assign_folds(formulas: dict[str, str], k: int, seed: int) -> dict[str, int]:
    """file -> fold, keeping every sum formula within one fold."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for file, formula in formulas.items():
        groups[formula].append(file)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    # stable sort by size after the shuffle: largest groups first, ties in random order
    order.sort(key=lambda g: len(groups[g]), reverse=True)
    sizes = [0] * k
    fold_of: dict[str, int] = {}
    for formula in order:
        fold = min(range(k), key=lambda i: sizes[i])
        sizes[fold] += len(groups[formula])
        for file in groups[formula]:
            fold_of[file] = fold
    return fold_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = ap.parse_args()

    paths = sorted(
        os.path.join(root, fn)
        for root, _, fnames in os.walk(args.data_dir)
        for fn in fnames
        if fn.endswith(".pkl")
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        formulas = dict(
            zip(
                (os.path.basename(p) for p in paths),
                ex.map(sum_formula, paths, chunksize=256),
            )
        )
    if len(formulas) != len(paths):
        raise ValueError("Duplicate pkl basenames; fold files are keyed by basename")
    fold_of = assign_folds(formulas, args.k, args.seed)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "fold", "formula"])
        for file in sorted(fold_of):
            writer.writerow([file, fold_of[file], formulas[file]])

    counts = collections.Counter(fold_of.values())
    print(f"{len(paths)} pkls, {len(set(formulas.values()))} sum formulas, k={args.k}")
    for fold in range(args.k):
        groups = {formulas[f] for f, g in fold_of.items() if g == fold}
        print(f"  fold {fold}: {counts[fold]} molecules, {len(groups)} formulas")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
