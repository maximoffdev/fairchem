#!/usr/bin/env python
"""Recompute IQA normalizer mean/rmsd values from a pkl dataset.

Reproduces the statistics that ``fairchem.core.modules.normalization.normalizer.
fit_normalizers`` would compute for the active IQA heads, so the results can be
pasted straight into ``configs/uma/training_release/train_iqa_pretrain.yaml``.

For atom-level targets that use element references in
``configs/uma/training_release/tasks/iqa_decompose.yaml`` (currently
``iqa_intra_a``), the per-atom reference from the element-refs yaml is
subtracted from every atom before the mean/rmsd are taken -- exactly what
``AtomElementReferences.apply_refs`` does during training. Edge-level targets
(``iqa_inter_ab``) have no element reference, so their per-edge values are used
directly.

All pkl values are converted Hartree -> eV (matching ``ht2ev: true`` in the
dataset config), while the element references are already stored in eV. The rmsd
uses the Bessel (n-1) correction that ``fit_normalizers`` applies whenever the
mean is non-zero.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# 1 Hartree = 27.211386245988 eV  (matches iqa_pkl_dataset.Ht_to_eV)
HT_TO_EV = 27.211386245988

# label name -> (pkl key, elem_refs yaml key or None)
# Keys mirror `key_mapping` in configs/uma/training_release/dataset/iqa_components.yaml
# and the element_references blocks in tasks/iqa_decompose.yaml.
ATOM_TARGETS = {
    "iqa_intra_a": ("E_IQA_Intra(A)", "iqa_intra"),
    "iqa_inter_a": ("E_IQA_Inter(A)", None),
}
EDGE_TARGETS = {
    "iqa_inter_ab": "E_IQA_Inter(A,B)/2",
}

ALL_TARGETS = list(ATOM_TARGETS) + list(EDGE_TARGETS)

DEFAULT_DATA_DIR = (
    "/media/data/qm_dataset/data/Pipeline/pkl/M062X_Jun-cc-pVDZ_HCNOSPClF/"
    "HCNOSPClF_combined_datasets_filtered0.001/sumformulasplit/train"
)
DEFAULT_ELEM_REFS = os.path.join(
    os.path.dirname(__file__),
    "../../../../configs/uma/training_release/element_refs/"
    "iqa_isolated_atom_elem_refs.yaml",
)


def _load_elem_refs(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        refs = yaml.safe_load(f)
    out = {}
    for _, ref_key in ATOM_TARGETS.values():
        if ref_key is None:
            continue
        if ref_key not in refs:
            raise KeyError(
                f"'{ref_key}' not found in element refs file {path}. "
                f"Available: {sorted(refs)}"
            )
        out[ref_key] = np.asarray(refs[ref_key], dtype=np.float64)
    return out


def _accumulate_file(path: str, elem_refs: dict[str, np.ndarray]):
    """Return per-target (count, sum, sumsq) accumulators for one pkl file."""
    acc = {t: [0, 0.0, 0.0] for t in ALL_TARGETS}
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        keys = set(d.keys())
    except Exception:
        return acc, 1  # failed file

    z = None
    if "atomic_numbers" in keys:
        z = np.asarray(d["atomic_numbers"]).astype(np.int64).reshape(-1)

    for label, (pkl_key, ref_key) in ATOM_TARGETS.items():
        if pkl_key not in keys or z is None:
            continue
        vals = np.asarray(d[pkl_key], dtype=np.float64).reshape(-1) * HT_TO_EV
        if vals.shape[0] != z.shape[0]:
            continue
        if ref_key is not None:
            vals = vals - elem_refs[ref_key][z]  # dereference, refs already in eV
        acc[label][0] += vals.size
        acc[label][1] += float(vals.sum())
        acc[label][2] += float(np.square(vals).sum())

    for label, pkl_key in EDGE_TARGETS.items():
        if pkl_key not in keys:
            continue
        vals = np.asarray(d[pkl_key], dtype=np.float64).reshape(-1) * HT_TO_EV
        if vals.size == 0:
            continue
        acc[label][0] += vals.size
        acc[label][1] += float(vals.sum())
        acc[label][2] += float(np.square(vals).sum())

    return acc, 0


def _worker(paths: list[str], elem_refs: dict[str, np.ndarray]):
    total = {t: [0, 0.0, 0.0] for t in ALL_TARGETS}
    failed = 0
    for p in paths:
        acc, fail = _accumulate_file(p, elem_refs)
        failed += fail
        for t in ALL_TARGETS:
            total[t][0] += acc[t][0]
            total[t][1] += acc[t][1]
            total[t][2] += acc[t][2]
    return total, failed


def _chunks(seq: list[str], n: int):
    k = math.ceil(len(seq) / n)
    for i in range(0, len(seq), k):
        yield seq[i : i + k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--elem-refs", default=os.path.abspath(DEFAULT_ELEM_REFS))
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument(
        "--limit", type=int, default=None, help="only use first N files (debug)"
    )
    args = ap.parse_args()

    elem_refs = _load_elem_refs(args.elem_refs)

    # walk recursively, matching IQAPKLDataset's file discovery
    files = sorted(
        os.path.join(root, fn)
        for root, _, fnames in os.walk(args.data_dir)
        for fn in fnames
        if fn.endswith(".pkl")
    )
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} pkl files in {args.data_dir}")
    print(f"Element refs from {args.elem_refs}\n")

    total = {t: [0, 0.0, 0.0] for t in ALL_TARGETS}
    failed = 0
    n_chunks = max(args.workers * 4, 1)
    chunks = list(_chunks(files, n_chunks))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, c, elem_refs) for c in chunks]
        done = 0
        for fut in as_completed(futs):
            part, fail = fut.result()
            failed += fail
            for t in ALL_TARGETS:
                total[t][0] += part[t][0]
                total[t][1] += part[t][1]
                total[t][2] += part[t][2]
            done += 1
            print(f"  chunk {done}/{len(chunks)} done", end="\r", flush=True)
    print()
    if failed:
        print(f"WARNING: {failed} files could not be read and were skipped.\n")

    print("=" * 64)
    print("Config-ready normalizer values (paste into train_iqa_pretrain.yaml):")
    print("=" * 64)
    results = {}
    for t in ALL_TARGETS:
        n, s, s2 = total[t]
        if n == 0:
            print(f"# {t}: NO DATA")
            continue
        mean = s / n
        # Bessel (n-1) correction, matching fit_normalizers when mean != 0
        var = (s2 - n * mean * mean) / max(n - 1, 1)
        rmsd = math.sqrt(max(var, 0.0))
        results[t] = (mean, rmsd, n)
        print(f"normalizer_mean_{t}: {mean:.6g}")
        print(f"normalizer_rmsd_{t}: {rmsd:.6g}")
    print("=" * 64)
    print("\nSummary (label: mean, rmsd, n_samples):")
    for t, (mean, rmsd, n) in results.items():
        print(f"  {t:16s} mean={mean:12.5f}  rmsd={rmsd:12.5f}  n={n}")


if __name__ == "__main__":
    main()
