"""Fit IQA normalizer mean/rmsd values from a pkl dataset split.

Samples are loaded through ``IQAPKLDataset`` with the ``key_mapping`` and
``lagrangian_cutoff`` of a dataset config (default: ``iqa_train`` of
``configs/uma/training_release/dataset/iqa_components.yaml``), and references are
subtracted with the same modules the tasks use, so units, Lagrangian masking and
baselines match training by construction:

- atom energies: ``AtomElementReferences`` (isolated-atom refs per element),
- pair subterms V_ne/V_en/V_ee(A,B): ``NeutralPointChargeEdgeReferences``,
- E_inter(A,B): additionally ``PredictedChargeEdgeReferences`` evaluated with the AIM
  q(A) labels (the charges the model's own predictions converge to).

Vector targets (atomic dipoles, forces) are rotation-equivariant, so their normalizer
mean is 0 and their rmsd is the root mean square over all components.

Every target is reported both raw and referenced. The rmsd uses the Bessel (n-1)
correction that ``fit_normalizers`` applies whenever the mean is non-zero. Values
masked to NaN (poorly integrated atoms and their edges) are skipped, like the loss
does. The largest |values| are listed with their files to expose outliers.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import yaml

from fairchem.core.datasets.iqa_pkl_dataset import IQAPKLDataset
from fairchem.core.modules.normalization.element_references import (
    AtomElementReferences,
    NeutralPointChargeEdgeReferences,
    PredictedChargeEdgeReferences,
)

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../configs/uma/training_release"
)
DEFAULT_DATASET_CONFIG = os.path.join(CONFIG_DIR, "dataset/iqa_components.yaml")
DEFAULT_ELEM_REFS = os.path.join(
    CONFIG_DIR, "element_refs/iqa_isolated_atom_elem_refs.yaml"
)
DEFAULT_DATA_DIR = (
    "/media/data/qm_dataset/data/Pipeline/pkl/M062X_Jun-cc-pVDZ_HCNOSPClF/"
    "HCNOSPClF_combined_datasets_filtered0.001/sumformulasplit/train"
)

# label -> element-refs yaml key, mirroring the element_references of the tasks
ATOM_ELEMENT_REFS: dict[str, str] = {
    "iqa_kinetic": "iqa_kinetic",
    "iqa_vne": "iqa_vne",
    "iqa_vee": "iqa_vee",
    "iqa_intra_a": "iqa_intra",
}
# iqa_charge keeps the identity normalizer when its head conserves the total charge
ATOM_RAW = ["iqa_inter_a", "iqa_charge"]
VECTOR_LABELS = ["dipole_vector", "iqa_forces_direct"]
# label -> sign of the neutral point-charge baseline sign * Z_A Z_B / 2R
EDGE_POINT_CHARGE_SIGNS: dict[str, int] = {
    "iqa_vne_ab": -1,
    "iqa_ven_ab": -1,
    "iqa_vee_ab": 1,
}
EDGE_RAW = ["iqa_inter_ab", "iqa_vnn_ab"]
# edge label -> charge label whose q_A q_B / 2R baseline it is also reported against
EDGE_CHARGE_REFS: dict[str, str] = {"iqa_inter_ab": "iqa_charge"}
ALL_LABELS = (
    list(ATOM_ELEMENT_REFS)
    + ATOM_RAW
    + list(EDGE_POINT_CHARGE_SIGNS)
    + EDGE_RAW
    + VECTOR_LABELS
)
NUM_LARGEST = 5

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    largest: list[tuple[float, str]] = field(default_factory=list)

    def add(self, values: torch.Tensor, path: str) -> None:
        values = values[torch.isfinite(values)].double()
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.total += float(values.sum())
        self.total_sq += float(values.square().sum())
        self._keep_largest([(float(values.abs().max()), path)])

    def merge(self, other: Moments) -> None:
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        self._keep_largest(other.largest)

    def _keep_largest(self, items: list[tuple[float, str]]) -> None:
        self.largest = heapq.nlargest(NUM_LARGEST, self.largest + items)

    @property
    def mean(self) -> float:
        return self.total / self.count

    @property
    def rmsd(self) -> float:
        # Bessel (n-1) correction, matching fit_normalizers when mean != 0
        var = (self.total_sq - self.count * self.mean**2) / max(self.count - 1, 1)
        return math.sqrt(max(var, 0.0))

    @property
    def rms(self) -> float:
        """Root mean square about zero, the rmsd of a zero-mean normalizer."""
        return math.sqrt(self.total_sq / self.count)


Accumulators = dict[str, Moments]
# label -> files that lack it (training would refuse these files)
MissingLabels = dict[str, list[str]]

_dataset: IQAPKLDataset
_atom_refs: dict[str, AtomElementReferences]
_edge_refs: dict[str, NeutralPointChargeEdgeReferences]
_charge_refs: dict[str, PredictedChargeEdgeReferences]


def _init_worker(
    dataset: IQAPKLDataset,
    atom_refs: dict[str, AtomElementReferences],
    edge_refs: dict[str, NeutralPointChargeEdgeReferences],
    charge_refs: dict[str, PredictedChargeEdgeReferences],
) -> None:
    global _dataset, _atom_refs, _edge_refs, _charge_refs
    _dataset, _atom_refs, _edge_refs, _charge_refs = (
        dataset,
        atom_refs,
        edge_refs,
        charge_refs,
    )


def _accumulate(indices: list[int]) -> tuple[Accumulators, MissingLabels]:
    acc: Accumulators = {}
    missing: MissingLabels = {}
    for idx in indices:
        sample: AtomicData = _dataset[idx]
        path = _dataset.file_paths[idx]
        # the backbone sets this on the batch before the loss reads the atom refs
        sample.atomic_numbers_full = sample.atomic_numbers
        for label in ALL_LABELS:
            if label not in _dataset.key_mapping:
                continue
            if label not in sample:
                missing.setdefault(label, []).append(path)
                continue
            values: torch.Tensor = getattr(sample, label)
            acc.setdefault(f"{label}:raw", Moments()).add(values, path)
            if label in _charge_refs:
                # the AIM labels stand in for the model's predicted charges
                charges = {
                    "iqa_charge": {"pred": getattr(sample, EDGE_CHARGE_REFS[label])}
                }
                charge_referenced = _charge_refs[label].apply_refs(
                    sample, values.double(), sample.edge_index, charges
                )
                acc.setdefault(f"{label}:charge_referenced", Moments()).add(
                    charge_referenced, path
                )
            if label in _atom_refs:
                referenced = _atom_refs[label].apply_refs(sample, values.double())
            elif label in _edge_refs:
                referenced = _edge_refs[label].apply_refs(
                    sample, values.double(), sample.edge_index, {}
                )
            else:
                continue
            acc.setdefault(f"{label}:referenced", Moments()).add(referenced, path)
    return acc, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    ap.add_argument("--dataset-key", default="iqa_train")
    ap.add_argument("--elem-refs", default=DEFAULT_ELEM_REFS)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=None, help="only use first N files")
    args = ap.parse_args()

    with open(args.dataset_config) as f:
        dataset_cfg = yaml.safe_load(f)[args.dataset_key]
    key_mapping: dict[str, str] = {
        out_key: in_key
        for out_key, in_key in dataset_cfg["key_mapping"].items()
        if out_key in ALL_LABELS
    }
    lagrangian_cutoff: float | None = dataset_cfg["lagrangian_cutoff"]
    dataset = IQAPKLDataset(
        src=args.data_dir,
        key_mapping=key_mapping,
        ht2ev=dataset_cfg["ht2ev"],
        lagrangian_cutoff=lagrangian_cutoff,
        # files lacking a label are listed below instead of aborting the fit
        allow_missing_labels=True,
    )

    with open(args.elem_refs) as f:
        refs_yaml = yaml.safe_load(f)
    atom_refs = {
        label: AtomElementReferences(torch.tensor(refs_yaml[key], dtype=torch.float64))
        for label, key in ATOM_ELEMENT_REFS.items()
    }
    edge_refs = {
        label: NeutralPointChargeEdgeReferences(sign)
        for label, sign in EDGE_POINT_CHARGE_SIGNS.items()
    }
    missing_charge_labels = set(EDGE_CHARGE_REFS.values()) - set(key_mapping)
    if missing_charge_labels:
        raise KeyError(f"key_mapping lacks the charge labels {missing_charge_labels}")
    charge_refs = {
        label: PredictedChargeEdgeReferences(charge_task="iqa_charge")
        for label in EDGE_CHARGE_REFS
    }

    n_files = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"{n_files} pkl files in {args.data_dir}")
    print(
        f"dataset config {args.dataset_config} [{args.dataset_key}], "
        f"lagrangian_cutoff={lagrangian_cutoff}, element refs {args.elem_refs}\n"
    )

    chunk = max(1, math.ceil(n_files / (args.workers * 8)))
    chunks = [list(range(i, min(i + chunk, n_files))) for i in range(0, n_files, chunk)]
    total: Accumulators = {}
    missing: MissingLabels = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(dataset, atom_refs, edge_refs, charge_refs),
    ) as ex:
        for done, (part, part_missing) in enumerate(
            ex.map(_accumulate, chunks), start=1
        ):
            for key, moments in part.items():
                total.setdefault(key, Moments()).merge(moments)
            for label, paths in part_missing.items():
                missing.setdefault(label, []).extend(paths)
            print(f"  chunk {done}/{len(chunks)}", end="\r", flush=True)
    print()
    for label, paths in sorted(missing.items()):
        print(
            f"WARNING: {len(paths)} file(s) lack '{label}' and would abort training: "
            + ", ".join(os.path.basename(p) for p in paths[:10])
        )

    print(
        f"{'target':34s} {'n':>10s} {'mean':>12s} {'rmsd':>12s} {'rms':>12s}"
        "  largest |x| (file)"
    )
    for key in sorted(total):
        m = total[key]
        top = ", ".join(f"{v:.1f} ({os.path.basename(p)})" for v, p in m.largest[:3])
        print(
            f"{key:34s} {m.count:10d} {m.mean:12.6g} {m.rmsd:12.6g} {m.rms:12.6g}  {top}"
        )

    print(
        "\n# Config-ready values (referenced where a reference exists, raw otherwise)"
    )
    for label in ALL_LABELS:
        key = (
            f"{label}:referenced" if f"{label}:referenced" in total else f"{label}:raw"
        )
        if key not in total:
            continue
        if label in VECTOR_LABELS:
            print(f"normalizer_mean_{label}: 0.0  # {key}, vector")
            print(f"normalizer_rmsd_{label}: {total[key].rms:.6g}")
            continue
        print(f"normalizer_mean_{label}: {total[key].mean:.6g}  # {key}")
        print(f"normalizer_rmsd_{label}: {total[key].rmsd:.6g}")
    print("\n# With PredictedChargeEdgeReferences (charge-referenced edge labels)")
    for label in EDGE_CHARGE_REFS:
        key = f"{label}:charge_referenced"
        print(f"normalizer_mean_{label}: {total[key].mean:.6g}  # {key}")
        print(f"normalizer_rmsd_{label}: {total[key].rmsd:.6g}")


if __name__ == "__main__":
    main()
