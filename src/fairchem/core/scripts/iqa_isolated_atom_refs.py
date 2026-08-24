"""Compute isolated-atom reference energies for IQA energy terms from a .pkl dataset.

The reference for each element is derived directly from the raw .pkl samples:

* Per-atom (intra-atomic) terms  -- e.g. ``E_IQA_Intra(A)``, ``T(A)``, ``Vne(A,A)``,
  ``Vee(A,A)`` -- are stored with one value per atom. The isolated-atom reference for
  an element is simply the mean of that atom-resolved quantity over all atoms of that
  element in the dataset.

* Molecular-total terms -- e.g. ``e_total`` -- are stored as a single scalar per sample.
  The references are obtained from a linear least-squares fit of the total against the
  per-element atom counts (the standard element linear-reference fit).

The .pkl values are assumed to be in Hartree (atomic units). Results are printed in both
Hartree and eV, and written to a YAML file under ``configs/uma/training_release/element_refs``
in the same layout as the existing ``*_elem_refs`` blocks: a flat list indexed by atomic
number (0..max_num_elements), values in eV, Hartree value in a comment, 0.0 for elements
not present in the dataset.

The output filename is made collision-safe: if the target file already exists, a numeric
suffix is appended so that no existing YAML is ever overwritten.

Example
-------
    python -m fairchem.core.scripts.iqa_isolated_atom_refs --input_dir /data/iqa/train
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

# 1 Hartree = 27.211386245988 eV  (matches fairchem.core.datasets.iqa_pkl_dataset)
HARTREE_TO_EV = 27.211386245988

# The 5 IQA energy terms (pkl keys) referenced by the training config
# configs/uma/training_release/dataset/iqa_components.yaml
DEFAULT_TERMS = [
    "e_total",
    "T(A)",
    "Vne(A,A)",
    "Vee(A,A)",
    "E_IQA_Intra(A)",
]

# Map known pkl keys to clean YAML block names (matches the config's output keys).
TERM_TO_REFNAME = {
    "e_total": "energy",
    "T(A)": "iqa_kinetic",
    "Vne(A,A)": "iqa_vne",
    "Vee(A,A)": "iqa_vee",
    "E_IQA_Intra(A)": "iqa_intra",
}

# Default output location, resolved relative to the repo (this file lives at
# src/fairchem/core/scripts/iqa_isolated_atom_refs.py).
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "uma"
    / "training_release"
    / "element_refs"
)


def _find_pkls(input_dir: str) -> list[str]:
    """Recursively collect non-empty .pkl files, matching IQAPKLDataset behavior."""
    paths = [
        p
        for p in glob.glob(os.path.join(input_dir, "**", "*.pkl"), recursive=True)
        if os.path.getsize(p) > 0
    ]
    paths.sort()
    return paths


def _symbol(z: int) -> str:
    return chemical_symbols[z] if 0 <= z < len(chemical_symbols) else f"Z{z}"


def _refname(term: str) -> str:
    """Turn a pkl key into a valid YAML block name ``<name>_elem_refs``."""
    base = TERM_TO_REFNAME.get(term)
    if base is None:
        base = "".join(c if c.isalnum() else "_" for c in term.lower()).strip("_")
        while "__" in base:
            base = base.replace("__", "_")
    return f"{base}_elem_refs"


def _nonclobbering_path(directory: Path, name: str) -> Path:
    """Return a path in `directory` for `name`, appending _1, _2, ... if it exists."""
    stem = Path(name).stem
    suffix = Path(name).suffix or ".yaml"
    candidate = directory / f"{stem}{suffix}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{suffix}"
        i += 1
    return candidate


def compute_references(file_paths: list[str], terms: list[str]) -> dict:
    """Scan pkl files and return {term: {"kind", "refs_ha": {Z: ha}, "meta": {...}}}."""
    # Per-atom accumulators: term -> {Z: [count, sum, sumsq]}
    atomic_acc: dict[str, dict[int, list[float]]] = {t: {} for t in terms}
    # Molecular accumulators: term -> (list of {Z: count}, list of total value)
    molec_counts: dict[str, list[dict[int, int]]] = {t: [] for t in terms}
    molec_totals: dict[str, list[float]] = {t: [] for t in terms}
    term_kind: dict[str, str] = {}
    missing_counts: dict[str, int] = {t: 0 for t in terms}

    for path in file_paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        z = np.asarray(d["atomic_numbers"]).reshape(-1).astype(int)
        n = z.size

        for term in terms:
            if term not in d:
                missing_counts[term] += 1
                continue
            arr = np.asarray(d[term], dtype=np.float64).reshape(-1)

            if arr.size == n and n > 1:
                kind = "atomic"
            elif arr.size == 1:
                kind = "molecular"
            elif arr.size == n and n == 1:
                # ambiguous single-atom sample; treat as atomic (value belongs to the atom)
                kind = "atomic"
            else:
                print(
                    f"  Skipping term {term!r} in {os.path.basename(path)}: "
                    f"unexpected length {arr.size} (natoms={n})"
                )
                continue

            prev = term_kind.setdefault(term, kind)
            if prev != kind:
                print(
                    f"  Warning: term {term!r} looks '{kind}' in {os.path.basename(path)} "
                    f"but was '{prev}' earlier; keeping '{prev}'."
                )
                kind = prev

            if kind == "atomic":
                acc = atomic_acc[term]
                for zi, vi in zip(z.tolist(), arr.tolist()):
                    slot = acc.setdefault(zi, [0.0, 0.0, 0.0])
                    slot[0] += 1.0
                    slot[1] += vi
                    slot[2] += vi * vi
            else:  # molecular
                unique, counts = np.unique(z, return_counts=True)
                molec_counts[term].append(dict(zip(unique.tolist(), counts.tolist())))
                molec_totals[term].append(float(arr[0]))

    # Reduce accumulators into per-element Hartree references.
    results: dict[str, dict] = {}
    for term in terms:
        entry = {"missing": missing_counts[term], "n_files": len(file_paths)}
        if term not in term_kind:
            entry["kind"] = None
            results[term] = entry
            continue

        kind = term_kind[term]
        entry["kind"] = kind
        if kind == "atomic":
            refs_ha, std_ha, counts = {}, {}, {}
            for zi, (count, s, ss) in atomic_acc[term].items():
                mean = s / count
                var = max(ss / count - mean * mean, 0.0)
                refs_ha[zi] = mean
                std_ha[zi] = var**0.5
                counts[zi] = int(count)
            entry.update(refs_ha=refs_ha, std_ha=std_ha, counts=counts)
        else:  # molecular linear reference fit
            rows = molec_counts[term]
            totals = np.asarray(molec_totals[term], dtype=np.float64)
            elements = sorted({zi for row in rows for zi in row})
            col = {zi: j for j, zi in enumerate(elements)}
            X = np.zeros((len(rows), len(elements)), dtype=np.float64)
            for i, row in enumerate(rows):
                for zi, c in row.items():
                    X[i, col[zi]] = c
            fit, _res, rank, _sv = np.linalg.lstsq(X, totals, rcond=None)
            atoms_per_element = X.sum(axis=0).astype(int)
            entry.update(
                refs_ha={zi: float(fit[col[zi]]) for zi in elements},
                counts={zi: int(atoms_per_element[col[zi]]) for zi in elements},
                rank=int(rank),
                n_elements=len(elements),
                n_samples=len(rows),
            )
        results[term] = entry
    return results


def print_report(results: dict, terms: list[str]) -> None:
    for term in terms:
        entry = results[term]
        print("=" * 68)
        print(f"Term: {term}")
        if entry["kind"] is None:
            print(f"  No usable data (missing in {entry['missing']}/{entry['n_files']} files).")
            print()
            continue
        if entry["missing"]:
            print(f"  Note: missing in {entry['missing']}/{entry['n_files']} files.")

        if entry["kind"] == "atomic":
            print("  Method: per-element mean of atom-resolved values")
            print(
                f"  {'Z':>3} {'El':>3} {'n_atoms':>10} "
                f"{'ref [Ha]':>16} {'ref [eV]':>16} {'std [Ha]':>14}"
            )
            for zi in sorted(entry["refs_ha"]):
                ha = entry["refs_ha"][zi]
                print(
                    f"  {zi:>3} {_symbol(zi):>3} {entry['counts'][zi]:>10} "
                    f"{ha:>16.8f} {ha * HARTREE_TO_EV:>16.8f} {entry['std_ha'][zi]:>14.8f}"
                )
        else:
            print(
                f"  Method: linear least-squares fit vs. element counts "
                f"({entry['n_samples']} samples, rank {entry['rank']}/{entry['n_elements']})"
            )
            if entry["rank"] < entry["n_elements"]:
                print(
                    "  Warning: fit is rank-deficient; per-element references are not "
                    "uniquely determined."
                )
            print(f"  {'Z':>3} {'El':>3} {'n_atoms':>10} {'ref [Ha]':>16} {'ref [eV]':>16}")
            for zi in sorted(entry["refs_ha"]):
                ha = entry["refs_ha"][zi]
                print(
                    f"  {zi:>3} {_symbol(zi):>3} {entry['counts'][zi]:>10} "
                    f"{ha:>16.8f} {ha * HARTREE_TO_EV:>16.8f}"
                )
        print()


def build_yaml(results: dict, terms: list[str], input_dir: str, max_num_elements: int) -> str:
    """Render the references as a YAML string in the *_elem_refs layout (values in eV)."""
    lines: list[str] = [
        "# Isolated-atom reference energies computed by iqa_isolated_atom_refs.py",
        f"# Source dataset: {input_dir}",
        "# Values are per-element references in eV, indexed by atomic number Z.",
        "# 0.0 marks elements not present in the source dataset.",
        "",
    ]
    for term in terms:
        entry = results[term]
        refname = _refname(term)
        if entry["kind"] is None:
            lines.append(f"# {refname}: term {term!r} had no usable data; block omitted.")
            lines.append("")
            continue

        method = (
            "per-element mean of atom-resolved values"
            if entry["kind"] == "atomic"
            else "linear least-squares fit vs. element counts"
        )
        lines.append(f"# from pkl key {term!r} ({method})")
        lines.append(f"{refname}:")
        refs_ha = entry["refs_ha"]

        z = 0
        while z <= max_num_elements:
            if z == 0:
                lines.append("# Index 0: placeholder")
                lines.append("- 0.0")
                z += 1
                continue
            if z in refs_ha:
                ha = refs_ha[z]
                ev = ha * HARTREE_TO_EV
                lines.append(f"# Index {z}: {_symbol(z)}  ({ha:.12g} Ha)")
                lines.append(f"- {ev:.11f}")
                z += 1
            else:
                # Collapse a run of absent elements into one comment.
                start = z
                while z <= max_num_elements and z not in refs_ha:
                    z += 1
                end = z - 1
                if end == start:
                    lines.append(f"# Index {start}: {_symbol(start)}  (not in dataset)")
                elif end - start <= 6:
                    syms = ", ".join(_symbol(k) for k in range(start, end + 1))
                    lines.append(f"# Index {start}-{end}: {syms}  (not in dataset)")
                else:
                    lines.append(f"# Index {start}-{end}: remaining elements (not in dataset)")
                lines.extend(["- 0.0"] * (end - start + 1))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compute isolated-atom reference energies for IQA terms from .pkl data."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .pkl files (searched recursively)"
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=DEFAULT_TERMS,
        help="pkl keys to compute references for (default: the 5 IQA energy terms)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the YAML file (default: configs/.../element_refs)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="iqa_isolated_atom_elem_refs.yaml",
        help="Base filename; a numeric suffix is added if the file already exists",
    )
    parser.add_argument(
        "--max_num_elements",
        type=int,
        default=118,
        help="Length of each reference list is max_num_elements + 1 (index 0..N)",
    )
    args = parser.parse_args()

    file_paths = _find_pkls(args.input_dir)
    if not file_paths:
        print(f"No .pkl files found in {args.input_dir}")
        return
    print(f"Found {len(file_paths)} .pkl files under {args.input_dir}\n")

    results = compute_references(file_paths, args.terms)
    print_report(results, args.terms)

    yaml_text = build_yaml(results, args.terms, args.input_dir, args.max_num_elements)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _nonclobbering_path(output_dir, args.output_name)
    out_path.write_text(yaml_text)
    print("=" * 68)
    print(f"Wrote reference YAML to: {out_path}")


if __name__ == "__main__":
    main()
