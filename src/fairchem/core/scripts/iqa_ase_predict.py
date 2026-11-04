from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_changes
from ase.io import read

from fairchem.core import load_predict_unit
from fairchem.core.calculate import IQACalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict IQA components for ASE-readable molecules."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MLIP checkpoint (e.g., inference_ckpt.pt).",
    )
    parser.add_argument(
        "--ase-input",
        required=True,
        help="ASE-readable structure file (xyz, traj, pdb, etc.).",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="ASE frame index (default: ':' for all frames).",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Dataset/task name in the checkpoint (e.g., iqa_pkl).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device selection (default: auto).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Neighbor cutoff radius for graph construction.",
    )
    parser.add_argument(
        "--max-neigh",
        type=int,
        default=None,
        help="Maximum neighbors for graph construction.",
    )
    parser.add_argument(
        "--molecule-cell-size",
        type=float,
        default=10.0,
        help="Size of cubic cell for molecule centering.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Default total charge when not in ASE info.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="Default spin when not in ASE info.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path (prints to stdout if omitted).",
    )
    return parser.parse_args()


def _ensure_info(atoms, charge: int, spin: int) -> None:
    if "charge" not in atoms.info:
        atoms.info["charge"] = charge
    if "spin" not in atoms.info:
        atoms.info["spin"] = spin


def main() -> None:
    args = _parse_args()

    settings = inference_settings_default()
    settings.external_graph_gen = True

    predict_unit = load_predict_unit(
        args.model,
        inference_settings=settings,
        device=args.device,
    )

    calculator = IQACalculator(
        predict_unit,
        task_name=args.task_name,
        radius=args.radius,
        max_neigh=args.max_neigh,
        molecule_cell_size=args.molecule_cell_size,
    )

    atoms_list = read(args.ase_input, index=args.index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    results = {
        "task_name": calculator.task_name,
        "input_path": str(Path(args.ase_input)),
        "structures": [],
    }

    for idx, atoms in enumerate(atoms_list):
        _ensure_info(atoms, args.charge, args.spin)
        calculator.calculate(
            atoms, properties=calculator.implemented_properties, system_changes=all_changes
        )

        predictions = {}
        for key in calculator.implemented_properties:
            if key in calculator.results:
                predictions[key] = (
                    np.asarray(calculator.results[key]).reshape(-1).tolist()
                )

        edge_index = np.asarray(calculator.results["edge_index"]).astype(int)
        total_energy = 0.0
        for key in ("iqa_intra_a", "iqa_inter_a", "iqa_inter_ab"):
            if key in predictions:
                total_energy += float(np.asarray(predictions[key]).sum())

        results["structures"].append(
            {
                "index": idx,
                "natoms": int(atoms.get_global_number_of_atoms()),
                "atomic_numbers": atoms.get_atomic_numbers().tolist(),
                "edge_index": edge_index.tolist(),
                "predictions": predictions,
                "total_energy": total_energy,
            }
        )

    payload = json.dumps(results, indent=2)
    if args.output is None:
        print(payload)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)


if __name__ == "__main__":
    main()
