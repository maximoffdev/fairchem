from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from fairchem.core import load_predict_unit
from fairchem.core.calculate import IQACalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an H2 IQA dispersion curve from node+edge sums."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MLIP checkpoint (e.g., inference_ckpt.pt).",
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
        "--dmin",
        type=float,
        default=0.4,
        help="Minimum H-H distance in Angstrom.",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        default=6.0,
        help="Maximum H-H distance in Angstrom.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=80,
        help="Number of points in the dispersion curve.",
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
        help="Total charge for H2.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="Spin for H2.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Shift energies so the last point is zero.",
    )
    parser.add_argument(
        "--plot",
        default="h2_iqa_dispersion.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path with distances and energies.",
    )
    return parser.parse_args()


def _build_h2(distance: float, charge: int, spin: int) -> Atoms:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


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

    distances = np.linspace(args.dmin, args.dmax, args.num)
    energies = []

    for distance in distances:
        atoms = _build_h2(distance, args.charge, args.spin)
        calculator.calculate(
            atoms, properties=calculator.implemented_properties, system_changes=all_changes
        )
        total_energy = 0.0
        for key in ("iqa_intra_a", "iqa_inter_a", "iqa_inter_ab"):
            if key in calculator.results:
                total_energy += float(np.asarray(calculator.results[key]).sum())
        energies.append(total_energy)

    energies = np.asarray(energies)
    if args.relative:
        energies = energies - energies[-1]

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(distances, energies, label="H2 IQA")
    plt.xlabel("H-H distance (Angstrom)")
    plt.ylabel("Total IQA energy (eV)")
    plt.title("H2 IQA dispersion curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    if args.output_json is not None:
        payload = {
            "task_name": calculator.task_name,
            "distances": distances.tolist(),
            "energies": energies.tolist(),
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
