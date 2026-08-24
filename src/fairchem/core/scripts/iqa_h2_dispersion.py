from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
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
        default="h2_iqa_dispersion.html",
        help="Output interactive HTML plot path.",
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
            atoms,
            properties=calculator.implemented_properties,
            system_changes=all_changes,
        )
        total_energy = 0.0
        for key in ("iqa_intra_a", "iqa_inter_a", "iqa_inter_ab"):
            if key in calculator.results:
                total_energy += float(np.asarray(calculator.results[key]).sum())
        energies.append(total_energy)

    energies = np.asarray(energies)
    if args.relative:
        energies = energies - energies[-1]

    min_idx = int(np.argmin(energies))
    min_energy = float(energies[min_idx])
    min_distance = float(distances[min_idx])
    idx_3a = int(np.argmin(np.abs(distances - 3.0)))
    distance_at_3a = float(distances[idx_3a])
    energy_at_3a = float(energies[idx_3a])
    delta_min_to_3a = energy_at_3a - min_energy

    plot_path = Path(args.plot)
    if plot_path.suffix.lower() != ".html":
        plot_path = plot_path.with_suffix(plot_path.suffix + ".html")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure(
        data=[go.Scatter(x=distances, y=energies, mode="lines", name="H2 IQA")]
    )
    fig.update_layout(
        title="H2 IQA dispersion curve",
        xaxis_title="H-H distance (Angstrom)",
        yaxis_title="Total IQA energy (eV)",
        template="plotly_white",
    )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "min @ {dist:.3f} A = {emin:.6f} eV, "
            "E(3 A) @ {dist3:.3f} A = {e3:.6f} eV, "
            "delta = {delta:.6f} eV"
        ).format(
            dist=min_distance,
            emin=min_energy,
            dist3=distance_at_3a,
            e3=energy_at_3a,
            delta=delta_min_to_3a,
        ),
    )
    fig.write_html(plot_path, include_plotlyjs="cdn")

    print(
        "min @ {dist:.3f} A = {emin:.6f} eV, "
        "E(3 A) @ {dist3:.3f} A = {e3:.6f} eV, "
        "delta = {delta:.6f} eV".format(
            dist=min_distance,
            emin=min_energy,
            dist3=distance_at_3a,
            e3=energy_at_3a,
            delta=delta_min_to_3a,
        )
    )

    if args.output_json is not None:
        payload = {
            "task_name": calculator.task_name,
            "distances": distances.tolist(),
            "energies": energies.tolist(),
            "summary": {
                "min_distance": min_distance,
                "min_energy": min_energy,
                "distance_at_3a": distance_at_3a,
                "energy_at_3a": energy_at_3a,
                "delta_min_to_3a": delta_min_to_3a,
            },
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
