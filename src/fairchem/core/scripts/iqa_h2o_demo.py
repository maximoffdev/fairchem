from __future__ import annotations

import json

from ase import Atoms
from ase.calculators.calculator import all_changes

from fairchem.core import load_predict_unit
from fairchem.core.calculate import IQACalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
import numpy as np

MODEL_PATH = "../models/inference_ckpt.pt"
TASK_NAME = "iqa_pkl"


def main() -> None:
    settings = inference_settings_default()
    settings.external_graph_gen = True

    predict_unit = load_predict_unit(
        MODEL_PATH,
        inference_settings=settings,
    )

    calculator = IQACalculator(
        predict_unit,
        task_name=TASK_NAME,
        molecule_cell_size=10.0,
    )

    atoms = Atoms(
        "H2O",
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.758602, 0.000000, 0.504284],
            [-0.758602, 0.000000, 0.504284],
        ],
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0

    calculator.calculate(
        atoms, properties=calculator.implemented_properties, system_changes=all_changes
    )

    predictions = {}
    for key in calculator.implemented_properties:
        if key in calculator.results:
            predictions[key] = calculator.results[key].reshape(-1).tolist()

    total_energy = 0.0
    for key in ("iqa_intra_a", "iqa_inter_a", "iqa_inter_ab"):
        if key in predictions:
            total_energy += float(np.asarray(predictions[key]).sum())

    payload = {
        "task_name": calculator.task_name,
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "predictions": predictions,
        "total_energy": total_energy,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
