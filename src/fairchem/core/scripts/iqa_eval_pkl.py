#!/usr/bin/env python3
"""Evaluate IQA predictions on PKL data with error stats and plots."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.data import chemical_symbols

from fairchem.core import load_predict_unit
from fairchem.core.calculate import predict_iqa_pkl
from fairchem.core.datasets.iqa_pkl_dataset import IQAPKLDataset
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =============================================================================
# 1. SETUP & LOAD PREDICTION UNIT
# =============================================================================
checkpoint_path = "/work2/rh41uvuh-iqa_train_2/iqa/202605-1212-3252-558f/checkpoints/final/inference_ckpt.pt"
data_path = "/work2/rh41uvuh-iqa_train_2/HCNOSPClF_combined_datasets_filtered0.001/test/5smallPeptides/"
task_name = "iqa_pkl"
max_items = 100
show_plots = False

REL_EPS = 1e-8


def _rel_error(abs_err: np.ndarray, target: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(target), REL_EPS)
    return abs_err / denom


def _to_numpy(values) -> np.ndarray:
    if values is None:
        return np.array([])
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values)


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

settings = inference_settings_default()
settings.external_graph_gen = True  # required for PKL edges

try:
    predict_unit = load_predict_unit(
        checkpoint_path,
        inference_settings=settings,
        device=device,
    )
except TypeError:
    predict_unit = load_predict_unit(
        checkpoint_path,
        inference_settings=settings,
    )

if hasattr(predict_unit, "device"):
    predict_unit.device = device
if hasattr(predict_unit, "move_to_device"):
    predict_unit.move_to_device()
if hasattr(predict_unit, "model"):
    predict_unit.model.eval()

tasks = predict_unit.dataset_to_tasks[task_name]
task_names = [t.name for t in tasks]

print("Prediction unit loaded successfully!")
print("Tasks:", task_names)

# =============================================================================
# 2. LOAD PREDICTIONS USING IQA CALCULATOR API
# =============================================================================
print("Running predict_iqa_pkl...")
pred_result = predict_iqa_pkl(
    predict_unit,
    data_path,
    task_name=task_name,
    max_items=max_items,
)
structures = pred_result["structures"]
print(f"Predicted {len(structures)} structures")

# =============================================================================
# 3. LOAD DATASET TARGETS
# =============================================================================
EV_TO_HARTREE = 1.0 / 27.211386245988
EV_PER_BOHR_TO_HARTREE_PER_ANGSTROM = 1.8897261245650618 / 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631


def convert_units(name: str, values: np.ndarray) -> np.ndarray:
    if "force" in name:
        return values * EV_PER_BOHR_TO_HARTREE_PER_ANGSTROM
    return values * EV_TO_HARTREE


path = Path(data_path)
dataset_root = path if path.is_dir() else path.parent
val_dataset = IQAPKLDataset(
    src=str(dataset_root),
    name=task_name,
    ht2ev=True,  # dataset labels converted to eV / eV per Bohr
    key_mapping={
        "energy": "e_total",
        "iqa_intra_a": "E_IQA_Intra(A)",
        "iqa_inter_a": "E_IQA_Inter(A)",
        "iqa_inter_ab": "E_IQA_Inter(A,B)/2",
        "iqa_forces_direct": "Fn(A,SumB)",
        "iqa_forces_grad": "Fn(A,SumB)",
    },
)
print(f"Validation dataset size: {len(val_dataset)}")

# =============================================================================
# 4. EVALUATE
# =============================================================================
results = {
    "molecule_id": [],
    "num_atoms": [],
    "num_edges": [],
    "error_total_atom_based": [],
    "rel_error_total_atom_based": [],
    "error_total_edge_based": [],
    "rel_error_total_edge_based": [],
}

per_task_mae = {name: [] for name in task_names}
per_task_rel_mae = {name: [] for name in task_names}
pred_buffers = {name: [] for name in task_names}
target_buffers = {name: [] for name in task_names}
paired_pred_buffers = {name: [] for name in task_names}
paired_target_buffers = {name: [] for name in task_names}

# Atom-level error bookkeeping
atom_errors = {name: defaultdict(list) for name in task_names}
atom_rel_errors = {name: defaultdict(list) for name in task_names}

# Edge-level pair errors
pair_errors = {name: defaultdict(list) for name in task_names}

for entry in tqdm(structures, desc="Evaluating", unit="mol"):
    idx = int(entry["index"])
    data = val_dataset[idx]

    results["molecule_id"].append(entry.get("sid", str(idx)))
    results["num_atoms"].append(int(entry.get("natoms", data.natoms.item())))
    results["num_edges"].append(int(entry.get("nedges", data.nedges.item())))

    energy_target = None
    if hasattr(data, "energy") and data.energy is not None:
        energy_target = convert_units(
            "energy", _to_numpy(data.energy.detach().cpu().numpy())
        ).reshape(-1)[0]

    pred_intra_sum = None
    pred_inter_a_sum = None
    pred_inter_ab_sum = None

    pred_by_task = entry.get("predictions", {})

    for task in tasks:
        name = task.name
        pred = _to_numpy(pred_by_task.get(name, None)).astype(float)
        if pred.size == 0:
            per_task_mae[name].append(np.nan)
            per_task_rel_mae[name].append(np.nan)
            continue

        pred = convert_units(name, pred)
        pred_buffers[name].append(pred.reshape(-1))

        target = None
        if hasattr(data, name):
            target = _to_numpy(getattr(data, name).detach().cpu().numpy()).astype(float)
            target = convert_units(name, target)
            target_buffers[name].append(target.reshape(-1))

        if target is None or target.size == 0:
            per_task_mae[name].append(np.nan)
            per_task_rel_mae[name].append(np.nan)
            continue

        abs_err = np.abs(pred - target)
        rel_err = _rel_error(abs_err, target)
        paired_pred_buffers[name].append(pred.reshape(-1))
        paired_target_buffers[name].append(target.reshape(-1))
        per_task_mae[name].append(float(abs_err.mean()))
        per_task_rel_mae[name].append(float(rel_err.mean()))

        if task.level == "atom":
            atomic_numbers = data.atomic_numbers.detach().cpu().numpy().astype(int)
            for z, err, rerr in zip(atomic_numbers, abs_err.reshape(-1), rel_err.reshape(-1)):
                atom_errors[name][int(z)].append(float(err))
                atom_rel_errors[name][int(z)].append(float(rerr))
        elif task.level == "edge" and hasattr(data, "edge_index"):
            edge_index = data.edge_index.detach().cpu().numpy()
            atomic_numbers = data.atomic_numbers.detach().cpu().numpy().astype(int)
            for (i, j), err in zip(edge_index.T, abs_err.reshape(-1)):
                zi, zj = int(atomic_numbers[i]), int(atomic_numbers[j])
                pair = tuple(sorted((zi, zj)))
                pair_errors[name][pair].append(float(err))

        if name == "iqa_intra_a":
            pred_intra_sum = float(pred.sum())
        elif name == "iqa_inter_a":
            pred_inter_a_sum = float(pred.sum())
        elif name == "iqa_inter_ab":
            pred_inter_ab_sum = float(pred.sum())

    if energy_target is not None and pred_intra_sum is not None and pred_inter_a_sum is not None:
        abs_err = abs(pred_intra_sum + pred_inter_a_sum - energy_target)
        rel_err = abs_err / max(abs(energy_target), REL_EPS)
        results["error_total_atom_based"].append(abs_err)
        results["rel_error_total_atom_based"].append(rel_err)
    else:
        results["error_total_atom_based"].append(np.nan)
        results["rel_error_total_atom_based"].append(np.nan)

    if energy_target is not None and pred_intra_sum is not None and pred_inter_ab_sum is not None:
        abs_err = abs(pred_intra_sum + pred_inter_ab_sum - energy_target)
        rel_err = abs_err / max(abs(energy_target), REL_EPS)
        results["error_total_edge_based"].append(abs_err)
        results["rel_error_total_edge_based"].append(rel_err)
    else:
        results["error_total_edge_based"].append(np.nan)
        results["rel_error_total_edge_based"].append(np.nan)

# Build DataFrame
for name in task_names:
    results[f"mae_{name}"] = per_task_mae[name]
    results[f"rel_mae_{name}"] = per_task_rel_mae[name]

df = pd.DataFrame(results)
print("\nEvaluation complete!")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================
summary_rows = []
for task in tasks:
    name = task.name
    unit = "Ha/Ang" if "force" in name else "Ha"

    pred_all = np.concatenate(pred_buffers[name]) if pred_buffers[name] else np.array([])
    targ_all = np.concatenate(target_buffers[name]) if target_buffers[name] else np.array([])
    paired_pred_all = (
        np.concatenate(paired_pred_buffers[name]) if paired_pred_buffers[name] else np.array([])
    )
    paired_targ_all = (
        np.concatenate(paired_target_buffers[name]) if paired_target_buffers[name] else np.array([])
    )

    mae_vals = df[f"mae_{name}"].dropna().values if f"mae_{name}" in df else np.array([])
    rel_vals = df[f"rel_mae_{name}"].dropna().values if f"rel_mae_{name}" in df else np.array([])

    pred_mean, pred_std = _mean_std(pred_all)
    targ_mean, targ_std = _mean_std(targ_all)
    mae_mean, mae_std = _mean_std(mae_vals)
    if paired_pred_all.size and paired_targ_all.size:
        rel_all = _rel_error(np.abs(paired_pred_all - paired_targ_all), paired_targ_all)
        rel_mean, rel_std = _mean_std(rel_all)
    else:
        rel_mean, rel_std = _mean_std(rel_vals)

    summary_rows.append(
        {
            "Task": name,
            "Unit": unit,
            "Pred mean": pred_mean,
            "Pred std": pred_std,
            "Target mean": targ_mean,
            "Target std": targ_std,
            "MAE mean": mae_mean,
            "MAE std": mae_std,
            "Rel MAE mean": rel_mean,
            "Rel MAE std": rel_std,
        }
    )

summary = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("TASK SUMMARY (Hartree / Hartree/Angstrom)")
print("=" * 80)
print(summary.to_string(index=False))

if df["error_total_atom_based"].notna().any():
    atom_err = df["error_total_atom_based"].dropna().values
    atom_rel = df["rel_error_total_atom_based"].dropna().values
    print("\n" + "=" * 80)
    print("TOTAL ENERGY ERROR (ATOM-BASED)")
    print("=" * 80)
    print(f"Mean ± std: {atom_err.mean():.6f} ± {atom_err.std():.6f} Ha")
    print(
        f"Mean ± std: {(atom_err.mean() * HARTREE_TO_KCAL_MOL):.2f} ± "
        f"{(atom_err.std() * HARTREE_TO_KCAL_MOL):.2f} kcal/mol"
    )
    print(f"Rel mean ± std: {atom_rel.mean():.6f} ± {atom_rel.std():.6f}")

if df["error_total_edge_based"].notna().any():
    edge_err = df["error_total_edge_based"].dropna().values
    edge_rel = df["rel_error_total_edge_based"].dropna().values
    print("\n" + "=" * 80)
    print("TOTAL ENERGY ERROR (EDGE-BASED)")
    print("=" * 80)
    print(f"Mean ± std: {edge_err.mean():.6f} ± {edge_err.std():.6f} Ha")
    print(
        f"Mean ± std: {(edge_err.mean() * HARTREE_TO_KCAL_MOL):.2f} ± "
        f"{(edge_err.std() * HARTREE_TO_KCAL_MOL):.2f} kcal/mol"
    )
    print(f"Rel mean ± std: {edge_rel.mean():.6f} ± {edge_rel.std():.6f}")

# =============================================================================
# 6. CORRELATION WITH MOLECULE SIZE
# =============================================================================
if df["error_total_atom_based"].notna().any():
    size = df["num_atoms"].astype(float)
    err = df["error_total_atom_based"].astype(float)
    rel_err = df["rel_error_total_atom_based"].astype(float)

    pearson_abs = size.corr(err)
    pearson_rel = size.corr(rel_err)
    spearman_abs = size.corr(err, method="spearman")
    spearman_rel = size.corr(rel_err, method="spearman")

    print("\n" + "=" * 80)
    print("SIZE CORRELATION (ATOM-BASED TOTAL ENERGY)")
    print("=" * 80)
    print(f"Pearson abs: {pearson_abs:.4f} | Spearman abs: {spearman_abs:.4f}")
    print(f"Pearson rel: {pearson_rel:.4f} | Spearman rel: {spearman_rel:.4f}")

# =============================================================================
# 7. PLOTS
# =============================================================================
plot_dir = Path("iqa_eval_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

mae_tasks = [t.name for t in tasks if f"mae_{t.name}" in df and df[f"mae_{t.name}"].notna().any()]
if mae_tasks:
    cols = min(3, len(mae_tasks))
    rows = (len(mae_tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(mae_tasks):
        unit = "Ha/Ang" if "force" in name else "Ha"
        axes[i].hist(df[f"mae_{name}"].dropna(), bins=50, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{name} MAE (Mean: {df[f'mae_{name}'].mean():.6f} {unit})")
        axes[i].set_xlabel(f"MAE ({unit})")
        axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "iqa_task_mae.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

if df["error_total_atom_based"].notna().any() or df["error_total_edge_based"].notna().any():
    fig = plt.figure(figsize=(8, 5))
    if df["error_total_atom_based"].notna().any():
        plt.hist(df["error_total_atom_based"].dropna(), bins=50, alpha=0.7, label="Atom-based")
    if df["error_total_edge_based"].notna().any():
        plt.hist(df["error_total_edge_based"].dropna(), bins=50, alpha=0.7, label="Edge-based")
    plt.xlabel("Total energy error (Ha)")
    plt.ylabel("Count")
    plt.title("Total Energy Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "iqa_total_energy_error.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

if df["error_total_atom_based"].notna().any():
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df["num_atoms"], df["error_total_atom_based"], alpha=0.6, s=20)
    plt.xlabel("Number of atoms")
    plt.ylabel("Total energy error (Ha)")
    plt.title("Total Energy Error vs Molecule Size (Atom-based)")
    plt.tight_layout()
    plt.savefig(plot_dir / "iqa_error_vs_size.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df["num_atoms"], df["rel_error_total_atom_based"], alpha=0.6, s=20)
    plt.xlabel("Number of atoms")
    plt.ylabel("Relative total energy error")
    plt.title("Relative Total Energy Error vs Molecule Size (Atom-based)")
    plt.tight_layout()
    plt.savefig(plot_dir / "iqa_rel_error_vs_size.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

# Atom-type error plots
atom_rows = []
for name, z_map in atom_errors.items():
    for z, errs in z_map.items():
        rels = atom_rel_errors[name].get(z, [])
        atom_rows.append(
            {
                "task": name,
                "element": chemical_symbols[z] if z < len(chemical_symbols) else str(z),
                "atomic_number": z,
                "count": len(errs),
                "mae": float(np.mean(errs)) if errs else np.nan,
                "rel_mae": float(np.mean(rels)) if rels else np.nan,
                "std": float(np.std(errs)) if errs else np.nan,
            }
        )

atom_df = pd.DataFrame(atom_rows)
if not atom_df.empty:
    atom_df.to_csv("iqa_atom_type_error_summary.csv", index=False)
    for task in atom_df["task"].unique():
        sub = atom_df[atom_df["task"] == task].sort_values("count", ascending=False)
        top = sub.head(12)
        fig = plt.figure(figsize=(8, 4))
        plt.bar(top["element"], top["mae"], color="#4C72B0")
        plt.ylabel("MAE (Ha)")
        plt.title(f"Atom-type MAE for {task} (top 12 by count)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"iqa_atom_type_mae_{task}.png", dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)

# Edge pair error summary (optional)
pair_rows = []
for name, pairs in pair_errors.items():
    for (zi, zj), errs in pairs.items():
        pair_rows.append(
            {
                "task": name,
                "pair": f"{chemical_symbols[zi]}-{chemical_symbols[zj]}",
                "atomic_numbers": f"{zi}-{zj}",
                "count": len(errs),
                "mae": float(np.mean(errs)) if errs else np.nan,
                "std": float(np.std(errs)) if errs else np.nan,
            }
        )

pair_df = pd.DataFrame(pair_rows)
if not pair_df.empty:
    pair_df.to_csv("iqa_edge_pair_error_summary.csv", index=False)
    for task in pair_df["task"].unique():
        sub = pair_df[pair_df["task"] == task].sort_values("count", ascending=False)
        top = sub.head(15)
        fig = plt.figure(figsize=(9, 4))
        plt.bar(top["pair"], top["mae"], color="#55A868")
        plt.ylabel("MAE (Ha)")
        plt.title(f"Edge-pair MAE for {task} (top 15 by count)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"iqa_edge_pair_mae_{task}.png", dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
df.to_csv("iqa_model_evaluation_results.csv", index=False)
summary.to_csv("iqa_model_summary.csv", index=False)
print("\nSaved: iqa_model_evaluation_results.csv")
print("Saved: iqa_model_summary.csv")
if not atom_df.empty:
    print("Saved: iqa_atom_type_error_summary.csv")
if not pair_df.empty:
    print("Saved: iqa_edge_pair_error_summary.csv")
print(f"Saved plots to: {plot_dir}")
