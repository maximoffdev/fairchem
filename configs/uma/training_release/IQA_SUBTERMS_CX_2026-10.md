# IQA subterm runs with the classical/exchange split, October 2026

Two subterm models that predict the electron-electron repulsion as its classical (Coulomb)
part V_ee^C and its exchange-correlation part V_ee^X instead of the combined V_ee. Both
share everything else with `iqa_subterms_atomchg_so2gate_6A.yaml`: K4L2 backbone, 6 Å
otf graph, q(A) conditioning with sigma 0.02, Lagrangian cutoff 5e-4, optimizer, and the
shared-trunk `IQA_MultiTaskSO2Head` with the exactly equivariant `so2_gate` activation.
Both inherit the task-independent `iqa_subterms_atomchg_common.yaml`.

| # | Config | What the head predicts |
|---|---|---|
| 1 | `iqa_subterms_cx_so2gate_6A.yaml` | the subterms directly |
| 2 | `iqa_subterm_fractions_so2gate_6A.yaml` | E_intra, E_inter and each subterm as a fraction |

## Identities in the labels

Checked on 5000 training pkls (all to < 0.06 kcal/mol):

    E_intra(A)   = T(A) + V_ne(A,A) + V_ee^C(A,A) + V_ee^X(A,A)
    E_inter(A,B) = V_ne(A,B) + V_en(A,B) + V_nn(A,B) + V_ee^C(A,B) + V_ee^X(A,B)
    V_ee = V_ee^C + V_ee^X,   VX_IQA_Inter(A,B) = V_ee^X(A,B)

All pair labels are stored as X(A,B)/2 per directed edge. V_ee^C(A,B) and V_ee^X(A,B)
are symmetric, and V_en(A,B) = V_ne(B,A).

## Run 1: direct subterms

Tasks: `tasks/iqa_decompose_subterms_cx.yaml`.

| Task | Level | Reference subtracted before the z-score | rmsd (eV) |
|---|---|---|---|
| T, V_ne, V_ee^C, V_ee^X (A,A) | atom | per-element mean | 6.32, 47.4, 41.2, 4.09 |
| V_ne(A,B)/2 (V_en by reversal) | edge | -Z_A Z_B / 2R | 11.7 |
| V_ee^C(A,B)/2 | edge | +Z_A Z_B / 2R | 12.8 |
| V_ee^X(A,B)/2 | edge | none | 1.35 |
| V_nn(A,B)/2 | edge | analytic head, not trained | - |

This is the previous subterm model with V_ee replaced by its two parts.

## Run 2: energies and fractions

Tasks: `tasks/iqa_decompose_subterm_fractions.yaml`. The fraction labels are built by the
dataset from `ratio_key_mapping` in `dataset/iqa_components.yaml` (numerator /
denominator in the pkl's units, no unit conversion, masked like the numerator).

| Task | Level | Label | Normalization |
|---|---|---|---|
| E_intra(A) | atom | eV | element refs + z-score (rmsd 5.86), as the intra/inter models |
| f_T, f_ne, f_C, f_X (A) | atom | X(A,A) / E_intra(A) | per-element mean + z-score (rmsd 0.143, 0.266, 0.239, 0.110) |
| E_inter(A,B)/2 | edge | eV | z-score (mean -0.69, rmsd 2.69), as the intra/inter models |
| g_ne, g_C, g_X (A,B) (g_en by reversal) | edge | X(A,B) / V_nn(A,B) | z-score (mean/rmsd -0.929/0.202, 0.866/0.267, -0.0083/0.0222) |

Reconstruction at inference, with V_nn(A,B)/2 from the analytic head:

    X(A)     = f_X(A) · E_intra(A)
    X(A,B)/2 = g_X(A,B) · V_nn(A,B)/2

The pair fractions are taken relative to V_nn, not E_inter. E_inter(A,B) crosses zero
(39 % of edges are positive, 32 % have |E_inter/2| < 0.5 kcal/mol). Because of that,
V_ne/E_inter has a median |f| of 241 at 2–4 Å and reaches 2e9. Even on the edges with
|E_inter/2| > 10 kcal/mol its std is 80. V_nn = Z_A Z_B / R never vanishes, and the
fractions relative to it stay within |g| ≤ 3.1.

Known limitations:

- The predicted fractions are not constrained to sum to 1 (intra) or to E_inter/V_nn
  (pair). The reconstructed subterms therefore do not add up exactly to the predicted
  energies.
- An intra fraction error δ costs δ·|E_intra| in the subterm. A 1σ error in the
  within-element fraction spread is 50–90 kcal/mol for H, ~1000 for C and ~3200 for S.
  The fraction loss weights all atoms equally, so this model is not expected to beat
  run 1 on absolute intra subterms.

## References and normalizers

`element_refs/iqa_isolated_atom_elem_refs.yaml` was regenerated with
`python -m fairchem.core.scripts.iqa_isolated_atom_refs --input_dir <sumformulasplit>/train`.
It adds `iqa_vee_c`, `iqa_vee_x` and the dimensionless `iqa_*_frac` blocks (per-element
mean fractions). The five earlier blocks came out identical.

The normalizers were fit with
`python -m fairchem.core.scripts.iqa_compute_normalizers` on the same split. All
previously configured values were reproduced exactly.

## Config refactor

The task-independent part of `iqa_subterms_atomchg_pretrain.yaml` moved to
`iqa_subterms_atomchg_common.yaml`. This was needed because `get_canonical_config`
rejects unused top-level keys, so a child config could not drop the V_ee tasks. Every
config in this directory that resolved before still resolves to the same config, apart
from timestamps. All 35 were compared leaf by leaf.

## Submitting (from the login node)

Commit and push first, then pull on the cluster checkout. The data tar is unchanged: the
new labels already exist in every pkl.

```bash
cd /data/horse/ws/maho053h-iqa-training/iqa_maxi/fairchem
export PATH=/data/horse/ws/maho053h-iqa-training/conda_maxi/bin:$PATH
for cfg in iqa_subterms_cx_so2gate_6A iqa_subterm_fractions_so2gate_6A; do
  fairchem -c configs/uma/training_release/$cfg.yaml cluster=dresden_alpha \
    cluster.ranks_per_node=4 cluster.max_atoms=800 cluster.timeout_hr=36
done
```

Each run uses 4 GPUs at 800 atoms per GPU. That is the same global batch (3200 atoms) as
the 2 x 1600 reruns and the 4 x 800 best-val resume of the 6 Å subterm model.
`epochs: 300` sets the cosine schedule (~160k steps at that batch), which is more than
36 h covers. submitit requeues a job that reaches its time limit (up to 3 times), so
`scancel` the runs at 36 h if they should stop there.
