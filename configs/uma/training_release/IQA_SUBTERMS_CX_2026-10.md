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
| 3 | `iqa_subterm_fractions_so2gate_triangle_6A.yaml` | as run 2, with triangle attention in the head |

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

## Run 3: fractions with triangle attention

Run 2 with `attention_type: triangle` in the shared head trunk; everything else,
including the validation metrics, is inherited. The per-edge graph attention is replaced
by AlphaFold3 triangle self-attention around the starting and ending node
(`models/uma/nn/triangle_attention.py`): the message of edge (A,B) mixes in the edges
(A,C) and (C,B) of every atom C that closes a triangle with A and B, with a bias from the
third edge. Node outputs average the incoming messages. Compare against run 2.

## Validation metrics

On top of the per-task `val/loss/<task>`, both runs log `val/mae/<name>` (eV). These are
exact MAEs over the whole val set, built from several task predictions
(`units/mlip_unit/derived_metrics.py`, configured in `derived_metrics/<tasks file>.yaml`):

| Key | Run 1 | Run 2 |
|---|---|---|
| `total_energy` | sum of all predicted subterms, incl. the analytic V_nn | sum_A E_intra + sum_(A,B) E_inter/2 |
| `iqa_kinetic`, ..., `iqa_vee_x_ab` | the predicted subterms | f_X · E_intra and g_X · V_nn/2 |

The subterm keys are the same in both runs, so the two can be compared directly. The other
subterm and intra/inter configs log `total_energy` too.

`total_energy` is compared against e_total. It therefore includes the E_inter of the pairs
beyond the otf cutoff, which the model never sees. On 300 val molecules that is a mean
|.| of 0.31 eV per molecule at 6 Å (max 12.8 eV, larger for ions) and 0.03 eV at 8 Å. The
IQA sum itself matches e_total to 0.011 eV. With the labels fed back as predictions,
`total_energy` comes out at 0.54 eV on a 6 Å graph (20 val molecules), and the subterms
reconstructed from the fractions at < 1e-4 eV.

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

Run 3 was submitted the same way with `cfg=iqa_subterm_fractions_so2gate_triangle_6A`.

Each run uses 4 GPUs at 800 atoms per GPU. That is the same global batch (3200 atoms) as
the 2 x 1600 reruns and the 4 x 800 best-val resume of the 6 Å subterm model.
`epochs: 300` sets the cosine schedule (~160k steps at that batch), which is more than
36 h covers. submitit requeues a job that reaches its time limit (up to 3 times), so
`scancel` the runs at 36 h if they should stop there.
