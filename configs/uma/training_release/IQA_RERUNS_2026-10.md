# IQA reruns, October 2026

Eight runs, all on Dresden alpha. Six retrain the SO(2)-head models with the exactly
equivariant activation. Two test more capacity for the pipeline dipole model.

## Why the SO(2)-head models are retrained

The SO(2) attention head used by the intra/inter and subterms models applied its
nonlinearity in the wrong coefficient order. Its outputs therefore depended on the random
roll of each edge frame, which is redrawn on every forward pass. On the trained 6 Å
intra/inter model the spread over rolls is 0.03 kcal/mol per edge on average and up to
0.75 kcal/mol. The head now has `activation_type: so2_gate`, which is exactly
equivariant, so repeated predictions agree to floating-point precision.

Checkpoints trained before the fix stay tied to the old activation and cannot be
patched. The dipole model uses an MLP head and is not affected.

## The runs

| # | Config | Cutoff | Purpose |
|---|---|---|---|
| 1 | `iqa_intra_inter_atomchg_so2gate_6A.yaml` | 6 Å | replaces pipeline 1 intra/inter |
| 2 | `iqa_intra_inter_atomchg_so2gate_8A.yaml` | 8 Å | replaces pipeline 2 intra/inter |
| 3 | `iqa_subterms_atomchg_local_so2gate_8A.yaml` | 8 Å | replaces the subterms model of both pipelines |
| 4 | `iqa_subterms_atomchg_local_so2gate_6A.yaml` | 6 Å | run 3 at 6 Å |
| 5 | `iqa_subterms_atomchg_so2gate_6A.yaml` | 6 Å | run 1 trained on the subterms |
| 6 | `iqa_subterms_atomchg_so2gate_8A.yaml` | 8 Å | run 2 trained on the subterms |
| 7 | `iqa_dipole_atomchg_sigma0p05_6A_5layers.yaml` | 6 Å | pipeline 1 dipole model, 5 backbone layers instead of 4 |
| 8 | `iqa_dipole_atomchg_sigma0p05_6A_lmax3.yaml` | 6 Å | pipeline 1 dipole model, lmax 3 instead of 2 |

## How each run relates to its original

Every config was resolved exactly as the submit command below resolves it. It was then
compared leaf by leaf with the resolved config of the run it is based on.

**Runs 1 to 4** inherit the config that produced the original run:
`iqa_intra_inter_atomchg_pretrain.yaml` for intra/inter and
`iqa_subterms_atomchg_local_pretrain.yaml` for the local subterms model. Only the
activation, the cutoff and the run name differ from the original's `resume.yaml`. The
original 6 Å intra/inter run left no `resume.yaml`. Its stored model and task configs
equal the 8 Å run's apart from the cutoff, so run 1 is built from the 8 Å config.

**Runs 5 and 6** inherit `iqa_subterms_atomchg_pretrain.yaml`. Its backbone is not
strictly local, its head uses softmax attention, and it uses sigma 0.02, all as in the
intra/inter runs. Outside the task and head sections they differ from runs 1 and 2 only
in the run name. The subterm tasks use the recent subterm normalization: neutral
point-charge baselines on V_ne, V_en and V_ee(A,B), element references on the atomic
terms, and the normalizers refit on the cleaned training split. V_nn is computed
analytically. The head is the shared-trunk multi-task head with the exact activation.

**Runs 7 and 8** inherit `iqa_dipole_atomchg_sigma0p05_6A.yaml`. That config replicates
the pipeline dipole model `local_charge_pipeline_ckpts/dipole_local_charge_cond`. The
model was trained in wandb run `lamerslab/iqa/202609-2923-2739-7817` on Leipzig, and
its weights load strictly into the replica. The replica is the base of runs 7 and 8 and
is not meant to be rerun. It pins two settings that the repo has changed since:
- `sigma_atomic_charge` is 0.05.
- The dipole normalizer rmsd is 1.0. The repo default was refit to 0.716 Debye on
  9 October.

Apart from the intended change, runs 7 and 8 differ from the original only in cluster
resources: 2 GPUs at 1600 atoms instead of 4 GPUs at 450 atoms (see Submitting).

- Run 7 sets `num_layers: 5`.
- Run 8 sets `lmax: 3` and keeps mmax at 2, as the larger UMA backbones do. That
  doubles the backbone parameters, from 6.3 M to 12.2 M. Expect it to train noticeably
  slower per step.

## Time budget

All runs are submitted with 40 h per job. submitit requeues a job that reaches its time
limit up to three times, so a run continues until it finishes or is cancelled. The
original dipole run, for example, accumulated 66 h over two jobs. The cosine schedule is
fixed by `epochs: 300`. When comparing runs, compare them at matched steps rather than
at matched wall time. The original 6 Å intra/inter run was submitted with 24 h.

## Before submitting

1. Commit and push the head fix together with these configs. Then pull on the cluster
   checkout `/data/horse/ws/maho053h-iqa-training/iqa_maxi/fairchem`. Without the fix,
   runs 1 to 6 fail at startup with an unexpected `activation_type` keyword.
2. Read `DRESDEN.md` in the repo root, in particular the data-tar staleness guard.
3. Eight runs take 16 GPUs at the same time.

## Submitting (from the login node)

```bash
cd /data/horse/ws/maho053h-iqa-training/iqa_maxi/fairchem
export PATH=/data/horse/ws/maho053h-iqa-training/conda_maxi/bin:$PATH
for cfg in iqa_intra_inter_atomchg_so2gate_6A iqa_intra_inter_atomchg_so2gate_8A \
           iqa_subterms_atomchg_local_so2gate_6A iqa_subterms_atomchg_local_so2gate_8A \
           iqa_subterms_atomchg_so2gate_6A iqa_subterms_atomchg_so2gate_8A \
           iqa_dipole_atomchg_sigma0p05_6A_5layers iqa_dipole_atomchg_sigma0p05_6A_lmax3; do
  fairchem -c configs/uma/training_release/$cfg.yaml cluster=dresden_alpha \
    cluster.timeout_hr=40 cluster.max_atoms=1600
done
```

Each run takes 2 GPUs. The original Dresden runs used `max_atoms=900` (2 x 900 = 4 x 450,
the Leipzig global batch). These reruns use 1600 atoms per GPU, so the global batch is
3200 instead of 1800 atoms at the same learning rate, and `epochs: 300` gives about 0.56x
as many optimizer steps as the originals. Submitted on 2026-10-15 as Slurm jobs
2370165 to 2370172 in the loop's order (run 4 is 2370167, run 3 is 2370168).

## After submitting

- Logs and checkpoints are written to `/data/horse/ws/maho053h-iqa-training/iqa/<timestamp_id>/`.
- The pipeline checkpoint is `checkpoints/best_val_checkpoint/inference_ckpt.pt`.
- Check that each run gets past the first epoch boundary and the first evaluations before
  trusting it (see `DRESDEN.md`, Monitoring).
- If the log of runs 1 to 6 contains a warning mentioning `legacy_sep_s2`, that run is
  training with the old activation. Cancel it.

## Old checkpoints

The head's default is still `activation_type: legacy_sep_s2`. That default lets the
existing pipeline checkpoints load and predict exactly as before, apart from GPU
nondeterminism. They log a warning that their predictions depend on the frame roll. New
checkpoints store `activation_type: so2_gate` in their model config, so they load with
the exact activation without any inference setting.
