# Working on the TU Dresden HPC (alpha partition)

Notes for agents that run IQA training on the Dresden cluster. Collected 2026-10-09/10
while moving the Leipzig (clara) runs here. Numbers are measurements, not estimates,
unless marked otherwise.

## Hardware

- Login node: `login1.alpha.hpc.tu-dresden.de`. No GPUs; treat it as slow for Python.
- Partition `alpha`: 37 nodes, each 8x NVIDIA A100-SXM4-40GB, 48 CPUs, 990 GB RAM,
  node-local NVMe at `/tmp` (3.5 TiB). Partition `MaxTime` is unlimited.
- Only GPU jobs are accepted (a Slurm cli_filter rejects anything without GPUs, even
  `srun --overlap` into your own job).

## Storage: the main pitfall

- The project, data and conda envs live on `/data/horse/ws/maho053h-iqa-training/`.
  The link from horse to the compute nodes is slow for many small files, and metadata
  operations are slow everywhere (`ls -la` on the 118k-file train dir took > 2 min on
  the login node; `du` on it timed out).
- Never train from horse or copy the pkls one by one: rsync of ~120k pkls took > 20 min.
- Stage one archive to node-local `/tmp` instead. `sumformulasplit_trainval.tar`
  (6.8 GB, train + val + their `metadata.npz`) unpacks in ~20 s; total setup is ~30 s.
- `cluster/dresden_alpha.yaml` does this per run in `setup_commands`: unpack into
  `/tmp/$USER_iqa_<run_name>` (one dir per run, because several 2-GPU jobs share a node),
  build/check `metadata.npz` once, and `trap 'rm -rf …' EXIT` for cleanup.
- Building `metadata.npz` unpickles every pkl (~10 min, CPU-bound, single process).
  That is why the archive carries the npz files. Rebuild recipe is in the yaml comment.
- The archive has a staleness guard: setup fails if `train/` or `val/` on horse is newer
  than the tar. Any change in those dirs (even creating and deleting a file) bumps their
  mtime. If only `metadata.npz` changed there, `touch` the tar; otherwise rebuild it.
- Keep the horse split dirs free of `metadata.npz`. Instantiating `IQAPKLDataset` on a
  horse path (e.g. a quick check on the login node) silently writes one there.

## Environment

- Use `/data/horse/ws/maho053h-iqa-training/conda_maxi` (editable install of this repo).
  The env `…/conda` points at a different fairchem checkout (`/home/h1/maho053h/fairchem`).
- `fairchem` is not on `PATH` by default. Submit with
  `export PATH=/data/horse/ws/maho053h-iqa-training/conda_maxi/bin:$PATH`. The sbatch
  script inherits this `PATH`, and the `python` in `setup_commands` relies on it.
- Fixed in `conda_maxi`: `torch_geometric` was missing (now also in `pyproject.toml`);
  setuptools 82 dropped `pkg_resources`, which torchtnt 0.2.4 imports, so it is pinned
  to `setuptools<81`.
- `pre-commit` is not installed in `conda_maxi`.

## Submitting

```bash
cd /data/horse/ws/maho053h-iqa-training/iqa_maxi/fairchem
export PATH=/data/horse/ws/maho053h-iqa-training/conda_maxi/bin:$PATH
fairchem -c configs/uma/training_release/<config>.yaml cluster=dresden_alpha \
  cluster.timeout_hr=40 cluster.max_atoms=900 [other overrides]
```

- Submit from the login node only; fairchem/submitit builds and submits the sbatch script.
- Use `cluster=dresden_alpha` for IQA runs. `dresden_cluster.yaml` and
  `dresden_cluster_escnbase.yaml` are older and change the model (`num_moe_experts: 32`,
  `max_neighbors: 30`/`100`) and stage the old dataset with a shared `/tmp/iqa_data`.
- Do not set `mem_gb: 0`; that requests the whole node. `dresden_alpha` uses 120 GB.
- `timeout_hr` is an `int`, so sub-hour jobs are impossible via the config. submitit
  also requeues a job that hits its time limit (`slurm_max_num_timeout=3`). For short
  test jobs submit with `cluster.timeout_hr=1` and `scancel` them yourself.
- The config is resolved at submission time, so `${oc.env:SLURM_JOB_ID}` is not
  available in it; use `${job.run_name}` or `${job.timestamp_id}` for per-run paths.
- Inspect a running job's node: `srun --jobid=<id> --overlap --nodes=1 --ntasks=1 --gpus=1 <cmd>`.

## Training behaviour and numbers

Two GPUs per run with the same global batch as Leipzig (4 GPUs x 450 atoms) means
`max_atoms` 900 for 8 A energy/dipole models and 1200 for the 4 A K2L2 charge models
(Leipzig: 4 x 600).

| Model | Steps/s | GPU util (median) | GPU memory |
|---|---|---|---|
| K4L2 8 A, 900 atoms, 0 workers | 0.72 | 18 % | 7 GB |
| K4L2 strict-local 8 A, 900 atoms, 4 workers | 1.05 | 51 % | 7 GB |
| K2L2_48 4 A, 1200 atoms, 0 workers | 0.76 | 0 % | 2.5 GB |
| K2L2_48 4 A, 1200 atoms, 4 workers | 1.39 | 23 % | 2.5 GB |

- Leipzig reference: 4x V100-32GB, K4L2 8 A, 450 atoms per GPU -> 0.67 steps/s, peak 7.3 GB.
- Throughput is CPU-bound (data loading, radius graph), not GPU-bound. Steps per hour
  depend on the per-GPU batch, not on the GPU count. Memory has a lot of headroom
  (~2000 atoms per GPU would fit at 8 A, untested).
- `epochs: 300` sets the cosine LR schedule (~290k steps at 1800 atoms per step), so
  a 40 h run (~150k steps) stops before the LR has decayed.

## Known bugs and their fixes

- **Forked dataloader workers abort** with `CUDA error: initialization error` at a random
  step; the other rank then hangs in an allreduce until the 30 min NCCL watchdog kills
  it. Cause: the model is on the GPU before the loaders fork. Fixed in
  `components/common/dataloader_builder.py` (`forkserver` + `persistent_workers`).
  Do not revert to `fork` with `num_workers > 0`.
- **wandb artifact names are capped at 128 characters** (wandb 0.27). The profiler trace
  used to be named after `run_name` and crashed runs with long names at step 12. It now
  uses `timestamp_id` (`common/profiler_utils.py`).
- A cancelled job can leave `FileNotFoundError` on a staged pkl in its log: the EXIT
  trap removes the `/tmp` dir while training is still reading. Not a data problem.

## wandb

- Compute nodes have internet; runs log online to `lamerslab/iqa` with the credentials in
  `~/.netrc`. No offline sync is needed.
- The public API is slow from the login node (listing runs can exceed 2 min); give calls
  a timeout of several minutes. `api.viewer` returns "relogin required" while reading
  runs works; this did not affect logging.
- Per-run step rate: `scan_history(keys=["train/step", "_timestamp"])`; GPU memory and
  utilisation: `history(stream="system", pandas=False)` (pandas is not installed).

## Monitoring

- Run outputs: `/data/horse/ws/maho053h-iqa-training/iqa/<timestamp_id>/`
  (`logs/<slurm_id>_<rank>_log.err` holds the `train/step` lines and tracebacks,
  `logs/profiles/` holds the profiler trace of steps 10-11, `checkpoints/`).
- `sacct -X -j <ids> -o JobID,JobName%40,State,Elapsed,NodeList,ExitCode`. Exit code
  `6:0` after ~30 min of silence means an NCCL watchdog abort; look at the other rank's
  log for the real error.
- Check that a run gets past the first epoch boundary and the first evals before
  trusting it: worker problems only showed up there.
