# IQA runs overview

Current and reference runs on the Dresden `alpha` partition, so none get lost. Run dirs
are under `/data/horse/ws/maho053h-iqa-training/iqa/`; the checkpoint used for
evaluation is `<run dir>/checkpoints/best_val_checkpoint/inference_ckpt.pt`. Details
per model family: [IQA_SUBTERMS_CX_2026-10.md](IQA_SUBTERMS_CX_2026-10.md) (energy and
subterms), [IQA_RERUNS_2026-10.md](IQA_RERUNS_2026-10.md) (dipole and charge).

Best val/loss is the value of the `New best val/loss` log line, which selects the
checkpoint.

Last updated: 2026-11-05.

## Running

| Model | Config | Slurm job | Run dir | Started | Limit | Status |
|---|---|---|---|---|---|---|
| Subterm fractions, triangle head, fraction closure (run 6) | `iqa_subterm_fractions_so2gate_triangle_attnnodes_closure_6A.yaml` | 2388410 | `202611-0517-0648-cf0d` | 2026-11-05 17:07 | 36 h | running |
| Dipole vector, triangle SO(2) head | `iqa_dipole_atomchg_sigma0p05_6A_so2head_triangle.yaml` | 2387864 | `202611-0414-5812-1217` | 2026-11-04 15:02 | 40 h | running; best val/loss 0.0260 at step 46.9k (of 49.5k) |

## Finished

| Model | Config | Slurm job | Run dir | Ended | Result |
|---|---|---|---|---|---|
| **Subterm fractions, triangle head with attention node aggregation (run 4, best energy model)** | `iqa_subterm_fractions_so2gate_triangle_attnnodes_6A.yaml` | 2382584 | `202610-2900-3547-fc3b` | 2026-10-30, 36 h limit, ~138k steps | **finished**; best val/loss 0.03434 at step 135000. Val MAE (eV): E_intra 0.0202, E_inter/2 0.0052, E_tot 0.379 (0.31 of it from pairs beyond 6 Å) |
| Charge, total-charge conditioned, triangle SO(2) head, no sum constraint | `iqa_charge_fulltrain_so2head_triangle.yaml` | 2387870 | `202611-0415-2605-7719` | 2026-11-05, 20 h limit, ~92k steps | finished; best val/loss 0.00404 at step 91500 |
| Charge, unconditioned, triangle SO(2) head | `iqa_charge_nochg_fulltrain_so2head_triangle.yaml` | 2387871 | `202611-0415-2614-dd9d` | 2026-11-05, 20 h limit, ~93k steps | finished; best val/loss 0.00590 at step 92880 |
