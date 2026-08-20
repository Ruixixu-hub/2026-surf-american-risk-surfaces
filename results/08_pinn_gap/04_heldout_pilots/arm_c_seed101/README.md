# Arm C seed-101 held-out follow-up

This directory records the completed **Arm C, seed 101, single-seed held-out
follow-up**. It is not the registered five-seed held-out experiment and must not
be described as random-seed robustness evidence.

## Scope and completion

- Method: Arm C `soft_lcp` vanilla PINN.
- Architecture: residual network with 4 blocks, 2 layers per block, and width 50.
- Seed: `101`, fixed to match the Arm D seed-101 pilot; it was not selected as
  the best Arm C validation seed.
- Held-out regimes: 67 total (43 test and 24 stress holdout).
- Training status: 67/67 `COMPLETE`, with 40,000 Adam steps per regime.
- Failures and budget exhaustion: none.
- Sum of recorded per-job training time: 116.5456 process-hours; median per
  regime: 6,492.46 seconds. Four jobs shared one RTX 5090 during most of the run,
  so the sum is not the actual wall-clock duration.
- Scoring: completed once against the frozen DIRK + Policy Iteration reference.

## Headline median metrics

| Method | Price RMSE | Delta RMSE | Scaled Gamma RMSE |
|---|---:|---:|---:|
| CN + PSOR (A) | 1.1924e-4 | 4.2058e-4 | 8.0539e-3 |
| CN + Policy Iteration (B) | 1.1924e-4 | 4.2058e-4 | 8.0539e-3 |
| Arm C, seed 101 | 4.4222e-2 | 6.1877e-2 | 5.0575e-1 |

Arm C's median price RMSE is 370.8673 times the strengthened classical
baseline. The direct PINN output is not a strict LCP solution: its median
`vi_p95` is 1.9538e-2 and its median maximum normalized discrete LCP residual is
1.7861e-3.

The aggregate median boundary-found rate of 1.0 is misleading because it
includes regimes with no reference early-exercise boundary. Among the 31
regimes with a reference boundary, 30 have a found rate below 0.5 and 25 have no
predicted boundary points.

## Files committed to Git

- `pilot_protocol_amendment.json`: declared single-seed protocol and selection disclosure.
- `single_seed_pilot_summary.json`: frozen headline summary.
- `scoring/`: complete per-regime PINN and classical metrics.
- `training/`: the initial single-process and final four-shard job manifests and
  status tables, plus per-regime history/status/heartbeat files. The initial
  `000_of_001` status table contains only the first completed job; the four
  `of_004` tables and the 67 per-regime status files form the complete record.
- `high_accuracy_reference/reference_manifest.json`: reference-generation provenance.

Binary checkpoints, prediction surfaces, and reference surfaces are intentionally
excluded from Git history. The complete 680,357,249-byte results archive is
attached to the GitHub release
[`arm-c-seed101-followup-v1`](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/releases/tag/arm-c-seed101-followup-v1).
Its SHA-256 is recorded in `result_package_manifest.json`. The release also
contains a supplemental partial log snapshot; it is not the canonical completion
record.
