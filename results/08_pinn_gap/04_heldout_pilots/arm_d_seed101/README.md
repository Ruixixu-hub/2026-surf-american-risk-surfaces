# Arm D seed-101 held-out pilot

This directory records the completed **Arm D, seed 101, single-seed held-out pilot**.
It is not the registered five-seed held-out experiment and must not be described as
random-seed robustness evidence.

## Scope and completion

- Method: Arm D `etc_fb_mixture` (exact-terminal lift plus Fischer--Burmeister loss
  and fixed mixture/curriculum sampling).
- Seed: `101`, selected on validation before held-out scoring.
- Held-out regimes: 67 total (43 test and 24 stress holdout).
- Training status: 67/67 `COMPLETE`, with 40,000 Adam steps per regime.
- Failures and budget exhaustion: none.
- Total recorded training time: 22.1549 hours; median per regime: 1,162.19 seconds.
- Scoring: completed once against the frozen DIRK + Policy Iteration reference.

## Headline median metrics

| Method | Price RMSE | Delta RMSE | Scaled Gamma RMSE |
|---|---:|---:|---:|
| CN + PSOR (A) | 1.1924e-4 | 4.2058e-4 | 8.0539e-3 |
| CN + Policy Iteration (B) | 1.1924e-4 | 4.2058e-4 | 8.0539e-3 |
| Arm D, seed 101 | 7.3710e-4 | 2.5433e-3 | 1.7583e-2 |

Arm D's median price RMSE is 6.1817 times the strengthened classical baseline.
The direct PINN output is not a strict LCP solution; its median `vi_p95` is
1.8567e-3. The median boundary-found rate is 1.0, but this aggregate hides
subgroup weakness: 25 of 67 regimes have a boundary-found rate below 0.5.

## Files committed to Git

- `pilot_protocol_amendment.json`: declared single-seed protocol and selection disclosure.
- `single_seed_pilot_summary.json`: frozen headline summary.
- `scoring/`: complete per-regime PINN and classical metrics.
- `training/`: job manifest, aggregate status table, and per-regime history/status/heartbeat files.
- `high_accuracy_reference/reference_manifest.json`: reference-generation provenance.

Binary checkpoints, prediction surfaces, and reference surfaces are intentionally
excluded from Git history. The complete 677,973,093-byte archive is attached to the
GitHub release [`arm-d-seed101-pilot-v1`](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/releases/tag/arm-d-seed101-pilot-v1).
Its SHA-256 is recorded in `result_package_manifest.json`.
