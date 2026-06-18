# Dataset Construction Backlog

This backlog records items that must be resolved before any surrogate dataset is generated.
It is a planning artifact only. It does not approve array export, label export, model files, or
neural training.

## Human Review Before Generation

- Approve the 288 planned regimes and confirm that q=0 call regimes remain validation controls.
- Approve the eight-regime dry-run subset before any full small-grid generation.
- Approve the deterministic regime-level split counts: 202 train, 19 validation, 43 test, and 24 stress-holdout regimes.
- Confirm the default label source: `american_crank_nicolson_psor_price`, `baseline_cn_psor`.
- Confirm that Rannacher-smoothed outputs remain excluded unless separately labelled in a later gate.
- Approve the storage package structure before any `.npz`, `.npy`, `.pt`, or training table is created.
- Approve diagnostic thresholds for PSOR convergence, obstacle violation, equation violation, and complementarity product.
- Approve exact mask widths for payoff-kink, boundary-near, maturity-row, Delta, and Gamma handling.
- Decide whether Delta is an optional diagnostic label in the first generated version or metadata-only.
- Decide whether Gamma is excluded from first training targets or included only under `gamma_allowed_mask`.

## Dry-Run Review Items

- Compare the future dry-run diagnostics against Pilot 01 and quantitative risk analytics.
- Check that all dry-run regimes record solver, grid, split, boundary, LCP, Greek-mask, and downstream-use metadata.
- Verify no-dividend call controls are interpreted as controls rather than dividend-call early-exercise evidence.
- Review any dry-run regime with high PSOR iteration counts or unexpectedly high runtime.
- Confirm boundary found/not-found reasons before accepting boundary metadata.

## Explicitly Blocked Until Later

- Full dataset construction.
- `.npz`, `.npy`, `.pt`, or training-table export.
- Label-array export.
- Neural surrogate training.
- Model-file creation.
- Broad production-scale sweeps.
- Production Greek labels.
- Final claims about all parameter regimes.
