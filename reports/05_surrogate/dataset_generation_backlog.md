# Dataset Generation Backlog

This backlog records decisions that must be reviewed before any surrogate dataset is generated.
It is not a dataset construction script and it does not approve label export.

## Human Review Items

- Approve or revise the first parameter grid: `T`, `sigma`, `r`, `q`, option families, and grid size.
- Decide whether the first generated dataset should include all planned parameter combinations or a smaller dry-run subset.
- Confirm the default solver source: `american_crank_nicolson_psor_price`, `baseline_cn_psor`.
- Confirm that Rannacher-smoothed outputs remain excluded unless separately labelled.
- Approve the boundary threshold and interpolation metadata policy.
- Approve the Greek mask width for payoff-kink, maturity, and boundary-near regions.
- Decide whether Delta is allowed as an early diagnostic label or should remain metadata-only in version 1.
- Decide whether Gamma is excluded entirely from first training targets or included only under `gamma_allowed_mask`.
- Define the exact regime-level train/test split before any rows are generated.
- Choose the future storage format, likely `.npz` arrays plus CSV/JSON manifests, without creating those files in this design step.

## Deferred Technical Checks

- Run selected higher-grid confirmation cases before broad generation.
- Compare a small dry-run subset against existing Pilot 01 and quantitative analytics diagnostics.
- Verify all future generated regimes include PSOR, LCP, boundary, and mask metadata.
- Define failure handling for non-converged regimes before generation starts.
- Define exact sample weighting only after boundary and Greek label policy is accepted.

## Explicitly Blocked Until Later

- Dataset construction.
- `.npz`, `.npy`, `.pt`, or training-table export.
- Neural surrogate training.
- Broad production-scale sweeps.
- Production Greek labels.
- Final claims about all parameter regimes.
