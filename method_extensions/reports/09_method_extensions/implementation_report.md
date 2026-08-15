# SURF Method-Extension Implementation Report

## Outcome

The frozen normalized LCP tolerance is `1e-12`. The requested `1e-10` target was not frozen because its maximum price difference from `1e-12` was `1.33065e-07`, above `1e-9 K`.

The selected online classical method is `policy_iteration_previous_slice`. The MLP warm-start speedup was `-44.429%`, so learned acceleration is stopped under the pre-registered gate.

The temporal Greek winner is `dirk_lstable_quadratic`. The current joint Gamma decision is `UNBLOCK_GAMMA_ON_STABLE_MASK`.

POD passed its rank test with `8` modes, but the simple coefficient map failed (`0.00680275` versus acceptance `0.00247495`).

## Method decisions

| Method | Role | Status | Decision |
|---|---|---|---|
| Policy Iteration | strict discrete-LCP solver | **GO** | policy_iteration_previous_slice |
| Positive-premium MLP warm-start | initializer only; never the final correctness mechanism | **STOP** | STOP_LEARNED_ACCELERATION_KEEP_POLICY_ITERATION |
| DIRK / Lobatto Greek reference | high-accuracy label and convergence audit | **GO** | UNBLOCK_GAMMA_ON_STABLE_MASK |
| POD/SVD rank diagnostic | falsification test before a large operator network | **GO** | 8 unaligned modes |
| Polynomial POD coefficient map | minimal nonintrusive basis surrogate | **STOP** | STOP_POD_COEFFICIENT_AT_DIAGNOSTIC |
| Primal/dual Reduced-Basis VI | intrusive many-query obstacle solver | **DEFER** | Needs full-grid primal and multiplier snapshots, stability enrichment, and an online VI estimator; a projected PCA solve is not an honest substitute. |
| DeepONet / FNO / localized operator | large learned operator | **DEFER** | Do not implement FNO: unaligned POD is already low-rank; revisit basis operator or DeepONet only after RB VI. |
| Multi-fidelity and UQ fallback | label-efficiency and deployment safety | **DEFER** | Requires a retained surrogate; no surrogate has passed its held-out gate yet. |

## Recommended next experiment

Implement a genuine primal/dual Reduced-Basis variational-inequality prototype. First regenerate full-grid train snapshots and multipliers with the frozen Policy Iteration solver; then add dual angle-greedy selection and stability enrichment. Do not call a direct POD coefficient regression a Reduced-Basis VI.

DeepONet, FNO, multi-fidelity, and UQ remain deferred because their prerequisite surrogate gate has not passed. This is the intended stage-gated stopping behavior, not missing evidence hidden from the report.
