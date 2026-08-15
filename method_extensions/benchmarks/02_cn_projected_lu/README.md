# Strengthened Benchmark 2: CN + Projected LU

The frozen candidate is the option-directed single projected sweep:

- American put: reverse UL factorization and projected forward substitution;
- American call: LU factorization and projected backward substitution.

Formal status: **GO_PROJECTED_LU_NUMERICALLY_CERTIFIED** on the frozen SURF
domain. It is strengthened benchmark 2.

On 67 held-out regimes, all trajectories passed the `1e-12` LCP gate. The
per-regime median LU/Policy time ratio was `0.680384`; the early-exercise-risk
subgroup ratio was `0.660768`. Four held-out q=0 calls lie outside the classic
M-matrix sufficient conditions, so the claim is numerical rather than an
unconditional theorem.

- Code: [`projected_lu.py`](../../src/american_risk_surfaces/solvers/projected_lu.py)
- Experiments: [`58`](../../experiments/58_projected_lu_protocol_and_eligibility.py),
  [`59`](../../experiments/59_projected_lu_validation_and_freeze.py),
  [`60`](../../experiments/60_projected_lu_heldout_benchmark.py),
  [`61`](../../experiments/61_projected_lu_synthesis.py)
- Decision: [`method_decision.json`](../../results/13_projected_lu/method_decision.json)
- Chinese report: [`projected_lu_结论_CN.md`](../../reports/15_projected_lu/projected_lu_结论_CN.md)
- English report: [`projected_lu_technical_report.md`](../../reports/15_projected_lu/projected_lu_technical_report.md)
