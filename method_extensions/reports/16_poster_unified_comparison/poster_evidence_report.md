# Unified Poster Evidence Report

## Benchmark roles

- CN+PSOR: **Basic / Original Classical Benchmark**.
- CN+Policy Iteration: **Strengthened Classical Benchmark 1**.
- CN+Projected LU: **Strengthened Classical Benchmark 2**.
- DIRK+Policy+sinh: **High-Accuracy Numerical Reference**.
- Penalty/Newton: candidate comparator; correctness is judged before speed.

## Strict same-CN-LCP comparison

| Arm | Median (s) | P95 (s) | Common residual gate | Strong Policy-match certification | Max normalized LCP residual |
|---|---:|---:|---:|---:|---:|
| psor | 0.309707 | 1.43094 | 67/67 | 66/67 | 9.99992e-13 |
| policy_iteration | 0.0159997 | 0.0224767 | 67/67 | 67/67 | 6.47702e-16 |
| projected_lu_single | 0.0102586 | 0.0144603 | 67/67 | 67/67 | 6.29798e-16 |
| penalty_newton | 0.020795 | 0.0622017 | 40/67 | 40/67 | 2.17228e-11 |

Penalty/Newton decision: **FAILED_CORRECTNESS**.

Paired median Policy/PSOR ratio: 0.0557; Projected-LU/Policy ratio: 0.6776; Penalty/Policy ratio: 1.0067.
For the 31 put/dividend-call regimes, Projected-LU/Policy is 0.6582 and Penalty/Policy is 2.6403.

## Accuracy-reference evidence

Audit decision: **REUSE_EXISTING_REFERENCE_EVIDENCE**.

Temporal and spatial evidence are reported separately from online solver timing. Gamma claims remain restricted to the validated stable mask.

### Temporal summary

| Method | Stable fraction | Median Gamma max error |
|---|---:|---:|
| cn_quadratic | 1.000 | 1.65742e-06 |
| rannacher_cn_quadratic | 1.000 | 1.6565e-06 |
| dirk_lstable_quadratic | 1.000 | 6.94696e-07 |
| lobatto_iiic_penalty_quadratic | 0.917 | 3.71535e-06 |

### Spatial summary

| Grid | Stable fraction | Median Gamma max error |
|---|---:|---:|
| uniform | 1.000 | 0.000152088 |
| sinh_strike_concentrated | 0.917 | 2.59175e-05 |

Reference limitation: Fine DIRK+sinh refinement is an internal numerical reference, not an analytic American-option theorem or a speed benchmark.
