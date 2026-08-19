# DIRK+sinh Projected-LU Solver-Substitution Audit

Decision: **RETAIN_DIRK_POLICY_SINH**

Only the algebraic LCP solver was changed. The Cash DIRK coefficients, quadratic time grid, two damping steps, M=480/N=960 resolution, current sinh mapping, boundaries, stable Greek query mask, and 1e-12 LCP gate were frozen.

| Regime | Structural | LCP | Price | Boundary | Delta | Gamma | LU/Policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| put_T100_s020_r005_q003 | True | True | True | True | True | True | 1.81992 |
| call_T100_s020_r005_q006 | True | True | True | True | True | True | 1.73645 |
| call_T100_s020_r005_q000 | True | True | True | True | True | True | 1.74546 |
| put_T025_s020_r001_q000 | True | True | True | True | True | True | 1.79282 |
| call_T025_s020_r001_q000 | True | True | True | True | True | True | 1.74586 |
| put_T200_s060_r010_q010 | True | True | True | True | True | True | 1.71851 |
| call_T200_s060_r010_q010 | True | True | True | True | True | True | 1.72648 |
| put_T200_s020_r001_q010 | False | True | True | True | True | True | 1.88185 |
| call_T200_s020_r010_q010 | True | True | True | True | True | True | 1.74045 |
| put_T050_s060_r005_q003 | True | True | True | True | True | True | 1.79239 |
| call_T050_s060_r005_q010 | True | True | True | True | True | True | 1.72995 |
| call_T100_s060_r001_q006 | True | True | True | True | True | True | 1.74103 |

Paired median runtime ratio: `1.74324`; speedup: `0.573644x`.

| Method | Pooled median (s) | Pooled p95 (s) | Pooled p99 (s) |
|---|---:|---:|---:|
| policy_iteration | 0.648357 | 0.824961 | 1.00319 |
| projected_lu_single | 1.13893 | 1.42365 | 1.58772 |

All 12 regimes passed the numerical LCP, price, boundary, Delta, Gamma, and financial-output gates. The largest observed differences were `1.05471e-14` for price, `5.19037e-11` for boundary, `1.29674e-13` for Delta, and `4.35283e-11` for stable-mask Gamma.

Structural failure counts: {"positive_offdiagonal": 1918}

A faster result alone does not promote the candidate; all structural, residual, price, boundary, Delta, and stable-mask Gamma gates must pass.
