# Projected LU / Brennan--Schwartz Technical Report

Decision: **GO_PROJECTED_LU_NUMERICALLY_CERTIFIED**

The frozen candidate was `projected_lu_single`. It was compared with CN+PSOR and
previous-slice CN+Policy on the identical 121x121 CN LCP at the frozen 1e-12
residual tolerance.

## Correctness

- All numerically certified: `True`
- All theorem-eligible: `False`
- Maximum full-trajectory difference versus Policy: `2.62013e-14`
- Maximum normalized LCP residual: `6.29798e-16`

## Runtime

| Method | Median (s) | p95 (s) | p99 (s) |
|---|---:|---:|---:|
| CN+PSOR | 0.243615 | 1.41543 | 1.47254 |
| CN+Policy | 0.0138143 | 0.0161276 | 0.0180133 |
| projected_lu_single | 0.00938379 | 0.00998221 | 0.0109708 |

The paired median LU/Policy ratio was
`0.680384`; on the put/dividend-call
subgroup it was `0.660768`.

The claim is limited to the frozen SURF numerical domain. Regimes outside the
classic M-matrix sufficient conditions are reported as numerically certified,
not theorem-certified.

Figures: `solver_runtime.png`, `paired_speed_ratio.png`,
`eligibility_by_split.png`.
