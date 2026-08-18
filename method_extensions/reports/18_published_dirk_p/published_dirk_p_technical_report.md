# Published DIRK-P High-Accuracy Candidate Audit

Decision: **RETAIN_DIRK_POLICY_SINH**

## Fidelity

The Cash DIRKa coefficients, DIRK-P stage equations, finite penalty (Large=1e7), tol=1e-7 stop, quadratic time mesh, first-two-step BE-P damping, the published uniform-to-2K/sinh-tail grid (d=K/10), and nonuniform central differences were implemented directly. No payoff projection was used to hide finite-penalty error.

Unavoidable SURF extensions are the dividend-paying and no-dividend calls, the existing Smax=4K regimes rather than the paper's representative 5K example, and SURF's exact far-call boundary. The paper's one-dimensional experiment itself is a put.

## Twelve-regime evidence

The paper stopping rule and the SURF acceptance rule are distinct: all runs may converge under the published penalty iteration while still failing the common frozen normalized obstacle/LCP tolerance of 1e-12.

Strict VI gate: 3/12 regimes pass; normalized obstacle and LCP tolerances are both 1e-12.

| Regime | Price max/gate | Boundary error/gate | Delta max/gate | Gamma max/gate | VI residual/gate | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| put_T100_s020_r005_q003 | 5.47e-06/1.61e-06 | 0.000118/0.00494 | 2.11e-05/1.22e-05 | 0.000633/0.000955 | 1.04e-11/1e-12 | no |
| call_T100_s020_r005_q006 | 6.06e-06/1.55e-06 | 0.00228/0.00527 | 1.85e-05/1.45e-05 | 0.000144/9.01e-05 | 1.32e-11/1e-12 | no |
| call_T100_s020_r005_q000 | 6.92e-06/1.52e-06 | 0/0.00494 | 3e-05/1.1e-05 | 0.000219/0.000135 | 5.89e-16/1e-12 | no |
| put_T025_s020_r001_q000 | 1.13e-05/1.57e-06 | 0.000232/0.00494 | 0.000145/4.11e-05 | 0.00127/0.000272 | 5.21e-13/1e-12 | no |
| call_T025_s020_r001_q000 | 1.3e-05/1.47e-06 | 0/0.00494 | 0.000148/4.05e-05 | 0.00134/0.000231 | 3.06e-16/1e-12 | no |
| put_T200_s060_r010_q010 | 2.69e-06/3.17e-06 | 0.00271/0.00949 | 2.62e-06/8.49e-06 | 7.55e-06/8.29e-06 | 4.14e-11/1e-12 | no |
| call_T200_s060_r010_q010 | 3.8e-06/3.51e-06 | 0.00687/0.0379 | 1.44e-06/4.21e-06 | 3.34e-06/1.74e-06 | 4.16e-11/1e-12 | no |
| put_T200_s020_r001_q010 | 3.18e-06/1.62e-06 | 0.00359/0.0135 | 1.06e-05/5.35e-06 | 2.53e-05/1.55e-05 | 3.96e-12/1e-12 | no |
| call_T200_s020_r010_q010 | 4.82e-06/1.59e-06 | 0.002/0.00625 | 4.86e-06/5.42e-06 | 3.84e-05/4.45e-05 | 4.16e-11/1e-12 | no |
| put_T050_s060_r005_q003 | 4.66e-06/2.74e-06 | 0.00154/0.00809 | 4.86e-06/5.7e-06 | 1.2e-05/1.7e-05 | 5.19e-12/1e-12 | no |
| call_T050_s060_r005_q010 | 4.57e-06/2.65e-06 | 0.00109/0.0133 | 4.23e-06/5.25e-06 | 1.02e-05/2.03e-05 | 1.22e-11/1e-12 | no |
| call_T100_s060_r001_q006 | 4.6e-06/3.38e-06 | 0.000229/0.0198 | 2.14e-06/4.54e-06 | 7.67e-06/7.94e-06 | 1.61e-11/1e-12 | no |

Joint second-order fraction: 1.000 (frozen gate 0.900).

## Runtime

Pooled median: 0.615159 s (candidate) versus 0.618124 s (current reference).

Paired median ratio candidate/reference: 0.9868; speedup: 1.0133x; pooled p95 ratio: 1.0312.

| Regime | Candidate median (s) | Current reference median (s) | Candidate/reference |
|---|---:|---:|---:|
| put_T100_s020_r005_q003 | 0.584631 | 0.603896 | 0.9681 |
| call_T100_s020_r005_q006 | 0.595551 | 0.607961 | 0.9796 |
| call_T100_s020_r005_q000 | 0.563444 | 0.591434 | 0.9527 |
| put_T025_s020_r001_q000 | 0.581525 | 0.604411 | 0.9621 |
| call_T025_s020_r001_q000 | 0.565045 | 0.593769 | 0.9516 |
| put_T200_s060_r010_q010 | 0.623487 | 0.619055 | 1.0072 |
| call_T200_s060_r010_q010 | 0.690275 | 0.630699 | 1.0945 |
| put_T200_s020_r001_q010 | 0.580301 | 0.600968 | 0.9656 |
| call_T200_s020_r010_q010 | 0.621609 | 0.62532 | 0.9941 |
| put_T050_s060_r005_q003 | 0.752776 | 0.741193 | 1.0156 |
| call_T050_s060_r005_q010 | 0.710065 | 0.668454 | 1.0622 |
| call_T100_s060_r001_q006 | 0.705929 | 0.636665 | 1.1088 |

The current reference and poster were not modified.
