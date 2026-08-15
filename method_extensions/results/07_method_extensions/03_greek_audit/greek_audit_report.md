# Greek Time-Integrator Audit

Decision: **GAMMA_REFERENCE_CANDIDATE**

Best temporal method: `dirk_lstable_quadratic`

| Method | Stable fraction | Median Delta max error | Median Gamma max error |
|---|---:|---:|---:|
| cn_quadratic | 1.000 | 1.88547e-07 | 1.65742e-06 |
| rannacher_cn_quadratic | 1.000 | 1.88249e-07 | 1.6565e-06 |
| dirk_lstable_quadratic | 1.000 | 7.53885e-08 | 6.94696e-07 |
| lobatto_iiic_penalty_quadratic | 0.917 | 4.4699e-07 | 3.71535e-06 |

Gamma remains blocked unless both this temporal gate and a later spatial-grid gate pass.
