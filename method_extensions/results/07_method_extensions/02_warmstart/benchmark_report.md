# Policy Iteration and Positive-Premium Warm-Start Benchmark

Decision: **STOP_LEARNED_ACCELERATION_KEEP_POLICY_ITERATION**

Selected method: `policy_iteration_previous_slice`

Frozen normalized LCP tolerance: `1e-12`

Protocol complete: `True`

| Arm | Median (s) | p95 (s) | Median iterations |
|---|---:|---:|---:|
| A_previous_psor | 0.311578 | 1.62845 | 3360 |
| B_previous_policy | 0.0159815 | 0.0209976 | 120 |
| C_mlp_psor | 0.323051 | 1.62715 | 3360 |
| D_mlp_policy | 0.0230818 | 0.0336051 | 179 |

Best learned-vs-classical median speedup: `-44.429%`.

The learned path is accepted only when the 20% end-to-end gate, p95 gate, strict-tolerance gate, and both held-out split gates pass together.
