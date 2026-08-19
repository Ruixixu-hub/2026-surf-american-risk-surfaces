# Method Status

The status labels below are decisions from pre-registered gates. A STOP result
is retained as evidence and is not rewritten as a successful method.

| Method | Status | Formal evidence? | Interpretation |
|---|---|---:|---|
| CN + PSOR | `BASIC_ORIGINAL_CLASSICAL_BENCHMARK` | Yes | Basic/original American-option classical benchmark |
| CN + Policy Iteration | `STRENGTHENED_BENCHMARK_1` | Yes | Best strict iterative solver from the warm-start study |
| CN + Projected LU | `STRENGTHENED_BENCHMARK_2_NUMERICALLY_CERTIFIED` | Yes | Fastest strict solver on the frozen SURF domain |
| Rannacher/DIRK/Lobatto/nonuniform-grid audit | `HIGH_ACCURACY_REFERENCE` | Yes | Stable-mask Gamma reference allowed |
| Published DIRK-P full framework | `RETAIN_DIRK_POLICY_SINH` | Yes, 12-regime audit | All runs met the paper stop, but only 3/12 met SURF's strict `1e-12` VI gate and 0/12 passed every frozen accuracy/structure gate; runtime was essentially tied |
| Penalty/Newton | `FAILED_CORRECTNESS` | Yes | Unified 67-regime comparator; only 40/67 passed the common LCP gate and it was not faster than Policy overall |
| Positive-premium MLP warm start | `STOP_LEARNED_ACCELERATION` | Yes | Did not improve end-to-end time over conventional Policy initialization |
| POD/SVD representation | `REPRESENTATION_GO` | Yes | Low-rank price representation is viable |
| Polynomial POD coefficient map | `STOP_MAPPING` | Yes | Fast but insufficient out-of-regime accuracy |
| Primal/dual RB-VI | `STOP_ACCURACY` | Yes, validation only | Price compression did not preserve boundary/Greeks reliably |
| Boundary-aligned/localized basis | `STOP_RB_ROUTE` | Yes, validation only | Oracle localization did not pass the complete structure gate |
| Positive-premium basis operator | `STOP_BASIS_OPERATOR` | Yes, validation only | Price gate passed; boundary/Greek/LCP structure gates failed |
| PINN Arms C/D/E | `NO_FORMAL_RESULTS_YET` | No | Code/protocol ready; formal five-seed GPU study not completed |
| Positive-premium DeepONet | `NO_FORMAL_RESULTS_YET` | No | Code/protocol ready; formal five-seed GPU study not completed |

Test/stress results were not opened for methods that failed their validation
gate.

The Penalty/Newton row is an exception only in workflow, not in claim scope:
it was preregistered directly inside the held-out strict-solver comparison and
is retained as a failed comparator, never as a benchmark.
