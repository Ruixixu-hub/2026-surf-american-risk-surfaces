# Method Status

The status labels below are decisions from pre-registered gates. A STOP result
is retained as evidence and is not rewritten as a successful method.

| Method | Status | Formal evidence? | Interpretation |
|---|---|---:|---|
| CN + PSOR | `HISTORICAL_BENCHMARK` | Yes | Original American-option benchmark |
| CN + Policy Iteration | `STRENGTHENED_BENCHMARK_1` | Yes | Best strict iterative solver from the warm-start study |
| CN + Projected LU | `STRENGTHENED_BENCHMARK_2_NUMERICALLY_CERTIFIED` | Yes | Fastest strict solver on the frozen SURF domain |
| Rannacher/DIRK/Lobatto/nonuniform-grid audit | `HIGH_ACCURACY_REFERENCE` | Yes | Stable-mask Gamma reference allowed |
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
