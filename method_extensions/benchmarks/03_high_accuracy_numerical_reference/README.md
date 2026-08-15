# High-Accuracy Numerical Reference

This benchmark family studies discretization accuracy rather than online
speed. It includes Rannacher smoothing, second-order DIRK, Lobatto IIIC, and
uniform versus strike-concentrated nonuniform spatial grids, with strict Policy
Iteration at each LCP stage.

The selected reference route is the L-stable DIRK formulation with a
strike-concentrated sinh grid. It supports a stable Gamma mask; Gamma is not
claimed reliable outside that mask.

- Time-integrator experiment: [`23_greek_time_integrator_audit.py`](../../experiments/23_greek_time_integrator_audit.py)
- Spatial-grid experiment: [`26_greek_spatial_grid_audit.py`](../../experiments/26_greek_spatial_grid_audit.py)
- Temporal decision: [`greek_decision.json`](../../results/07_method_extensions/03_greek_audit/greek_decision.json)
- Spatial decision: [`spatial_greek_decision.json`](../../results/07_method_extensions/03_greek_audit/spatial_greek_decision.json)

This reference does not replace the same-grid speed benchmarks.
