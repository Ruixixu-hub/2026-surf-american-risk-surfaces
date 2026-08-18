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

The published DIRKa-P framework from in 't Hout was also implemented as a
separate method-level candidate, including its finite penalty, quadratic time
mesh, first-two-step BE-P damping, and published uniform-to-2K/sinh-tail
spatial mesh. It retained regular second-order behaviour, but only 3 of 12
regimes met the existing strict `1e-12` normalized VI gate and no regime passed
every frozen accuracy/structure gate. The selected reference was not changed.

- Candidate protocol: [`protocol.json`](../../results/16_published_dirk_p/00_protocol/protocol.json)
- Metric-gate correction: [`metric_gate_correction.json`](../../results/16_published_dirk_p/00_protocol/metric_gate_correction.json)
- Twelve-regime metrics: [`regime_metrics.csv`](../../results/16_published_dirk_p/02_audit/regime_metrics.csv)
- Decision: [`method_decision.json`](../../results/16_published_dirk_p/method_decision.json)
- Technical report: [`published_dirk_p_technical_report.md`](../../reports/18_published_dirk_p/published_dirk_p_technical_report.md)

This reference does not replace the same-grid speed benchmarks.
