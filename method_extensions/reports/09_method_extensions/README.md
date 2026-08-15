# Method-extension reproducibility guide

Run from the repository root with `PYTHONPATH=src`. The scripts are deliberately
numbered after the existing Stage 1-6 workflow and write only below
`results/07_method_extensions/` and `reports/09_method_extensions/`.

1. `python3 experiments/21_freeze_method_extension_protocol.py`
2. `python3 experiments/22_policy_warmstart_benchmark.py`
3. `python3 experiments/23_greek_time_integrator_audit.py`
4. `python3 experiments/26_greek_spatial_grid_audit.py`
5. `python3 experiments/24_pod_rank_diagnostic.py`
6. `python3 experiments/25_pod_coefficient_map.py`
7. `python3 experiments/27_method_extension_synthesis.py`
8. `MPLBACKEND=Agg python3 experiments/28_method_extension_figures.py`

The benchmark defaults to 5 warm-ups and 30 randomized-order measured repeats
for all test and stress regimes. Use its CLI reduction flags only for smoke
testing; such output is marked `protocol_complete=false` and is not suitable for
the final decision.

The stage gates are binding. A `STOP` or `DEFER` result is retained as evidence;
the scripts do not continue automatically into DeepONet, FNO, multi-fidelity, or
UQ when their prerequisite method has failed.
