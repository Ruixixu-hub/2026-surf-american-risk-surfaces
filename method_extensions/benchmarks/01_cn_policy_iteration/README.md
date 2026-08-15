# Strengthened Benchmark 1: CN + Policy Iteration

Policy Iteration solves the same frozen American CN-LCP as PSOR by repeatedly
updating the exercise/continuation active set and solving the resulting
tridiagonal linear system.

Formal status: **STRENGTHENED_BENCHMARK_1**.

The learned MLP initializer failed the end-to-end acceleration gate, so the
selected method is conventional previous-time-slice Policy Iteration.

- Code: [`policy_iteration.py`](../../src/american_risk_surfaces/solvers/policy_iteration.py)
- Shared marcher: [`american_lcp.py`](../../src/american_risk_surfaces/solvers/american_lcp.py)
- Experiment: [`22_policy_warmstart_benchmark.py`](../../experiments/22_policy_warmstart_benchmark.py)
- Decision: [`method_decision.json`](../../results/07_method_extensions/02_warmstart/method_decision.json)
- Report: [`implementation_report.md`](../../reports/09_method_extensions/implementation_report.md)
