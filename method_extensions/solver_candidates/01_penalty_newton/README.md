# Penalty/Newton Candidate Comparator

This method replaces exact complementarity by a finite penalty equation and
uses semismooth Newton iterations with tridiagonal linear solves. It was added
to the same frozen CN-LCP comparison as PSOR, Policy Iteration, and Projected
LU, but it is **not** a benchmark.

On the 67 held-out regimes:

- pooled median: `0.0207950 s`;
- pooled p95: `0.0622017 s`;
- common residual gate: `40/67`;
- strong Policy-match certification: `40/67`;
- decision: **FAILED_CORRECTNESS**.

Its paired median time ratio relative to Policy Iteration was `1.00668`, and
on the 31 put/dividend-call regimes it was `2.64030`. The evidence therefore
does not support either correctness promotion or a speed advantage.

- Code: [`penalty_newton.py`](../../src/american_risk_surfaces/solvers/penalty_newton.py)
- Unified experiment: [`62_strict_solver_poster_comparison.py`](../../experiments/62_strict_solver_poster_comparison.py)
- Decision: [`strict_solver_decision.json`](../../results/14_poster_unified_comparison/01_strict_solvers/strict_solver_decision.json)
- Report: [`poster_evidence_report.md`](../../reports/16_poster_unified_comparison/poster_evidence_report.md)
