# Benchmark Hierarchy

| Level | Method | Role | Current state |
|---|---|---|---|
| Historical benchmark | CN + PSOR | Original American-option LCP benchmark | Retained |
| Strengthened benchmark 1 | CN + Policy Iteration | Strict iterative LCP solver | Formal benchmark |
| Strengthened benchmark 2 | CN + Projected LU / Brennan--Schwartz | Strict direct projected sweep on the frozen CN-LCP | Formal benchmark; numerically certified on the frozen SURF domain |
| High-accuracy reference | DIRK/Rannacher/Lobatto audit + Policy + sinh grid | Accuracy, boundary, Delta, and stable-mask Gamma reference | Reference only; not the main speed competitor |

All speed benchmarks solve the American-option obstacle problem. European BSM
prices and Greeks are controls used to validate the base PDE and the
no-dividend American-call theorem.

The headline same-Mac strict-solver timing is stored in
[`portfolio_summary/benchmark_comparison.csv`](portfolio_summary/benchmark_comparison.csv).
