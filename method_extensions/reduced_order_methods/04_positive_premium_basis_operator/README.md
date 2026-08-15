# Positive-Premium Basis Operator

A neural network maps option parameters to POD coefficients, followed by a
hard nonnegative-premium reconstruction.

Formal status: **STOP_BASIS_OPERATOR** at validation. Put and dividend-call
price gates passed, but the boundary, Greek, active-set, or LCP-structure gates
failed. Test/stress remained sealed, so there is no compliant claim that this
method beats a benchmark.

- Experiments: [`46`](../../experiments/46_basis_operator_protocol_and_data_audit.py)
  through [`51`](../../experiments/51_basis_operator_synthesis.py)
- Decision: [`method_decision.json`](../../results/11_positive_premium_basis_operator/07_synthesis/method_decision.json)
- Report: [`positive_premium_basis_operator_结论_CN.md`](../../reports/13_positive_premium_basis_operator/positive_premium_basis_operator_结论_CN.md)

Development checkpoints and POD artifacts are excluded; formal metrics and
decision evidence are included.
