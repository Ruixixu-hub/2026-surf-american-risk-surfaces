# Pointwise Positive-Premium MLP Warm Start

The supervised MLP predicts a nonnegative continuation premium and supplies a
warm start to PSOR or Policy Iteration. The strict LCP solver still determines
the final answer.

Formal status: **STOP_LEARNED_ACCELERATION_KEEP_POLICY_ITERATION**. The learned
initializer did not reduce end-to-end time relative to previous-slice Policy
Iteration under the frozen tolerance.

- Experiment: [`22_policy_warmstart_benchmark.py`](../../experiments/22_policy_warmstart_benchmark.py)
- Decision: [`method_decision.json`](../../results/07_method_extensions/02_warmstart/method_decision.json)
