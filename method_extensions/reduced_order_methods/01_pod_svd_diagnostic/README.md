# POD/SVD Diagnostic

Train-only POD/SVD shows that the value or continuation-premium surfaces are
low rank as price fields. Eight unaligned modes passed the representation
diagnostic, but a degree-3 ridge map from option parameters to coefficients did
not meet the held-out error threshold.

Statuses: **GO_POD_BASIS_LADDER** and
**STOP_POD_COEFFICIENT_AT_DIAGNOSTIC**.

- Experiments: [`24`](../../experiments/24_pod_rank_diagnostic.py) and
  [`25`](../../experiments/25_pod_coefficient_map.py)
- POD decision: [`pod_decision.json`](../../results/07_method_extensions/04_pod/pod_decision.json)
- Coefficient decision: [`coefficient_decision.json`](../../results/07_method_extensions/05_pod_coefficient/coefficient_decision.json)
