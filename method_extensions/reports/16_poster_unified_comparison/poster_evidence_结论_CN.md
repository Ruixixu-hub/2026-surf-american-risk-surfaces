# Poster 统一实验结论

- CN+PSOR 是 Basic / Original Classical Benchmark。
- CN+Policy Iteration 是 Strengthened Classical Benchmark 1。
- CN+Projected LU 是 Strengthened Classical Benchmark 2。
- DIRK+Policy+sinh 是 High-Accuracy Numerical Reference。

Strict comparison 状态：**STRICT_THREE_CONFIRMED**。Penalty/Newton 状态：**FAILED_CORRECTNESS**。

Projected LU median=0.0102586s，Policy median=0.0159997s，PSOR median=0.309707s。

Paired median ratios：Policy/PSOR=0.0557，Projected-LU/Policy=0.6776，Penalty/Policy=1.0067。在 31 个真正有提前行权风险的 regimes 中，Penalty/Policy=2.6403。

Penalty/Newton 通过共同 residual gate 40/67 regimes；其最大 normalized LCP residual 为 2.17228e-11。无论速度如何，只有全部共同正确性 gate 通过才可提升为 benchmark。

Accuracy-reference audit：**REUSE_EXISTING_REFERENCE_EVIDENCE**；Gamma 仅允许在 validated stable mask 上报告。
