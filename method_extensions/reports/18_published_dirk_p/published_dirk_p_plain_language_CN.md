# Published DIRK-P 高精度候选实验（中文结论）

最终决定：**RETAIN_DIRK_POLICY_SINH**。

本次实现忠实保留了论文的 L-stable Cash DIRK、两阶段 penalty 迭代、Large=1e7、tol=1e-7、二次时间网格、前两步 BE-P、论文的 [0,2K] 均匀/远端 sinh 网格以及非均匀二阶中心差分。没有在求解后把价格强行投影到 payoff 上。

不可避免的区别是：论文的一维实验是 American put；SURF 还要求 dividend call 和 q=0 call，并保留现有 Smax=4K 与 call 远端边界。这些属于明确披露的项目外推，不是论文原样案例。

12 个 regime 中有 1/12 个通过全部价格、边界、Delta、stable-mask Gamma 和 VI 门槛。

联合二阶收敛比例为 1.000（门槛 0.900）。

运行时间的 paired median candidate/reference 比值为 0.9868，即约 1.0133x speedup；p95 比值为 1.0312。

## 逐 regime 正式结果

| Regime | Price max | Boundary error | Delta max | Stable Gamma max | VI residual | 通过 |
|---|---:|---:|---:|---:|---:|:---:|
| `put_T100_s020_r005_q003` | 5.47e-06 | 0.000118 | 2.11e-05 | 0.000633 | 1.04e-11 | 否 |
| `call_T100_s020_r005_q006` | 6.06e-06 | 0.00228 | 1.85e-05 | 0.000144 | 1.32e-11 | 否 |
| `call_T100_s020_r005_q000` | 6.92e-06 | 0 | 3e-05 | 0.000219 | 5.89e-16 | 否 |
| `put_T025_s020_r001_q000` | 1.13e-05 | 0.000232 | 0.000145 | 0.00127 | 5.21e-13 | 否 |
| `call_T025_s020_r001_q000` | 1.3e-05 | 0 | 0.000148 | 0.00134 | 3.06e-16 | 否 |
| `put_T200_s060_r010_q010` | 2.69e-06 | 0.00271 | 2.62e-06 | 7.55e-06 | 4.14e-11 | 是 |
| `call_T200_s060_r010_q010` | 3.8e-06 | 0.00687 | 1.44e-06 | 3.34e-06 | 4.16e-11 | 否 |
| `put_T200_s020_r001_q010` | 3.18e-06 | 0.00359 | 1.06e-05 | 2.53e-05 | 3.96e-12 | 否 |
| `call_T200_s020_r010_q010` | 4.82e-06 | 0.002 | 4.86e-06 | 3.84e-05 | 4.16e-11 | 否 |
| `put_T050_s060_r005_q003` | 4.66e-06 | 0.00154 | 4.86e-06 | 1.2e-05 | 5.19e-12 | 否 |
| `call_T050_s060_r005_q010` | 4.57e-06 | 0.00109 | 4.23e-06 | 1.02e-05 | 1.22e-11 | 否 |
| `call_T100_s060_r001_q006` | 4.6e-06 | 0.000229 | 2.14e-06 | 7.67e-06 | 1.61e-11 | 否 |

未通过的 regimes 与原因：

- `put_T100_s020_r005_q003`：price, delta
- `call_T100_s020_r005_q006`：price, delta, gamma
- `call_T100_s020_r005_q000`：price, delta, gamma, q0_bsm
- `put_T025_s020_r001_q000`：price, delta, gamma
- `call_T025_s020_r001_q000`：price, delta, gamma, q0_bsm
- `call_T200_s060_r010_q010`：price, gamma
- `put_T200_s020_r001_q010`：price, delta, gamma
- `call_T200_s020_r010_q010`：price
- `put_T050_s060_r005_q003`：price
- `call_T050_s060_r005_q010`：price
- `call_T100_s060_r001_q006`：price

这项结果只决定候选资格；现有 DIRK+Policy+sinh 结果、当前 reference 和 poster 均未被覆盖或修改。
