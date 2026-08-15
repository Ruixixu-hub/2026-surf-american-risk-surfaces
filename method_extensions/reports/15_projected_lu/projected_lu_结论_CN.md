# Projected LU / Brennan--Schwartz 中文结论

最终状态：**GO_PROJECTED_LU_NUMERICALLY_CERTIFIED**

正式候选是 `projected_lu_single`。它与 CN+PSOR、CN+Policy 使用完全相同的
121x121 CN-LCP 和 1e-12 residual 容差。

## 正确性

- 67 个 held-out regimes 全部数值认证：`True`
- 全部满足经典理论充分条件：`False`
- 与 Policy 完整轨迹最大差异：`2.62013e-14`
- 最大 normalized LCP residual：`6.29798e-16`

## 速度

| 方法 | Median (秒) | p95 (秒) |
|---|---:|---:|
| CN+PSOR | 0.243615 | 1.41543 |
| CN+Policy | 0.0138143 | 0.0161276 |
| projected_lu_single | 0.00938379 | 0.00998221 |

逐 regime 的 median LU/Policy 比率为
`0.680384`；只看真正具有提前行权风险的
put/dividend-call，比率为
`0.660768`。

该结论只适用于冻结的 SURF 参数域。经典 M-matrix 充分条件之外的情况只能称为
“在当前数值集合中通过 residual 和 Policy 对照认证”，不能声称受到无条件理论保证。

图表：`solver_runtime.png`、`paired_speed_ratio.png`、
`eligibility_by_split.png`。
