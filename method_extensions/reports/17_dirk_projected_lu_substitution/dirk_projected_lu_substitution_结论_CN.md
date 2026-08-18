# DIRK+sinh 内部求解器替换实验：中文结论

最终决定：**RETAIN_DIRK_POLICY_SINH**。

本实验没有改变 DIRK、sinh 网格、分辨率、边界、Greek 算法或稳定 mask；只把每个隐式障碍问题的 Policy Iteration 换成了 Projected LU。

数值结果方面，12/12 个 regime 的 LCP residual、价格、boundary、Delta、stable-mask Gamma 和金融结构检查都通过。
结构适用性方面只有 11/12 通过；`put_T200_s020_r001_q010` 的全部 1,918 个 stages 出现 positive off-diagonal，因此不满足预注册的 M-matrix sufficient condition。
速度方面，Projected LU / Policy 的配对 median 时间比为 `1.74324`：Projected LU 约慢 `74.32%`，没有加速。
Policy pooled p95 为 `0.824961s`，Projected LU pooled p95 为 `1.42365s`。
观察到的最大差异为：价格 `1.05471e-14`，boundary `5.19037e-11`，Delta `1.29674e-13`，stable-mask Gamma `4.35283e-11`。

未通过的 regime：
- `put_T200_s020_r001_q010`：structural_or_pivot

结构条件失败统计：`{"positive_offdiagonal": 1918}`。

解释限制：Projected LU 即使成功，也只是更快地求解同一个离散 stage LCP；高精度仍主要来自 L-stable DIRK 与 strike-concentrated sinh grid。
