# Positive-Premium Basis Operator 实验结论

## 一句话结论

本方法最终状态为 **STOP_BASIS_OPERATOR**。POD basis 能很精确地表示价格面，12-mode P2 的五-seed 价格门也通过；但是行权边界、Greeks 或完整 LCP 结构没有达到预先冻结的要求，因此不允许打开 test/stress，也不能声称它优于 CN+PSOR 或 CN+Policy Iteration。

## 主要数据

| 期权族 | 8-mode POD 表示门 | 32-mode oracle 最差 RMSE | 12-mode P2 五-seed最差价格 RMSE | 价格门 | 最终 family 状态 |
|---|---:|---:|---:|---|---|
| American put | 通过 | 1.30404e-05 | 0.000340743 | 通过 | STOP_STRUCTURE |
| dividend American call | 通过 | 1.76267e-05 | 0.000447061 | 通过 | STOP_STRUCTURE |

冻结的价格门是 RMSE ≤ 4.94989e-4。put 与 dividend call 均满足，但完整 gate 的最大超限倍数分别为 14.151 和 34.880，说明仅凭价格 RMSE 会高估该方法对风险面的质量。

## 这代表什么

1. 失败原因不是“POD basis 不够低秩”：oracle projection 在 8 modes 已过门，32 modes 的误差更低。
2. 小网络确实能把总体价格学到门槛以内，但不能同时可靠重现移动行权边界、Delta/Gamma 与 LCP 互补结构。
3. 因 validation 失败，held-out 文件继续封存；没有运行速度竞争和 exact hybrid。因此不存在“超过 CN+PSOR/Policy”的合规结论。
4. 预注册路线要求停止继续增加 POD modes。下一方法是 positive-premium DeepONet，用更灵活的分支/主干表示检验全局 coefficient map 是否是瓶颈。
