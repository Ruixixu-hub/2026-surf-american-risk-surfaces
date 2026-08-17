# Zihan 方法进展修订摘要

## 证据范围

本摘要同时覆盖：

1. 默认分支中的 CN/PSOR、supervised risk-surface 和 Stage 21--23 PINN-inspired experiments；
2. [`campaign/independent-methods`](https://github.com/zihan-liang/american-option-risk-surfaces/tree/campaign/independent-methods) 分支中的 M01--M24 campaign。

Independent branch 核对 commit：`198b9bef9c7f7de22997ba5ea7b986c2a0e50c47`。

## 一、Classical solver 与数据基础

- 288 个 application regimes：144 put、108 dividend call、36 no-dividend call controls。
- 223 个 clean regimes、3,264,943 samples；split 156/33/34。
- American put obstacle violation 0，complementarity residual 4.009e-12。
- M01 在 9 个预注册 American-put screen regimes 上支持 CN+Policy：median 0.00731276 s，对比 CN+PSOR 0.140613 s，改善 94.799%；max natural residual 6.06898e-11。

M01 的准确结论是 scoped `candidate_supported`。它没有访问 blind holdout，不能与 Ruixi 的 67-regime Projected-LU experiment 合并为同一张公平 runtime table，但可以作为 Policy Iteration 强于 PSOR 的独立支持。

## 二、Supervised risk-surface 方法

| 方法 | 结果 | 解释 |
|---|---|---|
| Direct MLP | test RMSE 0.0282112；max obstacle violation 0.133762 | 可违反 payoff obstacle |
| Positive-premium MLP | test RMSE 0.0218009；obstacle violation 0 | 结构约束有效，但该实验中并非所有指标的最佳模型 |
| Boundary MLP | test RMSE 0.351305 | Boundary 仍是薄弱环节 |
| Price-autograd Delta | RMSE 0.097704 | 直接对 price net 求导不够稳 |
| Bounded Delta head | RMSE 0.0332401；bound violation 0 | 独立结构化 Greek head 更可靠 |
| Integrated workflow | 6 fresh regimes：price RMSE 0.0091874；boundary MAE 0.153897；Delta RMSE 0.0299804；mean speedup 87.5249 | 只支持 screening + PSOR audit，不是 strict solver replacement |

## 三、PINN 与 algorithm-unrolling

- Stage 21--23 是 **PINN-inspired supervised surrogates**：使用 PSOR labels 和 PSOR-derived residual masks，不是 standalone PINN solvers。
- Stage 23 的最佳配置改善了 price、masked residual、boundary 和 Delta，但仍是 0 diagnostic success；只能作为 capped-put preliminary result。
- M02 obstacle-aware PINN 比较四种 loss、五个 seeds；200 条 validation/legacy-test metric rows 均无法按冻结阈值得到 boundary，综合 gates 未通过，属于 negative result。
- M03 five-step unrolled PJOR + PSOR cleanup 有 11.847% timing improvement，但冻结的 price/Greek/boundary/complementarity non-regression gates 失败，不能提升为新 solver。

## 四、Free-boundary、quantum 与 low-rank campaign

| 方法 | 准确状态 |
|---|---|
| M04 Free-boundary Representation | 实现和测试通过；真实 dataset 未冻结，无 smoke/canonical evidence |
| M05 Schrödingerization | 两个 bounded numerical diagnostics 通过；只是 classical proxy，不是 quantum implementation 或完整 solver |
| M06 Quantum Linear Systems | VQLS 只在 N=8 通过，N=16/32 失败；无 QPU、finite-shot 或 speedup |
| M07 Nonlinear Quantum Reduction | LP 复现 value，但 dimension-growth ratio 96/98；没有实现降维 |
| M08 Tensor/Low Rank | 可以压缩存储，但没有配置同时维持 price、boundary、Greeks 和 LCP structure |
| M09 Multi-fidelity | 只有实现和问题修正后的 pre-canonical protocol；无正式结果 |

## 五、M10--M24

- M10--M21 是 structured surrogate、inverse reconstruction、diffusion、sampling、hardware、UQ、fallback、segmentation、transfer 和 XAI 的实验计划，尚未运行。
- M22--M24 已有 mechanistic audit、agentic search 和 formal verification 的 primary-source research/design，但没有 numerical results。

不能把这些 method packets、audit documents 或 implementations 写成“已经测试过并得到结果”。

## 六、Zihan 的整体状态

- **Scoped positive result：** M01 CN+Policy Iteration。
- **Formal negative results：** M02、M03、M06、M07、M08。
- **Diagnostic only：** M05；Stage 21--23 也只能作 scoped/preliminary evidence。
- **Implementation but no accepted experiment：** M04、M09。
- **Planned/research design only：** M10--M24。

Zihan 的总体贡献不只是 neural surrogates，而是：建立 CN/PSOR 数据和审计基础，独立验证 Policy Iteration 的速度价值，并用严格 gates 系统记录 neural、unrolling、quantum 和 low-rank 方法为何没有被错误提升为成功方法。

## 对海报的影响

1. M01 可作为 Policy-over-PSOR 的 scoped corroboration，但主 benchmark 数值仍建议使用 Ruixi 的 67-regime same-machine held-out comparison。
2. M02/M03/M06--M08 最适合放在 QR-linked negative-result/status matrix，不适合挤占主结果区域。
3. M05 必须明确写成 classical diagnostic，避免 quantum overclaim。
4. M04、M09--M24 只能列为 implementation/future work，不能进入 result table。
