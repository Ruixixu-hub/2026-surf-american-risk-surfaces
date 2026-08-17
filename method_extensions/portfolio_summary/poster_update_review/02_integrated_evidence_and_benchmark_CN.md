# 综合方法分类、可靠证据与建议 benchmark hierarchy

## 建议更新后的分类

原先的 “CN+PSOR / CN+Policy / CN+Projected LU / high-accuracy reference” 基本正确，但最好加一层 analytic controls，并把 learning methods 完全分开：

### A. Analytic and theorem controls

- European Black--Scholes analytic price：检查 PDE、边界和离散误差。
- No-dividend American call = European call：检查是否出现虚假提前行权。

这些是 validation controls，不是 American early-exercise solver benchmark。

### B. Same-discretization American LCP solver ladder

三者使用同一个 (121\times121) CN-LCP，区别只在每个时间层如何求 LCP：

1. `CN + PSOR` — Basic / Original Classical Benchmark。
2. `CN + Policy Iteration` — strengthened benchmark 1。
3. `CN + Projected LU` — strengthened benchmark 2，在冻结 SURF 域内数值认证。

这样分类可以把“solver improvement”与“换了更精细离散后的 accuracy improvement”分开。

### C. High-accuracy numerical reference

- `L-stable DIRK + Policy Iteration + strike-concentrated sinh grid`。
- Rannacher CN 和 Lobatto IIIC 是 audit comparators，不建议全部写成最终 reference。
- Gamma 只能在已经验证稳定收敛的 mask 上报告。

### D. Learned/reduced risk-surface methods

- Supervised: direct/positive-premium price surrogate、boundary head、bounded Delta head。
- Reduced/order: POD、RB-VI、boundary-aligned/localized basis、basis operator。
- Physics-informed: Member 3 的 PINN-inspired supervised regularization；Member 1 待运行的 Arm C/D/E standalone PINN。
- Operator learning: Positive-Premium DeepONet，尚无正式结果。

这些方法的任务可能是近似 screening、many-query acceleration、严格 hybrid initialization 或高维扩展，不能默认与 strict solver 完全等价。

## 当前最可靠、可放入海报的数值

### 1. Strict same-CN-LCP comparison（Member 1）

范围：67 个 held-out test/stress regimes；同一台 Mac、single CPU thread、float64；每个 arm 5 warm-ups + 30 repeats。

| Method | Pooled median (s) | P95 (s) | Correctness / interpretation |
|---|---:|---:|---|
| CN + PSOR | 0.309707 | 1.43094 | Basic / Original Classical Benchmark；67/67 common gates |
| CN + Policy Iteration | 0.0159997 | 0.0224767 | Strengthened benchmark 1；67/67 common gates |
| CN + Projected LU | 0.0102586 | 0.0144603 | Strengthened benchmark 2；67/67 common and strong-match gates |
| CN + Penalty/Newton | 0.0207950 | 0.0622017 | Candidate comparator；仅 40/67 common gates，`FAILED_CORRECTNESS` |

附加认证：

- 三个 benchmarks 均通过 67/67 common residual gates。
- Policy / PSOR paired median runtime ratio = 0.0556801。
- Projected-LU / Policy paired median ratio = 0.677606，即约快 (1.48\times)。
- Early-exercise subgroup 的 LU/Policy ratio = 0.658196。
- Penalty/Newton 不仅正确性失败；overall Penalty/Policy ratio 为 1.00668，在 31 个 early-exercise regimes 中为 2.64030。
- Full-trajectory max difference versus Policy = 2.62013e-14。
- Maximum normalized LCP residual = 6.29798e-16。

海报表述应为：`numerically certified on the frozen SURF domain`。不要写成对所有 American-option 参数都由经典理论无条件保证。

### 2. High-accuracy reference audit（Member 1）

- 12 audit regimes。
- L-stable DIRK 的 median Gamma max error 6.9470e-7；CN 为 1.657e-6。
- DIRK median boundary max error 6.3820e-6；CN 为 6.4596e-5。
- Spatial audit 选择 strike-concentrated sinh grid；stable second-order fraction 0.9167。
- 结论只允许：`Gamma can be evaluated on a validated stable mask`，不是所有区域的 Gamma 都已解决。

### 3. Structure-aware supervised learning（Member 3，需独立标注协议）

- Direct MLP 的 max obstacle violation 为 0.133762；positive-premium MLP 为 0。
- Bounded Delta head RMSE 0.0332401，优于 price-autograd Delta 的 0.097704，并保持 bound violation 0。
- 六个 fresh regimes 的 integrated workflow：mean price RMSE 0.0091874、boundary MAE 0.153897、strict-mask Delta RMSE 0.0299804、mean speedup 87.5249。

这些结果说明结构约束和分任务 head 有价值，但 boundary 仍然较弱；该 workflow 是 screening + PSOR audit，而不是替代 strict LCP solver。

### 4. Scientific negative findings（可简短放入海报）

- MLP warm start 没有加速 Policy；median 反而慢 44.4%。
- 统一 Penalty/Newton candidate 只通过 40/67 common LCP gates，且 overall timing 不优于 Policy；准确结论是 `failed correctness and no speed advantage`。
- Price surface 可以低秩，但这不保证 exercise boundary、Greeks 或 complementarity 同样低秩。
- RB-VI、localized basis 和 basis operator 都能得到较好的 price error，却因 boundary/Greek/LCP structure gate 未通过而停止。
- Member 3 的 M03 unrolled PJOR 有 11.85% speed signal，但未通过冻结的 price/Greek/boundary/complementarity non-regression gates。
- Member 3 的 M06--M08 分别表明：tested VQLS 没有通过规模扩展 gate、LP formulation 没有降维、low-rank compression 没有同时保持完整风险面结构。
- 这支持一个重要结论：American risk-surface 方法不能只比较 price RMSE。

### 5. Independent-campaign corroboration（Member 3，scoped）

- M01 在 9 个预注册 American-put screen regimes 上支持 CN+Policy：median 0.00731276 s vs CN+PSOR 0.140613 s，改善 94.799%，max natural residual 6.06898e-11。
- 该结果没有打开 blind holdout，且不是 Member 1 的同机/同 split protocol，因此只能作为“Policy 明显强化 PSOR”的独立 scoped corroboration，不能并入 67-regime Projected-LU runtime table。
- M02 obstacle-aware PINN 因 boundary diagnostic unavailable 且综合 gates 未通过而关闭；它不是正式 PINN 胜出结果。
- M05 是 classical proxy diagnostic，绝不能写成 quantum success。

## 不应当作为已完成结论的内容

- Member 1 PINN Arm C/D/E：尚无正式结果。
- Positive-Premium DeepONet：尚无正式结果。
- Member 2 DIRK attempt：没有仓库内定量结果。
- RB-VI、basis operator：没有通过最终 gate，不能写成“超过 benchmark”。
- M04/M09 只有实现或 smoke，尚无 canonical result。
- M10--M24、FNO、multi-fidelity、UQ/OOD：只有计划或设计，尚未完成正式实验；M22--M24 的 primary-source audits 也不是 numerical results。
- 当前海报中的 PINN 数值若无法找到对应 checkpoint、seed、grid、split 和 raw result，不应继续强化其结论。

## 证据等级建议

| Level | 含义 | 当前例子 |
|---|---|---|
| A — Poster-ready formal | 范围、配置、held-out、正确性和限制均可追溯 | Member 1 Projected LU；same-CN solver table |
| B — Poster-ready with scope | 结果可靠，但只能在明确的子任务/协议下解释 | Member 1 Greek stable-mask；Member 3 M01 Policy、positive-premium/Delta/integrated workflow |
| C — Validation, diagnostic or negative | 有科学价值，但未通过最终 gate或不能作胜出声明 | RB-VI、basis operator、M02/M03/M05--M08、PINN-inspired Stage 23 |
| D — Pending | 只有代码、smoke、audit design 或计划，没有正式结果 | M04、M09--M24、PINN C/D/E、DeepONet |

## 综合主结论（plain language）

目前最强且最干净的结论不是“某个神经网络已经胜过传统方法”，而是：

1. 对当前一维 American-option CN-LCP，Policy Iteration 已大幅强化 PSOR，而 Projected LU 又在同一离散问题上进一步提速，同时保持严格 residual 和几乎机器精度的一致性。
2. 对风险面代理，保证 premium 非负很重要，但仅 price 准确还不够；exercise boundary 和 Greeks 是主要瓶颈。
3. PINN 与 DeepONet 仍是待验证路线，不能在海报中写成已有胜出结果。
