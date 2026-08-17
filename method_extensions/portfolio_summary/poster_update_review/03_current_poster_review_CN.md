# 当前海报内容总结与修改建议

## 当前海报在讲什么

当前题目为 **Scientific Machine Learning and Numerical Methods in Derivative Pricing**。海报的主线是：

1. 用 Black--Scholes PDE 建立 European 和 American option 问题。
2. European call 使用 explicit/CN/implicit finite differences；American put 写成 LCP，并用 CN+PSOR。
3. PINN 使用 PDE、initial/boundary conditions、obstacle 和 Fischer--Burmeister residual。
4. 结果部分展示：
   - European call 的 CN 空间二阶收敛；
   - American put 的 PSOR 价格和 free boundary；
   - 一组 PINN 与传统方法的误差比较。
5. 当前结论是：低维问题中 FDM 更准确、更稳，PINN 尤其在 American put 上仍需改进；未来关注高维。

当前海报中的主要数值包括：

- European call CN：(M=100,200,400\) 时的 (L_\infty\) error 分别为 (6.92\times10^{-3},1.81\times10^{-3},4.41\times10^{-4})，observed order 约 1.93 和 2.04。
- American put CN+PSOR：相对 binomial 的 relative (L^2=1.82\times10^{-5})。
- PINN：European call relative (L^2=1.11\times10^{-3})，American put relative (L^2=3.73\times10^{-2})。

## 当前海报的优点

- LCP/obstacle/free-boundary 的数学问题表达清楚。
- European verification、American solver 和 PINN 尝试构成了容易理解的研究发展线。
- 色彩和模块风格统一，标题和主要 section 层次明确。
- 已经包含 free-boundary 和 prediction comparison 图，视觉上不是纯文字报告。

## 必须优先修正的内容

### Priority 0 — 证据与术语

1. **重新定义 benchmark。** European BSM 应称为 analytic/control validation；American benchmark 应是同一 CN-LCP 上的 PSOR → Policy → Projected LU solver ladder。
2. **区分 strict solver 与 surrogate。** CN solvers 求到统一 LCP tolerance；supervised/PINN/DeepONet 通常是近似预测器，除非再接 Policy finish。
3. **区分 standalone PINN 与 PINN-inspired supervised model。** Member 3 的 Stage 21--23 使用 PSOR labels 和 PSOR-derived masks，不是独立 PINN solver。
4. **补齐当前海报数字的 provenance。** 现有 PINN relative errors 和 American-put binomial comparison，需要明确对应的 code、seed、grid、training budget、binomial resolution 和 raw output。如果无法追溯，应删除或降级为 preliminary illustration。
5. **不要写尚未完成的方法结果。** Member 1 PINN C/D/E 和 DeepONet 只能放在 future work，不能进入 result table。

## 最值得加入的新结果

### A. Strict solver result — 建议成为海报中心结果

用一张 runtime bar chart 和一个小 correctness table 展示：

| Solver | Median (s) | P95 (s) |
|---|---:|---:|
| CN+PSOR | 0.309707 | 1.43094 |
| CN+Policy | 0.0159997 | 0.0224767 |
| CN+Projected LU | 0.0102586 | 0.0144603 |
| CN+Penalty/Newton | 0.0207950 | 0.0622017 |

在旁边写：前三个 benchmarks 均通过 67/67 common residual gates；Projected LU 的 LU-vs-Policy max trajectory difference 为 (2.62\times10^{-14})，maximum normalized LCP residual 为 (6.30\times10^{-16})。Penalty/Newton 只通过 40/67，应以红色星号标为 failed candidate，而不是 benchmark。

### B. Accuracy reference — 简短但重要

展示 `L-stable DIRK + Policy + sinh grid` 用于 boundary/Greek reference，并明确 Gamma 只在 validated stable mask 上报告。

### C. Learning/reduced-method finding — 不要堆所有方法

建议保留两条信息：

- Positive-premium output 可以消除 obstacle violation；bounded Delta head 比 price-autograd Delta 更可靠。
- Low-rank price representation does not guarantee an accurate exercise boundary or Greeks。

RB-VI、localized basis、basis operator 的全部 ladder 不适合塞进主海报；可以只做一个 status matrix，详细结果通过 QR/repository 提供。

Zihan 的 independent campaign 也建议采用同一处理：M01 可以作为“Policy strengthens PSOR”的 scoped corroboration；M02/M03/M06--M08 可在 supplement status matrix 中作为 negative results；M05 只能标成 classical diagnostic；M04、M09--M24 不应进入 results panel。

### D. PINN result 的保守处理

- 如果当前海报的原始 PINN run 有完整 provenance，可保留为 `initial standalone PINN attempt`，并写清不是正式 Arm C/D/E。
- Member 3 Stage 23 若加入，只能标记为 `supervised, PSOR-masked, capped-put pilot`，并同时写出 `diagnostic success: 0/3`。
- 更简单、安全的方案是：本版海报只把 formal PINN/DeepONet 放入 future work，等五-seed/held-out 后再更新。

## 建议的新内容结构

1. **Research question and contribution**：American risk surfaces 需要 price、boundary、Delta/Gamma 和 LCP structure 同时可靠。
2. **Problem and controls**：一条 VI/LCP 公式；European BSM 和 no-dividend call 作为 controls。
3. **Benchmark hierarchy**：PSOR → Policy → Projected LU；另列 high-accuracy DIRK reference。
4. **Strict solver results**：最大、最醒目的 runtime/correctness panel。
5. **Structure-aware learning evidence**：positive premium、Delta head、low-rank negative finding。
6. **Limitations and next steps**：formal PINN C/D/E、DeepONet、高维，以及尚未 canonical-run 的 M-methods；不声称已有结果。
7. **Reproducibility**：commit、protocol、sample count 和 QR code。

## 视觉和排版建议

- 当前海报下半部分正文和图表偏小；减少长公式推导，保留一条核心 VI/LCP 公式即可。
- 把 European theta-scheme 细节压缩成 verification badge/table，把版面让给新的 American solver results。
- 使用一致颜色：PSOR 灰、Policy 蓝、Projected LU 金/绿、surrogates 紫；high-accuracy reference 用虚线或深色框，不与速度 arms 混在一起。
- Runtime 图使用 log scale，否则 PSOR 会压扁 Policy/LU 差异。
- 每张图直接写 experiment scope，例如 `67 held-out regimes; same Mac; 30 repeats`。
- 对普通误差只保留 2--3 个有效数字；机器精度 residual 可以使用 scientific notation。
- 将 references 扩展到 Policy Iteration、Projected LU/Brennan--Schwartz、American-option DIRK/VI 和 PINN/DeepONet 的原始论文。

## 建议暂时删除或缩小的内容

- 过长的 finite-difference stencil 展开。
- 小字号的逐点 American value table；可以由 runtime/correctness table 替代。
- 缺少 provenance 的 PINN comparison 数值或图。
- 所有尚无正式结果的方法列表；只放 2--3 项 future work。

## 修改前需要团队确认的四件事

1. 当前海报 PINN 数值的原始 artifact 在哪里，是否可重现？
2. American put 相对 binomial 的 (1.82\times10^{-5}) 使用了多大的 binomial reference？
3. Member 3 的 six-regime integrated workflow 是否希望进入主海报，还是只放 repository supplement？
4. M01 的 scoped Policy result 是否作为独立 corroboration 放入正文，M02--M24 是否只放 QR-linked status matrix？
5. 新海报的中心是“strict American solver improvements”还是“classical versus learning methods”？当前证据更支持前者。
