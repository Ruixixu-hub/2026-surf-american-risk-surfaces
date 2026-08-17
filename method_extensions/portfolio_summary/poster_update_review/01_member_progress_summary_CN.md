# 三位成员的方法扩展进展与证据摘要

## 审阅原则

本摘要只把仓库中已经记录的方法、配置、结果和限制当作证据。仓库中的计划、prompt、TODO 或实验说明不自动视为已完成结果。不同成员使用的样本划分、网格、标签和计时协议不同，因此不能把数值直接拼成一个统一排行榜。

## Member 1 — Ruixi Xu

来源：[method_extensions](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/tree/main/method_extensions)、[方法状态](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/blob/main/method_extensions/METHOD_STATUS.md)、[结果摘要](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/blob/main/method_extensions/portfolio_summary/results_summary.md)。

Member 1 的工作形成了目前最完整的“同一 CN 离散问题上的 LCP solver ladder”，并系统测试了 reduced-order、basis/operator 和尚待正式运行的 PINN/DeepONet 路线。

### 已有正式或可明确判定的结果

| 方法 | 功能 | 范围与主要结果 | 当前状态 |
|---|---|---|---|
| CN + PSOR | Basic / Original Classical Benchmark | 2026-08-17 poster unified rerun：67/67 common residual gates；median 0.309707 s，p95 1.43094 s | `BASIC_ORIGINAL_CLASSICAL_BENCHMARK` |
| CN + Policy Iteration | 用 active-set/policy 更新求同一 CN-LCP | 67/67 common residual gates；median 0.0159997 s，p95 0.0224767 s；paired Policy/PSOR ratio 0.0556801 | `STRENGTHENED_BENCHMARK_1` |
| CN + option-directed Projected LU | 一次定向投影三对角 sweep 直接求同一 CN-LCP | 67/67 common residual 和 strong Policy-match gates；median 0.0102586 s，p95 0.0144603 s；paired LU/Policy ratio 0.677606；与 Policy trajectory 最大差 2.62013e-14；最大 normalized LCP residual 6.29798e-16 | `STRENGTHENED_BENCHMARK_2_NUMERICALLY_CERTIFIED` |
| CN + Penalty/Newton | 有限 penalty 的 semismooth-Newton candidate | Validation 冻结 penalty 1e8；held-out 只通过 40/67 common gates；median 0.0207950 s，p95 0.0622017 s；overall Penalty/Policy ratio 1.00668，early-exercise subgroup 2.64030 | `FAILED_CORRECTNESS`，不是 benchmark |
| L-stable DIRK + Policy + sinh grid | 更稳定的高精度 value/Greek reference | 12 个 audit regimes；stable second-order fraction 0.9167，boundary within one local cell fraction 1.0；Gamma 只在 stable mask 上解除 blocked | `HIGH_ACCURACY_REFERENCE` |
| Positive-premium MLP warm start | 用网络给 PSOR/Policy 初值，再严格收敛 | MLP→Policy 比 conventional Policy median 慢 44.4%，p95 也未保持；最终严格解正确但无加速价值 | `STOP_LEARNED_ACCELERATION` |
| POD/SVD representation | 检查 continuation-premium surface 是否低秩 | 8 个 unaligned modes 的 worst held-out RMSE 0.00115372，低于当时 label floor 0.001979957 | `GO_REPRESENTATION` |
| Polynomial POD coefficient map | 从参数预测 POD coefficients | worst held-out RMSE 0.00680275，高于 acceptance 0.002474946 | `STOP_MAPPING` |
| Primal/dual RB-VI | 降阶求解 variational inequality | Put 的 price reduction 可低至 6.91e-5 RMSE，但 boundary/Delta/Gamma 失败；call 更差；只到 validation | `STOP_ACCURACY` |
| Boundary-aligned/localized bases | 诊断 moving boundary 是否导致 RB 失败 | 对齐插值 pairing gate 失败；physical localization 可改善 price，但 boundary/F1 未达门槛；未打开 held-out | `STOP_RB_ROUTE` |
| Positive-premium POD Basis Operator | MLP 学参数到 POD coefficients，并保证 premium 非负 | 12-mode、5 seeds 的 price gate 通过：put worst RMSE 3.40743e-4，call 4.47061e-4；但 boundary/Greek/structure gate 严重失败，held-out 封存 | `STOP_STRUCTURE` |

Projected LU 的结论需要保留一个重要限定：4 个 held-out 的低波动 `q=0` calls 位于经典 M-matrix 充分条件之外。因此它是在冻结的 SURF 参数域中由 residual 和 Policy 对照严格数值认证，而不是对任意参数无条件的定理保证。

### 尚无正式结果

| 方法 | 已实现/计划内容 | 证据状态 |
|---|---|---|
| PINN Arm C | Soft-LCP vanilla PINN | 有代码和 protocol，尚无正式五-seed/held-out 结果 |
| PINN Arm D | Exact-terminal lift + Fischer--Burmeister + curriculum/adaptive sampling | 有代码和 protocol，尚无正式结果 |
| PINN Arm E | Arm D prediction → Policy Iteration strict hybrid | 依赖 Arm D 正式通过，尚无正式结果 |
| Positive-Premium DeepONet N0/N1/N2/H | Parameter-conditioned branch/trunk operator；可加入结构监督和 VI regularization | 有代码和 protocol，尚无正式 CUDA 结果 |

## Member 2 — Xiaoya Wu

来源仅限用户指定的 [research_automation](https://github.com/yaya0526/SURF2026/tree/main/research_automation)。

该文件夹的主要贡献是实验管理和可复现记录框架，而不是一套已经带完整数值表的求解器结果包。

| 方法/工作 | 仓库中能确认的内容 | 当前证据状态 |
|---|---|---|
| CN + PSOR | 被定义为 classical baseline | 方法定位明确；该文件夹没有独立数值表 |
| CN + Policy Iteration | 计划比较同一 LCP 下的替代求解器，并有“观察到更快”的定性记录 | exact runtime、iterations、grid 和 error 明确标记为需从本地输出补入；不应引用具体数值 |
| DIRK attempt | 作为更稳定时间离散方向进行尝试 | 尚缺 refinement、runtime 和 error 结果，不能宣称优于 CN |
| Research automation | experiment template、registry、log、poster summary 流程 | 适合作为团队统一实验记录规范；registry 仍为空，best-method 字段为 null |

因此，Member 2 当前最可靠的 poster contribution 是：帮助团队建立 benchmark hierarchy、实验注册和“结论必须绑定证据”的记录方式。仓库本身不支持把她的方法写成独立定量胜出结果。

## Member 3 — Zihan Liang

来源：[default branch](https://github.com/zihan-liang/american-option-risk-surfaces)、[verified metric ledger](https://github.com/zihan-liang/american-option-risk-surfaces/blob/main/docs/verified_metric_ledger.md)、[PINN report](https://github.com/zihan-liang/american-option-risk-surfaces/blob/main/reports/pinn_experiment_report.tex)，以及独立的 [`campaign/independent-methods`](https://github.com/zihan-liang/american-option-risk-surfaces/tree/campaign/independent-methods) 分支（本次核对 commit `198b9bef9c7f7de22997ba5ea7b986c2a0e50c47`）。

Member 3 的工作包括两部分：默认分支中的可复现 CN/PSOR 数据、supervised risk-surface 与 PINN-inspired 研究；以及独立分支中的 M01--M24 broad method campaign。后者不能被省略，但必须按 positive、negative、diagnostic、implementation-only 和 planned 严格区分。

### Independent campaign：M01--M09

| 编号 | 方法 | 已核实结果 | 证据状态 |
|---|---|---|---|
| M01 | Classical Solvers | 9 个预注册 American-put screen regimes；CN+Policy median 0.00731276 s，对比 CN+PSOR 0.140613 s，相对改善 94.799%；max natural residual 6.06898e-11。Penalty/Newton residual 5.33246e-8，未通过 | Policy 为 scoped `candidate_supported`；无 blind holdout，不能与 Member 1 的 67-regime timing 拼成公平排名 |
| M02 | Obstacle-aware PINN | supervised、masked PDE、separate complementarity、FB，5 seeds；Softplus premium 始终严格为正，200 条 validation/legacy-test metrics 都无法按冻结阈值提取 boundary，其他综合 gate 也未全部通过 | `insufficient_diagnostics` / `diagnostic_only` negative result |
| M03 | Algorithm Unrolling | 5-step unrolled PJOR + ordinary PSOR cleanup：0.1059998 s vs tuned PSOR 0.1202459 s，快 11.847%；但 price、Delta、Gamma、boundary、continuation 和 complementarity non-regression gates 失败 | `failed_metric_gate`；只有 speed signal，不是 accuracy-preserving improvement |
| M04 | Free-boundary Representation | premium boundary extraction、scalar boundary head、implicit level set；29 项 focused tests 和 71 项 campaign tests 通过 | implementation accepted；真实 dataset 未冻结，无 smoke/canonical result |
| M05 | Schrödingerization | frozen-continuation RMSE 9.204e-13；one obstacle update RMSE 1.667e-12；均低于 classical floor | `diagnostic_only`；只是 classical dissipative-recovery proxy，不是 quantum circuit、完整 solver 或 speedup |
| M06 | Quantum Linear Systems | exact-statevector VQLS 仅 N=8 通过；N=16、32 失败；estimated shot executions 快速增长 | `failed_metric_gate` / `diagnostic_only`；无 QPU、finite-shot 或 end-to-end speedup |
| M07 | Nonlinear Quantum Reduction | 两种 LP 都复现 stopping value，但 dimension-growth ratio 最大分别为 96 和 98 | `failed_metric_gate`；LP 扩维而非降维，无 quantum solver |
| M08 | Tensor and Low Rank | TT-SVD、TT completion、POD、primal-dual RB 均能压缩存储，但没有配置同时通过 price、boundary、Greeks、obstacle/complementarity 和 residual gates | `failed_metric_gate`，有价值的 negative result |
| M09 | Multi-fidelity | 有实现和非证据 smoke；smoke 暴露 seed 初始化和 test-label 访问顺序问题，随后修正 | 尚无 canonical run；不能报告正式效果 |

### Independent campaign：M10--M24

这些方法尚无 canonical numerical result：

| 范围 | 方法 | 当前状态 |
|---|---|---|
| M10--M14 | Structured Surrogate、Inverse Regularization、Diffusion/Score、Adaptive Sampling、Hardware Acceleration | method packets / planned experiments，未运行 |
| M15--M18 | MC Dropout、Deep Ensembles、Conformal Prediction、Selective Fallback | uncertainty/fallback plans，未运行 |
| M19--M21 | U-Net/V-Net Boundary Segmentation、Transfer Learning、Explainable AI | planned，未运行 |
| M22--M24 | Mechanistic Audit、Agentic Search、Formal Verification | 已有 primary-source audit/design documents；仍无实验结果 |

因此，M01 是该 campaign 唯一获得 scoped positive promotion 的方法；M02、M03、M06--M08 是正式 negative evidence；M05 只构成有限 diagnostic；M04/M09 尚无正式实验结果；M10--M24 不能写成已完成。

### Classical solver/data foundation

| 项目 | 结果 | 解释 |
|---|---:|---|
| European put smoke-grid max error | 2.782e-4 | 用于数值实现检查 |
| No-dividend American call max error | 2.782e-4 | 验证无提前行权 control |
| American put obstacle violation | 0 | 离散 obstacle 满足 |
| American put complementarity residual | 4.009e-12 | LCP 实现检查 |
| Application regimes | 288 | 144 put、108 dividend call、36 no-dividend call controls |
| Clean supervised dataset | 223 regimes / 3,264,943 samples | 65 个 diagnostic-failure regimes 被排除；split 为 156/33/34 |

### Supervised risk-surface methods

| 方法 | 结果 | 可靠解释 |
|---|---|---|
| Direct MLP price surrogate | validation RMSE 0.0280883；test 0.0282112；max obstacle violation 0.133762 | 直接 price MLP 可违反 payoff obstacle，属于 negative evidence |
| Positive-premium MLP | validation RMSE 0.0199883；test 0.0218009；obstacle violation 0 | 结构约束消除 obstacle violation；该次实验中 positive-premium linear 的 RMSE 反而更低，因此不能宣称 MLP 是总体最优 |
| Boundary MLP | test RMSE 0.351305；MAE 0.208745 | boundary prediction 是最弱环节 |
| Price-autograd Delta | RMSE 0.097704 | 单靠 price 网络导数不够可靠 |
| Bounded Delta head | test RMSE 0.0332401；MAE 0.014037；bound violation 0 | 独立、带界的 Delta head 明显更稳 |
| Integrated workflow | 6 个 fresh deterministic held-out regimes：mean price RMSE 0.0091874；boundary MAE 0.153897；strict-mask Delta RMSE 0.0299804；mean speedup 87.5249；obstacle/Delta-bound violations 0 | 只支持“快速 screening + PSOR audit”；不是严格 solver replacement，也不能与 Member 1 的同机 solver timing 直接对比 |

### PINN-inspired supervised experiments

这些阶段仍使用 PSOR labels，PDE residual mask 也由 PSOR 解确定；因此应称为 `PINN-inspired supervised surrogate`，不能称为独立求解 American VI 的 PINN。

| 阶段 | 主要结果 | 状态 |
|---|---|---|
| Stage 21 | 单个 capped put run；selected price RMSE 0.0928313，未优于 supervised 0.0927621；residual 也未改善 | negative result |
| Stage 22 | terminal scaling 改善 price/boundary/Delta，但 residual 变差；0/3 success | 未通过 |
| Stage 23 | 最佳 `terminal_pde_weight_5`：price RMSE 0.0636749→0.0222344，masked residual 0.0551382→0.0280643，boundary RMSE 0.0597350→0.0117206，strict-Delta RMSE 0.102513→0.0970616 | 3/3 candidate success，但 0 diagnostic success；只可作为 capped put、PSOR-masked、preliminary result |

## 三位成员的互补关系

- Member 1：严格 solver ladder、high-accuracy reference，以及对多条 reduced/operator 路线的可证伪实验。
- Member 2：实验治理、benchmark 组织和海报更新记录框架，目前没有可独立引用的定量结果。
- Member 3：CN/PSOR 数据、supervised risk-surface workflow 和 M01--M24 independent campaign；既提供 scoped Policy 支持，也系统保留 PINN、unrolling、quantum 和 low-rank negative evidence。

三人的结果可以在同一海报中形成一条共同故事线，但不能把不同协议的 runtime/RMSE 放在同一轴上声称公平排名。
