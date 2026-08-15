# SURF Positive-Premium DeepONet：Windows GPU 运行指南

## 1. 运行前准备

把完整项目复制到 Windows 电脑，保留 `results/09_reduced_basis_vi/`、
`results/11_positive_premium_basis_operator/`、`data/` 和全部源码。项目路径尽量不要包含中文或特殊符号。

需要：

- NVIDIA GPU 和较新的官方驱动；
- Python 3.11；
- PowerShell；
- 足够磁盘空间保存 18 个 development 模型、10 个 formal 模型和预测；
- 正式训练期间关闭休眠。

在项目根目录打开 PowerShell。若系统禁止本地脚本，可只对当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 2. 环境和 CUDA 审计

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Setup
```

该命令会创建 `.venv-deeponet`、安装项目依赖及 PyTorch 2.7.1 CUDA 12.8，随后检查：

- `torch.cuda.is_available()`；
- GPU、驱动、CUDA 和 Python 版本；
- float64 DeepONet 的完整 Cartesian contraction；
- 训练快照、validation cache、hash 和专项测试。

如果显卡驱动只适合 CUDA 12.6 wheel，改用：

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Setup -CudaWheel cu126
```

只有 `results/12_positive_premium_deeponet/00_protocol/windows_hardware_manifest.json`
显示 `PASS` 才继续。

## 3. 独立 tiny smoke

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Tiny -SkipPackageInstall
```

它只训练 2 steps，结果写入 `00_protocol/tiny_smoke/`，不会进入 development 表或方法结论。

## 4. 正式 development ladder

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Development -SkipPackageInstall -AcknowledgeLongRun
```

它运行 put/call × N0/N1/N2 × rank 32/64/128，共 18 个 seed-17 配置，每个 6,000 steps。
中断后用相同命令追加 `-Resume`。`-AcknowledgeLongRun` 用来确认这不是 smoke；Resume 会严格检查配置、数据和 protocol hash，不匹配会拒绝继续。

完成后检查：

- `01_development/development_decision.json`；
- 每个任务的 `checkpoint.pt`、`training_history.csv` 和 `validation_metrics.csv`。

## 5. 五-seed validation

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Validation -SkipPackageInstall
```

它只运行 development 选定的 family-specific arm/rank，seeds 为 17、29、43、71、101，每个 12,000 steps，
单 seed 最多一 GPU-hour。中断可用 `-Resume`；`FAILED` 或 `BUDGET_EXHAUSTED` seed 不允许私自延长或换 seed。

随后查看 `02_five_seed_validation/validation_decision.json`：

- `PROCEED_HELDOUT`：该 family 才能打开 held-out；
- `PARTIAL_PROCEED_HELDOUT`：只有一个 family 可继续；
- `STOP_BEFORE_HELDOUT`：test/stress 保持封存，不能继续评分。

## 6. 一次性 held-out

仅在 validation 允许时执行：

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Heldout -SkipPackageInstall
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Score -SkipPackageInstall
```

第一条只生成预测，不读取 held-out reference；第二条才生成 CN+Policy/DIRK reference 并评分，最后创建永久 scoring marker。
创建 marker 后，代码会禁止重新训练、覆盖预测或重复评分。不要手工删除 marker 来重新调参。

若需要记录 q=0 call 的神经网络 OOD 外推，只能在主评分完成后运行：

```powershell
.\.venv-deeponet\Scripts\python.exe experiments\55_deeponet_heldout_prediction_and_scoring.py q0-ood --device cuda
```

主 q=0 call 结果始终采用 European BSM 解析分支，OOD 不进入 GO/STOP。

## 7. 同机速度、hybrid 和报告

```powershell
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Runtime -SkipPackageInstall
.\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Report -SkipPackageInstall
```

Runtime 会分别生成：

- `runtime_samples_cuda.csv`：GPU DeepONet 与同机单线程 CN+PSOR/CN+Policy；
- `runtime_samples_cpu.csv`：DeepONet CPU-only 算法对照；
- DeepONet→Policy exact hybrid 的收敛、迭代数和最终解差异。

最终查看：

- `results/12_positive_premium_deeponet/06_synthesis/method_decision.json`；
- `reports/14_positive_premium_deeponet/positive_premium_deeponet_结论_CN.md`。

## 8. 重要纪律

- 不要在 test/stress 结果出来后调整 rank、loss、seed、steps 或 gate。
- 不要把 tiny smoke 当正式性能结果。
- 不要把 GPU 对 CPU 的速度优势全部写成算法优势；必须同时报告 CPU-only DeepONet。
- `GO_APPROXIMATE_DEEPONET` 代表合格近似代理，不代表严格等于 LCP 解。
- 只有 exact hybrid 达到相同 `1e-12` LCP 容差后，才能声明严格最终解相同。
