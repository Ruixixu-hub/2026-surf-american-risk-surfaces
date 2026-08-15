# Windows CUDA execution for SURF PINN Arms C/D/E

Formal neural training is intentionally separated from Mac CPU development. Use Python 3.11 and an NVIDIA GPU whose driver supports the selected PyTorch CUDA wheel.

## Recommended one-file entry point

The simplest manual workflow uses only:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\run_all_pinn_experiments.ps1 -Stage Setup
.\scripts\windows\run_all_pinn_experiments.ps1 -Stage Tiny -Resume
.\scripts\windows\run_all_pinn_experiments.ps1 -Stage Validation -Resume -AcknowledgeLongRun
```

Continue with `-Stage Heldout`, `Reference`, `Score`, and `ArmE` as described in
[`WINDOWS_PINN_运行说明_CN.txt`](WINDOWS_PINN_运行说明_CN.txt).
The master runner logs every stage under `results/08_pinn_gap/windows_master_logs/`.
The scripts below remain useful as smaller recovery entry points.

## 1. Set up and verify

From PowerShell at the repository root:

```powershell
.\scripts\windows\setup_pinn_cuda.ps1
```

The script installs the pinned CPU-side packages and PyTorch 2.7.1 CUDA 12.8, freezes Experiment 29, checks CUDA float64 second derivatives, and runs the PINN unit tests. Use `-CudaWheel cu126` only when the installed driver cannot support CUDA 12.8. Formal jobs must not start unless preflight passes.

## 2. Validation before held-out data

```powershell
.\scripts\windows\run_pinn_development_and_validation.ps1 -Resume
```

This command enforces the registered order: Arm C architecture sensitivity, architecture gate,
Arm D ablations, ablation gates, five-seed C/D validation, and the final frozen configuration.
It also generates the 19-regime DIRK+Policy M=480/N=960 validation reference. The held-out
script refuses to start until `frozen_pinn_configuration.json` exists. The smaller
`run_pinn_validation.ps1` remains available only for manually resuming one already-selected arm.

## 3. Frozen held-out training

Split 670 jobs across independent workers. For eight shards, each worker uses a different index from 0 through 7:

```powershell
.\scripts\windows\run_pinn_heldout_shard.ps1 -ShardIndex 0 -ShardCount 8 -Resume
```

Job identity is the sorted tuple `(arm, split, regime_id, seed)`. A checkpoint refuses to resume under a different configuration hash. The worst-case registered budget is 670 GPU-hours; actual usage may be lower through convergence or explicit budget exhaustion.
Each job writes an atomic heartbeat, history, checkpoint, status JSON, and label-free prediction
surface. Before scoring, the aggregator rejects duplicates, missing jobs, missing hashes,
mixed architectures, and missing prediction files.

## 4. One-time scoring and Arm E

Do not score until all 670 jobs have a terminal status. Scoring creates a permanent marker prohibiting held-out-driven retraining.

```powershell
.\scripts\windows\score_pinn_heldout_once.ps1
.\scripts\windows\run_arm_e.ps1
```

Arm E runs only when Arm D passes its held-out accuracy and structure gate. Reports keep training time, steady-state online time, first-query cost, failed seeds, and break-even query count separate.
Formal scoring also recomputes Arm A (CN+PSOR) and Arm B (CN+Policy Iteration) on the same
Windows machine and scores A/B/C/D against the same high-accuracy reference.
