<#
.SYNOPSIS
    One-file Windows entry point for the SURF Arm C/D/E PINN study.

.DESCRIPTION
    Creates/checks the CUDA environment and runs the registered stages in order.
    Python modules remain separate so checkpoints, tests, label isolation, and
    one-time held-out scoring remain auditable. Use -Resume after interruption.

.EXAMPLE
    Set-ExecutionPolicy -Scope Process Bypass
    .\scripts\windows\run_all_pinn_experiments.ps1 -Stage Setup

.EXAMPLE
    .\scripts\windows\run_all_pinn_experiments.ps1 -Stage Validation -Resume

.EXAMPLE
    .\scripts\windows\run_all_pinn_experiments.ps1 -Stage All -Resume -AcknowledgeLongRun
#>

[CmdletBinding()]
param(
    [ValidateSet("All", "Setup", "Tiny", "Validation", "Heldout", "Reference", "Score", "ArmE", "Report")]
    [string]$Stage = "All",

    [ValidateSet("cu126", "cu128")]
    [string]$CudaWheel = "cu128",

    [string]$PythonLauncher = "py",

    [int]$ShardIndex = 0,

    [int]$ShardCount = 1,

    [switch]$Resume,

    [switch]$SkipPackageInstall,

    [switch]$AcknowledgeLongRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Required by PyTorch deterministic algorithms for CUDA/cuBLAS. Child Python
# processes inherit this value, so formal seeds are reproducible on one stack.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv-pinn"
$PinnPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$ResultsRoot = Join-Path $ProjectRoot "results\08_pinn_gap"
$LogDirectory = Join-Path $ResultsRoot "windows_master_logs"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDirectory "master_${Stage}_${Timestamp}.log"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Assert-File {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing $Description at: $Path"
    }
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Description
    )
    Write-Host "> $Description" -ForegroundColor Yellow
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE. Re-run with -Resume after fixing the error."
    }
}

function Invoke-PinnPython {
    param([string[]]$Arguments, [string]$Description)
    Assert-File $PinnPython "PINN virtual-environment Python"
    Invoke-Checked -Executable $PinnPython -Arguments $Arguments -Description $Description
}

function Get-ResumeArguments {
    if ($Resume) { return @("--resume") }
    return @()
}

function Test-NeedsSetup {
    return -not (Test-Path $PinnPython -PathType Leaf)
}

function Invoke-Setup {
    Write-Step "STAGE SETUP: Python 3.11, dependencies, CUDA and tests"
    if (Test-NeedsSetup) {
        $LauncherName = [System.IO.Path]::GetFileNameWithoutExtension($PythonLauncher)
        $LauncherArguments = @()
        if ($LauncherName -eq "py") { $LauncherArguments += "-3.11" }
        $LauncherArguments += @("-m", "venv", ".venv-pinn")
        Invoke-Checked -Executable $PythonLauncher -Arguments $LauncherArguments -Description "Create Python 3.11 virtual environment"
    }
    if (-not $SkipPackageInstall) {
        Invoke-PinnPython -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Description "Upgrade pip"
        Invoke-PinnPython -Arguments @("-m", "pip", "install", "-r", "requirements-pinn.txt") -Description "Install project dependencies"
        Invoke-PinnPython -Arguments @("-m", "pip", "install", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/$CudaWheel") -Description "Install PyTorch 2.7.1 $CudaWheel"
    }
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    Invoke-PinnPython -Arguments @("experiments/29_pinn_protocol_and_operator_audit.py") -Description "Freeze and audit the PINN protocol"
    Invoke-PinnPython -Arguments @("scripts/windows/pinn_preflight.py") -Description "Check NVIDIA CUDA and float64 second derivatives"
    Invoke-PinnPython -Arguments @("-m", "unittest", "discover", "-s", "tests", "-p", "test_pinn_cde.py", "-v") -Description "Run PINN C/D/E tests"
}

function Assert-ReadyEnvironment {
    if (Test-NeedsSetup) {
        throw "The PINN environment does not exist. Run this file with -Stage Setup first, or use -Stage All."
    }
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    Invoke-PinnPython -Arguments @("scripts/windows/pinn_preflight.py") -Description "Re-check CUDA before running experiments"
}

function Invoke-Tiny {
    Write-Step "STAGE TINY: short Windows CUDA smoke tests"
    Assert-ReadyEnvironment
    $ResumeArguments = Get-ResumeArguments
    Invoke-PinnPython -Arguments (@("experiments/30_arm_c_soft_lcp_pinn.py", "--mode", "tiny", "--architecture", "resnet_4x2x50", "--device", "cuda") + $ResumeArguments) -Description "Run Arm C tiny smoke"
    Invoke-PinnPython -Arguments (@("experiments/31_arm_d_etc_fb_pinn.py", "--mode", "tiny", "--variant", "etc_fb_adaptive", "--architecture", "resnet_4x2x50", "--device", "cuda") + $ResumeArguments) -Description "Run Arm D tiny smoke"
}

function Invoke-Validation {
    Write-Step "STAGE VALIDATION: architecture, ablations, five seeds, frozen gate"
    if (-not $AcknowledgeLongRun) {
        throw "Validation includes many GPU trainings. Re-run with -AcknowledgeLongRun after confirming the compute budget."
    }
    Assert-ReadyEnvironment
    $ResumeArguments = Get-ResumeArguments

    Write-Step "VALIDATION 1/6: Arm C architecture sensitivity (32 training jobs)"
    Invoke-PinnPython -Arguments (@("experiments/30_arm_c_soft_lcp_pinn.py", "--mode", "sensitivity", "--architecture", "all", "--device", "cuda") + $ResumeArguments) -Description "Arm C architecture sensitivity"
    Write-Step "VALIDATION 2/6: required references and architecture decision"
    Invoke-PinnPython -Arguments @("experiments/32_pinn_validation_gates.py", "--architecture-only", "--generate-reference", "--device", "cuda") -Description "Select Arm C architecture and generate validation reference"

    $ArchitectureDecisionPath = Join-Path $ResultsRoot "03_validation_gates\arm_c_architecture_decision.json"
    Assert-File $ArchitectureDecisionPath "Arm C architecture decision"
    $ArchitectureDecision = Get-Content $ArchitectureDecisionPath -Raw | ConvertFrom-Json
    $Architecture = [string]$ArchitectureDecision.selected_architecture
    if ([string]::IsNullOrWhiteSpace($Architecture)) {
        throw "Architecture selection did not return a selected architecture."
    }
    Write-Host "Selected architecture: $Architecture" -ForegroundColor Green

    Write-Step "VALIDATION 3/6: Arm D attribution ablations (56 training jobs)"
    Invoke-PinnPython -Arguments (@("experiments/31_arm_d_etc_fb_pinn.py", "--mode", "ablation", "--variant", "all", "--architecture", $Architecture, "--device", "cuda") + $ResumeArguments) -Description "Run Arm D attribution ablations"
    $ValidationReference = Join-Path $ResultsRoot "03_validation_gates\high_accuracy_reference"
    Invoke-PinnPython -Arguments @("experiments/32_pinn_validation_gates.py", "--ablation-only", "--device", "cuda", "--reference-dir", $ValidationReference) -Description "Apply Arm D ablation gates"

    $AblationDecisionPath = Join-Path $ResultsRoot "03_validation_gates\arm_d_ablation_decision.json"
    Assert-File $AblationDecisionPath "Arm D ablation decision"
    $AblationDecision = Get-Content $AblationDecisionPath -Raw | ConvertFrom-Json
    $Variant = "etc_fb_mixture"
    if ($AblationDecision.adaptive_sampling.status -eq "GO") { $Variant = "etc_fb_adaptive" }
    if ($AblationDecision.positive_premium.status -eq "GO") { $Variant = "positive_premium" }
    Write-Host "Selected Arm D variant: $Variant" -ForegroundColor Green

    Write-Step "VALIDATION 4/6: Arm C five-seed validation (95 training jobs)"
    Invoke-PinnPython -Arguments (@("experiments/30_arm_c_soft_lcp_pinn.py", "--mode", "validation", "--architecture", $Architecture, "--device", "cuda") + $ResumeArguments) -Description "Run Arm C five-seed validation"
    Write-Step "VALIDATION 5/6: selected Arm D five-seed validation (95 training jobs)"
    Invoke-PinnPython -Arguments (@("experiments/31_arm_d_etc_fb_pinn.py", "--mode", "validation", "--variant", $Variant, "--architecture", $Architecture, "--device", "cuda") + $ResumeArguments) -Description "Run Arm D five-seed validation"

    $ArmCDirectory = Join-Path $ResultsRoot "01_arm_c\validation\$Architecture"
    $ArmDDirectory = Join-Path $ResultsRoot "02_arm_d\validation\$Variant\$Architecture"
    Write-Step "VALIDATION 6/6: score C/D and freeze held-out configuration"
    Invoke-PinnPython -Arguments @("experiments/32_pinn_validation_gates.py", "--device", "cuda", "--arm-c-dir", $ArmCDirectory, "--arm-d-dir", $ArmDDirectory, "--reference-dir", $ValidationReference) -Description "Score validation and freeze the held-out configuration"

    $FrozenConfiguration = Join-Path $ResultsRoot "03_validation_gates\frozen_pinn_configuration.json"
    if (-not (Test-Path $FrozenConfiguration -PathType Leaf)) {
        throw "Arm D did not pass the registered validation gate. Held-out training is intentionally blocked. See results\08_pinn_gap\03_validation_gates."
    }
}

function Invoke-Heldout {
    Write-Step "STAGE HELDOUT: physics-only C/D training shard"
    if (-not $AcknowledgeLongRun) {
        throw "Held-out training can consume up to 670 GPU-hours. Re-run with -AcknowledgeLongRun after confirming the budget."
    }
    if ($ShardCount -lt 1 -or $ShardIndex -lt 0 -or $ShardIndex -ge $ShardCount) {
        throw "Require 0 <= ShardIndex < ShardCount."
    }
    Assert-ReadyEnvironment
    Assert-File (Join-Path $ResultsRoot "03_validation_gates\frozen_pinn_configuration.json") "frozen validation configuration"
    $Arguments = @(
        "experiments/33_pinn_heldout_training_and_scoring.py", "train",
        "--device", "cuda",
        "--shard-index", [string]$ShardIndex,
        "--shard-count", [string]$ShardCount
    )
    if ($Resume) { $Arguments += "--resume" }
    Invoke-PinnPython -Arguments $Arguments -Description "Run held-out shard $ShardIndex of $ShardCount"
}

function Invoke-Reference {
    Write-Step "STAGE REFERENCE: 67 high-accuracy DIRK+Policy surfaces"
    Assert-ReadyEnvironment
    Invoke-PinnPython -Arguments @("experiments/33_pinn_heldout_training_and_scoring.py", "reference", "--device", "cpu") -Description "Generate held-out high-accuracy references"
}

function Invoke-Score {
    Write-Step "STAGE SCORE: one-time held-out scoring"
    Assert-ReadyEnvironment
    Write-Warning "This opens held-out labels and creates a permanent scoring lock. It must run only after all 670 jobs have a terminal status."
    Invoke-PinnPython -Arguments @("experiments/33_pinn_heldout_training_and_scoring.py", "score", "--device", "cuda") -Description "Score A/B/C/D once"
}

function Invoke-ArmE {
    Write-Step "STAGE ARM E: D prediction to strict Policy Iteration"
    Assert-ReadyEnvironment
    $DecisionPath = Join-Path $ResultsRoot "04_heldout\arm_d_heldout_decision.json"
    Assert-File $DecisionPath "Arm D held-out decision"
    $Decision = Get-Content $DecisionPath -Raw | ConvertFrom-Json
    if ($Decision.status -ne "GO") {
        Write-Warning "Arm D status is $($Decision.status). Arm E is correctly skipped by the registered gate."
        return
    }
    Invoke-PinnPython -Arguments @("experiments/34_arm_e_hybrid_benchmark.py", "--device", "cuda", "--warmups", "5", "--repeats", "30") -Description "Benchmark Arm B versus Arm E"
}

function Invoke-Report {
    Write-Step "STAGE REPORT: synthesize GO/STOP/DEFER decisions"
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    Invoke-PinnPython -Arguments @("experiments/35_pinn_gap_synthesis.py") -Description "Write final PINN gap synthesis"
}

Assert-File (Join-Path $ProjectRoot "requirements-pinn.txt") "PINN requirements"
Assert-File (Join-Path $ProjectRoot "results\04_surrogate_dataset\v1_small_grid\dataset_v1_small_grid.npz") "frozen SURF dataset"

Start-Transcript -Path $LogPath | Out-Null
try {
    Write-Host "SURF PINN Windows master runner" -ForegroundColor Green
    Write-Host "Project: $ProjectRoot"
    Write-Host "Stage: $Stage"
    Write-Host "Log: $LogPath"

    switch ($Stage) {
        "Setup"      { Invoke-Setup }
        "Tiny"       { Invoke-Tiny }
        "Validation" { Invoke-Validation }
        "Heldout"    { Invoke-Heldout }
        "Reference"  { Invoke-Reference }
        "Score"      { Invoke-Score }
        "ArmE"       { Invoke-ArmE; Invoke-Report }
        "Report"     { Invoke-Report }
        "All" {
            if ((Test-NeedsSetup) -or (-not $SkipPackageInstall)) {
                Invoke-Setup
            } else {
                Assert-ReadyEnvironment
            }
            Invoke-Tiny
            Invoke-Validation
            Invoke-Heldout
            if ($ShardCount -eq 1) {
                Invoke-Reference
                Invoke-Score
                Invoke-ArmE
                Invoke-Report
            } else {
                Write-Warning "Only shard $ShardIndex of $ShardCount was run. Merge all shard outputs into the same project results folder, then run -Stage Reference and -Stage Score."
            }
        }
    }
    Write-Step "REQUESTED STAGE FINISHED SUCCESSFULLY"
    Write-Host "Results: $ResultsRoot" -ForegroundColor Green
    Write-Host "Log: $LogPath" -ForegroundColor Green
}
finally {
    Stop-Transcript | Out-Null
}
