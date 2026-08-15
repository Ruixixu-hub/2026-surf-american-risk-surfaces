<#
.SYNOPSIS
  Windows CUDA entry point for SURF Positive-Premium DeepONet Experiments 52--57.

.EXAMPLE
  .\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Setup
  .\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Development -Resume
  .\scripts\windows\run_all_deeponet_experiments.ps1 -Stage Validation -Resume
#>

[CmdletBinding()]
param(
    [ValidateSet("All", "Setup", "Tiny", "Development", "Validation", "Heldout", "Score", "Runtime", "Report")]
    [string]$Stage = "All",
    [ValidateSet("cu126", "cu128")]
    [string]$CudaWheel = "cu128",
    [string]$PythonLauncher = "py",
    [switch]$Resume,
    [switch]$SkipPackageInstall,
    [switch]$AcknowledgeLongRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv-deeponet"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

function Invoke-Checked {
    param([string[]]$Arguments, [string]$Description)
    Write-Host "> $Description" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

function Ensure-Environment {
    if (-not (Test-Path $Python)) {
        & $PythonLauncher -3.11 -m venv $VirtualEnvironment
    }
    if (-not $SkipPackageInstall) {
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r requirements-pinn.txt
        & $Python -m pip install torch==2.7.1 --index-url "https://download.pytorch.org/whl/$CudaWheel"
    }
    Invoke-Checked @("scripts/windows/deeponet_preflight.py") "CUDA DeepONet preflight"
    Invoke-Checked @("experiments/52_deeponet_protocol_and_data_audit.py") "Protocol/data audit"
    Invoke-Checked @("-m", "unittest", "discover", "-s", "tests", "-p", "test_positive_premium_deeponet.py", "-v") "DeepONet unit tests"
}

$ResumeArguments = @()
if ($Resume) { $ResumeArguments = @("--resume") }

if ($Stage -in @("All", "Setup")) { Ensure-Environment }
if ($Stage -in @("All", "Tiny")) {
    Invoke-Checked @("experiments/53_deeponet_development.py", "--tiny-smoke", "--family", "put", "--arm", "N0", "--rank", "32", "--steps", "2", "--device", "cuda") "Tiny CUDA smoke"
}
if ($Stage -in @("All", "Development")) {
    if (-not $AcknowledgeLongRun) { throw "Use -AcknowledgeLongRun before the full development ladder." }
    Invoke-Checked (@("experiments/53_deeponet_development.py", "--device", "cuda") + $ResumeArguments) "18-configuration development ladder"
}
if ($Stage -in @("All", "Validation")) {
    Invoke-Checked (@("experiments/54_deeponet_five_seed_validation.py", "--device", "cuda") + $ResumeArguments) "Frozen five-seed validation"
}
if ($Stage -in @("All", "Heldout")) {
    Invoke-Checked @("experiments/55_deeponet_heldout_prediction_and_scoring.py", "predict-heldout", "--device", "cuda") "Heldout prediction"
}
if ($Stage -in @("All", "Score")) {
    Invoke-Checked @("experiments/55_deeponet_heldout_prediction_and_scoring.py", "score-heldout", "--device", "cpu") "One-time heldout scoring"
}
if ($Stage -in @("All", "Runtime")) {
    Invoke-Checked @("experiments/56_deeponet_hybrid_and_runtime.py", "--device", "cuda", "--warmups", "5", "--repeats", "30") "CUDA/CPU runtime and exact hybrid"
    Invoke-Checked @("experiments/56_deeponet_hybrid_and_runtime.py", "--device", "cpu", "--warmups", "5", "--repeats", "30") "CPU-only DeepONet timing"
}
if ($Stage -in @("All", "Report")) {
    Invoke-Checked @("experiments/57_deeponet_synthesis.py") "Layered synthesis"
}
