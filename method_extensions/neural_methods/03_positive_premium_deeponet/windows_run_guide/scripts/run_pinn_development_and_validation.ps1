param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$ResumeArgument = @()
if ($Resume) { $ResumeArgument = @("--resume") }

# Development-only Arm C architecture sensitivity (8 regimes, seed 17).
& $Python experiments/30_arm_c_soft_lcp_pinn.py --mode sensitivity --architecture all --device cuda @ResumeArgument
& $Python experiments/32_pinn_validation_gates.py --architecture-only --generate-reference --device cuda
$ArchitectureDecision = Get-Content results/08_pinn_gap/03_validation_gates/arm_c_architecture_decision.json | ConvertFrom-Json
$Architecture = $ArchitectureDecision.selected_architecture
if (-not $Architecture) { throw "Arm C architecture selection did not produce a frozen architecture." }

# Development-only Arm D ablations. Positive premium receives its registered 3 seeds.
& $Python experiments/31_arm_d_etc_fb_pinn.py --mode ablation --variant all --architecture $Architecture --device cuda @ResumeArgument
& $Python experiments/32_pinn_validation_gates.py --ablation-only --device cuda --reference-dir results/08_pinn_gap/03_validation_gates/high_accuracy_reference
$AblationDecision = Get-Content results/08_pinn_gap/03_validation_gates/arm_d_ablation_decision.json | ConvertFrom-Json
$Variant = "etc_fb_mixture"
if ($AblationDecision.adaptive_sampling.status -eq "GO") { $Variant = "etc_fb_adaptive" }
if ($AblationDecision.positive_premium.status -eq "GO") { $Variant = "positive_premium" }

# Five-seed validation for the selected, still label-frozen C/D configuration.
& $Python experiments/30_arm_c_soft_lcp_pinn.py --mode validation --architecture $Architecture --device cuda @ResumeArgument
& $Python experiments/31_arm_d_etc_fb_pinn.py --mode validation --variant $Variant --architecture $Architecture --device cuda @ResumeArgument
$ArmCDir = "results/08_pinn_gap/01_arm_c/validation/$Architecture"
$ArmDDir = "results/08_pinn_gap/02_arm_d/validation/$Variant/$Architecture"
& $Python experiments/32_pinn_validation_gates.py --device cuda --arm-c-dir $ArmCDir --arm-d-dir $ArmDDir --reference-dir results/08_pinn_gap/03_validation_gates/high_accuracy_reference
