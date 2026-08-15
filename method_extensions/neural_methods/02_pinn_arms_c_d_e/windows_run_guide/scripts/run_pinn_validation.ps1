param(
    [ValidateSet("C", "D")]
    [string]$Arm,
    [ValidateSet("resnet_4x2x50", "resnet_4x2x64", "mlp_4x64", "mlp_6x64")]
    [string]$Architecture = "resnet_4x2x50",
    [ValidateSet("etc_soft", "etc_fb_global", "etc_fb_mixture", "etc_fb_adaptive", "positive_premium")]
    [string]$DVariant = "etc_fb_adaptive",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if ($Arm -eq "C") {
    $Arguments = @("experiments/30_arm_c_soft_lcp_pinn.py", "--mode", "validation", "--architecture", $Architecture, "--device", "cuda")
} else {
    $Arguments = @("experiments/31_arm_d_etc_fb_pinn.py", "--mode", "validation", "--variant", $DVariant, "--architecture", $Architecture, "--device", "cuda")
}
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
