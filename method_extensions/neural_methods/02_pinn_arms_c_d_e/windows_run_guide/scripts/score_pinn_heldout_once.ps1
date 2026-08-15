$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python experiments/33_pinn_heldout_training_and_scoring.py reference --device cpu
& $Python experiments/33_pinn_heldout_training_and_scoring.py score --device cuda
