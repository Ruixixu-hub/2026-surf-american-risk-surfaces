$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python experiments/34_arm_e_hybrid_benchmark.py --device cuda --warmups 5 --repeats 30
& $Python experiments/35_pinn_gap_synthesis.py
