param(
    [string]$Python = "py -3.11",
    [ValidateSet("cu126", "cu128")]
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

Invoke-Expression "$Python -m venv .venv-pinn"
$PinnPython = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
& $PinnPython -m pip install --upgrade pip
& $PinnPython -m pip install -r requirements-pinn.txt
& $PinnPython -m pip install torch==2.7.1 --index-url "https://download.pytorch.org/whl/$CudaWheel"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $PinnPython experiments/29_pinn_protocol_and_operator_audit.py
& $PinnPython scripts/windows/pinn_preflight.py
& $PinnPython -m unittest discover -s tests -p "test_pinn_cde.py" -v
