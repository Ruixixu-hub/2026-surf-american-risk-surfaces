param(
    [int]$ShardIndex = 0,
    [int]$ShardCount = 1,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if ($ShardCount -lt 1 -or $ShardIndex -lt 0 -or $ShardIndex -ge $ShardCount) {
    throw "Require 0 <= ShardIndex < ShardCount"
}
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-pinn\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Arguments = @(
    "experiments/33_pinn_heldout_training_and_scoring.py", "train",
    "--device", "cuda",
    "--shard-index", $ShardIndex,
    "--shard-count", $ShardCount
)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
