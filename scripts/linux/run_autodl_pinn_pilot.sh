#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 C|D [seed] [train|reference|score|all]" >&2
  exit 2
}

arm="${1:-}"
seed="${2:-101}"
phase="${3:-all}"

case "$arm" in
  C) experiment="experiments/33b_arm_c_seed101_pilot.py" ;;
  D) experiment="experiments/33a_arm_d_seed101_pilot.py" ;;
  *) usage ;;
esac

case "$seed" in
  17|29|43|71|101) ;;
  *) echo "Seed must be one of: 17, 29, 43, 71, 101" >&2; exit 2 ;;
esac

case "$phase" in
  train|reference|score|all) ;;
  *) usage ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

export PYTHONPATH="$project_root/src"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python_bin="${PINN_PYTHON:-python}"
"$python_bin" - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected Python environment")
print(json.dumps({
    "cuda_available": True,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, indent=2))
PY

args=("$experiment" "$phase" --device cuda --seed "$seed")
if [[ "$phase" == "train" || "$phase" == "all" ]]; then
  args+=(--resume)
fi

echo "Running Arm $arm, seed $seed, phase $phase"
exec "$python_bin" "${args[@]}"
