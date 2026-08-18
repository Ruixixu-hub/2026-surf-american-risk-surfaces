#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 C|D [seed] [workers]" >&2
  exit 2
}

arm="${1:-}"
seed="${2:-101}"
workers="${3:-4}"

case "$arm" in C|D) ;; *) usage ;; esac
case "$seed" in 17|29|43|71|101) ;; *) usage ;; esac
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "workers must be a positive integer" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

log_root="$project_root/results/08_pinn_gap/04_heldout_pilots/parallel_logs"
mkdir -p "$log_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
pids=()
max_seconds="$((workers * 3600))"

stop_children() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap stop_children INT TERM

echo "Starting $workers independent Arm $arm training shards."
echo "Existing checkpoints will be reused with --resume."
echo "Each job keeps the frozen optimization steps and has a ${max_seconds}s wall-clock safety cap."
for ((index=0; index<workers; index++)); do
  log="$log_root/arm_${arm}_seed${seed}_shard_${index}_of_${workers}_${timestamp}.log"
  bash scripts/linux/run_autodl_pinn_pilot.sh \
    "$arm" "$seed" train "$index" "$workers" "$max_seconds" >"$log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "  shard $index/$workers pid=$pid log=$log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one training shard failed. Inspect $log_root and rerun this command." >&2
  exit 1
fi
trap - INT TERM

echo "All training shards finished. Generating the shared reference."
bash scripts/linux/run_autodl_pinn_pilot.sh "$arm" "$seed" reference

echo "Reference finished. Scoring the frozen predictions once."
bash scripts/linux/run_autodl_pinn_pilot.sh "$arm" "$seed" score

echo "Arm $arm seed $seed parallel pilot completed."
