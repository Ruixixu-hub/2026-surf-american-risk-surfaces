"""Experiment 56: same-machine classical, DeepONet, and exact-hybrid timing."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.deeponet.heldout import benchmark_runtime_and_hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    arguments = parser.parse_args()
    rows = benchmark_runtime_and_hybrid(
        device=arguments.device, warmups=arguments.warmups, repeats=arguments.repeats
    )
    print(json.dumps({"status": "COMPLETE", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
