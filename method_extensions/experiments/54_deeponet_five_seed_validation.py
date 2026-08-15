"""Experiment 54: frozen five-seed validation and grid-transfer diagnostic."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.deeponet.protocol import SEEDS
from american_risk_surfaces.deeponet.study import run_five_seed_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("put", "call"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    _rows, decision = run_five_seed_validation(
        families=(arguments.family,) if arguments.family else ("put", "call"),
        seeds=(arguments.seed,) if arguments.seed else SEEDS,
        steps=arguments.steps, device=arguments.device, resume=arguments.resume,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
