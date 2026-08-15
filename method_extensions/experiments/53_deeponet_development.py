"""Experiment 53: N0/N1/N2 by latent-rank development ladder."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.deeponet.protocol import ARMS, LATENT_RANKS
from american_risk_surfaces.deeponet.study import run_development, run_tiny_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("put", "call"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--rank", type=int, choices=LATENT_RANKS)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tiny-smoke", action="store_true",
        help="write an isolated smoke artifact; never enter the development decision",
    )
    arguments = parser.parse_args()
    if arguments.tiny_smoke:
        payload = run_tiny_smoke(
            family=arguments.family or "put", arm=arguments.arm or "N0",
            rank=arguments.rank or 32, steps=arguments.steps,
            device=arguments.device,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _rows, decision = run_development(
        families=(arguments.family,) if arguments.family else ("put", "call"),
        arms=(arguments.arm,) if arguments.arm else ARMS,
        ranks=(arguments.rank,) if arguments.rank else LATENT_RANKS,
        steps=arguments.steps, device=arguments.device, resume=arguments.resume,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
