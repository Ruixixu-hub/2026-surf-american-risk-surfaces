"""Experiment 30: train Arm C Soft-LCP vanilla PINNs."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.protocol import DEVELOPMENT_REGIME_IDS, RESULTS_DIR, SEEDS
from american_risk_surfaces.pinn.study import run_training_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tiny", "development", "sensitivity", "validation"), default="tiny")
    parser.add_argument(
        "--architecture",
        choices=("resnet_4x2x50", "resnet_4x2x64", "mlp_4x64", "mlp_6x64", "all"),
        default="resnet_4x2x50",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--lbfgs-evaluations", type=int)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "tiny":
        regime_ids = DEVELOPMENT_REGIME_IDS[:4]
        splits = ("train",)
        seeds = (17,)
        default_steps, default_lbfgs = 500, 0
    elif args.mode in {"development", "sensitivity"}:
        regime_ids = DEVELOPMENT_REGIME_IDS
        splits = ("train", "validation")
        seeds = (17,)
        default_steps, default_lbfgs = 40000, 2000
    else:
        regime_ids = None
        splits = ("validation",)
        seeds = SEEDS
        default_steps, default_lbfgs = 40000, 2000
    names = tuple(_architecture_specs()) if args.architecture == "all" else (args.architecture,)
    summaries = []
    for name in names:
        rows = run_training_jobs(
            arms=("C",),
            splits=splits,
            regime_ids=regime_ids,
            seeds=seeds,
            output_dir=RESULTS_DIR / "01_arm_c" / args.mode / name,
            device=args.device,
            resume=args.resume,
            adam_steps=args.adam_steps if args.adam_steps is not None else default_steps,
            lbfgs_max_evaluations=(
                args.lbfgs_evaluations if args.lbfgs_evaluations is not None else default_lbfgs
            ),
            max_seconds=args.max_seconds,
            network_spec=_architecture_specs()[name],
        )
        summaries.append({"architecture": name, "jobs": len(rows), "status_counts": _counts(rows)})
    print(json.dumps(summaries, indent=2))


def _counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {status: sum(row["status"] == status for row in rows) for status in ("COMPLETE", "FAILED", "BUDGET_EXHAUSTED")}


def _architecture_specs() -> dict[str, NetworkSpec]:
    return {
        "resnet_4x2x50": NetworkSpec("resnet", width=50, blocks=4, layers_per_block=2),
        "resnet_4x2x64": NetworkSpec("resnet", width=64, blocks=4, layers_per_block=2),
        "mlp_4x64": NetworkSpec("mlp", width=64, hidden_layers=4),
        "mlp_6x64": NetworkSpec("mlp", width=64, hidden_layers=6),
    }


if __name__ == "__main__":
    main()
