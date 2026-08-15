"""Experiment 31: Arm D ETC-FB ablations and formal validation training."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.protocol import DEVELOPMENT_REGIME_IDS, RESULTS_DIR, SEEDS
from american_risk_surfaces.pinn.study import run_training_jobs


VARIANTS = (
    "etc_soft",
    "etc_fb_global",
    "etc_fb_mixture",
    "etc_fb_adaptive",
    "positive_premium",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tiny", "ablation", "validation"), default="tiny")
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="etc_fb_adaptive")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--architecture",
        choices=("resnet_4x2x50", "resnet_4x2x64", "mlp_4x64", "mlp_6x64"),
        default="resnet_4x2x50",
    )
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--lbfgs-evaluations", type=int)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    summaries = []
    for variant in variants:
        if args.mode == "tiny":
            regime_ids, splits, seeds = DEVELOPMENT_REGIME_IDS[:4], ("train",), (17,)
            default_steps, default_lbfgs = 500, 0
        elif args.mode == "ablation":
            regime_ids, splits = DEVELOPMENT_REGIME_IDS, ("train", "validation")
            seeds = (17, 29, 43) if variant == "positive_premium" else (17,)
            default_steps, default_lbfgs = 40000, 2000
        else:
            regime_ids, splits, seeds = None, ("validation",), SEEDS
            default_steps, default_lbfgs = 40000, 2000
        rows = run_training_jobs(
            arms=("D",),
            splits=splits,
            regime_ids=regime_ids,
            seeds=seeds,
            output_dir=RESULTS_DIR / "02_arm_d" / args.mode / variant / args.architecture,
            device=args.device,
            resume=args.resume,
            adam_steps=args.adam_steps if args.adam_steps is not None else default_steps,
            lbfgs_max_evaluations=(
                args.lbfgs_evaluations if args.lbfgs_evaluations is not None else default_lbfgs
            ),
            max_seconds=args.max_seconds,
            d_variant=variant,
            network_spec=_architecture_specs()[args.architecture],
        )
        summaries.append({"variant": variant, "architecture": args.architecture, "jobs": len(rows), "complete": sum(row["status"] == "COMPLETE" for row in rows)})
    print(json.dumps(summaries, indent=2))


def _architecture_specs() -> dict[str, NetworkSpec]:
    return {
        "resnet_4x2x50": NetworkSpec("resnet", width=50, blocks=4, layers_per_block=2),
        "resnet_4x2x64": NetworkSpec("resnet", width=64, blocks=4, layers_per_block=2),
        "mlp_4x64": NetworkSpec("mlp", width=64, hidden_layers=4),
        "mlp_6x64": NetworkSpec("mlp", width=64, hidden_layers=6),
    }


if __name__ == "__main__":
    main()
