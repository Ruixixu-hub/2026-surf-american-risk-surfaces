"""Experiment 40: separated label-blind run and one-time held-out scoring."""

from __future__ import annotations

import argparse

from american_risk_surfaces.reduced_order.study import (
    run_heldout_predictions,
    score_heldout_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run-heldout")
    run.add_argument("--no-resume", action="store_true")
    score = subcommands.add_parser("score-heldout")
    score.add_argument("--reference-m", type=int, default=480)
    score.add_argument("--reference-n", type=int, default=960)
    score.add_argument("--runtime-repeats", type=int, default=30)
    score.add_argument("--warmups", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.command == "run-heldout":
        print(run_heldout_predictions(resume=not arguments.no_resume))
    else:
        metrics, runtime = score_heldout_predictions(
            reference_m=arguments.reference_m,
            reference_n=arguments.reference_n,
            runtime_repeats=arguments.runtime_repeats,
            warmups=arguments.warmups,
        )
        print(f"metrics={metrics}")
        print(f"runtime={runtime}")


if __name__ == "__main__":
    main()
