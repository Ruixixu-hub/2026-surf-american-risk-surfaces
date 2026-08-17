"""Experiment 62: freeze Penalty/Newton, then run the four-arm strict comparison."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from american_risk_surfaces.method_extensions.poster_unified_study import (
    resummarize_strict_solver_benchmark,
    run_protocol_and_penalty_validation,
    run_strict_solver_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("validate", "benchmark", "resummarize", "all")
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--regime-limit", type=int)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    outputs = {}
    if args.phase in {"validate", "all"}:
        validation = run_protocol_and_penalty_validation(
            regime_limit=args.regime_limit if args.phase == "validate" else None
        )
        outputs["validation"] = validation["frozen_data"]
    if args.phase in {"benchmark", "all"}:
        benchmark = run_strict_solver_benchmark(
            warmups=args.warmups,
            repeats=args.repeats,
            regime_limit=args.regime_limit,
            allow_existing=args.allow_existing,
        )
        outputs["benchmark"] = benchmark["decision_data"]
    if args.phase == "resummarize":
        summary = resummarize_strict_solver_benchmark()
        outputs["resummarize"] = summary["decision_data"]
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
