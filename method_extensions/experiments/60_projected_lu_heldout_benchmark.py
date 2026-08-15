"""Experiment 60: one-shot held-out Projected-LU benchmark."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from american_risk_surfaces.method_extensions.projected_lu_study import (
    run_heldout_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--regime-limit", type=int)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    result = run_heldout_benchmark(
        warmups=args.warmups,
        repeats=args.repeats,
        regime_limit=args.regime_limit,
        allow_existing=args.allow_existing,
    )
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
