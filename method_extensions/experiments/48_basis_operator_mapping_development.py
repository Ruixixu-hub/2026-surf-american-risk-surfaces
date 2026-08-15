"""Experiment 48: P0/P1/P2 coefficient-mapping development on validation."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.basis_operator.study import (
    configuration_gate,
    run_mapping_development,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--family", choices=("put", "call"))
    parser.add_argument("--modes", nargs="+", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.aggregate_only:
        import csv
        from american_risk_surfaces.basis_operator.protocol import RESULTS_DIR
        path = RESULTS_DIR / "04_mapping_development" / "mapping_development_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = run_mapping_development(
            steps=arguments.steps,
            families=(arguments.family,) if arguments.family else ("put", "call"),
            requested_modes=arguments.modes,
        )
    decision = configuration_gate(rows)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
