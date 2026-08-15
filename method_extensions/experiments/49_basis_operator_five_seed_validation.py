"""Experiment 49: frozen five-seed validation and heldout gate."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.basis_operator.study import (
    configuration_gate,
    run_five_seed_validation,
)
from american_risk_surfaces.basis_operator.protocol import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--family", choices=("put", "call"))
    parser.add_argument("--seed", type=int, choices=(17, 29, 43, 71, 101))
    arguments = parser.parse_args()
    mapping_path = RESULTS_DIR / "04_mapping_development" / "mapping_development_metrics.csv"
    import csv
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        mapping_rows = list(csv.DictReader(handle))
    configuration_gate(mapping_rows)
    _rows, decision = run_five_seed_validation(
        steps=arguments.steps,
        families=(arguments.family,) if arguments.family else ("put", "call"),
        seeds=(arguments.seed,) if arguments.seed else (17, 29, 43, 71, 101),
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
