"""Experiment 39: choose the smallest passing put/call basis on validation only."""

from __future__ import annotations

import argparse
import csv
import json

from american_risk_surfaces.reduced_order.protocol import DIMENSION_LADDER, RESULTS_DIR, load_regimes
from american_risk_surfaces.reduced_order.study import (
    evaluate_basis_ladder,
    freeze_validation_decision,
    select_validation_dimensions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", nargs="+", type=int, default=list(DIMENSION_LADDER))
    parser.add_argument("--reference-m", type=int, default=480)
    parser.add_argument("--reference-n", type=int, default=960)
    arguments = parser.parse_args()
    rows = evaluate_basis_ladder(
        load_regimes(splits=("validation",)),
        dimensions=arguments.dimensions,
        reference_m=arguments.reference_m,
        reference_n=arguments.reference_n,
    )
    output = RESULTS_DIR / "04_validation"
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "validation_ladder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    decision = select_validation_dimensions(rows)
    decision["reference_grid"] = {"M": arguments.reference_m, "N": arguments.reference_n}
    (output / "validation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    frozen = freeze_validation_decision(decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"frozen_config={frozen}")


if __name__ == "__main__":
    main()
