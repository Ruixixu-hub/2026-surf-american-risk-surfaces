"""Experiment 47: full-grid train-only POD representation ceiling."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.basis_operator.protocol import DIMENSION_LADDER
from american_risk_surfaces.basis_operator.study import (
    build_pod_basis_ladder,
    evaluate_representation_ceiling,
    select_representation_modes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", nargs="+", type=int, default=list(DIMENSION_LADDER))
    parser.add_argument("--reference-m", type=int, default=480)
    parser.add_argument("--reference-n", type=int, default=960)
    arguments = parser.parse_args()
    build_pod_basis_ladder(arguments.dimensions)
    rows = evaluate_representation_ceiling(
        dimensions=arguments.dimensions,
        reference_m=arguments.reference_m,
        reference_n=arguments.reference_n,
    )
    decision = select_representation_modes(rows)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
