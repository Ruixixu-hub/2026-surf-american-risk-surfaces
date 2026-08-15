"""Experiment 37: build put/call POD-greedy, angle-greedy, supremizer ladders."""

from __future__ import annotations

import argparse
import csv

from american_risk_surfaces.reduced_order import load_basis
from american_risk_surfaces.reduced_order.protocol import DIMENSION_LADDER
from american_risk_surfaces.reduced_order.study import build_basis_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DIMENSION_LADDER),
    )
    arguments = parser.parse_args()
    artifacts, manifest = build_basis_artifacts(dimensions=arguments.dimensions)
    history_rows = []
    for family in ("put", "call"):
        available = [
            dimension
            for dimension in arguments.dimensions
            if (manifest.parent / family / f"basis_{dimension:02d}.npz").exists()
        ]
        maximum = max(available)
        basis = load_basis(manifest.parent / family / f"basis_{maximum:02d}.npz")
        for kind, key in (("primal_pod", "primal_history"), ("dual_angle", "dual_history")):
            for row in basis.metadata[key]:
                history_rows.append({"option_type": family, "kind": kind, **row})
    with (manifest.parent / "greedy_decay_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in history_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"basis_count={len(artifacts)}")
    print(f"basis_manifest={manifest}")


if __name__ == "__main__":
    main()
