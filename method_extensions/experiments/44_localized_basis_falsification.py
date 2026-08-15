"""Experiment 44: physical L and, when gated, aligned-localized AL ladders."""

from __future__ import annotations

import json

from american_risk_surfaces.boundary_aligned_basis.protocol import RESULTS_DIR
from american_risk_surfaces.boundary_aligned_basis.study import build_basis_grid


def main() -> None:
    decision = json.loads(
        (RESULTS_DIR / "01_transformation_audit" / "transformation_decision.json").read_text(
            encoding="utf-8"
        )
    )
    points = {
        family: decision["families"][family]["selected_canonical_points"]
        for family in ("put", "call")
        if decision["families"][family]["status"] == "GO_TRANSFORM"
    }
    arms = ["L"] + (["AL"] if points else [])
    paths, manifest = build_basis_grid(
        arms=arms,
        canonical_points_by_family=points,
    )
    print(f"arms={arms}")
    print(f"artifact_count={len(paths)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
