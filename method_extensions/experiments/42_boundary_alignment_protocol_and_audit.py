"""Experiment 42: freeze protocol, audit extraction and alignment round trips."""

from __future__ import annotations

import csv
import json

from american_risk_surfaces.boundary_aligned_basis import (
    BoundaryAlignmentConfig,
    audit_alignment_resolution,
    audit_boundary_threshold_sensitivity,
)
from american_risk_surfaces.boundary_aligned_basis.protocol import (
    CANONICAL_POINT_LADDER,
    RESULTS_DIR,
    train_snapshot_paths,
    write_protocol,
)


def _write_csv(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol = write_protocol()
    output = RESULTS_DIR / "01_transformation_audit"
    output.mkdir(parents=True, exist_ok=True)
    decision: dict[str, object] = {"families": {}}
    for family in ("put", "call"):
        paths = train_snapshot_paths(family)
        sensitivity_rows, sensitivity = audit_boundary_threshold_sensitivity(paths)
        _write_csv(output / f"{family}_boundary_threshold_sensitivity.csv", sensitivity_rows)
        resolution_summaries = []
        selected = None
        for points in CANONICAL_POINT_LADDER:
            rows, summary = audit_alignment_resolution(
                paths, BoundaryAlignmentConfig(canonical_points=points)
            )
            _write_csv(output / f"{family}_roundtrip_{points}.csv", rows)
            resolution_summaries.append(summary)
            if summary["all_gates_pass"]:
                selected = points
                break
        if sensitivity["status"] != "GO_BOUNDARY_EXTRACTION":
            status = "DEFER_BOUNDARY_EXTRACTION"
        elif selected is None:
            status = "DEFER_INTERPOLATION"
        else:
            status = "GO_TRANSFORM"
        decision["families"][family] = {
            "status": status,
            "selected_canonical_points": selected,
            "threshold_sensitivity": sensitivity,
            "resolution_audits": resolution_summaries,
        }
    statuses = [decision["families"][family]["status"] for family in ("put", "call")]
    decision["status"] = "GO_TRANSFORM" if statuses == ["GO_TRANSFORM", "GO_TRANSFORM"] else (
        "PARTIAL_GO" if "GO_TRANSFORM" in statuses else statuses[0]
    )
    path = output / "transformation_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(f"protocol={protocol}")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
