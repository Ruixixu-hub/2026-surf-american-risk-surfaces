"""Stage 0: freeze the method-extension protocol and calibrate LCP tolerance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.method_extensions.protocol import (
    AUDIT_REGIME_IDS,
    DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    write_protocol_manifest,
)
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig, american_cn_lcp_price


TOLERANCES = (1e-8, 1e-10, 1e-12)


def run_tolerance_audit(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol_paths = write_protocol_manifest(output)
    regimes = _load_regimes()
    rows: list[dict[str, object]] = []
    solutions: dict[tuple[str, float], np.ndarray] = {}

    for regime_id in AUDIT_REGIME_IDS:
        row = regimes[regime_id]
        for tolerance in TOLERANCES:
            config = AmericanLCPConfig(
                option_type=row["option_type"],
                K=float(row["K"]),
                T=float(row["T"]),
                r=float(row["r"]),
                q=float(row["q"]),
                sigma=float(row["sigma"]),
                Smax=float(row["Smax"]),
                M=int(row["M"]),
                N=int(row["N"]),
                tolerance=tolerance,
                obstacle_tolerance=1e-12,
            )
            result = american_cn_lcp_price(config, lcp_solver="psor")
            solutions[(regime_id, tolerance)] = result.values
            residuals = [step.residual for step in result.lcp_results]
            iterations = [step.iterations for step in result.lcp_results]
            rows.append(
                {
                    "regime_id": regime_id,
                    "split": row["split"],
                    "option_type": row["option_type"],
                    "tolerance": tolerance,
                    "converged": result.converged,
                    "runtime_seconds": result.total_seconds,
                    "mean_iterations": float(np.mean(iterations)),
                    "max_iterations": int(np.max(iterations)),
                    "max_normalized_lcp_residual": max(
                        residual.normalized_lcp_residual for residual in residuals
                    ),
                    "max_normalized_obstacle_violation": max(
                        residual.normalized_obstacle_violation for residual in residuals
                    ),
                    "max_abs_difference_vs_1e12": "pending",
                }
            )

    for row in rows:
        key = (str(row["regime_id"]), float(row["tolerance"]))
        reference = solutions[(str(row["regime_id"]), 1e-12)]
        row["max_abs_difference_vs_1e12"] = float(np.max(np.abs(solutions[key] - reference)))

    csv_path = output / "tolerance_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows_1e10 = [row for row in rows if float(row["tolerance"]) == 1e-10]
    all_1e10 = all(bool(row["converged"]) for row in rows_1e10)
    max_difference = max(float(row["max_abs_difference_vs_1e12"]) for row in rows_1e10)
    all_1e12 = all(bool(row["converged"]) for row in rows if float(row["tolerance"]) == 1e-12)
    if all_1e10 and max_difference <= 1e-9:
        frozen_tolerance = 1e-10
        status = "FROZEN_1E-10"
    elif all_1e12:
        frozen_tolerance = 1e-12
        status = "FROZEN_1E-12_AFTER_1E-10_STABILITY_GATE_FAILED"
    else:
        frozen_tolerance = None
        status = "REVIEW_REQUIRED"
    decision = {
        "status": status,
        "frozen_normalized_lcp_tolerance": frozen_tolerance,
        "frozen_normalized_obstacle_tolerance": 1e-12,
        "all_1e10_converged": all_1e10,
        "max_1e10_vs_1e12_value_difference": max_difference,
        "all_1e12_converged": all_1e12,
        "comparison_threshold": 1e-9,
    }
    decision_path = output / "tolerance_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "protocol_paths": protocol_paths,
        "audit_csv": csv_path,
        "decision_path": decision_path,
        "decision": decision,
    }


def _load_regimes() -> dict[str, dict[str, str]]:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["regime_id"]: row for row in rows}
    missing = sorted(set(AUDIT_REGIME_IDS) - set(by_id))
    if missing:
        raise ValueError(f"audit regimes missing from frozen manifest: {missing}")
    return by_id


if __name__ == "__main__":
    result = run_tolerance_audit()
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
