"""Stage 3B: uniform versus strike-concentrated spatial Greek audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import continuation_premium, extract_boundary_at_time
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.method_extensions.protocol import AUDIT_REGIME_IDS, DATASET_DIR
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.greek_integrators import (
    AmericanGreekIntegratorResult,
    american_dirk_policy_price,
)
from american_risk_surfaces.solvers.grid import sinh_spot_grid, uniform_spot_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "07_method_extensions" / "03_greek_audit"
SPATIAL_LEVELS = (120, 240, 480)


def run_spatial_audit(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    regime_limit: int | None = None,
    time_steps: int = 960,
    reference_M: int = 960,
) -> dict[str, Any]:
    if reference_M <= max(SPATIAL_LEVELS):
        raise ValueError("reference_M must exceed the audited spatial grids")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tolerance = _frozen_tolerance()
    regimes = _regimes(regime_limit)
    rows: list[dict[str, Any]] = []

    for regime in regimes:
        reference_config = _config(regime, tolerance, reference_M, time_steps)
        reference_grid = sinh_spot_grid(
            reference_config.Smax, reference_config.K, reference_M
        )
        reference = american_dirk_policy_price(reference_config, spot_grid=reference_grid)
        reference_delta = finite_difference_delta_nonuniform(
            reference.spot_grid, reference.values
        )
        reference_gamma = finite_difference_gamma_nonuniform(
            reference.spot_grid, reference.values
        )
        reference_boundary = _boundary(reference)
        query = _query_spots(reference, reference_boundary)
        ref_value = np.interp(query, reference.spot_grid, reference.values)
        ref_delta = np.interp(query, reference.spot_grid, reference_delta)
        ref_gamma = np.interp(query, reference.spot_grid, reference_gamma)

        for grid_name in ("uniform", "sinh_strike_concentrated"):
            for M in SPATIAL_LEVELS:
                config = _config(regime, tolerance, M, time_steps)
                grid = (
                    uniform_spot_grid(config.Smax, M)[0]
                    if grid_name == "uniform"
                    else sinh_spot_grid(config.Smax, config.K, M)
                )
                result = american_dirk_policy_price(config, spot_grid=grid)
                delta = finite_difference_delta_nonuniform(result.spot_grid, result.values)
                gamma = finite_difference_gamma_nonuniform(result.spot_grid, result.values)
                value_error = np.interp(query, result.spot_grid, result.values) - ref_value
                delta_error = np.interp(query, result.spot_grid, delta) - ref_delta
                gamma_error = np.interp(query, result.spot_grid, gamma) - ref_gamma
                boundary = _boundary(result)
                comparison_spot = (
                    reference_boundary.boundary_spot
                    if reference_boundary.boundary_found
                    else config.K
                )
                local_spacing = _local_spacing(result.spot_grid, comparison_spot)
                boundary_consistent = (
                    not boundary.boundary_found and not reference_boundary.boundary_found
                ) or (
                    boundary.boundary_found
                    and reference_boundary.boundary_found
                    and abs(boundary.boundary_spot - reference_boundary.boundary_spot)
                    <= local_spacing
                )
                rows.append(
                    {
                        "regime_id": regime["regime_id"],
                        "option_type": regime["option_type"],
                        "grid": grid_name,
                        "M": M,
                        "N": time_steps,
                        "reference_M": reference_M,
                        "query_nodes": len(query),
                        "runtime_seconds": result.total_seconds,
                        "converged": result.converged,
                        "max_obstacle_violation": result.max_obstacle_violation,
                        "value_max_error": float(np.max(np.abs(value_error))),
                        "delta_max_error": float(np.max(np.abs(delta_error))),
                        "gamma_max_error": float(np.max(np.abs(gamma_error))),
                        "value_rmse": float(np.sqrt(np.mean(value_error**2))),
                        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
                        "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
                        "boundary_abs_error": (
                            abs(boundary.boundary_spot - reference_boundary.boundary_spot)
                            if boundary.boundary_found and reference_boundary.boundary_found
                            else float("nan")
                        ),
                        "local_grid_spacing": local_spacing,
                        "boundary_consistent": boundary_consistent,
                        "delta_empirical_order": float("nan"),
                        "gamma_empirical_order": float("nan"),
                    }
                )

    _attach_orders(rows)
    rows_path = output / "spatial_convergence.csv"
    _write_csv(rows_path, rows, tuple(rows[0]))
    summaries = _summaries(rows, len(regimes))
    summary_path = output / "spatial_grid_summary.csv"
    _write_csv(summary_path, summaries, tuple(summaries[0]))
    temporal_decision = json.loads((output / "greek_decision.json").read_text(encoding="utf-8"))
    decision = _decision(summaries, temporal_decision)
    decision.update(
        {
            "audit_regimes": len(regimes),
            "time_steps": time_steps,
            "spatial_levels": list(SPATIAL_LEVELS),
            "reference_M": reference_M,
            "reference_grid": "sinh_strike_concentrated",
            "frozen_lcp_tolerance": tolerance,
            "front_fixing_status": "DEFERRED; not needed to decide the Greek label gate",
        }
    )
    decision_path = output / "spatial_greek_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "spatial_greek_report.md"
    report_path.write_text(_report(decision, summaries), encoding="utf-8")
    return {
        "rows": rows_path,
        "summary": summary_path,
        "decision": decision_path,
        "report": report_path,
        "decision_data": decision,
    }


def _query_spots(result: AmericanGreekIntegratorResult, boundary: Any) -> np.ndarray:
    query = result.config.K * np.linspace(0.8, 1.2, 41)
    query = query[np.abs(query - result.config.K) > 0.03 * result.config.K]
    if boundary.boundary_found:
        query = query[np.abs(query - boundary.boundary_spot) > 0.03 * result.config.K]
    if len(query) < 5:
        raise RuntimeError("spatial Greek query region is too small")
    return query


def _boundary(result: AmericanGreekIntegratorResult):
    return extract_boundary_at_time(
        result.spot_grid,
        continuation_premium(result.values, result.payoff),
        result.config.option_type,
        tau=float(result.tau_grid[-1]),
        time_index=len(result.tau_grid) - 1,
    )


def _local_spacing(grid: np.ndarray, spot: float) -> float:
    index = int(np.clip(np.searchsorted(grid, spot), 1, len(grid) - 1))
    return float(grid[index] - grid[index - 1])


def _attach_orders(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["regime_id"]), str(row["grid"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["M"]))
        for previous, current in zip(group, group[1:]):
            for metric in ("delta", "gamma"):
                old = float(previous[f"{metric}_max_error"])
                new = float(current[f"{metric}_max_error"])
                current[f"{metric}_empirical_order"] = (
                    math.log(old / new, 2.0) if old > 0.0 and new > 0.0 else float("nan")
                )


def _summaries(rows: list[dict[str, Any]], regime_count: int) -> list[dict[str, Any]]:
    summaries = []
    for grid in ("uniform", "sinh_strike_concentrated"):
        finest = [row for row in rows if row["grid"] == grid and int(row["M"]) == 480]
        stable = [
            row
            for row in finest
            if float(row["delta_empirical_order"]) >= 1.5
            and float(row["gamma_empirical_order"]) >= 1.5
        ]
        boundary_ok = [row for row in finest if bool(row["boundary_consistent"])]
        summaries.append(
            {
                "grid": grid,
                "regime_count": regime_count,
                "all_converged": all(bool(row["converged"]) for row in finest),
                "stable_second_order_fraction": len(stable) / max(regime_count, 1),
                "boundary_within_local_cell_fraction": len(boundary_ok) / max(regime_count, 1),
                "median_delta_max_error": float(np.median([row["delta_max_error"] for row in finest])),
                "median_gamma_max_error": float(np.median([row["gamma_max_error"] for row in finest])),
                "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in finest])),
            }
        )
    return summaries


def _decision(summaries: list[dict[str, Any]], temporal: dict[str, Any]) -> dict[str, Any]:
    best = min(summaries, key=lambda row: float(row["median_gamma_max_error"]))
    spatial_pass = (
        bool(best["all_converged"])
        and float(best["stable_second_order_fraction"]) >= 0.90
        and float(best["boundary_within_local_cell_fraction"]) >= 0.90
    )
    temporal_pass = temporal["status"] == "GAMMA_REFERENCE_CANDIDATE"
    unblocked = temporal_pass and spatial_pass
    return {
        "status": "UNBLOCK_GAMMA_ON_STABLE_MASK" if unblocked else "KEEP_GAMMA_BLOCKED",
        "gamma_status": "REFERENCE_STABLE_MASK_ALLOWED" if unblocked else "BLOCKED",
        "selected_time_integrator": temporal["best_temporal_method"],
        "selected_spatial_grid": best["grid"],
        "temporal_gate_passed": temporal_pass,
        "spatial_gate_passed": spatial_pass,
        "stable_second_order_fraction": best["stable_second_order_fraction"],
        "boundary_within_local_cell_fraction": best["boundary_within_local_cell_fraction"],
    }


def _report(decision: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Spatial Greek Audit",
        "",
        f"Decision: **{decision['status']}**",
        "",
        f"Selected grid: `{decision['selected_spatial_grid']}`",
        "",
        "| Grid | Stable fraction | Median Delta max error | Median Gamma max error |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['grid']} | {row['stable_second_order_fraction']:.3f} | "
            f"{row['median_delta_max_error']:.6g} | {row['median_gamma_max_error']:.6g} |"
        )
    return "\n".join(lines) + "\n"


def _regimes(limit: int | None) -> list[dict[str, str]]:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        by_id = {row["regime_id"]: row for row in csv.DictReader(handle)}
    rows = [by_id[regime_id] for regime_id in AUDIT_REGIME_IDS]
    return rows if limit is None else rows[:limit]


def _config(regime: dict[str, str], tolerance: float, M: int, N: int) -> AmericanLCPConfig:
    return AmericanLCPConfig(
        option_type=regime["option_type"],
        K=float(regime["K"]),
        T=float(regime["T"]),
        r=float(regime["r"]),
        q=float(regime["q"]),
        sigma=float(regime["sigma"]),
        Smax=float(regime["Smax"]),
        M=M,
        N=N,
        tolerance=tolerance,
        obstacle_tolerance=1e-12,
    )


def _frozen_tolerance() -> float:
    decision = json.loads(
        (
            PROJECT_ROOT / "results" / "07_method_extensions" / "00_protocol" / "tolerance_decision.json"
        ).read_text(encoding="utf-8")
    )
    return float(decision["frozen_normalized_lcp_tolerance"])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-limit", type=int)
    parser.add_argument("--N", type=int, default=960)
    parser.add_argument("--reference-M", type=int, default=960)
    args = parser.parse_args()
    result = run_spatial_audit(
        regime_limit=args.regime_limit,
        time_steps=args.N,
        reference_M=args.reference_M,
    )
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
