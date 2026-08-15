"""Stage 3: temporal convergence audit for American-option Delta and Gamma."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    continuation_premium,
    extract_boundary_at_time,
)
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta,
    finite_difference_gamma,
)
from american_risk_surfaces.method_extensions.protocol import AUDIT_REGIME_IDS, DATASET_DIR
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.greek_integrators import (
    AmericanGreekIntegratorResult,
    american_dirk_policy_price,
    american_lobatto_penalty_price,
    american_theta_policy_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "07_method_extensions" / "03_greek_audit"
TIME_STEPS = (60, 120, 240, 480)
METHODS: dict[str, Callable[[AmericanLCPConfig], AmericanGreekIntegratorResult]] = {
    "cn_quadratic": lambda config: american_theta_policy_price(
        config, theta=0.5, quadratic_time=True, damping_steps=0
    ),
    "rannacher_cn_quadratic": lambda config: american_theta_policy_price(
        config, theta=0.5, quadratic_time=True, damping_steps=2
    ),
    "dirk_lstable_quadratic": lambda config: american_dirk_policy_price(
        config, quadratic_time=True, damping_steps=2
    ),
    "lobatto_iiic_penalty_quadratic": lambda config: american_lobatto_penalty_price(
        config, quadratic_time=True, damping_steps=2
    ),
}


def run_greek_audit(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    regime_limit: int | None = None,
    spatial_intervals: int = 240,
    reference_time_steps: int = 960,
) -> dict[str, Any]:
    if reference_time_steps <= max(TIME_STEPS):
        raise ValueError("reference_time_steps must exceed the largest audited N")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tolerance = _frozen_tolerance()
    regimes = _audit_regimes(regime_limit)
    rows: list[dict[str, Any]] = []

    for regime in regimes:
        reference_config = _config(
            regime, tolerance, M=spatial_intervals, N=reference_time_steps
        )
        reference = american_dirk_policy_price(reference_config)
        reference_delta = finite_difference_delta(reference.spot_grid, reference.values)
        reference_gamma = finite_difference_gamma(reference.spot_grid, reference.values)
        reference_boundary = _final_boundary(reference)
        mask = _evaluation_mask(reference, reference_delta, reference_gamma, reference_boundary)

        for method_name, solve in METHODS.items():
            for time_steps in TIME_STEPS:
                config = _config(
                    regime, tolerance, M=spatial_intervals, N=time_steps
                )
                result = solve(config)
                delta = finite_difference_delta(result.spot_grid, result.values)
                gamma = finite_difference_gamma(result.spot_grid, result.values)
                value_error = result.values[mask] - reference.values[mask]
                delta_error = delta[mask] - reference_delta[mask]
                gamma_error = gamma[mask] - reference_gamma[mask]
                boundary = _final_boundary(result)
                rows.append(
                    {
                        "regime_id": regime["regime_id"],
                        "option_type": regime["option_type"],
                        "method": method_name,
                        "M": spatial_intervals,
                        "N": time_steps,
                        "reference_N": reference_time_steps,
                        "evaluation_nodes": int(np.count_nonzero(mask)),
                        "runtime_seconds": result.total_seconds,
                        "converged": result.converged,
                        "max_obstacle_violation": result.max_obstacle_violation,
                        "value_max_error": float(np.max(np.abs(value_error))),
                        "delta_max_error": float(np.max(np.abs(delta_error))),
                        "gamma_max_error": float(np.max(np.abs(gamma_error))),
                        "value_rmse": float(np.sqrt(np.mean(value_error**2))),
                        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
                        "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
                        "boundary_found": boundary.boundary_found,
                        "reference_boundary_found": reference_boundary.boundary_found,
                        "boundary_abs_error": (
                            abs(boundary.boundary_spot - reference_boundary.boundary_spot)
                            if boundary.boundary_found and reference_boundary.boundary_found
                            else float("nan")
                        ),
                        "delta_empirical_order": float("nan"),
                        "gamma_empirical_order": float("nan"),
                    }
                )

    _attach_orders(rows)
    rows_path = output / "temporal_convergence.csv"
    _write_csv(rows_path, rows, tuple(rows[0]))
    summary_rows = _method_summary(rows, len(regimes))
    summary_path = output / "method_summary.csv"
    _write_csv(summary_path, summary_rows, tuple(summary_rows[0]))
    decision = _greek_decision(summary_rows, rows, spatial_intervals)
    decision.update(
        {
            "audit_regimes": len(regimes),
            "spatial_intervals": spatial_intervals,
            "time_steps": list(TIME_STEPS),
            "reference_time_steps": reference_time_steps,
            "frozen_lcp_tolerance": tolerance,
            "lobatto_constraint_method": "published_coupled_penalty_iteration",
            "nonuniform_spatial_grid_status": "DEFERRED_TO_SPATIAL_AUDIT",
            "front_fixing_status": "DEFERRED_TO_INDEPENDENT_BOUNDARY_STAGE",
        }
    )
    decision_path = output / "greek_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "greek_audit_report.md"
    report_path.write_text(_report(decision, summary_rows), encoding="utf-8")
    return {
        "rows": rows_path,
        "summary": summary_path,
        "decision": decision_path,
        "report": report_path,
        "decision_data": decision,
    }


def _final_boundary(result: AmericanGreekIntegratorResult):
    premium = continuation_premium(result.values, result.payoff)
    return extract_boundary_at_time(
        result.spot_grid,
        premium,
        result.config.option_type,
        tau=float(result.tau_grid[-1]),
        time_index=len(result.tau_grid) - 1,
    )


def _evaluation_mask(
    reference: AmericanGreekIntegratorResult,
    delta: np.ndarray,
    gamma: np.ndarray,
    boundary: Any,
) -> np.ndarray:
    spots = reference.spot_grid
    dS = float(spots[1] - spots[0])
    mask = (
        (spots / reference.config.K >= 0.8)
        & (spots / reference.config.K <= 1.2)
        & np.isfinite(delta)
        & np.isfinite(gamma)
        & (np.abs(spots - reference.config.K) > 3.0 * dS)
    )
    if boundary.boundary_found:
        mask &= np.abs(spots - boundary.boundary_spot) > 3.0 * dS
    if not np.any(mask):
        raise RuntimeError("Greek audit mask contains no nodes")
    return mask


def _attach_orders(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["regime_id"]), str(row["method"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["N"]))
        for previous, current in zip(group, group[1:]):
            for metric in ("delta", "gamma"):
                old_error = float(previous[f"{metric}_max_error"])
                new_error = float(current[f"{metric}_max_error"])
                current[f"{metric}_empirical_order"] = (
                    math.log(old_error / new_error, 2.0)
                    if old_error > 0.0 and new_error > 0.0
                    else float("nan")
                )


def _method_summary(rows: list[dict[str, Any]], regime_count: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        finest = [
            row for row in rows if row["method"] == method and int(row["N"]) == max(TIME_STEPS)
        ]
        stable = [
            row
            for row in finest
            if float(row["delta_empirical_order"]) >= 1.5
            and float(row["gamma_empirical_order"]) >= 1.5
        ]
        boundary_errors = np.array(
            [float(row["boundary_abs_error"]) for row in finest], dtype=float
        )
        summaries.append(
            {
                "method": method,
                "regime_count": regime_count,
                "all_converged": all(bool(row["converged"]) for row in finest),
                "stable_second_order_regimes": len(stable),
                "stable_second_order_fraction": len(stable) / max(regime_count, 1),
                "median_value_max_error": float(np.median([row["value_max_error"] for row in finest])),
                "median_delta_max_error": float(np.median([row["delta_max_error"] for row in finest])),
                "median_gamma_max_error": float(np.median([row["gamma_max_error"] for row in finest])),
                "max_boundary_abs_error": (
                    float(np.nanmax(boundary_errors))
                    if np.any(np.isfinite(boundary_errors))
                    else float("nan")
                ),
                "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in finest])),
            }
        )
    return summaries


def _greek_decision(
    summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]], M: int
) -> dict[str, Any]:
    best = min(summary_rows, key=lambda row: float(row["median_gamma_max_error"]))
    boundary_ok = (
        np.isfinite(float(best["max_boundary_abs_error"]))
        and float(best["max_boundary_abs_error"]) <= 4.0 / M
    )
    stable = float(best["stable_second_order_fraction"]) >= 0.90
    all_converged = bool(best["all_converged"])
    unblock = stable and boundary_ok and all_converged
    return {
        "status": "GAMMA_REFERENCE_CANDIDATE" if unblock else "KEEP_GAMMA_BLOCKED",
        "gamma_status": "CANDIDATE_FOR_SPATIAL_AUDIT" if unblock else "BLOCKED",
        "best_temporal_method": best["method"],
        "stable_second_order_fraction": best["stable_second_order_fraction"],
        "boundary_within_one_grid_cell": boundary_ok,
        "all_converged": all_converged,
        "required_stable_fraction": 0.90,
        "note": "Temporal evidence alone cannot unblock Gamma; spatial/nonuniform-grid audit remains required.",
    }


def _report(decision: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Greek Time-Integrator Audit",
        "",
        f"Decision: **{decision['status']}**",
        "",
        f"Best temporal method: `{decision['best_temporal_method']}`",
        "",
        "| Method | Stable fraction | Median Delta max error | Median Gamma max error |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['stable_second_order_fraction']:.3f} | "
            f"{row['median_delta_max_error']:.6g} | {row['median_gamma_max_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Gamma remains blocked unless both this temporal gate and a later spatial-grid gate pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_regimes(limit: int | None) -> list[dict[str, str]]:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        by_id = {row["regime_id"]: row for row in csv.DictReader(handle)}
    rows = [by_id[regime_id] for regime_id in AUDIT_REGIME_IDS]
    return rows if limit is None else rows[:limit]


def _config(
    regime: dict[str, str], tolerance: float, *, M: int, N: int
) -> AmericanLCPConfig:
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
    path = PROJECT_ROOT / "results" / "07_method_extensions" / "00_protocol" / "tolerance_decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    return float(decision["frozen_normalized_lcp_tolerance"])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-limit", type=int)
    parser.add_argument("--M", type=int, default=240)
    parser.add_argument("--reference-N", type=int, default=960)
    args = parser.parse_args()
    result = run_greek_audit(
        regime_limit=args.regime_limit,
        spatial_intervals=args.M,
        reference_time_steps=args.reference_N,
    )
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
