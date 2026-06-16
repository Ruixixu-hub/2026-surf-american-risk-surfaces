"""Ticket 09: continuation premium and boundary extraction experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    BoundaryExtractionSummary,
    extract_boundary_curve,
    selected_time_profile_rows,
    summarize_boundary_curve,
)
from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_09_boundary_extraction_summary.csv"
)
BY_TIME_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_09_boundary_by_time.csv"
)
SELECTED_TIMES_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_09_boundary_selected_times.csv"
)
PUT_BOUNDARY_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_09_american_put_boundary_curve.png"
)
CALL_BOUNDARY_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_09_dividend_call_boundary_curve.png"
)
PREMIUM_PROFILE_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_09_continuation_premium_profiles.png"
)
BOUNDARY_STATUS_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_09_boundary_found_status.png"
)

DEFAULT_THRESHOLD = 1e-6
SELECTED_TAU_FRACTIONS = (0.01, 0.5, 1.0)

DEFAULT_CASES = [
    {
        "case_name": "american_put_medium",
        "option_type": "put",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 120,
        "N": 120,
    },
    {
        "case_name": "dividend_call_medium",
        "option_type": "call",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.08,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 120,
        "N": 120,
    },
    {
        "case_name": "no_dividend_call_control",
        "option_type": "call",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.0,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 120,
        "N": 120,
    },
]

SUMMARY_FIELDNAMES = [
    "case_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "threshold",
    "search_direction",
    "total_time_rows",
    "positive_tau_rows",
    "found_boundary_count",
    "no_boundary_count",
    "maturity_ambiguous_count",
    "all_continuation_count",
    "all_exercise_count",
    "expected_exercise_side_absent_count",
    "no_clean_transition_count",
    "insufficient_interior_nodes_count",
    "first_boundary_tau",
    "last_boundary_tau",
    "min_boundary_spot",
    "max_boundary_spot",
    "status",
    "put_boundary_figure_created",
    "call_boundary_figure_created",
    "premium_profile_figure_created",
    "boundary_status_figure_created",
]

BY_TIME_FIELDNAMES = [
    "case_name",
    "option_type",
    "time_index",
    "tau",
    "boundary_found",
    "boundary_spot",
    "threshold",
    "search_direction",
    "extraction_method",
    "no_boundary_reason",
    "exercise_like_node_count",
    "continuation_like_node_count",
]

SELECTED_TIME_FIELDNAMES = [
    "case_name",
    "option_type",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "spot",
    "moneyness",
    "value",
    "payoff",
    "premium",
    "threshold",
    "premium_class",
]


def run_boundary_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run representative American cases and extract premium-based boundaries."""

    curves: list[BoundaryCurve] = []
    summaries: list[BoundaryExtractionSummary] = []
    for case in cases:
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        result = american_crank_nicolson_psor_price(**solver_kwargs)
        curve = extract_boundary_curve(result, case_name, threshold=threshold)
        curves.append(curve)
        summaries.append(summarize_boundary_curve(curve))

    put_boundary_figure_created = create_put_boundary_figure(curves, PUT_BOUNDARY_FIGURE)
    call_boundary_figure_created = create_dividend_call_boundary_figure(
        curves, CALL_BOUNDARY_FIGURE
    )
    premium_profile_figure_created = create_premium_profile_figure(
        curves, PREMIUM_PROFILE_FIGURE
    )
    boundary_status_figure_created = create_boundary_status_figure(
        curves, BOUNDARY_STATUS_FIGURE
    )

    figure_flags = {
        "put_boundary_figure_created": str(put_boundary_figure_created),
        "call_boundary_figure_created": str(call_boundary_figure_created),
        "premium_profile_figure_created": str(premium_profile_figure_created),
        "boundary_status_figure_created": str(boundary_status_figure_created),
    }
    summary_rows = [
        summary_row(summary) | figure_flags for summary in summaries
    ]
    by_time = [row for curve in curves for row in by_time_rows(curve)]
    selected_rows = [
        row
        for curve in curves
        for row in selected_time_rows(curve, selected_tau_fractions=SELECTED_TAU_FRACTIONS)
    ]

    metadata = {
        "summary_csv": str(SUMMARY_CSV),
        "by_time_csv": str(BY_TIME_CSV),
        "selected_times_csv": str(SELECTED_TIMES_CSV),
        "put_boundary_figure_path": str(PUT_BOUNDARY_FIGURE),
        "call_boundary_figure_path": str(CALL_BOUNDARY_FIGURE),
        "premium_profile_figure_path": str(PREMIUM_PROFILE_FIGURE),
        "boundary_status_figure_path": str(BOUNDARY_STATUS_FIGURE),
        "put_boundary_figure_created": str(put_boundary_figure_created),
        "call_boundary_figure_created": str(call_boundary_figure_created),
        "premium_profile_figure_created": str(premium_profile_figure_created),
        "boundary_status_figure_created": str(boundary_status_figure_created),
    }
    return summary_rows, by_time, selected_rows, metadata


def summary_row(summary: BoundaryExtractionSummary) -> dict[str, str]:
    """Convert a boundary extraction summary to a stable CSV row."""

    return {
        "case_name": summary.case_name,
        "option_type": summary.option_type,
        "K": _format_float(summary.K),
        "T": _format_float(summary.T),
        "r": _format_float(summary.r),
        "q": _format_float(summary.q),
        "sigma": _format_float(summary.sigma),
        "Smax": _format_float(summary.Smax),
        "M": str(summary.M),
        "N": str(summary.N),
        "threshold": _format_float(summary.threshold),
        "search_direction": summary.search_direction,
        "total_time_rows": str(summary.total_time_rows),
        "positive_tau_rows": str(summary.positive_tau_rows),
        "found_boundary_count": str(summary.found_boundary_count),
        "no_boundary_count": str(summary.no_boundary_count),
        "maturity_ambiguous_count": str(summary.maturity_ambiguous_count),
        "all_continuation_count": str(summary.all_continuation_count),
        "all_exercise_count": str(summary.all_exercise_count),
        "expected_exercise_side_absent_count": str(
            summary.expected_exercise_side_absent_count
        ),
        "no_clean_transition_count": str(summary.no_clean_transition_count),
        "insufficient_interior_nodes_count": str(
            summary.insufficient_interior_nodes_count
        ),
        "first_boundary_tau": _format_float(summary.first_boundary_tau),
        "last_boundary_tau": _format_float(summary.last_boundary_tau),
        "min_boundary_spot": _format_float(summary.min_boundary_spot),
        "max_boundary_spot": _format_float(summary.max_boundary_spot),
        "status": summary.status,
        "put_boundary_figure_created": "False",
        "call_boundary_figure_created": "False",
        "premium_profile_figure_created": "False",
        "boundary_status_figure_created": "False",
    }


def by_time_rows(curve: BoundaryCurve) -> list[dict[str, str]]:
    """Convert boundary points to stable by-time CSV rows."""

    return [
        {
            "case_name": curve.case_name,
            "option_type": curve.option_type,
            "time_index": str(point.time_index),
            "tau": _format_float(point.tau),
            "boundary_found": str(point.boundary_found),
            "boundary_spot": _format_float(point.boundary_spot),
            "threshold": _format_float(point.threshold),
            "search_direction": point.search_direction,
            "extraction_method": point.extraction_method,
            "no_boundary_reason": point.no_boundary_reason,
            "exercise_like_node_count": str(point.exercise_like_node_count),
            "continuation_like_node_count": str(point.continuation_like_node_count),
        }
        for point in curve.points
    ]


def selected_time_rows(
    curve: BoundaryCurve,
    selected_tau_fractions: tuple[float, ...] = SELECTED_TAU_FRACTIONS,
) -> list[dict[str, str]]:
    """Convert selected premium profile rows to stable CSV strings."""

    raw_rows = selected_time_profile_rows(curve, selected_tau_fractions)
    return [
        {
            "case_name": str(row["case_name"]),
            "option_type": str(row["option_type"]),
            "target_tau_fraction": _format_float(row["target_tau_fraction"]),
            "target_tau": _format_float(row["target_tau"]),
            "nearest_tau": _format_float(row["nearest_tau"]),
            "time_index": str(row["time_index"]),
            "spot": _format_float(row["spot"]),
            "moneyness": _format_float(row["moneyness"]),
            "value": _format_float(row["value"]),
            "payoff": _format_float(row["payoff"]),
            "premium": _format_float(row["premium"]),
            "threshold": _format_float(row["threshold"]),
            "premium_class": str(row["premium_class"]),
        }
        for row in raw_rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_put_boundary_figure(curves: list[BoundaryCurve], path: Path) -> bool:
    """Plot the extracted American put boundary against time-to-maturity."""

    put_curve = _find_curve(curves, "american_put_medium")
    if put_curve is None:
        return False
    return _create_boundary_curve_figure(
        put_curve,
        path,
        title="Ticket 09 American Put Boundary",
        ylabel="Boundary spot",
    )


def create_dividend_call_boundary_figure(curves: list[BoundaryCurve], path: Path) -> bool:
    """Plot the extracted dividend-paying American call boundary."""

    call_curve = _find_curve(curves, "dividend_call_medium")
    if call_curve is None:
        return False
    return _create_boundary_curve_figure(
        call_curve,
        path,
        title="Ticket 09 Dividend-Paying American Call Boundary",
        ylabel="Boundary spot",
    )


def create_premium_profile_figure(curves: list[BoundaryCurve], path: Path) -> bool:
    """Plot selected continuation-premium profiles for put and dividend-call cases."""

    put_curve = _find_curve(curves, "american_put_medium")
    call_curve = _find_curve(curves, "dividend_call_medium")
    if put_curve is None or call_curve is None:
        return False
    plt = _load_pyplot()
    if plt is None:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for axis, curve, title in (
        (axes[0], put_curve, "American put"),
        (axes[1], call_curve, "Dividend-paying American call"),
    ):
        for fraction in SELECTED_TAU_FRACTIONS:
            target_tau = curve.T * fraction
            time_index = int(np.argmin(np.abs(curve.tau_grid - target_tau)))
            axis.plot(
                curve.spot_grid / curve.K,
                curve.premium_grid[time_index],
                label=f"tau={curve.tau_grid[time_index]:.3f}",
            )
        axis.axhline(curve.threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("S / K")
        axis.set_ylabel("Continuation premium")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_boundary_status_figure(curves: list[BoundaryCurve], path: Path) -> bool:
    """Plot whether each case reports a boundary at each time-to-maturity row."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    offsets = np.arange(len(curves), dtype=float)
    for offset, curve in zip(offsets, curves):
        status = np.array([1.0 if point.boundary_found else 0.0 for point in curve.points])
        axis.step(curve.tau_grid, status + offset * 1.25, where="post", label=curve.case_name)
    axis.set_xlabel("Time to maturity tau")
    axis.set_ylabel("Boundary found status by case")
    axis.set_yticks(offsets * 1.25 + 0.5)
    axis.set_yticklabels([curve.case_name for curve in curves])
    axis.set_title("Ticket 09 Boundary-Found Status")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run the Ticket 09 boundary extraction experiment and write artifacts."""

    summary_rows, by_time, selected_rows, metadata = run_boundary_cases()
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(BY_TIME_CSV, by_time, BY_TIME_FIELDNAMES)
    write_csv(SELECTED_TIMES_CSV, selected_rows, SELECTED_TIME_FIELDNAMES)
    print(f"wrote {len(summary_rows)} rows to {SUMMARY_CSV}")
    print(f"wrote {len(by_time)} rows to {BY_TIME_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_TIMES_CSV}")
    print(
        "figures_created="
        f"{metadata['put_boundary_figure_created']},"
        f"{metadata['call_boundary_figure_created']},"
        f"{metadata['premium_profile_figure_created']},"
        f"{metadata['boundary_status_figure_created']}"
    )
    return summary_rows, by_time, selected_rows, metadata


def _create_boundary_curve_figure(
    curve: BoundaryCurve,
    path: Path,
    title: str,
    ylabel: str,
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    found_points = [point for point in curve.points if point.boundary_found]
    if not found_points:
        return False

    tau_values = [point.tau for point in found_points]
    boundary_values = [point.boundary_spot for point in found_points]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(tau_values, boundary_values, marker="o", markersize=2, linewidth=1.4)
    axis.axhline(curve.K, color="black", linestyle="--", linewidth=1, label="K")
    axis.set_xlabel("Time to maturity tau")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _find_curve(curves: list[BoundaryCurve], case_name: str) -> BoundaryCurve | None:
    for curve in curves:
        if curve.case_name == case_name:
            return curve
    return None


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _format_float(value: Any) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.12g}"


if __name__ == "__main__":
    main()
