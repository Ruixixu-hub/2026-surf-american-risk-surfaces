"""Ticket 10: Delta and Gamma diagnostics experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.greeks import (
    GreekDiagnostics,
    diagnose_greek_result,
    greek_by_time_rows,
    selected_greek_profile_rows,
)
from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10_greek_diagnostics_summary.csv"
)
SELECTED_PROFILES_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10_greek_selected_profiles.csv"
)
BY_TIME_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10_greek_by_time.csv"
)
PUT_PROFILE_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_10_american_put_delta_gamma_profiles.png"
)
CALL_PROFILE_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_10_dividend_call_delta_gamma_profiles.png"
)
GAMMA_BOUNDARY_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10_gamma_boundary_diagnostic.png"
)
GAMMA_BY_TIME_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10_gamma_by_time.png"
)

SELECTED_TAU_FRACTIONS = (0.01, 0.5, 1.0)
SELECTED_MONEYNESS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)

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
    "kink_band_steps",
    "boundary_band_steps",
    "delta_bound_tolerance",
    "gamma_negative_tolerance",
    "finite_delta_count",
    "finite_gamma_count",
    "nonfinite_delta_count",
    "nonfinite_gamma_count",
    "min_delta",
    "max_delta",
    "min_gamma",
    "max_gamma",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "strict_delta_lower_violation_count",
    "strict_delta_upper_violation_count",
    "strict_negative_gamma_count",
    "status",
    "put_profile_figure_created",
    "call_profile_figure_created",
    "gamma_boundary_figure_created",
    "gamma_by_time_figure_created",
]

SELECTED_FIELDNAMES = [
    "case_name",
    "option_type",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "delta",
    "gamma",
    "boundary_near",
    "kink_near",
    "maturity_row",
    "strict_interior",
]

BY_TIME_FIELDNAMES = [
    "case_name",
    "option_type",
    "time_index",
    "tau",
    "finite_delta_count",
    "finite_gamma_count",
    "boundary_near_node_count",
    "kink_near_node_count",
    "strict_interior_count",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "min_delta_strict",
    "max_delta_strict",
    "strict_delta_lower_violation_count",
    "strict_delta_upper_violation_count",
    "strict_negative_gamma_count",
    "warning_flag",
]


def run_greek_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run representative American cases and compute Delta/Gamma diagnostics."""

    diagnostics_list: list[GreekDiagnostics] = []
    for case in cases:
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        result = american_crank_nicolson_psor_price(**solver_kwargs)
        diagnostics_list.append(diagnose_greek_result(result, case_name))

    put_profile_figure_created = create_put_delta_gamma_profile_figure(
        diagnostics_list, PUT_PROFILE_FIGURE
    )
    call_profile_figure_created = create_call_delta_gamma_profile_figure(
        diagnostics_list, CALL_PROFILE_FIGURE
    )
    gamma_boundary_figure_created = create_gamma_boundary_diagnostic_figure(
        diagnostics_list, GAMMA_BOUNDARY_FIGURE
    )
    gamma_by_time_figure_created = create_gamma_by_time_figure(
        diagnostics_list, GAMMA_BY_TIME_FIGURE
    )
    figure_flags = {
        "put_profile_figure_created": str(put_profile_figure_created),
        "call_profile_figure_created": str(call_profile_figure_created),
        "gamma_boundary_figure_created": str(gamma_boundary_figure_created),
        "gamma_by_time_figure_created": str(gamma_by_time_figure_created),
    }

    summary_rows = [
        summary_row(diagnostics) | figure_flags for diagnostics in diagnostics_list
    ]
    selected_rows = [
        row
        for diagnostics in diagnostics_list
        for row in selected_profile_rows(diagnostics)
    ]
    time_rows = [
        row
        for diagnostics in diagnostics_list
        for row in by_time_rows(diagnostics)
    ]
    metadata = {
        "summary_csv": str(SUMMARY_CSV),
        "selected_profiles_csv": str(SELECTED_PROFILES_CSV),
        "by_time_csv": str(BY_TIME_CSV),
        "put_profile_figure_path": str(PUT_PROFILE_FIGURE),
        "call_profile_figure_path": str(CALL_PROFILE_FIGURE),
        "gamma_boundary_figure_path": str(GAMMA_BOUNDARY_FIGURE),
        "gamma_by_time_figure_path": str(GAMMA_BY_TIME_FIGURE),
        "put_profile_figure_created": str(put_profile_figure_created),
        "call_profile_figure_created": str(call_profile_figure_created),
        "gamma_boundary_figure_created": str(gamma_boundary_figure_created),
        "gamma_by_time_figure_created": str(gamma_by_time_figure_created),
    }
    return summary_rows, selected_rows, time_rows, metadata


def summary_row(diagnostics: GreekDiagnostics) -> dict[str, str]:
    """Convert one Greek diagnostic summary to a stable CSV row."""

    summary = diagnostics.summary
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
        "kink_band_steps": str(summary.kink_band_steps),
        "boundary_band_steps": str(summary.boundary_band_steps),
        "delta_bound_tolerance": _format_float(summary.delta_bound_tolerance),
        "gamma_negative_tolerance": _format_float(summary.gamma_negative_tolerance),
        "finite_delta_count": str(summary.finite_delta_count),
        "finite_gamma_count": str(summary.finite_gamma_count),
        "nonfinite_delta_count": str(summary.nonfinite_delta_count),
        "nonfinite_gamma_count": str(summary.nonfinite_gamma_count),
        "min_delta": _format_float(summary.min_delta),
        "max_delta": _format_float(summary.max_delta),
        "min_gamma": _format_float(summary.min_gamma),
        "max_gamma": _format_float(summary.max_gamma),
        "max_abs_gamma": _format_float(summary.max_abs_gamma),
        "max_abs_gamma_away_from_boundary": _format_float(
            summary.max_abs_gamma_away_from_boundary
        ),
        "max_abs_gamma_strict": _format_float(summary.max_abs_gamma_strict),
        "boundary_near_node_count": str(summary.boundary_near_node_count),
        "kink_near_node_count": str(summary.kink_near_node_count),
        "maturity_masked_node_count": str(summary.maturity_masked_node_count),
        "strict_interior_node_count": str(summary.strict_interior_node_count),
        "strict_delta_lower_violation_count": str(
            summary.strict_delta_lower_violation_count
        ),
        "strict_delta_upper_violation_count": str(
            summary.strict_delta_upper_violation_count
        ),
        "strict_negative_gamma_count": str(summary.strict_negative_gamma_count),
        "status": summary.status,
        "put_profile_figure_created": "False",
        "call_profile_figure_created": "False",
        "gamma_boundary_figure_created": "False",
        "gamma_by_time_figure_created": "False",
    }


def selected_profile_rows(diagnostics: GreekDiagnostics) -> list[dict[str, str]]:
    """Convert selected Delta/Gamma profile rows to stable CSV strings."""

    raw_rows = selected_greek_profile_rows(
        diagnostics,
        selected_tau_fractions=SELECTED_TAU_FRACTIONS,
        selected_moneyness=SELECTED_MONEYNESS,
    )
    return [
        {
            "case_name": str(row["case_name"]),
            "option_type": str(row["option_type"]),
            "target_tau_fraction": _format_float(row["target_tau_fraction"]),
            "target_tau": _format_float(row["target_tau"]),
            "nearest_tau": _format_float(row["nearest_tau"]),
            "time_index": str(row["time_index"]),
            "target_moneyness": _format_float(row["target_moneyness"]),
            "nearest_spot": _format_float(row["nearest_spot"]),
            "actual_moneyness": _format_float(row["actual_moneyness"]),
            "value": _format_float(row["value"]),
            "delta": _format_float(row["delta"]),
            "gamma": _format_float(row["gamma"]),
            "boundary_near": str(row["boundary_near"]),
            "kink_near": str(row["kink_near"]),
            "maturity_row": str(row["maturity_row"]),
            "strict_interior": str(row["strict_interior"]),
        }
        for row in raw_rows
    ]


def by_time_rows(diagnostics: GreekDiagnostics) -> list[dict[str, str]]:
    """Convert by-time Greek diagnostics to stable CSV strings."""

    return [
        {
            "case_name": diagnostics.case_name,
            "option_type": diagnostics.option_type,
            "time_index": str(row.time_index),
            "tau": _format_float(row.tau),
            "finite_delta_count": str(row.finite_delta_count),
            "finite_gamma_count": str(row.finite_gamma_count),
            "boundary_near_node_count": str(row.boundary_near_node_count),
            "kink_near_node_count": str(row.kink_near_node_count),
            "strict_interior_count": str(row.strict_interior_count),
            "max_abs_gamma": _format_float(row.max_abs_gamma),
            "max_abs_gamma_away_from_boundary": _format_float(
                row.max_abs_gamma_away_from_boundary
            ),
            "max_abs_gamma_strict": _format_float(row.max_abs_gamma_strict),
            "min_delta_strict": _format_float(row.min_delta_strict),
            "max_delta_strict": _format_float(row.max_delta_strict),
            "strict_delta_lower_violation_count": str(
                row.strict_delta_lower_violation_count
            ),
            "strict_delta_upper_violation_count": str(
                row.strict_delta_upper_violation_count
            ),
            "strict_negative_gamma_count": str(row.strict_negative_gamma_count),
            "warning_flag": row.warning_flag,
        }
        for row in greek_by_time_rows(diagnostics)
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_put_delta_gamma_profile_figure(
    diagnostics_list: list[GreekDiagnostics],
    path: Path,
) -> bool:
    """Plot American put Delta and Gamma profiles at selected tau values."""

    diagnostics = _find_diagnostics(diagnostics_list, "american_put_medium")
    if diagnostics is None:
        return False
    return _create_delta_gamma_profile_figure(
        diagnostics,
        path,
        title="Ticket 10 American Put Delta/Gamma Profiles",
    )


def create_call_delta_gamma_profile_figure(
    diagnostics_list: list[GreekDiagnostics],
    path: Path,
) -> bool:
    """Plot dividend-paying American call Delta and Gamma profiles."""

    diagnostics = _find_diagnostics(diagnostics_list, "dividend_call_medium")
    if diagnostics is None:
        return False
    return _create_delta_gamma_profile_figure(
        diagnostics,
        path,
        title="Ticket 10 Dividend Call Delta/Gamma Profiles",
    )


def create_gamma_boundary_diagnostic_figure(
    diagnostics_list: list[GreekDiagnostics],
    path: Path,
) -> bool:
    """Plot near-boundary versus away-from-boundary Gamma magnitudes over tau."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for axis, diagnostics in zip(axes, diagnostics_list):
        tau = diagnostics.tau_grid
        near_values = []
        away_values = []
        for index in range(len(tau)):
            gamma_row = diagnostics.arrays.gamma[index]
            finite = diagnostics.arrays.finite_gamma_mask[index]
            near = finite & diagnostics.masks.boundary_near[index]
            away = finite & ~diagnostics.masks.boundary_near[index]
            near_values.append(_max_abs_or_nan(gamma_row[near]))
            away_values.append(_max_abs_or_nan(gamma_row[away]))
        axis.plot(tau, near_values, label="near extracted boundary", linewidth=1.3)
        axis.plot(tau, away_values, label="away from boundary", linewidth=1.3)
        axis.set_title(diagnostics.case_name)
        axis.set_xlabel("Time to maturity tau")
        axis.set_ylabel("Max |Gamma|")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_gamma_by_time_figure(
    diagnostics_list: list[GreekDiagnostics],
    path: Path,
) -> bool:
    """Plot full versus strict-mask maximum absolute Gamma over time."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    for diagnostics in diagnostics_list:
        tau = [row.tau for row in diagnostics.by_time_rows]
        full_gamma = [row.max_abs_gamma for row in diagnostics.by_time_rows]
        strict_gamma = [row.max_abs_gamma_strict for row in diagnostics.by_time_rows]
        axis.plot(tau, full_gamma, linewidth=1.2, label=f"{diagnostics.case_name} full")
        axis.plot(
            tau,
            strict_gamma,
            linestyle="--",
            linewidth=1.2,
            label=f"{diagnostics.case_name} strict",
        )
    axis.set_xlabel("Time to maturity tau")
    axis.set_ylabel("Max |Gamma|")
    axis.set_title("Ticket 10 Gamma Diagnostics Over Time")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run the Ticket 10 Greek diagnostics experiment and write artifacts."""

    summary_rows, selected_rows, time_rows, metadata = run_greek_cases()
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(SELECTED_PROFILES_CSV, selected_rows, SELECTED_FIELDNAMES)
    write_csv(BY_TIME_CSV, time_rows, BY_TIME_FIELDNAMES)
    print(f"wrote {len(summary_rows)} rows to {SUMMARY_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_PROFILES_CSV}")
    print(f"wrote {len(time_rows)} rows to {BY_TIME_CSV}")
    print(
        "figures_created="
        f"{metadata['put_profile_figure_created']},"
        f"{metadata['call_profile_figure_created']},"
        f"{metadata['gamma_boundary_figure_created']},"
        f"{metadata['gamma_by_time_figure_created']}"
    )
    return summary_rows, selected_rows, time_rows, metadata


def _create_delta_gamma_profile_figure(
    diagnostics: GreekDiagnostics,
    path: Path,
    title: str,
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    moneyness = diagnostics.spot_grid / diagnostics.summary.K
    for fraction in SELECTED_TAU_FRACTIONS:
        target_tau = diagnostics.summary.T * fraction
        time_index = int(np.argmin(np.abs(diagnostics.tau_grid - target_tau)))
        label = f"tau={diagnostics.tau_grid[time_index]:.3f}"
        axes[0].plot(moneyness, diagnostics.arrays.delta[time_index], label=label)
        axes[1].plot(moneyness, diagnostics.arrays.gamma[time_index], label=label)
    axes[0].set_ylabel("Delta")
    axes[1].set_ylabel("Gamma")
    axes[1].set_xlabel("S / K")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _find_diagnostics(
    diagnostics_list: list[GreekDiagnostics],
    case_name: str,
) -> GreekDiagnostics | None:
    for diagnostics in diagnostics_list:
        if diagnostics.case_name == case_name:
            return diagnostics
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


def _max_abs_or_nan(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else float("nan")


if __name__ == "__main__":
    main()
