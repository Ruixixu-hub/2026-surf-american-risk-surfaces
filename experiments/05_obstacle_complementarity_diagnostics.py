"""Ticket 08: obstacle and complementarity diagnostics experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from american_risk_surfaces.diagnostics.lcp import (
    LCPDiagnostics,
    diagnose_lcp_result,
)
from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_08_lcp_diagnostics_summary.csv"
)
BY_STEP_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_08_lcp_diagnostics_by_step.csv"
)
OBSTACLE_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_08_lcp_obstacle_violation_by_step.png"
)
EQUATION_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_08_lcp_equation_violation_by_step.png"
)
COMPLEMENTARITY_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_08_lcp_complementarity_by_step.png"
)

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
    "value_gap_tolerance",
    "equation_gap_tolerance",
    "complementarity_tolerance",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_value_gap",
    "max_obstacle_violation",
    "min_equation_gap",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "mean_max_abs_complementarity_product",
    "max_exercise_like_node_count",
    "max_continuation_like_node_count",
    "max_ambiguous_node_count",
    "status",
    "obstacle_figure_created",
    "equation_figure_created",
    "complementarity_figure_created",
]

BY_STEP_FIELDNAMES = [
    "case_name",
    "option_type",
    "time_step",
    "tau",
    "psor_iterations",
    "psor_final_update",
    "min_value_gap",
    "max_obstacle_violation",
    "min_equation_gap",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "mean_abs_complementarity_product",
    "exercise_like_node_count",
    "continuation_like_node_count",
    "ambiguous_node_count",
]


def run_diagnostic_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run representative American CN/PSOR cases and summarize LCP diagnostics."""

    diagnostic_results = []
    for case in cases:
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        result = american_crank_nicolson_psor_price(**solver_kwargs)
        diagnostic_results.append(diagnose_lcp_result(result, case_name))

    summary_rows = [summary_row(diagnostics) for diagnostics in diagnostic_results]
    step_rows = [
        row
        for diagnostics in diagnostic_results
        for row in by_step_rows(diagnostics)
    ]

    obstacle_figure_created = create_obstacle_violation_figure(
        step_rows, OBSTACLE_FIGURE_PATH
    )
    equation_figure_created = create_equation_violation_figure(
        step_rows, EQUATION_FIGURE_PATH
    )
    complementarity_figure_created = create_complementarity_figure(
        step_rows, COMPLEMENTARITY_FIGURE_PATH
    )
    for row in summary_rows:
        row["obstacle_figure_created"] = str(obstacle_figure_created)
        row["equation_figure_created"] = str(equation_figure_created)
        row["complementarity_figure_created"] = str(complementarity_figure_created)

    metadata = {
        "summary_csv": str(SUMMARY_CSV),
        "by_step_csv": str(BY_STEP_CSV),
        "obstacle_figure_path": str(OBSTACLE_FIGURE_PATH),
        "equation_figure_path": str(EQUATION_FIGURE_PATH),
        "complementarity_figure_path": str(COMPLEMENTARITY_FIGURE_PATH),
        "obstacle_figure_created": str(obstacle_figure_created),
        "equation_figure_created": str(equation_figure_created),
        "complementarity_figure_created": str(complementarity_figure_created),
    }
    return summary_rows, step_rows, metadata


def summary_row(diagnostics: LCPDiagnostics) -> dict[str, str]:
    """Convert one diagnostic summary dataclass to a stable CSV row."""

    summary = diagnostics.summary
    return {
        "case_name": diagnostics.case_name,
        "option_type": summary.option_type,
        "K": _format_float(summary.K),
        "T": _format_float(summary.T),
        "r": _format_float(summary.r),
        "q": _format_float(summary.q),
        "sigma": _format_float(summary.sigma),
        "Smax": _format_float(summary.Smax),
        "M": str(summary.M),
        "N": str(summary.N),
        "value_gap_tolerance": _format_float(summary.value_gap_tolerance),
        "equation_gap_tolerance": _format_float(summary.equation_gap_tolerance),
        "complementarity_tolerance": _format_float(summary.complementarity_tolerance),
        "all_psor_steps_converged": str(summary.all_psor_steps_converged),
        "psor_step_count": str(summary.psor_step_count),
        "max_psor_iterations": str(summary.max_psor_iterations),
        "mean_psor_iterations": _format_float(summary.mean_psor_iterations),
        "max_final_update": _format_float(summary.max_final_update),
        "min_value_gap": _format_float(summary.min_value_gap),
        "max_obstacle_violation": _format_float(summary.max_obstacle_violation),
        "min_equation_gap": _format_float(summary.min_equation_gap),
        "max_equation_violation": _format_float(summary.max_equation_violation),
        "max_abs_complementarity_product": _format_float(
            summary.max_abs_complementarity_product
        ),
        "mean_max_abs_complementarity_product": _format_float(
            summary.mean_max_abs_complementarity_product
        ),
        "max_exercise_like_node_count": str(summary.max_exercise_like_node_count),
        "max_continuation_like_node_count": str(summary.max_continuation_like_node_count),
        "max_ambiguous_node_count": str(summary.max_ambiguous_node_count),
        "status": summary.status,
        "obstacle_figure_created": "False",
        "equation_figure_created": "False",
        "complementarity_figure_created": "False",
    }


def by_step_rows(diagnostics: LCPDiagnostics) -> list[dict[str, str]]:
    """Convert one diagnostic result bundle to stable by-step CSV rows."""

    return [
        {
            "case_name": diagnostics.case_name,
            "option_type": diagnostics.summary.option_type,
            "time_step": str(row.step_index),
            "tau": _format_float(row.tau),
            "psor_iterations": str(row.psor_iterations),
            "psor_final_update": _format_float(row.psor_final_update),
            "min_value_gap": _format_float(row.min_value_gap),
            "max_obstacle_violation": _format_float(row.max_obstacle_violation),
            "min_equation_gap": _format_float(row.min_equation_gap),
            "max_equation_violation": _format_float(row.max_equation_violation),
            "max_abs_complementarity_product": _format_float(
                row.max_abs_complementarity_product
            ),
            "mean_abs_complementarity_product": _format_float(
                row.mean_abs_complementarity_product
            ),
            "exercise_like_node_count": str(row.exercise_like_node_count),
            "continuation_like_node_count": str(row.continuation_like_node_count),
            "ambiguous_node_count": str(row.ambiguous_node_count),
        }
        for row in diagnostics.step_rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_obstacle_violation_figure(by_step_rows: list[dict[str, str]], path: Path) -> bool:
    """Plot maximum obstacle violation by time step for each diagnostic case."""

    return _create_metric_figure(
        by_step_rows,
        path,
        metric="max_obstacle_violation",
        ylabel="Max obstacle violation",
        title="Ticket 08 Obstacle Violation by Time Step",
    )


def create_equation_violation_figure(by_step_rows: list[dict[str, str]], path: Path) -> bool:
    """Plot maximum equation violation by time step for each diagnostic case."""

    return _create_metric_figure(
        by_step_rows,
        path,
        metric="max_equation_violation",
        ylabel="Max equation violation",
        title="Ticket 08 Equation Violation by Time Step",
    )


def create_complementarity_figure(by_step_rows: list[dict[str, str]], path: Path) -> bool:
    """Plot maximum absolute complementarity product by time step."""

    return _create_metric_figure(
        by_step_rows,
        path,
        metric="max_abs_complementarity_product",
        ylabel="Max absolute complementarity product",
        title="Ticket 08 Complementarity Product by Time Step",
    )


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run Ticket 08 diagnostics and write CSV/figure artifacts."""

    summary_rows, step_rows, metadata = run_diagnostic_cases()
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(BY_STEP_CSV, step_rows, BY_STEP_FIELDNAMES)
    print(f"wrote {len(summary_rows)} rows to {SUMMARY_CSV}")
    print(f"wrote {len(step_rows)} rows to {BY_STEP_CSV}")
    print(
        "obstacle_figure_created="
        f"{metadata['obstacle_figure_created']} path={OBSTACLE_FIGURE_PATH}"
    )
    print(
        "equation_figure_created="
        f"{metadata['equation_figure_created']} path={EQUATION_FIGURE_PATH}"
    )
    print(
        "complementarity_figure_created="
        f"{metadata['complementarity_figure_created']} path={COMPLEMENTARITY_FIGURE_PATH}"
    )
    return summary_rows, step_rows, metadata


def _create_metric_figure(
    rows: list[dict[str, str]],
    path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> bool:
    plot_modules = _matplotlib_modules()
    if plot_modules is None:
        return False
    _, plt = plot_modules

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for case_name in _case_names(rows):
        case_rows = [row for row in rows if row["case_name"] == case_name]
        x_values = [int(row["time_step"]) for row in case_rows]
        y_values = [float(row[metric]) for row in case_rows]
        ax.plot(x_values, y_values, label=case_name, linewidth=1.8)

    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _case_names(rows: list[dict[str, str]]) -> list[str]:
    names = []
    for row in rows:
        if row["case_name"] not in names:
            names.append(row["case_name"])
    return names


def _matplotlib_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return matplotlib, plt


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


if __name__ == "__main__":
    main()
