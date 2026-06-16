"""Ticket 07: dividend-paying American call validation experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.black_scholes import european_call_price
from american_risk_surfaces.solvers.cn import target_region_error_metrics as _target_region_error_metrics
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

VALIDATION_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_07_dividend_american_call_validation.csv"
)
SELECTED_SPOTS_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/"
    "ticket_07_dividend_american_call_selected_spots.csv"
)
VALUE_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_07_dividend_american_call_value_comparison.png"
)
DIFFERENCE_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_07_dividend_american_call_american_minus_european.png"
)
PSOR_FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/"
    "ticket_07_dividend_american_call_psor_iterations.png"
)

TARGET_LOWER_MONEYNESS = 0.4
TARGET_UPPER_MONEYNESS = 1.8
SELECTED_MONEYNESS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)

DEFAULT_CASES = [
    {
        "case_name": "medium",
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
        "case_name": "fine",
        "option_type": "call",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.08,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 180,
        "N": 180,
    },
]

VALIDATION_FIELDNAMES = [
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
    "target_lower_moneyness",
    "target_upper_moneyness",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_obstacle_gap",
    "max_obstacle_violation",
    "min_american_minus_european",
    "max_american_minus_european",
    "max_abs_american_european_difference",
    "rmse_american_european_difference",
    "max_difference_spot",
    "positive_american_minus_european_node_count",
    "medium_to_fine_selected_spot_max_abs_diff",
    "value_figure_created",
    "difference_figure_created",
    "psor_figure_created",
]

SELECTED_SPOTS_FIELDNAMES = [
    "case_name",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "M",
    "N",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "payoff",
    "european_call",
    "american_call",
    "american_minus_european",
    "american_minus_payoff",
]


def run_validation_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run the default positive-dividend American call validation cases."""

    results: list[tuple[str, AmericanCNPSORResult]] = []
    for case in cases:
        _validate_positive_dividend_call_case(case)
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        result = american_crank_nicolson_psor_price(**solver_kwargs)
        results.append((case_name, result))

    validation_rows = [
        summarize_dividend_call_case(result) | {"case_name": case_name}
        for case_name, result in results
    ]
    selected_rows = [
        row
        for case_name, result in results
        for row in selected_spot_rows(result, case_name=case_name)
    ]

    comparison = _medium_to_fine_comparison(results)
    for row in validation_rows:
        row["medium_to_fine_selected_spot_max_abs_diff"] = comparison
        row["value_figure_created"] = "False"
        row["difference_figure_created"] = "False"
        row["psor_figure_created"] = "False"

    fine_result = results[-1][1]
    value_figure_created = create_value_comparison_figure(fine_result, VALUE_FIGURE_PATH)
    difference_figure_created = create_american_minus_european_figure(
        fine_result, DIFFERENCE_FIGURE_PATH
    )
    psor_figure_created = create_psor_iterations_figure(fine_result, PSOR_FIGURE_PATH)
    validation_rows[-1]["value_figure_created"] = str(value_figure_created)
    validation_rows[-1]["difference_figure_created"] = str(difference_figure_created)
    validation_rows[-1]["psor_figure_created"] = str(psor_figure_created)

    metadata = {
        "validation_csv": str(VALIDATION_CSV),
        "selected_spots_csv": str(SELECTED_SPOTS_CSV),
        "value_figure_path": str(VALUE_FIGURE_PATH),
        "difference_figure_path": str(DIFFERENCE_FIGURE_PATH),
        "psor_figure_path": str(PSOR_FIGURE_PATH),
        "value_figure_created": str(value_figure_created),
        "difference_figure_created": str(difference_figure_created),
        "psor_figure_created": str(psor_figure_created),
    }
    return validation_rows, selected_rows, metadata


def summarize_dividend_call_case(result: AmericanCNPSORResult) -> dict[str, str]:
    """Summarize dividend-call premium, obstacle, and PSOR metadata for one case."""

    _validate_result_is_positive_dividend_call(result)
    european_values = _european_call_values(result)
    american_minus_european = result.values - european_values
    target_mask = _target_moneyness_mask(result.spot_grid, result.K)
    target_difference = american_minus_european[target_mask]
    metrics = _target_region_error_metrics(
        result.spot_grid,
        result.values,
        european_values,
        K=result.K,
        lower_moneyness=TARGET_LOWER_MONEYNESS,
        upper_moneyness=TARGET_UPPER_MONEYNESS,
    )
    obstacle_gap = result.value_grid - result.payoff[np.newaxis, :]
    iterations = np.array([step.iterations for step in result.psor_results], dtype=float)
    final_updates = np.array([step.final_update for step in result.psor_results], dtype=float)

    if len(iterations) == 0:
        max_iterations = 0
        mean_iterations = 0.0
        max_final_update = 0.0
    else:
        max_iterations = int(np.max(iterations))
        mean_iterations = float(np.mean(iterations))
        max_final_update = float(np.max(final_updates))

    return {
        "option_type": result.option_type,
        "K": _format_float(result.K),
        "T": _format_float(result.T),
        "r": _format_float(result.r),
        "q": _format_float(result.q),
        "sigma": _format_float(result.sigma),
        "Smax": _format_float(result.Smax),
        "M": str(result.M),
        "N": str(result.N),
        "target_lower_moneyness": _format_float(TARGET_LOWER_MONEYNESS),
        "target_upper_moneyness": _format_float(TARGET_UPPER_MONEYNESS),
        "all_psor_steps_converged": str(result.converged),
        "psor_step_count": str(len(result.psor_results)),
        "max_psor_iterations": str(max_iterations),
        "mean_psor_iterations": _format_float(mean_iterations),
        "max_final_update": _format_float(max_final_update),
        "min_obstacle_gap": _format_float(float(np.min(obstacle_gap))),
        "max_obstacle_violation": _format_float(result.max_obstacle_violation),
        "min_american_minus_european": _format_float(float(np.min(target_difference))),
        "max_american_minus_european": _format_float(float(np.max(target_difference))),
        "max_abs_american_european_difference": _format_float(metrics.max_abs_error),
        "rmse_american_european_difference": _format_float(metrics.rmse),
        "max_difference_spot": _format_float(metrics.max_error_spot),
        "positive_american_minus_european_node_count": str(
            int(np.count_nonzero(target_difference > 1e-8))
        ),
        "medium_to_fine_selected_spot_max_abs_diff": "",
        "value_figure_created": "False",
        "difference_figure_created": "False",
        "psor_figure_created": "False",
    }


def selected_spot_rows(
    result: AmericanCNPSORResult,
    selected_moneyness: tuple[float, ...] = SELECTED_MONEYNESS,
    case_name: str = "case",
) -> list[dict[str, str]]:
    """Return selected spot comparison rows using nearest grid nodes."""

    _validate_result_is_positive_dividend_call(result)
    european_values = _european_call_values(result)
    rows = []
    for moneyness in selected_moneyness:
        target_spot = float(moneyness) * result.K
        index = nearest_spot_index(result.spot_grid, target_spot)
        spot = float(result.spot_grid[index])
        payoff = float(result.payoff[index])
        european = float(european_values[index])
        american = float(result.values[index])
        rows.append(
            {
                "case_name": case_name,
                "K": _format_float(result.K),
                "T": _format_float(result.T),
                "r": _format_float(result.r),
                "q": _format_float(result.q),
                "sigma": _format_float(result.sigma),
                "M": str(result.M),
                "N": str(result.N),
                "target_moneyness": _format_float(float(moneyness)),
                "nearest_spot": _format_float(spot),
                "actual_moneyness": _format_float(spot / result.K),
                "payoff": _format_float(payoff),
                "european_call": _format_float(european),
                "american_call": _format_float(american),
                "american_minus_european": _format_float(american - european),
                "american_minus_payoff": _format_float(american - payoff),
            }
        )
    return rows


def nearest_spot_index(spot_grid: Any, target_spot: float) -> int:
    """Return the nearest grid index to the requested spot."""

    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1 or len(spots) == 0:
        raise ValueError("spot_grid must be a nonempty one-dimensional array.")
    return int(np.argmin(np.abs(spots - float(target_spot))))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_value_comparison_figure(result: AmericanCNPSORResult, path: Path) -> bool:
    """Create the Ticket 07 American, European, and payoff comparison figure."""

    plot_modules = _matplotlib_modules()
    if plot_modules is None:
        return False
    _, plt = plot_modules

    _validate_result_is_positive_dividend_call(result)
    european_values = _european_call_values(result)

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = result.spot_grid / result.K
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x_values, result.values, label="American call value", linewidth=2.0)
    ax.plot(x_values, european_values, label="European call value", linewidth=1.8, linestyle=":")
    ax.plot(x_values, result.payoff, label="Payoff", linewidth=1.6, linestyle="--")
    ax.set_xlabel("Moneyness S/K")
    ax.set_ylabel("Option value")
    ax.set_title("Ticket 07 Dividend-Paying American Call Validation")
    ax.set_xlim(0.0, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_american_minus_european_figure(
    result: AmericanCNPSORResult, path: Path
) -> bool:
    """Create the Ticket 07 American-minus-European dividend-call value figure."""

    plot_modules = _matplotlib_modules()
    if plot_modules is None:
        return False
    _, plt = plot_modules

    _validate_result_is_positive_dividend_call(result)
    european_values = _european_call_values(result)
    x_values = result.spot_grid / result.K
    differences = result.values - european_values

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_values, differences, label="American call - European call", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="Zero difference")
    ax.set_xlabel("Moneyness S/K")
    ax.set_ylabel("Value difference")
    ax.set_title("Ticket 07 Dividend-Call American Minus European")
    ax.set_xlim(0.0, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_psor_iterations_figure(result: AmericanCNPSORResult, path: Path) -> bool:
    """Create the Ticket 07 PSOR iteration-count summary figure."""

    plot_modules = _matplotlib_modules()
    if plot_modules is None:
        return False
    _, plt = plot_modules

    _validate_result_is_positive_dividend_call(result)
    iterations = np.array([step.iterations for step in result.psor_results], dtype=float)
    if len(iterations) == 0:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    step_numbers = np.arange(1, len(iterations) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(step_numbers, iterations, linewidth=1.8, color="#1f77b4")
    ax.set_xlabel("Time step")
    ax.set_ylabel("PSOR iterations")
    ax.set_title("Ticket 07 PSOR Iterations by Time Step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run Ticket 07 validation and write CSV/figure artifacts."""

    validation_rows, selected_rows, metadata = run_validation_cases()
    write_csv(VALIDATION_CSV, validation_rows, VALIDATION_FIELDNAMES)
    write_csv(SELECTED_SPOTS_CSV, selected_rows, SELECTED_SPOTS_FIELDNAMES)
    print(f"wrote {len(validation_rows)} rows to {VALIDATION_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_SPOTS_CSV}")
    print(
        "value_figure_created="
        f"{metadata['value_figure_created']} path={VALUE_FIGURE_PATH}"
    )
    print(
        "difference_figure_created="
        f"{metadata['difference_figure_created']} path={DIFFERENCE_FIGURE_PATH}"
    )
    print(
        "psor_figure_created="
        f"{metadata['psor_figure_created']} path={PSOR_FIGURE_PATH}"
    )
    return validation_rows, selected_rows, metadata


def _validate_positive_dividend_call_case(case: dict[str, Any]) -> None:
    if str(case.get("option_type", "")).lower() != "call":
        raise ValueError("Ticket 07 validation cases must use option_type='call'.")
    if float(case.get("q", float("nan"))) <= 0.0:
        raise ValueError("Ticket 07 validation cases must use q > 0.")


def _validate_result_is_positive_dividend_call(result: AmericanCNPSORResult) -> None:
    if result.option_type != "call":
        raise ValueError("Ticket 07 summaries require a call result.")
    if result.q <= 0.0:
        raise ValueError("Ticket 07 summaries require q > 0.")


def _european_call_values(result: AmericanCNPSORResult) -> np.ndarray:
    return np.asarray(
        european_call_price(
            result.spot_grid,
            K=result.K,
            T=result.T,
            r=result.r,
            q=result.q,
            sigma=result.sigma,
        ),
        dtype=float,
    )


def _target_moneyness_mask(spot_grid: np.ndarray, K: float) -> np.ndarray:
    moneyness = spot_grid / float(K)
    return (moneyness >= TARGET_LOWER_MONEYNESS) & (moneyness <= TARGET_UPPER_MONEYNESS)


def _medium_to_fine_comparison(results: list[tuple[str, AmericanCNPSORResult]]) -> str:
    if len(results) < 2:
        return ""

    medium = results[0][1]
    fine = results[-1][1]
    differences = []
    for moneyness in SELECTED_MONEYNESS:
        target_spot = float(moneyness) * medium.K
        medium_index = nearest_spot_index(medium.spot_grid, target_spot)
        fine_index = nearest_spot_index(fine.spot_grid, target_spot)
        differences.append(abs(float(medium.values[medium_index] - fine.values[fine_index])))
    return _format_float(float(np.max(differences)))


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
