"""Ticket 05: American put validation experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.black_scholes import european_put_price
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

VALIDATION_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_05_american_put_validation.csv"
)
SELECTED_SPOTS_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_05_american_put_selected_spots.csv"
)
FIGURE_PATH = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_05_american_put_value_vs_payoff.png"
)

SELECTED_MONEYNESS = (0.5, 0.8, 1.0, 1.2, 1.5)

DEFAULT_CASES = [
    {
        "case_name": "medium",
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
        "case_name": "fine",
        "option_type": "put",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
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
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_obstacle_gap",
    "max_obstacle_violation",
    "min_american_minus_european",
    "max_american_minus_european",
    "medium_to_fine_reference",
    "figure_created",
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
    "european_put",
    "american_put",
    "american_minus_european",
    "american_minus_payoff",
]


def run_validation_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run the default American put validation cases and build report rows."""

    results: list[tuple[str, AmericanCNPSORResult]] = []
    for case in cases:
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        result = american_crank_nicolson_psor_price(**solver_kwargs)
        results.append((case_name, result))

    validation_rows = [
        summarize_american_put_case(result) | {"case_name": case_name}
        for case_name, result in results
    ]
    selected_rows = [
        row
        for case_name, result in results
        for row in selected_spot_rows(result, case_name=case_name)
    ]

    comparison = _medium_to_fine_comparison(results)
    for row in validation_rows:
        row["medium_to_fine_reference"] = comparison
        row["figure_created"] = "False"

    fine_result = results[-1][1]
    figure_created = create_value_vs_payoff_figure(fine_result, FIGURE_PATH)
    validation_rows[-1]["figure_created"] = str(figure_created)

    metadata = {
        "validation_csv": str(VALIDATION_CSV),
        "selected_spots_csv": str(SELECTED_SPOTS_CSV),
        "figure_path": str(FIGURE_PATH),
        "figure_created": str(figure_created),
    }
    return validation_rows, selected_rows, metadata


def summarize_american_put_case(result: AmericanCNPSORResult) -> dict[str, str]:
    """Summarize obstacle, European comparison, and PSOR metadata."""

    european_values = np.asarray(
        european_put_price(
            result.spot_grid,
            K=result.K,
            T=result.T,
            r=result.r,
            q=result.q,
            sigma=result.sigma,
        ),
        dtype=float,
    )
    obstacle_gap = result.value_grid - result.payoff[np.newaxis, :]
    american_minus_european = result.values - european_values
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
        "all_psor_steps_converged": str(result.converged),
        "psor_step_count": str(len(result.psor_results)),
        "max_psor_iterations": str(max_iterations),
        "mean_psor_iterations": _format_float(mean_iterations),
        "max_final_update": _format_float(max_final_update),
        "min_obstacle_gap": _format_float(float(np.min(obstacle_gap))),
        "max_obstacle_violation": _format_float(result.max_obstacle_violation),
        "min_american_minus_european": _format_float(float(np.min(american_minus_european))),
        "max_american_minus_european": _format_float(float(np.max(american_minus_european))),
        "medium_to_fine_reference": "",
        "figure_created": "False",
    }


def selected_spot_rows(
    result: AmericanCNPSORResult,
    selected_moneyness: tuple[float, ...] = SELECTED_MONEYNESS,
    case_name: str = "case",
) -> list[dict[str, str]]:
    """Return selected spot comparison rows using nearest grid nodes."""

    european_values = np.asarray(
        european_put_price(
            result.spot_grid,
            K=result.K,
            T=result.T,
            r=result.r,
            q=result.q,
            sigma=result.sigma,
        ),
        dtype=float,
    )
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
                "european_put": _format_float(european),
                "american_put": _format_float(american),
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


def create_value_vs_payoff_figure(result: AmericanCNPSORResult, path: Path) -> bool:
    """Create the Ticket 05 American put validation figure if matplotlib imports."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    european_values = np.asarray(
        european_put_price(
            result.spot_grid,
            K=result.K,
            T=result.T,
            r=result.r,
            q=result.q,
            sigma=result.sigma,
        ),
        dtype=float,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = result.spot_grid / result.K
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x_values, result.values, label="American put value", linewidth=2.0)
    ax.plot(x_values, result.payoff, label="Payoff", linewidth=1.8, linestyle="--")
    ax.plot(x_values, european_values, label="European put value", linewidth=1.6, linestyle=":")
    ax.set_xlabel("Moneyness S/K")
    ax.set_ylabel("Option value")
    ax.set_title("Ticket 05 American Put Validation")
    ax.set_xlim(0.0, 2.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Run Ticket 05 validation and write CSV/figure artifacts."""

    validation_rows, selected_rows, metadata = run_validation_cases()
    write_csv(VALIDATION_CSV, validation_rows, VALIDATION_FIELDNAMES)
    write_csv(SELECTED_SPOTS_CSV, selected_rows, SELECTED_SPOTS_FIELDNAMES)
    print(f"wrote {len(validation_rows)} rows to {VALIDATION_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_SPOTS_CSV}")
    print(f"figure_created={metadata['figure_created']} path={FIGURE_PATH}")
    return validation_rows, selected_rows, metadata


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


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


if __name__ == "__main__":
    main()
