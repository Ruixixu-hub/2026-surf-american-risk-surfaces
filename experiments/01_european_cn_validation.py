"""Ticket 03: generate European Crank-Nicolson validation results."""

from __future__ import annotations

import csv
from pathlib import Path

from american_risk_surfaces.solvers.cn import EuropeanCNResult, european_crank_nicolson_price


OUTPUT_PATH = Path("results/01_solver_validation/tables/ticket_03_european_cn_validation.csv")

DEFAULT_CASES = [
    {
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
    {
        "option_type": "call",
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
        "option_type": "call",
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


def main() -> None:
    results = [european_crank_nicolson_price(**case) for case in DEFAULT_CASES]
    rows = [_summary_row(result) for result in results]
    _write_rows(OUTPUT_PATH, rows)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


def _summary_row(result: EuropeanCNResult) -> dict[str, str]:
    metrics = result.metrics
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
        "target_lower_moneyness": _format_float(metrics.target_lower),
        "target_upper_moneyness": _format_float(metrics.target_upper),
        "max_abs_error": _format_float(metrics.max_abs_error),
        "rmse": _format_float(metrics.rmse),
        "max_error_spot": _format_float(metrics.max_error_spot),
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "max_abs_error",
        "rmse",
        "max_error_spot",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


if __name__ == "__main__":
    main()
