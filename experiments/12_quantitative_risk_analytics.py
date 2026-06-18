"""Quantitative Risk Analytics: small controlled expansion after Pilot 01 stress maps."""

from __future__ import annotations

import os
from pathlib import Path

from american_risk_surfaces.downstream.risk_analytics import (
    BOUNDARY_METRIC_FIELDNAMES,
    GREEK_METRIC_FIELDNAMES,
    LCP_METRIC_FIELDNAMES,
    OUTPUT_MANIFEST_FIELDNAMES,
    RUN_SUMMARY_FIELDNAMES,
    RUNTIME_ITERATION_FIELDNAMES,
    RiskAnalyticsCase,
    boundary_metric_rows,
    create_call_boundary_vs_dividend_yield_figure,
    create_call_q_boundary_curves_figure,
    create_call_q_sigma_boundary_heatmap_figure,
    create_gamma_concentration_figure,
    create_lcp_stability_figure,
    create_psor_runtime_iterations_figure,
    create_put_boundary_vs_volatility_figure,
    create_put_vol_boundary_curves_figure,
    greek_metric_row,
    lcp_metric_row,
    manifest_row,
    risk_analytics_cases,
    run_risk_analytics_cases,
    run_summary_row,
    runtime_iteration_row,
    write_csv,
    _load_pyplot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "03_quantitative_risk_analytics" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "03_quantitative_risk_analytics" / "figures"

RUN_SUMMARY_CSV = DEFAULT_TABLE_DIR / "risk_analytics_run_summary.csv"
BOUNDARY_METRICS_CSV = DEFAULT_TABLE_DIR / "risk_analytics_boundary_metrics.csv"
GREEK_METRICS_CSV = DEFAULT_TABLE_DIR / "risk_analytics_greek_metrics.csv"
LCP_METRICS_CSV = DEFAULT_TABLE_DIR / "risk_analytics_lcp_metrics.csv"
RUNTIME_ITERATIONS_CSV = DEFAULT_TABLE_DIR / "risk_analytics_runtime_iterations.csv"
OUTPUT_MANIFEST_CSV = DEFAULT_TABLE_DIR / "risk_analytics_output_manifest.csv"

PUT_BOUNDARY_VS_VOL_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_put_boundary_vs_volatility.png"
CALL_BOUNDARY_VS_Q_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_call_boundary_vs_dividend_yield.png"
PUT_VOL_BOUNDARY_CURVES_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_put_vol_boundary_curves.png"
CALL_Q_BOUNDARY_CURVES_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_call_q_boundary_curves.png"
GAMMA_CONCENTRATION_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_gamma_concentration.png"
PSOR_RUNTIME_ITERATIONS_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_psor_runtime_iterations.png"
LCP_STABILITY_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_lcp_stability.png"
CALL_Q_SIGMA_HEATMAP_FIGURE = DEFAULT_FIGURE_DIR / "risk_analytics_call_q_sigma_boundary_heatmap.png"


def main(
    cases: tuple[RiskAnalyticsCase, ...] | None = None,
    table_dir: Path = DEFAULT_TABLE_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    create_figures: bool = True,
    include_q_sigma_heatmap: bool = True,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    """Run the controlled quantitative risk-analytics experiment."""

    selected_cases = (
        cases
        if cases is not None
        else risk_analytics_cases(include_q_sigma_heatmap=include_q_sigma_heatmap)
    )
    runs = run_risk_analytics_cases(selected_cases, cache_duplicate_parameters=True)

    run_rows = [run_summary_row(run) for run in runs]
    boundary_rows = [row for run in runs for row in boundary_metric_rows(run)]
    greek_rows = [greek_metric_row(run) for run in runs]
    lcp_rows = [lcp_metric_row(run) for run in runs]
    runtime_rows = [runtime_iteration_row(run) for run in runs]

    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    run_csv = table_dir / "risk_analytics_run_summary.csv"
    boundary_csv = table_dir / "risk_analytics_boundary_metrics.csv"
    greek_csv = table_dir / "risk_analytics_greek_metrics.csv"
    lcp_csv = table_dir / "risk_analytics_lcp_metrics.csv"
    runtime_csv = table_dir / "risk_analytics_runtime_iterations.csv"
    manifest_csv = table_dir / "risk_analytics_output_manifest.csv"

    write_csv(run_csv, run_rows, RUN_SUMMARY_FIELDNAMES)
    write_csv(boundary_csv, boundary_rows, BOUNDARY_METRIC_FIELDNAMES)
    write_csv(greek_csv, greek_rows, GREEK_METRIC_FIELDNAMES)
    write_csv(lcp_csv, lcp_rows, LCP_METRIC_FIELDNAMES)
    write_csv(runtime_csv, runtime_rows, RUNTIME_ITERATION_FIELDNAMES)

    figure_paths = {
        "put_boundary_vs_volatility": figure_dir
        / "risk_analytics_put_boundary_vs_volatility.png",
        "call_boundary_vs_dividend_yield": figure_dir
        / "risk_analytics_call_boundary_vs_dividend_yield.png",
        "put_vol_boundary_curves": figure_dir / "risk_analytics_put_vol_boundary_curves.png",
        "call_q_boundary_curves": figure_dir / "risk_analytics_call_q_boundary_curves.png",
        "gamma_concentration": figure_dir / "risk_analytics_gamma_concentration.png",
        "psor_runtime_iterations": figure_dir / "risk_analytics_psor_runtime_iterations.png",
        "lcp_stability": figure_dir / "risk_analytics_lcp_stability.png",
        "call_q_sigma_boundary_heatmap": figure_dir
        / "risk_analytics_call_q_sigma_boundary_heatmap.png",
    }
    figure_created = {name: False for name in figure_paths}
    if create_figures:
        figure_created["put_boundary_vs_volatility"] = (
            create_put_boundary_vs_volatility_figure(
                runs, figure_paths["put_boundary_vs_volatility"]
            )
        )
        figure_created["call_boundary_vs_dividend_yield"] = (
            create_call_boundary_vs_dividend_yield_figure(
                runs, figure_paths["call_boundary_vs_dividend_yield"]
            )
        )
        figure_created["put_vol_boundary_curves"] = create_put_vol_boundary_curves_figure(
            runs, figure_paths["put_vol_boundary_curves"]
        )
        figure_created["call_q_boundary_curves"] = create_call_q_boundary_curves_figure(
            runs, figure_paths["call_q_boundary_curves"]
        )
        figure_created["gamma_concentration"] = create_gamma_concentration_figure(
            runs, figure_paths["gamma_concentration"]
        )
        figure_created["psor_runtime_iterations"] = create_psor_runtime_iterations_figure(
            runs, figure_paths["psor_runtime_iterations"]
        )
        figure_created["lcp_stability"] = create_lcp_stability_figure(
            runs, figure_paths["lcp_stability"]
        )
        figure_created["call_q_sigma_boundary_heatmap"] = (
            create_call_q_sigma_boundary_heatmap_figure(
                runs, figure_paths["call_q_sigma_boundary_heatmap"]
            )
        )

    case_count = len(runs)
    manifest_rows = [
        manifest_row(
            run_csv,
            "table",
            "risk_analytics_run_summary",
            "Run-level quantitative risk analytics summary.",
            run_csv.exists(),
            case_count,
            contains_boundary_metrics=True,
            contains_greek_metrics=True,
            contains_lcp_metrics=True,
        ),
        manifest_row(
            boundary_csv,
            "table",
            "risk_analytics_boundary_metrics",
            "Selected-time threshold boundary and continuation-premium metrics.",
            boundary_csv.exists(),
            case_count,
            contains_boundary_metrics=True,
        ),
        manifest_row(
            greek_csv,
            "table",
            "risk_analytics_greek_metrics",
            "Finite-difference Greek diagnostics with caution-mask metrics.",
            greek_csv.exists(),
            case_count,
            contains_greek_metrics=True,
        ),
        manifest_row(
            lcp_csv,
            "table",
            "risk_analytics_lcp_metrics",
            "Obstacle, equation, and complementarity diagnostic metrics.",
            lcp_csv.exists(),
            case_count,
            contains_lcp_metrics=True,
        ),
        manifest_row(
            runtime_csv,
            "table",
            "risk_analytics_runtime_iterations",
            "Runtime and PSOR iteration diagnostics.",
            runtime_csv.exists(),
            case_count,
        ),
        manifest_row(
            manifest_csv,
            "table",
            "risk_analytics_output_manifest",
            "Manifest for every quantitative risk analytics table and figure.",
            True,
            case_count,
            contains_boundary_metrics=True,
            contains_greek_metrics=True,
            contains_lcp_metrics=True,
        ),
    ]
    figure_manifest = [
        (
            "put_boundary_vs_volatility",
            "American put final-time boundary moneyness versus volatility.",
            True,
            False,
            False,
        ),
        (
            "call_boundary_vs_dividend_yield",
            "Dividend-call final-time boundary moneyness versus dividend yield.",
            True,
            False,
            False,
        ),
        (
            "put_vol_boundary_curves",
            "American put boundary curves across volatility.",
            True,
            False,
            False,
        ),
        (
            "call_q_boundary_curves",
            "Dividend-call boundary curves across dividend yield.",
            True,
            False,
            False,
        ),
        (
            "gamma_concentration",
            "Full and strict-mask Gamma concentration across key stress sweeps.",
            False,
            True,
            False,
        ),
        (
            "psor_runtime_iterations",
            "PSOR iteration and runtime diagnostics across analytics cases.",
            False,
            False,
            False,
        ),
        (
            "lcp_stability",
            "LCP diagnostic stability across analytics cases.",
            False,
            False,
            True,
        ),
        (
            "call_q_sigma_boundary_heatmap",
            "Small aggregate dividend-call boundary heatmap over q and sigma.",
            True,
            False,
            False,
        ),
    ]
    for output_id, description, has_boundary, has_greeks, has_lcp in figure_manifest:
        manifest_rows.append(
            manifest_row(
                figure_paths[output_id],
                "figure",
                output_id,
                description,
                figure_created[output_id],
                case_count,
                contains_boundary_metrics=has_boundary,
                contains_greek_metrics=has_greeks,
                contains_lcp_metrics=has_lcp,
            )
        )
    write_csv(manifest_csv, manifest_rows, OUTPUT_MANIFEST_FIELDNAMES)

    metadata = {
        "run_summary_csv": str(run_csv),
        "boundary_metrics_csv": str(boundary_csv),
        "greek_metrics_csv": str(greek_csv),
        "lcp_metrics_csv": str(lcp_csv),
        "runtime_iterations_csv": str(runtime_csv),
        "output_manifest_csv": str(manifest_csv),
        "put_boundary_vs_volatility_figure_created": str(
            figure_created["put_boundary_vs_volatility"]
        ),
        "call_boundary_vs_dividend_yield_figure_created": str(
            figure_created["call_boundary_vs_dividend_yield"]
        ),
        "put_vol_boundary_curves_figure_created": str(
            figure_created["put_vol_boundary_curves"]
        ),
        "call_q_boundary_curves_figure_created": str(
            figure_created["call_q_boundary_curves"]
        ),
        "gamma_concentration_figure_created": str(figure_created["gamma_concentration"]),
        "psor_runtime_iterations_figure_created": str(
            figure_created["psor_runtime_iterations"]
        ),
        "lcp_stability_figure_created": str(figure_created["lcp_stability"]),
        "call_q_sigma_boundary_heatmap_figure_created": str(
            figure_created["call_q_sigma_boundary_heatmap"]
        ),
    }
    print(f"wrote {len(run_rows)} rows to {run_csv}")
    print(f"wrote {len(boundary_rows)} rows to {boundary_csv}")
    print(f"wrote {len(greek_rows)} rows to {greek_csv}")
    print(f"wrote {len(lcp_rows)} rows to {lcp_csv}")
    print(f"wrote {len(runtime_rows)} rows to {runtime_csv}")
    print(f"wrote {len(manifest_rows)} rows to {manifest_csv}")
    print(
        "figures_created="
        + ",".join(f"{name}:{created}" for name, created in figure_created.items())
    )
    return (
        run_rows,
        boundary_rows,
        greek_rows,
        lcp_rows,
        runtime_rows,
        manifest_rows,
        metadata,
    )


if __name__ == "__main__":
    main()
