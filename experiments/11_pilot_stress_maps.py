"""Pilot 01: controlled pilot stress maps for validated American CN/PSOR solver."""

from __future__ import annotations

import os
from pathlib import Path

from american_risk_surfaces.downstream.pilot_stress_maps import (
    BOUNDARY_SUMMARY_FIELDNAMES,
    DIAGNOSTIC_SUMMARY_FIELDNAMES,
    OUTPUT_MANIFEST_FIELDNAMES,
    RUN_SUMMARY_FIELDNAMES,
    SELECTED_SLICE_FIELDNAMES,
    PilotCase,
    create_base_indicator_boundary_maps,
    create_base_premium_heatmaps,
    create_base_selected_value_slices,
    create_base_value_heatmaps,
    create_boundary_variation_comparison,
    create_greek_diagnostic_slices,
    create_premium_slice_variation_comparison,
    boundary_summary_row,
    diagnostic_summary_row,
    manifest_row,
    pilot_cases,
    run_pilot_case,
    run_summary_row,
    selected_slice_rows,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "02_pilot_stress_maps" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "02_pilot_stress_maps" / "figures"

RUN_SUMMARY_CSV = DEFAULT_TABLE_DIR / "pilot_01_run_summary.csv"
DIAGNOSTIC_SUMMARY_CSV = DEFAULT_TABLE_DIR / "pilot_01_diagnostic_summary.csv"
BOUNDARY_SUMMARY_CSV = DEFAULT_TABLE_DIR / "pilot_01_boundary_summary.csv"
SELECTED_SLICES_CSV = DEFAULT_TABLE_DIR / "pilot_01_selected_slices.csv"
OUTPUT_MANIFEST_CSV = DEFAULT_TABLE_DIR / "pilot_01_output_manifest.csv"

VALUE_HEATMAP_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_base_value_heatmaps.png"
PREMIUM_HEATMAP_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_base_premium_heatmaps.png"
INDICATOR_BOUNDARY_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_base_indicator_boundary_maps.png"
VALUE_SLICES_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_base_selected_value_slices.png"
BOUNDARY_VARIATION_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_boundary_variation_comparison.png"
PREMIUM_VARIATION_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_premium_slice_variation_comparison.png"
GREEK_DIAGNOSTIC_FIGURE = DEFAULT_FIGURE_DIR / "pilot_01_greek_diagnostic_slices.png"


def main(
    cases: tuple[PilotCase, ...] | None = None,
    table_dir: Path = DEFAULT_TABLE_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    create_figures: bool = True,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    """Run the controlled Pilot 01 stress-map experiment."""

    selected_cases = cases if cases is not None else pilot_cases(include_higher_grid_checks=True)
    artifacts = [run_pilot_case(case) for case in selected_cases]

    run_rows = [run_summary_row(artifact) for artifact in artifacts]
    diagnostic_rows = [diagnostic_summary_row(artifact) for artifact in artifacts]
    boundary_rows = [boundary_summary_row(artifact) for artifact in artifacts]
    slice_rows = [
        row
        for artifact in artifacts
        for row in selected_slice_rows(artifact)
    ]

    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    run_csv = table_dir / "pilot_01_run_summary.csv"
    diagnostic_csv = table_dir / "pilot_01_diagnostic_summary.csv"
    boundary_csv = table_dir / "pilot_01_boundary_summary.csv"
    slices_csv = table_dir / "pilot_01_selected_slices.csv"
    manifest_csv = table_dir / "pilot_01_output_manifest.csv"

    write_csv(run_csv, run_rows, RUN_SUMMARY_FIELDNAMES)
    write_csv(diagnostic_csv, diagnostic_rows, DIAGNOSTIC_SUMMARY_FIELDNAMES)
    write_csv(boundary_csv, boundary_rows, BOUNDARY_SUMMARY_FIELDNAMES)
    write_csv(slices_csv, slice_rows, SELECTED_SLICE_FIELDNAMES)

    figure_paths = {
        "value_heatmap": figure_dir / "pilot_01_base_value_heatmaps.png",
        "premium_heatmap": figure_dir / "pilot_01_base_premium_heatmaps.png",
        "indicator_boundary": figure_dir / "pilot_01_base_indicator_boundary_maps.png",
        "value_slices": figure_dir / "pilot_01_base_selected_value_slices.png",
        "boundary_variation": figure_dir / "pilot_01_boundary_variation_comparison.png",
        "premium_variation": figure_dir / "pilot_01_premium_slice_variation_comparison.png",
        "greek_diagnostic": figure_dir / "pilot_01_greek_diagnostic_slices.png",
    }
    figure_created = {
        "value_heatmap": False,
        "premium_heatmap": False,
        "indicator_boundary": False,
        "value_slices": False,
        "boundary_variation": False,
        "premium_variation": False,
        "greek_diagnostic": False,
    }
    if create_figures:
        figure_created["value_heatmap"] = create_base_value_heatmaps(
            artifacts, figure_paths["value_heatmap"]
        )
        figure_created["premium_heatmap"] = create_base_premium_heatmaps(
            artifacts, figure_paths["premium_heatmap"]
        )
        figure_created["indicator_boundary"] = create_base_indicator_boundary_maps(
            artifacts, figure_paths["indicator_boundary"]
        )
        figure_created["value_slices"] = create_base_selected_value_slices(
            artifacts, figure_paths["value_slices"]
        )
        figure_created["boundary_variation"] = create_boundary_variation_comparison(
            artifacts, figure_paths["boundary_variation"]
        )
        figure_created["premium_variation"] = create_premium_slice_variation_comparison(
            artifacts, figure_paths["premium_variation"]
        )
        figure_created["greek_diagnostic"] = create_greek_diagnostic_slices(
            artifacts, figure_paths["greek_diagnostic"]
        )

    manifest_rows = [
        manifest_row(
            run_csv,
            "table",
            "pilot_01_run_summary",
            "Run-level solver, grid, convergence, boundary, and acceptance summary.",
            run_csv.exists(),
        ),
        manifest_row(
            diagnostic_csv,
            "table",
            "pilot_01_diagnostic_summary",
            "LCP and Greek diagnostic summary for each pilot case.",
            diagnostic_csv.exists(),
            contains_greek_diagnostics=True,
        ),
        manifest_row(
            boundary_csv,
            "table",
            "pilot_01_boundary_summary",
            "Threshold-based boundary extraction metadata for each pilot case.",
            boundary_csv.exists(),
            contains_boundary_overlay=True,
        ),
        manifest_row(
            slices_csv,
            "table",
            "pilot_01_selected_slices",
            "Selected tau and moneyness rows with value, premium, boundary, and Greek diagnostics.",
            slices_csv.exists(),
            contains_greek_diagnostics=True,
            contains_boundary_overlay=True,
        ),
        manifest_row(
            manifest_csv,
            "table",
            "pilot_01_output_manifest",
            "Manifest for every Pilot 01 table and figure.",
            True,
        ),
    ]
    figure_manifest = [
        (
            "value_heatmap",
            "Base put/call option value heatmaps over the interpretation region.",
            False,
            False,
        ),
        (
            "premium_heatmap",
            "Base put/call continuation-premium heatmaps.",
            False,
            False,
        ),
        (
            "indicator_boundary",
            "Base exercise/continuation maps with threshold boundary overlays.",
            False,
            True,
        ),
        (
            "value_slices",
            "Base value slices at selected time-to-maturity levels.",
            False,
            False,
        ),
        (
            "boundary_variation",
            "Boundary curve comparison across one-at-a-time pilot variations.",
            False,
            True,
        ),
        (
            "premium_variation",
            "Mid-time continuation-premium slices across pilot variations.",
            False,
            False,
        ),
        (
            "greek_diagnostic",
            "Base Gamma diagnostic slices with kink and boundary cautions.",
            True,
            True,
        ),
    ]
    for output_id, description, has_greeks, has_boundary in figure_manifest:
        manifest_rows.append(
            manifest_row(
                figure_paths[output_id],
                "figure",
                f"pilot_01_{output_id}",
                description,
                figure_created[output_id],
                contains_greek_diagnostics=has_greeks,
                contains_boundary_overlay=has_boundary,
            )
        )
    write_csv(manifest_csv, manifest_rows, OUTPUT_MANIFEST_FIELDNAMES)

    metadata = {
        "run_summary_csv": str(run_csv),
        "diagnostic_summary_csv": str(diagnostic_csv),
        "boundary_summary_csv": str(boundary_csv),
        "selected_slices_csv": str(slices_csv),
        "output_manifest_csv": str(manifest_csv),
        "value_heatmap_figure_created": str(figure_created["value_heatmap"]),
        "premium_heatmap_figure_created": str(figure_created["premium_heatmap"]),
        "indicator_boundary_figure_created": str(figure_created["indicator_boundary"]),
        "value_slices_figure_created": str(figure_created["value_slices"]),
        "boundary_variation_figure_created": str(figure_created["boundary_variation"]),
        "premium_variation_figure_created": str(figure_created["premium_variation"]),
        "greek_diagnostic_figure_created": str(figure_created["greek_diagnostic"]),
    }
    print(f"wrote {len(run_rows)} rows to {run_csv}")
    print(f"wrote {len(diagnostic_rows)} rows to {diagnostic_csv}")
    print(f"wrote {len(boundary_rows)} rows to {boundary_csv}")
    print(f"wrote {len(slice_rows)} rows to {slices_csv}")
    print(f"wrote {len(manifest_rows)} rows to {manifest_csv}")
    print(
        "figures_created="
        + ",".join(f"{name}:{created}" for name, created in figure_created.items())
    )
    return run_rows, diagnostic_rows, boundary_rows, slice_rows, manifest_rows, metadata


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


if __name__ == "__main__":
    main()
