"""Create the required method-extension comparison and structure figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve
from american_risk_surfaces.method_extensions.protocol import DATASET_DIR
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    as_legacy_cn_psor_result,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "07_method_extensions"
FIGURES = RESULTS / "figures"


def create_figures() -> list[Path]:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [
        _solver_pareto(plt),
        _greek_pareto(plt),
        _pod_curve(plt),
        _active_boundary_overlay(plt),
    ]
    manifest = RESULTS / "figure_manifest.json"
    manifest.write_text(
        json.dumps([str(path.relative_to(ROOT)) for path in paths], indent=2),
        encoding="utf-8",
    )
    return paths


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _solver_pareto(plt) -> Path:
    runtime = {
        row["arm"]: row
        for row in _rows(RESULTS / "02_warmstart" / "runtime_summary.csv")
    }
    accuracy = _rows(RESULTS / "02_warmstart" / "accuracy_structure_metrics.csv")
    errors = {
        arm: max(float(row["max_abs_value_error"]) for row in accuracy if row["arm"] == arm)
        for arm in runtime
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for arm, row in runtime.items():
        x = float(row["median_total_seconds"])
        y = max(errors[arm], 1e-16)
        axis.scatter(x, y, s=65)
        axis.annotate(arm, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(xlabel="Median end-to-end seconds", ylabel="Maximum value error",
             title="Strict-LCP solver runtime/error comparison")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    return _save(plt, fig, "solver_error_runtime_pareto.png")


def _greek_pareto(plt) -> Path:
    rows = _rows(RESULTS / "03_greek_audit" / "method_summary.csv")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        x, y = float(row["median_runtime_seconds"]), float(row["median_gamma_max_error"])
        axis.scatter(x, y, s=65)
        axis.annotate(row["method"], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(xlabel="Median seconds", ylabel="Median Gamma max error",
             title="Greek reference runtime/error comparison")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    return _save(plt, fig, "greek_error_runtime_pareto.png")


def _pod_curve(plt) -> Path:
    rows = _rows(RESULTS / "04_pod" / "pod_reconstruction_metrics.csv")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for representation in ("unaligned", "oracle_boundary_aligned"):
        selected = [
            row for row in rows
            if row["representation"] == representation
            and row["split"] in {"test", "stress_holdout"}
            and row["region"] == "all"
        ]
        grouped: dict[int, float] = {}
        for row in selected:
            mode = int(row["modes"])
            grouped[mode] = max(grouped.get(mode, 0.0), float(row["premium_rmse"]))
        axis.plot(sorted(grouped), [grouped[key] for key in sorted(grouped)], marker="o",
                  label=representation)
    decision = json.loads((RESULTS / "04_pod" / "pod_decision.json").read_text())
    axis.axhline(float(decision["label_floor"]), linestyle="--", color="black", label="label floor")
    axis.set(xlabel="POD modes", ylabel="Worst held-out premium RMSE",
             title="POD falsification ladder")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    return _save(plt, fig, "pod_mode_error.png")


def _active_boundary_overlay(plt) -> Path:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        regime = next(row for row in csv.DictReader(handle) if row["regime_id"] == "put_T100_s020_r005_q003")
    tolerance = json.loads(
        (RESULTS / "00_protocol" / "tolerance_decision.json").read_text()
    )["frozen_normalized_lcp_tolerance"]
    config = AmericanLCPConfig(
        regime["option_type"], float(regime["K"]), float(regime["T"]),
        float(regime["r"]), float(regime["q"]), float(regime["sigma"]),
        float(regime["Smax"]), 120, 120, tolerance=float(tolerance),
        obstacle_tolerance=1e-12,
    )
    psor = american_cn_lcp_price(config, lcp_solver="psor")
    policy = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    curves = {
        "PSOR": extract_boundary_curve(as_legacy_cn_psor_result(psor), "PSOR"),
        "Policy Iteration": extract_boundary_curve(as_legacy_cn_psor_result(policy), "Policy"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for label, curve in curves.items():
        points = [point for point in curve.points if point.boundary_found]
        axes[0].plot([point.tau for point in points], [point.boundary_spot for point in points], label=label)
    axes[0].set(xlabel="Time to maturity", ylabel="Exercise boundary S/K",
                title="Boundary overlay")
    axes[0].legend()
    changes = [sum(result.active_set_changes) for result in policy.lcp_results]
    axes[1].plot(policy.tau_grid[1:], changes)
    axes[1].set(xlabel="Time to maturity", ylabel="Active-set changes",
                title="Policy active-set work by time slice")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save(plt, fig, "active_set_boundary_overlay.png")


def _save(plt, figure, name: str) -> Path:
    path = FIGURES / name
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    print("\n".join(str(path) for path in create_figures()))
