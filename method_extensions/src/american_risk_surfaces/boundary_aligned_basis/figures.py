"""Small validation-only diagnostics for the oracle basis experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from american_risk_surfaces.boundary_aligned_basis.protocol import RESULTS_DIR


def make_validation_figures() -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = RESULTS_DIR / "04_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "03_validation" / "oracle_validation_ladder.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    paths: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, family in zip(axes, ("put", "call"), strict=True):
        family_rows = [row for row in rows if row["option_type"] == family and row["arm"] == "L"]
        configurations = sorted(
            {(int(row["dimension"]), int(row["bin_count"])) for row in family_rows}
        )
        for dimension, bins in configurations:
            selected = [
                row for row in family_rows
                if int(row["dimension"]) == dimension
                and int(row["bin_count"]) == bins
                and row["finite_reconstruction"].lower() == "true"
            ]
            expected = 11 if family == "put" else 8
            if len(selected) != expected:
                continue
            boundary = max(float(row["reduction_boundary_conditional_mae"]) for row in selected)
            price = max(float(row["reduction_projected_price_rmse"]) for row in selected)
            axis.scatter(price, boundary, s=38)
            axis.annotate(f"m{dimension}/b{bins}", (price, boundary), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.axvline(4.94989e-4, color="black", linestyle="--", linewidth=1, label="price gate")
        axis.axhline(0.066667, color="tab:red", linestyle="--", linewidth=1, label="boundary gate")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("worst validation price RMSE")
        axis.set_ylabel("worst boundary conditional MAE")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "localized_price_boundary_gates.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, family in zip(axes, ("put", "call"), strict=True):
        audit = RESULTS_DIR / "01_transformation_audit" / f"{family}_roundtrip_3841.csv"
        with audit.open(newline="", encoding="utf-8") as handle:
            audit_rows = [row for row in csv.DictReader(handle) if row["status"] == "COMPLETE"]
        pairing = np.asarray([float(row["trajectory_pairing_relative_error"]) for row in audit_rows])
        axis.hist(pairing, bins=20, color="tab:blue", alpha=0.75)
        axis.axvline(1e-4, color="tab:red", linestyle="--", label="pre-registered gate")
        axis.set_title(f"{family}: 3841-point pairing error")
        axis.set_xlabel("relative pairing error")
        axis.set_ylabel("train regimes")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    path = output / "alignment_pairing_gate.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths
