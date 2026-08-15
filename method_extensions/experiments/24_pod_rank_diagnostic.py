"""Stage 4: train-only POD and oracle boundary-alignment rank diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.method_extensions.pod import (
    fit_pod_basis,
    load_premium_surface_dataset,
    oracle_align_surfaces,
    undo_oracle_alignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "07_method_extensions" / "04_pod"
MODES = (4, 8, 16, 24, 32, 48, 64)


def run_pod_diagnostic(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_premium_surface_dataset()
    train = dataset.split_by_regime == "train"
    aligned, shifts = oracle_align_surfaces(dataset)
    representations = {
        "unaligned": (dataset.premium_surfaces, None),
        "oracle_boundary_aligned": (aligned, shifts),
    }
    metric_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []

    for representation, (surfaces, representation_shifts) in representations.items():
        basis = fit_pod_basis(
            surfaces[train],
            dataset.regime_ids[train],
            representation=representation,
        )
        energy = basis.singular_values**2
        cumulative = np.cumsum(energy) / np.sum(energy)
        singular_rows.extend(
            {
                "representation": representation,
                "mode": mode,
                "singular_value": float(value),
                "cumulative_energy": float(cumulative[mode - 1]),
            }
            for mode, value in enumerate(basis.singular_values, start=1)
        )
        for modes in MODES:
            reconstructed = basis.reconstruct(surfaces, modes)
            if representation_shifts is not None:
                reconstructed = undo_oracle_alignment(
                    reconstructed, representation_shifts, dataset.moneyness_grid
                )
            projected = np.maximum(reconstructed, 0.0)
            for split in ("train", "validation", "test", "stress_holdout"):
                regime_mask = dataset.split_by_regime == split
                for region, node_masks in (
                    ("all", np.ones_like(dataset.strict_interior_masks)),
                    ("boundary_near", dataset.boundary_near_masks),
                    ("strict_interior", dataset.strict_interior_masks),
                ):
                    mask = node_masks[regime_mask]
                    target = dataset.premium_surfaces[regime_mask][mask]
                    raw_prediction = reconstructed[regime_mask][mask]
                    prediction = projected[regime_mask][mask]
                    error = prediction - target
                    metric_rows.append(
                        {
                            "representation": representation,
                            "modes": modes,
                            "split": split,
                            "region": region,
                            "row_count": int(target.size),
                            "premium_mae": float(np.mean(np.abs(error))),
                            "premium_rmse": float(np.sqrt(np.mean(error**2))),
                            "premium_max_abs_error": float(np.max(np.abs(error))),
                            "raw_negative_premium_rate": float(np.mean(raw_prediction < 0.0)),
                            "projected_negative_premium_rate": float(np.mean(prediction < 0.0)),
                        }
                    )

    metrics_path = output / "pod_reconstruction_metrics.csv"
    singular_path = output / "singular_value_decay.csv"
    _write_csv(metrics_path, metric_rows, tuple(metric_rows[0]))
    _write_csv(singular_path, singular_rows, tuple(singular_rows[0]))
    label_floor = _label_floor()
    decision = _decision(metric_rows, label_floor)
    decision.update(
        {
            "label_floor": label_floor,
            "label_floor_source": "median mean_abs_value_difference in v1 higher-grid confirmations",
            "basis_fit_split": "train_only",
            "oracle_alignment_warning": (
                "True boundaries are used only to diagnose moving-interface rank; "
                "an online aligned model would require a predicted boundary."
            ),
            "tested_modes": list(MODES),
        }
    )
    decision_path = output / "pod_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "pod_report.md"
    report_path.write_text(_report(decision), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "singular_values": singular_path,
        "decision": decision_path,
        "report": report_path,
        "decision_data": decision,
    }


def _decision(rows: list[dict[str, Any]], label_floor: float) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for representation in ("unaligned", "oracle_boundary_aligned"):
        for modes in MODES:
            if modes > 30:
                continue
            selected = [
                row
                for row in rows
                if row["representation"] == representation
                and int(row["modes"]) == modes
                and row["region"] == "all"
                and row["split"] in {"test", "stress_holdout"}
            ]
            if selected and all(float(row["premium_rmse"]) <= label_floor for row in selected):
                candidates.append(
                    {
                        "representation": representation,
                        "modes": modes,
                        "worst_heldout_rmse": max(float(row["premium_rmse"]) for row in selected),
                    }
                )
    if candidates:
        winner = min(candidates, key=lambda row: (int(row["modes"]), row["worst_heldout_rmse"]))
        return {
            "status": "GO_POD_BASIS_LADDER",
            "selected_representation": winner["representation"],
            "selected_modes": winner["modes"],
            "worst_heldout_rmse": winner["worst_heldout_rmse"],
            "next_method": "pod_coefficient_map_then_reduced_basis_vi",
        }
    return {
        "status": "HIGH_RANK_OPERATOR_JUSTIFIED",
        "selected_representation": None,
        "selected_modes": None,
        "worst_heldout_rmse": None,
        "next_method": "positive_premium_deeponet_or_localized_operator",
    }


def _label_floor() -> float:
    path = (
        PROJECT_ROOT
        / "results"
        / "04_surrogate_dataset"
        / "v1_small_grid"
        / "v1_higher_grid_confirmation.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(np.median([float(row["mean_abs_value_difference"]) for row in rows]))


def _report(decision: dict[str, Any]) -> str:
    return (
        "# POD Rank Diagnostic\n\n"
        f"Decision: **{decision['status']}**\n\n"
        f"Selected representation: `{decision['selected_representation']}`\n\n"
        f"Selected modes: `{decision['selected_modes']}`\n\n"
        f"Label floor: `{decision['label_floor']:.6g}`\n\n"
        f"Next method: `{decision['next_method']}`\n\n"
        "The aligned result is an oracle diagnostic and is not an online boundary solution.\n"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_pod_diagnostic()
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))
