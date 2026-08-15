"""Stage 4B: deterministic polynomial map from regimes to POD coefficients."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from american_risk_surfaces.method_extensions.pod import (
    fit_pod_basis,
    load_premium_surface_dataset,
)
from american_risk_surfaces.method_extensions.protocol import DATASET_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "07_method_extensions" / "05_pod_coefficient"
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def run_pod_coefficient_experiment(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pod_decision = json.loads(
        (
            PROJECT_ROOT
            / "results"
            / "07_method_extensions"
            / "04_pod"
            / "pod_decision.json"
        ).read_text(encoding="utf-8")
    )
    if pod_decision["status"] != "GO_POD_BASIS_LADDER":
        raise RuntimeError("POD rank gate did not authorize a coefficient model")
    if pod_decision["selected_representation"] != "unaligned":
        raise RuntimeError("online coefficient model currently requires an unaligned POD basis")
    modes = int(pod_decision["selected_modes"])
    label_floor = float(pod_decision["label_floor"])
    dataset = load_premium_surface_dataset()
    parameters = _regime_parameters(dataset.regime_ids)
    train = dataset.split_by_regime == "train"
    validation = dataset.split_by_regime == "validation"
    basis = fit_pod_basis(
        dataset.premium_surfaces[train],
        dataset.regime_ids[train],
        representation="unaligned",
    )
    flat = dataset.premium_surfaces.reshape(len(dataset.premium_surfaces), -1)
    true_scores = np.einsum(
        "ij,kj->ik", flat - basis.mean, basis.components[:modes], optimize=False
    )
    scaler = StandardScaler().fit(parameters[train])
    polynomial = PolynomialFeatures(degree=3, include_bias=True)
    train_features = polynomial.fit_transform(scaler.transform(parameters[train]))
    all_features = polynomial.transform(scaler.transform(parameters))

    validation_rows: list[dict[str, Any]] = []
    models: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        coefficients = _fit_ridge(train_features, true_scores[train], alpha)
        models[alpha] = coefficients
        predicted_scores = _predict_scores(all_features[validation], coefficients)
        reconstructed = basis.mean + np.einsum(
            "ik,kj->ij", predicted_scores, basis.components[:modes], optimize=False
        )
        reconstructed = np.maximum(reconstructed, 0.0)
        error = reconstructed - flat[validation]
        validation_rows.append(
            {
                "alpha": alpha,
                "validation_premium_mae": float(np.mean(np.abs(error))),
                "validation_premium_rmse": float(np.sqrt(np.mean(error**2))),
                "validation_premium_max_abs_error": float(np.max(np.abs(error))),
            }
        )
    selected_alpha = min(
        validation_rows, key=lambda row: float(row["validation_premium_rmse"])
    )["alpha"]
    coefficients = models[float(selected_alpha)]

    started = perf_counter()
    predicted_scores = _predict_scores(all_features, coefficients)
    reconstructed = basis.mean + np.einsum(
        "ik,kj->ij", predicted_scores, basis.components[:modes], optimize=False
    )
    projected = np.maximum(reconstructed, 0.0)
    batch_inference_seconds = perf_counter() - started
    predicted_surfaces = projected.reshape(dataset.premium_surfaces.shape)
    metric_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "stress_holdout"):
        regime_mask = dataset.split_by_regime == split
        for region, node_masks in (
            ("all", np.ones_like(dataset.strict_interior_masks)),
            ("boundary_near", dataset.boundary_near_masks),
            ("strict_interior", dataset.strict_interior_masks),
        ):
            mask = node_masks[regime_mask]
            target = dataset.premium_surfaces[regime_mask][mask]
            prediction = predicted_surfaces[regime_mask][mask]
            error = prediction - target
            metric_rows.append(
                {
                    "split": split,
                    "region": region,
                    "row_count": int(target.size),
                    "premium_mae": float(np.mean(np.abs(error))),
                    "premium_rmse": float(np.sqrt(np.mean(error**2))),
                    "premium_max_abs_error": float(np.max(np.abs(error))),
                    "negative_premium_rate": float(np.mean(prediction < 0.0)),
                }
            )

    latency_samples = []
    sample_feature = all_features[:1]
    for _ in range(20):
        _predict_scores(sample_feature, coefficients)
    for _ in range(1000):
        tick = perf_counter()
        score = _predict_scores(sample_feature, coefficients)
        np.maximum(
            basis.mean
            + np.einsum("ik,kj->ij", score, basis.components[:modes], optimize=False),
            0.0,
        )
        latency_samples.append(perf_counter() - tick)

    validation_path = output / "alpha_validation.csv"
    metrics_path = output / "coefficient_map_metrics.csv"
    _write_csv(validation_path, validation_rows, tuple(validation_rows[0]))
    _write_csv(metrics_path, metric_rows, tuple(metric_rows[0]))
    artifact_path = output / "pod_coefficient_model.npz"
    np.savez_compressed(
        artifact_path,
        pod_mean=basis.mean,
        pod_components=basis.components[:modes],
        pod_singular_values=basis.singular_values,
        parameter_scaler_mean=scaler.mean_,
        parameter_scaler_scale=scaler.scale_,
        polynomial_powers=polynomial.powers_,
        ridge_coef=coefficients.T,
        modes=np.array(modes),
        alpha=np.array(selected_alpha),
        train_regime_ids=dataset.regime_ids[train],
    )
    decision = _decision(metric_rows, label_floor)
    decision.update(
        {
            "selected_alpha": selected_alpha,
            "modes": modes,
            "polynomial_degree": 3,
            "parameter_order": ["T", "sigma", "r", "q", "is_call"],
            "basis_fit_split": "train_only",
            "hyperparameter_split": "validation_only",
            "batch_inference_seconds": batch_inference_seconds,
            "median_single_surface_seconds": float(np.median(latency_samples)),
            "p95_single_surface_seconds": float(np.percentile(latency_samples, 95)),
            "artifact_path": str(artifact_path),
        }
    )
    decision_path = output / "coefficient_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "coefficient_report.md"
    report_path.write_text(_report(decision), encoding="utf-8")
    return {
        "validation": validation_path,
        "metrics": metrics_path,
        "artifact": artifact_path,
        "decision": decision_path,
        "report": report_path,
        "decision_data": decision,
    }


def _fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    """Fit a small multi-output ridge map without opaque BLAS prediction calls."""

    gram = np.einsum("ni,nj->ij", features, features, optimize=False)
    cross = np.einsum("ni,nk->ik", features, targets, optimize=False)
    regularized = gram + float(alpha) * np.eye(gram.shape[0])
    return np.linalg.solve(regularized, cross)


def _predict_scores(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->ik", features, coefficients, optimize=False)


def _decision(rows: list[dict[str, Any]], label_floor: float) -> dict[str, Any]:
    heldout = [
        row
        for row in rows
        if row["region"] == "all" and row["split"] in {"test", "stress_holdout"}
    ]
    worst = max(float(row["premium_rmse"]) for row in heldout)
    go = worst <= 1.25 * label_floor and all(
        float(row["negative_premium_rate"]) == 0.0 for row in heldout
    )
    return {
        "status": "GO_REDUCED_BASIS_VI" if go else "STOP_POD_COEFFICIENT_AT_DIAGNOSTIC",
        "worst_heldout_rmse": worst,
        "acceptance_threshold": 1.25 * label_floor,
        "obstacle_preserved_by_projection": True,
        "next_method": "primal_dual_reduced_basis_vi" if go else "retain_pod_as_rank_diagnostic_only",
    }


def _regime_parameters(regime_ids: np.ndarray) -> np.ndarray:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        by_id = {row["regime_id"]: row for row in csv.DictReader(handle)}
    return np.asarray(
        [
            [
                float(by_id[str(regime_id)]["T"]),
                float(by_id[str(regime_id)]["sigma"]),
                float(by_id[str(regime_id)]["r"]),
                float(by_id[str(regime_id)]["q"]),
                1.0 if by_id[str(regime_id)]["option_type"] == "call" else 0.0,
            ]
            for regime_id in regime_ids
        ],
        dtype=float,
    )


def _report(decision: dict[str, Any]) -> str:
    return (
        "# POD Coefficient Map\n\n"
        f"Decision: **{decision['status']}**\n\n"
        f"Worst held-out RMSE: `{decision['worst_heldout_rmse']:.6g}`\n\n"
        f"Acceptance threshold: `{decision['acceptance_threshold']:.6g}`\n\n"
        f"Median online surface latency: `{decision['median_single_surface_seconds']:.6g}` seconds.\n"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_pod_coefficient_experiment()
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))
