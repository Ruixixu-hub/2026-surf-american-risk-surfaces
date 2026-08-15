"""Stage 3: price and positive-premium surrogate comparison."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from american_risk_surfaces.workspace import EXTENSION_ROOT, frozen_input

try:  # pragma: no cover - import itself is environment dependent.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


PROJECT_ROOT = EXTENSION_ROOT
DEFAULT_DATASET_PATH = frozen_input(
    "results/04_surrogate_dataset/v1_small_grid/dataset_v1_small_grid.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "05_surrogate_models" / "price_premium"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "06_surrogate" / "price_premium_surrogate_report.tex"

RANDOM_SEED = 20260619
TRAIN_ROW_CAP = 250_000
RELATIVE_ERROR_DENOMINATOR_FLOOR = 1e-4
NEAR_ZERO_RATE_TOLERANCE = 1e-4
VIOLATION_AMOUNT_TOLERANCE = 1e-10
DOWNSTREAM_USE_STATUS = "stage3_surrogate_diagnostic_only"

FEATURE_NAMES = (
    "log_moneyness",
    "tau_fraction",
    "r",
    "q",
    "sigma",
    "T",
    "is_call",
)
LABEL_NAMES = (
    "value_over_K",
    "payoff_over_K",
    "premium_over_K",
    "exercise_indicator",
    "boundary_spot_over_K",
    "delta",
    "scaled_gamma",
)
MASK_NAMES = (
    "payoff_kink_near",
    "boundary_near",
    "maturity_row",
    "strict_interior",
    "gamma_allowed_mask",
    "delta_allowed_mask",
    "exercise_region",
    "continuation_region",
)
AUDIT_NUMERIC_NAMES = (
    "S_over_K",
    "tau",
    "S",
    "K",
    "Smax",
    "M",
    "N",
    "dS",
    "dtau",
    "regime_index",
    "split_index",
)
SPLIT_NAMES = ("train", "validation", "test", "stress_holdout")
EXPECTED_DATASET_KEYS = (
    "X",
    "y_value",
    "y_payoff",
    "y_premium",
    "y_exercise_indicator",
    "y_boundary",
    "y_delta",
    "y_scaled_gamma",
    "masks",
    "regime_index",
    "feature_names",
    "label_names",
    "mask_names",
    "audit_numeric",
    "audit_numeric_names",
    "regime_ids",
    "split_names",
)

MASK_INDEX = {name: index for index, name in enumerate(MASK_NAMES)}
AUDIT_NUMERIC_INDEX = {name: index for index, name in enumerate(AUDIT_NUMERIC_NAMES)}

METRICS_BY_SPLIT_FIELDNAMES = [
    "model_name",
    "split",
    "row_count",
    "value_mae",
    "value_rmse",
    "value_max_abs_error",
    "value_stable_relative_mae",
    "premium_mae",
    "premium_rmse",
    "premium_max_abs_error",
    "premium_stable_relative_mae",
    "relative_error_denominator_floor",
    "downstream_use_status",
]
METRICS_BY_REGION_FIELDNAMES = [
    "model_name",
    "split",
    "region",
    "row_count",
    "value_mae",
    "value_rmse",
    "value_max_abs_error",
    "value_stable_relative_mae",
    "premium_mae",
    "premium_rmse",
    "premium_max_abs_error",
    "premium_stable_relative_mae",
    "downstream_use_status",
]
OBSTACLE_FIELDNAMES = [
    "model_name",
    "split",
    "row_count",
    "obstacle_violation_rate",
    "negative_premium_rate",
    "max_obstacle_violation",
    "min_predicted_premium",
    "near_zero_rate_tolerance",
    "violation_amount_tolerance",
    "review_flag",
    "downstream_use_status",
]
PREDICTION_AUDIT_FIELDNAMES = [
    "model_name",
    "split",
    "sample_id",
    "row_index",
    "regime_id",
    "value_target",
    "value_prediction",
    "value_error",
    "premium_target",
    "premium_prediction",
    "premium_error",
    "payoff_over_K",
    "boundary_near",
    "strict_interior",
    "downstream_use_status",
]
MODEL_RUN_MANIFEST_FIELDNAMES = [
    "run_id",
    "model_name",
    "dataset_path",
    "dataset_file_bytes",
    "dataset_sha256_16",
    "torch_version",
    "sklearn_version",
    "random_seed",
    "train_row_cap",
    "train_sampling_rule",
    "selected_train_row_count",
    "preprocessing_policy",
    "input_columns",
    "target_name",
    "output_activation",
    "loss",
    "optimizer",
    "learning_rate",
    "epochs",
    "batch_size",
    "final_train_loss",
    "final_validation_loss",
    "model_weights_saved",
    "relative_error_denominator_floor",
    "near_zero_rate_tolerance",
    "violation_amount_tolerance",
    "review_decision",
    "downstream_use_status",
]


__all__ = (
    "AUDIT_NUMERIC_INDEX",
    "AUDIT_NUMERIC_NAMES",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_DATASET_KEYS",
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "MASK_INDEX",
    "MASK_NAMES",
    "METRICS_BY_REGION_FIELDNAMES",
    "METRICS_BY_SPLIT_FIELDNAMES",
    "ModelPrediction",
    "NEAR_ZERO_RATE_TOLERANCE",
    "PredictionPreprocessor",
    "RANDOM_SEED",
    "RELATIVE_ERROR_DENOMINATOR_FLOOR",
    "SPLIT_NAMES",
    "SurrogateDatasetBundle",
    "SurrogateExperimentResult",
    "TrainingConfig",
    "capped_train_indices",
    "fit_preprocessor",
    "load_v1_dataset",
    "metrics_by_region_rows",
    "metrics_by_split_rows",
    "obstacle_violation_rows",
    "run_surrogate_experiment",
    "split_masks",
    "validate_training_target",
    "write_csv",
)


@dataclass(frozen=True)
class DatasetFileHandle:
    """Small stand-in for numpy's NpzFile file list after arrays are loaded."""

    files: tuple[str, ...]


@dataclass(frozen=True)
class SurrogateDatasetBundle:
    """Loaded v1 small-grid dataset arrays for Stage 3."""

    dataset_path: Path
    X: np.ndarray
    y_value: np.ndarray
    y_payoff: np.ndarray
    y_premium: np.ndarray
    y_exercise_indicator: np.ndarray
    y_boundary: np.ndarray
    y_delta: np.ndarray
    y_scaled_gamma: np.ndarray
    masks: np.ndarray
    regime_index: np.ndarray
    feature_names: np.ndarray
    label_names: np.ndarray
    mask_names: np.ndarray
    audit_numeric: np.ndarray
    audit_numeric_names: np.ndarray
    regime_ids: np.ndarray
    split_names: np.ndarray
    arrays: DatasetFileHandle


@dataclass(frozen=True)
class PredictionPreprocessor:
    """Train-only scalers for the two approved surrogate inputs."""

    direct_scaler: StandardScaler
    premium_scaler: StandardScaler

    def transform_direct(self, X: np.ndarray) -> np.ndarray:
        return self.direct_scaler.transform(np.asarray(X, dtype=float))

    def transform_premium(self, X: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        augmented = premium_input_matrix(X, payoff)
        return self.premium_scaler.transform(augmented)


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed Stage 3 training configuration."""

    seed: int = RANDOM_SEED
    train_cap: int = TRAIN_ROW_CAP
    epochs: int = 10
    batch_size: int = 8192
    hidden_units: int = 64
    learning_rate: float = 1e-3
    validation_eval_cap: int = 120_000


@dataclass(frozen=True)
class ModelPrediction:
    """Full-dataset predictions for one surrogate."""

    model_name: str
    predicted_value: np.ndarray
    predicted_premium: np.ndarray


@dataclass(frozen=True)
class SurrogateExperimentResult:
    """Stage 3 experiment outputs and in-memory predictions."""

    output_dir: Path
    metrics_by_split_path: Path
    metrics_by_region_path: Path
    obstacle_summary_path: Path
    prediction_sample_audit_path: Path
    model_run_manifest_path: Path
    report_tex_path: Path
    predictions: dict[str, ModelPrediction]
    training_history: dict[str, list[dict[str, float]]]
    review_decision: str
    figure_paths: tuple[Path, ...]


class _MLP(nn.Module):
    """Small fixed MLP used for both Stage 3 comparison models."""

    def __init__(self, input_dim: int, hidden_units: int, output_activation: str) -> None:
        super().__init__()
        if output_activation not in {"linear", "softplus"}:
            raise ValueError(f"unsupported output activation {output_activation!r}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )
        self.output_activation = output_activation
        self.softplus = nn.Softplus()

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803
        raw = self.net(X).squeeze(-1)
        if self.output_activation == "softplus":
            return self.softplus(raw)
        return raw

    def set_softplus_mean_bias(self, premium_mean: float) -> None:
        if self.output_activation != "softplus":
            return
        premium_mean = max(float(premium_mean), 1e-6)
        bias = math.log(math.expm1(premium_mean))
        final = self.net[-1]
        with torch.no_grad():
            final.bias.fill_(bias)


def load_v1_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> SurrogateDatasetBundle:
    """Load and validate the v1 small-grid dataset without regenerating it."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as arrays:
        files = tuple(arrays.files)
        missing = [key for key in EXPECTED_DATASET_KEYS if key not in files]
        if missing:
            raise ValueError(f"v1 dataset is missing keys: {missing}")
        bundle = SurrogateDatasetBundle(
            dataset_path=dataset_path,
            X=arrays["X"].astype(float, copy=False),
            y_value=arrays["y_value"].astype(float, copy=False),
            y_payoff=arrays["y_payoff"].astype(float, copy=False),
            y_premium=arrays["y_premium"].astype(float, copy=False),
            y_exercise_indicator=arrays["y_exercise_indicator"].astype(float, copy=False),
            y_boundary=arrays["y_boundary"].astype(float, copy=False),
            y_delta=arrays["y_delta"].astype(float, copy=False),
            y_scaled_gamma=arrays["y_scaled_gamma"].astype(float, copy=False),
            masks=arrays["masks"].astype(bool, copy=False),
            regime_index=arrays["regime_index"].astype(int, copy=False),
            feature_names=arrays["feature_names"],
            label_names=arrays["label_names"],
            mask_names=arrays["mask_names"],
            audit_numeric=arrays["audit_numeric"].astype(float, copy=False),
            audit_numeric_names=arrays["audit_numeric_names"],
            regime_ids=arrays["regime_ids"],
            split_names=arrays["split_names"],
            arrays=DatasetFileHandle(files=files),
        )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: SurrogateDatasetBundle) -> None:
    row_count = bundle.X.shape[0]
    if bundle.X.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected Stage 3 feature count")
    for name in (
        "y_value",
        "y_payoff",
        "y_premium",
        "y_exercise_indicator",
        "y_boundary",
        "y_delta",
        "y_scaled_gamma",
        "regime_index",
    ):
        if getattr(bundle, name).shape[0] != row_count:
            raise ValueError(f"{name} row count does not match X")
    if bundle.masks.shape != (row_count, len(MASK_NAMES)):
        raise ValueError("mask matrix shape does not match Stage 3 schema")
    if bundle.audit_numeric.shape[0] != row_count:
        raise ValueError("audit row count does not match X")
    if tuple(map(str, bundle.feature_names)) != FEATURE_NAMES:
        raise ValueError("feature names do not match v1 schema")
    if tuple(map(str, bundle.mask_names)) != MASK_NAMES:
        raise ValueError("mask names do not match v1 schema")
    if np.nanmax(np.abs(bundle.y_premium - (bundle.y_value - bundle.y_payoff))) > 1e-10:
        raise ValueError("premium is not value minus payoff")


def split_masks(bundle: SurrogateDatasetBundle) -> dict[str, np.ndarray]:
    split_index = bundle.audit_numeric[:, AUDIT_NUMERIC_INDEX["split_index"]].astype(int)
    split_names = tuple(map(str, bundle.split_names))
    masks = {name: split_index == index for index, name in enumerate(split_names)}
    return {name: masks[name] for name in SPLIT_NAMES}


def capped_train_indices(
    train_indices: np.ndarray,
    cap: int = TRAIN_ROW_CAP,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=int)
    if cap <= 0:
        raise ValueError("train cap must be positive")
    if train_indices.size <= cap:
        return np.sort(train_indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_indices, size=cap, replace=False))


def premium_input_matrix(X: np.ndarray, payoff: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(X, dtype=float), np.asarray(payoff, dtype=float)])


def fit_preprocessor(
    bundle: SurrogateDatasetBundle,
    train_indices: np.ndarray,
    model_name: str | None = None,
) -> PredictionPreprocessor:
    """Fit both scalers on training rows only."""

    del model_name
    train_indices = np.asarray(train_indices, dtype=int)
    direct_scaler = StandardScaler().fit(bundle.X[train_indices])
    premium_scaler = StandardScaler().fit(
        premium_input_matrix(bundle.X[train_indices], bundle.y_payoff[train_indices])
    )
    return PredictionPreprocessor(
        direct_scaler=direct_scaler,
        premium_scaler=premium_scaler,
    )


def validate_training_target(target_name: str) -> None:
    if target_name in {"value_over_K", "premium_over_K"}:
        return
    if target_name in {"delta", "scaled_gamma", "boundary_spot_over_K"}:
        raise ValueError(f"{target_name} is diagnostic only and is not a Stage 3 training target")
    raise ValueError(f"unsupported Stage 3 training target {target_name!r}")


def run_surrogate_experiment(
    bundle: SurrogateDatasetBundle | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_tex_path: Path | str = DEFAULT_REPORT_PATH,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
    create_figures: bool = True,
) -> SurrogateExperimentResult:
    """Train and evaluate the two approved Stage 3 surrogates."""

    _require_torch()
    _set_random_seed(RANDOM_SEED)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    bundle = load_v1_dataset() if bundle is None else bundle
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if create_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    split_map = split_masks(bundle)
    train_indices = np.flatnonzero(split_map["train"])
    selected_train = capped_train_indices(train_indices, cap=train_cap, seed=RANDOM_SEED)
    validation_indices = np.flatnonzero(split_map["validation"])
    config = TrainingConfig(
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )
    preprocessor = fit_preprocessor(bundle, train_indices)

    direct_model, direct_history = _train_direct_value_model(
        bundle, preprocessor, selected_train, validation_indices, config
    )
    premium_model, premium_history = _train_positive_premium_model(
        bundle, preprocessor, selected_train, validation_indices, config
    )

    predictions = {
        "direct_value_mlp": _predict_direct(bundle, preprocessor, direct_model),
        "positive_premium_mlp": _predict_premium(bundle, preprocessor, premium_model),
    }
    metric_split_rows = metrics_by_split_rows(bundle, predictions)
    metric_region_rows = metrics_by_region_rows(bundle, predictions)
    obstacle_rows = obstacle_violation_rows(bundle, predictions)
    review_decision = stage3_review_decision(metric_split_rows, obstacle_rows)

    metrics_by_split_path = output_dir / "surrogate_metrics_by_split.csv"
    metrics_by_region_path = output_dir / "surrogate_metrics_by_region.csv"
    obstacle_summary_path = output_dir / "obstacle_violation_summary.csv"
    prediction_sample_audit_path = output_dir / "prediction_sample_audit.csv"
    model_run_manifest_path = output_dir / "model_run_manifest.csv"

    write_csv(metrics_by_split_path, metric_split_rows, METRICS_BY_SPLIT_FIELDNAMES)
    write_csv(metrics_by_region_path, metric_region_rows, METRICS_BY_REGION_FIELDNAMES)
    write_csv(obstacle_summary_path, obstacle_rows, OBSTACLE_FIELDNAMES)
    write_csv(
        prediction_sample_audit_path,
        prediction_sample_audit_rows(bundle, predictions),
        PREDICTION_AUDIT_FIELDNAMES,
    )
    histories = {
        "direct_value_mlp": direct_history,
        "positive_premium_mlp": premium_history,
    }
    write_csv(
        model_run_manifest_path,
        model_run_manifest_rows(
            bundle=bundle,
            selected_train_count=int(selected_train.size),
            train_cap=train_cap,
            epochs=epochs,
            batch_size=batch_size,
            histories=histories,
            review_decision=review_decision,
        ),
        MODEL_RUN_MANIFEST_FIELDNAMES,
    )

    figure_paths: tuple[Path, ...] = ()
    if create_figures:
        figure_paths = tuple(_create_figures(output_dir, histories, metric_split_rows, obstacle_rows, bundle, predictions))

    report_tex_path = Path(report_tex_path)
    write_stage3_report(
        report_tex_path,
        bundle=bundle,
        metric_split_rows=metric_split_rows,
        metric_region_rows=metric_region_rows,
        obstacle_rows=obstacle_rows,
        histories=histories,
        figure_paths=figure_paths,
        review_decision=review_decision,
        selected_train_count=int(selected_train.size),
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )

    return SurrogateExperimentResult(
        output_dir=output_dir,
        metrics_by_split_path=metrics_by_split_path,
        metrics_by_region_path=metrics_by_region_path,
        obstacle_summary_path=obstacle_summary_path,
        prediction_sample_audit_path=prediction_sample_audit_path,
        model_run_manifest_path=model_run_manifest_path,
        report_tex_path=report_tex_path,
        predictions=predictions,
        training_history=histories,
        review_decision=review_decision,
        figure_paths=figure_paths,
    )


def metrics_by_split_rows(
    bundle: SurrogateDatasetBundle,
    predictions: dict[str, ModelPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = split_masks(bundle)
    for model_name, prediction in predictions.items():
        for split, mask in split_map.items():
            rows.append(
                _metric_row(
                    METRICS_BY_SPLIT_FIELDNAMES,
                    model_name=model_name,
                    split=split,
                    region=None,
                    mask=mask,
                    bundle=bundle,
                    prediction=prediction,
                )
            )
    return rows


def metrics_by_region_rows(
    bundle: SurrogateDatasetBundle,
    predictions: dict[str, ModelPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = split_masks(bundle)
    regions = {
        "all": np.ones(bundle.X.shape[0], dtype=bool),
        "boundary_near": bundle.masks[:, MASK_INDEX["boundary_near"]],
        "strict_interior": bundle.masks[:, MASK_INDEX["strict_interior"]],
    }
    for model_name, prediction in predictions.items():
        for split, split_mask in split_map.items():
            for region, region_mask in regions.items():
                rows.append(
                    _metric_row(
                        METRICS_BY_REGION_FIELDNAMES,
                        model_name=model_name,
                        split=split,
                        region=region,
                        mask=split_mask & region_mask,
                        bundle=bundle,
                        prediction=prediction,
                    )
                )
    return rows


def obstacle_violation_rows(
    bundle: SurrogateDatasetBundle,
    predictions: dict[str, ModelPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = split_masks(bundle)
    for model_name, prediction in predictions.items():
        for split, mask in split_map.items():
            indices = np.flatnonzero(mask)
            obstacle = np.maximum(
                bundle.y_payoff[indices] - prediction.predicted_value[indices],
                0.0,
            )
            premium = prediction.predicted_premium[indices]
            obstacle_rate = float(np.mean(obstacle > VIOLATION_AMOUNT_TOLERANCE)) if indices.size else float("nan")
            negative_rate = float(np.mean(premium < -VIOLATION_AMOUNT_TOLERANCE)) if indices.size else float("nan")
            review_flag = (
                "PASS"
                if obstacle_rate <= NEAR_ZERO_RATE_TOLERANCE
                and negative_rate <= NEAR_ZERO_RATE_TOLERANCE
                else "REVIEW"
            )
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "row_count": int(indices.size),
                    "obstacle_violation_rate": obstacle_rate,
                    "negative_premium_rate": negative_rate,
                    "max_obstacle_violation": float(np.max(obstacle)) if indices.size else float("nan"),
                    "min_predicted_premium": float(np.min(premium)) if indices.size else float("nan"),
                    "near_zero_rate_tolerance": NEAR_ZERO_RATE_TOLERANCE,
                    "violation_amount_tolerance": VIOLATION_AMOUNT_TOLERANCE,
                    "review_flag": review_flag,
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def prediction_sample_audit_rows(
    bundle: SurrogateDatasetBundle,
    predictions: dict[str, ModelPrediction],
    per_split: int = 40,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    split_map = split_masks(bundle)
    for model_name, prediction in predictions.items():
        for split, split_mask in split_map.items():
            indices = np.flatnonzero(split_mask)
            if indices.size > per_split:
                selected = np.sort(rng.choice(indices, size=per_split, replace=False))
            else:
                selected = indices
            for sample_id, row_index in enumerate(selected):
                regime_id = str(bundle.regime_ids[int(bundle.regime_index[row_index])])
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "sample_id": sample_id,
                        "row_index": int(row_index),
                        "regime_id": regime_id,
                        "value_target": float(bundle.y_value[row_index]),
                        "value_prediction": float(prediction.predicted_value[row_index]),
                        "value_error": float(prediction.predicted_value[row_index] - bundle.y_value[row_index]),
                        "premium_target": float(bundle.y_premium[row_index]),
                        "premium_prediction": float(prediction.predicted_premium[row_index]),
                        "premium_error": float(prediction.predicted_premium[row_index] - bundle.y_premium[row_index]),
                        "payoff_over_K": float(bundle.y_payoff[row_index]),
                        "boundary_near": bool(bundle.masks[row_index, MASK_INDEX["boundary_near"]]),
                        "strict_interior": bool(bundle.masks[row_index, MASK_INDEX["strict_interior"]]),
                        "downstream_use_status": DOWNSTREAM_USE_STATUS,
                    }
                )
    return rows


def model_run_manifest_rows(
    *,
    bundle: SurrogateDatasetBundle,
    selected_train_count: int,
    train_cap: int,
    epochs: int,
    batch_size: int,
    histories: dict[str, list[dict[str, float]]],
    review_decision: str,
) -> list[dict[str, Any]]:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:  # pragma: no cover
        sklearn_version = "unknown"
    dataset_bytes = bundle.dataset_path.stat().st_size if bundle.dataset_path.exists() else 0
    dataset_hash = _sha256_16(bundle.dataset_path) if bundle.dataset_path.exists() else "synthetic"
    sampling_rule = (
        "If train rows exceed 250000, select 250000 positions with "
        "np.random.default_rng(20260619).choice(train_indices, replace=False), "
        "then sort selected indices for deterministic batching."
    )
    rows: list[dict[str, Any]] = []
    for model_name in ("direct_value_mlp", "positive_premium_mlp"):
        final = histories[model_name][-1]
        rows.append(
            {
                "run_id": "stage3_price_premium",
                "model_name": model_name,
                "dataset_path": str(bundle.dataset_path),
                "dataset_file_bytes": dataset_bytes,
                "dataset_sha256_16": dataset_hash,
                "torch_version": getattr(torch, "__version__", "unavailable"),
                "sklearn_version": sklearn_version,
                "random_seed": RANDOM_SEED,
                "train_row_cap": train_cap,
                "train_sampling_rule": sampling_rule,
                "selected_train_row_count": selected_train_count,
                "preprocessing_policy": "StandardScaler fit on train split rows only",
                "input_columns": "X" if model_name == "direct_value_mlp" else "X plus payoff_over_K",
                "target_name": "value_over_K" if model_name == "direct_value_mlp" else "premium_over_K",
                "output_activation": "linear" if model_name == "direct_value_mlp" else "softplus",
                "loss": "MSE",
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "epochs": epochs,
                "batch_size": batch_size,
                "final_train_loss": final["train_loss"],
                "final_validation_loss": final["validation_loss"],
                "model_weights_saved": "no",
                "relative_error_denominator_floor": RELATIVE_ERROR_DENOMINATOR_FLOOR,
                "near_zero_rate_tolerance": NEAR_ZERO_RATE_TOLERANCE,
                "violation_amount_tolerance": VIOLATION_AMOUNT_TOLERANCE,
                "review_decision": review_decision,
                "downstream_use_status": DOWNSTREAM_USE_STATUS,
            }
        )
    return rows


def stage3_review_decision(
    metric_split_rows: list[dict[str, Any]],
    obstacle_rows: list[dict[str, Any]],
) -> str:
    if not metric_split_rows or not obstacle_rows:
        return "REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE"
    for row in metric_split_rows:
        numeric = [
            row["value_mae"],
            row["value_rmse"],
            row["value_max_abs_error"],
            row["premium_mae"],
            row["premium_rmse"],
            row["premium_max_abs_error"],
        ]
        if not all(np.isfinite(float(value)) for value in numeric):
            return "REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE"

    pp_rows = [row for row in obstacle_rows if row["model_name"] == "positive_premium_mlp"]
    if any(float(row["obstacle_violation_rate"]) > NEAR_ZERO_RATE_TOLERANCE for row in pp_rows):
        return "REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE"
    if any(float(row["negative_premium_rate"]) > NEAR_ZERO_RATE_TOLERANCE for row in pp_rows):
        return "REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE"

    rmse: dict[tuple[str, str], float] = {
        (str(row["model_name"]), str(row["split"])): float(row["value_rmse"])
        for row in metric_split_rows
    }
    for split in ("validation", "test"):
        direct = rmse.get(("direct_value_mlp", split), float("nan"))
        premium = rmse.get(("positive_premium_mlp", split), float("nan"))
        if not np.isfinite(direct) or not np.isfinite(premium) or premium > 1.25 * direct:
            return "REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE"
    return "READY_FOR_BOUNDARY_DIAGNOSTIC_STAGE"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage3_report(
    path: Path,
    *,
    bundle: SurrogateDatasetBundle,
    metric_split_rows: list[dict[str, Any]],
    metric_region_rows: list[dict[str, Any]],
    obstacle_rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float]]],
    figure_paths: tuple[Path, ...],
    review_decision: str,
    selected_train_count: int,
    train_cap: int,
    epochs: int,
    batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = {name: int(mask.sum()) for name, mask in split_masks(bundle).items()}
    value_rows = [
        row
        for row in metric_split_rows
        if row["split"] in {"validation", "test", "stress_holdout"}
    ]
    obstacle_pp = [
        row
        for row in obstacle_rows
        if row["model_name"] == "positive_premium_mlp"
    ]
    figure_block = "\n".join(_latex_figure(path, figure_path) for figure_path in figure_paths)
    metrics_table = _latex_metrics_table(value_rows)
    obstacle_table = _latex_obstacle_table(obstacle_pp)
    region_table = _latex_region_table(metric_region_rows)
    final_direct = histories["direct_value_mlp"][-1]
    final_premium = histories["positive_premium_mlp"][-1]

    path.write_text(
        rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{fontspec}}
\IfFontExistsTF{{Times New Roman}}{{\setmainfont{{Times New Roman}}}}{{\setmainfont{{TeX Gyre Termes}}}}
\IfFontExistsTF{{Menlo}}{{\setmonofont{{Menlo}}}}{{\setmonofont{{Latin Modern Mono}}}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{fancyhdr}}
\usepackage[round,authoryear]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{xurl}}
\usepackage{{microtype}}

\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.65em}}
\setlist[itemize]{{leftmargin=1.4em}}
\newcommand{{\file}}[1]{{\path{{#1}}}}
\newcommand{{\code}}[1]{{\path{{#1}}}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}

\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{\small Stage 3 Surrogate Experiment}}
\rhead{{\small Price / Premium Comparison}}
\cfoot{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\title{{\textbf{{Stage 3 Price / Premium Surrogate Report}}\\
\large Direct Value Versus Positive Continuation-Premium Prediction}}
\author{{Codex-assisted surrogate modelling review}}
\date{{June 19, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report reviews the Stage 3 surrogate experiment on the v1 small-grid American option dataset.
Two fixed multilayer perceptrons were compared: a direct value surrogate and a positive-premium
surrogate that predicts continuation premium and reconstructs value as payoff plus premium. The
experiment uses the existing v1 dataset read-only, keeps Delta, Gamma, and boundary fields as
diagnostics only, does not save model weights, and does not train boundary or Greek heads. The
Stage 3 review decision is \code{{{review_decision}}}.
\end{{abstract}}

\tableofcontents
\newpage

\section{{Purpose of Stage 3}}

Stage 3 asks a narrow modelling question: does predicting nonnegative continuation premium help
preserve the American payoff obstacle better than predicting option value directly? The experiment
is intentionally small and fixed. It is not an architecture search, not a production-risk model,
and not a final research conclusion.

\section{{Link to v1 Dataset QA}}

The source dataset is \file{{results/04_surrogate_dataset/v1_small_grid/dataset_v1_small_grid.npz}}.
The v1 QA report accepted 288 regimes and 1,498,464 sampled rows for price-surrogate planning. The
solver source remains the validated baseline American CN/PSOR solver, grounded in the finite
difference American option formulation used throughout the validation ladder
\citep{{black_scholes_1973,merton_1973,brennan_schwartz_1977,wilmott_howison_dewynne_1995}}.

\begin{{longtable}}{{L{{0.26\textwidth}} L{{0.22\textwidth}} L{{0.40\textwidth}}}}
\toprule
Split & Rows & Role \\
\midrule
\endhead
Train & {split_counts['train']:,} & Training split; preprocessing fit only here. \\
Validation & {split_counts['validation']:,} & Regime-level validation split. \\
Test & {split_counts['test']:,} & Regime-level test split. \\
Stress holdout & {split_counts['stress_holdout']:,} & Held-out stress combinations. \\
\bottomrule
\end{{longtable}}

\section{{Model Families}}

\textbf{{Direct value model.}} The direct baseline uses the seven normalized v1 features and predicts
\code{{value_over_K}} with a linear output layer.

\textbf{{Positive-premium model.}} The premium model uses the same seven features plus
\code{{payoff_over_K}}, predicts \code{{premium_over_K}}, and applies a Softplus output so the
predicted premium is nonnegative. The value prediction is reconstructed as
\[
\widehat U/K = \Phi/K + \widehat{{(U-\Phi)}}/K.
\]
The payoff is analytic contract information computed from spot, strike, and option type. It is not
target leakage: it is known before pricing and is used only for reconstruction and obstacle-aware
structure.

\section{{Target Policy}}

The only training targets are \code{{value_over_K}} and \code{{premium_over_K}}. The fields
\code{{boundary_spot_over_K}}, \code{{delta}}, and \code{{scaled_gamma}} remain diagnostic only.
No boundary head, Delta head, Gamma head, or production Greek target is trained in this stage.

\section{{Preprocessing Policy}}

Both models use \code{{StandardScaler}} fit on training rows only. Validation, test, and stress
holdout rows do not influence scaling. The train subset is capped deterministically:
\[
\text{{seed}}=20260619,\quad \text{{cap}}={train_cap:,},\quad
\text{{selected rows}}={selected_train_count:,}.
\]
If the train split exceeds the cap, rows are selected with
\code{{np.random.default_rng(20260619).choice(train_indices, replace=False)}} and sorted before
batching.

\section{{Loss Functions and Training}}

Both MLPs use mean squared error and Adam with learning rate \(10^{-3}\). Training used
{epochs} epochs and batch size {batch_size}. No model weights or \code{{.pt}} files were saved.

\begin{{longtable}}{{L{{0.28\textwidth}} L{{0.24\textwidth}} L{{0.24\textwidth}}}}
\toprule
Model & Final train loss & Final validation loss \\
\midrule
\endhead
Direct value MLP & {_fmt(final_direct['train_loss'])} & {_fmt(final_direct['validation_loss'])} \\
Positive-premium MLP & {_fmt(final_premium['train_loss'])} & {_fmt(final_premium['validation_loss'])} \\
\bottomrule
\end{{longtable}}

\section{{Evaluation Protocol}}

Metrics are reported by regime-level split and by diagnostic region. Relative errors use a stable
denominator floor of \(10^{-4}\). Obstacle and negative-premium rates use a near-zero rate tolerance
of \(10^{-4}\), with violation amounts counted only above \(10^{-10}\).

\section{{Results by Split}}

{metrics_table}

\section{{Direct Value Versus Positive Premium}}

The direct model is the plain baseline. The positive-premium model adds American-option structure by
learning the continuation premium and reconstructing value from the analytic payoff. This comparison
therefore tests whether the obstacle-aware representation reduces payoff-obstacle violations while
keeping price accuracy close to the direct value baseline.

\section{{Obstacle and Negative-Premium Diagnostics}}

{obstacle_table}

\section{{Near-Boundary and Strict-Interior Diagnostics}}

Boundary-near rows use the v1 diagnostic mask inherited from Ticket 09/10 logic. Strict-interior rows
exclude maturity, payoff-kink, boundary-near, and nonfinite Greek regions. These regions are not
separate training targets; they are evaluation slices.

{region_table}

\section{{Figures}}

{figure_block}

\section{{Limitations}}

This experiment uses one fixed MLP configuration and a capped training subset. It does not search
architectures, tune hyperparameters broadly, train a boundary head, train Greek heads, or save model
weights. Delta and Gamma remain diagnostic fields. The stress-holdout results are informative but
not a production-risk guarantee.

\section{{What Stage 3 Supports}}

Stage 3 supports a first comparison between direct value prediction and an obstacle-aware
positive-premium representation. It also verifies that preprocessing can respect regime-level split
separation and that metrics can be reported by split, boundary-near region, and strict-interior
region.

\section{{What Stage 3 Does Not Support}}

This stage does not approve production deployment, production Greek labels, broad architecture
selection, boundary-head training, or final paper claims. It does not modify the validated solver and
does not regenerate the v1 dataset.

\section{{Recommended Next Stage}}

If the review decision is \code{{READY_FOR_BOUNDARY_DIAGNOSTIC_STAGE}}, the next stage may plan a
boundary-diagnostic modelling experiment. If the decision is \code{{REVIEW_REQUIRED_BEFORE_BOUNDARY_STAGE}},
the next step is human review of the price/premium errors, obstacle diagnostics, and training setup
before adding any new target heads.

\bibliographystyle{{plainnat}}
\bibliography{{reports/03_solver/references}}

\end{{document}}
""",
        encoding="utf-8",
    )


def _metric_row(
    fieldnames: list[str],
    *,
    model_name: str,
    split: str,
    region: str | None,
    mask: np.ndarray,
    bundle: SurrogateDatasetBundle,
    prediction: ModelPrediction,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    value_metrics = _error_metrics(
        bundle.y_value[indices],
        prediction.predicted_value[indices],
    )
    premium_metrics = _error_metrics(
        bundle.y_premium[indices],
        prediction.predicted_premium[indices],
    )
    row: dict[str, Any] = {
        "model_name": model_name,
        "split": split,
    }
    if "region" in fieldnames:
        row["region"] = region
    row.update(
        {
            "row_count": int(indices.size),
            "value_mae": value_metrics["mae"],
            "value_rmse": value_metrics["rmse"],
            "value_max_abs_error": value_metrics["max_abs_error"],
            "value_stable_relative_mae": value_metrics["stable_relative_mae"],
            "premium_mae": premium_metrics["mae"],
            "premium_rmse": premium_metrics["rmse"],
            "premium_max_abs_error": premium_metrics["max_abs_error"],
            "premium_stable_relative_mae": premium_metrics["stable_relative_mae"],
        }
    )
    if "relative_error_denominator_floor" in fieldnames:
        row["relative_error_denominator_floor"] = RELATIVE_ERROR_DENOMINATOR_FLOOR
    row["downstream_use_status"] = DOWNSTREAM_USE_STATUS
    return {name: row[name] for name in fieldnames}


def _error_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs_error": float("nan"),
            "stable_relative_mae": float("nan"),
        }
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    abs_error = np.abs(error)
    denominator = np.maximum(np.abs(target), RELATIVE_ERROR_DENOMINATOR_FLOOR)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(abs_error)),
        "stable_relative_mae": float(np.mean(abs_error / denominator)),
    }


def _train_direct_value_model(
    bundle: SurrogateDatasetBundle,
    preprocessor: PredictionPreprocessor,
    selected_train: np.ndarray,
    validation_indices: np.ndarray,
    config: TrainingConfig,
) -> tuple[_MLP, list[dict[str, float]]]:
    validate_training_target("value_over_K")
    _set_random_seed(config.seed)
    model = _MLP(input_dim=len(FEATURE_NAMES), hidden_units=config.hidden_units, output_activation="linear")
    X_train = preprocessor.transform_direct(bundle.X[selected_train]).astype(np.float32)
    y_train = bundle.y_value[selected_train].astype(np.float32)
    X_val = preprocessor.transform_direct(bundle.X[validation_indices]).astype(np.float32)
    y_val = bundle.y_value[validation_indices].astype(np.float32)
    return model, _fit_torch_model(model, X_train, y_train, X_val, y_val, config)


def _train_positive_premium_model(
    bundle: SurrogateDatasetBundle,
    preprocessor: PredictionPreprocessor,
    selected_train: np.ndarray,
    validation_indices: np.ndarray,
    config: TrainingConfig,
) -> tuple[_MLP, list[dict[str, float]]]:
    validate_training_target("premium_over_K")
    _set_random_seed(config.seed + 17)
    model = _MLP(input_dim=len(FEATURE_NAMES) + 1, hidden_units=config.hidden_units, output_activation="softplus")
    model.set_softplus_mean_bias(float(np.mean(bundle.y_premium[selected_train])))
    X_train = preprocessor.transform_premium(
        bundle.X[selected_train],
        bundle.y_payoff[selected_train],
    ).astype(np.float32)
    y_train = bundle.y_premium[selected_train].astype(np.float32)
    X_val = preprocessor.transform_premium(
        bundle.X[validation_indices],
        bundle.y_payoff[validation_indices],
    ).astype(np.float32)
    y_val = bundle.y_premium[validation_indices].astype(np.float32)
    return model, _fit_torch_model(model, X_train, y_train, X_val, y_val, config)


def _fit_torch_model(
    model: _MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    device = torch.device("cpu")
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    val_X = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.0
        rows = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * batch_X.shape[0]
            rows += batch_X.shape[0]
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(val_X), val_y).detach().cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": weighted_loss / max(rows, 1),
                "validation_loss": validation_loss,
            }
        )
    return history


def _predict_direct(
    bundle: SurrogateDatasetBundle,
    preprocessor: PredictionPreprocessor,
    model: _MLP,
    batch_size: int = 131_072,
) -> ModelPrediction:
    predicted_value = _batched_predict(
        model,
        bundle.X,
        lambda slc: preprocessor.transform_direct(bundle.X[slc]).astype(np.float32),
        batch_size=batch_size,
    )
    predicted_premium = predicted_value - bundle.y_payoff
    return ModelPrediction(
        model_name="direct_value_mlp",
        predicted_value=predicted_value,
        predicted_premium=predicted_premium,
    )


def _predict_premium(
    bundle: SurrogateDatasetBundle,
    preprocessor: PredictionPreprocessor,
    model: _MLP,
    batch_size: int = 131_072,
) -> ModelPrediction:
    predicted_premium = _batched_predict(
        model,
        bundle.X,
        lambda slc: preprocessor.transform_premium(
            bundle.X[slc],
            bundle.y_payoff[slc],
        ).astype(np.float32),
        batch_size=batch_size,
    )
    predicted_value = bundle.y_payoff + predicted_premium
    return ModelPrediction(
        model_name="positive_premium_mlp",
        predicted_value=predicted_value,
        predicted_premium=predicted_premium,
    )


def _batched_predict(
    model: _MLP,
    X: np.ndarray,
    transform: Any,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cpu")
    model.eval()
    out = np.empty(X.shape[0], dtype=float)
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            batch = torch.from_numpy(transform(slice(start, stop))).to(device)
            out[start:stop] = model(batch).detach().cpu().numpy()
    return out


def _create_figures(
    output_dir: Path,
    histories: dict[str, list[dict[str, float]]],
    metric_split_rows: list[dict[str, Any]],
    obstacle_rows: list[dict[str, Any]],
    bundle: SurrogateDatasetBundle,
    predictions: dict[str, ModelPrediction],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    path = figure_dir / "loss_curves.png"
    plt.figure(figsize=(8, 4.8))
    for model_name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["train_loss"] for row in history], label=f"{model_name} train")
        plt.plot(epochs, [row["validation_loss"] for row in history], linestyle="--", label=f"{model_name} validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Stage 3 Training and Validation Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    audit_indices = _deterministic_eval_sample(bundle, split="test", count=20_000)
    path = figure_dir / "value_prediction_scatter.png"
    plt.figure(figsize=(5.8, 5.4))
    for model_name, prediction in predictions.items():
        plt.scatter(
            bundle.y_value[audit_indices],
            prediction.predicted_value[audit_indices],
            s=5,
            alpha=0.35,
            label=model_name,
        )
    lim = [0.0, max(1e-6, float(np.max(bundle.y_value[audit_indices]))) * 1.05]
    plt.plot(lim, lim, color="black", linewidth=1)
    plt.xlabel("Target value / K")
    plt.ylabel("Predicted value / K")
    plt.title("Value Prediction Scatter on Test Sample")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    path = figure_dir / "premium_prediction_scatter.png"
    plt.figure(figsize=(5.8, 5.4))
    for model_name, prediction in predictions.items():
        plt.scatter(
            bundle.y_premium[audit_indices],
            prediction.predicted_premium[audit_indices],
            s=5,
            alpha=0.35,
            label=model_name,
        )
    lim = [0.0, max(1e-6, float(np.max(bundle.y_premium[audit_indices]))) * 1.05]
    plt.plot(lim, lim, color="black", linewidth=1)
    plt.xlabel("Target premium / K")
    plt.ylabel("Predicted premium / K")
    plt.title("Premium Prediction Scatter on Test Sample")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    path = figure_dir / "error_by_split.png"
    split_order = list(SPLIT_NAMES)
    x = np.arange(len(split_order))
    width = 0.35
    plt.figure(figsize=(7.2, 4.6))
    for offset, model_name in enumerate(("direct_value_mlp", "positive_premium_mlp")):
        rmse = [
            float(
                next(
                    row["value_rmse"]
                    for row in metric_split_rows
                    if row["model_name"] == model_name and row["split"] == split
                )
            )
            for split in split_order
        ]
        plt.bar(x + (offset - 0.5) * width, rmse, width=width, label=model_name)
    plt.xticks(x, split_order, rotation=20)
    plt.ylabel("Value RMSE")
    plt.title("Value RMSE by Regime-Level Split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    path = figure_dir / "obstacle_violation_comparison.png"
    plt.figure(figsize=(7.2, 4.6))
    x = np.arange(len(split_order))
    for offset, model_name in enumerate(("direct_value_mlp", "positive_premium_mlp")):
        rates = [
            float(
                next(
                    row["obstacle_violation_rate"]
                    for row in obstacle_rows
                    if row["model_name"] == model_name and row["split"] == split
                )
            )
            for split in split_order
        ]
        plt.bar(x + (offset - 0.5) * width, rates, width=width, label=model_name)
    plt.axhline(NEAR_ZERO_RATE_TOLERANCE, color="black", linestyle="--", linewidth=1, label="tolerance")
    plt.xticks(x, split_order, rotation=20)
    plt.ylabel("Obstacle violation rate")
    plt.title("Obstacle Violation Rate by Split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)
    return paths


def _deterministic_eval_sample(bundle: SurrogateDatasetBundle, split: str, count: int) -> np.ndarray:
    indices = np.flatnonzero(split_masks(bundle)[split])
    if indices.size <= count:
        return indices
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(indices, size=count, replace=False))


def _latex_figure(report_path: Path, figure_path: Path) -> str:
    relative = figure_path.relative_to(PROJECT_ROOT)
    caption = figure_path.stem.replace("_", " ").capitalize()
    return rf"""\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\textwidth]{{{relative}}}
\caption{{{caption}.}}
\end{{figure}}
"""


def _latex_metrics_table(rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in rows if row["split"] in {"validation", "test", "stress_holdout"}
    ]
    lines = [
        r"\begin{longtable}{L{0.24\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Model & Split & Value RMSE & Value MAE & Premium RMSE \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in selected:
        lines.append(
            rf"{_escape(row['model_name'])} & {_escape(row['split'])} & {_fmt(row['value_rmse'])} & {_fmt(row['value_mae'])} & {_fmt(row['premium_rmse'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_obstacle_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.18\textwidth} L{0.20\textwidth} L{0.20\textwidth} L{0.20\textwidth}}",
        r"\toprule",
        r"Split & Obstacle rate & Negative premium rate & Review \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            rf"{_escape(row['split'])} & {_fmt(row['obstacle_violation_rate'])} & {_fmt(row['negative_premium_rate'])} & {_escape(row['review_flag'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_region_table(rows: list[dict[str, Any]]) -> str:
    selected = [
        row
        for row in rows
        if row["split"] in {"test", "stress_holdout"}
        and row["region"] in {"boundary_near", "strict_interior"}
    ]
    lines = [
        r"\begin{longtable}{L{0.22\textwidth} L{0.16\textwidth} L{0.18\textwidth} L{0.15\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Model & Split & Region & Rows & Value RMSE \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in selected:
        lines.append(
            rf"{_escape(row['model_name'])} & {_escape(row['split'])} & {_escape(row['region'])} & {int(row['row_count']):,} & {_fmt(row['value_rmse'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _escape(value: Any) -> str:
    return str(value).replace("_", r"\_")


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for Stage 3") from _TORCH_IMPORT_ERROR


def _sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
