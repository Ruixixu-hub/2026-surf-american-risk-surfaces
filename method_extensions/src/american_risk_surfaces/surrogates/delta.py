"""Stage 5: Delta diagnostic and supervised Delta-head comparison."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

from american_risk_surfaces.surrogates import price_premium as stage3


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "05_surrogate_models" / "delta"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "06_surrogate" / "delta_diagnostic_report.tex"

STAGE5_DOCSTRING = '"""Stage 5: Delta diagnostic and supervised Delta-head comparison."""'
RANDOM_SEED = stage3.RANDOM_SEED
TRAIN_ROW_CAP = stage3.TRAIN_ROW_CAP
LOG_MONEYNESS_STEP = 1e-3
DELTA_BOUND_TOLERANCE = 1e-4
RELATIVE_ERROR_DENOMINATOR_FLOOR = stage3.RELATIVE_ERROR_DENOMINATOR_FLOOR
DOWNSTREAM_USE_STATUS = "stage5_delta_diagnostic_only"

DELTA_METRICS_BY_SPLIT_FIELDNAMES = [
    "method_name",
    "split",
    "row_count",
    "delta_mae",
    "delta_rmse",
    "delta_max_abs_error",
    "delta_stable_relative_mae",
    "relative_error_denominator_floor",
    "downstream_use_status",
]
DELTA_METRICS_BY_OPTION_TYPE_FIELDNAMES = [
    "method_name",
    "option_type",
    "row_count",
    "delta_mae",
    "delta_rmse",
    "delta_max_abs_error",
    "delta_stable_relative_mae",
    "downstream_use_status",
]
DELTA_METRICS_BY_REGION_FIELDNAMES = [
    "method_name",
    "split",
    "region",
    "row_count",
    "delta_mae",
    "delta_rmse",
    "delta_max_abs_error",
    "delta_stable_relative_mae",
    "downstream_use_status",
]
DELTA_BOUNDS_FIELDNAMES = [
    "method_name",
    "split",
    "row_count",
    "bounds_violation_rate",
    "sign_violation_rate",
    "max_bounds_violation",
    "max_sign_violation",
    "delta_bound_tolerance",
    "review_flag",
    "downstream_use_status",
]
DELTA_CURVE_SAMPLE_AUDIT_FIELDNAMES = [
    "method_name",
    "sample_id",
    "row_index",
    "regime_id",
    "split",
    "option_type",
    "S_over_K",
    "tau_fraction",
    "target_delta",
    "predicted_delta",
    "absolute_error",
    "delta_allowed_mask",
    "strict_interior",
    "boundary_near",
    "payoff_kink_near",
    "maturity_row",
    "downstream_use_status",
]
DELTA_MODEL_MANIFEST_FIELDNAMES = [
    "run_id",
    "method_name",
    "dataset_path",
    "random_seed",
    "train_row_cap",
    "stage3_models_retrained",
    "input_columns",
    "target_name",
    "training_rows",
    "method_family",
    "finite_difference_step",
    "mask_policy",
    "model_weights_saved",
    "final_train_loss",
    "final_validation_loss",
    "review_decision",
    "downstream_use_status",
]

__all__ = (
    "DELTA_BOUNDS_FIELDNAMES",
    "DELTA_CURVE_SAMPLE_AUDIT_FIELDNAMES",
    "DELTA_METRICS_BY_OPTION_TYPE_FIELDNAMES",
    "DELTA_METRICS_BY_REGION_FIELDNAMES",
    "DELTA_METRICS_BY_SPLIT_FIELDNAMES",
    "DeltaExperimentResult",
    "DeltaPrediction",
    "LOG_MONEYNESS_STEP",
    "central_log_moneyness_delta",
    "delta_allowed_mask",
    "delta_bounds_violation_rows",
    "delta_curve_sample_audit_rows",
    "delta_metrics_by_option_type_rows",
    "delta_metrics_by_region_rows",
    "delta_metrics_by_split_rows",
    "delta_training_indices",
    "price_implied_delta_predictions",
    "run_delta_diagnostics_experiment",
    "stage5_review_decision",
    "train_supervised_delta_head",
    "validate_delta_training_target",
    "write_csv",
    "write_delta_outputs",
)


@dataclass(frozen=True)
class DeltaPrediction:
    """Predicted row-level Delta for one diagnostic method."""

    method_name: str
    predicted_delta: np.ndarray


@dataclass(frozen=True)
class DeltaExperimentResult:
    """Stage 5 output paths and review decision."""

    output_dir: Path
    metrics_by_split_path: Path
    metrics_by_option_type_path: Path
    metrics_by_region_path: Path
    bounds_summary_path: Path
    curve_sample_audit_path: Path
    model_manifest_path: Path
    report_tex_path: Path
    predictions: dict[str, DeltaPrediction]
    training_histories: dict[str, list[dict[str, float]]]
    review_decision: str
    figure_paths: tuple[Path, ...]


def validate_delta_training_target(target_name: str) -> None:
    if target_name in {"delta", "y_delta"}:
        return
    if target_name in {"scaled_gamma", "gamma", "y_scaled_gamma"}:
        raise ValueError(f"{target_name} is blocked: Stage 5 does not train Gamma targets.")
    if target_name == "boundary_spot_over_K":
        raise ValueError("boundary_spot_over_K is not a Stage 5 Delta target.")
    raise ValueError(f"unsupported Stage 5 Delta training target {target_name!r}")


def delta_allowed_mask(bundle: stage3.SurrogateDatasetBundle) -> np.ndarray:
    return bundle.masks[:, stage3.MASK_INDEX["delta_allowed_mask"]] & np.isfinite(bundle.y_delta)


def delta_training_indices(
    bundle: stage3.SurrogateDatasetBundle,
    *,
    split_name: str = "train",
) -> np.ndarray:
    split_mask = stage3.split_masks(bundle)[split_name]
    return np.flatnonzero(split_mask & delta_allowed_mask(bundle))


def central_log_moneyness_delta(
    X: np.ndarray,
    value_function: Callable[[np.ndarray], np.ndarray],
    *,
    step: float = LOG_MONEYNESS_STEP,
) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    X = np.asarray(X, dtype=float)
    plus = X.copy()
    minus = X.copy()
    log_index = stage3.FEATURE_NAMES.index("log_moneyness")
    plus[:, log_index] += step
    minus[:, log_index] -= step
    value_plus = np.asarray(value_function(plus), dtype=float)
    value_minus = np.asarray(value_function(minus), dtype=float)
    s_plus = np.exp(plus[:, log_index])
    s_minus = np.exp(minus[:, log_index])
    return (value_plus - value_minus) / (s_plus - s_minus)


def price_implied_delta_predictions(
    bundle: stage3.SurrogateDatasetBundle,
    *,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
    step: float = LOG_MONEYNESS_STEP,
) -> tuple[dict[str, DeltaPrediction], dict[str, list[dict[str, float]]]]:
    stage3._require_torch()  # noqa: SLF001 - internal project helper reuse.
    stage3._set_random_seed(RANDOM_SEED)  # noqa: SLF001
    if hasattr(stage3.torch, "set_num_threads"):
        stage3.torch.set_num_threads(min(4, os.cpu_count() or 1))
    split_map = stage3.split_masks(bundle)
    train_indices = np.flatnonzero(split_map["train"])
    validation_indices = np.flatnonzero(split_map["validation"])
    selected_train = stage3.capped_train_indices(train_indices, cap=train_cap, seed=RANDOM_SEED)
    preprocessor = stage3.fit_preprocessor(bundle, train_indices)
    config = stage3.TrainingConfig(train_cap=train_cap, epochs=epochs, batch_size=batch_size)

    direct_model, direct_history = stage3._train_direct_value_model(  # noqa: SLF001
        bundle,
        preprocessor,
        selected_train,
        validation_indices,
        config,
    )
    premium_model, premium_history = stage3._train_positive_premium_model(  # noqa: SLF001
        bundle,
        preprocessor,
        selected_train,
        validation_indices,
        config,
    )

    direct_delta = central_log_moneyness_delta(
        bundle.X,
        lambda X: _predict_direct_value_for_features(X, preprocessor, direct_model),
        step=step,
    )
    premium_delta = central_log_moneyness_delta(
        bundle.X,
        lambda X: _predict_premium_value_for_features(X, preprocessor, premium_model),
        step=step,
    )
    predictions = {
        "direct_value_implied_delta": DeltaPrediction("direct_value_implied_delta", direct_delta),
        "positive_premium_implied_delta": DeltaPrediction("positive_premium_implied_delta", premium_delta),
    }
    histories = {
        "direct_value_implied_delta": direct_history,
        "positive_premium_implied_delta": premium_history,
    }
    return predictions, histories


def train_supervised_delta_head(
    bundle: stage3.SurrogateDatasetBundle,
    *,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
) -> tuple[DeltaPrediction, list[dict[str, float]]]:
    validate_delta_training_target("delta")
    stage3._require_torch()  # noqa: SLF001
    stage3._set_random_seed(RANDOM_SEED + 51)  # noqa: SLF001
    if hasattr(stage3.torch, "set_num_threads"):
        stage3.torch.set_num_threads(min(4, os.cpu_count() or 1))

    train_indices = delta_training_indices(bundle, split_name="train")
    if train_indices.size == 0:
        raise ValueError("no allowed train rows for supervised Delta head")
    selected_train = stage3.capped_train_indices(train_indices, cap=train_cap, seed=RANDOM_SEED)
    validation_indices = delta_training_indices(bundle, split_name="validation")
    if validation_indices.size == 0:
        validation_indices = selected_train[: min(selected_train.size, 8192)]

    scaler = StandardScaler().fit(bundle.X[train_indices])
    config = stage3.TrainingConfig(train_cap=train_cap, epochs=epochs, batch_size=batch_size)
    model = stage3._MLP(  # noqa: SLF001
        input_dim=len(stage3.FEATURE_NAMES),
        hidden_units=config.hidden_units,
        output_activation="linear",
    )
    history = stage3._fit_torch_model(  # noqa: SLF001
        model,
        scaler.transform(bundle.X[selected_train]).astype(np.float32),
        bundle.y_delta[selected_train].astype(np.float32),
        scaler.transform(bundle.X[validation_indices]).astype(np.float32),
        bundle.y_delta[validation_indices].astype(np.float32),
        config,
    )
    raw_delta = stage3._batched_predict(  # noqa: SLF001
        model,
        bundle.X,
        lambda slc: scaler.transform(bundle.X[slc]).astype(np.float32),
        batch_size=131_072,
    )
    clipped = clip_delta_bounds(bundle, raw_delta)
    return DeltaPrediction("supervised_delta_head", clipped), history


def clip_delta_bounds(bundle: stage3.SurrogateDatasetBundle, delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float).copy()
    is_call = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    delta[is_call] = np.clip(delta[is_call], 0.0, 1.0)
    delta[~is_call] = np.clip(delta[~is_call], -1.0, 0.0)
    return delta


def delta_metrics_by_split_rows(
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = stage3.split_masks(bundle)
    allowed = delta_allowed_mask(bundle)
    for method_name, prediction in predictions.items():
        for split, split_mask in split_map.items():
            rows.append(
                _metric_row(
                    DELTA_METRICS_BY_SPLIT_FIELDNAMES,
                    bundle,
                    prediction,
                    method_name=method_name,
                    split=split,
                    region=None,
                    mask=split_mask & allowed,
                )
            )
    return rows


def delta_metrics_by_option_type_rows(
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = delta_allowed_mask(bundle)
    is_call = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    for method_name, prediction in predictions.items():
        rows.append(
            _option_metric_row(
                bundle,
                prediction,
                method_name=method_name,
                option_type="put",
                mask=(~is_call) & allowed,
            )
        )
        rows.append(
            _option_metric_row(
                bundle,
                prediction,
                method_name=method_name,
                option_type="call",
                mask=is_call & allowed,
            )
        )
    return rows


def delta_metrics_by_region_rows(
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = stage3.split_masks(bundle)
    allowed = delta_allowed_mask(bundle)
    regions = {
        "all_allowed": allowed,
        "strict_interior": allowed & bundle.masks[:, stage3.MASK_INDEX["strict_interior"]],
        "boundary_near": bundle.masks[:, stage3.MASK_INDEX["boundary_near"]] & np.isfinite(bundle.y_delta),
        "payoff_kink_near": bundle.masks[:, stage3.MASK_INDEX["payoff_kink_near"]] & np.isfinite(bundle.y_delta),
        "maturity_row": bundle.masks[:, stage3.MASK_INDEX["maturity_row"]] & np.isfinite(bundle.y_delta),
    }
    for method_name, prediction in predictions.items():
        for split, split_mask in split_map.items():
            for region, region_mask in regions.items():
                rows.append(
                    _metric_row(
                        DELTA_METRICS_BY_REGION_FIELDNAMES,
                        bundle,
                        prediction,
                        method_name=method_name,
                        split=split,
                        region=region,
                        mask=split_mask & region_mask,
                    )
                )
    return rows


def delta_bounds_violation_rows(
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_map = stage3.split_masks(bundle)
    is_call = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    allowed = delta_allowed_mask(bundle)
    for method_name, prediction in predictions.items():
        delta = prediction.predicted_delta
        lower = np.where(is_call, 0.0, -1.0)
        upper = np.where(is_call, 1.0, 0.0)
        lower_violation = np.maximum(lower - delta, 0.0)
        upper_violation = np.maximum(delta - upper, 0.0)
        bound_violation = np.maximum(lower_violation, upper_violation)
        sign_violation = np.where(is_call, np.maximum(-delta, 0.0), np.maximum(delta, 0.0))
        for split, split_mask in split_map.items():
            mask = split_mask & allowed
            idx = np.flatnonzero(mask)
            if idx.size:
                bounds_rate = float(np.mean(bound_violation[idx] > DELTA_BOUND_TOLERANCE))
                sign_rate = float(np.mean(sign_violation[idx] > DELTA_BOUND_TOLERANCE))
                max_bounds = float(np.max(bound_violation[idx]))
                max_sign = float(np.max(sign_violation[idx]))
            else:
                bounds_rate = sign_rate = max_bounds = max_sign = float("nan")
            rows.append(
                {
                    "method_name": method_name,
                    "split": split,
                    "row_count": int(idx.size),
                    "bounds_violation_rate": bounds_rate,
                    "sign_violation_rate": sign_rate,
                    "max_bounds_violation": max_bounds,
                    "max_sign_violation": max_sign,
                    "delta_bound_tolerance": DELTA_BOUND_TOLERANCE,
                    "review_flag": (
                        "PASS"
                        if bounds_rate <= DELTA_BOUND_TOLERANCE and sign_rate <= DELTA_BOUND_TOLERANCE
                        else "REVIEW"
                    ),
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def delta_curve_sample_audit_rows(
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
    per_method: int = 200,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    allowed_indices = np.flatnonzero(delta_allowed_mask(bundle))
    split_index = bundle.audit_numeric[:, stage3.AUDIT_NUMERIC_INDEX["split_index"]].astype(int)
    split_names = tuple(map(str, bundle.split_names))
    is_call = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    for method_name, prediction in predictions.items():
        selected = allowed_indices
        if selected.size > per_method:
            selected = np.sort(rng.choice(selected, size=per_method, replace=False))
        for sample_id, row_index in enumerate(selected):
            rows.append(
                {
                    "method_name": method_name,
                    "sample_id": sample_id,
                    "row_index": int(row_index),
                    "regime_id": str(bundle.regime_ids[int(bundle.regime_index[row_index])]),
                    "split": split_names[int(split_index[row_index])],
                    "option_type": "call" if is_call[row_index] else "put",
                    "S_over_K": float(bundle.audit_numeric[row_index, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]]),
                    "tau_fraction": float(bundle.X[row_index, stage3.FEATURE_NAMES.index("tau_fraction")]),
                    "target_delta": float(bundle.y_delta[row_index]),
                    "predicted_delta": float(prediction.predicted_delta[row_index]),
                    "absolute_error": float(abs(prediction.predicted_delta[row_index] - bundle.y_delta[row_index])),
                    "delta_allowed_mask": bool(bundle.masks[row_index, stage3.MASK_INDEX["delta_allowed_mask"]]),
                    "strict_interior": bool(bundle.masks[row_index, stage3.MASK_INDEX["strict_interior"]]),
                    "boundary_near": bool(bundle.masks[row_index, stage3.MASK_INDEX["boundary_near"]]),
                    "payoff_kink_near": bool(bundle.masks[row_index, stage3.MASK_INDEX["payoff_kink_near"]]),
                    "maturity_row": bool(bundle.masks[row_index, stage3.MASK_INDEX["maturity_row"]]),
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def run_delta_diagnostics_experiment(
    *,
    dataset_path: Path | str = stage3.DEFAULT_DATASET_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_tex_path: Path | str = DEFAULT_REPORT_PATH,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
    create_figures: bool = True,
) -> DeltaExperimentResult:
    bundle = stage3.load_v1_dataset(dataset_path)
    price_predictions, price_histories = price_implied_delta_predictions(
        bundle,
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )
    delta_head_prediction, delta_head_history = train_supervised_delta_head(
        bundle,
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )
    predictions = dict(price_predictions)
    predictions[delta_head_prediction.method_name] = delta_head_prediction
    histories = dict(price_histories)
    histories[delta_head_prediction.method_name] = delta_head_history
    return write_delta_outputs(
        output_dir=output_dir,
        report_tex_path=report_tex_path,
        bundle=bundle,
        predictions=predictions,
        training_histories=histories,
        create_figures=create_figures,
        dataset_path=Path(dataset_path),
        train_cap=train_cap,
    )


def write_delta_outputs(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_tex_path: Path | str = DEFAULT_REPORT_PATH,
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
    training_histories: dict[str, list[dict[str, float]]],
    create_figures: bool = True,
    dataset_path: Path | str = stage3.DEFAULT_DATASET_PATH,
    train_cap: int = TRAIN_ROW_CAP,
) -> DeltaExperimentResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_tex_path = Path(report_tex_path)

    split_rows = delta_metrics_by_split_rows(bundle, predictions)
    option_rows = delta_metrics_by_option_type_rows(bundle, predictions)
    region_rows = delta_metrics_by_region_rows(bundle, predictions)
    bounds_rows = delta_bounds_violation_rows(bundle, predictions)
    audit_rows = delta_curve_sample_audit_rows(bundle, predictions)
    review_decision = stage5_review_decision(split_rows, bounds_rows)

    metrics_by_split_path = output_dir / "delta_metrics_by_split.csv"
    metrics_by_option_type_path = output_dir / "delta_metrics_by_option_type.csv"
    metrics_by_region_path = output_dir / "delta_metrics_by_region.csv"
    bounds_summary_path = output_dir / "delta_bounds_violation_summary.csv"
    curve_sample_audit_path = output_dir / "delta_curve_sample_audit.csv"
    model_manifest_path = output_dir / "delta_model_manifest.csv"

    write_csv(metrics_by_split_path, split_rows, DELTA_METRICS_BY_SPLIT_FIELDNAMES)
    write_csv(metrics_by_option_type_path, option_rows, DELTA_METRICS_BY_OPTION_TYPE_FIELDNAMES)
    write_csv(metrics_by_region_path, region_rows, DELTA_METRICS_BY_REGION_FIELDNAMES)
    write_csv(bounds_summary_path, bounds_rows, DELTA_BOUNDS_FIELDNAMES)
    write_csv(curve_sample_audit_path, audit_rows, DELTA_CURVE_SAMPLE_AUDIT_FIELDNAMES)
    write_csv(
        model_manifest_path,
        _model_manifest_rows(
            bundle=bundle,
            predictions=predictions,
            histories=training_histories,
            dataset_path=Path(dataset_path),
            train_cap=train_cap,
            review_decision=review_decision,
        ),
        DELTA_MODEL_MANIFEST_FIELDNAMES,
    )

    figure_paths: tuple[Path, ...] = ()
    if create_figures:
        figure_paths = tuple(_create_figures(output_dir, bundle, predictions, split_rows, option_rows, region_rows, bounds_rows))

    write_stage5_report(
        report_tex_path,
        bundle=bundle,
        split_rows=split_rows,
        option_rows=option_rows,
        region_rows=region_rows,
        bounds_rows=bounds_rows,
        figure_paths=figure_paths,
        review_decision=review_decision,
    )

    return DeltaExperimentResult(
        output_dir=output_dir,
        metrics_by_split_path=metrics_by_split_path,
        metrics_by_option_type_path=metrics_by_option_type_path,
        metrics_by_region_path=metrics_by_region_path,
        bounds_summary_path=bounds_summary_path,
        curve_sample_audit_path=curve_sample_audit_path,
        model_manifest_path=model_manifest_path,
        report_tex_path=report_tex_path,
        predictions=predictions,
        training_histories=training_histories,
        review_decision=review_decision,
        figure_paths=figure_paths,
    )


def stage5_review_decision(
    split_rows: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
) -> str:
    if _passing_methods(split_rows, bounds_rows):
        return "READY_FOR_INTEGRATED_WORKFLOW_STAGE"
    return "REVIEW_REQUIRED_BEFORE_INTEGRATED_STAGE"


def _passing_methods(
    split_rows: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
) -> list[str]:
    passing: list[str] = []
    methods = sorted({str(row["method_name"]) for row in split_rows})
    for method in methods:
        validation = _row_lookup(split_rows, method, "validation")
        test = _row_lookup(split_rows, method, "test")
        stress = _row_lookup(split_rows, method, "stress_holdout")
        if validation is None or test is None or stress is None:
            continue
        metric_ok = (
            float(validation["delta_rmse"]) <= 0.12
            and float(test["delta_rmse"]) <= 0.12
            and float(stress["delta_rmse"]) <= 0.18
        )
        bounds_ok = True
        for split in ("validation", "test", "stress_holdout"):
            row = _row_lookup(bounds_rows, method, split)
            if row is None:
                bounds_ok = False
                break
            bounds_ok = bounds_ok and float(row["bounds_violation_rate"]) <= DELTA_BOUND_TOLERANCE
            bounds_ok = bounds_ok and float(row["sign_violation_rate"]) <= DELTA_BOUND_TOLERANCE
        if metric_ok and bounds_ok:
            passing.append(method)
    return passing


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage5_report(
    path: Path,
    *,
    bundle: stage3.SurrogateDatasetBundle,
    split_rows: list[dict[str, Any]],
    option_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
    figure_paths: tuple[Path, ...],
    review_decision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed_count = int(np.count_nonzero(delta_allowed_mask(bundle)))
    split_table = _latex_split_table(split_rows)
    option_table = _latex_option_table(option_rows)
    region_table = _latex_region_table(region_rows)
    bounds_table = _latex_bounds_table(bounds_rows)
    figure_block = "\n".join(_latex_figure(figure_path) for figure_path in figure_paths)
    method_interpretation = _method_interpretation(split_rows, bounds_rows)
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
\lhead{{\small Stage 5 Delta Diagnostic}}
\rhead{{\small Delta Head Comparison}}
\cfoot{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\title{{\textbf{{Stage 5 Delta Diagnostic Report}}\\
\large Price-Implied Delta and a Supervised Delta-Head Baseline}}
\author{{Codex-assisted surrogate modelling review}}
\date{{June 21, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report reviews Stage 5 of the compressed roadmap. It compares Delta implied by retrained
Stage 3 price/premium surrogates against a supervised Delta-head baseline trained only on approved
finite-difference Delta diagnostic labels. No Gamma head is trained, Gamma is not used as a target,
and the existing v1 dataset is read-only. The Stage 5 review decision is \code{{{review_decision}}}.
\end{{abstract}}

\tableofcontents
\newpage

\section{{Purpose of Stage 5}}

Stage 5 asks whether the existing price/premium surrogates give coherent Delta behavior, and whether
a separate supervised Delta head is needed. Delta is more delicate than price because it is a
derivative diagnostic of the value surface, especially near payoff kinks, maturity rows, and
exercise-boundary regions. The numerical setting follows the finite-difference American-option
framework used in the solver validation reports \citep{{brennan_schwartz_1977,wilmott_howison_dewynne_1995}}.

\section{{Link to Stage 3 and Stage 4}}

Stage 3 accepted the positive-premium price surrogate for boundary diagnostics. Stage 4 showed that
boundary behavior needed a separate boundary-focused model rather than relying only on threshold
extraction from price/premium predictions. Stage 5 applies the same caution to Delta diagnostics.

\section{{Delta Target Construction and Mask Policy}}

The target \code{{y_delta}} is the finite-difference Delta stored in the v1 dataset. It is not an
analytical Greek and is not a production risk label. Primary training and evaluation use
\code{{delta_allowed_mask}} and finite \code{{y_delta}} rows, giving {allowed_count:,} allowed rows.
The strict-interior mask is the main reliable diagnostic region; boundary-near, payoff-kink-near,
and maturity rows are reported as difficult diagnostic slices.

\section{{Price-Implied Delta Method}}

The direct-value and positive-premium Stage 3 MLPs are retrained deterministically in memory. Delta
is estimated by central finite difference in log moneyness with step \(10^{{-3}}\):
\[
\widehat\Delta =
\frac{{\widehat V(\log(S/K)+h)-\widehat V(\log(S/K)-h)}}
{{\exp(\log(S/K)+h)-\exp(\log(S/K)-h)}}.
\]
For the premium model, payoff is recomputed analytically at the perturbed moneyness and value is
reconstructed as payoff plus predicted premium.

\section{{Supervised Delta-Head Method}}

The supervised Delta head uses the seven v1 features and trains only on train-split rows satisfying
\code{{delta_allowed_mask}}. Predictions are clipped to the basic diagnostic bounds: call Delta in
\([0,1]\) and put Delta in \([-1,0]\). This is a diagnostic baseline, not production Greek
infrastructure.

\section{{Metrics and Evaluation Protocol}}

Metrics include Delta MAE, RMSE, maximum absolute error, stable relative MAE, bounds violation rate,
and sign violation rate. Bounds and sign rates use tolerance \(10^{{-4}}\). Review readiness requires
validation and test RMSE at most 0.12, stress-holdout RMSE at most 0.18, and near-zero bounds/sign
violations for at least one method.

\section{{Results by Split}}

{split_table}

{method_interpretation}

\section{{Results by Option Family}}

{option_table}

\section{{Results by Diagnostic Region}}

{region_table}

\section{{Bounds and Sign Diagnostics}}

{bounds_table}

\section{{Figures}}

{figure_block}

\section{{Limitations}}

Delta labels are finite-difference diagnostics from the CN/PSOR dataset. Good price or premium
performance does not automatically imply good Delta behavior. Boundary-near, payoff-kink-near, and
maturity rows remain delicate. Gamma is deliberately excluded from training and remains blocked from
this stage.

\section{{What Stage 5 Supports}}

Stage 5 supports comparing price-implied Delta diagnostics with a supervised Delta-head baseline
under the existing v1 masks and regime-level splits. If the supervised head is the passing method,
the evidence supports using a separate Delta-focused diagnostic model.

\section{{What Stage 5 Does Not Support}}

This stage does not support production Greek reliability, Gamma-head training, exact analytical
Greek claims, final paper claims, solver changes, or v1 dataset regeneration.

\section{{Recommended Next Stage}}

If the review decision is \code{{READY_FOR_INTEGRATED_WORKFLOW_STAGE}}, the next stage may integrate
the price, boundary, and Delta diagnostic components into a workflow review. If the decision is
\code{{REVIEW_REQUIRED_BEFORE_INTEGRATED_STAGE}}, human review should inspect the Delta failures
before adding new components.

\bibliographystyle{{plainnat}}
\bibliography{{reports/03_solver/references}}

\end{{document}}
""",
        encoding="utf-8",
    )


def _predict_direct_value_for_features(
    X: np.ndarray,
    preprocessor: stage3.PredictionPreprocessor,
    model: Any,
) -> np.ndarray:
    return stage3._batched_predict(  # noqa: SLF001
        model,
        X,
        lambda slc: preprocessor.transform_direct(X[slc]).astype(np.float32),
        batch_size=131_072,
    )


def _predict_premium_value_for_features(
    X: np.ndarray,
    preprocessor: stage3.PredictionPreprocessor,
    model: Any,
) -> np.ndarray:
    payoff = _analytic_payoff_over_K(X)
    premium = stage3._batched_predict(  # noqa: SLF001
        model,
        X,
        lambda slc: preprocessor.transform_premium(X[slc], payoff[slc]).astype(np.float32),
        batch_size=131_072,
    )
    return payoff + premium


def _analytic_payoff_over_K(X: np.ndarray) -> np.ndarray:
    moneyness = np.exp(X[:, stage3.FEATURE_NAMES.index("log_moneyness")])
    is_call = X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    return np.where(is_call, np.maximum(moneyness - 1.0, 0.0), np.maximum(1.0 - moneyness, 0.0))


def _metric_row(
    fieldnames: list[str],
    bundle: stage3.SurrogateDatasetBundle,
    prediction: DeltaPrediction,
    *,
    method_name: str,
    split: str,
    region: str | None,
    mask: np.ndarray,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    metrics = _error_metrics(bundle.y_delta[idx], prediction.predicted_delta[idx])
    row: dict[str, Any] = {
        "method_name": method_name,
        "split": split,
        "row_count": int(idx.size),
        "delta_mae": metrics["mae"],
        "delta_rmse": metrics["rmse"],
        "delta_max_abs_error": metrics["max_abs_error"],
        "delta_stable_relative_mae": metrics["stable_relative_mae"],
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }
    if "region" in fieldnames:
        row["region"] = region
    if "relative_error_denominator_floor" in fieldnames:
        row["relative_error_denominator_floor"] = RELATIVE_ERROR_DENOMINATOR_FLOOR
    return {name: row[name] for name in fieldnames}


def _option_metric_row(
    bundle: stage3.SurrogateDatasetBundle,
    prediction: DeltaPrediction,
    *,
    method_name: str,
    option_type: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    metrics = _error_metrics(bundle.y_delta[idx], prediction.predicted_delta[idx])
    return {
        "method_name": method_name,
        "option_type": option_type,
        "row_count": int(idx.size),
        "delta_mae": metrics["mae"],
        "delta_rmse": metrics["rmse"],
        "delta_max_abs_error": metrics["max_abs_error"],
        "delta_stable_relative_mae": metrics["stable_relative_mae"],
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _error_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs_error": float("nan"), "stable_relative_mae": float("nan")}
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    abs_error = np.abs(error)
    denominator = np.maximum(np.abs(target), RELATIVE_ERROR_DENOMINATOR_FLOOR)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(abs_error)),
        "stable_relative_mae": float(np.mean(abs_error / denominator)),
    }


def _row_lookup(rows: list[dict[str, Any]], method_name: str, split: str) -> dict[str, Any] | None:
    for row in rows:
        if row["method_name"] == method_name and row.get("split") == split:
            return row
    return None


def _model_manifest_rows(
    *,
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
    histories: dict[str, list[dict[str, float]]],
    dataset_path: Path,
    train_cap: int,
    review_decision: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in predictions:
        history = histories.get(method_name, [])
        final = history[-1] if history else {"train_loss": float("nan"), "validation_loss": float("nan")}
        if method_name == "supervised_delta_head":
            target = "y_delta on train split and delta_allowed_mask rows"
            training_rows = int(delta_training_indices(bundle).size)
            family = "small fixed MLP with call/put Delta bounds clipping"
            inputs = "X"
        else:
            target = "value or premium model; Delta extracted by finite difference"
            training_rows = train_cap
            family = "retrained Stage 3 MLP plus central log-moneyness finite difference"
            inputs = "X" if method_name == "direct_value_implied_delta" else "X plus analytic payoff reconstruction"
        rows.append(
            {
                "run_id": "stage5_delta_diagnostic",
                "method_name": method_name,
                "dataset_path": str(dataset_path),
                "random_seed": RANDOM_SEED,
                "train_row_cap": train_cap,
                "stage3_models_retrained": method_name != "supervised_delta_head",
                "input_columns": inputs,
                "target_name": target,
                "training_rows": training_rows,
                "method_family": family,
                "finite_difference_step": LOG_MONEYNESS_STEP,
                "mask_policy": "primary rows require delta_allowed_mask; difficult regions reported separately",
                "model_weights_saved": "no",
                "final_train_loss": final["train_loss"],
                "final_validation_loss": final["validation_loss"],
                "review_decision": review_decision,
                "downstream_use_status": DOWNSTREAM_USE_STATUS,
            }
        )
    return rows


def _method_interpretation(
    split_rows: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
) -> str:
    passing = set(_passing_methods(split_rows, bounds_rows))
    if "supervised_delta_head" in passing:
        return (
            "The Stage 5 gate is satisfied by \\code{supervised_delta_head}. The price-implied "
            "Delta methods remain useful diagnostics, but they do not replace the supervised Delta "
            "head under the combined RMSE and bounds/sign checks. This supports using a separate "
            "Delta-focused diagnostic model rather than assuming price or premium accuracy is enough "
            "for Greek behavior."
        )
    if passing:
        methods = ", ".join(rf"\code{{{method}}}" for method in sorted(passing))
        return (
            f"The Stage 5 gate is satisfied by {methods}. The supervised Delta head remains a "
            "diagnostic comparison rather than production Greek infrastructure."
        )
    return (
        "No Stage 5 method satisfies the combined RMSE and bounds/sign checks. The appropriate "
        "decision is human review before integrating Delta diagnostics into a broader workflow."
    )


def _create_figures(
    output_dir: Path,
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
    split_rows: list[dict[str, Any]],
    option_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
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

    paths.append(_bar_figure(figure_dir / "delta_error_by_split.png", split_rows, predictions, "split", ("train", "validation", "test", "stress_holdout"), "delta_rmse", "Delta RMSE by Split"))
    paths.append(_bar_figure(figure_dir / "delta_error_by_option_type.png", option_rows, predictions, "option_type", ("put", "call"), "delta_rmse", "Delta RMSE by Option Type"))
    region_subset = [row for row in region_rows if row["split"] == "test"]
    paths.append(_bar_figure(figure_dir / "delta_error_by_region.png", region_subset, predictions, "region", ("all_allowed", "strict_interior", "boundary_near", "payoff_kink_near", "maturity_row"), "delta_rmse", "Delta RMSE by Region on Test Split"))
    paths.append(_sample_delta_curve_figure(figure_dir / "sample_put_delta_curves.png", bundle, predictions, "put"))
    paths.append(_sample_delta_curve_figure(figure_dir / "sample_call_delta_curves.png", bundle, predictions, "call"))
    paths.append(_bounds_figure(figure_dir / "delta_bounds_violation_comparison.png", bounds_rows, predictions))
    return paths


def _bar_figure(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: dict[str, DeltaPrediction],
    group_key: str,
    group_values: tuple[str, ...],
    metric: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    x = np.arange(len(group_values))
    width = 0.24
    plt.figure(figsize=(9, 5.0))
    methods = tuple(predictions)
    for offset, method_name in enumerate(methods):
        values = [_metric_value(rows, method_name, group_key, group, metric) for group in group_values]
        plt.bar(x + (offset - (len(methods) - 1) / 2) * width, values, width=width, label=method_name)
    plt.xticks(x, group_values, rotation=20)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _sample_delta_curve_figure(
    path: Path,
    bundle: stage3.SurrogateDatasetBundle,
    predictions: dict[str, DeltaPrediction],
    option_type: str,
) -> Path:
    import matplotlib.pyplot as plt

    is_call = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
    option_mask = is_call if option_type == "call" else ~is_call
    allowed = delta_allowed_mask(bundle)
    candidate_regimes = []
    for regime_index, regime_id in enumerate(bundle.regime_ids):
        mask = option_mask & allowed & (bundle.regime_index == regime_index)
        if np.count_nonzero(mask) >= 80:
            candidate_regimes.append(str(regime_id))
        if len(candidate_regimes) >= 2:
            break
    plt.figure(figsize=(7.8, 5.0))
    for regime_id in candidate_regimes:
        regime_index = int(np.where(bundle.regime_ids.astype(str) == regime_id)[0][0])
        mask = option_mask & allowed & (bundle.regime_index == regime_index)
        tau = bundle.X[:, stage3.FEATURE_NAMES.index("tau_fraction")]
        chosen_tau = float(np.median(tau[mask]))
        slice_mask = mask & (np.abs(tau - chosen_tau) <= 1e-12)
        if np.count_nonzero(slice_mask) < 5:
            continue
        order = np.argsort(bundle.audit_numeric[slice_mask, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]])
        indices = np.flatnonzero(slice_mask)[order]
        moneyness = bundle.audit_numeric[indices, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]]
        plt.plot(moneyness, bundle.y_delta[indices], linewidth=1.7, label=f"{regime_id} target")
        for method_name, prediction in predictions.items():
            plt.plot(moneyness, prediction.predicted_delta[indices], linestyle="--", linewidth=1, label=f"{regime_id} {method_name}")
    plt.xlabel("S / K")
    plt.ylabel("Delta")
    plt.title(f"Sample {option_type.title()} Delta Curves")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _bounds_figure(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: dict[str, DeltaPrediction],
) -> Path:
    import matplotlib.pyplot as plt

    split_names = ("validation", "test", "stress_holdout")
    x = np.arange(len(split_names))
    width = 0.24
    plt.figure(figsize=(8.4, 4.8))
    methods = tuple(predictions)
    for offset, method_name in enumerate(methods):
        values = [_metric_value(rows, method_name, "split", split, "bounds_violation_rate") for split in split_names]
        plt.bar(x + (offset - (len(methods) - 1) / 2) * width, values, width=width, label=method_name)
    plt.axhline(DELTA_BOUND_TOLERANCE, color="black", linestyle="--", linewidth=1, label="tolerance")
    plt.xticks(x, split_names, rotation=20)
    plt.ylabel("Bounds violation rate")
    plt.title("Delta Bounds Violation Comparison")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _metric_value(
    rows: list[dict[str, Any]],
    method_name: str,
    group_key: str,
    group_value: str,
    metric: str,
) -> float:
    for row in rows:
        if row["method_name"] == method_name and row.get(group_key) == group_value:
            value = float(row[metric])
            return value if np.isfinite(value) else 0.0
    return 0.0


def _latex_figure(figure_path: Path) -> str:
    relative = figure_path.relative_to(PROJECT_ROOT)
    caption = figure_path.stem.replace("_", " ").capitalize()
    return rf"""\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\textwidth]{{{relative}}}
\caption{{{caption}.}}
\end{{figure}}
"""


def _latex_split_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.26\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Method & Split & Rows & RMSE & MAE \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        if row["split"] in {"validation", "test", "stress_holdout"}:
            lines.append(rf"{_escape(row['method_name'])} & {_escape(row['split'])} & {int(row['row_count']):,} & {_fmt(row['delta_rmse'])} & {_fmt(row['delta_mae'])} \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_option_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.28\textwidth} L{0.14\textwidth} L{0.16\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Method & Option & RMSE & MAE \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(rf"{_escape(row['method_name'])} & {_escape(row['option_type'])} & {_fmt(row['delta_rmse'])} & {_fmt(row['delta_mae'])} \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_region_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.25\textwidth} L{0.16\textwidth} L{0.19\textwidth} L{0.12\textwidth} L{0.13\textwidth}}",
        r"\toprule",
        r"Method & Split & Region & Rows & RMSE \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        if row["split"] in {"test", "stress_holdout"}:
            lines.append(rf"{_escape(row['method_name'])} & {_escape(row['split'])} & {_escape(row['region'])} & {int(row['row_count']):,} & {_fmt(row['delta_rmse'])} \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_bounds_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.28\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.12\textwidth}}",
        r"\toprule",
        r"Method & Split & Bounds rate & Sign rate & Review \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        if row["split"] in {"validation", "test", "stress_holdout"}:
            lines.append(rf"{_escape(row['method_name'])} & {_escape(row['split'])} & {_fmt(row['bounds_violation_rate'])} & {_fmt(row['sign_violation_rate'])} & {_escape(row['review_flag'])} \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _escape(value: Any) -> str:
    return str(value).replace("_", r"\_")
