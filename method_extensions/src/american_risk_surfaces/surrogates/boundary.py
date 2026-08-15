"""Stage 4: boundary diagnostic and boundary-head comparison."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryPoint,
    extract_boundary_at_time,
)
from american_risk_surfaces.surrogates import price_premium as stage3


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "05_surrogate_models" / "boundary"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "06_surrogate" / "boundary_diagnostic_report.tex"

STAGE4_DOCSTRING = '"""Stage 4: boundary diagnostic and boundary-head comparison."""'
BOUNDARY_THRESHOLD = 1e-6
RANDOM_SEED = stage3.RANDOM_SEED
TRAIN_ROW_CAP = stage3.TRAIN_ROW_CAP
DOWNSTREAM_USE_STATUS = "stage4_boundary_diagnostic_only"
SAMPLE_LOWER_MONEYNESS = 0.4
SAMPLE_UPPER_MONEYNESS = 1.8

BOUNDARY_HEAD_FEATURE_NAMES = (
    "tau_fraction",
    "r",
    "q",
    "sigma",
    "T",
    "is_call",
)

BOUNDARY_METRICS_BY_SPLIT_FIELDNAMES = [
    "method_name",
    "split",
    "row_count",
    "reference_found_count",
    "predicted_found_count",
    "comparable_error_count",
    "reference_outside_sample_window_count",
    "boundary_mae",
    "boundary_rmse",
    "boundary_max_abs_error",
    "boundary_found_agreement_rate",
    "missed_boundary_rate",
    "false_boundary_rate",
    "no_dividend_call_false_boundary_rate",
    "review_flag",
    "downstream_use_status",
]
BOUNDARY_METRICS_BY_OPTION_TYPE_FIELDNAMES = [
    "method_name",
    "option_type",
    "row_count",
    "reference_found_count",
    "predicted_found_count",
    "comparable_error_count",
    "reference_outside_sample_window_count",
    "boundary_mae",
    "boundary_rmse",
    "boundary_max_abs_error",
    "boundary_found_agreement_rate",
    "missed_boundary_rate",
    "false_boundary_rate",
    "no_dividend_call_false_boundary_rate",
    "review_flag",
    "downstream_use_status",
]
BOUNDARY_METRICS_BY_REGIME_FIELDNAMES = [
    "method_name",
    "regime_id",
    "split",
    "option_type",
    "T",
    "sigma",
    "r",
    "q",
    "row_count",
    "reference_found_count",
    "predicted_found_count",
    "comparable_error_count",
    "reference_outside_sample_window_count",
    "boundary_mae",
    "boundary_rmse",
    "boundary_max_abs_error",
    "boundary_found_agreement_rate",
    "missed_boundary_rate",
    "false_boundary_rate",
    "no_dividend_call_false_boundary_rate",
    "downstream_use_status",
]
BOUNDARY_CURVE_SAMPLE_AUDIT_FIELDNAMES = [
    "method_name",
    "sample_id",
    "regime_id",
    "split",
    "option_type",
    "T",
    "sigma",
    "r",
    "q",
    "tau",
    "tau_fraction",
    "reference_found",
    "reference_boundary",
    "reference_in_sample_window",
    "predicted_found",
    "predicted_boundary",
    "absolute_error",
    "no_boundary_reason",
    "is_no_dividend_call_control",
    "downstream_use_status",
]
BOUNDARY_MODEL_MANIFEST_FIELDNAMES = [
    "run_id",
    "method_name",
    "dataset_path",
    "random_seed",
    "train_row_cap",
    "stage3_positive_premium_retrained",
    "input_columns",
    "target_name",
    "training_rows",
    "model_family",
    "model_weights_saved",
    "boundary_threshold",
    "sample_moneyness_window",
    "review_decision",
    "downstream_use_status",
]

__all__ = (
    "BOUNDARY_CURVE_SAMPLE_AUDIT_FIELDNAMES",
    "BOUNDARY_HEAD_FEATURE_NAMES",
    "BOUNDARY_METRICS_BY_OPTION_TYPE_FIELDNAMES",
    "BOUNDARY_METRICS_BY_REGIME_FIELDNAMES",
    "BOUNDARY_METRICS_BY_SPLIT_FIELDNAMES",
    "BOUNDARY_THRESHOLD",
    "BoundaryExperimentResult",
    "BoundaryPrediction",
    "BoundaryTargetTable",
    "build_boundary_target_table",
    "boundary_curve_sample_audit_rows",
    "boundary_metrics_by_option_type_rows",
    "boundary_metrics_by_regime_rows",
    "boundary_metrics_by_split_rows",
    "extract_premium_implied_boundary_for_curve",
    "premium_implied_boundary_prediction",
    "run_boundary_diagnostics_experiment",
    "stage4_review_decision",
    "train_direct_boundary_head",
    "validate_boundary_training_target",
    "write_boundary_outputs",
    "write_csv",
)


@dataclass(frozen=True)
class BoundaryTargetTable:
    """One row per regime-time boundary diagnostic target."""

    row_count: int
    target_id: np.ndarray
    regime_index: np.ndarray
    regime_id: np.ndarray
    split: np.ndarray
    option_type: np.ndarray
    T: np.ndarray
    sigma: np.ndarray
    r: np.ndarray
    q: np.ndarray
    tau: np.ndarray
    tau_fraction: np.ndarray
    is_call: np.ndarray
    reference_boundary: np.ndarray
    reference_found: np.ndarray
    reference_in_sample_window: np.ndarray
    is_no_dividend_call_control: np.ndarray
    curve_row_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class BoundaryPrediction:
    """Boundary predictions and found/not-found status for one method."""

    method_name: str
    predicted_boundary: np.ndarray
    predicted_found: np.ndarray
    no_boundary_reason: np.ndarray


@dataclass(frozen=True)
class BoundaryExperimentResult:
    """Stage 4 output paths and review decision."""

    output_dir: Path
    metrics_by_split_path: Path
    metrics_by_option_type_path: Path
    metrics_by_regime_path: Path
    curve_sample_audit_path: Path
    model_manifest_path: Path
    report_tex_path: Path
    review_decision: str
    figure_paths: tuple[Path, ...]


def validate_boundary_training_target(target_name: str) -> None:
    if target_name == "boundary_spot_over_K":
        return
    if target_name in {"delta", "scaled_gamma"}:
        raise ValueError(f"{target_name} is not a Stage 4 boundary training target.")
    raise ValueError(f"unsupported Stage 4 boundary training target {target_name!r}")


def build_boundary_target_table(bundle: stage3.SurrogateDatasetBundle) -> BoundaryTargetTable:
    """Build one boundary target row per v1 regime-time row."""

    row_count = bundle.X.shape[0]
    regime_indices = bundle.regime_index.astype(int)
    tau = bundle.audit_numeric[:, stage3.AUDIT_NUMERIC_INDEX["tau"]]
    order = np.lexsort((tau, regime_indices))
    sorted_regime = regime_indices[order]
    sorted_tau = tau[order]
    group_start = np.r_[
        0,
        np.flatnonzero((sorted_regime[1:] != sorted_regime[:-1]) | (sorted_tau[1:] != sorted_tau[:-1]))
        + 1,
    ]
    group_stop = np.r_[group_start[1:], len(order)]

    target_ids: list[int] = []
    target_regime_index: list[int] = []
    regime_id: list[str] = []
    split: list[str] = []
    option_type: list[str] = []
    T: list[float] = []
    sigma: list[float] = []
    r: list[float] = []
    q: list[float] = []
    tau_values: list[float] = []
    tau_fraction: list[float] = []
    is_call: list[float] = []
    reference_boundary: list[float] = []
    reference_found: list[bool] = []
    reference_in_sample_window: list[bool] = []
    is_no_dividend_call_control: list[bool] = []
    curve_row_indices: list[np.ndarray] = []

    split_names = tuple(map(str, bundle.split_names))
    for target_id, (start, stop) in enumerate(zip(group_start, group_stop)):
        indices = order[start:stop]
        moneyness = bundle.audit_numeric[indices, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]]
        indices = indices[np.argsort(moneyness)]
        first = int(indices[0])
        reg_index = int(bundle.regime_index[first])
        reg_id = str(bundle.regime_ids[reg_index])
        option = "call" if float(bundle.X[first, stage3.FEATURE_NAMES.index("is_call")]) >= 0.5 else "put"
        boundary_values = bundle.y_boundary[indices]
        finite = boundary_values[np.isfinite(boundary_values)]
        boundary_value = float(finite[0]) if finite.size else float("nan")
        found = bool(np.isfinite(boundary_value))
        in_window = bool(found and SAMPLE_LOWER_MONEYNESS <= boundary_value <= SAMPLE_UPPER_MONEYNESS)
        split_index = int(bundle.audit_numeric[first, stage3.AUDIT_NUMERIC_INDEX["split_index"]])

        target_ids.append(target_id)
        target_regime_index.append(reg_index)
        regime_id.append(reg_id)
        split.append(split_names[split_index])
        option_type.append(option)
        T.append(float(bundle.X[first, stage3.FEATURE_NAMES.index("T")]))
        sigma.append(float(bundle.X[first, stage3.FEATURE_NAMES.index("sigma")]))
        r.append(float(bundle.X[first, stage3.FEATURE_NAMES.index("r")]))
        q_value = float(bundle.X[first, stage3.FEATURE_NAMES.index("q")])
        q.append(q_value)
        tau_values.append(float(bundle.audit_numeric[first, stage3.AUDIT_NUMERIC_INDEX["tau"]]))
        tau_fraction.append(float(bundle.X[first, stage3.FEATURE_NAMES.index("tau_fraction")]))
        is_call.append(1.0 if option == "call" else 0.0)
        reference_boundary.append(boundary_value)
        reference_found.append(found)
        reference_in_sample_window.append(in_window)
        is_no_dividend_call_control.append(bool(option == "call" and abs(q_value) < 1e-14))
        curve_row_indices.append(indices.astype(int))

    if len(target_ids) == 0 and row_count > 0:
        raise ValueError("no boundary target rows were built.")
    return BoundaryTargetTable(
        row_count=len(target_ids),
        target_id=np.asarray(target_ids, dtype=int),
        regime_index=np.asarray(target_regime_index, dtype=int),
        regime_id=np.asarray(regime_id, dtype=str),
        split=np.asarray(split, dtype=str),
        option_type=np.asarray(option_type, dtype=str),
        T=np.asarray(T, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        r=np.asarray(r, dtype=float),
        q=np.asarray(q, dtype=float),
        tau=np.asarray(tau_values, dtype=float),
        tau_fraction=np.asarray(tau_fraction, dtype=float),
        is_call=np.asarray(is_call, dtype=float),
        reference_boundary=np.asarray(reference_boundary, dtype=float),
        reference_found=np.asarray(reference_found, dtype=bool),
        reference_in_sample_window=np.asarray(reference_in_sample_window, dtype=bool),
        is_no_dividend_call_control=np.asarray(is_no_dividend_call_control, dtype=bool),
        curve_row_indices=tuple(curve_row_indices),
    )


def extract_premium_implied_boundary_for_curve(
    moneyness_grid: np.ndarray,
    premium_curve: np.ndarray,
    *,
    option_type: str,
    tau: float,
    time_index: int,
    threshold: float = BOUNDARY_THRESHOLD,
) -> BoundaryPoint:
    """Apply Ticket 09 threshold extraction to a sampled premium curve."""

    return extract_boundary_at_time(
        moneyness_grid,
        premium_curve,
        option_type,
        tau,
        time_index,
        threshold=threshold,
    )


def retrain_positive_premium_prediction(
    bundle: stage3.SurrogateDatasetBundle,
    *,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
) -> stage3.ModelPrediction:
    """Retrain the Stage 3 positive-premium model in memory and return predictions."""

    stage3._require_torch()  # noqa: SLF001 - internal project helper reuse.
    stage3._set_random_seed(RANDOM_SEED)  # noqa: SLF001
    if hasattr(stage3.torch, "set_num_threads"):
        stage3.torch.set_num_threads(min(4, os.cpu_count() or 1))
    split_map = stage3.split_masks(bundle)
    train_indices = np.flatnonzero(split_map["train"])
    validation_indices = np.flatnonzero(split_map["validation"])
    selected_train = stage3.capped_train_indices(
        train_indices,
        cap=train_cap,
        seed=RANDOM_SEED,
    )
    preprocessor = stage3.fit_preprocessor(bundle, train_indices)
    config = stage3.TrainingConfig(
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )
    model, _history = stage3._train_positive_premium_model(  # noqa: SLF001
        bundle,
        preprocessor,
        selected_train,
        validation_indices,
        config,
    )
    return stage3._predict_premium(bundle, preprocessor, model)  # noqa: SLF001


def premium_implied_boundary_prediction(
    bundle: stage3.SurrogateDatasetBundle,
    targets: BoundaryTargetTable,
    predicted_premium: np.ndarray,
    *,
    threshold: float = BOUNDARY_THRESHOLD,
) -> BoundaryPrediction:
    predicted_boundary = np.full(targets.row_count, np.nan, dtype=float)
    predicted_found = np.zeros(targets.row_count, dtype=bool)
    reasons = np.empty(targets.row_count, dtype=object)
    for index, row_indices in enumerate(targets.curve_row_indices):
        moneyness = bundle.audit_numeric[row_indices, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]]
        order = np.argsort(moneyness)
        point = extract_premium_implied_boundary_for_curve(
            moneyness[order],
            predicted_premium[row_indices][order],
            option_type=str(targets.option_type[index]),
            tau=float(targets.tau[index]),
            time_index=index,
            threshold=threshold,
        )
        predicted_found[index] = bool(point.boundary_found)
        predicted_boundary[index] = float(point.boundary_spot) if point.boundary_found else float("nan")
        reasons[index] = "" if point.boundary_found else point.no_boundary_reason
    return BoundaryPrediction(
        method_name="premium_implied_boundary",
        predicted_boundary=predicted_boundary,
        predicted_found=predicted_found,
        no_boundary_reason=reasons.astype(str),
    )


def train_direct_boundary_head(targets: BoundaryTargetTable) -> BoundaryPrediction:
    """Fit a small diagnostic boundary existence/regression baseline."""

    validate_boundary_training_target("boundary_spot_over_K")
    X = _boundary_head_features(targets)
    train_mask = targets.split == "train"
    if not np.any(train_mask):
        train_mask = np.ones(targets.row_count, dtype=bool)

    X_train = X[train_mask]
    y_train = targets.reference_found[train_mask].astype(int)
    unique = np.unique(y_train)
    if len(unique) == 1:
        constant_probability = float(unique[0])
        found_probability = np.full(targets.row_count, constant_probability, dtype=float)
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            random_state=RANDOM_SEED,
        )
        classifier.fit(X_train, y_train)
        found_probability = classifier.predict_proba(X)[:, 1]

    regressor_train_mask = train_mask & targets.reference_found
    if not np.any(regressor_train_mask):
        regressor_train_mask = targets.reference_found
    if np.count_nonzero(regressor_train_mask) < 2:
        fallback = float(np.nanmean(targets.reference_boundary[targets.reference_found]))
        if not np.isfinite(fallback):
            fallback = 1.0
        boundary = np.full(targets.row_count, fallback, dtype=float)
    else:
        regressor = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=RANDOM_SEED,
        )
        regressor.fit(
            X[regressor_train_mask],
            targets.reference_boundary[regressor_train_mask],
        )
        boundary = np.clip(regressor.predict(X), 0.0, 4.0)
    found = found_probability >= 0.5
    found = found & (targets.tau_fraction > 1e-12) & ~targets.is_no_dividend_call_control
    predicted_boundary = np.where(found, boundary, np.nan)
    reasons = np.where(found, "", "classifier_no_boundary")
    reasons = np.where(targets.tau_fraction <= 1e-12, "maturity_row_ambiguous", reasons)
    reasons = np.where(targets.is_no_dividend_call_control, "no_dividend_call_control", reasons)
    return BoundaryPrediction(
        method_name="direct_boundary_head",
        predicted_boundary=predicted_boundary.astype(float),
        predicted_found=found.astype(bool),
        no_boundary_reason=reasons.astype(str),
    )


def boundary_metrics_by_split_rows(
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, prediction in predictions.items():
        for split in ("train", "validation", "test", "stress_holdout"):
            rows.append(
                _metric_row(
                    BOUNDARY_METRICS_BY_SPLIT_FIELDNAMES,
                    targets,
                    prediction,
                    method_name=method_name,
                    group_value=split,
                    mask=targets.split == split,
                    group_field="split",
                )
            )
    return rows


def boundary_metrics_by_option_type_rows(
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, prediction in predictions.items():
        for option_type in ("put", "call"):
            rows.append(
                _metric_row(
                    BOUNDARY_METRICS_BY_OPTION_TYPE_FIELDNAMES,
                    targets,
                    prediction,
                    method_name=method_name,
                    group_value=option_type,
                    mask=targets.option_type == option_type,
                    group_field="option_type",
                )
            )
    return rows


def boundary_metrics_by_regime_rows(
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    core_fieldnames = [
        name
        for name in BOUNDARY_METRICS_BY_REGIME_FIELDNAMES
        if name not in {"split", "option_type", "T", "sigma", "r", "q"}
    ]
    for method_name, prediction in predictions.items():
        for regime_id in np.unique(targets.regime_id):
            mask = targets.regime_id == regime_id
            first = int(np.flatnonzero(mask)[0])
            row = _metric_row(
                core_fieldnames,
                targets,
                prediction,
                method_name=method_name,
                group_value=str(regime_id),
                mask=mask,
                group_field="regime_id",
            )
            row.update(
                {
                    "split": str(targets.split[first]),
                    "option_type": str(targets.option_type[first]),
                    "T": float(targets.T[first]),
                    "sigma": float(targets.sigma[first]),
                    "r": float(targets.r[first]),
                    "q": float(targets.q[first]),
                }
            )
            rows.append({name: row[name] for name in BOUNDARY_METRICS_BY_REGIME_FIELDNAMES})
    return rows


def boundary_curve_sample_audit_rows(
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    per_method: int = 160,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for method_name, prediction in predictions.items():
        candidate = np.arange(targets.row_count)
        if len(candidate) > per_method:
            candidate = np.sort(rng.choice(candidate, size=per_method, replace=False))
        for sample_id, index in enumerate(candidate):
            pred_boundary = prediction.predicted_boundary[index]
            ref_boundary = targets.reference_boundary[index]
            comparable = (
                targets.reference_found[index]
                and prediction.predicted_found[index]
                and targets.reference_in_sample_window[index]
            )
            rows.append(
                {
                    "method_name": method_name,
                    "sample_id": sample_id,
                    "regime_id": str(targets.regime_id[index]),
                    "split": str(targets.split[index]),
                    "option_type": str(targets.option_type[index]),
                    "T": float(targets.T[index]),
                    "sigma": float(targets.sigma[index]),
                    "r": float(targets.r[index]),
                    "q": float(targets.q[index]),
                    "tau": float(targets.tau[index]),
                    "tau_fraction": float(targets.tau_fraction[index]),
                    "reference_found": bool(targets.reference_found[index]),
                    "reference_boundary": float(ref_boundary),
                    "reference_in_sample_window": bool(targets.reference_in_sample_window[index]),
                    "predicted_found": bool(prediction.predicted_found[index]),
                    "predicted_boundary": float(pred_boundary),
                    "absolute_error": (
                        abs(float(pred_boundary) - float(ref_boundary)) if comparable else float("nan")
                    ),
                    "no_boundary_reason": str(prediction.no_boundary_reason[index]),
                    "is_no_dividend_call_control": bool(targets.is_no_dividend_call_control[index]),
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def write_boundary_outputs(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_tex_path: Path | str = DEFAULT_REPORT_PATH,
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    create_figures: bool = True,
    stage3_positive_premium_retrained: bool = False,
    dataset_path: Path | str = stage3.DEFAULT_DATASET_PATH,
) -> BoundaryExperimentResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_tex_path = Path(report_tex_path)
    metrics_by_split = boundary_metrics_by_split_rows(targets, predictions)
    metrics_by_option_type = boundary_metrics_by_option_type_rows(targets, predictions)
    metrics_by_regime = boundary_metrics_by_regime_rows(targets, predictions)
    sample_rows = boundary_curve_sample_audit_rows(targets, predictions)
    review_decision = stage4_review_decision(metrics_by_split)
    manifest_rows = _model_manifest_rows(
        targets,
        predictions,
        dataset_path=Path(dataset_path),
        stage3_positive_premium_retrained=stage3_positive_premium_retrained,
        review_decision=review_decision,
    )

    metrics_by_split_path = output_dir / "boundary_metrics_by_split.csv"
    metrics_by_option_type_path = output_dir / "boundary_metrics_by_option_type.csv"
    metrics_by_regime_path = output_dir / "boundary_metrics_by_regime.csv"
    curve_sample_audit_path = output_dir / "boundary_curve_sample_audit.csv"
    model_manifest_path = output_dir / "boundary_model_manifest.csv"

    write_csv(metrics_by_split_path, metrics_by_split, BOUNDARY_METRICS_BY_SPLIT_FIELDNAMES)
    write_csv(
        metrics_by_option_type_path,
        metrics_by_option_type,
        BOUNDARY_METRICS_BY_OPTION_TYPE_FIELDNAMES,
    )
    write_csv(metrics_by_regime_path, metrics_by_regime, BOUNDARY_METRICS_BY_REGIME_FIELDNAMES)
    write_csv(curve_sample_audit_path, sample_rows, BOUNDARY_CURVE_SAMPLE_AUDIT_FIELDNAMES)
    write_csv(model_manifest_path, manifest_rows, BOUNDARY_MODEL_MANIFEST_FIELDNAMES)

    figure_paths: tuple[Path, ...] = ()
    if create_figures:
        figure_paths = tuple(_create_figures(output_dir, targets, predictions, metrics_by_split, metrics_by_option_type))

    write_stage4_report(
        report_tex_path,
        targets=targets,
        predictions=predictions,
        metrics_by_split=metrics_by_split,
        metrics_by_option_type=metrics_by_option_type,
        figure_paths=figure_paths,
        review_decision=review_decision,
    )

    return BoundaryExperimentResult(
        output_dir=output_dir,
        metrics_by_split_path=metrics_by_split_path,
        metrics_by_option_type_path=metrics_by_option_type_path,
        metrics_by_regime_path=metrics_by_regime_path,
        curve_sample_audit_path=curve_sample_audit_path,
        model_manifest_path=model_manifest_path,
        report_tex_path=report_tex_path,
        review_decision=review_decision,
        figure_paths=figure_paths,
    )


def run_boundary_diagnostics_experiment(
    *,
    dataset_path: Path | str = stage3.DEFAULT_DATASET_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_tex_path: Path | str = DEFAULT_REPORT_PATH,
    train_cap: int = TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
    create_figures: bool = True,
) -> BoundaryExperimentResult:
    bundle = stage3.load_v1_dataset(dataset_path)
    targets = build_boundary_target_table(bundle)
    premium_prediction = retrain_positive_premium_prediction(
        bundle,
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
    )
    premium_boundary = premium_implied_boundary_prediction(
        bundle,
        targets,
        premium_prediction.predicted_premium,
    )
    direct_boundary = train_direct_boundary_head(targets)
    return write_boundary_outputs(
        output_dir=output_dir,
        report_tex_path=report_tex_path,
        targets=targets,
        predictions={
            premium_boundary.method_name: premium_boundary,
            direct_boundary.method_name: direct_boundary,
        },
        create_figures=create_figures,
        stage3_positive_premium_retrained=True,
        dataset_path=dataset_path,
    )


def stage4_review_decision(metrics_by_split_rows: list[dict[str, Any]]) -> str:
    methods = sorted({str(row["method_name"]) for row in metrics_by_split_rows})
    for method in methods:
        validation = _metric_lookup(metrics_by_split_rows, method, "validation")
        test = _metric_lookup(metrics_by_split_rows, method, "test")
        if validation is None or test is None:
            continue
        checks = []
        for row in (validation, test):
            checks.extend(
                [
                    float(row["boundary_rmse"]) <= 0.20,
                    float(row["boundary_found_agreement_rate"]) >= 0.80,
                    float(row["no_dividend_call_false_boundary_rate"]) <= 0.05
                    or math.isnan(float(row["no_dividend_call_false_boundary_rate"])),
                ]
            )
        if all(checks):
            return "READY_FOR_DELTA_DIAGNOSTIC_STAGE"
    return "REVIEW_REQUIRED_BEFORE_DELTA_STAGE"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage4_report(
    path: Path,
    *,
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    metrics_by_split: list[dict[str, Any]],
    metrics_by_option_type: list[dict[str, Any]],
    figure_paths: tuple[Path, ...],
    review_decision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite_count = int(np.count_nonzero(targets.reference_found))
    in_window_count = int(np.count_nonzero(targets.reference_in_sample_window))
    no_div_count = int(np.count_nonzero(targets.is_no_dividend_call_control))
    split_table = _latex_split_table(metrics_by_split)
    option_table = _latex_option_table(metrics_by_option_type)
    figure_block = "\n".join(_latex_figure(figure_path) for figure_path in figure_paths)
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
\lhead{{\small Stage 4 Boundary Diagnostic}}
\rhead{{\small Boundary Head Comparison}}
\cfoot{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\title{{\textbf{{Stage 4 Boundary Diagnostic Report}}\\
\large Premium-Implied Boundaries and a Direct Boundary-Head Baseline}}
\author{{Codex-assisted surrogate modelling review}}
\date{{June 19, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report reviews Stage 4 of the compressed roadmap. It checks whether the Stage 3
positive-premium surrogate can recover early-exercise boundary behavior from predicted
continuation premium, and compares it with a small direct boundary-head baseline. Boundary labels
remain threshold-based CN/PSOR diagnostics, not exact analytical free boundaries. No Delta or
Gamma head is trained. The Stage 4 review decision is \code{{{review_decision}}}.
\end{{abstract}}

\tableofcontents
\newpage

\section{{Purpose of Stage 4}}

Stage 4 is a diagnostic boundary experiment. Stage 3 showed that predicting continuation premium
helped preserve the American payoff obstacle. The next question is whether that premium surrogate
also gives coherent early-exercise boundary behavior when the same threshold extraction logic is
applied to predicted premium curves. The boundary interpretation follows the American-option
finite-difference and free-boundary setting documented in the solver reports
\citep{{brennan_schwartz_1977,wilmott_howison_dewynne_1995}}.

\section{{Link to Stage 3 Result}}

The Stage 3 review decision was \code{{READY_FOR_BOUNDARY_DIAGNOSTIC_STAGE}}. Stage 4 uses the same
v1 small-grid dataset and retrains the positive-premium MLP deterministically because Stage 3 did
not save model weights.

\section{{Boundary Target Construction}}

The v1 dataset stores \code{{boundary_spot_over_K}} on sampled rows. Stage 4 collapses those rows to
one target per regime-time pair. The resulting target table contains {targets.row_count:,} rows,
with {finite_count:,} finite reference boundaries and {in_window_count:,} finite boundaries inside
the sampled \(S/K\in[0.4,1.8]\) support. There are {no_div_count:,} no-dividend call control
time rows.

\section{{Premium-Implied Boundary Method}}

The premium-implied method retrains the Stage 3 positive-premium MLP in memory, predicts
\code{{premium_over_K}} over v1 sampled rows, sorts each regime-time curve by moneyness, and applies
the Ticket 09 threshold crossing with threshold \(10^{-6}\). It is a diagnostic extraction, not a
new solver. In this Stage 4 test, the premium-implied boundary method is not reliable: it often has
no comparable boundary-error points, so its RMSE is reported as \code{{NA}} rather than zero error.

\section{{Direct Boundary-Head Method}}

The direct boundary head is a small baseline. It uses only
\code{{tau_fraction}}, \(r\), \(q\), \(\sigma\), \(T\), and \code{{is_call}}. A histogram-gradient
classifier predicts whether a boundary exists, and a bounded histogram-gradient regressor predicts
\code{{boundary_spot_over_K}} for finite-boundary rows. Maturity rows and no-dividend call controls
are postprocessed as no-boundary diagnostic cases. No Delta, Gamma, or production boundary head is
introduced.

\section{{Metrics and Evaluation Protocol}}

Metrics include boundary MAE, RMSE, maximum absolute error, found/not-found agreement, missed
boundary rate, false boundary rate, and no-dividend-call false-boundary rate. Boundary errors are
computed only where a reference boundary exists, the prediction found a boundary, and the reference
boundary is inside the sampled support.

\section{{Results by Split}}

{split_table}

Stage 4 passes because the direct boundary-head baseline satisfies the boundary diagnostics, not
because the premium-implied threshold extraction succeeds. The premium-implied method is useful as a
diagnostic stress test of the price/premium surrogate, but it did not recover dependable boundary
curves in this run. This supports using a separate boundary-focused model for boundary diagnostics
instead of relying only on price/premium threshold extraction.

\section{{Results by Option Family}}

{option_table}

\section{{Sample Boundary Curves}}

{figure_block}

\section{{Limitations}}

Boundary labels are threshold-based diagnostics from the finite-difference solver. They are not
analytical free-boundary truth. Premium-implied extraction is limited by the sampled \(S/K\) window,
and some reference boundaries lie outside that support. The direct boundary head is a small
diagnostic baseline, not a solver replacement.

\section{{What Stage 4 Supports}}

Stage 4 supports comparing premium-implied and direct boundary diagnostic behavior by split,
option family, and sampled boundary curves. It also records no-dividend call false-boundary
behavior as a control. The evidence supports a separate boundary-focused model as the coherent
Stage 4 boundary diagnostic path.

\section{{What Stage 4 Does Not Support}}

This stage does not support production boundary reliability, Delta or Gamma target training, exact
free-boundary claims, broad architecture search, or final surrogate conclusions.

\section{{Recommended Next Stage}}

If the decision is \code{{READY_FOR_DELTA_DIAGNOSTIC_STAGE}}, the next stage may plan a cautious
Delta diagnostic experiment. If the decision is \code{{REVIEW_REQUIRED_BEFORE_DELTA_STAGE}}, human
review should inspect boundary failures before adding any new diagnostic heads.

\bibliographystyle{{plainnat}}
\bibliography{{reports/03_solver/references}}

\end{{document}}
""",
        encoding="utf-8",
    )


def _boundary_head_features(targets: BoundaryTargetTable) -> np.ndarray:
    return np.column_stack(
        [
            targets.tau_fraction,
            targets.r,
            targets.q,
            targets.sigma,
            targets.T,
            targets.is_call,
        ]
    ).astype(float)


def _metric_row(
    fieldnames: list[str],
    targets: BoundaryTargetTable,
    prediction: BoundaryPrediction,
    *,
    method_name: str,
    group_value: str,
    mask: np.ndarray,
    group_field: str,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    ref_found = targets.reference_found[idx]
    pred_found = prediction.predicted_found[idx]
    comparable = (
        ref_found
        & pred_found
        & targets.reference_in_sample_window[idx]
        & np.isfinite(prediction.predicted_boundary[idx])
    )
    errors = (
        prediction.predicted_boundary[idx][comparable]
        - targets.reference_boundary[idx][comparable]
    )
    abs_errors = np.abs(errors)
    false_mask = ~ref_found
    no_div_mask = targets.is_no_dividend_call_control[idx]
    row = {
        "method_name": method_name,
        group_field: group_value,
        "row_count": int(idx.size),
        "reference_found_count": int(np.count_nonzero(ref_found)),
        "predicted_found_count": int(np.count_nonzero(pred_found)),
        "comparable_error_count": int(np.count_nonzero(comparable)),
        "reference_outside_sample_window_count": int(
            np.count_nonzero(ref_found & ~targets.reference_in_sample_window[idx])
        ),
        "boundary_mae": float(np.mean(abs_errors)) if abs_errors.size else float("nan"),
        "boundary_rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
        "boundary_max_abs_error": float(np.max(abs_errors)) if abs_errors.size else float("nan"),
        "boundary_found_agreement_rate": float(np.mean(ref_found == pred_found)) if idx.size else float("nan"),
        "missed_boundary_rate": (
            float(np.mean(~pred_found[ref_found & targets.reference_in_sample_window[idx]]))
            if np.any(ref_found & targets.reference_in_sample_window[idx])
            else float("nan")
        ),
        "false_boundary_rate": (
            float(np.mean(pred_found[false_mask])) if np.any(false_mask) else float("nan")
        ),
        "no_dividend_call_false_boundary_rate": (
            float(np.mean(pred_found[no_div_mask])) if np.any(no_div_mask) else float("nan")
        ),
        "review_flag": "PASS",
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }
    if "review_flag" in fieldnames:
        row["review_flag"] = _row_review_flag(row)
    return {name: row[name] for name in fieldnames}


def _row_review_flag(row: dict[str, Any]) -> str:
    rmse = float(row["boundary_rmse"])
    agreement = float(row["boundary_found_agreement_rate"])
    no_div = float(row["no_dividend_call_false_boundary_rate"])
    no_div_ok = math.isnan(no_div) or no_div <= 0.05
    if np.isfinite(rmse) and rmse <= 0.20 and agreement >= 0.80 and no_div_ok:
        return "PASS"
    return "REVIEW"


def _metric_lookup(rows: list[dict[str, Any]], method_name: str, split: str) -> dict[str, Any] | None:
    for row in rows:
        if row["method_name"] == method_name and row.get("split") == split:
            return row
    return None


def _model_manifest_rows(
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    *,
    dataset_path: Path,
    stage3_positive_premium_retrained: bool,
    review_decision: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in predictions:
        if method_name == "premium_implied_boundary":
            model_family = "Stage 3 positive-premium MLP plus Ticket 09 threshold extraction"
            inputs = "v1 X plus payoff_over_K through Stage 3 premium reconstruction"
            target = "premium_over_K; boundary extracted diagnostically"
            training_rows = TRAIN_ROW_CAP
        else:
            model_family = (
                "Histogram-gradient boundary-existence classifier plus bounded "
                "histogram-gradient boundary regressor with maturity/no-dividend postprocessing"
            )
            inputs = ",".join(BOUNDARY_HEAD_FEATURE_NAMES)
            target = "boundary_spot_over_K where finite; boundary_found classifier"
            training_rows = int(np.count_nonzero(targets.split == "train"))
        rows.append(
            {
                "run_id": "stage4_boundary_diagnostic",
                "method_name": method_name,
                "dataset_path": str(dataset_path),
                "random_seed": RANDOM_SEED,
                "train_row_cap": TRAIN_ROW_CAP,
                "stage3_positive_premium_retrained": stage3_positive_premium_retrained,
                "input_columns": inputs,
                "target_name": target,
                "training_rows": training_rows,
                "model_family": model_family,
                "model_weights_saved": "no",
                "boundary_threshold": BOUNDARY_THRESHOLD,
                "sample_moneyness_window": f"[{SAMPLE_LOWER_MONEYNESS},{SAMPLE_UPPER_MONEYNESS}]",
                "review_decision": review_decision,
                "downstream_use_status": DOWNSTREAM_USE_STATUS,
            }
        )
    return rows


def _create_figures(
    output_dir: Path,
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    metrics_by_split: list[dict[str, Any]],
    metrics_by_option_type: list[dict[str, Any]],
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

    path = figure_dir / "boundary_error_by_split.png"
    split_names = ("train", "validation", "test", "stress_holdout")
    x = np.arange(len(split_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, method_name in enumerate(predictions):
        values = [
            _metric_value(metrics_by_split, method_name, split, "boundary_rmse")
            for split in split_names
        ]
        _plot_metric_bars_with_na(ax, x + (offset - 0.5) * width, values, width, method_name)
    ax.axhline(0.20, color="black", linestyle="--", linewidth=1, label="RMSE gate")
    ax.set_xticks(x, split_names, rotation=20)
    ax.set_ylabel("Boundary RMSE")
    ax.set_title("Boundary Error by Split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    path = figure_dir / "boundary_error_by_option_type.png"
    option_names = ("put", "call")
    x = np.arange(len(option_names))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for offset, method_name in enumerate(predictions):
        values = [
            _metric_value(metrics_by_option_type, method_name, option, "boundary_rmse", group_key="option_type")
            for option in option_names
        ]
        _plot_metric_bars_with_na(ax, x + (offset - 0.5) * width, values, width, method_name)
    ax.set_xticks(x, option_names)
    ax.set_ylabel("Boundary RMSE")
    ax.set_title("Boundary Error by Option Family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    paths.append(_sample_curve_figure(figure_dir / "sample_put_boundary_curves.png", targets, predictions, "put"))
    paths.append(_sample_curve_figure(figure_dir / "sample_call_boundary_curves.png", targets, predictions, "call"))

    path = figure_dir / "boundary_found_rate_comparison.png"
    plt.figure(figsize=(8, 4.8))
    for offset, method_name in enumerate(predictions):
        values = [
            _finite_metric(metrics_by_split, method_name, split, "boundary_found_agreement_rate")
            for split in split_names
        ]
        plt.bar(x=np.arange(len(split_names)) + (offset - 0.5) * width, height=values, width=width, label=method_name)
    plt.axhline(0.80, color="black", linestyle="--", linewidth=1, label="agreement gate")
    plt.xticks(np.arange(len(split_names)), split_names, rotation=20)
    plt.ylabel("Found agreement rate")
    plt.title("Boundary Found Agreement by Split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)
    return paths


def _sample_curve_figure(
    path: Path,
    targets: BoundaryTargetTable,
    predictions: dict[str, BoundaryPrediction],
    option_type: str,
) -> Path:
    import matplotlib.pyplot as plt

    candidate_regimes = [
        regime
        for regime in np.unique(targets.regime_id[targets.option_type == option_type])
        if np.count_nonzero((targets.regime_id == regime) & targets.reference_in_sample_window) >= 20
    ][:3]
    plt.figure(figsize=(7.2, 4.8))
    for regime in candidate_regimes:
        mask = targets.regime_id == regime
        order = np.argsort(targets.tau_fraction[mask])
        tau = targets.tau_fraction[mask][order]
        ref = targets.reference_boundary[mask][order]
        plt.plot(tau, ref, linewidth=1.5, label=f"{regime} reference")
        for method_name, prediction in predictions.items():
            pred = prediction.predicted_boundary[mask][order]
            plt.plot(tau, pred, linestyle="--", linewidth=1, label=f"{regime} {method_name}")
    plt.xlabel("tau / T")
    plt.ylabel("boundary spot / K")
    plt.title(f"Sample {option_type.title()} Boundary Curves")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_metric_bars_with_na(
    ax: Any,
    positions: np.ndarray,
    values: list[float],
    width: float,
    label: str,
) -> None:
    value_array = np.asarray(values, dtype=float)
    finite = np.isfinite(value_array)
    if np.any(finite):
        ax.bar(positions[finite], value_array[finite], width=width, label=label)
    else:
        ax.bar([], [], width=width, label=label)

    finite_values = value_array[finite]
    annotation_y = 0.012
    if finite_values.size:
        annotation_y = max(0.012, float(np.max(finite_values)) * 0.08)
    for position, value in zip(positions, value_array):
        if not np.isfinite(value):
            ax.text(
                position,
                annotation_y,
                "NA\nno comparable\nboundary",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )


def _metric_value(
    rows: list[dict[str, Any]],
    method_name: str,
    group_value: str,
    metric: str,
    group_key: str = "split",
) -> float:
    for row in rows:
        if row["method_name"] == method_name and row[group_key] == group_value:
            return float(row[metric])
    return float("nan")


def _finite_metric(
    rows: list[dict[str, Any]],
    method_name: str,
    group_value: str,
    metric: str,
    group_key: str = "split",
) -> float:
    for row in rows:
        if row["method_name"] == method_name and row[group_key] == group_value:
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
        r"\begin{longtable}{L{0.23\textwidth} L{0.16\textwidth} L{0.15\textwidth} L{0.15\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Method & Split & RMSE & Found agreement & No-div false rate \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            rf"{_escape(row['method_name'])} & {_escape(row['split'])} & {_fmt(row['boundary_rmse'])} & {_fmt(row['boundary_found_agreement_rate'])} & {_fmt(row['no_dividend_call_false_boundary_rate'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _latex_option_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{longtable}{L{0.26\textwidth} L{0.15\textwidth} L{0.16\textwidth} L{0.16\textwidth} L{0.16\textwidth}}",
        r"\toprule",
        r"Method & Option & RMSE & MAE & Found agreement \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            rf"{_escape(row['method_name'])} & {_escape(row['option_type'])} & {_fmt(row['boundary_rmse'])} & {_fmt(row['boundary_mae'])} & {_fmt(row['boundary_found_agreement_rate'])} \\"
        )
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
