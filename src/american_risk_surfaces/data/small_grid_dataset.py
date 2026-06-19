"""v1 Small-Grid Dataset: 288-regime surrogate dataset generation and QA."""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    BoundaryExtractionSummary,
    continuation_premium,
    extract_boundary_curve,
    summarize_boundary_curve,
)
from american_risk_surfaces.diagnostics.greeks import (
    GreekDiagnostics,
    diagnose_greek_result,
)
from american_risk_surfaces.diagnostics.lcp import LCPDiagnostics, diagnose_lcp_result
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SURROGATE_PLAN_DIR = PROJECT_ROOT / "reports" / "05_surrogate"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "04_surrogate_dataset" / "v1_small_grid"
)

SOLVER_NAME = "american_crank_nicolson_psor_price"
SOLVER_VARIANT = "baseline_cn_psor"
DOWNSTREAM_USE_STATUS = "v1_small_grid_only"
PREMIUM_THRESHOLD = 1e-6
INTERPRETATION_MONEYNESS_BOUNDS = (0.4, 1.8)
OBSTACLE_TOLERANCE = 1e-8
EQUATION_TOLERANCE = 1e-6
COMPLEMENTARITY_TOLERANCE = 1e-6

EXPECTED_SPLIT_COUNTS = {
    "train": 202,
    "validation": 19,
    "test": 43,
    "stress_holdout": 24,
}
SPLIT_ORDER = ("train", "validation", "test", "stress_holdout")
EXPECTED_PARAMETER_VALUES = {
    "option_type": ("put", "call"),
    "T": (0.25, 0.5, 1.0, 2.0),
    "sigma": (0.20, 0.40, 0.60),
    "r": (0.01, 0.05, 0.10),
    "q": (0.00, 0.03, 0.06, 0.10),
}
HIGHER_GRID_CONFIRMATION_IDS = (
    "put_T100_s020_r005_q003",
    "put_T100_s060_r005_q003",
    "call_T100_s020_r005_q006",
    "call_T100_s020_r005_q010",
)

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
EXPECTED_NPZ_KEYS = (
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

REGIME_MANIFEST_FIELDNAMES = [
    "regime_id",
    "option_type",
    "T",
    "sigma",
    "r",
    "q",
    "K",
    "Smax",
    "M",
    "N",
    "dS",
    "dtau",
    "split",
    "split_reason",
    "stress_holdout_flag",
    "planned_use",
    "solver_name",
    "solver_variant",
    "runtime_seconds",
    "full_grid_rows",
    "total_sample_rows",
    "accepted_sample_rows",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "boundary_found_count",
    "boundary_status",
    "finite_delta_count",
    "finite_gamma_count",
    "strict_interior_node_count",
    "acceptance_status",
    "acceptance_reason",
    "downstream_use_status",
]

DIAGNOSTIC_SUMMARY_FIELDNAMES = [
    "regime_id",
    "option_type",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "lcp_status",
    "boundary_found_count",
    "boundary_threshold",
    "boundary_status",
    "finite_delta_count",
    "finite_gamma_count",
    "nonfinite_delta_count",
    "nonfinite_gamma_count",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "strict_negative_gamma_count",
    "greek_status",
    "runtime_seconds",
    "acceptance_status",
    "acceptance_reason",
    "downstream_use_status",
]

SPLIT_ASSIGNMENT_FIELDNAMES = [
    "regime_id",
    "split",
    "split_reason",
    "option_family_balance",
    "stress_holdout_flag",
    "total_sample_rows",
    "accepted_sample_rows",
    "acceptance_status",
    "downstream_use_status",
]

SCHEMA_SNAPSHOT_FIELDNAMES = [
    "schema_group",
    "field_name",
    "field_order",
    "definition",
    "dtype",
    "role",
    "downstream_use_status",
]

OUTPUT_MANIFEST_FIELDNAMES = [
    "output_id",
    "output_type",
    "path",
    "created",
    "description",
    "row_count",
    "solver_name",
    "solver_variant",
    "downstream_use_status",
    "review_status",
]

QUALITY_SUMMARY_FIELDNAMES = [
    "metric",
    "metric_value",
    "status",
    "review_decision",
    "notes",
]

HIGHER_GRID_CONFIRMATION_FIELDNAMES = [
    "regime_id",
    "option_type",
    "T",
    "sigma",
    "r",
    "q",
    "baseline_M",
    "baseline_N",
    "confirmation_M",
    "confirmation_N",
    "baseline_runtime_seconds",
    "confirmation_runtime_seconds",
    "max_abs_value_difference",
    "mean_abs_value_difference",
    "max_abs_premium_difference",
    "baseline_boundary_found_count",
    "confirmation_boundary_found_count",
    "confirmation_acceptance_status",
    "notes",
    "downstream_use_status",
]

__all__ = (
    "SmallGridRegime",
    "SmallGridArtifacts",
    "SmallGridArrayChunk",
    "SmallGridDatasetPackage",
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "MASK_NAMES",
    "AUDIT_NUMERIC_NAMES",
    "EXPECTED_NPZ_KEYS",
    "REGIME_MANIFEST_FIELDNAMES",
    "DIAGNOSTIC_SUMMARY_FIELDNAMES",
    "SPLIT_ASSIGNMENT_FIELDNAMES",
    "SCHEMA_SNAPSHOT_FIELDNAMES",
    "OUTPUT_MANIFEST_FIELDNAMES",
    "QUALITY_SUMMARY_FIELDNAMES",
    "HIGHER_GRID_CONFIRMATION_FIELDNAMES",
    "load_small_grid_plan",
    "validate_small_grid_plan",
    "split_counts",
    "run_small_grid_regime",
    "continuation_premium_grid",
    "exercise_indicator",
    "evaluate_acceptance",
    "sample_regime_chunk",
    "build_dataset_arrays",
    "write_npz_package",
    "generate_v1_small_grid_dataset",
    "write_csv",
    "create_split_row_counts_figure",
    "create_diagnostic_thresholds_figure",
    "create_premium_distribution_figure",
    "create_boundary_availability_figure",
    "create_gamma_mask_summary_figure",
)


@dataclass(frozen=True)
class SmallGridRegime:
    """One planned v1 small-grid regime."""

    regime_id: str
    option_type: str
    T: float
    sigma: float
    r: float
    q: float
    K: float
    Smax: float
    M: int
    N: int
    split: str
    split_reason: str
    option_family_balance: str
    stress_holdout_flag: str
    solver_variant: str
    planned_use: str
    notes: str = ""


@dataclass(frozen=True)
class SmallGridArtifacts:
    """Solver result and diagnostics for one v1 regime."""

    regime: SmallGridRegime
    result: AmericanCNPSORResult
    premium_grid: np.ndarray
    exercise_indicator_grid: np.ndarray
    boundary_curve: BoundaryCurve
    boundary_summary: BoundaryExtractionSummary
    lcp_diagnostics: LCPDiagnostics
    greek_diagnostics: GreekDiagnostics
    runtime_seconds: float
    dS: float
    dtau: float
    premium_threshold: float
    solver_name: str = SOLVER_NAME
    solver_variant: str = SOLVER_VARIANT


@dataclass(frozen=True)
class SmallGridArrayChunk:
    """Vectorized accepted-row arrays for one accepted regime."""

    regime_id: str
    row_count: int
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
    audit_numeric: np.ndarray


@dataclass(frozen=True)
class SmallGridDatasetPackage:
    """Paths, rows, and decision produced by the v1 generator."""

    output_dir: Path
    npz_path: Path
    regime_manifest_path: Path
    diagnostic_summary_path: Path
    split_assignment_path: Path
    schema_snapshot_path: Path
    output_manifest_path: Path
    generation_log_path: Path
    quality_summary_path: Path
    higher_grid_confirmation_path: Path
    figure_paths: tuple[Path, ...]
    regime_rows: list[dict[str, Any]]
    diagnostic_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    schema_rows: list[dict[str, Any]]
    output_rows: list[dict[str, Any]]
    quality_rows: list[dict[str, Any]]
    higher_grid_rows: list[dict[str, Any]]
    review_decision: str
    accepted_row_count: int
    total_regime_count: int


def load_small_grid_plan(plan_dir: Path = SURROGATE_PLAN_DIR) -> tuple[SmallGridRegime, ...]:
    """Load and validate the approved 288-regime v1 plan."""

    plan_path = Path(plan_dir)
    regime_rows = _read_csv(plan_path / "dataset_regime_plan.csv")
    split_rows = _read_csv(plan_path / "dataset_split_assignment_plan.csv")
    splits_by_id = {row["regime_id"]: row for row in split_rows}
    regimes: list[SmallGridRegime] = []
    for row in regime_rows:
        regime_id = row["regime_id"]
        if regime_id not in splits_by_id:
            raise ValueError(f"regime {regime_id} is missing from split plan.")
        split = splits_by_id[regime_id]
        regimes.append(
            SmallGridRegime(
                regime_id=regime_id,
                option_type=row["option_type"],
                T=float(row["T"]),
                sigma=float(row["sigma"]),
                r=float(row["r"]),
                q=float(row["q"]),
                K=float(row["K"]),
                Smax=float(row["Smax"]),
                M=int(row["M"]),
                N=int(row["N"]),
                split=split["split"],
                split_reason=split["split_reason"],
                option_family_balance=split["option_family_balance"],
                stress_holdout_flag=split["stress_holdout_flag"],
                solver_variant=row["solver_variant"],
                planned_use=row["planned_use"],
                notes=row.get("notes", ""),
            )
        )
    validate_small_grid_plan(tuple(regimes))
    return tuple(regimes)


def validate_small_grid_plan(regimes: tuple[SmallGridRegime, ...]) -> None:
    """Validate that the full v1 plan exactly matches the approved construction grid."""

    if len(regimes) != 288:
        raise ValueError("v1 small-grid generation requires exactly 288 regimes.")
    regime_ids = [regime.regime_id for regime in regimes]
    if len(set(regime_ids)) != len(regime_ids):
        raise ValueError("v1 regime IDs must be unique.")
    if split_counts(regimes) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("v1 split counts do not match the approved construction plan.")

    expected_tuples = {
        (option_type, T, sigma, r, q)
        for option_type in EXPECTED_PARAMETER_VALUES["option_type"]
        for T in EXPECTED_PARAMETER_VALUES["T"]
        for sigma in EXPECTED_PARAMETER_VALUES["sigma"]
        for r in EXPECTED_PARAMETER_VALUES["r"]
        for q in EXPECTED_PARAMETER_VALUES["q"]
    }
    actual_tuples = {
        (
            regime.option_type,
            round(regime.T, 10),
            round(regime.sigma, 10),
            round(regime.r, 10),
            round(regime.q, 10),
        )
        for regime in regimes
    }
    if actual_tuples != expected_tuples:
        raise ValueError("v1 regimes do not match the approved parameter grid.")

    for regime in regimes:
        _validate_regime(regime)
        if regime.K != 1.0 or regime.Smax != 4.0:
            raise ValueError("v1 regimes must use K=1 and Smax=4.")
        if (regime.M, regime.N) != (120, 120):
            raise ValueError("approved v1 regimes must use M=N=120.")
        if regime.planned_use != regime.split:
            raise ValueError("regime planned_use must match the split assignment.")


def split_counts(regimes: tuple[SmallGridRegime, ...]) -> dict[str, int]:
    """Return deterministic split counts for v1 regimes."""

    counts = Counter(regime.split for regime in regimes)
    return {split: int(counts.get(split, 0)) for split in SPLIT_ORDER}


def run_small_grid_regime(
    regime: SmallGridRegime,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> SmallGridArtifacts:
    """Run one v1 regime with the baseline American CN/PSOR solver."""

    validated = _validate_regime(regime)
    threshold = _validate_threshold(premium_threshold)
    start = perf_counter()
    result = american_crank_nicolson_psor_price(
        option_type=validated.option_type,
        K=validated.K,
        T=validated.T,
        r=validated.r,
        q=validated.q,
        sigma=validated.sigma,
        Smax=validated.Smax,
        M=validated.M,
        N=validated.N,
    )
    runtime_seconds = perf_counter() - start
    premium_grid = continuation_premium_grid(result)
    indicator = exercise_indicator(premium_grid, threshold=threshold)
    boundary_curve = extract_boundary_curve(result, validated.regime_id, threshold=threshold)
    boundary_summary = summarize_boundary_curve(boundary_curve)
    lcp_diagnostics = diagnose_lcp_result(result, validated.regime_id)
    greek_diagnostics = diagnose_greek_result(
        result,
        validated.regime_id,
        boundary_curve=boundary_curve,
        boundary_threshold=threshold,
    )
    return SmallGridArtifacts(
        regime=validated,
        result=result,
        premium_grid=premium_grid,
        exercise_indicator_grid=indicator,
        boundary_curve=boundary_curve,
        boundary_summary=boundary_summary,
        lcp_diagnostics=lcp_diagnostics,
        greek_diagnostics=greek_diagnostics,
        runtime_seconds=float(runtime_seconds),
        dS=_spacing(result.spot_grid),
        dtau=_spacing(result.tau_grid),
        premium_threshold=threshold,
    )


def continuation_premium_grid(result: AmericanCNPSORResult) -> np.ndarray:
    """Return the v1 continuation premium grid, U minus payoff."""

    return continuation_premium(result.value_grid, result.payoff)


def exercise_indicator(premium_grid: Any, threshold: float = PREMIUM_THRESHOLD) -> np.ndarray:
    """Return 1 for exercise-like nodes and 0 for continuation-like nodes."""

    premium = np.asarray(premium_grid, dtype=float)
    if premium.ndim not in (1, 2):
        raise ValueError("premium_grid must be one- or two-dimensional.")
    if np.any(~np.isfinite(premium)):
        raise ValueError("premium_grid must contain finite values.")
    return (premium <= _validate_threshold(threshold)).astype(int)


def evaluate_acceptance(
    all_psor_steps_converged: bool,
    max_obstacle_violation: float,
    max_equation_violation: float,
    max_abs_complementarity_product: float,
    metadata_complete: bool,
) -> tuple[str, str]:
    """Return v1 acceptance status and reason text."""

    reasons: list[str] = []
    if not all_psor_steps_converged:
        reasons.append("psor_not_converged")
    if float(max_obstacle_violation) > OBSTACLE_TOLERANCE:
        reasons.append("obstacle_violation_above_tolerance")
    if float(max_equation_violation) > EQUATION_TOLERANCE:
        reasons.append("equation_violation_above_tolerance")
    if float(max_abs_complementarity_product) > COMPLEMENTARITY_TOLERANCE:
        reasons.append("complementarity_above_tolerance")
    if not metadata_complete:
        reasons.append("metadata_or_masks_missing")
    if reasons:
        return "review_required", ";".join(reasons)
    return "accepted", "diagnostics_passed"


def sample_regime_chunk(
    artifacts: SmallGridArtifacts,
    regime_index: int,
) -> SmallGridArrayChunk:
    """Return vectorized accepted-row arrays in the approved reporting region."""

    if not isinstance(artifacts, SmallGridArtifacts):
        raise ValueError("artifacts must be a SmallGridArtifacts instance.")
    result = artifacts.result
    regime = artifacts.regime
    lower, upper = INTERPRETATION_MONEYNESS_BOUNDS
    moneyness_grid = result.spot_grid / regime.K
    spot_indices = np.where(
        (moneyness_grid >= lower - 1e-14) & (moneyness_grid <= upper + 1e-14)
    )[0]
    if len(spot_indices) == 0:
        raise ValueError("approved reporting region contains no spot nodes.")

    time_count = len(result.tau_grid)
    time_indices = np.repeat(np.arange(time_count), len(spot_indices))
    spot_index_rows = np.tile(spot_indices, time_count)

    spots = result.spot_grid[spot_index_rows]
    moneyness = spots / regime.K
    tau = result.tau_grid[time_indices]
    tau_fraction = np.divide(tau, regime.T, out=np.zeros_like(tau), where=regime.T != 0.0)

    values = result.value_grid[time_indices, spot_index_rows] / regime.K
    payoff = result.payoff[spot_index_rows] / regime.K
    premium = artifacts.premium_grid[time_indices, spot_index_rows] / regime.K
    exercise = artifacts.exercise_indicator_grid[time_indices, spot_index_rows].astype(float)
    boundary_by_time = np.array(
        [
            point.boundary_spot / regime.K
            if point.boundary_found and np.isfinite(point.boundary_spot)
            else np.nan
            for point in artifacts.boundary_curve.points
        ],
        dtype=float,
    )
    boundary = boundary_by_time[time_indices]
    delta = artifacts.greek_diagnostics.arrays.delta[time_indices, spot_index_rows]
    gamma = artifacts.greek_diagnostics.arrays.gamma[time_indices, spot_index_rows]
    finite_delta = artifacts.greek_diagnostics.arrays.finite_delta_mask[
        time_indices, spot_index_rows
    ]
    finite_gamma = artifacts.greek_diagnostics.arrays.finite_gamma_mask[
        time_indices, spot_index_rows
    ]
    kink = artifacts.greek_diagnostics.masks.payoff_kink_near[time_indices, spot_index_rows]
    boundary_near = artifacts.greek_diagnostics.masks.boundary_near[
        time_indices, spot_index_rows
    ]
    maturity = artifacts.greek_diagnostics.masks.maturity_row[time_indices, spot_index_rows]
    strict = artifacts.greek_diagnostics.masks.strict_interior[time_indices, spot_index_rows]

    X = np.column_stack(
        [
            np.log(moneyness),
            tau_fraction,
            np.full_like(tau, regime.r, dtype=float),
            np.full_like(tau, regime.q, dtype=float),
            np.full_like(tau, regime.sigma, dtype=float),
            np.full_like(tau, regime.T, dtype=float),
            np.full_like(tau, 1.0 if regime.option_type == "call" else 0.0, dtype=float),
        ]
    ).astype(float)
    masks = np.column_stack(
        [
            kink,
            boundary_near,
            maturity,
            strict,
            strict & finite_gamma,
            strict & finite_delta,
            exercise == 1.0,
            exercise == 0.0,
        ]
    ).astype(bool)
    audit_numeric = np.column_stack(
        [
            moneyness,
            tau,
            spots,
            np.full_like(tau, regime.K, dtype=float),
            np.full_like(tau, regime.Smax, dtype=float),
            np.full_like(tau, float(regime.M), dtype=float),
            np.full_like(tau, float(regime.N), dtype=float),
            np.full_like(tau, artifacts.dS, dtype=float),
            np.full_like(tau, artifacts.dtau, dtype=float),
            np.full_like(tau, float(regime_index), dtype=float),
            np.full_like(tau, float(_split_index(regime.split)), dtype=float),
        ]
    ).astype(float)
    row_count = int(len(time_indices))
    return SmallGridArrayChunk(
        regime_id=regime.regime_id,
        row_count=row_count,
        X=X,
        y_value=values.astype(float),
        y_payoff=payoff.astype(float),
        y_premium=premium.astype(float),
        y_exercise_indicator=exercise.astype(float),
        y_boundary=boundary.astype(float),
        y_delta=delta.astype(float),
        y_scaled_gamma=(regime.K * gamma).astype(float),
        masks=masks,
        regime_index=np.full(row_count, int(regime_index), dtype=int),
        audit_numeric=audit_numeric,
    )


def build_dataset_arrays(
    chunks: list[SmallGridArrayChunk],
    regimes: tuple[SmallGridRegime, ...],
) -> dict[str, np.ndarray]:
    """Build the accepted-row v1 arrays for compressed storage."""

    arrays = {
        "X": _concat_2d([chunk.X for chunk in chunks], len(FEATURE_NAMES), float),
        "y_value": _concat_1d([chunk.y_value for chunk in chunks], float),
        "y_payoff": _concat_1d([chunk.y_payoff for chunk in chunks], float),
        "y_premium": _concat_1d([chunk.y_premium for chunk in chunks], float),
        "y_exercise_indicator": _concat_1d(
            [chunk.y_exercise_indicator for chunk in chunks], float
        ),
        "y_boundary": _concat_1d([chunk.y_boundary for chunk in chunks], float),
        "y_delta": _concat_1d([chunk.y_delta for chunk in chunks], float),
        "y_scaled_gamma": _concat_1d([chunk.y_scaled_gamma for chunk in chunks], float),
        "masks": _concat_2d([chunk.masks for chunk in chunks], len(MASK_NAMES), bool),
        "regime_index": _concat_1d([chunk.regime_index for chunk in chunks], int),
        "feature_names": np.array(FEATURE_NAMES, dtype=str),
        "label_names": np.array(LABEL_NAMES, dtype=str),
        "mask_names": np.array(MASK_NAMES, dtype=str),
        "audit_numeric": _concat_2d(
            [chunk.audit_numeric for chunk in chunks], len(AUDIT_NUMERIC_NAMES), float
        ),
        "audit_numeric_names": np.array(AUDIT_NUMERIC_NAMES, dtype=str),
        "regime_ids": np.array([regime.regime_id for regime in regimes], dtype=str),
        "split_names": np.array(SPLIT_ORDER, dtype=str),
    }
    return arrays


def write_npz_package(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write exactly one compressed v1 small-grid dataset package."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: arrays[key] for key in EXPECTED_NPZ_KEYS}
    np.savez_compressed(destination, **ordered)


def generate_v1_small_grid_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    regimes: tuple[SmallGridRegime, ...] | None = None,
    create_figures: bool = True,
    include_higher_grid_confirmation: bool = True,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> SmallGridDatasetPackage:
    """Generate the approved v1 small-grid package and QA manifests."""

    selected_regimes = regimes if regimes is not None else load_small_grid_plan()
    if regimes is None:
        validate_small_grid_plan(selected_regimes)
    else:
        for regime in selected_regimes:
            _validate_regime(regime)

    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    generation_log: list[str] = [
        "v1 Small-Grid Dataset generation log",
        f"regime_count={len(selected_regimes)}",
        f"downstream_use_status={DOWNSTREAM_USE_STATUS}",
    ]
    artifacts_by_id: dict[str, SmallGridArtifacts] = {}
    chunks: list[SmallGridArrayChunk] = []
    regime_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for index, regime in enumerate(selected_regimes):
        generation_log.append(f"start regime_index={index} regime_id={regime.regime_id}")
        try:
            artifact = run_small_grid_regime(regime, premium_threshold=premium_threshold)
            artifacts_by_id[regime.regime_id] = artifact
            status, reason = _artifact_acceptance(artifact)
            chunk = sample_regime_chunk(artifact, regime_index=index)
            total_sample_rows = chunk.row_count
            accepted_sample_rows = chunk.row_count if status == "accepted" else 0
            if status == "accepted":
                chunks.append(chunk)
            regime_rows.append(
                _regime_manifest_row(
                    artifact,
                    total_sample_rows,
                    accepted_sample_rows,
                    status,
                    reason,
                )
            )
            diagnostic_rows.append(_diagnostic_summary_row(artifact, status, reason))
            split_rows.append(
                _split_assignment_row(artifact, total_sample_rows, accepted_sample_rows, status)
            )
            generation_log.append(
                f"finish regime_id={regime.regime_id} status={status} rows={accepted_sample_rows}"
            )
        except Exception as exc:  # pragma: no cover - defensive audit path
            reason = f"solver_or_diagnostic_exception:{type(exc).__name__}:{exc}"
            regime_rows.append(_failed_regime_manifest_row(regime, reason))
            diagnostic_rows.append(_failed_diagnostic_row(regime, reason))
            split_rows.append(_failed_split_row(regime, reason))
            generation_log.append(f"fail regime_id={regime.regime_id} reason={reason}")

    arrays = build_dataset_arrays(chunks, selected_regimes)
    npz_path = output_path / "dataset_v1_small_grid.npz"
    write_npz_package(npz_path, arrays)

    regime_manifest_path = output_path / "regime_manifest.csv"
    diagnostic_summary_path = output_path / "diagnostic_summary.csv"
    split_assignment_path = output_path / "split_assignment.csv"
    schema_snapshot_path = output_path / "schema_snapshot.csv"
    output_manifest_path = output_path / "output_manifest.csv"
    generation_log_path = output_path / "generation_log.txt"
    quality_summary_path = output_path / "v1_dataset_quality_summary.csv"
    higher_grid_confirmation_path = output_path / "v1_higher_grid_confirmation.csv"

    review_decision = (
        "READY_FOR_PRICE_SURROGATE_PLANNING"
        if len(regime_rows) == 288
        and all(row["acceptance_status"] == "accepted" for row in regime_rows)
        and npz_path.exists()
        else "REVIEW_REQUIRED_BEFORE_SURROGATE"
    )
    schema_rows = _schema_snapshot_rows()
    quality_rows = _quality_summary_rows(
        regime_rows,
        diagnostic_rows,
        arrays,
        npz_path,
        review_decision,
    )
    if include_higher_grid_confirmation:
        higher_grid_rows = _higher_grid_confirmation_rows(
            selected_regimes,
            artifacts_by_id,
            premium_threshold=premium_threshold,
        )
    else:
        higher_grid_rows = [
            _higher_grid_deferred_row(
                "not_requested",
                "Higher-grid confirmation disabled for this run.",
            )
        ]

    write_csv(regime_manifest_path, regime_rows, REGIME_MANIFEST_FIELDNAMES)
    write_csv(diagnostic_summary_path, diagnostic_rows, DIAGNOSTIC_SUMMARY_FIELDNAMES)
    write_csv(split_assignment_path, split_rows, SPLIT_ASSIGNMENT_FIELDNAMES)
    write_csv(schema_snapshot_path, schema_rows, SCHEMA_SNAPSHOT_FIELDNAMES)
    write_csv(quality_summary_path, quality_rows, QUALITY_SUMMARY_FIELDNAMES)
    write_csv(
        higher_grid_confirmation_path,
        higher_grid_rows,
        HIGHER_GRID_CONFIRMATION_FIELDNAMES,
    )

    figure_paths: list[Path] = []
    figure_status: dict[str, bool] = {}
    if create_figures:
        figure_specs = [
            (
                "split_row_counts",
                figure_dir / "v1_split_row_counts.png",
                create_split_row_counts_figure,
                (split_rows,),
            ),
            (
                "diagnostic_thresholds",
                figure_dir / "v1_diagnostic_thresholds.png",
                create_diagnostic_thresholds_figure,
                (diagnostic_rows,),
            ),
            (
                "premium_distribution",
                figure_dir / "v1_premium_distribution.png",
                create_premium_distribution_figure,
                (arrays,),
            ),
            (
                "boundary_availability",
                figure_dir / "v1_boundary_availability.png",
                create_boundary_availability_figure,
                (diagnostic_rows,),
            ),
            (
                "gamma_mask_summary",
                figure_dir / "v1_gamma_mask_summary.png",
                create_gamma_mask_summary_figure,
                (diagnostic_rows,),
            ),
        ]
        for name, path, helper, args in figure_specs:
            figure_status[name] = helper(*args, path)
            figure_paths.append(path)

    output_rows = _output_manifest_rows(
        npz_path=npz_path,
        regime_manifest_path=regime_manifest_path,
        diagnostic_summary_path=diagnostic_summary_path,
        split_assignment_path=split_assignment_path,
        schema_snapshot_path=schema_snapshot_path,
        output_manifest_path=output_manifest_path,
        generation_log_path=generation_log_path,
        quality_summary_path=quality_summary_path,
        higher_grid_confirmation_path=higher_grid_confirmation_path,
        figure_paths=figure_paths,
        figure_status=figure_status,
        regime_count=len(regime_rows),
        accepted_row_count=int(arrays["X"].shape[0]),
        quality_row_count=len(quality_rows),
        higher_grid_row_count=len(higher_grid_rows),
        review_decision=review_decision,
    )
    write_csv(output_manifest_path, output_rows, OUTPUT_MANIFEST_FIELDNAMES)
    generation_log.append(f"review_decision={review_decision}")
    generation_log.append(f"accepted_row_count={int(arrays['X'].shape[0])}")
    generation_log_path.write_text("\n".join(generation_log) + "\n")

    return SmallGridDatasetPackage(
        output_dir=output_path,
        npz_path=npz_path,
        regime_manifest_path=regime_manifest_path,
        diagnostic_summary_path=diagnostic_summary_path,
        split_assignment_path=split_assignment_path,
        schema_snapshot_path=schema_snapshot_path,
        output_manifest_path=output_manifest_path,
        generation_log_path=generation_log_path,
        quality_summary_path=quality_summary_path,
        higher_grid_confirmation_path=higher_grid_confirmation_path,
        figure_paths=tuple(figure_paths),
        regime_rows=regime_rows,
        diagnostic_rows=diagnostic_rows,
        split_rows=split_rows,
        schema_rows=schema_rows,
        output_rows=output_rows,
        quality_rows=quality_rows,
        higher_grid_rows=higher_grid_rows,
        review_decision=review_decision,
        accepted_row_count=int(arrays["X"].shape[0]),
        total_regime_count=len(regime_rows),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows with a stable schema using safe CSV escaping."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_split_row_counts_figure(split_rows: list[dict[str, Any]], path: Path) -> bool:
    """Create a split row-count figure for the v1 QA review."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not split_rows:
        return False
    counts = {
        split: sum(
            int(row["accepted_sample_rows"]) for row in split_rows if row["split"] == split
        )
        for split in SPLIT_ORDER
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(counts), list(counts.values()), color="#4f7cac")
    ax.set_title("v1 accepted row counts by regime-level split")
    ax.set_ylabel("accepted sampled rows")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_diagnostic_thresholds_figure(
    diagnostic_rows: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Create a compact diagnostic-threshold figure for v1 QA."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not diagnostic_rows:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(diagnostic_rows))
    obstacle = np.array([float(row["max_obstacle_violation"]) for row in diagnostic_rows])
    equation = np.array([float(row["max_equation_violation"]) for row in diagnostic_rows])
    comp = np.array([float(row["max_abs_complementarity_product"]) for row in diagnostic_rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, obstacle, label="obstacle", linewidth=1.0)
    ax.plot(x, equation, label="equation", linewidth=1.0)
    ax.plot(x, comp, label="complementarity", linewidth=1.0)
    ax.axhline(OBSTACLE_TOLERANCE, color="#555555", linestyle=":", linewidth=1.0)
    ax.axhline(EQUATION_TOLERANCE, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_title("v1 LCP diagnostic thresholds by regime")
    ax.set_xlabel("regime index")
    ax.set_ylabel("diagnostic magnitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_premium_distribution_figure(arrays: dict[str, np.ndarray], path: Path) -> bool:
    """Create a continuation-premium distribution figure for accepted rows."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    premiums = np.asarray(arrays.get("y_premium", []), dtype=float)
    if premiums.size == 0:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(premiums, bins=80, color="#4f7cac", alpha=0.85)
    ax.set_title("v1 continuation-premium distribution")
    ax.set_xlabel("premium_over_K")
    ax.set_ylabel("sample count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_boundary_availability_figure(
    diagnostic_rows: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Create a boundary-availability figure for v1 QA."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not diagnostic_rows:
        return False
    grouped: dict[str, list[float]] = {"put": [], "call": []}
    for row in diagnostic_rows:
        grouped.setdefault(row["option_type"], []).append(float(row["boundary_found_count"]))
    labels = list(grouped)
    means = [float(np.mean(grouped[label])) if grouped[label] else 0.0 for label in labels]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, color=["#4f7cac", "#b55d4c"][: len(labels)])
    ax.set_title("v1 mean boundary-found rows by option family")
    ax.set_ylabel("mean boundary-found time rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_gamma_mask_summary_figure(
    diagnostic_rows: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Create a Gamma mask and strict-region summary figure for v1 QA."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not diagnostic_rows:
        return False
    labels = [row["option_type"] for row in diagnostic_rows]
    unique = list(dict.fromkeys(labels))
    strict_means = []
    boundary_near_means = []
    for option_type in unique:
        rows = [row for row in diagnostic_rows if row["option_type"] == option_type]
        strict_means.append(float(np.mean([float(row["strict_interior_node_count"]) for row in rows])))
        boundary_near_means.append(
            float(np.mean([float(row["boundary_near_node_count"]) for row in rows]))
        )
    x = np.arange(len(unique))
    width = 0.35
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, strict_means, width, label="strict interior")
    ax.bar(x + width / 2, boundary_near_means, width, label="boundary near")
    ax.set_xticks(x, unique)
    ax.set_title("v1 Greek mask node counts by option family")
    ax.set_ylabel("mean full-grid node count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def _artifact_acceptance(artifact: SmallGridArtifacts) -> tuple[str, str]:
    summary = artifact.lcp_diagnostics.summary
    metadata_complete = (
        artifact.premium_grid.shape == artifact.result.value_grid.shape
        and artifact.exercise_indicator_grid.shape == artifact.result.value_grid.shape
        and artifact.greek_diagnostics.arrays.delta.shape == artifact.result.value_grid.shape
        and artifact.greek_diagnostics.masks.strict_interior.shape
        == artifact.result.value_grid.shape
    )
    return evaluate_acceptance(
        all_psor_steps_converged=summary.all_psor_steps_converged,
        max_obstacle_violation=summary.max_obstacle_violation,
        max_equation_violation=summary.max_equation_violation,
        max_abs_complementarity_product=summary.max_abs_complementarity_product,
        metadata_complete=metadata_complete,
    )


def _regime_manifest_row(
    artifact: SmallGridArtifacts,
    total_sample_rows: int,
    accepted_sample_rows: int,
    acceptance_status: str,
    acceptance_reason: str,
) -> dict[str, Any]:
    regime = artifact.regime
    lcp = artifact.lcp_diagnostics.summary
    boundary = artifact.boundary_summary
    greek = artifact.greek_diagnostics.summary
    return {
        "regime_id": regime.regime_id,
        "option_type": regime.option_type,
        "T": regime.T,
        "sigma": regime.sigma,
        "r": regime.r,
        "q": regime.q,
        "K": regime.K,
        "Smax": regime.Smax,
        "M": regime.M,
        "N": regime.N,
        "dS": artifact.dS,
        "dtau": artifact.dtau,
        "split": regime.split,
        "split_reason": regime.split_reason,
        "stress_holdout_flag": regime.stress_holdout_flag,
        "planned_use": regime.planned_use,
        "solver_name": artifact.solver_name,
        "solver_variant": artifact.solver_variant,
        "runtime_seconds": artifact.runtime_seconds,
        "full_grid_rows": int(artifact.result.value_grid.size),
        "total_sample_rows": total_sample_rows,
        "accepted_sample_rows": accepted_sample_rows,
        "all_psor_steps_converged": lcp.all_psor_steps_converged,
        "max_psor_iterations": lcp.max_psor_iterations,
        "mean_psor_iterations": lcp.mean_psor_iterations,
        "max_final_update": lcp.max_final_update,
        "max_obstacle_violation": lcp.max_obstacle_violation,
        "max_equation_violation": lcp.max_equation_violation,
        "max_abs_complementarity_product": lcp.max_abs_complementarity_product,
        "boundary_found_count": boundary.found_boundary_count,
        "boundary_status": boundary.status,
        "finite_delta_count": greek.finite_delta_count,
        "finite_gamma_count": greek.finite_gamma_count,
        "strict_interior_node_count": greek.strict_interior_node_count,
        "acceptance_status": acceptance_status,
        "acceptance_reason": acceptance_reason,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _diagnostic_summary_row(
    artifact: SmallGridArtifacts,
    acceptance_status: str,
    acceptance_reason: str,
) -> dict[str, Any]:
    regime = artifact.regime
    lcp = artifact.lcp_diagnostics.summary
    boundary = artifact.boundary_summary
    greek = artifact.greek_diagnostics.summary
    return {
        "regime_id": regime.regime_id,
        "option_type": regime.option_type,
        "all_psor_steps_converged": lcp.all_psor_steps_converged,
        "psor_step_count": lcp.psor_step_count,
        "max_psor_iterations": lcp.max_psor_iterations,
        "mean_psor_iterations": lcp.mean_psor_iterations,
        "max_final_update": lcp.max_final_update,
        "max_obstacle_violation": lcp.max_obstacle_violation,
        "max_equation_violation": lcp.max_equation_violation,
        "max_abs_complementarity_product": lcp.max_abs_complementarity_product,
        "lcp_status": lcp.status,
        "boundary_found_count": boundary.found_boundary_count,
        "boundary_threshold": artifact.premium_threshold,
        "boundary_status": boundary.status,
        "finite_delta_count": greek.finite_delta_count,
        "finite_gamma_count": greek.finite_gamma_count,
        "nonfinite_delta_count": greek.nonfinite_delta_count,
        "nonfinite_gamma_count": greek.nonfinite_gamma_count,
        "max_abs_gamma": greek.max_abs_gamma,
        "max_abs_gamma_strict": greek.max_abs_gamma_strict,
        "boundary_near_node_count": greek.boundary_near_node_count,
        "kink_near_node_count": greek.kink_near_node_count,
        "maturity_masked_node_count": greek.maturity_masked_node_count,
        "strict_interior_node_count": greek.strict_interior_node_count,
        "strict_negative_gamma_count": greek.strict_negative_gamma_count,
        "greek_status": greek.status,
        "runtime_seconds": artifact.runtime_seconds,
        "acceptance_status": acceptance_status,
        "acceptance_reason": acceptance_reason,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _split_assignment_row(
    artifact: SmallGridArtifacts,
    total_sample_rows: int,
    accepted_sample_rows: int,
    acceptance_status: str,
) -> dict[str, Any]:
    regime = artifact.regime
    return {
        "regime_id": regime.regime_id,
        "split": regime.split,
        "split_reason": regime.split_reason,
        "option_family_balance": regime.option_family_balance,
        "stress_holdout_flag": regime.stress_holdout_flag,
        "total_sample_rows": total_sample_rows,
        "accepted_sample_rows": accepted_sample_rows,
        "acceptance_status": acceptance_status,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _failed_regime_manifest_row(regime: SmallGridRegime, reason: str) -> dict[str, Any]:
    row = {field: "" for field in REGIME_MANIFEST_FIELDNAMES}
    row.update(
        {
            "regime_id": regime.regime_id,
            "option_type": regime.option_type,
            "T": regime.T,
            "sigma": regime.sigma,
            "r": regime.r,
            "q": regime.q,
            "K": regime.K,
            "Smax": regime.Smax,
            "M": regime.M,
            "N": regime.N,
            "split": regime.split,
            "split_reason": regime.split_reason,
            "stress_holdout_flag": regime.stress_holdout_flag,
            "planned_use": regime.planned_use,
            "solver_name": SOLVER_NAME,
            "solver_variant": SOLVER_VARIANT,
            "total_sample_rows": 0,
            "accepted_sample_rows": 0,
            "acceptance_status": "review_required",
            "acceptance_reason": reason,
            "downstream_use_status": DOWNSTREAM_USE_STATUS,
        }
    )
    return row


def _failed_diagnostic_row(regime: SmallGridRegime, reason: str) -> dict[str, Any]:
    row = {field: "" for field in DIAGNOSTIC_SUMMARY_FIELDNAMES}
    row.update(
        {
            "regime_id": regime.regime_id,
            "option_type": regime.option_type,
            "acceptance_status": "review_required",
            "acceptance_reason": reason,
            "downstream_use_status": DOWNSTREAM_USE_STATUS,
        }
    )
    return row


def _failed_split_row(regime: SmallGridRegime, reason: str) -> dict[str, Any]:
    return {
        "regime_id": regime.regime_id,
        "split": regime.split,
        "split_reason": f"{regime.split_reason}; {reason}",
        "option_family_balance": regime.option_family_balance,
        "stress_holdout_flag": regime.stress_holdout_flag,
        "total_sample_rows": 0,
        "accepted_sample_rows": 0,
        "acceptance_status": "review_required",
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _schema_snapshot_rows() -> list[dict[str, Any]]:
    definitions = {
        "feature": {
            "log_moneyness": "log(S/K)",
            "tau_fraction": "tau/T",
            "r": "risk-free rate",
            "q": "continuous dividend yield",
            "sigma": "volatility",
            "T": "maturity in years",
            "is_call": "1 for call and 0 for put",
        },
        "label": {
            "value_over_K": "American value divided by K",
            "payoff_over_K": "payoff divided by K",
            "premium_over_K": "continuation premium divided by K",
            "exercise_indicator": "1 when premium is within threshold of zero",
            "boundary_spot_over_K": "threshold boundary spot divided by K",
            "delta": "finite-difference Delta",
            "scaled_gamma": "K times finite-difference Gamma",
        },
        "mask": {
            "payoff_kink_near": "node is near S=K",
            "boundary_near": "node is near extracted boundary",
            "maturity_row": "tau equals zero row",
            "strict_interior": "finite Greek node away from kink, boundary, and maturity",
            "gamma_allowed_mask": "strict finite-Gamma node",
            "delta_allowed_mask": "strict finite-Delta node",
            "exercise_region": "premium-threshold exercise-like node",
            "continuation_region": "premium-threshold continuation-like node",
        },
        "audit_numeric": {
            name: "numeric audit metadata" for name in AUDIT_NUMERIC_NAMES
        },
    }
    groups = (
        ("feature", FEATURE_NAMES, "float64", "model_input_candidate"),
        ("label", LABEL_NAMES, "float64", "label_or_diagnostic"),
        ("mask", MASK_NAMES, "bool", "mask_metadata"),
        ("audit_numeric", AUDIT_NUMERIC_NAMES, "float64", "audit_metadata"),
    )
    rows: list[dict[str, Any]] = []
    for group, names, dtype, role in groups:
        for order, name in enumerate(names):
            rows.append(
                {
                    "schema_group": group,
                    "field_name": name,
                    "field_order": order,
                    "definition": definitions[group][name],
                    "dtype": dtype,
                    "role": role,
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def _quality_summary_rows(
    regime_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    npz_path: Path,
    review_decision: str,
) -> list[dict[str, Any]]:
    accepted_regimes = sum(row["acceptance_status"] == "accepted" for row in regime_rows)
    total_rows = int(arrays["X"].shape[0])
    rows = [
        _quality_row("total_regimes", len(regime_rows), "INFO", review_decision, "Planned v1 regimes."),
        _quality_row(
            "accepted_regimes",
            accepted_regimes,
            "PASS" if accepted_regimes == len(regime_rows) else "REVIEW",
            review_decision,
            "Regimes accepted into arrays.",
        ),
        _quality_row("accepted_rows", total_rows, "INFO", review_decision, "Accepted sampled rows."),
        _quality_row(
            "npz_file_bytes",
            npz_path.stat().st_size if npz_path.exists() else 0,
            "INFO",
            review_decision,
            "Compressed package size.",
        ),
    ]
    for split in SPLIT_ORDER:
        rows.append(
            _quality_row(
                f"{split}_accepted_rows",
                sum(int(row["accepted_sample_rows"]) for row in regime_rows if row["split"] == split),
                "INFO",
                review_decision,
                "Accepted rows by split.",
            )
        )
    if diagnostic_rows:
        numeric_metrics = (
            ("max_obstacle_violation", OBSTACLE_TOLERANCE),
            ("max_equation_violation", EQUATION_TOLERANCE),
            ("max_abs_complementarity_product", COMPLEMENTARITY_TOLERANCE),
        )
        for metric, tolerance in numeric_metrics:
            values = [float(row[metric]) for row in diagnostic_rows if row.get(metric) not in ("", None)]
            max_value = max(values) if values else float("nan")
            rows.append(
                _quality_row(
                    metric,
                    max_value,
                    "PASS" if np.isfinite(max_value) and max_value <= tolerance else "REVIEW",
                    review_decision,
                    f"Tolerance {tolerance}.",
                )
            )
        rows.append(
            _quality_row(
                "boundary_found_total",
                sum(float(row["boundary_found_count"]) for row in diagnostic_rows if row.get("boundary_found_count") not in ("", None)),
                "INFO",
                review_decision,
                "Boundary-found time rows across regimes.",
            )
        )
        rows.append(
            _quality_row(
                "strict_interior_node_total",
                sum(float(row["strict_interior_node_count"]) for row in diagnostic_rows if row.get("strict_interior_node_count") not in ("", None)),
                "INFO",
                review_decision,
                "Full-grid strict Greek mask nodes across regimes.",
            )
        )
    return rows


def _quality_row(
    metric: str,
    value: Any,
    status: str,
    review_decision: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "metric_value": value,
        "status": status,
        "review_decision": review_decision,
        "notes": notes,
    }


def _higher_grid_confirmation_rows(
    regimes: tuple[SmallGridRegime, ...],
    baseline_artifacts: dict[str, SmallGridArtifacts],
    premium_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regimes_by_id = {regime.regime_id: regime for regime in regimes}
    for regime_id in HIGHER_GRID_CONFIRMATION_IDS:
        if regime_id not in regimes_by_id or regime_id not in baseline_artifacts:
            rows.append(
                _higher_grid_deferred_row(
                    regime_id,
                    "Baseline regime was not part of this generation run.",
                )
            )
            continue
        baseline = baseline_artifacts[regime_id]
        confirmation_regime = replace(regimes_by_id[regime_id], M=180, N=180)
        try:
            confirmation = run_small_grid_regime(
                confirmation_regime,
                premium_threshold=premium_threshold,
            )
            status, reason = _artifact_acceptance(confirmation)
            value_diffs, premium_diffs = _selected_confirmation_differences(
                baseline.result,
                baseline.premium_grid,
                confirmation.result,
                confirmation.premium_grid,
                baseline.regime.K,
            )
            rows.append(
                {
                    "regime_id": regime_id,
                    "option_type": baseline.regime.option_type,
                    "T": baseline.regime.T,
                    "sigma": baseline.regime.sigma,
                    "r": baseline.regime.r,
                    "q": baseline.regime.q,
                    "baseline_M": baseline.regime.M,
                    "baseline_N": baseline.regime.N,
                    "confirmation_M": confirmation.regime.M,
                    "confirmation_N": confirmation.regime.N,
                    "baseline_runtime_seconds": baseline.runtime_seconds,
                    "confirmation_runtime_seconds": confirmation.runtime_seconds,
                    "max_abs_value_difference": _nanmax_abs(value_diffs),
                    "mean_abs_value_difference": _nanmean_abs(value_diffs),
                    "max_abs_premium_difference": _nanmax_abs(premium_diffs),
                    "baseline_boundary_found_count": baseline.boundary_summary.found_boundary_count,
                    "confirmation_boundary_found_count": confirmation.boundary_summary.found_boundary_count,
                    "confirmation_acceptance_status": status,
                    "notes": reason,
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive audit path
            rows.append(
                _higher_grid_deferred_row(
                    regime_id,
                    f"confirmation_exception:{type(exc).__name__}:{exc}",
                )
            )
    return rows


def _higher_grid_deferred_row(regime_id: str, notes: str) -> dict[str, Any]:
    row = {field: "" for field in HIGHER_GRID_CONFIRMATION_FIELDNAMES}
    row.update(
        {
            "regime_id": regime_id,
            "confirmation_acceptance_status": "deferred_or_review_required",
            "notes": notes,
            "downstream_use_status": DOWNSTREAM_USE_STATUS,
        }
    )
    return row


def _selected_confirmation_differences(
    baseline_result: AmericanCNPSORResult,
    baseline_premium: np.ndarray,
    confirmation_result: AmericanCNPSORResult,
    confirmation_premium: np.ndarray,
    K: float,
) -> tuple[np.ndarray, np.ndarray]:
    moneyness_values = (0.5, 0.8, 1.0, 1.2, 1.5, 1.8)
    tau_fractions = (0.01, 0.25, 0.50, 0.75, 1.00)
    value_diffs: list[float] = []
    premium_diffs: list[float] = []
    for tau_fraction in tau_fractions:
        base_tau_index = _nearest_index(baseline_result.tau_grid, tau_fraction * baseline_result.T)
        confirm_tau_index = _nearest_index(
            confirmation_result.tau_grid, tau_fraction * confirmation_result.T
        )
        for moneyness in moneyness_values:
            spot = moneyness * K
            base_spot_index = _nearest_index(baseline_result.spot_grid, spot)
            confirm_spot_index = _nearest_index(confirmation_result.spot_grid, spot)
            value_diffs.append(
                float(
                    confirmation_result.value_grid[confirm_tau_index, confirm_spot_index]
                    - baseline_result.value_grid[base_tau_index, base_spot_index]
                )
            )
            premium_diffs.append(
                float(
                    confirmation_premium[confirm_tau_index, confirm_spot_index]
                    - baseline_premium[base_tau_index, base_spot_index]
                )
            )
    return np.array(value_diffs, dtype=float), np.array(premium_diffs, dtype=float)


def _output_manifest_rows(
    npz_path: Path,
    regime_manifest_path: Path,
    diagnostic_summary_path: Path,
    split_assignment_path: Path,
    schema_snapshot_path: Path,
    output_manifest_path: Path,
    generation_log_path: Path,
    quality_summary_path: Path,
    higher_grid_confirmation_path: Path,
    figure_paths: list[Path],
    figure_status: dict[str, bool],
    regime_count: int,
    accepted_row_count: int,
    quality_row_count: int,
    higher_grid_row_count: int,
    review_decision: str,
) -> list[dict[str, Any]]:
    base_rows = [
        ("dataset_v1_small_grid_npz", "npz", npz_path, "Compressed v1 arrays for accepted rows only.", accepted_row_count),
        ("regime_manifest", "csv", regime_manifest_path, "Per-regime v1 parameter and acceptance manifest.", regime_count),
        ("diagnostic_summary", "csv", diagnostic_summary_path, "Per-regime PSOR, LCP, boundary, and Greek diagnostics.", regime_count),
        ("split_assignment", "csv", split_assignment_path, "Regime-level split assignments and accepted row counts.", regime_count),
        ("schema_snapshot", "csv", schema_snapshot_path, "Feature, label, mask, and audit schema snapshot.", len(_schema_snapshot_rows())),
        ("quality_summary", "csv", quality_summary_path, "Aggregate v1 dataset QA summary.", quality_row_count),
        ("higher_grid_confirmation", "csv", higher_grid_confirmation_path, "QA-only M=N=180 confirmation rows.", higher_grid_row_count),
        ("generation_log", "txt", generation_log_path, "Plain-text generation progress log.", 0),
        ("output_manifest", "csv", output_manifest_path, "Manifest for every v1 output.", 0),
    ]
    rows = [
        _output_row(
            output_id,
            output_type,
            path,
            description,
            row_count,
            review_decision,
            created=True if output_id in {"generation_log", "output_manifest"} else None,
        )
        for output_id, output_type, path, description, row_count in base_rows
    ]
    for path in figure_paths:
        name = path.stem.replace("v1_", "")
        rows.append(
            _output_row(
                path.stem,
                "png",
                path,
                "Optional v1 dataset QA figure.",
                0,
                review_decision,
                created=figure_status.get(name, path.exists()),
            )
        )
    return rows


def _output_row(
    output_id: str,
    output_type: str,
    path: Path,
    description: str,
    row_count: int,
    review_decision: str,
    created: bool | None = None,
) -> dict[str, Any]:
    return {
        "output_id": output_id,
        "output_type": output_type,
        "path": str(path),
        "created": path.exists() if created is None else bool(created),
        "description": description,
        "row_count": row_count,
        "solver_name": SOLVER_NAME,
        "solver_variant": SOLVER_VARIANT,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
        "review_status": review_decision,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        raise ValueError(f"required planning CSV is missing: {path}")
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV row-length drift at {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise ValueError(f"required planning CSV has no data rows: {path}")
    return rows


def _validate_regime(regime: SmallGridRegime) -> SmallGridRegime:
    if not isinstance(regime, SmallGridRegime):
        raise ValueError("regime must be a SmallGridRegime.")
    if regime.option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    for name in ("regime_id", "split", "solver_variant"):
        if not getattr(regime, name):
            raise ValueError(f"{name} must be nonempty.")
    if regime.solver_variant != SOLVER_VARIANT:
        raise ValueError("v1 generation requires baseline_cn_psor.")
    if regime.split not in SPLIT_ORDER:
        raise ValueError("split must be one of the approved regime-level splits.")
    for name in ("K", "T", "Smax", "sigma"):
        if float(getattr(regime, name)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if regime.M < 3 or regime.N < 1:
        raise ValueError("M must be at least 3 and N must be positive.")
    return regime


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if value < 0.0:
        raise ValueError("threshold must be nonnegative.")
    return value


def _spacing(grid: Any) -> float:
    values = np.asarray(grid, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(values[1] - values[0])


def _split_index(split: str) -> int:
    mapping = {split_name: index for index, split_name in enumerate(SPLIT_ORDER)}
    return mapping.get(split, -1)


def _nearest_index(values: Any, target: float) -> int:
    array = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(array - float(target))))


def _concat_1d(chunks: list[np.ndarray], dtype: Any) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=dtype)
    return np.concatenate(chunks).astype(dtype, copy=False)


def _concat_2d(chunks: list[np.ndarray], columns: int, dtype: Any) -> np.ndarray:
    if not chunks:
        return np.empty((0, columns), dtype=dtype)
    return np.vstack(chunks).astype(dtype, copy=False)


def _nanmax_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def _nanmean_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))
