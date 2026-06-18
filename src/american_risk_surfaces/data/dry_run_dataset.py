"""v0 Dry-Run Dataset: eight-regime surrogate dataset generation test."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
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
    PROJECT_ROOT / "results" / "04_surrogate_dataset" / "v0_dry_run"
)

SOLVER_NAME = "american_crank_nicolson_psor_price"
SOLVER_VARIANT = "baseline_cn_psor"
DOWNSTREAM_USE_STATUS = "v0_dry_run_only"
PREMIUM_THRESHOLD = 1e-6
INTERPRETATION_MONEYNESS_BOUNDS = (0.4, 1.8)
OBSTACLE_TOLERANCE = 1e-8
EQUATION_TOLERANCE = 1e-6
COMPLEMENTARITY_TOLERANCE = 1e-6

APPROVED_DRY_RUN_IDS = (
    "dry_01",
    "dry_02",
    "dry_03",
    "dry_04",
    "dry_05",
    "dry_06",
    "dry_07",
    "dry_08",
)
APPROVED_REGIME_IDS = (
    "put_T100_s020_r005_q003",
    "put_T100_s060_r005_q003",
    "put_T100_s020_r001_q003",
    "put_T200_s040_r010_q010",
    "call_T100_s020_r005_q006",
    "call_T100_s020_r005_q010",
    "call_T100_s060_r005_q006",
    "call_T100_s020_r005_q000",
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

REGIME_MANIFEST_FIELDNAMES = [
    "dry_run_id",
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
    "dry_run_id",
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
    "dry_run_id",
    "regime_id",
    "split",
    "split_reason",
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

__all__ = (
    "DryRunRegime",
    "DryRunArtifacts",
    "DryRunDatasetPackage",
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "MASK_NAMES",
    "AUDIT_NUMERIC_NAMES",
    "REGIME_MANIFEST_FIELDNAMES",
    "DIAGNOSTIC_SUMMARY_FIELDNAMES",
    "SPLIT_ASSIGNMENT_FIELDNAMES",
    "SCHEMA_SNAPSHOT_FIELDNAMES",
    "OUTPUT_MANIFEST_FIELDNAMES",
    "load_dry_run_plan",
    "validate_dry_run_plan",
    "expected_full_regime_count",
    "run_dry_run_regime",
    "continuation_premium_grid",
    "exercise_indicator",
    "evaluate_acceptance",
    "sample_regime_rows",
    "build_dataset_arrays",
    "write_npz_package",
    "generate_v0_dry_run_dataset",
    "write_csv",
    "create_row_counts_figure",
    "create_diagnostics_figure",
    "create_premium_distribution_figure",
)


@dataclass(frozen=True)
class DryRunRegime:
    """One approved v0 dry-run regime."""

    dry_run_id: str
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
    solver_variant: str
    reason_selected: str
    expected_check: str
    required_manual_review: str
    stress_holdout_flag: str = "no"


@dataclass(frozen=True)
class DryRunArtifacts:
    """Solver result and diagnostics for one dry-run regime."""

    regime: DryRunRegime
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
class DryRunDatasetPackage:
    """Paths, rows, and decision produced by the v0 dry-run generator."""

    output_dir: Path
    npz_path: Path
    regime_manifest_path: Path
    diagnostic_summary_path: Path
    split_assignment_path: Path
    schema_snapshot_path: Path
    output_manifest_path: Path
    figure_paths: tuple[Path, ...]
    regime_rows: list[dict[str, Any]]
    diagnostic_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    schema_rows: list[dict[str, Any]]
    output_rows: list[dict[str, Any]]
    review_decision: str
    accepted_row_count: int
    total_regime_count: int


def expected_full_regime_count(plan_dir: Path = SURROGATE_PLAN_DIR) -> int:
    """Return the planned full-grid regime count used only for validation."""

    return len(_read_csv(plan_dir / "dataset_regime_plan.csv"))


def load_dry_run_plan(plan_dir: Path = SURROGATE_PLAN_DIR) -> tuple[DryRunRegime, ...]:
    """Load and validate the approved eight-regime dry-run plan."""

    plan_path = Path(plan_dir)
    regime_rows = _read_csv(plan_path / "dataset_regime_plan.csv")
    dry_rows = _read_csv(plan_path / "dataset_dry_run_subset_plan.csv")
    split_rows = _read_csv(plan_path / "dataset_split_assignment_plan.csv")
    regimes_by_id = {row["regime_id"]: row for row in regime_rows}
    splits_by_id = {row["regime_id"]: row for row in split_rows}
    regimes = []
    for dry_row in dry_rows:
        regime_id = dry_row["regime_id"]
        if regime_id not in regimes_by_id:
            raise ValueError(f"dry-run regime {regime_id} is missing from regime plan.")
        if regime_id not in splits_by_id:
            raise ValueError(f"dry-run regime {regime_id} is missing from split plan.")
        regime = regimes_by_id[regime_id]
        split = splits_by_id[regime_id]
        regimes.append(
            DryRunRegime(
                dry_run_id=dry_row["dry_run_id"],
                regime_id=regime_id,
                option_type=regime["option_type"],
                T=float(regime["T"]),
                sigma=float(regime["sigma"]),
                r=float(regime["r"]),
                q=float(regime["q"]),
                K=float(regime["K"]),
                Smax=float(regime["Smax"]),
                M=int(regime["M"]),
                N=int(regime["N"]),
                split=split["split"],
                split_reason=split["split_reason"],
                solver_variant=regime["solver_variant"],
                reason_selected=dry_row["reason_selected"],
                expected_check=dry_row["expected_check"],
                required_manual_review=dry_row["required_manual_review"],
                stress_holdout_flag=split.get("stress_holdout_flag", "no"),
            )
        )
    validate_dry_run_plan(tuple(regimes), full_regime_count=len(regime_rows))
    return tuple(regimes)


def validate_dry_run_plan(
    regimes: tuple[DryRunRegime, ...],
    full_regime_count: int = 288,
) -> None:
    """Validate that only the approved v0 dry-run regimes will be generated."""

    if full_regime_count != 288:
        raise ValueError("dataset_regime_plan.csv must contain exactly 288 planned regimes.")
    if len(regimes) != 8:
        raise ValueError("v0 dry run must contain exactly eight regimes.")
    if tuple(regime.dry_run_id for regime in regimes) != APPROVED_DRY_RUN_IDS:
        raise ValueError("dry-run IDs do not match the approved construction plan.")
    if tuple(regime.regime_id for regime in regimes) != APPROVED_REGIME_IDS:
        raise ValueError("dry-run regime IDs do not match the approved construction plan.")

    for regime in regimes:
        _validate_regime(regime)
        if regime.K != 1.0:
            raise ValueError("dry-run regimes must use K=1.")
        if regime.Smax != 4.0:
            raise ValueError("dry-run regimes must use Smax=4.")
        if (regime.M, regime.N) != (120, 120):
            raise ValueError("approved dry-run regimes must use M=N=120.")
        if regime.solver_variant != SOLVER_VARIANT:
            raise ValueError("dry-run regimes must use baseline_cn_psor.")


def run_dry_run_regime(
    regime: DryRunRegime,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> DryRunArtifacts:
    """Run one dry-run regime with the baseline American CN/PSOR solver."""

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
    return DryRunArtifacts(
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
    """Return the dry-run continuation premium grid, U minus payoff."""

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
    """Return v0 dry-run acceptance status and reason text."""

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


def sample_regime_rows(artifacts: DryRunArtifacts) -> list[dict[str, Any]]:
    """Return regular-grid sample rows in the approved reporting region."""

    if not isinstance(artifacts, DryRunArtifacts):
        raise ValueError("artifacts must be a DryRunArtifacts instance.")
    result = artifacts.result
    regime = artifacts.regime
    lower, upper = INTERPRETATION_MONEYNESS_BOUNDS
    rows: list[dict[str, Any]] = []
    split_index = _split_index(regime.split)
    for time_index, tau in enumerate(result.tau_grid):
        point = artifacts.boundary_curve.points[time_index]
        boundary_over_K = (
            float(point.boundary_spot / regime.K)
            if point.boundary_found and np.isfinite(point.boundary_spot)
            else float("nan")
        )
        tau_fraction = float(tau / regime.T) if regime.T > 0.0 else 0.0
        for spot_index, spot in enumerate(result.spot_grid):
            moneyness = float(spot / regime.K)
            if moneyness < lower - 1e-14 or moneyness > upper + 1e-14:
                continue
            value = float(result.value_grid[time_index, spot_index])
            payoff = float(result.payoff[spot_index])
            premium = float(artifacts.premium_grid[time_index, spot_index])
            delta = float(artifacts.greek_diagnostics.arrays.delta[time_index, spot_index])
            gamma = float(artifacts.greek_diagnostics.arrays.gamma[time_index, spot_index])
            boundary_near = bool(
                artifacts.greek_diagnostics.masks.boundary_near[time_index, spot_index]
            )
            kink_near = bool(
                artifacts.greek_diagnostics.masks.payoff_kink_near[time_index, spot_index]
            )
            maturity_row = bool(
                artifacts.greek_diagnostics.masks.maturity_row[time_index, spot_index]
            )
            strict = bool(
                artifacts.greek_diagnostics.masks.strict_interior[time_index, spot_index]
            )
            finite_delta = bool(
                artifacts.greek_diagnostics.arrays.finite_delta_mask[time_index, spot_index]
            )
            finite_gamma = bool(
                artifacts.greek_diagnostics.arrays.finite_gamma_mask[time_index, spot_index]
            )
            exercise_flag = int(artifacts.exercise_indicator_grid[time_index, spot_index])
            rows.append(
                {
                    "dry_run_id": regime.dry_run_id,
                    "regime_id": regime.regime_id,
                    "option_type": regime.option_type,
                    "split": regime.split,
                    "split_index": split_index,
                    "regime_index": -1,
                    "log_moneyness": float(np.log(moneyness)),
                    "tau_fraction": tau_fraction,
                    "r": regime.r,
                    "q": regime.q,
                    "sigma": regime.sigma,
                    "T": regime.T,
                    "is_call": 1.0 if regime.option_type == "call" else 0.0,
                    "value_over_K": value / regime.K,
                    "payoff_over_K": payoff / regime.K,
                    "premium_over_K": premium / regime.K,
                    "exercise_indicator": exercise_flag,
                    "boundary_spot_over_K": boundary_over_K,
                    "delta": delta,
                    "scaled_gamma": regime.K * gamma,
                    "payoff_kink_near": kink_near,
                    "boundary_near": boundary_near,
                    "maturity_row": maturity_row,
                    "strict_interior": strict,
                    "gamma_allowed_mask": bool(strict and finite_gamma),
                    "delta_allowed_mask": bool(strict and finite_delta),
                    "exercise_region": bool(exercise_flag == 1),
                    "continuation_region": bool(exercise_flag == 0),
                    "S_over_K": moneyness,
                    "tau": float(tau),
                    "S": float(spot),
                    "K": regime.K,
                    "Smax": regime.Smax,
                    "M": float(regime.M),
                    "N": float(regime.N),
                    "dS": artifacts.dS,
                    "dtau": artifacts.dtau,
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def build_dataset_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build the accepted-row v0 dry-run arrays for compressed storage."""

    regime_ids = _ordered_unique(row["regime_id"] for row in rows)
    dry_run_ids = _ordered_unique(row["dry_run_id"] for row in rows)
    split_names = _ordered_unique(row["split"] for row in rows)
    regime_lookup = {regime_id: index for index, regime_id in enumerate(regime_ids)}
    for row in rows:
        row["regime_index"] = regime_lookup[row["regime_id"]]

    X = _matrix(rows, FEATURE_NAMES)
    masks = _matrix(rows, MASK_NAMES).astype(bool)
    audit_numeric = _matrix(rows, AUDIT_NUMERIC_NAMES)
    arrays = {
        "X": X,
        "y_value": _vector(rows, "value_over_K"),
        "y_payoff": _vector(rows, "payoff_over_K"),
        "y_premium": _vector(rows, "premium_over_K"),
        "y_exercise_indicator": _vector(rows, "exercise_indicator"),
        "y_boundary": _vector(rows, "boundary_spot_over_K"),
        "y_delta": _vector(rows, "delta"),
        "y_scaled_gamma": _vector(rows, "scaled_gamma"),
        "masks": masks,
        "regime_index": _vector(rows, "regime_index").astype(int),
        "feature_names": np.array(FEATURE_NAMES, dtype=str),
        "label_names": np.array(LABEL_NAMES, dtype=str),
        "mask_names": np.array(MASK_NAMES, dtype=str),
        "audit_numeric": audit_numeric,
        "audit_numeric_names": np.array(AUDIT_NUMERIC_NAMES, dtype=str),
        "regime_ids": np.array(regime_ids, dtype=str),
        "dry_run_ids": np.array(dry_run_ids, dtype=str),
        "split_names": np.array(split_names, dtype=str),
    }
    return arrays


def write_npz_package(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write exactly one compressed dry-run dataset package."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def generate_v0_dry_run_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    regimes: tuple[DryRunRegime, ...] | None = None,
    create_figures: bool = True,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> DryRunDatasetPackage:
    """Generate the approved v0 dry-run package and audit manifests."""

    selected_regimes = regimes if regimes is not None else load_dry_run_plan()
    if regimes is None:
        validate_dry_run_plan(selected_regimes)
    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = [
        run_dry_run_regime(regime, premium_threshold=premium_threshold)
        for regime in selected_regimes
    ]
    accepted_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    sampled_rows_by_id: dict[str, list[dict[str, Any]]] = {}

    for artifact in artifacts:
        sampled_rows = sample_regime_rows(artifact)
        sampled_rows_by_id[artifact.regime.regime_id] = sampled_rows
        status, reason = _artifact_acceptance(artifact)
        accepted_sample_rows = len(sampled_rows) if status == "accepted" else 0
        if status == "accepted":
            accepted_rows.extend(sampled_rows)
        regime_rows.append(
            _regime_manifest_row(artifact, len(sampled_rows), accepted_sample_rows, status, reason)
        )
        diagnostic_rows.append(_diagnostic_summary_row(artifact, status, reason))
        split_rows.append(_split_assignment_row(artifact, len(sampled_rows), accepted_sample_rows, status))

    arrays = build_dataset_arrays(accepted_rows)
    npz_path = output_path / "dataset_v0_dry_run.npz"
    write_npz_package(npz_path, arrays)

    regime_manifest_path = output_path / "regime_manifest.csv"
    diagnostic_summary_path = output_path / "diagnostic_summary.csv"
    split_assignment_path = output_path / "split_assignment.csv"
    schema_snapshot_path = output_path / "schema_snapshot.csv"
    output_manifest_path = output_path / "output_manifest.csv"

    schema_rows = _schema_snapshot_rows()
    write_csv(regime_manifest_path, regime_rows, REGIME_MANIFEST_FIELDNAMES)
    write_csv(diagnostic_summary_path, diagnostic_rows, DIAGNOSTIC_SUMMARY_FIELDNAMES)
    write_csv(split_assignment_path, split_rows, SPLIT_ASSIGNMENT_FIELDNAMES)
    write_csv(schema_snapshot_path, schema_rows, SCHEMA_SNAPSHOT_FIELDNAMES)

    figure_paths: list[Path] = []
    figure_status: dict[str, bool] = {}
    if create_figures:
        row_counts_path = figure_dir / "v0_dry_run_row_counts.png"
        diagnostics_path = figure_dir / "v0_dry_run_diagnostics.png"
        premium_path = figure_dir / "v0_dry_run_premium_distribution.png"
        figure_status["row_counts"] = create_row_counts_figure(regime_rows, row_counts_path)
        figure_status["diagnostics"] = create_diagnostics_figure(
            diagnostic_rows, diagnostics_path
        )
        figure_status["premium_distribution"] = create_premium_distribution_figure(
            accepted_rows, premium_path
        )
        figure_paths.extend([row_counts_path, diagnostics_path, premium_path])

    review_decision = (
        "READY_FOR_V1_SMALL_GRID_PLANNING"
        if len(regime_rows) == 8
        and all(row["acceptance_status"] == "accepted" for row in regime_rows)
        and npz_path.exists()
        else "REVIEW_REQUIRED_BEFORE_V1"
    )
    output_rows = _output_manifest_rows(
        npz_path=npz_path,
        regime_manifest_path=regime_manifest_path,
        diagnostic_summary_path=diagnostic_summary_path,
        split_assignment_path=split_assignment_path,
        schema_snapshot_path=schema_snapshot_path,
        output_manifest_path=output_manifest_path,
        figure_paths=figure_paths,
        figure_status=figure_status,
        regime_count=len(regime_rows),
        accepted_row_count=len(accepted_rows),
        review_decision=review_decision,
    )
    write_csv(output_manifest_path, output_rows, OUTPUT_MANIFEST_FIELDNAMES)

    return DryRunDatasetPackage(
        output_dir=output_path,
        npz_path=npz_path,
        regime_manifest_path=regime_manifest_path,
        diagnostic_summary_path=diagnostic_summary_path,
        split_assignment_path=split_assignment_path,
        schema_snapshot_path=schema_snapshot_path,
        output_manifest_path=output_manifest_path,
        figure_paths=tuple(figure_paths),
        regime_rows=regime_rows,
        diagnostic_rows=diagnostic_rows,
        split_rows=split_rows,
        schema_rows=schema_rows,
        output_rows=output_rows,
        review_decision=review_decision,
        accepted_row_count=len(accepted_rows),
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


def create_row_counts_figure(regime_rows: list[dict[str, Any]], path: Path) -> bool:
    """Create a small row-count figure for the dry-run review."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not regime_rows:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["dry_run_id"] for row in regime_rows]
    counts = [int(row["accepted_sample_rows"]) for row in regime_rows]
    colors = ["#4f7cac" if row["split"] == "train" else "#b55d4c" for row in regime_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, counts, color=colors)
    ax.set_title("v0 dry-run accepted row counts")
    ax.set_ylabel("accepted sampled rows")
    ax.set_xlabel("dry-run regime")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_diagnostics_figure(
    diagnostic_rows: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Create a compact LCP diagnostic figure for the dry-run review."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not diagnostic_rows:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["dry_run_id"] for row in diagnostic_rows]
    obstacle = [float(row["max_obstacle_violation"]) for row in diagnostic_rows]
    equation = [float(row["max_equation_violation"]) for row in diagnostic_rows]
    comp = [float(row["max_abs_complementarity_product"]) for row in diagnostic_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(labels, obstacle, marker="o", label="max obstacle")
    ax.plot(labels, equation, marker="s", label="max equation")
    ax.plot(labels, comp, marker="^", label="max complementarity")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_title("v0 dry-run LCP diagnostic stability")
    ax.set_ylabel("diagnostic magnitude")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def create_premium_distribution_figure(
    sampled_rows: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Create a continuation-premium distribution figure for accepted rows."""

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not sampled_rows:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    premiums = np.array([float(row["premium_over_K"]) for row in sampled_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(premiums, bins=60, color="#4f7cac", alpha=0.85)
    ax.set_title("v0 dry-run continuation-premium distribution")
    ax.set_xlabel("premium_over_K")
    ax.set_ylabel("sample count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination.exists()


def _artifact_acceptance(artifact: DryRunArtifacts) -> tuple[str, str]:
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
    artifact: DryRunArtifacts,
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
        "dry_run_id": regime.dry_run_id,
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
    artifact: DryRunArtifacts,
    acceptance_status: str,
    acceptance_reason: str,
) -> dict[str, Any]:
    regime = artifact.regime
    lcp = artifact.lcp_diagnostics.summary
    boundary = artifact.boundary_summary
    greek = artifact.greek_diagnostics.summary
    return {
        "dry_run_id": regime.dry_run_id,
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
    artifact: DryRunArtifacts,
    total_sample_rows: int,
    accepted_sample_rows: int,
    acceptance_status: str,
) -> dict[str, Any]:
    regime = artifact.regime
    return {
        "dry_run_id": regime.dry_run_id,
        "regime_id": regime.regime_id,
        "split": regime.split,
        "split_reason": regime.split_reason,
        "stress_holdout_flag": regime.stress_holdout_flag,
        "total_sample_rows": total_sample_rows,
        "accepted_sample_rows": accepted_sample_rows,
        "acceptance_status": acceptance_status,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def _schema_snapshot_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        ("feature", FEATURE_NAMES, "float64", "model_input"),
        ("label", LABEL_NAMES, "float64", "dry_run_label_or_diagnostic"),
        ("mask", MASK_NAMES, "bool", "mask_metadata"),
        ("audit_numeric", AUDIT_NUMERIC_NAMES, "float64", "audit_metadata"),
    )
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


def _output_manifest_rows(
    npz_path: Path,
    regime_manifest_path: Path,
    diagnostic_summary_path: Path,
    split_assignment_path: Path,
    schema_snapshot_path: Path,
    output_manifest_path: Path,
    figure_paths: list[Path],
    figure_status: dict[str, bool],
    regime_count: int,
    accepted_row_count: int,
    review_decision: str,
) -> list[dict[str, Any]]:
    base_rows = [
        (
            "dataset_v0_dry_run_npz",
            "npz",
            npz_path,
            "Compressed dry-run arrays for accepted rows only.",
            accepted_row_count,
        ),
        (
            "regime_manifest",
            "csv",
            regime_manifest_path,
            "Per-regime dry-run parameter and acceptance manifest.",
            regime_count,
        ),
        (
            "diagnostic_summary",
            "csv",
            diagnostic_summary_path,
            "Per-regime PSOR, LCP, boundary, and Greek diagnostics.",
            regime_count,
        ),
        (
            "split_assignment",
            "csv",
            split_assignment_path,
            "Regime-level split assignments and accepted row counts.",
            regime_count,
        ),
        (
            "schema_snapshot",
            "csv",
            schema_snapshot_path,
            "Feature, label, mask, and audit schema snapshot.",
            len(_schema_snapshot_rows()),
        ),
        (
            "output_manifest",
            "csv",
            output_manifest_path,
            "Manifest for every v0 dry-run output.",
            0,
        ),
    ]
    rows = []
    for output_id, output_type, path, description, row_count in base_rows:
        rows.append(
            _output_row(
                output_id,
                output_type,
                path,
                description,
                row_count,
                review_decision,
                created=True if output_id == "output_manifest" else None,
            )
        )
    for path in figure_paths:
        output_id = path.stem
        rows.append(
            _output_row(
                output_id,
                "png",
                path,
                "Optional v0 dry-run diagnostic figure.",
                0,
                review_decision,
                created=figure_status.get(output_id.replace("v0_dry_run_", ""), path.exists()),
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
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"required planning CSV has no data rows: {path}")
    return rows


def _validate_regime(regime: DryRunRegime) -> DryRunRegime:
    if not isinstance(regime, DryRunRegime):
        raise ValueError("regime must be a DryRunRegime.")
    if regime.option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    for name in ("dry_run_id", "regime_id", "split", "solver_variant"):
        if not getattr(regime, name):
            raise ValueError(f"{name} must be nonempty.")
    if regime.solver_variant != SOLVER_VARIANT:
        raise ValueError("dry-run generation requires baseline_cn_psor.")
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
    mapping = {"train": 0, "validation": 1, "test": 2, "stress_holdout": 3}
    return mapping.get(split, -1)


def _ordered_unique(values: Any) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return tuple(seen)


def _vector(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    if not rows:
        return np.empty((0,), dtype=float)
    return np.array([float(row[name]) for row in rows], dtype=float)


def _matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(names)), dtype=float)
    return np.array([[float(row[name]) for name in names] for row in rows], dtype=float)
