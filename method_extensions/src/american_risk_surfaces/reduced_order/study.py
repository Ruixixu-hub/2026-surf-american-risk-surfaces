"""Leakage-safe orchestration helpers for Experiments 36--41."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.reduced_order.basis import (
    build_primal_dual_basis_ladder,
    load_basis,
    save_basis,
)
from american_risk_surfaces.reduced_order.metrics import score_value_trajectory
from american_risk_surfaces.reduced_order.protocol import (
    DIMENSION_LADDER,
    REDUCTION_RMSE_LIMIT,
    RESULTS_DIR,
    RBRegime,
    load_regimes,
    protocol_hash,
)
from american_risk_surfaces.reduced_order.snapshots import (
    generate_fom_snapshot,
    trajectory_multipliers,
)
from american_risk_surfaces.reduced_order.solver import (
    assemble_affine_rb_operator,
    solve_reduced_american_vi,
)
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid
from american_risk_surfaces.reduced_order.metrics import interpolate_reference_surface


SNAPSHOT_DIR = RESULTS_DIR / "01_snapshots" / "train_only"
BASIS_DIR = RESULTS_DIR / "02_basis"
VALIDATION_DIR = RESULTS_DIR / "04_validation"
HELDOUT_DIR = RESULTS_DIR / "05_heldout"


def generate_train_snapshots(
    *,
    option_type: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> Path:
    regimes = load_regimes(splits=("train",), option_type=option_type)
    if limit is not None:
        regimes = regimes[: int(limit)]
    rows: list[dict[str, object]] = []
    for regime in regimes:
        path = SNAPSHOT_DIR / regime.option_type / f"{regime.regime_id}.npz"
        started = perf_counter()
        status = "REUSED" if resume and path.exists() else "GENERATED"
        if status == "GENERATED":
            snapshot = generate_fom_snapshot(regime, path)
            residual = float(np.max(snapshot.residual_by_time[:, 3]))
        else:
            with np.load(path, allow_pickle=False) as data:
                residual = float(np.max(data["residual_by_time"][:, 3]))
                generation_seconds = float(
                    json.loads(str(data["metadata_json"])).get("generation_seconds", 0.0)
                )
        rows.append(
            {
                "regime_id": regime.regime_id,
                "option_type": regime.option_type,
                "split": regime.split,
                "path": str(path.relative_to(RESULTS_DIR)),
                "status": status,
                "max_lcp_residual": residual,
                "elapsed_seconds": (
                    float(snapshot.metadata["generation_seconds"])
                    if status == "GENERATED"
                    else generation_seconds
                ),
                "protocol_hash": protocol_hash(),
            }
        )
    manifest = RESULTS_DIR / "01_snapshots" / "snapshot_manifest.csv"
    _write_csv(manifest, rows)
    return manifest


def train_snapshot_paths(option_type: str | None = None) -> list[Path]:
    paths = sorted(SNAPSHOT_DIR.glob("*/*.npz"))
    if option_type is not None:
        paths = [path for path in paths if path.parent.name == option_type]
    expected = {
        regime.regime_id
        for regime in load_regimes(splits=("train",), option_type=option_type)
    }
    actual = {path.stem for path in paths}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"snapshot manifest mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return paths


def build_basis_artifacts(
    *, dimensions: Iterable[int] = DIMENSION_LADDER
) -> tuple[list[Path], Path]:
    rows: list[dict[str, object]] = []
    paths: list[Path] = []
    for family in ("put", "call"):
        for dimension in sorted(set(map(int, dimensions))):
            started = perf_counter()
            try:
                basis = build_primal_dual_basis_ladder(
                    train_snapshot_paths(family), family, (dimension,)
                )[dimension]
                failure_reason = ""
            except RuntimeError as error:
                rows.append(
                    {
                        "option_type": family,
                        "requested_dimension": dimension,
                        "primal_dimension": "",
                        "dual_dimension": "",
                        "inf_sup_constant": 0.0,
                        "condition_number": float("inf"),
                        "construction_seconds": perf_counter() - started,
                        "artifact_bytes": 0,
                        "path": "",
                        "status": "FAILED_STABILITY",
                        "failure_reason": str(error),
                        "protocol_hash": protocol_hash(),
                    }
                )
                continue
            path = BASIS_DIR / family / f"basis_{dimension:02d}.npz"
            save_basis(basis, path)
            paths.append(path)
            rows.append(
                {
                    "option_type": family,
                    "requested_dimension": dimension,
                    "primal_dimension": basis.primal_dimension,
                    "dual_dimension": basis.dual_dimension,
                    "inf_sup_constant": basis.inf_sup_constant,
                    "condition_number": basis.condition_number,
                    "construction_seconds": perf_counter() - started,
                    "artifact_bytes": path.stat().st_size,
                    "path": str(path.relative_to(RESULTS_DIR)),
                    "status": "COMPLETE",
                    "failure_reason": failure_reason,
                    "protocol_hash": protocol_hash(),
                }
            )
    manifest = BASIS_DIR / "basis_manifest.csv"
    _write_csv(manifest, rows)
    return paths, manifest


def evaluate_basis_ladder(
    regimes: Iterable[RBRegime],
    *,
    dimensions: Iterable[int] = DIMENSION_LADDER,
    reference_m: int = 480,
    reference_n: int = 960,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    artifacts = {}
    for family in ("put", "call"):
        for dimension in dimensions:
            path = BASIS_DIR / family / f"basis_{dimension:02d}.npz"
            if path.exists():
                artifacts[(family, dimension)] = assemble_affine_rb_operator(load_basis(path))
    for regime in sorted(regimes, key=lambda item: item.regime_id):
        config = regime.config()
        fom_started = perf_counter()
        full = american_cn_lcp_price(config, lcp_solver="policy_iteration")
        fom_seconds = perf_counter() - fom_started
        reference_multiplier, _, _ = trajectory_multipliers(config, full.value_grid)
        fine_config = type(config)(
            config.option_type,
            config.K,
            config.T,
            config.r,
            config.q,
            config.sigma,
            config.Smax,
            int(reference_m),
            int(reference_n),
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        reference_started = perf_counter()
        fine = american_dirk_policy_price(
            fine_config,
            quadratic_time=True,
            damping_steps=2,
            spot_grid=sinh_spot_grid(config.Smax, config.K, int(reference_m)),
        )
        high_reference = interpolate_reference_surface(
            fine.value_grid,
            fine.spot_grid,
            fine.tau_grid,
            full.spot_grid,
            full.tau_grid,
        )
        reference_seconds = perf_counter() - reference_started
        cn_high_metrics = score_value_trajectory(
            full.value_grid,
            high_reference,
            full.payoff,
            full.spot_grid,
            full.tau_grid,
            regime.option_type,
        )
        for dimension in dimensions:
            if (regime.option_type, dimension) not in artifacts:
                rows.append(
                    {
                        "regime_id": regime.regime_id,
                        "option_type": regime.option_type,
                        "split": regime.split,
                        "requested_dimension": dimension,
                        "primal_dimension": "",
                        "dual_dimension": dimension,
                        "converged": False,
                        "failure_reason": "basis artifact failed stability gate",
                        "reduced_residual_max": float("inf"),
                        "raw_full_lcp_residual": float("inf"),
                        "projected_full_lcp_residual": float("inf"),
                        "raw_obstacle_violation": float("inf"),
                        "projected_obstacle_violation": float("inf"),
                        "cn_high_price_rmse": cn_high_metrics["price_rmse"],
                        "cn_high_delta_rmse": cn_high_metrics["delta_rmse"],
                        "cn_high_stable_gamma_rmse": cn_high_metrics["stable_gamma_rmse"],
                        "fom_generation_seconds": fom_seconds,
                        "high_reference_generation_seconds": reference_seconds,
                        "timing_total_seconds": 0.0,
                    }
                )
                continue
            result = solve_reduced_american_vi(artifacts[(regime.option_type, dimension)], config)
            base: dict[str, object] = {
                "regime_id": regime.regime_id,
                "option_type": regime.option_type,
                "split": regime.split,
                "requested_dimension": dimension,
                "primal_dimension": artifacts[(regime.option_type, dimension)].basis.primal_dimension,
                "dual_dimension": dimension,
                "converged": result.converged,
                "failure_reason": result.failure_reason or "",
                "reduced_residual_max": result.reduced_residual_max,
                "raw_full_lcp_residual": result.raw_audit["normalized_lcp_residual_max"],
                "projected_full_lcp_residual": result.projected_audit["normalized_lcp_residual_max"],
                "raw_obstacle_violation": result.raw_audit["normalized_obstacle_violation_max"],
                "projected_obstacle_violation": result.projected_audit["normalized_obstacle_violation_max"],
                "cn_high_price_rmse": cn_high_metrics["price_rmse"],
                "cn_high_delta_rmse": cn_high_metrics["delta_rmse"],
                "cn_high_stable_gamma_rmse": cn_high_metrics["stable_gamma_rmse"],
                "fom_generation_seconds": fom_seconds,
                "high_reference_generation_seconds": reference_seconds,
                **{f"timing_{key}": value for key, value in result.timing.items()},
            }
            if result.converged:
                raw_metrics = score_value_trajectory(
                    result.raw_value_grid,
                    full.value_grid,
                    full.payoff,
                    full.spot_grid,
                    full.tau_grid,
                    regime.option_type,
                    predicted_multiplier=result.reconstructed_multiplier_grid,
                    reference_multiplier=reference_multiplier,
                )
                projected_metrics = score_value_trajectory(
                    result.projected_value_grid,
                    full.value_grid,
                    full.payoff,
                    full.spot_grid,
                    full.tau_grid,
                    regime.option_type,
                )
                raw_high = score_value_trajectory(
                    result.raw_value_grid,
                    high_reference,
                    full.payoff,
                    full.spot_grid,
                    full.tau_grid,
                    regime.option_type,
                )
                projected_high = score_value_trajectory(
                    result.projected_value_grid,
                    high_reference,
                    full.payoff,
                    full.spot_grid,
                    full.tau_grid,
                    regime.option_type,
                )
                base.update({f"raw_reduction_{key}": value for key, value in raw_metrics.items()})
                base.update({f"projected_reduction_{key}": value for key, value in projected_metrics.items()})
                base.update({f"raw_high_{key}": value for key, value in raw_high.items()})
                base.update({f"projected_high_{key}": value for key, value in projected_high.items()})
            rows.append(base)
    return rows


def select_validation_dimensions(rows: list[dict[str, object]]) -> dict[str, object]:
    decision: dict[str, object] = {
        "protocol_hash": protocol_hash(),
        "reduction_rmse_limit": REDUCTION_RMSE_LIMIT,
        "families": {},
    }
    for family in ("put", "call"):
        family_decisions = []
        for dimension in DIMENSION_LADDER:
            selected = [
                row
                for row in rows
                if row["option_type"] == family and int(row["requested_dimension"]) == dimension
            ]
            converged = bool(selected) and all(bool(row["converged"]) for row in selected)
            worst_rmse = (
                max(float(row.get("projected_reduction_price_rmse", float("inf"))) for row in selected)
                if selected
                else float("inf")
            )
            worst_boundary = (
                max(
                    float(row.get("projected_reduction_boundary_conditional_mae", float("inf")))
                    for row in selected
                    if np.isfinite(float(row.get("projected_reduction_boundary_conditional_mae", float("nan"))))
                )
                if any(
                    np.isfinite(float(row.get("projected_reduction_boundary_conditional_mae", float("nan"))))
                    for row in selected
                )
                else float("inf")
            )
            raw_lcp_median = (
                float(np.median([float(row["raw_full_lcp_residual"]) for row in selected]))
                if selected
                else float("inf")
            )
            active_values = [
                float(row.get("raw_reduction_active_set_f1", float("nan")))
                for row in selected
            ]
            active_values = [value for value in active_values if np.isfinite(value)]
            active_f1_median = (
                float(np.median(active_values)) if active_values else float("nan")
            )
            prior = family_decisions[-1] if family_decisions else None
            nonworsening = (
                prior is None
                or raw_lcp_median <= 1.10 * max(float(prior["median_raw_full_lcp_residual"]), 1e-15)
            ) and (
                prior is None
                or not np.isfinite(active_f1_median)
                or not np.isfinite(float(prior["median_active_set_f1"]))
                or active_f1_median >= float(prior["median_active_set_f1"]) - 0.02
            )
            passes = (
                converged
                and worst_rmse <= REDUCTION_RMSE_LIMIT
                and worst_boundary <= 4.0 / 120.0
                and all(float(row["projected_obstacle_violation"]) <= 1e-12 for row in selected)
                and all(
                    float(row.get("projected_high_price_rmse", float("inf")))
                    <= max(0.002474946, 1.25 * float(row["cn_high_price_rmse"]))
                    for row in selected
                )
                and all(
                    float(row.get("projected_high_delta_rmse", float("inf")))
                    <= 1.1 * max(float(row["cn_high_delta_rmse"]), 1e-15)
                    for row in selected
                )
                and all(
                    float(row.get("projected_high_stable_gamma_rmse", float("inf")))
                    <= 1.1 * max(float(row["cn_high_stable_gamma_rmse"]), 1e-15)
                    for row in selected
                )
                and nonworsening
            )
            family_decisions.append(
                {
                    "dimension": dimension,
                    "passes": passes,
                    "all_converged": converged,
                    "worst_surface_rmse": worst_rmse,
                    "worst_boundary_mae": worst_boundary,
                    "median_raw_full_lcp_residual": raw_lcp_median,
                    "median_active_set_f1": active_f1_median,
                    "raw_structure_nonworsening": nonworsening,
                    "median_online_seconds": (
                        float(np.median([float(row["timing_total_seconds"]) for row in selected]))
                        if selected
                        else float("inf")
                    ),
                }
            )
        passing = [item for item in family_decisions if item["passes"]]
        chosen = min(passing, key=lambda item: (item["dimension"], item["median_online_seconds"])) if passing else None
        calibration = None
        if chosen:
            chosen_rows = [
                row
                for row in rows
                if row["option_type"] == family
                and int(row["requested_dimension"]) == int(chosen["dimension"])
            ]
            calibration = max(
                float(row["projected_reduction_price_rmse"])
                / max(float(row["projected_full_lcp_residual"]), 1e-15)
                for row in chosen_rows
            )
        decision["families"][family] = {
            "status": "GO_VALIDATION" if chosen else "STOP_VALIDATION",
            "selected_dimension": chosen["dimension"] if chosen else None,
            "estimator_calibration_factor": calibration,
            "ladder": family_decisions,
        }
    statuses = [decision["families"][family]["status"] for family in ("put", "call")]
    decision["status"] = (
        "GO_VALIDATION" if statuses.count("GO_VALIDATION") == 2 else
        "PARTIAL_GO" if statuses.count("GO_VALIDATION") == 1 else
        "STOP_ACCURACY"
    )
    return decision


def freeze_validation_decision(decision: dict[str, object]) -> Path:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    destination = VALIDATION_DIR / "frozen_rb_config.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _decision_identity(existing) != _decision_identity(decision):
            raise RuntimeError("a different RB validation configuration is already frozen")
        return destination
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def create_scoring_marker(
    frozen_config: Path | str, *, output_dir: Path | str = HELDOUT_DIR
) -> Path:
    """Permanently mark that held-out labels have been opened."""

    path = Path(frozen_config)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    marker_root = Path(output_dir)
    marker_root.mkdir(parents=True, exist_ok=True)
    marker = marker_root / "SCORING_STARTED.json"
    payload = {"frozen_config_sha256": digest, "protocol_hash": protocol_hash()}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("held-out scoring marker does not match frozen configuration")
    if not marker.exists():
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, marker)
    return marker


def run_heldout_predictions(*, resume: bool = True) -> Path:
    """Label-blind entry point: solve frozen RB models and write predictions only."""

    frozen_path = VALIDATION_DIR / "frozen_rb_config.json"
    decision = json.loads(frozen_path.read_text(encoding="utf-8"))
    predictions_dir = HELDOUT_DIR / "predictions"
    rows: list[dict[str, object]] = []
    for family in ("put", "call"):
        dimension = decision["families"][family]["selected_dimension"]
        if dimension is None:
            continue
        basis_path = BASIS_DIR / family / f"basis_{int(dimension):02d}.npz"
        artifact = assemble_affine_rb_operator(load_basis(basis_path))
        for regime in load_regimes(splits=("test", "stress_holdout"), option_type=family):
            destination = predictions_dir / family / f"{regime.regime_id}.npz"
            if resume and destination.exists():
                status = "REUSED"
                with np.load(destination, allow_pickle=False) as data:
                    converged = bool(data["converged"])
                    total_seconds = float(data["total_seconds"])
                    failure_reason = str(data["failure_reason"])
            else:
                result = solve_reduced_american_vi(artifact, regime.config())
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        regime_id=regime.regime_id,
                        option_type=family,
                        split=regime.split,
                        requested_dimension=dimension,
                        raw_value_grid=result.raw_value_grid,
                        projected_value_grid=result.projected_value_grid,
                        multiplier_grid=result.reconstructed_multiplier_grid,
                        converged=result.converged,
                        reduced_residual_max=result.reduced_residual_max,
                        full_lcp_residual_max=result.full_lcp_residual_max,
                        raw_full_lcp_residual=result.raw_audit["normalized_lcp_residual_max"],
                        raw_obstacle_violation=result.raw_audit["normalized_obstacle_violation_max"],
                        projected_obstacle_violation=result.projected_audit["normalized_obstacle_violation_max"],
                        total_seconds=result.timing["total_seconds"],
                        failure_reason=result.failure_reason or "",
                        protocol_hash=protocol_hash(),
                    )
                os.replace(temporary, destination)
                status = "GENERATED"
                converged = result.converged
                total_seconds = result.timing["total_seconds"]
                failure_reason = result.failure_reason or ""
            rows.append(
                {
                    "regime_id": regime.regime_id,
                    "option_type": family,
                    "split": regime.split,
                    "dimension": dimension,
                    "status": status,
                    "converged": converged,
                    "total_seconds": total_seconds,
                    "failure_reason": failure_reason,
                    "path": str(destination.relative_to(RESULTS_DIR)),
                }
            )
    if not rows:
        raise RuntimeError("no option family passed validation; held-out execution is forbidden")
    manifest = HELDOUT_DIR / "prediction_manifest.csv"
    _write_csv(manifest, rows)
    return manifest


def score_heldout_predictions(
    *,
    reference_m: int = 480,
    reference_n: int = 960,
    runtime_repeats: int = 30,
    warmups: int = 5,
) -> tuple[Path, Path]:
    """Open references only after every frozen prediction exists, then score once."""

    frozen_path = VALIDATION_DIR / "frozen_rb_config.json"
    decision = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected: list[tuple[RBRegime, int, Path]] = []
    for family in ("put", "call"):
        dimension = decision["families"][family]["selected_dimension"]
        if dimension is None:
            continue
        for regime in load_regimes(splits=("test", "stress_holdout"), option_type=family):
            path = HELDOUT_DIR / "predictions" / family / f"{regime.regime_id}.npz"
            if not path.exists():
                raise RuntimeError(f"missing held-out prediction {path}")
            expected.append((regime, int(dimension), path))
    if not expected:
        raise RuntimeError("no option family passed validation; held-out scoring is forbidden")
    create_scoring_marker(frozen_path)
    metric_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260812)
    artifacts: dict[tuple[str, int], object] = {}
    for regime, dimension, prediction_path in expected:
        config = regime.config()
        with np.load(prediction_path, allow_pickle=False) as data:
            raw = data["raw_value_grid"].copy()
            projected = data["projected_value_grid"].copy()
            multiplier = data["multiplier_grid"].copy()
            rb_full_lcp_residual = float(data["full_lcp_residual_max"])
            rb_raw_lcp_residual = float(data["raw_full_lcp_residual"])
            rb_raw_obstacle = float(data["raw_obstacle_violation"])
            rb_projected_obstacle = float(data["projected_obstacle_violation"])
            if not bool(data["converged"]):
                metric_rows.append(
                    {
                        "regime_id": regime.regime_id,
                        "option_type": regime.option_type,
                        "split": regime.split,
                        "method": "RB_VI",
                        "converged": False,
                    }
                )
                continue
        full = american_cn_lcp_price(config, lcp_solver="policy_iteration")
        psor = american_cn_lcp_price(config, lcp_solver="psor")
        reference_multiplier, _, _ = trajectory_multipliers(config, full.value_grid)
        fine_config = type(config)(
            config.option_type,
            config.K,
            config.T,
            config.r,
            config.q,
            config.sigma,
            config.Smax,
            int(reference_m),
            int(reference_n),
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        fine_spots = sinh_spot_grid(config.Smax, config.K, int(reference_m))
        fine = american_dirk_policy_price(
            fine_config,
            theta=1.0 - np.sqrt(2.0) / 2.0,
            quadratic_time=True,
            damping_steps=2,
            spot_grid=fine_spots,
        )
        high_reference = interpolate_reference_surface(
            fine.value_grid,
            fine.spot_grid,
            fine.tau_grid,
            full.spot_grid,
            full.tau_grid,
        )
        for method, values, predicted_lambda in (
            ("CN_PSOR", psor.value_grid, trajectory_multipliers(config, psor.value_grid)[0]),
            ("CN_POLICY", full.value_grid, reference_multiplier),
            ("RB_VI_RAW", raw, multiplier),
            ("RB_VI_PROJECTED", projected, multiplier),
        ):
            metrics = score_value_trajectory(
                values,
                high_reference,
                full.payoff,
                full.spot_grid,
                full.tau_grid,
                regime.option_type,
                predicted_multiplier=predicted_lambda,
                reference_multiplier=reference_multiplier,
            )
            reduction_rmse = float(np.sqrt(np.mean((values - full.value_grid) ** 2)))
            calibration = decision["families"][regime.option_type].get(
                "estimator_calibration_factor"
            )
            estimator = (
                float(calibration) * rb_full_lcp_residual
                if method == "RB_VI_PROJECTED" and calibration is not None
                else float("nan")
            )
            metric_rows.append(
                {
                    "regime_id": regime.regime_id,
                    "option_type": regime.option_type,
                    "split": regime.split,
                    "method": method,
                    "dimension": dimension if method.startswith("RB") else 0,
                    "converged": True,
                    "full_lcp_residual": (
                        rb_raw_lcp_residual if method == "RB_VI_RAW" else
                        rb_full_lcp_residual if method == "RB_VI_PROJECTED" else 0.0
                    ),
                    "obstacle_violation": (
                        rb_raw_obstacle if method == "RB_VI_RAW" else
                        rb_projected_obstacle if method == "RB_VI_PROJECTED" else 0.0
                    ),
                    "reduction_price_rmse": reduction_rmse,
                    "estimated_reduction_error": estimator,
                    "estimator_covers_reduction_error": (
                        bool(reduction_rmse <= estimator)
                        if np.isfinite(estimator)
                        else ""
                    ),
                    "estimator_effectivity": (
                        estimator / max(reduction_rmse, 1e-15)
                        if np.isfinite(estimator)
                        else float("nan")
                    ),
                    **metrics,
                }
            )
        key = (regime.option_type, dimension)
        if key not in artifacts:
            load_started = perf_counter()
            artifacts[key] = assemble_affine_rb_operator(
                load_basis(BASIS_DIR / regime.option_type / f"basis_{dimension:02d}.npz")
            )
            runtime_rows.append(
                {
                    "regime_id": "ARTIFACT",
                    "option_type": regime.option_type,
                    "split": "cold_start",
                    "repeat": 0,
                    "method": "RB_ARTIFACT_LOAD",
                    "elapsed_seconds": perf_counter() - load_started,
                    "full_lcp_residual": 0.0,
                }
            )
        artifact = artifacts[key]
        for _ in range(int(warmups)):
            american_cn_lcp_price(config, lcp_solver="psor")
            american_cn_lcp_price(config, lcp_solver="policy_iteration")
            solve_reduced_american_vi(artifact, config)
        for repeat in range(int(runtime_repeats)):
            order = ["CN_PSOR", "CN_POLICY", "RB_VI"]
            rng.shuffle(order)
            for method in order:
                started = perf_counter()
                if method in {"CN_PSOR", "CN_POLICY"}:
                    timed = american_cn_lcp_price(
                        config,
                        lcp_solver="psor" if method == "CN_PSOR" else "policy_iteration",
                    )
                    elapsed = perf_counter() - started
                    residual = max(
                        result.residual.normalized_lcp_residual for result in timed.lcp_results
                    )
                else:
                    timed = solve_reduced_american_vi(artifact, config)
                    elapsed = perf_counter() - started
                    residual = timed.full_lcp_residual_max
                runtime_rows.append(
                    {
                        "regime_id": regime.regime_id,
                        "option_type": regime.option_type,
                        "split": regime.split,
                        "repeat": repeat,
                        "method": method,
                        "elapsed_seconds": elapsed,
                        "full_lcp_residual": residual,
                    }
                )
    metrics_path = HELDOUT_DIR / "heldout_metrics.csv"
    runtime_path = HELDOUT_DIR / "runtime_samples.csv"
    _write_csv(metrics_path, metric_rows)
    _write_csv(runtime_path, runtime_rows)
    return metrics_path, runtime_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _decision_identity(decision: dict[str, object]) -> dict[str, object]:
    """Remove volatile timing fields before comparing frozen model choices."""

    copied = json.loads(json.dumps(decision))
    for family in copied.get("families", {}).values():
        for row in family.get("ladder", []):
            row.pop("median_online_seconds", None)
    return copied
