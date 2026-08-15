"""Leakage-safe orchestration for Experiments 46--51."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.basis_operator.basis import (
    fit_full_grid_premium_basis,
    load_premium_basis,
    premium_vector_from_value,
    project_premium_coefficients,
    save_premium_basis,
)
from american_risk_surfaces.basis_operator.evaluation import audit_basis_operator_surface
from american_risk_surfaces.basis_operator.prediction import reconstruct_full_prediction
from american_risk_surfaces.basis_operator.model import (
    infer_coefficients,
    load_basis_operator_artifact,
    regime_features,
    train_basis_coefficient_operator,
)
from american_risk_surfaces.basis_operator.protocol import (
    DIMENSION_LADDER,
    REDUCTION_RMSE_LIMIT,
    RESULTS_DIR,
    TOTAL_RMSE_FLOOR,
    assert_mapping_regime_allowed,
    load_regimes,
    protocol_hash,
    train_snapshot_paths,
)
from american_risk_surfaces.basis_operator.types import BasisOperatorTrainingConfig
from american_risk_surfaces.reduced_order.metrics import (
    interpolate_reference_surface,
    score_value_trajectory,
)
from american_risk_surfaces.reduced_order.protocol import RBRegime
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid


BASIS_DIR = RESULTS_DIR / "01_pod_basis"
VALIDATION_CACHE_DIR = RESULTS_DIR / "02_validation_cache"
REPRESENTATION_DIR = RESULTS_DIR / "03_representation_ceiling"
MAPPING_DIR = RESULTS_DIR / "04_mapping_development"
FIVE_SEED_DIR = RESULTS_DIR / "05_five_seed_validation"
HELDOUT_DIR = RESULTS_DIR / "06_heldout"


def write_csv(path: Path | str, rows: list[dict[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def build_pod_basis_ladder(
    dimensions: Iterable[int] = DIMENSION_LADDER,
) -> tuple[list[Path], list[dict[str, object]]]:
    dimensions = tuple(sorted(set(map(int, dimensions))))
    paths: list[Path] = []
    rows: list[dict[str, object]] = []
    for family in ("put", "call"):
        started = perf_counter()
        maximum = fit_full_grid_premium_basis(
            train_snapshot_paths(family), family, max(dimensions)
        )
        total_energy = max(float(np.sum(maximum.singular_values**2)), 1e-30)
        for dimension in dimensions:
            basis = type(maximum)(
                maximum.option_type,
                maximum.mean_premium.copy(),
                maximum.components[:dimension].copy(),
                maximum.singular_values.copy(),
                maximum.positive_tau_grid.copy(),
                maximum.interior_moneyness_grid.copy(),
                maximum.train_regime_ids,
                {
                    **maximum.metadata,
                    "requested_modes": dimension,
                    "retained_energy": float(
                        np.sum(maximum.singular_values[:dimension] ** 2) / total_energy
                    ),
                },
            )
            path = BASIS_DIR / family / f"premium_basis_{dimension:02d}.npz"
            save_premium_basis(basis, path)
            paths.append(path)
            rows.append({
                "option_type": family,
                "modes": dimension,
                "retained_energy": basis.metadata["retained_energy"],
                "singular_value": float(basis.singular_values[dimension - 1]),
                "artifact_bytes": path.stat().st_size,
                "family_svd_seconds": perf_counter() - started,
                "artifact_path": str(path.relative_to(RESULTS_DIR)),
                "protocol_hash": protocol_hash(),
            })
    write_csv(BASIS_DIR / "basis_manifest.csv", rows)
    return paths, rows


def validation_reference_bundle(
    regime: RBRegime,
    *,
    reference_m: int = 480,
    reference_n: int = 960,
) -> dict[str, np.ndarray | float]:
    assert_mapping_regime_allowed(regime.split, regime.option_type, regime.q)
    if regime.split != "validation":
        raise PermissionError("this cache is validation-only")
    path = VALIDATION_CACHE_DIR / f"{regime.regime_id}_M{reference_m}_N{reference_n}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() if data[key].shape else float(data[key]) for key in data.files}
    config = regime.config()
    started = perf_counter()
    cn = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    cn_seconds = perf_counter() - started
    fine_config = type(config)(
        config.option_type, config.K, config.T, config.r, config.q, config.sigma,
        config.Smax, int(reference_m), int(reference_n),
        tolerance=1e-12, obstacle_tolerance=1e-12,
    )
    started = perf_counter()
    fine = american_dirk_policy_price(
        fine_config, quadratic_time=True, damping_steps=2,
        spot_grid=sinh_spot_grid(config.Smax, config.K, int(reference_m)),
    )
    high = interpolate_reference_surface(
        fine.value_grid, fine.spot_grid, fine.tau_grid,
        cn.spot_grid, cn.tau_grid,
    )
    reference_seconds = perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        value_grid=cn.value_grid,
        high_reference=high,
        payoff=cn.payoff,
        spot_grid=cn.spot_grid,
        tau_grid=cn.tau_grid,
        cn_seconds=cn_seconds,
        high_reference_seconds=reference_seconds,
    )
    return {
        "value_grid": cn.value_grid, "high_reference": high,
        "payoff": cn.payoff, "spot_grid": cn.spot_grid, "tau_grid": cn.tau_grid,
        "cn_seconds": cn_seconds, "high_reference_seconds": reference_seconds,
    }


def evaluate_representation_ceiling(
    *,
    dimensions: Iterable[int] = DIMENSION_LADDER,
    reference_m: int = 480,
    reference_n: int = 960,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for regime in load_regimes(splits=("validation",)):
        if regime.option_type == "call" and regime.q <= 0.0:
            continue
        bundle = validation_reference_bundle(
            regime, reference_m=reference_m, reference_n=reference_n
        )
        config = regime.config()
        value_grid = np.asarray(bundle["value_grid"])
        vector = premium_vector_from_value(value_grid, np.asarray(bundle["payoff"]))
        cn_high = score_value_trajectory(
            value_grid, np.asarray(bundle["high_reference"]), np.asarray(bundle["payoff"]),
            np.asarray(bundle["spot_grid"]), np.asarray(bundle["tau_grid"]), regime.option_type,
        )
        for dimension in dimensions:
            basis = load_premium_basis(
                BASIS_DIR / regime.option_type / f"premium_basis_{int(dimension):02d}.npz"
            )
            coefficients = project_premium_coefficients(basis, vector)
            for projection in ("raw", "hard", "softplus"):
                prediction = reconstruct_full_prediction(
                    basis, coefficients, config, projection=projection
                )
                reduction = audit_basis_operator_surface(
                    prediction, config, reference_value_grid=value_grid, prefix="reduction"
                )
                high = audit_basis_operator_surface(
                    prediction, config, reference_value_grid=np.asarray(bundle["high_reference"]),
                    prefix="high",
                )
                rows.append({
                    "regime_id": regime.regime_id,
                    "option_type": regime.option_type,
                    "split": regime.split,
                    "arm": "O",
                    "modes": int(dimension),
                    "projection": projection,
                    "cn_high_price_rmse": cn_high["price_rmse"],
                    "cn_high_delta_rmse": cn_high["delta_rmse"],
                    "cn_high_stable_gamma_rmse": cn_high["stable_gamma_rmse"],
                    "cn_generation_seconds": bundle["cn_seconds"],
                    "high_reference_generation_seconds": bundle["high_reference_seconds"],
                    **reduction,
                    **high,
                })
    write_csv(REPRESENTATION_DIR / "oracle_ceiling_metrics.csv", rows)
    return rows


def select_representation_modes(rows: list[dict[str, object]]) -> dict[str, object]:
    decision: dict[str, object] = {
        "protocol_hash": protocol_hash(),
        "status": "COMPLETE",
        "reduction_rmse_limit": REDUCTION_RMSE_LIMIT,
        "families": {},
    }
    for family in ("put", "call"):
        ladder = []
        for dimension in DIMENSION_LADDER:
            selected = [
                row for row in rows
                if row["option_type"] == family and int(row["modes"]) == dimension
                and row["projection"] == "hard"
            ]
            finite = bool(selected) and all(
                np.isfinite(float(row["reduction_price_rmse"])) for row in selected
            )
            worst_rmse = max(
                (float(row["reduction_price_rmse"]) for row in selected), default=float("inf")
            )
            worst_obstacle = max(
                (float(row["projected_obstacle_violation"]) for row in selected), default=float("inf")
            )
            passes = finite and worst_rmse <= REDUCTION_RMSE_LIMIT and worst_obstacle <= 1e-12
            ladder.append({
                "modes": dimension,
                "passes_mapping_entry": passes,
                "worst_reduction_rmse": worst_rmse,
                "worst_projected_obstacle": worst_obstacle,
            })
        passing = [item["modes"] for item in ladder if item["passes_mapping_entry"]]
        decision["families"][family] = {
            "ladder": ladder,
            "passing_modes": passing,
            "status": "PROCEED_MAPPING" if passing else "STOP_REPRESENTATION",
        }
    if not any(item["passing_modes"] for item in decision["families"].values()):
        decision["status"] = "STOP_REPRESENTATION"
    path = REPRESENTATION_DIR / "representation_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return decision


def _validation_metric_row(
    regime: RBRegime,
    prediction,
    *,
    arm: str,
    modes: int,
    seed: int,
    oracle_lcp_p95: float,
) -> dict[str, object]:
    bundle = validation_reference_bundle(regime)
    config = regime.config()
    reduction = audit_basis_operator_surface(
        prediction, config, reference_value_grid=np.asarray(bundle["value_grid"]),
        prefix="reduction",
    )
    high = audit_basis_operator_surface(
        prediction, config, reference_value_grid=np.asarray(bundle["high_reference"]),
        prefix="high",
    )
    cn_high = score_value_trajectory(
        np.asarray(bundle["value_grid"]), np.asarray(bundle["high_reference"]),
        np.asarray(bundle["payoff"]), np.asarray(bundle["spot_grid"]),
        np.asarray(bundle["tau_grid"]), regime.option_type,
    )
    return {
        "regime_id": regime.regime_id,
        "option_type": regime.option_type,
        "split": regime.split,
        "arm": arm,
        "modes": modes,
        "seed": seed,
        "oracle_lcp_p95": oracle_lcp_p95,
        "cn_high_price_rmse": cn_high["price_rmse"],
        "cn_high_delta_rmse": cn_high["delta_rmse"],
        "cn_high_stable_gamma_rmse": cn_high["stable_gamma_rmse"],
        **prediction.timing,
        **reduction,
        **high,
    }


def _oracle_lcp_lookup() -> dict[tuple[str, int], float]:
    path = REPRESENTATION_DIR / "oracle_ceiling_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["regime_id"], int(row["modes"])): float(row["normalized_full_lcp_residual_p95"])
        for row in rows if row["projection"] == "hard"
    }


def _features_for_regime(regime: RBRegime) -> np.ndarray:
    import math
    return np.asarray([[math.log(regime.T), regime.sigma, regime.r, regime.q]], dtype=float)


def _fit_polynomial_ridge(
    basis, family: str, *, alpha: float = 1e-6
) -> dict[str, object]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    paths = train_snapshot_paths(family)
    matrix, identifiers, _tau, _spots = __import__(
        "american_risk_surfaces.basis_operator.basis", fromlist=["load_training_matrix"]
    ).load_training_matrix(paths, family)
    features = regime_features(paths, family)
    scaler = StandardScaler().fit(features)
    polynomial = PolynomialFeatures(degree=3, include_bias=False).fit(scaler.transform(features))
    expanded = polynomial.transform(scaler.transform(features))
    coefficients = np.einsum(
        "ni,mi->nm", matrix - basis.mean_premium, basis.components, optimize=False
    )
    model = Ridge(alpha=alpha).fit(expanded, coefficients)
    return {
        "input_mean": scaler.mean_,
        "input_scale": scaler.scale_,
        "powers": polynomial.powers_,
        "ridge_coef": model.coef_,
        "ridge_intercept": model.intercept_,
        "train_regime_ids": identifiers,
    }


def _polynomial_predict(payload: dict[str, object], features: np.ndarray) -> np.ndarray:
    scaled = (features - payload["input_mean"]) / payload["input_scale"]
    powers = np.asarray(payload["powers"], dtype=int)
    expanded = np.prod(scaled[:, None, :] ** powers[None, :, :], axis=2)
    return np.einsum(
        "nf,mf->nm", expanded, np.asarray(payload["ridge_coef"]), optimize=False
    ) + np.asarray(payload["ridge_intercept"])


def evaluate_checkpoint_on_validation(
    checkpoint_path: Path | str,
    *,
    arm: str,
    modes: int,
    seed: int,
) -> list[dict[str, object]]:
    artifact = load_basis_operator_artifact(checkpoint_path)
    oracle = _oracle_lcp_lookup()
    rows = []
    for regime in load_regimes(splits=("validation",), option_type=artifact.basis.option_type):
        if regime.option_type == "call" and regime.q <= 0.0:
            continue
        started = perf_counter()
        coefficients = infer_coefficients(artifact, _features_for_regime(regime))[0]
        inference_seconds = perf_counter() - started
        prediction = reconstruct_full_prediction(
            artifact.basis, coefficients, regime.config(), projection="hard",
            coefficient_seconds=inference_seconds,
        )
        rows.append(_validation_metric_row(
            regime, prediction, arm=arm, modes=modes, seed=seed,
            oracle_lcp_p95=oracle[(regime.regime_id, modes)],
        ))
        if arm == "P2":
            soft = reconstruct_full_prediction(
                artifact.basis, coefficients, regime.config(), projection="softplus",
                coefficient_seconds=inference_seconds,
            )
            rows.append(_validation_metric_row(
                regime, soft, arm="P2-S", modes=modes, seed=seed,
                oracle_lcp_p95=oracle[(regime.regime_id, modes)],
            ))
    return rows


def run_mapping_development(
    *, steps: int = 4000, families: Iterable[str] = ("put", "call"),
    requested_modes: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    decision = json.loads(
        (REPRESENTATION_DIR / "representation_decision.json").read_text(encoding="utf-8")
    )
    oracle = _oracle_lcp_lookup()
    rows: list[dict[str, object]] = []
    for family in families:
        modes_to_run = decision["families"][family]["passing_modes"]
        if requested_modes is not None:
            allowed_modes = set(map(int, requested_modes))
            modes_to_run = [item for item in modes_to_run if item in allowed_modes]
        for modes in modes_to_run:
            basis_path = BASIS_DIR / family / f"premium_basis_{modes:02d}.npz"
            basis = load_premium_basis(basis_path)
            ridge_dir = MAPPING_DIR / family / f"modes_{modes:02d}" / "P0_seed_17"
            ridge_dir.mkdir(parents=True, exist_ok=True)
            ridge_path = ridge_dir / "ridge_artifact.npz"
            if ridge_path.exists():
                with np.load(ridge_path, allow_pickle=False) as data:
                    ridge = {key: data[key].copy() for key in data.files}
            else:
                ridge = _fit_polynomial_ridge(basis, family)
                np.savez_compressed(ridge_path, **ridge)
            for regime in load_regimes(splits=("validation",), option_type=family):
                coefficients = _polynomial_predict(ridge, _features_for_regime(regime))[0]
                prediction = reconstruct_full_prediction(basis, coefficients, regime.config())
                rows.append(_validation_metric_row(
                    regime, prediction, arm="P0", modes=modes, seed=17,
                    oracle_lcp_p95=oracle[(regime.regime_id, modes)],
                ))
            for arm, loss_variant in (("P1", "coefficient"), ("P2", "structure_aware")):
                output = MAPPING_DIR / family / f"modes_{modes:02d}" / f"{arm}_seed_17"
                config = BasisOperatorTrainingConfig(
                    family, modes, loss_variant, 17, steps, 16, 1e-3, "float64", 500
                )
                history_path = output / "training_history.csv"
                checkpoint_path = output / "checkpoint.pt"
                completed = False
                if history_path.exists() and checkpoint_path.exists():
                    with history_path.open(newline="", encoding="utf-8") as handle:
                        history_rows = list(csv.DictReader(handle))
                    completed = bool(history_rows) and int(history_rows[-1]["step"]) == steps
                result = (
                    type("ReusedResult", (), {
                        "status": "COMPLETE", "artifact_path": checkpoint_path,
                        "failure_reason": None,
                    })()
                    if completed else
                    train_basis_coefficient_operator(
                        basis, train_snapshot_paths(family), config,
                        output_dir=output, basis_path=basis_path,
                    )
                )
                if result.status == "COMPLETE":
                    rows.extend(evaluate_checkpoint_on_validation(
                        result.artifact_path, arm=arm, modes=modes, seed=17
                    ))
                else:
                    rows.append({
                        "regime_id": "TRAINING_FAILURE", "option_type": family,
                        "split": "validation", "arm": arm, "modes": modes,
                        "seed": 17, "failure_reason": result.failure_reason,
                    })
            write_csv(
                MAPPING_DIR / family / f"modes_{modes:02d}" / "validation_metrics.csv",
                [row for row in rows if row["option_type"] == family and int(row["modes"]) == modes],
            )
    metric_paths = sorted(MAPPING_DIR.glob("*/modes_*/validation_metrics.csv"))
    combined: list[dict[str, object]] = []
    for path in metric_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            combined.extend(csv.DictReader(handle))
    write_csv(MAPPING_DIR / "mapping_development_metrics.csv", combined)
    return combined


def configuration_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: list[dict[str, object]] = []
    for family in ("put", "call"):
        candidates = sorted({
            (str(row["arm"]), int(row["modes"])) for row in rows
            if row["option_type"] == family and row["arm"] in {"P1", "P2"}
        })
        for arm, modes in candidates:
            selected = [
                row for row in rows
                if row["option_type"] == family and row["arm"] == arm
                and int(row["modes"]) == modes and row.get("regime_id") != "TRAINING_FAILURE"
            ]
            gate = _rows_pass_approximate_gate(selected, multi_seed=False)
            grouped.append({
                "option_type": family, "arm": arm, "modes": modes,
                **gate,
            })
    decision: dict[str, object] = {
        "protocol_hash": protocol_hash(), "families": {},
    }
    for family in ("put", "call"):
        candidates = [item for item in grouped if item["option_type"] == family]
        if not candidates:
            decision["families"][family] = {"status": "NOT_RUN"}
            continue
        passing = [item for item in candidates if item["passes"]]
        if passing:
            mode = min(item["modes"] for item in passing)
            same_mode = {item["arm"]: item for item in passing if item["modes"] == mode}
            chosen = same_mode.get("P2", same_mode.get("P1"))
            same_mode = {item["arm"]: item for item in candidates if item["modes"] == chosen["modes"]}
            if "P1" in same_mode and "P2" in same_mode:
                p1, p2 = same_mode["P1"], same_mode["P2"]
                p1_can_replace = (
                    p1["worst_reduction_rmse"] <= 0.8 * p2["worst_reduction_rmse"]
                    and p1["max_gate_ratio_excluding_price"] <= 1.05 * p2["max_gate_ratio_excluding_price"]
                )
                chosen = p1 if p1_can_replace else p2
        else:
            p2_candidates = [item for item in candidates if item["arm"] == "P2"]
            chosen = min(p2_candidates or candidates, key=lambda item: item["max_gate_ratio"])
            p1_same_mode = next(
                (item for item in candidates if item["arm"] == "P1" and item["modes"] == chosen["modes"]),
                None,
            )
            if p1_same_mode is not None:
                p1_can_replace = (
                    p1_same_mode["worst_reduction_rmse"] <= 0.8 * chosen["worst_reduction_rmse"]
                    and p1_same_mode["max_gate_ratio_excluding_price"]
                    <= 1.05 * chosen["max_gate_ratio_excluding_price"]
                )
                if p1_can_replace:
                    chosen = p1_same_mode
        decision["families"][family] = {
            "status": "DEVELOPMENT_PASS" if chosen["passes"] else "DEVELOPMENT_FAIL_CONFIRM",
            "selected_arm": chosen["arm"],
            "selected_modes": chosen["modes"],
            "metrics": chosen,
            "all_candidates": candidates,
        }
    path = MAPPING_DIR / "mapping_development_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return decision


def _rows_pass_approximate_gate(
    rows: list[dict[str, object]], *, multi_seed: bool
) -> dict[str, object]:
    if not rows:
        return {"passes": False, "max_gate_ratio": float("inf"), "reason": "no rows"}
    by_regime: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_regime.setdefault(str(row["regime_id"]), []).append(row)
    reductions = []
    ratios_other = []
    failure = []
    for regime_id, regime_rows in by_regime.items():
        def median(name: str) -> float:
            return float(np.median([float(item[name]) for item in regime_rows]))
        price = median("reduction_price_rmse")
        high_price = median("high_price_rmse")
        high_limit = max(TOTAL_RMSE_FLOOR, 1.25 * median("cn_high_price_rmse"))
        boundary_values = [
            float(item["reduction_boundary_conditional_mae"]) for item in regime_rows
            if np.isfinite(float(item["reduction_boundary_conditional_mae"]))
        ]
        boundary = float(np.median(boundary_values)) if boundary_values else float("inf")
        delta_ratio = median("high_delta_rmse") / max(median("cn_high_delta_rmse"), 1e-15)
        gamma_ratio = median("high_stable_gamma_rmse") / max(median("cn_high_stable_gamma_rmse"), 1e-15)
        f1 = median("reduction_exercise_f1")
        lcp_ratio = median("normalized_full_lcp_residual_p95") / max(median("oracle_lcp_p95"), 1e-15)
        obstacle = median("projected_obstacle_violation")
        ratios = [
            price / REDUCTION_RMSE_LIMIT,
            high_price / high_limit,
            boundary / 0.066667,
            delta_ratio / 1.25,
            gamma_ratio / 1.25,
            0.98 / max(f1, 1e-15),
            lcp_ratio / 1.05,
            obstacle / 1e-12,
        ]
        reductions.append(price)
        ratios_other.append(max(ratios[1:]))
        if max(ratios) > 1.0:
            failure.append(regime_id)
        if multi_seed:
            for item in regime_rows:
                single_ratios = [
                    float(item["reduction_price_rmse"]) / (1.5 * REDUCTION_RMSE_LIMIT),
                    float(item["reduction_boundary_conditional_mae"]) / (1.5 * 0.066667)
                    if np.isfinite(float(item["reduction_boundary_conditional_mae"])) else float("inf"),
                    float(item["high_delta_rmse"]) / max(1.5 * 1.25 * float(item["cn_high_delta_rmse"]), 1e-15),
                    float(item["high_stable_gamma_rmse"]) / max(1.5 * 1.25 * float(item["cn_high_stable_gamma_rmse"]), 1e-15),
                ]
                if max(single_ratios) > 1.0:
                    failure.append(f"{regime_id}:seed{item['seed']}")
    all_ratios = []
    for item in ratios_other:
        all_ratios.append(item)
    max_price_ratio = max(reductions) / REDUCTION_RMSE_LIMIT
    return {
        "passes": not failure,
        "failed_regimes": sorted(set(failure)),
        "worst_reduction_rmse": max(reductions),
        "max_gate_ratio": max([max_price_ratio, *all_ratios]),
        "max_gate_ratio_excluding_price": max(all_ratios),
    }


def run_five_seed_validation(
    *, steps: int = 8000, families: Iterable[str] = ("put", "call"),
    seeds: Iterable[int] = (17, 29, 43, 71, 101),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    development = json.loads(
        (MAPPING_DIR / "mapping_development_decision.json").read_text(encoding="utf-8")
    )
    new_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for family in families:
        selection = development["families"][family]
        modes = int(selection["selected_modes"])
        arm = str(selection["selected_arm"])
        loss = "structure_aware" if arm == "P2" else "coefficient"
        basis_path = BASIS_DIR / family / f"premium_basis_{modes:02d}.npz"
        basis = load_premium_basis(basis_path)
        for seed in map(int, seeds):
            output = FIVE_SEED_DIR / family / f"modes_{modes:02d}" / f"{arm}_seed_{seed}"
            config = BasisOperatorTrainingConfig(
                family, modes, loss, seed, steps, 16, 1e-3, "float64", 500
            )
            metric_path = output / "validation_metrics.csv"
            if metric_path.exists():
                with metric_path.open(newline="", encoding="utf-8") as handle:
                    new_rows.extend(csv.DictReader(handle))
                continue
            checkpoint_path = output / "checkpoint.pt"
            checkpoint_complete = False
            if checkpoint_path.exists():
                try:
                    artifact = load_basis_operator_artifact(checkpoint_path)
                    checkpoint_complete = (
                        int(artifact.config["steps"]) == steps
                        and int(artifact.config["seed"]) == seed
                        and artifact.config["option_type"] == family
                    )
                except Exception:
                    checkpoint_complete = False
            result = (
                type("ReusedResult", (), {
                    "status": "COMPLETE", "artifact_path": checkpoint_path,
                    "failure_reason": None,
                })()
                if checkpoint_complete else
                train_basis_coefficient_operator(
                    basis, train_snapshot_paths(family), config,
                    output_dir=output, basis_path=basis_path,
                )
            )
            if result.status == "COMPLETE":
                seed_rows = [
                    row for row in evaluate_checkpoint_on_validation(
                        result.artifact_path, arm=arm, modes=modes, seed=seed
                    )
                    if row["arm"] == arm
                ]
                new_rows.extend(seed_rows)
                write_csv(output / "validation_metrics.csv", seed_rows)
            else:
                failures.append({
                    "option_type": family, "seed": seed,
                    "failure_reason": result.failure_reason,
                })
    all_rows: list[dict[str, object]] = []
    for path in sorted(FIVE_SEED_DIR.glob("*/modes_*/P*_seed_*/validation_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            all_rows.extend(csv.DictReader(handle))
    if all_rows:
        write_csv(FIVE_SEED_DIR / "five_seed_validation_metrics.csv", all_rows)
    if failures:
        write_csv(FIVE_SEED_DIR / "training_failures.csv", failures)
    decision: dict[str, object] = {
        "protocol_hash": protocol_hash(),
        "heldout_remains_sealed": True,
        "families": {},
    }
    for family in ("put", "call"):
        rows = [row for row in all_rows if row["option_type"] == family]
        family_failures = [item for item in failures if item["option_type"] == family]
        expected = 5 * len(load_regimes(splits=("validation",), option_type=family))
        gate = _rows_pass_approximate_gate(rows, multi_seed=True)
        complete = len(rows) == expected
        gate["passes"] = bool(gate["passes"] and not family_failures and complete)
        price_pass = bool(rows) and all(
            float(np.median([
                float(item["reduction_price_rmse"])
                for item in rows if item["regime_id"] == regime_id
            ])) <= REDUCTION_RMSE_LIMIT
            for regime_id in sorted({str(item["regime_id"]) for item in rows})
        )
        status = (
            "INCOMPLETE" if not complete else
            "PROCEED_HELDOUT" if gate["passes"] else
            "STOP_STRUCTURE" if price_pass else "STOP_MAPPING"
        )
        development_family = development["families"][family]
        decision["families"][family] = {
            "status": status,
            "selected_arm": development_family["selected_arm"],
            "selected_modes": development_family["selected_modes"],
            "steps": steps,
            "seeds": [17, 29, 43, 71, 101],
            "expected_rows": expected,
            "actual_rows": len(rows),
            "price_gate_passes": price_pass,
            "approximate_gate": gate,
            "training_failures": family_failures,
            "checkpoint_paths": [
                str(path.relative_to(RESULTS_DIR))
                for path in sorted((FIVE_SEED_DIR / family).glob("**/checkpoint.pt"))
            ],
        }
    passing = [
        family for family, payload in decision["families"].items()
        if payload["status"] == "PROCEED_HELDOUT"
    ]
    incomplete = any(payload["status"] == "INCOMPLETE" for payload in decision["families"].values())
    decision["status"] = (
        "INCOMPLETE" if incomplete else
        "PROCEED_HELDOUT" if len(passing) == 2 else
        "PARTIAL_PROCEED_HELDOUT" if passing else "STOP_BEFORE_HELDOUT"
    )
    decision["heldout_remains_sealed"] = incomplete or not bool(passing)
    path = FIVE_SEED_DIR / "validation_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    frozen = FIVE_SEED_DIR / "frozen_config.json"
    frozen.write_text(json.dumps({
        "protocol_hash": protocol_hash(),
        "families": {
            family: {
                "arm": payload["selected_arm"], "modes": payload["selected_modes"],
                "steps": steps, "seeds": payload["seeds"], "status": payload["status"],
            }
            for family, payload in decision["families"].items()
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return all_rows, decision
