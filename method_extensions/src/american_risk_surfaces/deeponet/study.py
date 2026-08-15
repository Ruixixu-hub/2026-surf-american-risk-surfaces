"""Leakage-safe orchestration and frozen decisions for Experiments 52--54."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.basis_operator.study import validation_reference_bundle
from american_risk_surfaces.deeponet.data import build_deeponet_training_bundle
from american_risk_surfaces.deeponet.evaluation import audit_deeponet_surface
from american_risk_surfaces.deeponet.model import (
    load_deeponet_artifact,
    train_positive_premium_deeponet,
)
from american_risk_surfaces.deeponet.prediction import (
    make_deeponet_policy_initializer,
    predict_deeponet_surface,
)
from american_risk_surfaces.deeponet.protocol import (
    ARMS,
    BASIS_VALIDATION_METRICS,
    BOUNDARY_LIMIT,
    DEVELOPMENT_STEPS,
    FORMAL_STEPS,
    LATENT_RANKS,
    LCP_TOLERANCE,
    REDUCTION_RMSE_LIMIT,
    RESULTS_DIR,
    SEEDS,
    TOTAL_RMSE_FLOOR,
    protocol_hash,
    train_snapshot_paths,
)
from american_risk_surfaces.deeponet.types import DeepONetTrainingConfig
from american_risk_surfaces.reduced_order.metrics import score_value_trajectory
from american_risk_surfaces.reduced_order.protocol import load_regimes
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price


DEVELOPMENT_DIR = RESULTS_DIR / "01_development"
FIVE_SEED_DIR = RESULTS_DIR / "02_five_seed_validation"
GRID_TRANSFER_DIR = RESULTS_DIR / "03_grid_transfer"
HELDOUT_DIR = RESULTS_DIR / "04_heldout"
RUNTIME_DIR = RESULTS_DIR / "05_runtime_hybrid"
SYNTHESIS_DIR = RESULTS_DIR / "06_synthesis"


def write_csv(path: Path | str, rows: list[dict[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def basis_operator_oracle_lcp_lookup() -> dict[str, float]:
    path = (
        RESULTS_DIR.parent / "11_positive_premium_basis_operator"
        / "03_representation_ceiling" / "oracle_ceiling_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError("frozen basis-operator oracle metrics are missing")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["regime_id"]): float(row["normalized_full_lcp_residual_p95"])
        for row in rows
        if row["projection"] == "hard" and int(row["modes"]) == 12
    }


def evaluate_checkpoint_on_validation(checkpoint_path: Path | str) -> list[dict[str, object]]:
    artifact = load_deeponet_artifact(checkpoint_path)
    family = str(artifact.config["option_type"])
    oracle = basis_operator_oracle_lcp_lookup()
    p2 = basis_operator_validation_lookup()
    rows: list[dict[str, object]] = []
    for regime in load_regimes(splits=("validation",), option_type=family):
        if family == "call" and regime.q <= 0.0:
            continue
        bundle = validation_reference_bundle(regime)
        prediction = predict_deeponet_surface(
            artifact, regime.config(), device="cpu", compute_ad_greeks=True
        )
        reduction = audit_deeponet_surface(
            prediction, regime.config(), np.asarray(bundle["value_grid"]), prefix="reduction"
        )
        high = audit_deeponet_surface(
            prediction, regime.config(), np.asarray(bundle["high_reference"]), prefix="high"
        )
        cn_high = score_value_trajectory(
            np.asarray(bundle["value_grid"]), np.asarray(bundle["high_reference"]),
            np.asarray(bundle["payoff"]), np.asarray(bundle["spot_grid"]),
            np.asarray(bundle["tau_grid"]), family,
        )
        rows.append({
            "regime_id": regime.regime_id,
            "protocol_hash": protocol_hash(),
            "option_type": family,
            "split": "validation",
            "arm": artifact.config["arm"],
            "latent_rank": int(artifact.config["latent_rank"]),
            "seed": int(artifact.config["seed"]),
            "training_steps": int(artifact.config["steps"]),
            "oracle_lcp_p95": oracle[regime.regime_id],
            "cn_high_price_rmse": cn_high["price_rmse"],
            "cn_high_delta_rmse": cn_high["delta_rmse"],
            "cn_high_stable_gamma_rmse": cn_high["stable_gamma_rmse"],
            "p2_reduction_price_rmse": p2[regime.regime_id]["reduction_price_rmse"],
            "p2_reduction_boundary_conditional_mae": p2[regime.regime_id][
                "reduction_boundary_conditional_mae"
            ],
            "p2_high_delta_rmse": p2[regime.regime_id]["high_delta_rmse"],
            "p2_high_stable_gamma_rmse": p2[regime.regime_id][
                "high_stable_gamma_rmse"
            ],
            **prediction.timing, **reduction, **high,
        })
    return rows


def basis_operator_validation_lookup() -> dict[str, dict[str, float]]:
    with BASIS_VALIDATION_METRICS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["arm"] == "P2" and int(row["modes"]) == 12:
            grouped.setdefault(row["regime_id"], []).append(row)
    keys = (
        "reduction_price_rmse", "reduction_boundary_conditional_mae",
        "high_delta_rmse", "high_stable_gamma_rmse",
    )
    return {
        regime_id: {
            key: float(np.median([float(row[key]) for row in regime_rows]))
            for key in keys
        }
        for regime_id, regime_rows in grouped.items()
    }


def run_development(
    *,
    families: Iterable[str] = ("put", "call"),
    arms: Iterable[str] = ARMS,
    ranks: Iterable[int] = LATENT_RANKS,
    steps: int = 6000,
    device: str = "auto",
    resume: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    new_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for family in families:
        bundle = build_deeponet_training_bundle(train_snapshot_paths(family), family)
        for arm in arms:
            for rank in map(int, ranks):
                output = DEVELOPMENT_DIR / family / f"{arm}_rank_{rank:03d}_seed_17"
                config = DeepONetTrainingConfig(
                    family, arm, rank, 17, int(steps), 8, 1e-3, device,
                    "float64", max(1, min(1000, int(steps))), None,
                )
                checkpoint = output / "checkpoint.pt"
                result = train_positive_premium_deeponet(
                    bundle, config, output_dir=output, resume=resume
                )
                if result.status == "COMPLETE":
                    seed_rows = evaluate_checkpoint_on_validation(checkpoint)
                    write_csv(output / "validation_metrics.csv", seed_rows)
                    new_rows.extend(seed_rows)
                else:
                    failures.append({
                        "option_type": family, "arm": arm, "latent_rank": rank,
                        "seed": 17, "status": result.status,
                        "failure_reason": result.failure_reason,
                    })
    all_rows = _collect_metrics(DEVELOPMENT_DIR)
    if all_rows:
        write_csv(DEVELOPMENT_DIR / "development_metrics.csv", all_rows)
    if failures:
        write_csv(DEVELOPMENT_DIR / "training_failures.csv", failures)
    write_checkpoint_manifest(DEVELOPMENT_DIR)
    decision = select_development_configuration(all_rows, failures)
    (DEVELOPMENT_DIR / "development_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    return all_rows, decision


def run_tiny_smoke(
    *, family: str = "put", arm: str = "N0", rank: int = 32,
    steps: int = 2, device: str = "auto",
) -> dict[str, object]:
    """Isolated smoke run that can never be collected as development evidence."""

    bundle = build_deeponet_training_bundle(train_snapshot_paths(family), family)
    output = RESULTS_DIR / "00_protocol" / "tiny_smoke" / (
        f"{family}_{arm}_rank_{int(rank):03d}"
    )
    config = DeepONetTrainingConfig(
        family, arm, int(rank), 17, int(steps), 8, 1e-3, device,
        "float64", max(1, int(steps)), None,
    )
    result = train_positive_premium_deeponet(bundle, config, output_dir=output)
    payload: dict[str, object] = {
        "status": result.status,
        "scope": "TINY_SMOKE_ONLY_NOT_FORMAL_EVIDENCE",
        "family": family,
        "arm": arm,
        "latent_rank": int(rank),
        "steps": int(steps),
        "training_seconds": result.training_seconds,
        "checkpoint_path": str(result.checkpoint_path),
        "failure_reason": result.failure_reason,
        "protocol_hash": protocol_hash(),
    }
    if result.status == "COMPLETE":
        artifact = load_deeponet_artifact(result.checkpoint_path)
        example = next(
            item for item in load_regimes(splits=("train",), option_type=family)
            if family == "put" or item.q > 0.0
        )
        prediction = predict_deeponet_surface(artifact, example.config(), device=device)
        payload.update({
            "finite": bool(np.all(np.isfinite(prediction.value_grid))),
            "minimum_premium": float(np.min(prediction.projected_premium_grid)),
            "prediction_shape": list(prediction.value_grid.shape),
        })
    (output / "smoke_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_checkpoint_manifest(RESULTS_DIR / "00_protocol" / "tiny_smoke")
    return payload


def select_development_configuration(
    rows: list[dict[str, object]], failures: list[dict[str, object]] | None = None
) -> dict[str, object]:
    failures = failures or []
    decision: dict[str, object] = {"protocol_hash": protocol_hash(), "families": {}}
    for family in ("put", "call"):
        candidates = []
        validation_count = len([
            item for item in load_regimes(splits=("validation",), option_type=family)
            if family == "put" or item.q > 0.0
        ])
        for rank in LATENT_RANKS:
            gates = {}
            for arm in ARMS:
                selected = [
                    row for row in rows if row["option_type"] == family
                    and row["arm"] == arm and int(row["latent_rank"]) == rank
                    and int(row.get("training_steps", -1)) == DEVELOPMENT_STEPS
                ]
                gate = rows_pass_approximate_gate(selected, multi_seed=False)
                configuration_complete = bool(
                    len(selected) == validation_count
                    and all(
                        int(item.get("training_steps", -1)) == DEVELOPMENT_STEPS
                        for item in selected
                    )
                )
                gate["configuration_complete"] = configuration_complete
                gate["passes"] = bool(gate["passes"] and configuration_complete)
                if any(
                    item["option_type"] == family and item["arm"] == arm
                    and int(item["latent_rank"]) == rank for item in failures
                ):
                    gate["passes"] = False
                    gate["training_failure"] = True
                gates[arm] = gate
                candidates.append({"arm": arm, "latent_rank": rank, **gate})
            n2, n1 = gates["N2"], gates["N1"]
            n2_attributable = (
                "reason" not in n2 and "reason" not in n1
                and
                n2.get("worst_reduction_rmse", float("inf"))
                <= 1.05 * n1.get("worst_reduction_rmse", float("inf"))
                and n2.get("max_gate_ratio_excluding_price", float("inf"))
                <= 0.85 * n1.get("max_gate_ratio_excluding_price", float("inf"))
            )
            for item in candidates[-3:]:
                if item["arm"] == "N2":
                    item["n2_attribution_passes"] = bool(n2_attributable)
                    item["passes"] = bool(item["passes"] and n2_attributable)
        passing = [item for item in candidates if item["passes"]]
        if passing:
            minimum_rank = min(int(item["latent_rank"]) for item in passing)
            same_rank = [item for item in passing if int(item["latent_rank"]) == minimum_rank]
            priority = {"N2": 0, "N1": 1, "N0": 2}
            chosen = min(same_rank, key=lambda item: priority[item["arm"]])
            better_price = [
                item for item in same_rank
                if item["arm"] != chosen["arm"]
                and item["worst_reduction_rmse"] <= 0.8 * chosen["worst_reduction_rmse"]
                and item["max_gate_ratio_excluding_price"]
                <= 1.05 * chosen["max_gate_ratio_excluding_price"]
            ]
            if better_price:
                chosen = min(better_price, key=lambda item: item["worst_reduction_rmse"])
            status = "DEVELOPMENT_PASS"
        else:
            best_ratio = min((item["max_gate_ratio"] for item in candidates), default=float("inf"))
            near = [item for item in candidates if item["max_gate_ratio"] <= 1.05 * best_ratio]
            priority = {"N2": 0, "N1": 1, "N0": 2}
            chosen = min(
                near, key=lambda item: (int(item["latent_rank"]), priority[item["arm"]])
            ) if near else {"arm": "N2", "latent_rank": 32, "max_gate_ratio": float("inf")}
            complete_ladder = all(
                bool(item.get("configuration_complete")) for item in candidates
            )
            status = (
                "DEVELOPMENT_FAIL_CONFIRM" if complete_ladder
                else "DEVELOPMENT_INCOMPLETE"
            )
        complete_ladder = all(
            bool(item.get("configuration_complete")) for item in candidates
        )
        decision["families"][family] = {
            "status": status,
            "complete_ladder": complete_ladder,
            "selected_arm": chosen["arm"],
            "selected_latent_rank": int(chosen["latent_rank"]),
            "metrics": chosen,
            "all_candidates": candidates,
        }
    return decision


def rows_pass_approximate_gate(
    rows: list[dict[str, object]], *, multi_seed: bool
) -> dict[str, object]:
    if not rows:
        return {
            "passes": False, "max_gate_ratio": float("inf"),
            "max_gate_ratio_excluding_price": float("inf"),
            "worst_reduction_rmse": float("inf"), "reason": "no rows",
        }
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["regime_id"]), []).append(row)
    failures, prices, other_ratios = [], [], []
    for regime_id, regime_rows in grouped.items():
        def median(name: str) -> float:
            return float(np.median([float(item[name]) for item in regime_rows]))

        price = median("reduction_price_rmse")
        high_price = median("high_price_rmse")
        high_limit = max(TOTAL_RMSE_FLOOR, 1.25 * median("cn_high_price_rmse"))
        finite_boundary = [
            float(item["reduction_boundary_conditional_mae"]) for item in regime_rows
            if np.isfinite(float(item["reduction_boundary_conditional_mae"]))
        ]
        boundary = float(np.median(finite_boundary)) if finite_boundary else float("inf")
        delta_ratio = median("high_delta_rmse") / max(median("cn_high_delta_rmse"), 1e-15)
        gamma_ratio = median("high_stable_gamma_rmse") / max(
            median("cn_high_stable_gamma_rmse"), 1e-15
        )
        f1 = median("reduction_exercise_f1")
        lcp_ratio = median("normalized_full_lcp_residual_p95") / max(
            median("oracle_lcp_p95"), 1e-15
        )
        obstacle = median("projected_obstacle_violation")
        ratios = [
            price / REDUCTION_RMSE_LIMIT,
            high_price / high_limit,
            boundary / BOUNDARY_LIMIT,
            delta_ratio / 1.25,
            gamma_ratio / 1.25,
            0.98 / max(f1, 1e-15),
            lcp_ratio / 1.05,
            obstacle / 1e-12,
        ]
        required_values = [
            price, high_price, boundary, delta_ratio, gamma_ratio, f1,
            lcp_ratio, obstacle,
        ]
        if not all(np.isfinite(value) for value in required_values):
            failures.append(f"{regime_id}:non_finite_gate_metric")
        prices.append(price)
        other_ratios.append(max(ratios[1:]))
        if max(ratios) > 1.0:
            failures.append(regime_id)
        if multi_seed:
            seeds = [int(item["seed"]) for item in regime_rows]
            if len(seeds) != 5 or len(set(seeds)) != 5:
                failures.append(f"{regime_id}:missing_or_duplicate_seed")
            failed_seeds = 0
            for item in regime_rows:
                single = [
                    float(item["reduction_price_rmse"]) / (1.5 * REDUCTION_RMSE_LIMIT),
                    (
                        float(item["reduction_boundary_conditional_mae"])
                        / (1.5 * BOUNDARY_LIMIT)
                        if np.isfinite(float(item["reduction_boundary_conditional_mae"]))
                        else float("inf")
                    ),
                    float(item["high_delta_rmse"])
                    / max(1.5 * 1.25 * float(item["cn_high_delta_rmse"]), 1e-15),
                    float(item["high_stable_gamma_rmse"])
                    / max(1.5 * 1.25 * float(item["cn_high_stable_gamma_rmse"]), 1e-15),
                ]
                if max(single) > 1.0:
                    failed_seeds += 1
                    failures.append(f"{regime_id}:seed{item['seed']}")
            if failed_seeds >= 2:
                failures.append(f"{regime_id}:two_or_more_failed_seeds")
    price_ratio = max(prices) / REDUCTION_RMSE_LIMIT
    return {
        "passes": not failures,
        "failed_regimes": sorted(set(failures)),
        "worst_reduction_rmse": max(prices),
        "max_gate_ratio": max([price_ratio, *other_ratios]),
        "max_gate_ratio_excluding_price": max(other_ratios),
    }


def run_five_seed_validation(
    *, families: Iterable[str] = ("put", "call"), seeds: Iterable[int] = SEEDS,
    steps: int = FORMAL_STEPS, device: str = "auto", resume: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    development = json.loads(
        (DEVELOPMENT_DIR / "development_decision.json").read_text(encoding="utf-8")
    )
    if development.get("protocol_hash") != protocol_hash():
        raise RuntimeError("development decision protocol hash mismatch")
    failures: list[dict[str, object]] = []
    for family in families:
        selected = development["families"][family]
        if not selected.get("complete_ladder", False):
            raise RuntimeError(
                f"{family} development ladder is incomplete; formal validation is sealed"
            )
        arm, rank = selected["selected_arm"], int(selected["selected_latent_rank"])
        bundle = build_deeponet_training_bundle(train_snapshot_paths(family), family)
        for seed in map(int, seeds):
            output = FIVE_SEED_DIR / family / f"{arm}_rank_{rank:03d}_seed_{seed}"
            checkpoint_interval = max(1, min(1000, int(steps)))
            config = DeepONetTrainingConfig(
                family, arm, rank, seed, int(steps), 8, 1e-3, device,
                "float64", checkpoint_interval,
                3600.0 if int(steps) == FORMAL_STEPS else None,
            )
            result = train_positive_premium_deeponet(
                bundle, config, output_dir=output, resume=resume
            )
            if result.status == "COMPLETE":
                rows = evaluate_checkpoint_on_validation(result.checkpoint_path)
                write_csv(output / "validation_metrics.csv", rows)
            else:
                failures.append({
                    "option_type": family, "arm": arm, "latent_rank": rank,
                    "seed": seed, "status": result.status,
                    "failure_reason": result.failure_reason,
                })
    rows = _collect_metrics(FIVE_SEED_DIR)
    if rows:
        write_csv(FIVE_SEED_DIR / "five_seed_validation_metrics.csv", rows)
    if failures:
        write_csv(FIVE_SEED_DIR / "training_failures.csv", failures)
    decision: dict[str, object] = {
        "protocol_hash": protocol_hash(), "heldout_remains_sealed": True,
        "families": {},
    }
    for family in ("put", "call"):
        selected = development["families"][family]
        family_rows = [row for row in rows if row["option_type"] == family]
        family_failures = [item for item in failures if item["option_type"] == family]
        validation_count = len([
            item for item in load_regimes(splits=("validation",), option_type=family)
            if family == "put" or item.q > 0.0
        ])
        expected = 5 * validation_count
        complete = len(family_rows) == expected
        gate = rows_pass_approximate_gate(family_rows, multi_seed=True)
        gate["passes"] = bool(gate["passes"] and complete and not family_failures)
        oracle_values = [
            float(item["oracle_lcp_p95"]) for item in family_rows
            if np.isfinite(float(item["oracle_lcp_p95"]))
        ]
        # Test/stress Oracle-POD projections remain sealed.  Freeze a single
        # family-level LCP ceiling from validation and carry that number into
        # heldout scoring instead of opening the stopped Basis Operator there.
        heldout_lcp_p95_limit = (
            1.05 * max(oracle_values) if oracle_values else float("nan")
        )
        price_pass = bool(family_rows) and all(
            float(np.median([
                float(item["reduction_price_rmse"]) for item in family_rows
                if item["regime_id"] == regime_id
            ])) <= REDUCTION_RMSE_LIMIT
            for regime_id in sorted({str(item["regime_id"]) for item in family_rows})
        )
        status = (
            "INCOMPLETE" if not complete else
            "PROCEED_HELDOUT" if gate["passes"] else
            "STOP_TRAINING" if family_failures else
            "STOP_STRUCTURE" if price_pass else "STOP_PRICE_MAPPING"
        )
        decision["families"][family] = {
            "status": status,
            "selected_arm": selected["selected_arm"],
            "selected_latent_rank": selected["selected_latent_rank"],
            "steps": int(steps), "seeds": list(SEEDS),
            "expected_rows": expected, "actual_rows": len(family_rows),
            "price_gate_passes": price_pass, "approximate_gate": gate,
            "heldout_lcp_p95_limit": heldout_lcp_p95_limit,
            "heldout_lcp_comparator": "validation_frozen_12_mode_oracle_pod_ceiling",
            "training_failures": family_failures,
        }
    passing = [
        family for family, payload in decision["families"].items()
        if payload["status"] == "PROCEED_HELDOUT"
    ]
    decision["status"] = (
        "PROCEED_HELDOUT" if len(passing) == 2 else
        "PARTIAL_PROCEED_HELDOUT" if passing else "STOP_BEFORE_HELDOUT"
    )
    decision["heldout_remains_sealed"] = not bool(passing)
    FIVE_SEED_DIR.mkdir(parents=True, exist_ok=True)
    (FIVE_SEED_DIR / "validation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (FIVE_SEED_DIR / "frozen_config.json").write_text(
        json.dumps({
            "protocol_hash": protocol_hash(), "families": decision["families"],
        }, indent=2, sort_keys=True), encoding="utf-8"
    )
    if passing:
        run_grid_transfer_diagnostic(passing)
        run_validation_hybrid_audit(passing, device=device)
    write_checkpoint_manifest(FIVE_SEED_DIR)
    return rows, decision


def run_grid_transfer_diagnostic(families: Iterable[str]) -> list[dict[str, object]]:
    rows = []
    for family in families:
        checkpoints = sorted((FIVE_SEED_DIR / family).glob("*/checkpoint.pt"))
        for checkpoint in checkpoints:
            artifact = load_deeponet_artifact(checkpoint)
            for regime in load_regimes(splits=("validation",), option_type=family):
                if family == "call" and regime.q <= 0.0:
                    continue
                config = regime.config()
                fine = type(config)(
                    config.option_type, config.K, config.T, config.r, config.q,
                    config.sigma, config.Smax, 240, 240,
                    tolerance=LCP_TOLERANCE, obstacle_tolerance=LCP_TOLERANCE,
                )
                prediction = predict_deeponet_surface(
                    artifact, fine, np.linspace(0.0, 4.0, 241),
                    np.linspace(0.0, config.T, 241), device="cpu",
                )
                reference = american_cn_lcp_price(fine, lcp_solver="policy_iteration")
                metrics = audit_deeponet_surface(
                    prediction, fine, reference.value_grid, prefix="grid_transfer"
                )
                rows.append({
                    "regime_id": regime.regime_id, "option_type": family,
                    "seed": artifact.config["seed"], "M": 240, "N": 240, **metrics,
                })
    if rows:
        write_csv(GRID_TRANSFER_DIR / "grid_transfer_metrics.csv", rows)
    return rows


def run_validation_hybrid_audit(
    families: Iterable[str], *, device: str = "auto",
) -> list[dict[str, object]]:
    """One-run validation audit; it is not a formal latency headline."""

    rows = []
    for family in families:
        checkpoints = sorted((FIVE_SEED_DIR / family).glob("*/checkpoint.pt"))
        regimes = [
            item for item in load_regimes(splits=("validation",), option_type=family)
            if family == "put" or item.q > 0.0
        ]
        for checkpoint in checkpoints:
            artifact = load_deeponet_artifact(checkpoint)
            for regime in regimes:
                config = regime.config()
                prediction = predict_deeponet_surface(artifact, config, device=device)
                conventional = american_cn_lcp_price(
                    config, lcp_solver="policy_iteration"
                )
                started = perf_counter()
                hybrid = american_cn_lcp_price(
                    config, lcp_solver="policy_iteration",
                    initializer=make_deeponet_policy_initializer(prediction),
                )
                rows.append({
                    "regime_id": regime.regime_id,
                    "option_type": family,
                    "seed": int(artifact.config["seed"]),
                    "hybrid_finish_seconds_one_run": perf_counter() - started,
                    "hybrid_converged": hybrid.converged,
                    "hybrid_iterations": sum(
                        item.iterations for item in hybrid.lcp_results
                    ),
                    "conventional_iterations": sum(
                        item.iterations for item in conventional.lcp_results
                    ),
                    "max_lcp_residual": max(
                        item.residual.normalized_lcp_residual
                        for item in hybrid.lcp_results
                    ),
                    "max_solution_difference_vs_policy": float(np.max(np.abs(
                        hybrid.value_grid - conventional.value_grid
                    ))),
                    "scope": "VALIDATION_ONE_RUN_NOT_FORMAL_TIMING",
                })
    if rows:
        write_csv(FIVE_SEED_DIR / "validation_hybrid_audit.csv", rows)
    return rows


def _collect_metrics(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("**/validation_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(
                row for row in csv.DictReader(handle)
                if row.get("protocol_hash") == protocol_hash()
            )
    return rows


def write_checkpoint_manifest(root: Path) -> Path:
    rows = []
    for checkpoint in sorted(root.glob("**/checkpoint.pt")):
        status_path = checkpoint.parent / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists() else {"status": "UNKNOWN"}
        )
        rows.append({
            "checkpoint_path": str(checkpoint.relative_to(RESULTS_DIR)),
            "status": status.get("status"),
            "step": status.get("step"),
            "target_steps": status.get("target_steps"),
            "training_seconds": status.get("training_seconds"),
            "protocol_hash": status.get("protocol_hash"),
            "size_bytes": checkpoint.stat().st_size,
        })
    destination = root / "checkpoint_manifest.csv"
    if rows:
        write_csv(destination, rows)
    return destination
