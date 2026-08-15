"""Validation-gated heldout prediction, scoring, q=0 diagnostics, and timing."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from american_risk_surfaces.deeponet.evaluation import audit_deeponet_surface
from american_risk_surfaces.deeponet.model import (
    load_deeponet_artifact,
    model_from_artifact,
)
from american_risk_surfaces.deeponet.prediction import (
    make_deeponet_policy_initializer,
    predict_deeponet_surface,
    predict_q0_call_analytic_control,
)
from american_risk_surfaces.deeponet.protocol import RESULTS_DIR, protocol_hash
from american_risk_surfaces.deeponet.study import (
    FIVE_SEED_DIR,
    HELDOUT_DIR,
    RUNTIME_DIR,
    write_csv,
)
from american_risk_surfaces.reduced_order.protocol import load_regimes
from american_risk_surfaces.reduced_order.metrics import (
    interpolate_reference_surface,
    score_value_trajectory,
)
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid


PREDICTION_DIR = HELDOUT_DIR / "predictions"
SCORING_MARKER = HELDOUT_DIR / "SCORING_COMPLETE_DO_NOT_RETRAIN.json"


def heldout_gate() -> dict[str, object]:
    path = FIVE_SEED_DIR / "validation_decision.json"
    if not path.exists():
        raise RuntimeError("five-seed DeepONet validation decision is missing")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision["protocol_hash"] != protocol_hash():
        raise RuntimeError("validation decision protocol hash mismatch")
    return decision


def run_heldout_predictions(*, device: str = "cuda") -> dict[str, object]:
    decision = heldout_gate()
    passing = {
        family for family, payload in decision["families"].items()
        if payload["status"] == "PROCEED_HELDOUT"
    }
    if not passing:
        return _write_status(
            "prediction_status.json",
            {"status": "SKIPPED_VALIDATION_GATE", "heldout_read": False,
             "reason": "no DeepONet family passed validation", "protocol_hash": protocol_hash()},
        )
    if SCORING_MARKER.exists():
        raise PermissionError("heldout was scored permanently; prediction is locked")
    rows = []
    for regime in load_regimes(splits=("test", "stress_holdout")):
        family = regime.option_type
        if family == "call" and regime.q == 0.0:
            if family not in passing:
                continue
            checkpoints: list[Path | None] = [None]
            branch = "q0_analytic"
        elif family in passing:
            checkpoints = sorted((FIVE_SEED_DIR / family).glob("*/checkpoint.pt"))
            if len(checkpoints) != 5:
                raise RuntimeError(
                    f"{family} heldout requires exactly five frozen checkpoints"
                )
            branch = "deeponet"
        else:
            continue
        for checkpoint in checkpoints:
            if checkpoint is None:
                prediction = predict_q0_call_analytic_control(regime.config())
                seed = -1
            else:
                artifact = load_deeponet_artifact(checkpoint)
                prediction = predict_deeponet_surface(
                    artifact, regime.config(), device=device, compute_ad_greeks=True
                )
                seed = int(artifact.config["seed"])
            path = PREDICTION_DIR / regime.split / regime.regime_id / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, value_grid=prediction.value_grid,
                raw_premium_grid=prediction.raw_premium_grid,
                projected_premium_grid=prediction.projected_premium_grid,
                ad_delta_grid=(
                    prediction.ad_delta_grid
                    if prediction.ad_delta_grid is not None else np.empty((0, 0))
                ),
                ad_gamma_grid=(
                    prediction.ad_gamma_grid
                    if prediction.ad_gamma_grid is not None else np.empty((0, 0))
                ),
                timing_json=json.dumps(prediction.timing, sort_keys=True),
                control_branch=prediction.control_branch, protocol_hash=protocol_hash(),
            )
            rows.append({
                "regime_id": regime.regime_id, "split": regime.split,
                "option_type": family, "q": regime.q, "seed": seed,
                "branch": branch, "path": str(path.relative_to(RESULTS_DIR)),
            })
    write_csv(HELDOUT_DIR / "prediction_manifest.csv", rows)
    return _write_status(
        "prediction_status.json",
        {"status": "PREDICTIONS_COMPLETE", "count": len(rows),
         "protocol_hash": protocol_hash()},
    )


def score_heldout_predictions() -> dict[str, object]:
    if SCORING_MARKER.exists():
        raise PermissionError("heldout scoring marker already exists")
    gate = heldout_gate()
    if not any(
        item["status"] == "PROCEED_HELDOUT" for item in gate["families"].values()
    ):
        return _write_status(
            "scoring_status.json",
            {"status": "SKIPPED_VALIDATION_GATE", "heldout_labels_read": False,
             "protocol_hash": protocol_hash()},
        )
    manifest_path = HELDOUT_DIR / "prediction_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("complete all heldout predictions before scoring")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    expected_paths = [RESULTS_DIR / item["path"] for item in manifest]
    if not expected_paths or not all(path.exists() for path in expected_paths):
        raise RuntimeError("heldout prediction manifest is incomplete")
    regimes = {
        item.regime_id: item for item in load_regimes(splits=("test", "stress_holdout"))
    }
    expected = _expected_heldout_keys(gate, regimes)
    actual = {(item["regime_id"], int(item["seed"])) for item in manifest}
    if actual != expected or len(actual) != len(manifest):
        raise RuntimeError(
            "heldout manifest task mismatch; refusing partial, duplicate, or extra scoring"
        )
    rows = []
    for item, path in zip(manifest, expected_paths):
        regime = regimes[item["regime_id"]]
        truth, high_reference = _heldout_reference_bundle(regime)
        with np.load(path, allow_pickle=False) as data:
            from american_risk_surfaces.deeponet.types import DeepONetPrediction

            prediction = DeepONetPrediction(
                data["raw_premium_grid"], data["projected_premium_grid"],
                data["value_grid"], json.loads(str(data["timing_json"])),
                str(data["control_branch"]),
                data["ad_delta_grid"] if data["ad_delta_grid"].size else None,
                data["ad_gamma_grid"] if data["ad_gamma_grid"].size else None,
            )
        metrics = audit_deeponet_surface(
            prediction, regime.config(), truth.value_grid, prefix="reduction"
        )
        high = audit_deeponet_surface(
            prediction, regime.config(), high_reference, prefix="high"
        )
        cn_high = score_value_trajectory(
            truth.value_grid, high_reference, truth.payoff, truth.spot_grid,
            truth.tau_grid, regime.option_type,
        )
        rows.append({
            **item,
            # This is deliberately a validation-frozen family ceiling.  It is
            # not a heldout Oracle-POD score, so the stopped P2 comparator stays
            # sealed on test/stress.
            "oracle_lcp_p95": (
                float(gate["families"][regime.option_type]["heldout_lcp_p95_limit"])
                / 1.05
                if int(item["seed"]) >= 0 else float("nan")
            ),
            "lcp_comparator": "validation_frozen_12_mode_oracle_pod_ceiling",
            "cn_high_price_rmse": cn_high["price_rmse"],
            "cn_high_delta_rmse": cn_high["delta_rmse"],
            "cn_high_stable_gamma_rmse": cn_high["stable_gamma_rmse"],
            **metrics, **high,
        })
    write_csv(HELDOUT_DIR / "heldout_metrics.csv", rows)
    marker = {
        "status": "SCORING_COMPLETE", "row_count": len(rows),
        "protocol_hash": protocol_hash(),
        "policy": "permanent marker: heldout results may not trigger retraining",
    }
    temporary = SCORING_MARKER.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, SCORING_MARKER)
    return marker


def _expected_heldout_keys(gate, regimes):
    expected: set[tuple[str, int]] = set()
    for regime in regimes.values():
        if regime.option_type == "call" and regime.q == 0.0:
            if gate["families"]["call"]["status"] == "PROCEED_HELDOUT":
                expected.add((regime.regime_id, -1))
            continue
        if gate["families"][regime.option_type]["status"] == "PROCEED_HELDOUT":
            expected.update((regime.regime_id, seed) for seed in (17, 29, 43, 71, 101))
    return expected


def run_q0_raw_ood_diagnostic(*, device: str = "cuda") -> dict[str, object]:
    if not SCORING_MARKER.exists():
        raise PermissionError("raw q=0 DeepONet OOD is allowed only after main scoring")
    gate = heldout_gate()
    if gate["families"]["call"]["status"] != "PROCEED_HELDOUT":
        raise PermissionError("dividend-call DeepONet did not pass validation")
    rows = []
    checkpoints = sorted((FIVE_SEED_DIR / "call").glob("*/checkpoint.pt"))
    q0 = [
        item for item in load_regimes(splits=("test", "stress_holdout"))
        if item.option_type == "call" and item.q == 0.0
    ]
    for checkpoint in checkpoints:
        artifact = load_deeponet_artifact(checkpoint)
        seed = int(artifact.config["seed"])
        for regime in q0:
            prediction = predict_deeponet_surface(
                artifact, regime.config(), device=device,
                compute_ad_greeks=True, _allow_q0_neural_ood=True,
            )
            truth = predict_q0_call_analytic_control(regime.config())
            metrics = audit_deeponet_surface(
                prediction, regime.config(), truth.value_grid, prefix="analytic"
            )
            path = HELDOUT_DIR / "q0_raw_ood" / regime.regime_id / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, value_grid=prediction.value_grid,
                raw_premium_grid=prediction.raw_premium_grid,
                projected_premium_grid=prediction.projected_premium_grid,
                protocol_hash=protocol_hash(),
            )
            rows.append({
                "regime_id": regime.regime_id, "seed": seed,
                "status": "OOD_DIAGNOSTIC_ONLY",
                "excluded_from_main_decision": True,
                "path": str(path.relative_to(RESULTS_DIR)),
                **metrics,
            })
    write_csv(HELDOUT_DIR / "q0_raw_ood_metrics.csv", rows)
    return {"status": "OOD_DIAGNOSTIC_ONLY_COMPLETE", "count": len(rows)}


def benchmark_runtime_and_hybrid(
    *, device: str = "cuda", warmups: int = 5, repeats: int = 30
) -> list[dict[str, object]]:
    gate = heldout_gate()
    passing = [
        family for family, payload in gate["families"].items()
        if payload["status"] == "PROCEED_HELDOUT"
    ]
    if not passing:
        raise PermissionError("runtime/hybrid is allowed only for validation-passing families")
    if not SCORING_MARKER.exists():
        raise PermissionError("formal heldout scoring must finish before runtime/hybrid")
    rows = []
    rng = np.random.default_rng(1701)
    for family in passing:
        checkpoints = sorted((FIVE_SEED_DIR / family).glob("*/checkpoint.pt"))
        regimes = [
            item for item in load_regimes(
                splits=("test", "stress_holdout"), option_type=family
            )
            if family == "put" or item.q > 0.0
        ]
        for checkpoint in checkpoints:
            load_started = perf_counter_ns()
            artifact = load_deeponet_artifact(checkpoint)
            artifact_load_seconds = (perf_counter_ns() - load_started) * 1e-9
            model_started = perf_counter_ns()
            prepared_model = model_from_artifact(artifact, device=device)
            model_materialization_seconds = (perf_counter_ns() - model_started) * 1e-9
            for regime in regimes:
                config = regime.config()
                strict_reference = american_cn_lcp_price(
                    config, lcp_solver="policy_iteration"
                )
                for _ in range(warmups):
                    prediction = predict_deeponet_surface(
                        artifact, config, device=device,
                        _prepared_model=prepared_model,
                    )
                    american_cn_lcp_price(config, lcp_solver="psor")
                    american_cn_lcp_price(config, lcp_solver="policy_iteration")
                    american_cn_lcp_price(
                        config, lcp_solver="policy_iteration",
                        initializer=make_deeponet_policy_initializer(prediction),
                    )
                for repeat in range(repeats):
                    order = ["psor", "policy", "deeponet_safe", "hybrid"]
                    rng.shuffle(order)
                    cached_prediction = None
                    for arm in order:
                        started = perf_counter_ns()
                        iterations = 0
                        residual = float("nan")
                        solution_difference = float("nan")
                        converged = True
                        phase_timing: dict[str, float] = {}
                        if arm == "psor":
                            result = american_cn_lcp_price(config, lcp_solver="psor")
                        elif arm == "policy":
                            result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
                        elif arm == "deeponet_safe":
                            cached_prediction = predict_deeponet_surface(
                                artifact, config, device=device,
                                _prepared_model=prepared_model,
                            )
                            audit_started = perf_counter_ns()
                            audit_deeponet_surface(cached_prediction, config)
                            phase_timing = dict(cached_prediction.timing)
                            phase_timing["full_lcp_audit_seconds"] = (
                                perf_counter_ns() - audit_started
                            ) * 1e-9
                            result = None
                        else:
                            cached_prediction = predict_deeponet_surface(
                                artifact, config, device=device,
                                _prepared_model=prepared_model,
                            )
                            phase_timing = dict(cached_prediction.timing)
                            finish_started = perf_counter_ns()
                            result = american_cn_lcp_price(
                                config, lcp_solver="policy_iteration",
                                initializer=make_deeponet_policy_initializer(cached_prediction),
                            )
                            phase_timing["policy_finish_seconds"] = (
                                perf_counter_ns() - finish_started
                            ) * 1e-9
                        seconds = (perf_counter_ns() - started) * 1e-9
                        if result is not None:
                            iterations = sum(item.iterations for item in result.lcp_results)
                            residual = max(item.residual.normalized_lcp_residual for item in result.lcp_results)
                            converged = result.converged
                            solution_difference = float(np.max(np.abs(
                                result.value_grid - strict_reference.value_grid
                            )))
                        rows.append({
                            "regime_id": regime.regime_id, "option_type": family,
                            "seed": artifact.config["seed"], "repeat": repeat,
                            "arm": arm, "seconds": seconds, "iterations": iterations,
                            "max_lcp_residual": residual, "converged": converged,
                            "max_solution_difference_vs_policy": solution_difference,
                            "device": device,
                            "artifact_load_seconds": artifact_load_seconds,
                            "cold_model_materialization_seconds": (
                                model_materialization_seconds
                            ),
                            **phase_timing,
                        })
    if "call" in passing:
        q0_regimes = [
            item for item in load_regimes(
                splits=("test", "stress_holdout"), option_type="call"
            )
            if item.q == 0.0
        ]
        for regime in q0_regimes:
            for _ in range(warmups):
                predict_q0_call_analytic_control(regime.config())
            for repeat in range(repeats):
                started = perf_counter_ns()
                prediction = predict_q0_call_analytic_control(regime.config())
                audit_started = perf_counter_ns()
                audit_deeponet_surface(prediction, regime.config())
                audit_seconds = (perf_counter_ns() - audit_started) * 1e-9
                rows.append({
                    "regime_id": regime.regime_id,
                    "option_type": "call", "seed": -1, "repeat": repeat,
                    "arm": "q0_analytic_safe",
                    "seconds": (perf_counter_ns() - started) * 1e-9,
                    "iterations": 0, "max_lcp_residual": float("nan"),
                    "converged": True,
                    "max_solution_difference_vs_policy": float("nan"),
                    "device": "cpu", "artifact_load_seconds": 0.0,
                    "cold_model_materialization_seconds": 0.0,
                    "full_lcp_audit_seconds": audit_seconds,
                    **prediction.timing,
                })
    write_csv(RUNTIME_DIR / f"runtime_samples_{device}.csv", rows)
    return rows


def _write_status(name: str, payload: dict[str, object]) -> dict[str, object]:
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    (HELDOUT_DIR / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _heldout_reference_bundle(regime):
    cache = HELDOUT_DIR / "reference_cache" / f"{regime.regime_id}_M480_N960.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cn = american_cn_lcp_price(regime.config(), lcp_solver="policy_iteration")
            return cn, data["high_reference"].copy()
    config = regime.config()
    cn = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    fine_config = type(config)(
        config.option_type, config.K, config.T, config.r, config.q, config.sigma,
        config.Smax, 480, 960, tolerance=1e-12, obstacle_tolerance=1e-12,
    )
    fine = american_dirk_policy_price(
        fine_config, quadratic_time=True, damping_steps=2,
        spot_grid=sinh_spot_grid(config.Smax, config.K, 480),
    )
    high = interpolate_reference_surface(
        fine.value_grid, fine.spot_grid, fine.tau_grid,
        cn.spot_grid, cn.tau_grid,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, high_reference=high, protocol_hash=protocol_hash())
    return cn, high
