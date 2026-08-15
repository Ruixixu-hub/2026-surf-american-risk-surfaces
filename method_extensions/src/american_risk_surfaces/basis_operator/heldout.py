"""Permanently gated heldout prediction, scoring, and timing helpers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from american_risk_surfaces.basis_operator.evaluation import audit_basis_operator_surface
from american_risk_surfaces.basis_operator.model import load_basis_operator_artifact
from american_risk_surfaces.basis_operator.prediction import (
    make_basis_operator_policy_initializer,
    predict_basis_operator_surface,
)
from american_risk_surfaces.basis_operator.protocol import RESULTS_DIR, protocol_hash
from american_risk_surfaces.basis_operator.study import FIVE_SEED_DIR, HELDOUT_DIR, write_csv
from american_risk_surfaces.reduced_order.protocol import load_regimes
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price


PREDICTION_DIR = HELDOUT_DIR / "predictions"
SCORING_MARKER = HELDOUT_DIR / "SCORING_COMPLETE_DO_NOT_RETRAIN.json"


def heldout_gate() -> dict[str, object]:
    path = FIVE_SEED_DIR / "validation_decision.json"
    if not path.exists():
        raise RuntimeError("five-seed validation decision is missing")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision["protocol_hash"] != protocol_hash():
        raise RuntimeError("validation decision protocol hash mismatch")
    return decision


def run_heldout_predictions() -> dict[str, object]:
    decision = heldout_gate()
    passing = {
        family for family, payload in decision["families"].items()
        if payload["status"] == "PROCEED_HELDOUT"
    }
    if not passing:
        payload = {
            "status": "SKIPPED_VALIDATION_GATE",
            "heldout_read": False,
            "reason": "no family passed five-seed approximate-quality validation",
            "protocol_hash": protocol_hash(),
        }
        HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
        (HELDOUT_DIR / "prediction_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return payload
    if SCORING_MARKER.exists():
        raise PermissionError("heldout was scored permanently; retraining/prediction is locked")
    rows = []
    regimes = load_regimes(splits=("test", "stress_holdout"))
    for regime in regimes:
        family = regime.option_type
        if family == "call" and regime.q == 0.0:
            branch = "q0_analytic"
            checkpoint_paths = [None]
        elif family in passing:
            branch = "basis_operator"
            checkpoint_paths = sorted((FIVE_SEED_DIR / family).glob("**/checkpoint.pt"))
        else:
            continue
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path is None:
                from american_risk_surfaces.basis_operator.prediction import predict_no_dividend_call_control
                prediction = predict_no_dividend_call_control(regime.config())
                seed = -1
            else:
                artifact = load_basis_operator_artifact(checkpoint_path)
                prediction = predict_basis_operator_surface(artifact, regime.config())
                seed = int(artifact.config["seed"])
            path = PREDICTION_DIR / regime.split / regime.regime_id / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                value_grid=prediction.value_grid,
                raw_premium_grid=prediction.raw_premium_grid,
                projected_premium_grid=prediction.projected_premium_grid,
                timing_json=json.dumps(prediction.timing, sort_keys=True),
                control_branch=prediction.control_branch,
                protocol_hash=protocol_hash(),
            )
            rows.append({
                "regime_id": regime.regime_id, "split": regime.split,
                "option_type": family, "q": regime.q, "seed": seed,
                "branch": branch, "path": str(path.relative_to(RESULTS_DIR)),
            })
    write_csv(HELDOUT_DIR / "prediction_manifest.csv", rows)
    payload = {"status": "PREDICTIONS_COMPLETE", "count": len(rows), "protocol_hash": protocol_hash()}
    (HELDOUT_DIR / "prediction_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def score_heldout_predictions() -> dict[str, object]:
    if SCORING_MARKER.exists():
        raise PermissionError("heldout scoring marker already exists")
    gate = heldout_gate()
    if not any(
        payload["status"] == "PROCEED_HELDOUT" for payload in gate["families"].values()
    ):
        payload = {
            "status": "SKIPPED_VALIDATION_GATE", "heldout_labels_read": False,
            "protocol_hash": protocol_hash(),
        }
        HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
        (HELDOUT_DIR / "scoring_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return payload
    manifest_path = HELDOUT_DIR / "prediction_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("complete heldout predictions before scoring")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    expected_paths = [RESULTS_DIR / row["path"] for row in manifest]
    if not expected_paths or not all(path.exists() for path in expected_paths):
        raise RuntimeError("prediction manifest is incomplete")
    regime_map = {
        item.regime_id: item for item in load_regimes(splits=("test", "stress_holdout"))
    }
    rows = []
    for item, path in zip(manifest, expected_paths):
        regime = regime_map[item["regime_id"]]
        truth = american_cn_lcp_price(regime.config(), lcp_solver="policy_iteration")
        with np.load(path, allow_pickle=False) as data:
            from american_risk_surfaces.basis_operator.types import BasisOperatorPrediction
            prediction = BasisOperatorPrediction(
                data["raw_premium_grid"], data["projected_premium_grid"], data["value_grid"],
                json.loads(str(data["timing_json"])), str(data["control_branch"]),
            )
        metrics = audit_basis_operator_surface(
            prediction, regime.config(), reference_value_grid=truth.value_grid,
            prefix="reduction",
        )
        rows.append({**item, **metrics})
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


def benchmark_hybrid(config, prediction, *, repeats: int = 30) -> list[dict[str, object]]:
    initializer = make_basis_operator_policy_initializer(prediction.value_grid)
    for _ in range(5):
        american_cn_lcp_price(config, lcp_solver="policy_iteration")
        american_cn_lcp_price(config, lcp_solver="policy_iteration", initializer=initializer)
    rows = []
    rng = np.random.default_rng(1701)
    for repeat in range(repeats):
        arms = ["policy", "hybrid"]
        rng.shuffle(arms)
        for arm in arms:
            started = perf_counter_ns()
            result = american_cn_lcp_price(
                config, lcp_solver="policy_iteration",
                initializer="previous_slice" if arm == "policy" else initializer,
            )
            elapsed = (perf_counter_ns() - started) * 1e-9
            rows.append({
                "repeat": repeat, "arm": arm, "seconds": elapsed,
                "converged": result.converged,
                "iterations": sum(item.iterations for item in result.lcp_results),
                "max_lcp_residual": max(item.residual.normalized_lcp_residual for item in result.lcp_results),
            })
    return rows


def run_q0_raw_ood_diagnostic() -> dict[str, object]:
    """Run raw neural q=0 extrapolation only after the permanent main scoring marker."""

    if not SCORING_MARKER.exists():
        raise PermissionError("raw q=0 OOD diagnostics are allowed only after main heldout scoring")
    decision = heldout_gate()
    if decision["families"]["call"]["status"] != "PROCEED_HELDOUT":
        raise PermissionError("dividend-call network did not pass validation")
    from american_risk_surfaces.basis_operator.model import infer_coefficients
    from american_risk_surfaces.basis_operator.prediction import reconstruct_full_prediction
    import math

    rows = []
    checkpoints = sorted((FIVE_SEED_DIR / "call").glob("**/checkpoint.pt"))
    q0_regimes = [
        item for item in load_regimes(splits=("test", "stress_holdout"))
        if item.option_type == "call" and item.q == 0.0
    ]
    for checkpoint in checkpoints:
        artifact = load_basis_operator_artifact(checkpoint)
        seed = int(artifact.config["seed"])
        for regime in q0_regimes:
            features = np.asarray([[math.log(regime.T), regime.sigma, regime.r, 0.0]])
            coefficients = infer_coefficients(artifact, features)[0]
            prediction = reconstruct_full_prediction(artifact.basis, coefficients, regime.config())
            path = HELDOUT_DIR / "q0_raw_ood" / regime.regime_id / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, value_grid=prediction.value_grid, protocol_hash=protocol_hash())
            rows.append({
                "regime_id": regime.regime_id, "seed": seed,
                "status": "OOD_DIAGNOSTIC_ONLY", "path": str(path.relative_to(RESULTS_DIR)),
            })
    write_csv(HELDOUT_DIR / "q0_raw_ood_manifest.csv", rows)
    return {"status": "OOD_DIAGNOSTIC_ONLY_COMPLETE", "count": len(rows)}
