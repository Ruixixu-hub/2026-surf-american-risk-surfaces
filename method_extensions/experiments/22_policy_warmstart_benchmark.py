"""Stage 1/2: 2x2 policy-iteration and learned-warm-start benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "surf-matplotlib-cache")
)

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve
from american_risk_surfaces.diagnostics.greeks import finite_difference_delta
from american_risk_surfaces.method_extensions.premium_warmstart import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_WARMSTART_DIR,
    GatedSurfaceInitializer,
    PositivePremiumSurfaceModel,
    train_positive_premium_checkpoint,
)
from american_risk_surfaces.method_extensions.protocol import DATASET_DIR
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    AmericanLCPResult,
    american_cn_lcp_price,
    as_legacy_cn_psor_result,
)
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid


ARMS = {
    "A_previous_psor": ("psor", False),
    "B_previous_policy": ("policy_iteration", False),
    "C_mlp_psor": ("psor", True),
    "D_mlp_policy": ("policy_iteration", True),
}
RUNTIME_FIELDS = (
    "regime_id",
    "split",
    "arm",
    "repeat",
    "option_type",
    "prediction_seconds",
    "projection_seconds",
    "lcp_finish_seconds",
    "marcher_other_seconds",
    "total_seconds",
    "converged",
    "total_iterations",
    "mean_iterations",
    "max_iterations",
    "max_normalized_lcp_residual",
    "max_normalized_obstacle_violation",
)


def run_benchmark(
    *,
    output_dir: Path | str = DEFAULT_WARMSTART_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    warmups: int = 5,
    repeats: int = 30,
    regime_limit: int | None = None,
    retrain: bool = False,
    include_raw_extrapolation: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    if retrain or not checkpoint_path.exists():
        train_positive_premium_checkpoint(checkpoint_path)
    load_started = perf_counter()
    model = PositivePremiumSurfaceModel.load(checkpoint_path)
    model_load_seconds = perf_counter() - load_started

    tolerance = _frozen_tolerance()
    regimes = _evaluation_regimes(regime_limit)
    runtime_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260811)

    for regime in regimes:
        config = _config(regime, tolerance)
        reference = american_cn_lcp_price(
            config, lcp_solver="policy_iteration", initializer="previous_slice"
        )

        for arm in ARMS:
            for _ in range(warmups):
                _run_arm(config, arm, model)

        first_results: dict[str, AmericanLCPResult] = {}
        for repeat in range(repeats):
            arm_order = list(ARMS)
            rng.shuffle(arm_order)
            for arm in arm_order:
                result, timing = _run_arm(config, arm, model)
                first_results.setdefault(arm, result)
                runtime_rows.append(
                    _runtime_row(regime, arm, repeat, result, timing)
                )

        for arm, result in first_results.items():
            accuracy_rows.append(_accuracy_row(regime, arm, result, reference))

        if include_raw_extrapolation:
            for solver in ("psor", "policy_iteration"):
                result, timing = _run_arm(
                    config,
                    "C_mlp_psor" if solver == "psor" else "D_mlp_policy",
                    model,
                    raw_extrapolation=True,
                )
                raw_rows.append(
                    {
                        **_accuracy_row(
                            regime,
                            f"raw_extrapolation_{solver}",
                            result,
                            reference,
                        ),
                        "total_seconds": timing["total_seconds"],
                    }
                )

    runtime_path = output / "runtime_samples.csv"
    accuracy_path = output / "accuracy_structure_metrics.csv"
    raw_path = output / "raw_extrapolation_diagnostic.csv"
    _write_csv(runtime_path, runtime_rows, RUNTIME_FIELDS)
    _write_csv(accuracy_path, accuracy_rows, tuple(accuracy_rows[0]))
    if raw_rows:
        _write_csv(raw_path, raw_rows, tuple(raw_rows[0]))

    summary_rows = _summary_rows(runtime_rows)
    summary_path = output / "runtime_summary.csv"
    _write_csv(summary_path, summary_rows, tuple(summary_rows[0]))
    decision = _decision(summary_rows, runtime_rows, accuracy_rows, model.metadata)
    decision.update(
        {
            "warmups": warmups,
            "repeats": repeats,
            "evaluated_regimes": len(regimes),
            "model_load_seconds": model_load_seconds,
            "frozen_tolerance": tolerance,
            "checkpoint_path": str(checkpoint_path),
            "protocol_complete": warmups == 5 and repeats == 30 and regime_limit is None,
        }
    )
    decision_path = output / "method_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "benchmark_report.md"
    report_path.write_text(_markdown_report(decision, summary_rows), encoding="utf-8")
    return {
        "runtime": runtime_path,
        "accuracy": accuracy_path,
        "raw_extrapolation": raw_path if raw_rows else None,
        "summary": summary_path,
        "decision": decision_path,
        "report": report_path,
        "decision_data": decision,
    }


def _run_arm(
    config: AmericanLCPConfig,
    arm: str,
    model: PositivePremiumSurfaceModel,
    *,
    raw_extrapolation: bool = False,
) -> tuple[AmericanLCPResult, dict[str, float]]:
    solver, use_mlp = ARMS[arm]
    prediction_seconds = 0.0
    initializer: Any = "previous_slice"
    if use_mlp:
        spots, _ = uniform_spot_grid(config.Smax, config.M)
        taus, _ = uniform_tau_grid(config.T, config.N)
        prediction = model.predict_surface(config, spots, taus)
        prediction_seconds = prediction.inference_seconds
        initializer = GatedSurfaceInitializer(
            prediction.value_grid,
            spots[1:-1],
            config.K,
            raw_extrapolation=raw_extrapolation,
        )
    result = american_cn_lcp_price(
        config,
        lcp_solver=solver,
        initializer=initializer,
    )
    other = max(
        result.total_seconds - result.initialization_seconds - result.lcp_finish_seconds,
        0.0,
    )
    return result, {
        "prediction_seconds": prediction_seconds,
        "projection_seconds": result.initialization_seconds,
        "lcp_finish_seconds": result.lcp_finish_seconds,
        "marcher_other_seconds": other,
        "total_seconds": prediction_seconds + result.total_seconds,
    }


def _runtime_row(
    regime: dict[str, str],
    arm: str,
    repeat: int,
    result: AmericanLCPResult,
    timing: dict[str, float],
) -> dict[str, Any]:
    iterations = np.array([step.iterations for step in result.lcp_results], dtype=float)
    residuals = [step.residual for step in result.lcp_results]
    return {
        "regime_id": regime["regime_id"],
        "split": regime["split"],
        "arm": arm,
        "repeat": repeat,
        "option_type": regime["option_type"],
        **timing,
        "converged": result.converged,
        "total_iterations": int(np.sum(iterations)),
        "mean_iterations": float(np.mean(iterations)),
        "max_iterations": int(np.max(iterations)),
        "max_normalized_lcp_residual": max(
            residual.normalized_lcp_residual for residual in residuals
        ),
        "max_normalized_obstacle_violation": max(
            residual.normalized_obstacle_violation for residual in residuals
        ),
    }


def _accuracy_row(
    regime: dict[str, str],
    arm: str,
    result: AmericanLCPResult,
    reference: AmericanLCPResult,
) -> dict[str, Any]:
    error = result.value_grid - reference.value_grid
    result_legacy = as_legacy_cn_psor_result(result)
    reference_legacy = as_legacy_cn_psor_result(reference)
    result_boundary = extract_boundary_curve(result_legacy, f"{regime['regime_id']}_{arm}")
    reference_boundary = extract_boundary_curve(reference_legacy, f"{regime['regime_id']}_reference")
    boundary_pairs = [
        (candidate.boundary_spot, target.boundary_spot)
        for candidate, target in zip(result_boundary.points, reference_boundary.points)
        if candidate.boundary_found and target.boundary_found
    ]
    boundary_error = (
        np.array([candidate - target for candidate, target in boundary_pairs], dtype=float)
        if boundary_pairs
        else np.array([], dtype=float)
    )
    delta = finite_difference_delta(result.spot_grid, result.value_grid)
    reference_delta = finite_difference_delta(reference.spot_grid, reference.value_grid)
    finite = np.isfinite(delta) & np.isfinite(reference_delta)
    delta_error = delta[finite] - reference_delta[finite]
    return {
        "regime_id": regime["regime_id"],
        "split": regime["split"],
        "arm": arm,
        "converged": result.converged,
        "max_abs_value_error": float(np.max(np.abs(error))),
        "value_rmse": float(np.sqrt(np.mean(error**2))),
        "boundary_matched_rows": len(boundary_pairs),
        "boundary_mae": float(np.mean(np.abs(boundary_error))) if boundary_pairs else float("nan"),
        "boundary_max_abs_error": float(np.max(np.abs(boundary_error))) if boundary_pairs else float("nan"),
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
        "delta_max_abs_error": float(np.max(np.abs(delta_error))),
        "max_normalized_lcp_residual": max(
            step.residual.normalized_lcp_residual for step in result.lcp_results
        ),
        "max_normalized_obstacle_violation": max(
            step.residual.normalized_obstacle_violation for step in result.lcp_results
        ),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        times = np.array([float(row["total_seconds"]) for row in selected])
        summary.append(
            {
                "arm": arm,
                "sample_count": len(selected),
                "median_total_seconds": float(np.median(times)),
                "p95_total_seconds": float(np.percentile(times, 95)),
                "p99_total_seconds": float(np.percentile(times, 99)),
                "median_prediction_seconds": float(
                    np.median([float(row["prediction_seconds"]) for row in selected])
                ),
                "median_lcp_finish_seconds": float(
                    np.median([float(row["lcp_finish_seconds"]) for row in selected])
                ),
                "median_total_iterations": float(
                    np.median([float(row["total_iterations"]) for row in selected])
                ),
                "all_converged": all(bool(row["converged"]) for row in selected),
            }
        )
    return summary


def _decision(
    summary_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    by_arm = {str(row["arm"]): row for row in summary_rows}
    best_classical = min(("A_previous_psor", "B_previous_policy"), key=lambda arm: by_arm[arm]["median_total_seconds"])
    best_ml = min(("C_mlp_psor", "D_mlp_policy"), key=lambda arm: by_arm[arm]["median_total_seconds"])
    classical_median = float(by_arm[best_classical]["median_total_seconds"])
    ml_median = float(by_arm[best_ml]["median_total_seconds"])
    speedup = 1.0 - ml_median / classical_median
    p95_preserved = float(by_arm[best_ml]["p95_total_seconds"]) <= float(
        by_arm[best_classical]["p95_total_seconds"]
    )
    split_speedups: dict[str, float | None] = {}
    for split in ("test", "stress_holdout"):
        classical = [
            float(row["total_seconds"])
            for row in runtime_rows
            if row["split"] == split and row["arm"] == best_classical
        ]
        learned = [
            float(row["total_seconds"])
            for row in runtime_rows
            if row["split"] == split and row["arm"] == best_ml
        ]
        split_speedups[split] = (
            1.0 - float(np.median(learned)) / float(np.median(classical))
            if classical and learned
            else None
        )
    all_tolerance_pass = all(bool(row["converged"]) for row in accuracy_rows)
    generalizes = all(
        value is not None and value > 0.0 for value in split_speedups.values()
    )
    go = speedup >= 0.20 and p95_preserved and all_tolerance_pass and generalizes
    if go and best_ml == "D_mlp_policy":
        selection = "positive_premium_mlp_to_policy_iteration_finish"
        status = "GO_LEARNED_ACCELERATION"
    elif best_classical == "B_previous_policy":
        selection = "policy_iteration_previous_slice"
        status = "STOP_LEARNED_ACCELERATION_KEEP_POLICY_ITERATION"
    else:
        selection = "psor_previous_slice"
        status = "STOP_LEARNED_ACCELERATION_KEEP_PSOR"
    saving = classical_median - ml_median
    training_seconds = float(model_metadata.get("training_seconds", 0.0))
    break_even = training_seconds / saving if saving > 0.0 else None
    return {
        "status": status,
        "selected_method": selection,
        "best_classical_arm": best_classical,
        "best_ml_arm": best_ml,
        "median_speedup_fraction": speedup,
        "p95_preserved": p95_preserved,
        "all_tolerance_pass": all_tolerance_pass,
        "positive_speedup_on_test_and_stress": generalizes,
        "split_speedup_fraction": split_speedups,
        "go_threshold_fraction": 0.20,
        "incremental_training_seconds": training_seconds,
        "break_even_surface_queries": break_even,
        "existing_v1_label_cost_treatment": "sunk_cost_reported_separately",
    }


def _markdown_report(decision: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Policy Iteration and Positive-Premium Warm-Start Benchmark",
        "",
        f"Decision: **{decision['status']}**",
        "",
        f"Selected method: `{decision['selected_method']}`",
        "",
        f"Frozen normalized LCP tolerance: `{decision['frozen_tolerance']}`",
        "",
        f"Protocol complete: `{decision['protocol_complete']}`",
        "",
        "| Arm | Median (s) | p95 (s) | Median iterations |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['median_total_seconds']:.6g} | "
            f"{row['p95_total_seconds']:.6g} | {row['median_total_iterations']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Best learned-vs-classical median speedup: `{decision['median_speedup_fraction']:.3%}`.",
            "",
            "The learned path is accepted only when the 20% end-to-end gate, p95 gate, "
            "strict-tolerance gate, and both held-out split gates pass together.",
        ]
    )
    return "\n".join(lines) + "\n"


def _frozen_tolerance() -> float:
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "07_method_extensions"
        / "00_protocol"
        / "tolerance_decision.json"
    )
    if not path.exists():
        raise FileNotFoundError("run experiment 21 before experiment 22")
    decision = json.loads(path.read_text(encoding="utf-8"))
    tolerance = decision.get("frozen_normalized_lcp_tolerance")
    if tolerance is None:
        raise RuntimeError("Stage 0 did not freeze an LCP tolerance")
    return float(tolerance)


def _evaluation_regimes(limit: int | None) -> list[dict[str, str]]:
    with (DATASET_DIR / "regime_manifest.csv").open(newline="", encoding="utf-8") as handle:
        regimes = [
            row
            for row in csv.DictReader(handle)
            if row["split"] in {"test", "stress_holdout"}
        ]
    regimes.sort(key=lambda row: row["regime_id"])
    return regimes if limit is None else regimes[:limit]


def _config(regime: dict[str, str], tolerance: float) -> AmericanLCPConfig:
    return AmericanLCPConfig(
        option_type=regime["option_type"],
        K=float(regime["K"]),
        T=float(regime["T"]),
        r=float(regime["r"]),
        q=float(regime["q"]),
        sigma=float(regime["sigma"]),
        Smax=float(regime["Smax"]),
        M=int(regime["M"]),
        N=int(regime["N"]),
        tolerance=tolerance,
        obstacle_tolerance=1e-12,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--regime-limit", type=int)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-raw-extrapolation", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        warmups=args.warmups,
        repeats=args.repeats,
        regime_limit=args.regime_limit,
        retrain=args.retrain,
        include_raw_extrapolation=not args.skip_raw_extrapolation,
    )
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
