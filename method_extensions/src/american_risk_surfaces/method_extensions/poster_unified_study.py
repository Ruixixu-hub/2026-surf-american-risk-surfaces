"""Poster-only unified solver comparison and accuracy-reference audit.

All outputs are isolated from the original method-extension evidence.  The
module treats CN+PSOR as the Basic / Original Classical Benchmark, not as a
historical label.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.method_extensions.projected_lu_study import (
    FROZEN_OBSTACLE_TOLERANCE,
    FROZEN_TOLERANCE,
    VALUE_MATCH_TOLERANCE,
    config_from_regime,
    load_regimes,
    runtime_row,
    trajectory_metrics,
)
from american_risk_surfaces.method_extensions.protocol import (
    DATASET_DIR,
    environment_manifest,
    sha256_file,
)
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPResult,
    american_cn_lcp_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "14_poster_unified_comparison"
REPORTS_DIR = PROJECT_ROOT / "reports" / "16_poster_unified_comparison"
PROTOCOL_DIR = RESULTS_DIR / "00_protocol"
STRICT_DIR = RESULTS_DIR / "01_strict_solvers"
ACCURACY_DIR = RESULTS_DIR / "02_accuracy_reference_audit"

PENALTY_LADDER = (1e8, 1e10, 1e12, 1e14)
STRICT_ARMS = ("psor", "policy_iteration", "projected_lu_single", "penalty_newton")
STRICT_ORDER_SEED = 20260817
POSTER_SPEED_GATE = 0.8

BENCHMARK_ROLES = {
    "psor": "Basic / Original Classical Benchmark",
    "policy_iteration": "Strengthened Classical Benchmark 1",
    "projected_lu_single": "Strengthened Classical Benchmark 2",
    "penalty_newton": "Candidate comparator; promotion requires the common correctness gate",
    "dirk_policy_sinh": "High-Accuracy Numerical Reference",
}


def run_protocol_and_penalty_validation(
    output_dir: Path | str = PROTOCOL_DIR,
    *,
    regime_limit: int | None = None,
    penalty_ladder: Iterable[float] = PENALTY_LADDER,
) -> dict[str, Any]:
    """Freeze the poster protocol and select a penalty on validation only."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    penalties = tuple(float(value) for value in penalty_ladder)
    if not penalties or any(not np.isfinite(value) or value <= 0.0 for value in penalties):
        raise ValueError("penalty_ladder must contain positive finite values")
    if tuple(sorted(set(penalties))) != penalties:
        raise ValueError("penalty_ladder must be unique and increasing")

    required_inputs = _required_protocol_inputs()
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required poster-comparison inputs are missing: {missing}")

    regimes = [row for row in load_regimes() if row["split"] == "validation"]
    regimes.sort(key=lambda row: row["regime_id"])
    if regime_limit is not None:
        if regime_limit < 1:
            raise ValueError("regime_limit must be positive")
        regimes = regimes[:regime_limit]

    protocol = {
        "protocol_version": "poster_unified_comparison_v1",
        "benchmark_roles": BENCHMARK_ROLES,
        "strict_solver_question": "same frozen CN-LCP; only the LCP solver changes",
        "accuracy_reference_question": "time/spatial discretization and Greek stability",
        "frozen_grid": {"K": 1.0, "Smax": 4.0, "M": 120, "N": 120},
        "frozen_normalized_lcp_tolerance": FROZEN_TOLERANCE,
        "frozen_normalized_obstacle_tolerance": FROZEN_OBSTACLE_TOLERANCE,
        "policy_match_tolerance_K": VALUE_MATCH_TOLERANCE,
        "penalty_ladder": list(penalties),
        "penalty_newton_max_iter": 100,
        "penalty_selection_split": "validation_only",
        "penalty_selection_rule": (
            "smallest penalty passing every common correctness gate; otherwise freeze "
            "the smallest worst-residual candidate as a failed diagnostic without retuning"
        ),
        "heldout_policy": (
            "one diagnostic evaluation after validation freeze; heldout never changes penalty"
        ),
        "timing": {
            "threads": 1,
            "dtype": "float64",
            "warmups": 5,
            "repeats": 30,
            "randomized_arm_order": True,
        },
        "input_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in required_inputs
        },
        "environment": environment_manifest(),
    }
    protocol["protocol_hash"] = _json_hash(protocol)
    protocol_path = output / "protocol_manifest.json"
    _write_json(protocol_path, protocol)

    metric_rows: list[dict[str, Any]] = []
    for regime in regimes:
        base_config = config_from_regime(regime)
        policy = american_cn_lcp_price(
            base_config, lcp_solver="policy_iteration", initializer="previous_slice"
        )
        for penalty in penalties:
            config = replace(
                base_config, penalty=penalty, penalty_newton_max_iter=100
            )
            candidate = american_cn_lcp_price(
                config, lcp_solver="penalty_newton", initializer="previous_slice"
            )
            row = trajectory_metrics(
                regime, f"penalty_newton_{penalty:.0e}", candidate, policy
            )
            row["penalty"] = penalty
            metric_rows.append(row)

    metrics_path = output / "penalty_validation_metrics.csv"
    _write_csv(metrics_path, metric_rows)
    summary_rows = _penalty_validation_summary(metric_rows, penalties, len(regimes))
    summary_path = output / "penalty_validation_summary.csv"
    _write_csv(summary_path, summary_rows)

    passing = [row for row in summary_rows if bool(row["all_numerically_certified"])]
    if passing:
        selected = min(passing, key=lambda row: float(row["penalty"]))
        status = "PENALTY_CORRECTNESS_PASSED_VALIDATION"
    else:
        selected = min(
            summary_rows,
            key=lambda row: (
                float(row["max_normalized_lcp_residual"]),
                float(row["penalty"]),
            ),
        )
        status = "PENALTY_FAILED_VALIDATION_FROZEN_DIAGNOSTIC"
    frozen = {
        "status": status,
        "selected_penalty": float(selected["penalty"]),
        "validation_regime_count": len(regimes),
        "full_validation_protocol": regime_limit is None and len(regimes) == 19,
        "all_numerically_certified": bool(selected["all_numerically_certified"]),
        "certified_regime_count": int(selected["certified_regime_count"]),
        "max_normalized_lcp_residual": float(selected["max_normalized_lcp_residual"]),
        "max_abs_trajectory_difference": float(selected["max_abs_trajectory_difference"]),
        "common_lcp_tolerance": FROZEN_TOLERANCE,
        "protocol_hash": protocol["protocol_hash"],
        "validation_metrics_sha256": sha256_file(metrics_path),
        "no_post_heldout_retuning": True,
    }
    frozen["frozen_config_hash"] = _json_hash(frozen)
    frozen_path = output / "frozen_penalty_config.json"
    _write_json(frozen_path, frozen)
    return {
        "protocol": protocol_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "frozen": frozen_path,
        "frozen_data": frozen,
    }


def run_strict_solver_benchmark(
    output_dir: Path | str = STRICT_DIR,
    *,
    warmups: int = 5,
    repeats: int = 30,
    regime_limit: int | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Run the matched four-arm poster comparison on frozen held-out regimes."""

    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "strict_solver_complete.marker.json"
    if marker_path.exists() and not allow_existing:
        raise RuntimeError("strict-solver poster benchmark already exists; refusing overwrite")
    frozen = _load_json(PROTOCOL_DIR / "frozen_penalty_config.json")
    if not bool(frozen.get("full_validation_protocol")) and regime_limit is None:
        raise RuntimeError("full validation penalty freeze is required before full heldout")
    penalty = float(frozen["selected_penalty"])

    regimes = [
        row for row in load_regimes() if row["split"] in {"test", "stress_holdout"}
    ]
    regimes.sort(key=lambda row: row["regime_id"])
    if regime_limit is not None:
        if regime_limit < 1:
            raise ValueError("regime_limit must be positive")
        regimes = regimes[:regime_limit]

    rng = np.random.default_rng(STRICT_ORDER_SEED)
    runtime_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    benchmark_started = perf_counter_ns()
    for regime_index, regime in enumerate(regimes, start=1):
        base_config = config_from_regime(regime)
        policy_reference = american_cn_lcp_price(
            base_config, lcp_solver="policy_iteration", initializer="previous_slice"
        )
        for arm in STRICT_ARMS:
            for _ in range(warmups):
                _solve_arm(base_config, arm, penalty)

        first_results: dict[str, AmericanLCPResult] = {}
        for repeat_index in range(repeats):
            order = list(STRICT_ARMS)
            rng.shuffle(order)
            for arm in order:
                started = perf_counter_ns()
                result = _solve_arm(base_config, arm, penalty)
                measured = (perf_counter_ns() - started) * 1e-9
                first_results.setdefault(arm, result)
                row = runtime_row(regime, arm, repeat_index, result, measured)
                row["penalty"] = penalty if arm == "penalty_newton" else ""
                runtime_rows.append(row)
        for arm, result in first_results.items():
            row = trajectory_metrics(regime, arm, result, policy_reference)
            row["penalty"] = penalty if arm == "penalty_newton" else ""
            accuracy_rows.append(row)
        elapsed = (perf_counter_ns() - benchmark_started) * 1e-9
        print(
            f"[{regime_index}/{len(regimes)}] {regime['regime_id']} complete "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )

    runtime_path = output / "runtime_samples.csv"
    accuracy_path = output / "correctness_and_structure.csv"
    _write_csv(runtime_path, runtime_rows)
    _write_csv(accuracy_path, accuracy_rows)
    summary_rows = _runtime_summary(runtime_rows)
    correctness_rows = _correctness_summary(accuracy_rows)
    paired_rows = _paired_runtime(runtime_rows)
    summary_path = output / "runtime_summary.csv"
    correctness_path = output / "correctness_summary.csv"
    paired_path = output / "paired_regime_timing.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(correctness_path, correctness_rows)
    _write_csv(paired_path, paired_rows)

    protocol_complete = bool(
        warmups == 5 and repeats == 30 and regime_limit is None and len(regimes) == 67
    )
    decision = _strict_decision(
        summary_rows,
        correctness_rows,
        paired_rows,
        penalty=penalty,
        regime_count=len(regimes),
        protocol_complete=protocol_complete,
    )
    decision["frozen_penalty_config_hash"] = frozen["frozen_config_hash"]
    decision_path = output / "strict_solver_decision.json"
    _write_json(decision_path, decision)
    marker = {
        "status": "COMPLETE" if protocol_complete else "SMOKE_COMPLETE",
        "regime_count": len(regimes),
        "warmups": warmups,
        "repeats": repeats,
        "runtime_sha256": sha256_file(runtime_path),
        "accuracy_sha256": sha256_file(accuracy_path),
        "decision_sha256": sha256_file(decision_path),
        "frozen_penalty_config_hash": frozen["frozen_config_hash"],
    }
    _write_json(marker_path, marker)
    return {
        "runtime": runtime_path,
        "accuracy": accuracy_path,
        "summary": summary_path,
        "correctness": correctness_path,
        "paired": paired_path,
        "decision": decision_path,
        "marker": marker_path,
        "decision_data": decision,
    }


def resummarize_strict_solver_benchmark(
    output_dir: Path | str = STRICT_DIR,
) -> dict[str, Any]:
    """Regenerate summaries/decision from immutable raw timing and accuracy CSVs."""

    output = Path(output_dir)
    runtime_path = output / "runtime_samples.csv"
    accuracy_path = output / "correctness_and_structure.csv"
    marker_path = output / "strict_solver_complete.marker.json"
    frozen_path = PROTOCOL_DIR / "frozen_penalty_config.json"
    for path in (runtime_path, accuracy_path, marker_path, frozen_path):
        if not path.exists():
            raise FileNotFoundError(f"required strict-solver artifact is missing: {path}")
    runtime_rows = _read_csv(runtime_path)
    accuracy_rows = _read_csv(accuracy_path)
    marker = _load_json(marker_path)
    frozen = _load_json(frozen_path)
    summary_rows = _runtime_summary(runtime_rows)
    correctness_rows = _correctness_summary(accuracy_rows)
    paired_rows = _paired_runtime(runtime_rows)
    summary_path = output / "runtime_summary.csv"
    correctness_path = output / "correctness_summary.csv"
    paired_path = output / "paired_regime_timing.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(correctness_path, correctness_rows)
    _write_csv(paired_path, paired_rows)
    protocol_complete = bool(
        int(marker["warmups"]) == 5
        and int(marker["repeats"]) == 30
        and int(marker["regime_count"]) == 67
        and len(runtime_rows) == 67 * len(STRICT_ARMS) * 30
    )
    decision = _strict_decision(
        summary_rows,
        correctness_rows,
        paired_rows,
        penalty=float(frozen["selected_penalty"]),
        regime_count=int(marker["regime_count"]),
        protocol_complete=protocol_complete,
    )
    decision["frozen_penalty_config_hash"] = frozen["frozen_config_hash"]
    decision_path = output / "strict_solver_decision.json"
    _write_json(decision_path, decision)
    marker["decision_sha256"] = sha256_file(decision_path)
    marker["resummarized_from_immutable_raw"] = True
    _write_json(marker_path, marker)
    return {
        "summary": summary_path,
        "correctness": correctness_path,
        "paired": paired_path,
        "decision": decision_path,
        "marker": marker_path,
        "decision_data": decision,
    }


def audit_accuracy_reference_evidence(
    output_dir: Path | str = ACCURACY_DIR,
) -> dict[str, Any]:
    """Audit the existing temporal/spatial Greek evidence before any rerun."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "results" / "07_method_extensions" / "03_greek_audit"
    required = {
        "temporal": source / "temporal_convergence.csv",
        "temporal_summary": source / "method_summary.csv",
        "temporal_decision": source / "greek_decision.json",
        "spatial": source / "spatial_convergence.csv",
        "spatial_summary": source / "spatial_grid_summary.csv",
        "spatial_decision": source / "spatial_greek_decision.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"accuracy-reference evidence is incomplete: {missing}")

    temporal = _read_csv(required["temporal"])
    temporal_summary = _read_csv(required["temporal_summary"])
    spatial = _read_csv(required["spatial"])
    spatial_summary = _read_csv(required["spatial_summary"])
    temporal_decision = _load_json(required["temporal_decision"])
    spatial_decision = _load_json(required["spatial_decision"])

    temporal_methods = sorted({row["method"] for row in temporal})
    temporal_levels = sorted({int(row["N"]) for row in temporal})
    temporal_regimes = sorted({row["regime_id"] for row in temporal})
    spatial_grids = sorted({row["grid"] for row in spatial})
    spatial_levels = sorted({int(row["M"]) for row in spatial})
    spatial_regimes = sorted({row["regime_id"] for row in spatial})
    expected_temporal_methods = sorted(
        [
            "cn_quadratic",
            "rannacher_cn_quadratic",
            "dirk_lstable_quadratic",
            "lobatto_iiic_penalty_quadratic",
        ]
    )
    completeness = {
        "temporal_row_count_ok": len(temporal) == 12 * 4 * 4,
        "temporal_regime_count_ok": len(temporal_regimes) == 12,
        "temporal_methods_ok": temporal_methods == expected_temporal_methods,
        "temporal_levels_ok": temporal_levels == [60, 120, 240, 480],
        "spatial_row_count_ok": len(spatial) == 12 * 2 * 3,
        "spatial_regime_count_ok": spatial_regimes == temporal_regimes,
        "spatial_grids_ok": spatial_grids
        == ["sinh_strike_concentrated", "uniform"],
        "spatial_levels_ok": spatial_levels == [120, 240, 480],
        "selected_time_integrator_ok": temporal_decision.get("best_temporal_method")
        == "dirk_lstable_quadratic",
        "selected_grid_ok": spatial_decision.get("selected_spatial_grid")
        == "sinh_strike_concentrated",
        "stable_mask_status_ok": spatial_decision.get("status")
        == "UNBLOCK_GAMMA_ON_STABLE_MASK",
    }
    selected_temporal = [
        row for row in temporal if row["method"] == "dirk_lstable_quadratic"
    ]
    selected_spatial = [
        row for row in spatial if row["grid"] == "sinh_strike_concentrated"
    ]
    completeness["selected_temporal_all_converged"] = all(
        _as_bool(row["converged"]) for row in selected_temporal
    )
    completeness["selected_spatial_all_converged"] = all(
        _as_bool(row["converged"]) for row in selected_spatial
    )
    all_complete = all(completeness.values())

    poster_temporal = [
        {
            "method": row["method"],
            "stable_second_order_fraction": float(row["stable_second_order_fraction"]),
            "median_value_max_error": float(row["median_value_max_error"]),
            "median_delta_max_error": float(row["median_delta_max_error"]),
            "median_gamma_max_error": float(row["median_gamma_max_error"]),
            "max_boundary_abs_error": float(row["max_boundary_abs_error"]),
            "median_runtime_seconds": float(row["median_runtime_seconds"]),
            "all_converged": _as_bool(row["all_converged"]),
        }
        for row in temporal_summary
    ]
    poster_spatial = [
        {
            "grid": row["grid"],
            "stable_second_order_fraction": float(row["stable_second_order_fraction"]),
            "boundary_within_local_cell_fraction": float(
                row["boundary_within_local_cell_fraction"]
            ),
            "median_delta_max_error": float(row["median_delta_max_error"]),
            "median_gamma_max_error": float(row["median_gamma_max_error"]),
            "median_runtime_seconds": float(row["median_runtime_seconds"]),
            "all_converged": _as_bool(row["all_converged"]),
        }
        for row in spatial_summary
    ]
    temporal_path = output / "poster_temporal_summary.csv"
    spatial_path = output / "poster_spatial_summary.csv"
    _write_csv(temporal_path, poster_temporal)
    _write_csv(spatial_path, poster_spatial)
    decision = {
        "status": "REUSE_EXISTING_REFERENCE_EVIDENCE" if all_complete else "RERUN_REQUIRED",
        "benchmark_role": "High-Accuracy Numerical Reference",
        "selected_time_integrator": temporal_decision.get("best_temporal_method"),
        "selected_spatial_grid": spatial_decision.get("selected_spatial_grid"),
        "gamma_scope": "validated stable mask only",
        "audit_regime_count": len(temporal_regimes),
        "completeness_checks": completeness,
        "source_hashes": {
            name: sha256_file(path) for name, path in required.items()
        },
        "reference_limit": (
            "Fine DIRK+sinh refinement is an internal numerical reference, not an "
            "analytic American-option theorem or a speed benchmark."
        ),
        "rerun_recommendation": (
            "No rerun needed for the poster evidence panel"
            if all_complete
            else "Repair the failed evidence checks before poster use"
        ),
    }
    decision_path = output / "accuracy_reference_audit.json"
    _write_json(decision_path, decision)
    return {
        "temporal_summary": temporal_path,
        "spatial_summary": spatial_path,
        "decision": decision_path,
        "decision_data": decision,
    }


def synthesize_poster_comparison() -> dict[str, Path]:
    """Create poster-ready tables, figures, and an integrated decision report."""

    strict_decision_path = STRICT_DIR / "strict_solver_decision.json"
    accuracy_decision_path = ACCURACY_DIR / "accuracy_reference_audit.json"
    if not strict_decision_path.exists():
        raise FileNotFoundError("run Experiment 62 strict benchmark first")
    if not accuracy_decision_path.exists():
        raise FileNotFoundError("run Experiment 63 accuracy audit first")
    strict = _load_json(strict_decision_path)
    accuracy = _load_json(accuracy_decision_path)
    runtime = _read_csv(STRICT_DIR / "runtime_summary.csv")
    correctness = _read_csv(STRICT_DIR / "correctness_summary.csv")
    temporal = _read_csv(ACCURACY_DIR / "poster_temporal_summary.csv")
    spatial = _read_csv(ACCURACY_DIR / "poster_spatial_summary.csv")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _make_figures(figures_dir, runtime, temporal, spatial)
    integrated = {
        "strict_solver_status": strict["status"],
        "penalty_newton_status": strict["penalty_newton_status"],
        "accuracy_reference_status": accuracy["status"],
        "benchmark_roles": BENCHMARK_ROLES,
        "poster_headline_solver": "projected_lu_single",
        "poster_accuracy_reference": "dirk_policy_sinh",
        "claim_rules": {
            "psor": "Basic / Original Classical Benchmark",
            "policy": "Strengthened Classical Benchmark 1",
            "projected_lu": "Strengthened Classical Benchmark 2",
            "penalty_newton": "Candidate or failed comparator; never promoted on speed alone",
        },
        "source_decisions": {
            "strict": str(strict_decision_path.relative_to(PROJECT_ROOT)),
            "accuracy": str(accuracy_decision_path.relative_to(PROJECT_ROOT)),
        },
    }
    decision_path = RESULTS_DIR / "method_decision.json"
    _write_json(decision_path, integrated)
    technical_path = REPORTS_DIR / "poster_evidence_report.md"
    chinese_path = REPORTS_DIR / "poster_evidence_结论_CN.md"
    technical_path.write_text(
        _technical_report(strict, accuracy, runtime, correctness, temporal, spatial),
        encoding="utf-8",
    )
    chinese_path.write_text(
        _chinese_report(strict, accuracy, runtime, correctness), encoding="utf-8"
    )
    manifest_path = RESULTS_DIR / "figure_manifest.json"
    _write_json(
        manifest_path,
        {name: str(path.relative_to(PROJECT_ROOT)) for name, path in figure_paths.items()},
    )
    return {
        "decision": decision_path,
        "technical_report": technical_path,
        "chinese_report": chinese_path,
        "figure_manifest": manifest_path,
    }


def _required_protocol_inputs() -> tuple[Path, ...]:
    return (
        DATASET_DIR / "regime_manifest.csv",
        DATASET_DIR / "split_assignment.csv",
        PROJECT_ROOT / "results" / "13_projected_lu" / "method_decision.json",
        PROJECT_ROOT
        / "results"
        / "13_projected_lu"
        / "02_heldout"
        / "runtime_samples.csv",
        PROJECT_ROOT
        / "results"
        / "13_projected_lu"
        / "02_heldout"
        / "correctness_and_structure.csv",
        PROJECT_ROOT
        / "results"
        / "07_method_extensions"
        / "03_greek_audit"
        / "temporal_convergence.csv",
        PROJECT_ROOT
        / "results"
        / "07_method_extensions"
        / "03_greek_audit"
        / "spatial_convergence.csv",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "american_lcp.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "penalty_newton.py",
    )


def _solve_arm(config: Any, arm: str, penalty: float) -> AmericanLCPResult:
    selected = (
        replace(config, penalty=penalty, penalty_newton_max_iter=100)
        if arm == "penalty_newton"
        else config
    )
    if arm == "projected_lu_single":
        return american_cn_lcp_price(selected, lcp_solver="projected_lu_single")
    return american_cn_lcp_price(
        selected,
        lcp_solver=arm,  # type: ignore[arg-type]
        initializer="previous_slice",
    )


def _penalty_validation_summary(
    rows: list[dict[str, Any]], penalties: tuple[float, ...], regime_count: int
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for penalty in penalties:
        selected = [row for row in rows if float(row["penalty"]) == penalty]
        summaries.append(
            {
                "penalty": penalty,
                "regime_count": regime_count,
                "certified_regime_count": sum(
                    bool(row["numerically_certified"]) for row in selected
                ),
                "all_numerically_certified": bool(selected)
                and all(bool(row["numerically_certified"]) for row in selected),
                "all_converged": bool(selected)
                and all(bool(row["converged"]) for row in selected),
                "max_normalized_lcp_residual": max(
                    (float(row["max_normalized_lcp_residual"]) for row in selected),
                    default=float("inf"),
                ),
                "max_normalized_obstacle_violation": max(
                    (float(row["max_normalized_obstacle_violation"]) for row in selected),
                    default=float("inf"),
                ),
                "max_abs_trajectory_difference": max(
                    (float(row["max_abs_trajectory_difference"]) for row in selected),
                    default=float("inf"),
                ),
            }
        )
    return summaries


def _runtime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm in STRICT_ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        times = np.array([float(row["measured_total_seconds"]) for row in selected])
        summaries.append(
            {
                "arm": arm,
                "benchmark_role": BENCHMARK_ROLES[arm],
                "sample_count": len(selected),
                "median_seconds": float(np.median(times)),
                "p95_seconds": float(np.percentile(times, 95)),
                "p99_seconds": float(np.percentile(times, 99)),
                "all_timed_runs_converged": all(_as_bool(row["converged"]) for row in selected),
                "median_iterations": float(
                    np.median([float(row["total_sweeps_or_iterations"]) for row in selected])
                ),
            }
        )
    return summaries


def _correctness_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm in STRICT_ARMS:
        selected = [row for row in rows if row["solver"] == arm]
        common_gate = [
            _as_bool(row["converged"])
            and float(row["max_normalized_lcp_residual"]) <= FROZEN_TOLERANCE
            and float(row["max_normalized_obstacle_violation"])
            <= FROZEN_OBSTACLE_TOLERANCE
            for row in selected
        ]
        summaries.append(
            {
                "arm": arm,
                "benchmark_role": BENCHMARK_ROLES[arm],
                "regime_count": len(selected),
                "common_gate_regime_count": sum(common_gate),
                "all_common_residual_gates_passed": bool(selected) and all(common_gate),
                "certified_regime_count": sum(
                    _as_bool(row["numerically_certified"]) for row in selected
                ),
                "all_numerically_certified": bool(selected)
                and all(_as_bool(row["numerically_certified"]) for row in selected),
                "max_normalized_lcp_residual": max(
                    (float(row["max_normalized_lcp_residual"]) for row in selected),
                    default=float("nan"),
                ),
                "max_normalized_obstacle_violation": max(
                    (float(row["max_normalized_obstacle_violation"]) for row in selected),
                    default=float("nan"),
                ),
                "max_abs_trajectory_difference": max(
                    (float(row["max_abs_trajectory_difference"]) for row in selected),
                    default=float("nan"),
                ),
            }
        )
    return summaries


def _paired_runtime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for regime_id in sorted({str(row["regime_id"]) for row in rows}):
        selected = [row for row in rows if row["regime_id"] == regime_id]
        medians = {
            arm: float(
                np.median(
                    [
                        float(row["measured_total_seconds"])
                        for row in selected
                        if row["arm"] == arm
                    ]
                )
            )
            for arm in STRICT_ARMS
        }
        metadata = selected[0]
        paired.append(
            {
                "regime_id": regime_id,
                "split": metadata["split"],
                "option_type": metadata["option_type"],
                "q": metadata["q"],
                "early_exercise_risk": not (
                    metadata["option_type"] == "call" and float(metadata["q"]) == 0.0
                ),
                **{f"{arm}_median_seconds": medians[arm] for arm in STRICT_ARMS},
                "policy_over_psor_ratio": medians["policy_iteration"] / medians["psor"],
                "projected_lu_over_policy_ratio": medians["projected_lu_single"]
                / medians["policy_iteration"],
                "penalty_over_policy_ratio": medians["penalty_newton"]
                / medians["policy_iteration"],
                "penalty_over_projected_lu_ratio": medians["penalty_newton"]
                / medians["projected_lu_single"],
            }
        )
    return paired


def _strict_decision(
    runtime: list[dict[str, Any]],
    correctness: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    penalty: float,
    regime_count: int,
    protocol_complete: bool,
) -> dict[str, Any]:
    runtime_by_arm = {row["arm"]: row for row in runtime}
    correctness_by_arm = {row["arm"]: row for row in correctness}
    strict_three_correct = all(
        bool(correctness_by_arm[arm]["all_common_residual_gates_passed"])
        for arm in ("psor", "policy_iteration", "projected_lu_single")
    )
    penalty_correct = bool(
        correctness_by_arm["penalty_newton"]["all_common_residual_gates_passed"]
    )
    penalty_ratio_policy = float(
        np.median([float(row["penalty_over_policy_ratio"]) for row in paired])
    )
    penalty_ratio_lu = float(
        np.median([float(row["penalty_over_projected_lu_ratio"]) for row in paired])
    )
    penalty_p95 = float(runtime_by_arm["penalty_newton"]["p95_seconds"])
    policy_p95 = float(runtime_by_arm["policy_iteration"]["p95_seconds"])
    projected_p95 = float(runtime_by_arm["projected_lu_single"]["p95_seconds"])
    policy_ratio_psor = float(
        np.median([float(row["policy_over_psor_ratio"]) for row in paired])
    )
    projected_ratio_policy = float(
        np.median([float(row["projected_lu_over_policy_ratio"]) for row in paired])
    )
    early = [row for row in paired if _as_bool(row["early_exercise_risk"])]
    early_penalty_ratio_policy = float(
        np.median([float(row["penalty_over_policy_ratio"]) for row in early])
    )
    early_projected_ratio_policy = float(
        np.median([float(row["projected_lu_over_policy_ratio"]) for row in early])
    )
    if not protocol_complete:
        status = "SMOKE_ONLY"
    elif not strict_three_correct:
        status = "STOP_STRICT_REFERENCE_REGRESSION"
    else:
        status = "STRICT_THREE_CONFIRMED"

    if not penalty_correct:
        penalty_status = "FAST_BUT_FAILED_CORRECTNESS" if penalty_ratio_policy < 1.0 else "FAILED_CORRECTNESS"
    elif penalty_ratio_lu <= POSTER_SPEED_GATE and penalty_p95 <= projected_p95:
        penalty_status = "CANDIDATE_BEATS_PROJECTED_LU"
    elif penalty_ratio_policy <= POSTER_SPEED_GATE and penalty_p95 <= policy_p95:
        penalty_status = "CANDIDATE_BEATS_POLICY_NOT_PROJECTED_LU"
    else:
        penalty_status = "CORRECT_BUT_NO_20_PERCENT_SPEED_VALUE"
    return {
        "status": status,
        "protocol_complete": protocol_complete,
        "regime_count": regime_count,
        "benchmark_roles": BENCHMARK_ROLES,
        "selected_penalty": penalty,
        "penalty_newton_status": penalty_status,
        "penalty_all_common_residual_gates_passed": penalty_correct,
        "penalty_all_strong_policy_match_certified": bool(
            correctness_by_arm["penalty_newton"]["all_numerically_certified"]
        ),
        "penalty_common_gate_regime_count": int(
            correctness_by_arm["penalty_newton"]["common_gate_regime_count"]
        ),
        "penalty_certified_regime_count": int(
            correctness_by_arm["penalty_newton"]["certified_regime_count"]
        ),
        "penalty_max_normalized_lcp_residual": float(
            correctness_by_arm["penalty_newton"]["max_normalized_lcp_residual"]
        ),
        "penalty_max_abs_trajectory_difference": float(
            correctness_by_arm["penalty_newton"]["max_abs_trajectory_difference"]
        ),
        "paired_median_penalty_over_policy": penalty_ratio_policy,
        "paired_median_penalty_over_projected_lu": penalty_ratio_lu,
        "paired_median_policy_over_psor": policy_ratio_psor,
        "paired_median_projected_lu_over_policy": projected_ratio_policy,
        "early_exercise_regime_count": len(early),
        "early_exercise_paired_median_penalty_over_policy": early_penalty_ratio_policy,
        "early_exercise_paired_median_projected_lu_over_policy": early_projected_ratio_policy,
        "speed_gate": POSTER_SPEED_GATE,
        "promotion_rule": "Penalty/Newton cannot be promoted on timing unless every common correctness gate passes.",
    }


def _make_figures(
    output: Path,
    runtime: list[dict[str, str]],
    temporal: list[dict[str, str]],
    spatial: list[dict[str, str]],
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    labels = {
        "psor": "CN+PSOR",
        "policy_iteration": "CN+Policy",
        "projected_lu_single": "CN+Projected LU",
        "penalty_newton": "CN+Penalty/Newton*",
    }
    colors = ["#7f8c8d", "#2878b5", "#2ca25f", "#c44e52"]
    arms = [row["arm"] for row in runtime]
    medians = [float(row["median_seconds"]) for row in runtime]
    p95s = [float(row["p95_seconds"]) for row in runtime]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - 0.18, medians, 0.36, label="Median", color=colors)
    ax.bar(x + 0.18, p95s, 0.36, label="P95", color=colors, alpha=0.45)
    ax.set_yscale("log")
    ax.set_ylabel("End-to-end time (s, log scale)")
    ax.set_xticks(x, [labels[arm] for arm in arms], rotation=12, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Same CN-LCP: 67 held-out regimes")
    for index, value in enumerate(p95s):
        ax.text(index + 0.18, value * 1.07, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "* Penalty/Newton failed the common LCP correctness gate (40/67).",
        ha="center",
        fontsize=9,
        color="#9b2d30",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    runtime_path = output / "strict_solver_runtime_log.png"
    fig.savefig(runtime_path, dpi=220)
    plt.close(fig)

    temporal_names = {
        "cn_quadratic": "CN",
        "rannacher_cn_quadratic": "Rannacher CN",
        "dirk_lstable_quadratic": "L-stable DIRK",
        "lobatto_iiic_penalty_quadratic": "Lobatto IIIC",
    }
    method_labels = [temporal_names.get(row["method"], row["method"]) for row in temporal]
    gamma = [float(row["median_gamma_max_error"]) for row in temporal]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(np.arange(len(gamma)), gamma, color="#4c72b0")
    ax.set_yscale("log")
    ax.set_ylabel("Median Gamma max error (log scale)")
    ax.set_xticks(np.arange(len(gamma)), method_labels, rotation=15, ha="right")
    ax.set_title("Temporal Gamma audit (validated mask)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    temporal_path = output / "temporal_gamma_reference.png"
    fig.savefig(temporal_path, dpi=220)
    plt.close(fig)

    grid_labels = [row["grid"].replace("_strike_concentrated", "") for row in spatial]
    spatial_gamma = [float(row["median_gamma_max_error"]) for row in spatial]
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.bar(np.arange(len(spatial_gamma)), spatial_gamma, color=["#9ecae1", "#2ca25f"])
    ax.set_yscale("log")
    ax.set_ylabel("Median Gamma max error (log scale)")
    ax.set_xticks(np.arange(len(spatial_gamma)), grid_labels)
    ax.set_title("Spatial-grid Gamma audit (validated mask)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    spatial_path = output / "spatial_grid_gamma_reference.png"
    fig.savefig(spatial_path, dpi=220)
    plt.close(fig)
    return {
        "strict_runtime": runtime_path,
        "temporal_gamma": temporal_path,
        "spatial_gamma": spatial_path,
    }


def _technical_report(
    strict: dict[str, Any],
    accuracy: dict[str, Any],
    runtime: list[dict[str, str]],
    correctness: list[dict[str, str]],
    temporal: list[dict[str, str]],
    spatial: list[dict[str, str]],
) -> str:
    lines = [
        "# Unified Poster Evidence Report",
        "",
        "## Benchmark roles",
        "",
        "- CN+PSOR: **Basic / Original Classical Benchmark**.",
        "- CN+Policy Iteration: **Strengthened Classical Benchmark 1**.",
        "- CN+Projected LU: **Strengthened Classical Benchmark 2**.",
        "- DIRK+Policy+sinh: **High-Accuracy Numerical Reference**.",
        "- Penalty/Newton: candidate comparator; correctness is judged before speed.",
        "",
        "## Strict same-CN-LCP comparison",
        "",
        "| Arm | Median (s) | P95 (s) | Common residual gate | Strong Policy-match certification | Max normalized LCP residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    correctness_by = {row["arm"]: row for row in correctness}
    for row in runtime:
        correct = correctness_by[row["arm"]]
        lines.append(
            f"| {row['arm']} | {float(row['median_seconds']):.6g} | "
            f"{float(row['p95_seconds']):.6g} | {correct['common_gate_regime_count']}/"
            f"{correct['regime_count']} | {correct['certified_regime_count']}/"
            f"{correct['regime_count']} | {float(correct['max_normalized_lcp_residual']):.6g} |"
        )
    lines.extend(
        [
            "",
            f"Penalty/Newton decision: **{strict['penalty_newton_status']}**.",
            "",
            f"Paired median Policy/PSOR ratio: {strict['paired_median_policy_over_psor']:.4f}; "
            f"Projected-LU/Policy ratio: {strict['paired_median_projected_lu_over_policy']:.4f}; "
            f"Penalty/Policy ratio: {strict['paired_median_penalty_over_policy']:.4f}.",
            f"For the {strict['early_exercise_regime_count']} put/dividend-call regimes, "
            f"Projected-LU/Policy is {strict['early_exercise_paired_median_projected_lu_over_policy']:.4f} "
            f"and Penalty/Policy is {strict['early_exercise_paired_median_penalty_over_policy']:.4f}.",
            "",
            "## Accuracy-reference evidence",
            "",
            f"Audit decision: **{accuracy['status']}**.",
            "",
            "Temporal and spatial evidence are reported separately from online solver timing. "
            "Gamma claims remain restricted to the validated stable mask.",
            "",
            "### Temporal summary",
            "",
            "| Method | Stable fraction | Median Gamma max error |",
            "|---|---:|---:|",
        ]
    )
    for row in temporal:
        lines.append(
            f"| {row['method']} | {float(row['stable_second_order_fraction']):.3f} | "
            f"{float(row['median_gamma_max_error']):.6g} |"
        )
    lines.extend(["", "### Spatial summary", "", "| Grid | Stable fraction | Median Gamma max error |", "|---|---:|---:|"])
    for row in spatial:
        lines.append(
            f"| {row['grid']} | {float(row['stable_second_order_fraction']):.3f} | "
            f"{float(row['median_gamma_max_error']):.6g} |"
        )
    lines.extend(["", f"Reference limitation: {accuracy['reference_limit']}"])
    return "\n".join(lines) + "\n"


def _chinese_report(
    strict: dict[str, Any],
    accuracy: dict[str, Any],
    runtime: list[dict[str, str]],
    correctness: list[dict[str, str]],
) -> str:
    runtime_by = {row["arm"]: row for row in runtime}
    correct_by = {row["arm"]: row for row in correctness}
    return (
        "# Poster 统一实验结论\n\n"
        "- CN+PSOR 是 Basic / Original Classical Benchmark。\n"
        "- CN+Policy Iteration 是 Strengthened Classical Benchmark 1。\n"
        "- CN+Projected LU 是 Strengthened Classical Benchmark 2。\n"
        "- DIRK+Policy+sinh 是 High-Accuracy Numerical Reference。\n\n"
        f"Strict comparison 状态：**{strict['status']}**。Penalty/Newton 状态："
        f"**{strict['penalty_newton_status']}**。\n\n"
        f"Projected LU median={float(runtime_by['projected_lu_single']['median_seconds']):.6g}s，"
        f"Policy median={float(runtime_by['policy_iteration']['median_seconds']):.6g}s，"
        f"PSOR median={float(runtime_by['psor']['median_seconds']):.6g}s。\n\n"
        f"Paired median ratios：Policy/PSOR={strict['paired_median_policy_over_psor']:.4f}，"
        f"Projected-LU/Policy={strict['paired_median_projected_lu_over_policy']:.4f}，"
        f"Penalty/Policy={strict['paired_median_penalty_over_policy']:.4f}。"
        f"在 {strict['early_exercise_regime_count']} 个真正有提前行权风险的 regimes 中，"
        f"Penalty/Policy={strict['early_exercise_paired_median_penalty_over_policy']:.4f}。\n\n"
        f"Penalty/Newton 通过共同 residual gate "
        f"{correct_by['penalty_newton']['common_gate_regime_count']}/"
        f"{correct_by['penalty_newton']['regime_count']} regimes；其最大 normalized LCP residual 为 "
        f"{float(correct_by['penalty_newton']['max_normalized_lcp_residual']):.6g}。"
        "无论速度如何，只有全部共同正确性 gate 通过才可提升为 benchmark。\n\n"
        f"Accuracy-reference audit：**{accuracy['status']}**；Gamma 仅允许在 validated stable mask 上报告。\n"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
