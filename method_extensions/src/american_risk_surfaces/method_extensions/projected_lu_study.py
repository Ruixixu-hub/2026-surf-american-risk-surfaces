"""Protocol, validation, benchmark, and reporting for projected LU."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta,
    finite_difference_gamma,
)
from american_risk_surfaces.method_extensions.protocol import (
    AUDIT_REGIME_IDS,
    DATASET_DIR,
    environment_manifest,
    sha256_file,
)
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    AmericanLCPResult,
    american_cn_lcp_price,
    as_legacy_cn_psor_result,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.black_scholes import (
    call_payoff,
    european_call_price,
    put_payoff,
)
from american_risk_surfaces.solvers.lcp import TridiagonalLCP
from american_risk_surfaces.solvers.projected_lu import (
    audit_projected_lu_eligibility,
    factorize_projected_lu,
    reconstruct_projected_lu_matrix,
)
from american_risk_surfaces.workspace import portable_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "13_projected_lu"
REPORTS_DIR = PROJECT_ROOT / "reports" / "15_projected_lu"
PROTOCOL_DIR = RESULTS_DIR / "00_protocol"
VALIDATION_DIR = RESULTS_DIR / "01_validation"
HELDOUT_DIR = RESULTS_DIR / "02_heldout"

FROZEN_TOLERANCE = 1e-12
FROZEN_OBSTACLE_TOLERANCE = 1e-12
VALUE_MATCH_TOLERANCE = 1e-9
CONTACT_TOLERANCE = 1e-10
PIVOT_TOLERANCE = 1e-14
SIGN_TOLERANCE = 1e-15
RECONSTRUCTION_TOLERANCE = 1e-13
SPEED_RATIO_GATE = 0.8
HISTORICAL_MEDIAN_TARGET = 0.012785166827961803
HISTORICAL_P95_TARGET = 0.02099762330763042
BOOTSTRAP_SEED = 20260814
BENCHMARK_ORDER_SEED = 20260815


def run_protocol_and_eligibility(
    output_dir: Path | str = PROTOCOL_DIR,
) -> dict[str, Any]:
    """Freeze inputs and audit matrix/contact eligibility for all regimes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regimes = load_regimes()
    input_paths = (
        DATASET_DIR / "regime_manifest.csv",
        DATASET_DIR / "split_assignment.csv",
        PROJECT_ROOT
        / "results"
        / "07_method_extensions"
        / "00_protocol"
        / "tolerance_decision.json",
        PROJECT_ROOT
        / "results"
        / "07_method_extensions"
        / "02_warmstart"
        / "runtime_summary.csv",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "american_lcp.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "projected_lu.py",
    )
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required projected-LU inputs are missing: {missing}")

    rows: list[dict[str, Any]] = []
    for regime in regimes:
        config = config_from_regime(regime)
        payoff = _payoff(config)
        first_system = assemble_american_cn_lcp_step(config, payoff, 1)
        factorization = factorize_projected_lu(
            first_system,
            directions=("lu", "ul"),
            pivot_tolerance=PIVOT_TOLERANCE,
        )
        matrix_eligibility = audit_projected_lu_eligibility(
            first_system,
            pivot_tolerance=PIVOT_TOLERANCE,
            sign_tolerance=SIGN_TOLERANCE,
            contact_tolerance=CONTACT_TOLERANCE,
        )
        dense = _dense_matrix(first_system)
        lu_error = float(
            np.max(
                np.abs(
                    reconstruct_projected_lu_matrix(
                        first_system, factorization, direction="lu"
                    )
                    - dense
                )
            )
        )
        ul_error = float(
            np.max(
                np.abs(
                    reconstruct_projected_lu_matrix(
                        first_system, factorization, direction="ul"
                    )
                    - dense
                )
            )
        )

        contact_audited = regime["split"] in {"train", "validation"}
        all_expected_contact: bool | str = "not_audited_heldout"
        max_contact_components: int | str = "not_audited_heldout"
        all_step_theorem_eligible: bool | str = "not_audited_heldout"
        if contact_audited:
            policy = american_cn_lcp_price(
                config, lcp_solver="policy_iteration", initializer="previous_slice"
            )
            step_expected: list[bool] = []
            step_components: list[int] = []
            step_theorem: list[bool] = []
            for step_index in range(1, config.N + 1):
                system = assemble_american_cn_lcp_step(
                    config, policy.value_grid[step_index - 1], step_index
                )
                eligibility = audit_projected_lu_eligibility(
                    system,
                    policy.value_grid[step_index, 1:-1],
                    option_type=config.option_type,
                    pivot_tolerance=PIVOT_TOLERANCE,
                    sign_tolerance=SIGN_TOLERANCE,
                    contact_tolerance=CONTACT_TOLERANCE,
                )
                step_expected.append(eligibility.expected_contact_geometry is not False)
                step_components.append(eligibility.contact_components or 0)
                step_theorem.append(eligibility.theorem_eligible)
            all_expected_contact = all(step_expected)
            max_contact_components = max(step_components)
            all_step_theorem_eligible = all(step_theorem)

        rows.append(
            {
                "regime_id": regime["regime_id"],
                "split": regime["split"],
                "option_type": regime["option_type"],
                "T": float(regime["T"]),
                "sigma": float(regime["sigma"]),
                "r": float(regime["r"]),
                "q": float(regime["q"]),
                "positive_diagonal": matrix_eligibility.positive_diagonal,
                "nonpositive_offdiagonals": matrix_eligibility.nonpositive_offdiagonals,
                "strict_diagonal_dominance": matrix_eligibility.strictly_diagonally_dominant,
                "positive_lu_pivots": matrix_eligibility.positive_lu_pivots,
                "positive_ul_pivots": matrix_eligibility.positive_ul_pivots,
                "m_matrix_sufficient_conditions": matrix_eligibility.m_matrix_sufficient_conditions,
                "lu_reconstruction_max_abs": lu_error,
                "ul_reconstruction_max_abs": ul_error,
                "contact_topology_audited": contact_audited,
                "all_expected_contact_geometry": all_expected_contact,
                "max_contact_components": max_contact_components,
                "all_step_theorem_eligible": all_step_theorem_eligible,
                "matrix_reasons": "|".join(matrix_eligibility.reasons),
            }
        )

    eligibility_path = output / "matrix_and_contact_eligibility.csv"
    _write_csv(eligibility_path, rows)
    counts = _eligibility_counts(rows)
    protocol = {
        "protocol_version": "projected_lu_v1",
        "regime_count": len(regimes),
        "frozen_grid": {"K": 1.0, "Smax": 4.0, "M": 120, "N": 120},
        "frozen_lcp_tolerance": FROZEN_TOLERANCE,
        "frozen_obstacle_tolerance": FROZEN_OBSTACLE_TOLERANCE,
        "value_match_tolerance_K": VALUE_MATCH_TOLERANCE,
        "contact_tolerance_value_scale": CONTACT_TOLERANCE,
        "pivot_tolerance": PIVOT_TOLERANCE,
        "sign_tolerance": SIGN_TOLERANCE,
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "speed_ratio_gate": SPEED_RATIO_GATE,
        "audit_regime_ids": list(AUDIT_REGIME_IDS),
        "heldout_contact_policy": "matrix_only_until_frozen_solver_selection",
        "stencil_policy": "frozen_central_difference_no_fitting_or_upwinding",
        "input_hashes": {
            portable_path(path): sha256_file(path) for path in input_paths
        },
        "environment": _environment_with_blas(),
        "eligibility_counts": counts,
    }
    protocol["protocol_hash"] = _json_hash(protocol)
    protocol_path = output / "protocol_manifest.json"
    _write_json(protocol_path, protocol)
    decision = {
        "status": "PROTOCOL_READY"
        if len(regimes) == 288
        and counts["positive_diagonal"] == 288
        and counts["strict_diagonal_dominance"] == 288
        and counts["reconstruction_pass"] == 288
        else "STOP_PROTOCOL",
        "regime_count": len(regimes),
        **counts,
        "protocol_hash": protocol["protocol_hash"],
    }
    decision_path = output / "eligibility_decision.json"
    _write_json(decision_path, decision)
    return {
        "protocol": protocol_path,
        "eligibility": eligibility_path,
        "decision": decision_path,
        "decision_data": decision,
    }


def run_validation_and_freeze(
    output_dir: Path | str = VALIDATION_DIR,
) -> dict[str, Any]:
    """Run single/double projected-LU validation and freeze one global mode."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = _require_protocol()
    eligibility_rows = _read_csv(PROTOCOL_DIR / "matrix_and_contact_eligibility.csv")
    regimes = {row["regime_id"]: row for row in load_regimes()}
    edge_ids = {
        row["regime_id"]
        for row in eligibility_rows
        if row["split"] == "train"
        and row["nonpositive_offdiagonals"].lower() == "false"
    }
    selected_ids = set(AUDIT_REGIME_IDS) | edge_ids | {
        regime_id for regime_id, row in regimes.items() if row["split"] == "validation"
    }
    selected = [regimes[regime_id] for regime_id in sorted(selected_ids)]

    metric_rows: list[dict[str, Any]] = []
    for regime in selected:
        config = config_from_regime(regime)
        policy = american_cn_lcp_price(
            config, lcp_solver="policy_iteration", initializer="previous_slice"
        )
        for solver in ("projected_lu_single", "projected_lu_double"):
            candidate = american_cn_lcp_price(config, lcp_solver=solver)
            metric_rows.append(
                trajectory_metrics(regime, solver, candidate, policy)
            )

    metrics_path = output / "validation_metrics.csv"
    _write_csv(metrics_path, metric_rows)
    by_solver = {
        solver: [row for row in metric_rows if row["solver"] == solver]
        for solver in ("projected_lu_single", "projected_lu_double")
    }
    single_pass = all(bool(row["numerically_certified"]) for row in by_solver["projected_lu_single"])
    double_pass = all(bool(row["numerically_certified"]) for row in by_solver["projected_lu_double"])
    if single_pass:
        selected_solver = "projected_lu_single"
        status = "PROCEED_HELDOUT_SINGLE"
    elif double_pass:
        selected_solver = "projected_lu_double"
        status = "PROCEED_HELDOUT_DOUBLE"
    else:
        selected_solver = None
        status = "STOP_CORRECTNESS"

    summary_rows = []
    for solver, rows in by_solver.items():
        summary_rows.append(
            {
                "solver": solver,
                "case_count": len(rows),
                "all_numerically_certified": all(
                    bool(row["numerically_certified"]) for row in rows
                ),
                "all_converged": all(bool(row["converged"]) for row in rows),
                "max_abs_trajectory_difference": max(
                    float(row["max_abs_trajectory_difference"]) for row in rows
                ),
                "max_normalized_lcp_residual": max(
                    float(row["max_normalized_lcp_residual"]) for row in rows
                ),
                "all_theorem_eligible": all(
                    bool(row["all_steps_theorem_eligible"]) for row in rows
                ),
            }
        )
    summary_path = output / "validation_summary.csv"
    _write_csv(summary_path, summary_rows)
    frozen = {
        "status": status,
        "selected_solver": selected_solver,
        "selection_rule": "single_if_all_correct_else_double_if_all_correct",
        "selection_does_not_use_validation_timing": True,
        "evaluated_case_count": len(selected),
        "audit_case_count": len(AUDIT_REGIME_IDS),
        "non_z_train_edge_count": len(edge_ids),
        "validation_case_count": sum(row["split"] == "validation" for row in selected),
        "frozen_lcp_tolerance": FROZEN_TOLERANCE,
        "frozen_obstacle_tolerance": FROZEN_OBSTACLE_TOLERANCE,
        "value_match_tolerance_K": VALUE_MATCH_TOLERANCE,
        "protocol_hash": protocol["protocol_hash"],
        "validation_metrics_sha256": sha256_file(metrics_path),
    }
    frozen["frozen_config_hash"] = _json_hash(frozen)
    frozen_path = output / "frozen_solver_config.json"
    _write_json(frozen_path, frozen)
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "frozen": frozen_path,
        "frozen_data": frozen,
    }


def run_heldout_benchmark(
    output_dir: Path | str = HELDOUT_DIR,
    *,
    warmups: int = 5,
    repeats: int = 30,
    regime_limit: int | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Run the one-shot matched held-out benchmark."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "heldout_complete.marker.json"
    if marker_path.exists() and not allow_existing:
        raise RuntimeError("held-out benchmark is already complete; refusing to overwrite it")
    frozen = _require_frozen_solver()
    selected_solver = frozen.get("selected_solver")
    if selected_solver not in {"projected_lu_single", "projected_lu_double"}:
        raise RuntimeError("validation did not freeze a projected-LU solver")
    regimes = [
        row
        for row in load_regimes()
        if row["split"] in {"test", "stress_holdout"}
    ]
    regimes.sort(key=lambda row: row["regime_id"])
    if regime_limit is not None:
        regimes = regimes[:regime_limit]
    arms = ("psor", "policy_iteration", str(selected_solver))
    rng = np.random.default_rng(BENCHMARK_ORDER_SEED)
    runtime_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []

    benchmark_started = perf_counter_ns()
    for regime_index, regime in enumerate(regimes, start=1):
        config = config_from_regime(regime)
        policy_reference = american_cn_lcp_price(
            config, lcp_solver="policy_iteration", initializer="previous_slice"
        )
        for arm in arms:
            for _ in range(warmups):
                american_cn_lcp_price(config, lcp_solver=arm)  # type: ignore[arg-type]

        first_results: dict[str, AmericanLCPResult] = {}
        for repeat in range(repeats):
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                wall_started = perf_counter_ns()
                result = american_cn_lcp_price(
                    config, lcp_solver=arm  # type: ignore[arg-type]
                )
                measured_seconds = (perf_counter_ns() - wall_started) * 1e-9
                first_results.setdefault(arm, result)
                runtime_rows.append(
                    runtime_row(
                        regime,
                        arm,
                        repeat,
                        result,
                        measured_seconds,
                    )
                )
        for arm, result in first_results.items():
            accuracy_rows.append(
                trajectory_metrics(regime, arm, result, policy_reference)
            )
        elapsed_seconds = (perf_counter_ns() - benchmark_started) * 1e-9
        print(
            f"[{regime_index}/{len(regimes)}] {regime['regime_id']} complete "
            f"({elapsed_seconds:.1f}s elapsed)",
            flush=True,
        )

    runtime_path = output / "runtime_samples.csv"
    accuracy_path = output / "correctness_and_structure.csv"
    _write_csv(runtime_path, runtime_rows)
    _write_csv(accuracy_path, accuracy_rows)
    summary_rows, paired_rows, bootstrap = summarize_runtime(
        runtime_rows, selected_solver=str(selected_solver)
    )
    summary_path = output / "runtime_summary.csv"
    paired_path = output / "paired_regime_timing.csv"
    bootstrap_path = output / "paired_bootstrap.json"
    _write_csv(summary_path, summary_rows)
    _write_csv(paired_path, paired_rows)
    _write_json(bootstrap_path, bootstrap)
    decision = heldout_decision(
        regimes,
        accuracy_rows,
        runtime_rows,
        summary_rows,
        paired_rows,
        frozen,
        warmups=warmups,
        repeats=repeats,
        protocol_complete=(warmups == 5 and repeats == 30 and regime_limit is None),
    )
    decision_path = RESULTS_DIR / "method_decision.json"
    _write_json(decision_path, decision)
    marker = {
        "status": "COMPLETE",
        "frozen_config_hash": frozen["frozen_config_hash"],
        "runtime_sha256": sha256_file(runtime_path),
        "accuracy_sha256": sha256_file(accuracy_path),
        "decision_sha256": sha256_file(decision_path),
        "regime_count": len(regimes),
        "warmups": warmups,
        "repeats": repeats,
    }
    _write_json(marker_path, marker)
    return {
        "runtime": runtime_path,
        "accuracy": accuracy_path,
        "summary": summary_path,
        "paired": paired_path,
        "bootstrap": bootstrap_path,
        "decision": decision_path,
        "marker": marker_path,
        "decision_data": decision,
    }


def synthesize_projected_lu() -> dict[str, Path]:
    """Generate figures and the English/Chinese decision reports."""

    decision_path = RESULTS_DIR / "method_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError("run experiment 60 before experiment 61")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    eligibility = _read_csv(PROTOCOL_DIR / "matrix_and_contact_eligibility.csv")
    summary = _read_csv(HELDOUT_DIR / "runtime_summary.csv")
    paired = _read_csv(HELDOUT_DIR / "paired_regime_timing.csv")
    accuracy = _read_csv(HELDOUT_DIR / "correctness_and_structure.csv")
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _make_figures(figures_dir, eligibility, summary, paired)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    english_path = REPORTS_DIR / "projected_lu_technical_report.md"
    chinese_path = REPORTS_DIR / "projected_lu_结论_CN.md"
    english_path.write_text(
        _english_report(decision, summary, accuracy, figure_paths), encoding="utf-8"
    )
    chinese_path.write_text(
        _chinese_report(decision, summary, accuracy, figure_paths), encoding="utf-8"
    )
    manifest_path = RESULTS_DIR / "figure_manifest.json"
    _write_json(
        manifest_path,
        {name: str(path.relative_to(PROJECT_ROOT)) for name, path in figure_paths.items()},
    )
    return {
        "decision": decision_path,
        "english_report": english_path,
        "chinese_report": chinese_path,
        "figure_manifest": manifest_path,
    }


def trajectory_metrics(
    regime: dict[str, str],
    solver: str,
    result: AmericanLCPResult,
    policy: AmericanLCPResult,
) -> dict[str, Any]:
    """Compute full-trajectory strict-solver agreement and structure metrics."""

    error = result.value_grid - policy.value_grid
    absolute = np.abs(error)
    residuals = [step.residual for step in result.lcp_results]
    max_lcp = max(
        (residual.normalized_lcp_residual for residual in residuals), default=0.0
    )
    max_obstacle = max(
        (residual.normalized_obstacle_violation for residual in residuals), default=0.0
    )
    max_equation = max(
        (residual.normalized_equation_violation for residual in residuals), default=0.0
    )
    max_complementarity = max(
        (residual.normalized_complementarity for residual in residuals), default=0.0
    )
    theorem_flags: list[bool] = []
    active_components: list[int] = []
    active_disagreements: list[int] = []
    config = result.config
    for step_index in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(
            config, result.value_grid[step_index - 1], step_index
        )
        eligibility = audit_projected_lu_eligibility(
            system,
            result.value_grid[step_index, 1:-1],
            option_type=config.option_type,
            contact_tolerance=CONTACT_TOLERANCE,
        )
        theorem_flags.append(eligibility.theorem_eligible)
        active_components.append(eligibility.contact_components or 0)
        candidate_gap = result.value_grid[step_index, 1:-1] - system.obstacle
        policy_gap = policy.value_grid[step_index, 1:-1] - system.obstacle
        scale = max(1.0, float(np.max(np.abs(policy.value_grid[step_index, 1:-1]))))
        candidate_active = candidate_gap <= CONTACT_TOLERANCE * scale
        policy_active = policy_gap <= CONTACT_TOLERANCE * scale
        active_disagreements.append(int(np.count_nonzero(candidate_active != policy_active)))

    boundary_candidate = extract_boundary_curve(
        as_legacy_cn_psor_result(result), f"{regime['regime_id']}_{solver}"
    )
    boundary_policy = extract_boundary_curve(
        as_legacy_cn_psor_result(policy), f"{regime['regime_id']}_policy"
    )
    boundary_errors = [
        candidate.boundary_spot - target.boundary_spot
        for candidate, target in zip(boundary_candidate.points, boundary_policy.points)
        if candidate.boundary_found and target.boundary_found
    ]
    boundary_found_disagreement = sum(
        candidate.boundary_found != target.boundary_found
        for candidate, target in zip(boundary_candidate.points, boundary_policy.points)
    )
    delta_error = finite_difference_delta(result.spot_grid, result.value_grid) - finite_difference_delta(
        policy.spot_grid, policy.value_grid
    )
    gamma_error = finite_difference_gamma(result.spot_grid, result.value_grid) - finite_difference_gamma(
        policy.spot_grid, policy.value_grid
    )
    delta_finite = delta_error[np.isfinite(delta_error)]
    gamma_finite = gamma_error[np.isfinite(gamma_error)]
    q0_bsm_max = float("nan")
    false_exercise_nodes = 0
    if config.option_type == "call" and config.q == 0.0:
        bsm = np.asarray(
            european_call_price(
                result.spot_grid,
                K=config.K,
                T=config.T,
                r=config.r,
                q=config.q,
                sigma=config.sigma,
            ),
            dtype=float,
        )
        target = (result.spot_grid / config.K >= 0.4) & (
            result.spot_grid / config.K <= 1.8
        )
        q0_bsm_max = float(np.max(np.abs(result.values[target] - bsm[target])))
        premium = result.value_grid[1:, 1:-1] - result.payoff[np.newaxis, 1:-1]
        false_exercise_nodes = int(np.count_nonzero(premium <= 1e-10))

    max_difference = float(np.max(absolute))
    certified = bool(
        result.converged
        and np.all(np.isfinite(result.value_grid))
        and max_lcp <= FROZEN_TOLERANCE
        and max_obstacle <= FROZEN_OBSTACLE_TOLERANCE
        and max_difference <= VALUE_MATCH_TOLERANCE * config.K
        and boundary_found_disagreement == 0
        and max(active_disagreements, default=0) == 0
    )
    return {
        "regime_id": regime["regime_id"],
        "split": regime["split"],
        "option_type": regime["option_type"],
        "q": float(regime["q"]),
        "solver": solver,
        "converged": result.converged,
        "numerically_certified": certified,
        "all_steps_theorem_eligible": all(theorem_flags),
        "max_active_components": max(active_components, default=0),
        "max_active_set_disagreement_nodes": max(active_disagreements, default=0),
        "max_abs_trajectory_difference": max_difference,
        "trajectory_mae": float(np.mean(absolute)),
        "trajectory_rmse": float(np.sqrt(np.mean(error**2))),
        "max_normalized_lcp_residual": max_lcp,
        "max_normalized_obstacle_violation": max_obstacle,
        "max_normalized_equation_violation": max_equation,
        "max_normalized_complementarity": max_complementarity,
        "exact_boundary_columns": bool(
            np.array_equal(result.value_grid[:, 0], policy.value_grid[:, 0])
            and np.array_equal(result.value_grid[:, -1], policy.value_grid[:, -1])
        ),
        "boundary_found_disagreement_rows": boundary_found_disagreement,
        "boundary_mae_difference": float(np.mean(np.abs(boundary_errors)))
        if boundary_errors
        else 0.0,
        "boundary_max_abs_difference": float(np.max(np.abs(boundary_errors)))
        if boundary_errors
        else 0.0,
        "delta_rmse_difference": float(np.sqrt(np.mean(delta_finite**2)))
        if delta_finite.size
        else 0.0,
        "delta_max_abs_difference": float(np.max(np.abs(delta_finite)))
        if delta_finite.size
        else 0.0,
        "gamma_rmse_difference": float(np.sqrt(np.mean(gamma_finite**2)))
        if gamma_finite.size
        else 0.0,
        "gamma_max_abs_difference": float(np.max(np.abs(gamma_finite)))
        if gamma_finite.size
        else 0.0,
        "q0_bsm_max_abs_error": q0_bsm_max,
        "q0_false_exercise_nodes": false_exercise_nodes,
    }


def runtime_row(
    regime: dict[str, str],
    arm: str,
    repeat: int,
    result: AmericanLCPResult,
    measured_seconds: float,
) -> dict[str, Any]:
    component = {
        "obstacle_transform": 0.0,
        "projected_sweep": 0.0,
        "inverse_transform": 0.0,
        "residual_audit": 0.0,
    }
    for step in result.lcp_results:
        for name, seconds in step.component_timing:
            component[name] = component.get(name, 0.0) + float(seconds)
    max_lcp = max(
        (step.residual.normalized_lcp_residual for step in result.lcp_results),
        default=0.0,
    )
    accounted = (
        result.initialization_seconds
        + result.solver_setup_seconds
        + result.lcp_finish_seconds
    )
    return {
        "regime_id": regime["regime_id"],
        "split": regime["split"],
        "option_type": regime["option_type"],
        "q": float(regime["q"]),
        "early_exercise_risk": not (
            regime["option_type"] == "call" and float(regime["q"]) == 0.0
        ),
        "arm": arm,
        "repeat": repeat,
        "measured_total_seconds": measured_seconds,
        "internal_total_seconds": result.total_seconds,
        "solver_setup_seconds": result.solver_setup_seconds,
        "initialization_seconds": result.initialization_seconds,
        "lcp_finish_seconds": result.lcp_finish_seconds,
        "marcher_other_seconds": max(result.total_seconds - accounted, 0.0),
        **{f"{name}_seconds": value for name, value in component.items()},
        "total_sweeps_or_iterations": int(
            sum(step.iterations for step in result.lcp_results)
        ),
        "converged": result.converged,
        "max_normalized_lcp_residual": max_lcp,
    }


def summarize_runtime(
    runtime_rows: list[dict[str, Any]],
    *,
    selected_solver: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arms = ("psor", "policy_iteration", selected_solver)
    summary_rows: list[dict[str, Any]] = []
    for arm in arms:
        selected = [row for row in runtime_rows if row["arm"] == arm]
        times = np.array([float(row["measured_total_seconds"]) for row in selected])
        summary_rows.append(
            {
                "arm": arm,
                "sample_count": len(selected),
                "median_seconds": float(np.median(times)),
                "p95_seconds": float(np.percentile(times, 95)),
                "p99_seconds": float(np.percentile(times, 99)),
                "median_solver_setup_seconds": float(
                    np.median([float(row["solver_setup_seconds"]) for row in selected])
                ),
                "median_lcp_finish_seconds": float(
                    np.median([float(row["lcp_finish_seconds"]) for row in selected])
                ),
                "all_converged": all(bool(row["converged"]) for row in selected),
            }
        )

    regime_ids = sorted({str(row["regime_id"]) for row in runtime_rows})
    paired_rows: list[dict[str, Any]] = []
    for regime_id in regime_ids:
        regime_rows = [row for row in runtime_rows if row["regime_id"] == regime_id]
        medians = {
            arm: float(
                np.median(
                    [
                        float(row["measured_total_seconds"])
                        for row in regime_rows
                        if row["arm"] == arm
                    ]
                )
            )
            for arm in arms
        }
        metadata = regime_rows[0]
        paired_rows.append(
            {
                "regime_id": regime_id,
                "split": metadata["split"],
                "option_type": metadata["option_type"],
                "q": metadata["q"],
                "early_exercise_risk": metadata["early_exercise_risk"],
                "psor_median_seconds": medians["psor"],
                "policy_median_seconds": medians["policy_iteration"],
                "projected_lu_median_seconds": medians[selected_solver],
                "lu_over_policy_ratio": medians[selected_solver]
                / medians["policy_iteration"],
                "lu_over_psor_ratio": medians[selected_solver] / medians["psor"],
            }
        )
    ratios = np.array([float(row["lu_over_policy_ratio"]) for row in paired_rows])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(10000, dtype=float)
    for index in range(len(boot)):
        sample = rng.choice(ratios, size=len(ratios), replace=True)
        boot[index] = np.median(sample)
    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "replicates": len(boot),
        "paired_median_lu_over_policy": float(np.median(ratios)),
        "confidence_interval_95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
    }
    return summary_rows, paired_rows, bootstrap


def heldout_decision(
    regimes: list[dict[str, str]],
    accuracy_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    warmups: int,
    repeats: int,
    protocol_complete: bool,
) -> dict[str, Any]:
    selected_solver = str(frozen["selected_solver"])
    projected_accuracy = [
        row for row in accuracy_rows if row["solver"] == selected_solver
    ]
    all_correct = bool(
        len(projected_accuracy) == len(regimes)
        and all(bool(row["numerically_certified"]) for row in projected_accuracy)
    )
    all_theorem = bool(
        projected_accuracy
        and all(bool(row["all_steps_theorem_eligible"]) for row in projected_accuracy)
    )
    by_arm = {str(row["arm"]): row for row in summary_rows}
    paired_policy = np.array(
        [float(row["lu_over_policy_ratio"]) for row in paired_rows], dtype=float
    )
    paired_psor = np.array(
        [float(row["lu_over_psor_ratio"]) for row in paired_rows], dtype=float
    )
    early_policy = np.array(
        [
            float(row["lu_over_policy_ratio"])
            for row in paired_rows
            if str(row["early_exercise_risk"]).lower() == "true"
            or row["early_exercise_risk"] is True
        ],
        dtype=float,
    )
    paired_policy_median = float(np.median(paired_policy))
    paired_psor_median = float(np.median(paired_psor))
    early_policy_median = float(np.median(early_policy))
    projected_p95 = float(by_arm[selected_solver]["p95_seconds"])
    policy_p95 = float(by_arm["policy_iteration"]["p95_seconds"])
    psor_p95 = float(by_arm["psor"]["p95_seconds"])
    beats_policy = bool(
        paired_policy_median <= SPEED_RATIO_GATE
        and early_policy_median <= SPEED_RATIO_GATE
        and projected_p95 <= policy_p95
    )
    beats_psor = bool(
        paired_psor_median <= SPEED_RATIO_GATE and projected_p95 <= psor_p95
    )
    if not protocol_complete:
        status = "DEFER_TIMING_REPRODUCTION"
    elif not all_correct:
        status = "STOP_CORRECTNESS"
    elif beats_policy and all_theorem:
        status = "GO_PROJECTED_LU_THEOREM_CERTIFIED"
    elif beats_policy:
        status = "GO_PROJECTED_LU_NUMERICALLY_CERTIFIED"
    elif beats_psor:
        status = "GO_BEATS_PSOR_ONLY_KEEP_POLICY"
    else:
        status = "STOP_ONLINE_VALUE_KEEP_POLICY"
    return {
        "status": status,
        "selected_solver": selected_solver,
        "new_primary_strict_solver": selected_solver if status.startswith("GO_PROJECTED") else "policy_iteration",
        "regime_count": len(regimes),
        "early_exercise_regime_count": len(early_policy),
        "warmups": warmups,
        "repeats": repeats,
        "protocol_complete": protocol_complete,
        "all_numerically_certified": all_correct,
        "all_theorem_eligible": all_theorem,
        "paired_median_lu_over_policy": paired_policy_median,
        "paired_median_lu_over_psor": paired_psor_median,
        "early_exercise_paired_median_lu_over_policy": early_policy_median,
        "projected_lu_p95_seconds": projected_p95,
        "policy_p95_seconds": policy_p95,
        "psor_p95_seconds": psor_p95,
        "pooled_projected_lu_median_seconds": float(by_arm[selected_solver]["median_seconds"]),
        "pooled_policy_median_seconds": float(by_arm["policy_iteration"]["median_seconds"]),
        "pooled_psor_median_seconds": float(by_arm["psor"]["median_seconds"]),
        "historical_absolute_targets": {
            "median_seconds": HISTORICAL_MEDIAN_TARGET,
            "p95_seconds": HISTORICAL_P95_TARGET,
            "decision_role": "secondary_continuity_check_only",
        },
        "speed_ratio_gate": SPEED_RATIO_GATE,
        "frozen_config_hash": frozen["frozen_config_hash"],
        "claim_limit": (
            "Frozen SURF numerical domain only; classical sufficient-condition theorem "
            "coverage is reported separately."
        ),
    }


def load_regimes() -> list[dict[str, str]]:
    path = DATASET_DIR / "regime_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["regime_id"])
    return rows


def config_from_regime(regime: dict[str, str]) -> AmericanLCPConfig:
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
        tolerance=FROZEN_TOLERANCE,
        obstacle_tolerance=FROZEN_OBSTACLE_TOLERANCE,
    )


def _payoff(config: AmericanLCPConfig) -> np.ndarray:
    spots = np.linspace(0.0, config.Smax, config.M + 1)
    function = put_payoff if config.option_type == "put" else call_payoff
    return np.asarray(function(spots, config.K), dtype=float)


def _dense_matrix(system: TridiagonalLCP) -> np.ndarray:
    dense = np.diag(system.diagonal)
    if system.size > 1:
        dense += np.diag(system.lower, -1)
        dense += np.diag(system.upper, 1)
    return dense


def _eligibility_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "positive_diagonal": sum(bool(row["positive_diagonal"]) for row in rows),
        "nonpositive_offdiagonals": sum(
            bool(row["nonpositive_offdiagonals"]) for row in rows
        ),
        "strict_diagonal_dominance": sum(
            bool(row["strict_diagonal_dominance"]) for row in rows
        ),
        "m_matrix_sufficient_conditions": sum(
            bool(row["m_matrix_sufficient_conditions"]) for row in rows
        ),
        "reconstruction_pass": sum(
            max(
                float(row["lu_reconstruction_max_abs"]),
                float(row["ul_reconstruction_max_abs"]),
            )
            <= RECONSTRUCTION_TOLERANCE
            for row in rows
        ),
        "heldout_non_z": sum(
            row["split"] in {"test", "stress_holdout"}
            and not bool(row["nonpositive_offdiagonals"])
            for row in rows
        ),
    }


def _environment_with_blas() -> dict[str, Any]:
    manifest = environment_manifest()
    try:
        from threadpoolctl import threadpool_info

        manifest["threadpools"] = threadpool_info()
    except Exception as exc:  # pragma: no cover - optional diagnostic
        manifest["threadpools"] = [{"status": "unavailable", "reason": str(exc)}]
    manifest["thread_policy"] = "single_cpu_thread"
    return manifest


def _require_protocol() -> dict[str, Any]:
    decision_path = PROTOCOL_DIR / "eligibility_decision.json"
    protocol_path = PROTOCOL_DIR / "protocol_manifest.json"
    if not decision_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("run experiment 58 before experiment 59")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "PROTOCOL_READY":
        raise RuntimeError("projected-LU protocol gate did not pass")
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def _require_frozen_solver() -> dict[str, Any]:
    path = VALIDATION_DIR / "frozen_solver_config.json"
    if not path.exists():
        raise FileNotFoundError("run experiment 59 before experiment 60")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not str(frozen.get("status", "")).startswith("PROCEED_HELDOUT"):
        raise RuntimeError("validation did not permit held-out evaluation")
    return frozen


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_figures(
    figures_dir: Path,
    eligibility: list[dict[str, str]],
    summary: list[dict[str, str]],
    paired: list[dict[str, str]],
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    labels = [row["arm"] for row in summary]
    medians = [float(row["median_seconds"]) for row in summary]
    p95s = [float(row["p95_seconds"]) for row in summary]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, medians, width, label="median")
    ax.bar(x + width / 2, p95s, width, label="p95")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("End-to-end seconds (log scale)")
    ax.set_title("Strict CN-LCP runtime on the same Mac")
    ax.legend()
    fig.tight_layout()
    paths["runtime"] = figures_dir / "solver_runtime.png"
    fig.savefig(paths["runtime"], dpi=180)
    plt.close(fig)

    ratios = np.array([float(row["lu_over_policy_ratio"]) for row in paired])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ratios, bins=18, color="#4472C4", alpha=0.85)
    ax.axvline(SPEED_RATIO_GATE, color="#C00000", linestyle="--", label="20% gate")
    ax.axvline(1.0, color="black", linestyle=":", label="equal time")
    ax.set_xlabel("Per-regime median Projected LU / Policy time")
    ax.set_ylabel("Regime count")
    ax.set_title("Paired runtime ratios")
    ax.legend()
    fig.tight_layout()
    paths["paired_ratio"] = figures_dir / "paired_speed_ratio.png"
    fig.savefig(paths["paired_ratio"], dpi=180)
    plt.close(fig)

    splits = sorted({row["split"] for row in eligibility})
    eligible = [
        sum(
            row["split"] == split
            and row["m_matrix_sufficient_conditions"].lower() == "true"
            for row in eligibility
        )
        for split in splits
    ]
    ineligible = [
        sum(
            row["split"] == split
            and row["m_matrix_sufficient_conditions"].lower() == "false"
            for row in eligibility
        )
        for split in splits
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(splits, eligible, label="M-matrix sufficient conditions")
    ax.bar(splits, ineligible, bottom=eligible, label="outside sufficient conditions")
    ax.set_ylabel("Regime count")
    ax.set_title("Projected-LU matrix eligibility")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    paths["eligibility"] = figures_dir / "eligibility_by_split.png"
    fig.savefig(paths["eligibility"], dpi=180)
    plt.close(fig)
    return paths


def _english_report(
    decision: dict[str, Any],
    summary: list[dict[str, str]],
    accuracy: list[dict[str, str]],
    figures: dict[str, Path],
) -> str:
    selected = decision["selected_solver"]
    rows = {row["arm"]: row for row in summary}
    projected_accuracy = [row for row in accuracy if row["solver"] == selected]
    max_difference = max(
        float(row["max_abs_trajectory_difference"]) for row in projected_accuracy
    )
    max_residual = max(
        float(row["max_normalized_lcp_residual"]) for row in projected_accuracy
    )
    return f"""# Projected LU / Brennan--Schwartz Technical Report

Decision: **{decision['status']}**

The frozen candidate was `{selected}`. It was compared with CN+PSOR and
previous-slice CN+Policy on the identical 121x121 CN LCP at the frozen 1e-12
residual tolerance.

## Correctness

- All numerically certified: `{decision['all_numerically_certified']}`
- All theorem-eligible: `{decision['all_theorem_eligible']}`
- Maximum full-trajectory difference versus Policy: `{max_difference:.6g}`
- Maximum normalized LCP residual: `{max_residual:.6g}`

## Runtime

| Method | Median (s) | p95 (s) | p99 (s) |
|---|---:|---:|---:|
| CN+PSOR | {float(rows['psor']['median_seconds']):.6g} | {float(rows['psor']['p95_seconds']):.6g} | {float(rows['psor']['p99_seconds']):.6g} |
| CN+Policy | {float(rows['policy_iteration']['median_seconds']):.6g} | {float(rows['policy_iteration']['p95_seconds']):.6g} | {float(rows['policy_iteration']['p99_seconds']):.6g} |
| {selected} | {float(rows[selected]['median_seconds']):.6g} | {float(rows[selected]['p95_seconds']):.6g} | {float(rows[selected]['p99_seconds']):.6g} |

The paired median LU/Policy ratio was
`{decision['paired_median_lu_over_policy']:.6g}`; on the put/dividend-call
subgroup it was `{decision['early_exercise_paired_median_lu_over_policy']:.6g}`.

The claim is limited to the frozen SURF numerical domain. Regimes outside the
classic M-matrix sufficient conditions are reported as numerically certified,
not theorem-certified.

Figures: `{figures['runtime'].name}`, `{figures['paired_ratio'].name}`,
`{figures['eligibility'].name}`.
"""


def _chinese_report(
    decision: dict[str, Any],
    summary: list[dict[str, str]],
    accuracy: list[dict[str, str]],
    figures: dict[str, Path],
) -> str:
    selected = decision["selected_solver"]
    rows = {row["arm"]: row for row in summary}
    projected_accuracy = [row for row in accuracy if row["solver"] == selected]
    max_difference = max(
        float(row["max_abs_trajectory_difference"]) for row in projected_accuracy
    )
    max_residual = max(
        float(row["max_normalized_lcp_residual"]) for row in projected_accuracy
    )
    return f"""# Projected LU / Brennan--Schwartz 中文结论

最终状态：**{decision['status']}**

正式候选是 `{selected}`。它与 CN+PSOR、CN+Policy 使用完全相同的
121x121 CN-LCP 和 1e-12 residual 容差。

## 正确性

- 67 个 held-out regimes 全部数值认证：`{decision['all_numerically_certified']}`
- 全部满足经典理论充分条件：`{decision['all_theorem_eligible']}`
- 与 Policy 完整轨迹最大差异：`{max_difference:.6g}`
- 最大 normalized LCP residual：`{max_residual:.6g}`

## 速度

| 方法 | Median (秒) | p95 (秒) |
|---|---:|---:|
| CN+PSOR | {float(rows['psor']['median_seconds']):.6g} | {float(rows['psor']['p95_seconds']):.6g} |
| CN+Policy | {float(rows['policy_iteration']['median_seconds']):.6g} | {float(rows['policy_iteration']['p95_seconds']):.6g} |
| {selected} | {float(rows[selected]['median_seconds']):.6g} | {float(rows[selected]['p95_seconds']):.6g} |

逐 regime 的 median LU/Policy 比率为
`{decision['paired_median_lu_over_policy']:.6g}`；只看真正具有提前行权风险的
put/dividend-call，比率为
`{decision['early_exercise_paired_median_lu_over_policy']:.6g}`。

该结论只适用于冻结的 SURF 参数域。经典 M-matrix 充分条件之外的情况只能称为
“在当前数值集合中通过 residual 和 Policy 对照认证”，不能声称受到无条件理论保证。

图表：`{figures['runtime'].name}`、`{figures['paired_ratio'].name}`、
`{figures['eligibility'].name}`。
"""
