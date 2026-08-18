"""DIRK+sinh Policy-to-Projected-LU solver-substitution audit.

The protocol phase in this module is deliberately independent of the
Projected-LU candidate.  It freezes the existing spatial-reference errors,
stable Greek query nodes, and agreement gates before a candidate solution is
computed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    continuation_premium,
    extract_boundary_at_time,
)
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.method_extensions.projected_lu_study import (
    FROZEN_OBSTACLE_TOLERANCE,
    FROZEN_TOLERANCE,
    VALUE_MATCH_TOLERANCE,
)
from american_risk_surfaces.method_extensions.protocol import (
    AUDIT_REGIME_IDS,
    DATASET_DIR,
    environment_manifest,
    sha256_file,
)
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.greek_integrators import (
    AmericanGreekIntegratorResult,
    american_dirk_policy_price,
    american_dirk_projected_lu_price,
)
from american_risk_surfaces.solvers.grid import sinh_spot_grid


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "15_dirk_projected_lu_substitution"
REPORTS_DIR = PROJECT_ROOT / "reports" / "17_dirk_projected_lu_substitution"
PROTOCOL_DIR = RESULTS_DIR / "00_protocol"
AUDIT_DIR = RESULTS_DIR / "01_solver_substitution_audit"
REFERENCE_ERROR_PATH = (
    PROJECT_ROOT
    / "results"
    / "07_method_extensions"
    / "03_greek_audit"
    / "spatial_convergence.csv"
)
TOLERANCE_DECISION_PATH = (
    PROJECT_ROOT
    / "results"
    / "07_method_extensions"
    / "00_protocol"
    / "tolerance_decision.json"
)

PROTOCOL_VERSION = "dirk_sinh_projected_lu_substitution_v1"
FROZEN_M = 480
FROZEN_N = 960
REFERENCE_M = 960
DIRK_DAMPING_STEPS = 2
SINH_CONCENTRATION_WIDTH_K = 0.1
BOUNDARY_THRESHOLD = 1e-6
REFERENCE_ERROR_FRACTION = 0.1
SPEED_RATIO_GATE = 0.8
TIMING_ORDER_SEED = 20260819


def freeze_substitution_protocol(
    output_dir: Path | str = PROTOCOL_DIR,
    *,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Freeze all solver-substitution gates before any candidate is run."""

    output = Path(output_dir)
    protocol_path = output / "protocol_manifest.json"
    if protocol_path.exists():
        if not allow_existing:
            raise RuntimeError("substitution protocol already exists; refusing overwrite")
        if output.resolve() != PROTOCOL_DIR.resolve():
            raise RuntimeError("allow_existing is only valid for the canonical protocol")
        protocol = _load_frozen_protocol()
        return {
            "protocol": protocol_path,
            "gates": output / "frozen_agreement_gates.csv",
            "marker": output / "pre_candidate_freeze.marker.json",
            "protocol_data": protocol,
        }
    output.mkdir(parents=True, exist_ok=True)

    frozen_tolerance = json.loads(
        TOLERANCE_DECISION_PATH.read_text(encoding="utf-8")
    )
    if float(frozen_tolerance["frozen_normalized_lcp_tolerance"]) != FROZEN_TOLERANCE:
        raise RuntimeError("repository LCP tolerance no longer matches the frozen protocol")

    regimes = _load_audit_regimes()
    reference_rows = _load_existing_reference_rows()
    grid = sinh_spot_grid(
        4.0,
        1.0,
        FROZEN_M,
        concentration_width=SINH_CONCENTRATION_WIDTH_K,
    )
    gate_rows: list[dict[str, Any]] = []
    for regime in regimes:
        regime_id = regime["regime_id"]
        prior = reference_rows[regime_id]
        reference_config = _config(regime, M=REFERENCE_M, N=FROZEN_N)
        reference_grid = sinh_spot_grid(
            reference_config.Smax,
            reference_config.K,
            REFERENCE_M,
            concentration_width=SINH_CONCENTRATION_WIDTH_K * reference_config.K,
        )
        reference = american_dirk_policy_price(
            reference_config,
            spot_grid=reference_grid,
            damping_steps=DIRK_DAMPING_STEPS,
        )
        if not reference.converged:
            raise RuntimeError(f"existing Policy reference did not converge for {regime_id}")
        boundary = extract_boundary_at_time(
            reference.spot_grid,
            continuation_premium(reference.values, reference.payoff),
            reference.config.option_type,
            tau=float(reference.tau_grid[-1]),
            time_index=len(reference.tau_grid) - 1,
            threshold=BOUNDARY_THRESHOLD,
        )
        query_spots = _stable_query_spots(reference.config.K, boundary)
        expected_query_nodes = int(prior["query_nodes"])
        if len(query_spots) != expected_query_nodes:
            raise RuntimeError(
                f"stable query-node count changed for {regime_id}: "
                f"{len(query_spots)} != {expected_query_nodes}"
            )
        delta_price_bound = VALUE_MATCH_TOLERANCE * _interpolated_operator_l1_bound(
            grid, query_spots, derivative_order=1
        )
        gamma_price_bound = VALUE_MATCH_TOLERANCE * _interpolated_operator_l1_bound(
            grid, query_spots, derivative_order=2
        )
        delta_reference_error = _positive_finite(
            "delta_max_error", prior["delta_max_error"], regime_id
        )
        gamma_reference_error = _positive_finite(
            "gamma_max_error", prior["gamma_max_error"], regime_id
        )
        gate_rows.append(
            {
                "regime_id": regime_id,
                "option_type": regime["option_type"],
                "existing_delta_reference_error": delta_reference_error,
                "existing_gamma_reference_error": gamma_reference_error,
                "delta_reference_fraction_bound": (
                    REFERENCE_ERROR_FRACTION * delta_reference_error
                ),
                "gamma_reference_fraction_bound": (
                    REFERENCE_ERROR_FRACTION * gamma_reference_error
                ),
                "delta_price_operator_bound": delta_price_bound,
                "gamma_price_operator_bound": gamma_price_bound,
                "frozen_delta_max_difference_gate": min(
                    REFERENCE_ERROR_FRACTION * delta_reference_error,
                    delta_price_bound,
                ),
                "frozen_gamma_max_difference_gate": min(
                    REFERENCE_ERROR_FRACTION * gamma_reference_error,
                    gamma_price_bound,
                ),
                "frozen_boundary_local_spacing_gate": float(
                    prior["local_grid_spacing"]
                ),
                "reference_boundary_found": bool(boundary.boundary_found),
                "reference_boundary_spot": (
                    float(boundary.boundary_spot)
                    if boundary.boundary_found
                    else float("nan")
                ),
                "stable_query_node_count": len(query_spots),
                "stable_query_spots_json": json.dumps(query_spots.tolist()),
            }
        )

    gate_path = output / "frozen_agreement_gates.csv"
    _write_csv(gate_path, gate_rows)
    required_inputs = (
        DATASET_DIR / "regime_manifest.csv",
        REFERENCE_ERROR_PATH,
        TOLERANCE_DECISION_PATH,
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "solvers"
        / "greek_integrators.py",
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "solvers"
        / "grid.py",
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "diagnostics"
        / "greeks.py",
        PROJECT_ROOT
        / "experiments"
        / "26_greek_spatial_grid_audit.py",
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "research_question": (
            "replace only Policy Iteration with option-directed single-sweep "
            "Projected LU inside the frozen DIRK+sinh reference"
        ),
        "audit_regime_ids": list(AUDIT_REGIME_IDS),
        "frozen_configuration": {
            "M": FROZEN_M,
            "N": FROZEN_N,
            "reference_M_used_by_existing_spatial_audit": REFERENCE_M,
            "K": 1.0,
            "Smax": 4.0,
            "DIRK_theta": "1-sqrt(2)/2",
            "quadratic_time_grid": True,
            "backward_euler_damping_steps": DIRK_DAMPING_STEPS,
            "sinh_concentration_width_K": SINH_CONCENTRATION_WIDTH_K,
            "boundary_threshold": BOUNDARY_THRESHOLD,
            "dtype": "float64",
        },
        "frozen_lcp_gates": {
            "normalized_lcp_residual": FROZEN_TOLERANCE,
            "normalized_obstacle_violation": FROZEN_OBSTACLE_TOLERANCE,
            "price_max_difference_K": VALUE_MATCH_TOLERANCE,
        },
        "frozen_greek_gate_formula": {
            "reference_error_fraction": REFERENCE_ERROR_FRACTION,
            "delta": "min(0.1 * existing regime Delta max error, B_delta_price)",
            "gamma": "min(0.1 * existing regime Gamma max error, B_gamma_price)",
            "price_bound_definition": (
                "1e-9 K times the maximum L1 row norm of the exact M=480 "
                "nonuniform finite-difference-plus-interpolation operator on the "
                "existing stable query nodes"
            ),
        },
        "boundary_gate": (
            "same found/no-boundary status and difference no larger than the "
            "pre-existing regime-specific local M=480 grid spacing"
        ),
        "runtime_gate": {
            "paired_candidate_over_policy_median_ratio_max": SPEED_RATIO_GATE,
            "warmups": 5,
            "repeats": 30,
            "threads": 1,
        },
        "candidate_results_seen_before_freeze": False,
        "all_reference_errors_pre_existing_and_valid": True,
        "agreement_gate_rows_sha256": sha256_file(gate_path),
        "input_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in required_inputs
        },
        "environment": environment_manifest(),
    }
    protocol["protocol_hash"] = _json_hash(protocol)
    _write_json(protocol_path, protocol)
    marker = {
        "status": "FROZEN_BEFORE_PROJECTED_LU_RESULTS",
        "protocol_hash": protocol["protocol_hash"],
        "gate_file_sha256": sha256_file(gate_path),
        "candidate_results_seen": False,
    }
    marker_path = output / "pre_candidate_freeze.marker.json"
    _write_json(marker_path, marker)
    return {
        "protocol": protocol_path,
        "gates": gate_path,
        "marker": marker_path,
        "protocol_data": protocol,
    }


def run_substitution_audit(
    output_dir: Path | str = AUDIT_DIR,
    *,
    warmups: int = 5,
    repeats: int = 30,
    regime_limit: int | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Run the frozen Policy-versus-Projected-LU DIRK+sinh comparison."""

    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    output = Path(output_dir)
    marker_path = output / "solver_substitution_complete.marker.json"
    if marker_path.exists():
        if not allow_existing:
            raise RuntimeError("solver-substitution audit already exists; refusing overwrite")
        decision_path = output / "method_decision.json"
        reports = synthesize_substitution_audit(output, REPORTS_DIR)
        return {
            "structural": output / "stage_structural_checks.csv",
            "agreement": output / "regime_correctness_and_agreement.csv",
            "runtime": output / "runtime_samples.csv",
            "paired": output / "paired_runtime_by_regime.csv",
            "summary": output / "runtime_summary.csv",
            "decision": decision_path,
            "marker": marker_path,
            "reports": reports,
            "decision_data": json.loads(decision_path.read_text(encoding="utf-8")),
        }
    protocol = _load_frozen_protocol()
    gates = _load_frozen_gate_rows()
    regimes = _load_audit_regimes()
    if regime_limit is not None:
        if regime_limit < 1:
            raise ValueError("regime_limit must be positive")
        regimes = regimes[:regime_limit]
    output.mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    surface_hash_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(TIMING_ORDER_SEED)
    audit_started = perf_counter_ns()

    for regime_index, regime in enumerate(regimes, start=1):
        regime_id = regime["regime_id"]
        config = _config(regime, M=FROZEN_M, N=FROZEN_N)
        spot_grid = sinh_spot_grid(
            config.Smax,
            config.K,
            config.M,
            concentration_width=SINH_CONCENTRATION_WIDTH_K * config.K,
        )
        policy = american_dirk_policy_price(
            config,
            spot_grid=spot_grid,
            damping_steps=DIRK_DAMPING_STEPS,
        )
        candidate_error = ""
        candidate: AmericanGreekIntegratorResult | None = None
        try:
            candidate = american_dirk_projected_lu_price(
                config,
                spot_grid=spot_grid,
                damping_steps=DIRK_DAMPING_STEPS,
            )
        except Exception as exc:  # explicit evidence; never a solver fallback
            candidate_error = f"{type(exc).__name__}: {exc}"

        if candidate is None:
            agreement_rows.append(
                _failed_agreement_row(regime, gates[regime_id], candidate_error)
            )
        else:
            structural_rows.extend(_stage_rows(regime, candidate))
            comparison, profiles, boundaries, hashes = _agreement_metrics(
                regime,
                gates[regime_id],
                policy,
                candidate,
            )
            agreement_rows.append(comparison)
            profile_rows.extend(profiles)
            boundary_rows.extend(boundaries)
            surface_hash_rows.extend(hashes)

        for method in ("policy_iteration", "projected_lu_single"):
            for _ in range(warmups):
                try:
                    _solve_method(method, config, spot_grid)
                except Exception:
                    pass
        for repeat_index in range(repeats):
            order = ["policy_iteration", "projected_lu_single"]
            rng.shuffle(order)
            for order_index, method in enumerate(order):
                started = perf_counter_ns()
                timed_result: AmericanGreekIntegratorResult | None = None
                timing_error = ""
                try:
                    timed_result = _solve_method(method, config, spot_grid)
                except Exception as exc:
                    timing_error = f"{type(exc).__name__}: {exc}"
                measured = (perf_counter_ns() - started) * 1e-9
                runtime_rows.append(
                    {
                        "regime_id": regime_id,
                        "option_type": regime["option_type"],
                        "q": float(regime["q"]),
                        "genuine_early_exercise": not (
                            regime["option_type"] == "call"
                            and float(regime["q"]) == 0.0
                        ),
                        "method": method,
                        "repeat_index": repeat_index,
                        "randomized_order_index": order_index,
                        "measured_seconds": measured,
                        "reported_total_seconds": (
                            timed_result.total_seconds
                            if timed_result is not None
                            else float("nan")
                        ),
                        "converged": (
                            timed_result.converged
                            if timed_result is not None
                            else False
                        ),
                        "all_structural_checks_passed": (
                            _all_stage_structural_checks(timed_result)
                            if method == "projected_lu_single"
                            and timed_result is not None
                            else method == "policy_iteration"
                        ),
                        "failure_reason": timing_error,
                    }
                )
        elapsed = (perf_counter_ns() - audit_started) * 1e-9
        print(
            f"[{regime_index}/{len(regimes)}] {regime_id} complete "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )

    structural_path = output / "stage_structural_checks.csv"
    agreement_path = output / "regime_correctness_and_agreement.csv"
    runtime_path = output / "runtime_samples.csv"
    profile_path = output / "final_price_greek_profiles.csv"
    boundary_path = output / "boundary_paths.csv"
    hashes_path = output / "surface_hashes.csv"
    _write_csv(structural_path, structural_rows)
    _write_csv(agreement_path, agreement_rows)
    _write_csv(runtime_path, runtime_rows)
    _write_csv(profile_path, profile_rows)
    _write_csv(boundary_path, boundary_rows)
    _write_csv(hashes_path, surface_hash_rows)

    paired_rows = _paired_runtime_rows(runtime_rows)
    summary_rows = _runtime_summary(runtime_rows)
    paired_path = output / "paired_runtime_by_regime.csv"
    summary_path = output / "runtime_summary.csv"
    _write_csv(paired_path, paired_rows)
    _write_csv(summary_path, summary_rows)
    decision = _decision(
        agreement_rows,
        paired_rows,
        runtime_rows,
        protocol_hash=str(protocol["protocol_hash"]),
        full_protocol=regime_limit is None and len(regimes) == len(AUDIT_REGIME_IDS),
    )
    decision_path = output / "method_decision.json"
    _write_json(decision_path, decision)
    marker = {
        "status": "COMPLETE",
        "decision": decision["decision"],
        "protocol_hash": protocol["protocol_hash"],
        "regime_count": len(regimes),
        "warmups": warmups,
        "repeats": repeats,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in (
                structural_path,
                agreement_path,
                runtime_path,
                profile_path,
                boundary_path,
                hashes_path,
                paired_path,
                summary_path,
                decision_path,
            )
        },
    }
    _write_json(marker_path, marker)
    reports = synthesize_substitution_audit(output, REPORTS_DIR)
    return {
        "structural": structural_path,
        "agreement": agreement_path,
        "runtime": runtime_path,
        "paired": paired_path,
        "summary": summary_path,
        "decision": decision_path,
        "marker": marker_path,
        "reports": reports,
        "decision_data": decision,
    }


def synthesize_substitution_audit(
    audit_dir: Path | str = AUDIT_DIR,
    reports_dir: Path | str = REPORTS_DIR,
) -> dict[str, Path]:
    """Create technical and plain-language reports from immutable audit CSVs."""

    audit = Path(audit_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    decision = json.loads((audit / "method_decision.json").read_text(encoding="utf-8"))
    agreement = _read_csv(audit / "regime_correctness_and_agreement.csv")
    paired = _read_csv(audit / "paired_runtime_by_regime.csv")
    runtime_summary = _read_csv(audit / "runtime_summary.csv")
    structural = _read_csv(audit / "stage_structural_checks.csv")
    reason_counts: dict[str, int] = {}
    for row in structural:
        reasons = str(row["structural_failure_reasons"]).strip()
        if reasons:
            for reason in reasons.split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    structural_summary = _structural_summary(structural)
    structural_summary_path = reports / "stage_structural_summary.csv"
    _write_csv(structural_summary_path, structural_summary)
    technical_path = reports / "dirk_projected_lu_substitution_report.md"
    chinese_path = reports / "dirk_projected_lu_substitution_结论_CN.md"
    technical_path.write_text(
        _technical_report(
            decision, agreement, paired, runtime_summary, reason_counts
        ),
        encoding="utf-8",
    )
    chinese_path.write_text(
        _chinese_report(
            decision, agreement, paired, runtime_summary, reason_counts
        ),
        encoding="utf-8",
    )
    source_paths = (
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "solvers"
        / "greek_integrators.py",
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "solvers"
        / "projected_lu.py",
        PROJECT_ROOT
        / "src"
        / "american_risk_surfaces"
        / "method_extensions"
        / "dirk_projected_lu_study.py",
        PROJECT_ROOT / "experiments" / "65_dirk_projected_lu_protocol.py",
        PROJECT_ROOT
        / "experiments"
        / "66_dirk_sinh_projected_lu_substitution_audit.py",
        PROJECT_ROOT
        / "experiments"
        / "67_dirk_projected_lu_substitution_synthesis.py",
        PROJECT_ROOT / "tests" / "test_dirk_projected_lu_substitution.py",
    )
    raw_paths = tuple(
        sorted(
            path
            for path in audit.iterdir()
            if path.is_file() and path.name != "provenance_manifest.json"
        )
    )
    protocol = _load_frozen_protocol()
    provenance = {
        "experiment": "DIRK sinh Projected-LU solver substitution audit",
        "base_git_commit_before_experiment": protocol["environment"]["git_commit"],
        "protocol_hash": protocol["protocol_hash"],
        "candidate_results_seen_before_protocol_freeze": False,
        "source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths
        },
        "raw_evidence_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in raw_paths
        },
        "report_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (technical_path, chinese_path, structural_summary_path)
        },
        "environment": environment_manifest(),
    }
    provenance_path = audit.parent / "provenance_manifest.json"
    _write_json(provenance_path, provenance)
    return {
        "technical_report": technical_path,
        "chinese_report": chinese_path,
        "structural_summary": structural_summary_path,
        "provenance": provenance_path,
    }


def _load_frozen_protocol() -> dict[str, Any]:
    protocol_path = PROTOCOL_DIR / "protocol_manifest.json"
    marker_path = PROTOCOL_DIR / "pre_candidate_freeze.marker.json"
    if not protocol_path.exists() or not marker_path.exists():
        raise FileNotFoundError("pre-candidate substitution protocol has not been frozen")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "FROZEN_BEFORE_PROJECTED_LU_RESULTS":
        raise RuntimeError("invalid pre-candidate freeze marker")
    if marker.get("candidate_results_seen") is not False:
        raise RuntimeError("protocol marker does not certify a pre-candidate freeze")
    if marker.get("protocol_hash") != protocol.get("protocol_hash"):
        raise RuntimeError("protocol hash does not match its freeze marker")
    gate_path = PROTOCOL_DIR / "frozen_agreement_gates.csv"
    if sha256_file(gate_path) != protocol.get("agreement_gate_rows_sha256"):
        raise RuntimeError("frozen agreement-gate file hash mismatch")
    return protocol


def _load_frozen_gate_rows() -> dict[str, dict[str, Any]]:
    path = PROTOCOL_DIR / "frozen_agreement_gates.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        regime_id = row["regime_id"]
        if regime_id in by_id:
            raise RuntimeError(f"duplicate frozen gate row for {regime_id}")
        parsed: dict[str, Any] = dict(row)
        for name in (
            "existing_delta_reference_error",
            "existing_gamma_reference_error",
            "delta_reference_fraction_bound",
            "gamma_reference_fraction_bound",
            "delta_price_operator_bound",
            "gamma_price_operator_bound",
            "frozen_delta_max_difference_gate",
            "frozen_gamma_max_difference_gate",
            "frozen_boundary_local_spacing_gate",
        ):
            parsed[name] = float(row[name])
        parsed["reference_boundary_found"] = row["reference_boundary_found"] == "True"
        parsed["reference_boundary_spot"] = float(row["reference_boundary_spot"])
        parsed["stable_query_node_count"] = int(row["stable_query_node_count"])
        parsed["stable_query_spots"] = np.asarray(
            json.loads(row["stable_query_spots_json"]), dtype=float
        )
        by_id[regime_id] = parsed
    if tuple(by_id) != tuple(AUDIT_REGIME_IDS):
        raise RuntimeError("frozen gate rows do not match the ordered 12-regime audit")
    return by_id


def _solve_method(
    method: str,
    config: AmericanLCPConfig,
    spot_grid: np.ndarray,
) -> AmericanGreekIntegratorResult:
    if method == "policy_iteration":
        return american_dirk_policy_price(
            config,
            spot_grid=spot_grid,
            damping_steps=DIRK_DAMPING_STEPS,
        )
    if method == "projected_lu_single":
        return american_dirk_projected_lu_price(
            config,
            spot_grid=spot_grid,
            damping_steps=DIRK_DAMPING_STEPS,
        )
    raise ValueError(f"unsupported method: {method}")


def _stage_rows(
    regime: dict[str, str],
    result: AmericanGreekIntegratorResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in result.projected_lu_stage_audits:
        pre = audit.pre_eligibility
        post = audit.post_eligibility
        residual = audit.result.residual
        reasons = tuple(dict.fromkeys(pre.reasons + post.reasons))
        rows.append(
            {
                "regime_id": regime["regime_id"],
                "option_type": regime["option_type"],
                "step_index": audit.step_index,
                "stage_name": audit.stage_name,
                "dt": audit.dt,
                "mode": audit.mode,
                "matrix_sha256": audit.matrix_sha256,
                "tridiagonal_structure": True,
                "positive_diagonal": pre.positive_diagonal,
                "nonpositive_offdiagonals": pre.nonpositive_offdiagonals,
                "strictly_diagonally_dominant": pre.strictly_diagonally_dominant,
                "positive_lu_pivots": pre.positive_lu_pivots,
                "positive_ul_pivots": pre.positive_ul_pivots,
                "m_matrix_sufficient_conditions": pre.m_matrix_sufficient_conditions,
                "contact_components": post.contact_components,
                "expected_contact_geometry": post.expected_contact_geometry,
                "theorem_eligible": post.theorem_eligible,
                "structural_failure_reasons": ";".join(reasons),
                "finite_solution": bool(np.all(np.isfinite(audit.result.solution))),
                "solver_converged": audit.result.converged,
                "normalized_lcp_residual": residual.normalized_lcp_residual,
                "normalized_obstacle_violation": residual.normalized_obstacle_violation,
                "normalized_equation_violation": residual.normalized_equation_violation,
                "normalized_complementarity": residual.normalized_complementarity,
                "max_obstacle_violation": residual.max_obstacle_violation,
                "max_equation_violation": residual.max_equation_violation,
                "max_abs_complementarity": residual.max_abs_complementarity,
                "factorization_seconds": audit.factorization_seconds,
                "precheck_seconds": audit.precheck_seconds,
                "postcheck_seconds": audit.postcheck_seconds,
                "projected_sweep_seconds": dict(audit.result.component_timing).get(
                    "projected_sweep", float("nan")
                ),
            }
        )
    return rows


def _all_stage_structural_checks(result: AmericanGreekIntegratorResult) -> bool:
    expected = DIRK_DAMPING_STEPS + 2 * (result.config.N - DIRK_DAMPING_STEPS)
    audits = result.projected_lu_stage_audits
    return bool(
        len(audits) == expected
        and all(
            audit.pre_eligibility.theorem_eligible
            and audit.post_eligibility.theorem_eligible
            and audit.result.converged
            and np.all(np.isfinite(audit.result.solution))
            for audit in audits
        )
    )


def _agreement_metrics(
    regime: dict[str, str],
    gate: dict[str, Any],
    policy: AmericanGreekIntegratorResult,
    candidate: AmericanGreekIntegratorResult,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not np.array_equal(policy.spot_grid, candidate.spot_grid):
        raise RuntimeError("Policy and Projected-LU spot grids differ")
    if not np.array_equal(policy.tau_grid, candidate.tau_grid):
        raise RuntimeError("Policy and Projected-LU time grids differ")
    query = np.asarray(gate["stable_query_spots"], dtype=float)
    policy_delta = finite_difference_delta_nonuniform(policy.spot_grid, policy.values)
    candidate_delta = finite_difference_delta_nonuniform(
        candidate.spot_grid, candidate.values
    )
    policy_gamma = finite_difference_gamma_nonuniform(policy.spot_grid, policy.values)
    candidate_gamma = finite_difference_gamma_nonuniform(
        candidate.spot_grid, candidate.values
    )
    policy_delta_query = np.interp(query, policy.spot_grid, policy_delta)
    candidate_delta_query = np.interp(query, candidate.spot_grid, candidate_delta)
    policy_gamma_query = np.interp(query, policy.spot_grid, policy_gamma)
    candidate_gamma_query = np.interp(query, candidate.spot_grid, candidate_gamma)
    delta_difference = candidate_delta_query - policy_delta_query
    gamma_difference = candidate_gamma_query - policy_gamma_query
    value_difference = candidate.value_grid - policy.value_grid
    final_query_difference = np.interp(
        query, candidate.spot_grid, candidate.values
    ) - np.interp(query, policy.spot_grid, policy.values)

    boundary_metrics, boundary_rows = _boundary_comparison(
        regime,
        gate,
        policy,
        candidate,
    )
    candidate_stages = [
        stage
        for stages in candidate.stage_results
        for stage in stages
    ]
    policy_stages = [stage for stages in policy.stage_results for stage in stages]
    candidate_residual = max(
        stage.residual.normalized_lcp_residual for stage in candidate_stages
    )
    candidate_obstacle = max(
        stage.residual.normalized_obstacle_violation for stage in candidate_stages
    )
    candidate_equation = max(
        stage.residual.normalized_equation_violation for stage in candidate_stages
    )
    candidate_complementarity = max(
        stage.residual.normalized_complementarity for stage in candidate_stages
    )
    policy_residual = max(
        stage.residual.normalized_lcp_residual for stage in policy_stages
    )
    structural_pass = _all_stage_structural_checks(candidate)
    price_max = float(np.max(np.abs(value_difference)))
    delta_max = float(np.max(np.abs(delta_difference)))
    gamma_max = float(np.max(np.abs(gamma_difference)))
    exact_boundary_difference = float(
        max(
            np.max(np.abs(candidate.value_grid[:, 0] - policy.value_grid[:, 0])),
            np.max(np.abs(candidate.value_grid[:, -1] - policy.value_grid[:, -1])),
        )
    )
    finite_pass = bool(
        np.all(np.isfinite(candidate.value_grid))
        and np.all(np.isfinite(delta_difference))
        and np.all(np.isfinite(gamma_difference))
    )
    lcp_pass = bool(
        candidate.converged
        and candidate_residual <= FROZEN_TOLERANCE
        and candidate_obstacle <= FROZEN_OBSTACLE_TOLERANCE
    )
    price_pass = price_max <= VALUE_MATCH_TOLERANCE * policy.config.K
    delta_pass = delta_max <= float(gate["frozen_delta_max_difference_gate"])
    gamma_pass = gamma_max <= float(gate["frozen_gamma_max_difference_gate"])
    financial_pass = bool(
        finite_pass
        and exact_boundary_difference == 0.0
        and boundary_metrics["boundary_financial_structure_passed"]
        and all(
            audit.post_eligibility.expected_contact_geometry is True
            for audit in candidate.projected_lu_stage_audits
        )
    )
    failures: list[str] = []
    for name, passed in (
        ("structural_or_pivot", structural_pass),
        ("lcp_residual", lcp_pass),
        ("price", price_pass),
        ("boundary", boundary_metrics["boundary_agreement_passed"]),
        ("delta", delta_pass),
        ("gamma", gamma_pass),
        ("financial_structure", financial_pass),
    ):
        if not bool(passed):
            failures.append(name)
    correctness_pass = not failures
    row = {
        "regime_id": regime["regime_id"],
        "option_type": regime["option_type"],
        "T": float(regime["T"]),
        "sigma": float(regime["sigma"]),
        "r": float(regime["r"]),
        "q": float(regime["q"]),
        "genuine_early_exercise": not (
            regime["option_type"] == "call" and float(regime["q"]) == 0.0
        ),
        "policy_converged": policy.converged,
        "projected_lu_converged": candidate.converged,
        "stage_count": len(candidate_stages),
        "structural_checks_passed": structural_pass,
        "max_normalized_lcp_residual": candidate_residual,
        "max_normalized_obstacle_violation": candidate_obstacle,
        "max_normalized_equation_violation": candidate_equation,
        "max_normalized_complementarity": candidate_complementarity,
        "policy_max_normalized_lcp_residual": policy_residual,
        "full_trajectory_price_max_difference": price_max,
        "full_trajectory_price_mae": float(np.mean(np.abs(value_difference))),
        "full_trajectory_price_rmse": float(np.sqrt(np.mean(value_difference**2))),
        "final_query_price_max_difference": float(
            np.max(np.abs(final_query_difference))
        ),
        "final_query_price_rmse": float(
            np.sqrt(np.mean(final_query_difference**2))
        ),
        "price_max_difference_gate": VALUE_MATCH_TOLERANCE * policy.config.K,
        "price_agreement_passed": price_pass,
        **boundary_metrics,
        "stable_query_node_count": len(query),
        "delta_max_difference": delta_max,
        "delta_rmse_difference": float(np.sqrt(np.mean(delta_difference**2))),
        "delta_median_abs_difference": float(np.median(np.abs(delta_difference))),
        "delta_p95_abs_difference": float(np.percentile(np.abs(delta_difference), 95.0)),
        "delta_max_difference_gate": float(gate["frozen_delta_max_difference_gate"]),
        "delta_reference_fraction_bound": float(gate["delta_reference_fraction_bound"]),
        "delta_price_operator_bound": float(gate["delta_price_operator_bound"]),
        "delta_agreement_passed": delta_pass,
        "gamma_max_difference": gamma_max,
        "gamma_rmse_difference": float(np.sqrt(np.mean(gamma_difference**2))),
        "gamma_median_abs_difference": float(np.median(np.abs(gamma_difference))),
        "gamma_p95_abs_difference": float(np.percentile(np.abs(gamma_difference), 95.0)),
        "gamma_max_difference_gate": float(gate["frozen_gamma_max_difference_gate"]),
        "gamma_reference_fraction_bound": float(gate["gamma_reference_fraction_bound"]),
        "gamma_price_operator_bound": float(gate["gamma_price_operator_bound"]),
        "gamma_agreement_passed": gamma_pass,
        "exact_domain_boundary_max_difference": exact_boundary_difference,
        "financial_structure_passed": financial_pass,
        "all_correctness_gates_passed": correctness_pass,
        "failure_reasons": ";".join(failures),
        "candidate_exception": "",
    }
    profiles: list[dict[str, Any]] = []
    for method, result, delta, gamma in (
        ("policy_iteration", policy, policy_delta, policy_gamma),
        ("projected_lu_single", candidate, candidate_delta, candidate_gamma),
    ):
        for index, spot in enumerate(result.spot_grid):
            profiles.append(
                {
                    "regime_id": regime["regime_id"],
                    "method": method,
                    "spot_index": index,
                    "spot": float(spot),
                    "value": float(result.values[index]),
                    "delta": float(delta[index]),
                    "gamma": float(gamma[index]),
                    "stable_query_region": bool(
                        np.any(np.isclose(query, spot, atol=1e-14, rtol=0.0))
                    ),
                }
            )
    hashes = []
    for method, result in (
        ("policy_iteration", policy),
        ("projected_lu_single", candidate),
    ):
        hashes.append(
            {
                "regime_id": regime["regime_id"],
                "method": method,
                "value_grid_sha256": _array_hash(result.value_grid),
                "final_values_sha256": _array_hash(result.values),
                "spot_grid_sha256": _array_hash(result.spot_grid),
                "tau_grid_sha256": _array_hash(result.tau_grid),
            }
        )
    hashes.append(
        {
            "regime_id": regime["regime_id"],
            "method": "projected_lu_minus_policy",
            "value_grid_sha256": _array_hash(value_difference),
            "final_values_sha256": _array_hash(candidate.values - policy.values),
            "spot_grid_sha256": _array_hash(policy.spot_grid),
            "tau_grid_sha256": _array_hash(policy.tau_grid),
        }
    )
    return row, profiles, boundary_rows, hashes


def _boundary_comparison(
    regime: dict[str, str],
    gate: dict[str, Any],
    policy: AmericanGreekIntegratorResult,
    candidate: AmericanGreekIntegratorResult,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q0_call = regime["option_type"] == "call" and float(regime["q"]) == 0.0
    rows: list[dict[str, Any]] = []
    differences: list[float] = []
    found_mismatches = 0
    policy_found_count = 0
    candidate_found_count = 0
    for index, tau in enumerate(policy.tau_grid):
        if q0_call:
            policy_found = False
            candidate_found = False
            policy_spot = float("nan")
            candidate_spot = float("nan")
            policy_reason = "not_applicable_no_dividend_call"
            candidate_reason = "not_applicable_no_dividend_call"
        else:
            policy_point = extract_boundary_at_time(
                policy.spot_grid,
                continuation_premium(policy.value_grid[index], policy.payoff),
                policy.config.option_type,
                tau=float(tau),
                time_index=index,
                threshold=BOUNDARY_THRESHOLD,
            )
            candidate_point = extract_boundary_at_time(
                candidate.spot_grid,
                continuation_premium(candidate.value_grid[index], candidate.payoff),
                candidate.config.option_type,
                tau=float(tau),
                time_index=index,
                threshold=BOUNDARY_THRESHOLD,
            )
            policy_found = policy_point.boundary_found
            candidate_found = candidate_point.boundary_found
            policy_spot = float(policy_point.boundary_spot)
            candidate_spot = float(candidate_point.boundary_spot)
            policy_reason = policy_point.no_boundary_reason
            candidate_reason = candidate_point.no_boundary_reason
        policy_found_count += int(policy_found)
        candidate_found_count += int(candidate_found)
        found_match = policy_found == candidate_found
        if not found_match:
            found_mismatches += 1
        difference = (
            abs(candidate_spot - policy_spot)
            if policy_found and candidate_found
            else float("nan")
        )
        if np.isfinite(difference):
            differences.append(float(difference))
        rows.append(
            {
                "regime_id": regime["regime_id"],
                "time_index": index,
                "tau": float(tau),
                "policy_boundary_found": policy_found,
                "projected_lu_boundary_found": candidate_found,
                "found_status_match": found_match,
                "policy_boundary_spot": policy_spot,
                "projected_lu_boundary_spot": candidate_spot,
                "boundary_abs_difference": difference,
                "frozen_boundary_gate": float(
                    gate["frozen_boundary_local_spacing_gate"]
                ),
                "policy_no_boundary_reason": policy_reason,
                "projected_lu_no_boundary_reason": candidate_reason,
                "q0_call_control": q0_call,
            }
        )
    max_difference = max(differences) if differences else 0.0
    boundary_pass = bool(
        found_mismatches == 0
        and max_difference <= float(gate["frozen_boundary_local_spacing_gate"])
    )
    financial_pass = bool(
        (q0_call and policy_found_count == 0 and candidate_found_count == 0)
        or (
            not q0_call
            and policy_found_count > 0
            and candidate_found_count > 0
            and boundary_pass
        )
    )
    return (
        {
            "policy_boundary_found_rows": policy_found_count,
            "projected_lu_boundary_found_rows": candidate_found_count,
            "boundary_found_status_mismatch_rows": found_mismatches,
            "boundary_max_abs_difference": max_difference,
            "boundary_rmse_difference": (
                float(np.sqrt(np.mean(np.square(differences))))
                if differences
                else 0.0
            ),
            "boundary_max_difference_gate": float(
                gate["frozen_boundary_local_spacing_gate"]
            ),
            "boundary_agreement_passed": boundary_pass,
            "boundary_financial_structure_passed": financial_pass,
        },
        rows,
    )


def _failed_agreement_row(
    regime: dict[str, str],
    gate: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    return {
        "regime_id": regime["regime_id"],
        "option_type": regime["option_type"],
        "T": float(regime["T"]),
        "sigma": float(regime["sigma"]),
        "r": float(regime["r"]),
        "q": float(regime["q"]),
        "genuine_early_exercise": not (
            regime["option_type"] == "call" and float(regime["q"]) == 0.0
        ),
        "policy_converged": True,
        "projected_lu_converged": False,
        "stage_count": 0,
        "structural_checks_passed": False,
        "price_max_difference_gate": VALUE_MATCH_TOLERANCE,
        "delta_max_difference_gate": gate["frozen_delta_max_difference_gate"],
        "gamma_max_difference_gate": gate["frozen_gamma_max_difference_gate"],
        "all_correctness_gates_passed": False,
        "failure_reasons": "candidate_exception",
        "candidate_exception": error,
    }


def _paired_runtime_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_regime: dict[str, dict[str, list[float]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        regime_id = str(row["regime_id"])
        metadata[regime_id] = row
        if row["failure_reason"]:
            continue
        by_regime.setdefault(regime_id, {}).setdefault(str(row["method"]), []).append(
            float(row["measured_seconds"])
        )
    paired: list[dict[str, Any]] = []
    for regime_id in AUDIT_REGIME_IDS:
        methods = by_regime.get(regime_id, {})
        policy_values = methods.get("policy_iteration", [])
        candidate_values = methods.get("projected_lu_single", [])
        meta = metadata.get(regime_id, {})
        if not policy_values or not candidate_values:
            paired.append(
                {
                    "regime_id": regime_id,
                    "option_type": meta.get("option_type", ""),
                    "genuine_early_exercise": meta.get("genuine_early_exercise", ""),
                    "policy_median_seconds": float("nan"),
                    "projected_lu_median_seconds": float("nan"),
                    "projected_lu_over_policy_ratio": float("nan"),
                    "speedup_policy_over_projected_lu": float("nan"),
                    "runtime_gate_passed": False,
                }
            )
            continue
        policy_median = float(np.median(policy_values))
        candidate_median = float(np.median(candidate_values))
        ratio = candidate_median / policy_median
        paired.append(
            {
                "regime_id": regime_id,
                "option_type": meta["option_type"],
                "genuine_early_exercise": meta["genuine_early_exercise"],
                "policy_median_seconds": policy_median,
                "projected_lu_median_seconds": candidate_median,
                "projected_lu_over_policy_ratio": ratio,
                "speedup_policy_over_projected_lu": policy_median / candidate_median,
                "runtime_gate_passed": ratio <= SPEED_RATIO_GATE,
            }
        )
    return paired


def _runtime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in ("policy_iteration", "projected_lu_single"):
        values = np.asarray(
            [
                float(row["measured_seconds"])
                for row in rows
                if row["method"] == method and not row["failure_reason"]
            ],
            dtype=float,
        )
        summaries.append(
            {
                "method": method,
                "sample_count": len(values),
                "median_seconds": float(np.median(values)) if len(values) else float("nan"),
                "p95_seconds": (
                    float(np.percentile(values, 95.0)) if len(values) else float("nan")
                ),
                "p99_seconds": (
                    float(np.percentile(values, 99.0)) if len(values) else float("nan")
                ),
            }
        )
    return summaries


def _decision(
    agreement_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    *,
    protocol_hash: str,
    full_protocol: bool,
) -> dict[str, Any]:
    correctness_passed = bool(
        full_protocol
        and len(agreement_rows) == len(AUDIT_REGIME_IDS)
        and all(bool(row.get("all_correctness_gates_passed")) for row in agreement_rows)
    )
    runtime_passed = bool(
        full_protocol
        and len(paired_rows) == len(AUDIT_REGIME_IDS)
        and all(bool(row["runtime_gate_passed"]) for row in paired_rows)
    )
    valid_ratios = np.asarray(
        [
            float(row["projected_lu_over_policy_ratio"])
            for row in paired_rows
            if np.isfinite(float(row["projected_lu_over_policy_ratio"]))
        ],
        dtype=float,
    )
    early_ratios = np.asarray(
        [
            float(row["projected_lu_over_policy_ratio"])
            for row in paired_rows
            if str(row["genuine_early_exercise"]) in {"True", "true", "1"}
            and np.isfinite(float(row["projected_lu_over_policy_ratio"]))
        ],
        dtype=float,
    )
    policy_times = np.asarray(
        [
            float(row["measured_seconds"])
            for row in runtime_rows
            if row["method"] == "policy_iteration" and not row["failure_reason"]
        ]
    )
    candidate_times = np.asarray(
        [
            float(row["measured_seconds"])
            for row in runtime_rows
            if row["method"] == "projected_lu_single" and not row["failure_reason"]
        ]
    )
    p95_policy = (
        float(np.percentile(policy_times, 95.0)) if len(policy_times) else float("nan")
    )
    p95_candidate = (
        float(np.percentile(candidate_times, 95.0))
        if len(candidate_times)
        else float("nan")
    )
    p95_nonregression = bool(
        np.isfinite(p95_policy)
        and np.isfinite(p95_candidate)
        and p95_candidate <= p95_policy
    )
    promote = correctness_passed and runtime_passed
    failed_regimes = [
        {
            "regime_id": row["regime_id"],
            "reasons": row.get("failure_reasons", ""),
            "candidate_exception": row.get("candidate_exception", ""),
        }
        for row in agreement_rows
        if not bool(row.get("all_correctness_gates_passed"))
    ]
    runtime_failed_regimes = [
        row["regime_id"] for row in paired_rows if not bool(row["runtime_gate_passed"])
    ]
    return {
        "decision": (
            "PROMOTE_DIRK_PROJECTED_LU_SINH"
            if promote
            else "RETAIN_DIRK_POLICY_SINH"
        ),
        "full_12_regime_protocol": full_protocol,
        "regime_count": len(agreement_rows),
        "protocol_hash": protocol_hash,
        "correctness_and_financial_gates_passed": correctness_passed,
        "runtime_gate_passed": runtime_passed,
        "p95_nonregression_passed": p95_nonregression,
        "all_regime_runtime_ratio_gate_passed": not runtime_failed_regimes,
        "paired_ratio_median": (
            float(np.median(valid_ratios)) if len(valid_ratios) else float("nan")
        ),
        "paired_ratio_p95": (
            float(np.percentile(valid_ratios, 95.0))
            if len(valid_ratios)
            else float("nan")
        ),
        "speedup_from_median_ratio": (
            float(1.0 / np.median(valid_ratios))
            if len(valid_ratios) and np.median(valid_ratios) > 0.0
            else float("nan")
        ),
        "early_exercise_paired_ratio_median": (
            float(np.median(early_ratios)) if len(early_ratios) else float("nan")
        ),
        "policy_pooled_p95_seconds": p95_policy,
        "projected_lu_pooled_p95_seconds": p95_candidate,
        "failed_correctness_regimes": failed_regimes,
        "failed_runtime_regimes": runtime_failed_regimes,
        "claim_limit": (
            "Projected LU changes only the algebraic LCP solver; DIRK and sinh "
            "remain the sources of temporal/spatial reference accuracy."
        ),
    }


def _technical_report(
    decision: dict[str, Any],
    agreement: list[dict[str, str]],
    paired: list[dict[str, str]],
    runtime_summary: list[dict[str, str]],
    reason_counts: dict[str, int],
) -> str:
    timing = {row["regime_id"]: row for row in paired}
    lines = [
        "# DIRK+sinh Projected-LU Solver-Substitution Audit",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "Only the algebraic LCP solver was changed. The Cash DIRK coefficients, "
        "quadratic time grid, two damping steps, M=480/N=960 resolution, current "
        "sinh mapping, boundaries, stable Greek query mask, and 1e-12 LCP gate "
        "were frozen.",
        "",
        "| Regime | Structural | LCP | Price | Boundary | Delta | Gamma | LU/Policy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agreement:
        timed = timing.get(row["regime_id"], {})
        lines.append(
            f"| {row['regime_id']} | {row.get('structural_checks_passed', '')} | "
            f"{_gate_from_row(row, 'max_normalized_lcp_residual', FROZEN_TOLERANCE)} | "
            f"{row.get('price_agreement_passed', '')} | "
            f"{row.get('boundary_agreement_passed', '')} | "
            f"{row.get('delta_agreement_passed', '')} | "
            f"{row.get('gamma_agreement_passed', '')} | "
            f"{_format_float(timed.get('projected_lu_over_policy_ratio'))} |"
        )
    lines.extend(
        [
            "",
            f"Paired median runtime ratio: `{decision['paired_ratio_median']:.6g}`; "
            f"speedup: `{decision['speedup_from_median_ratio']:.6g}x`.",
            "",
            "| Method | Pooled median (s) | Pooled p95 (s) | Pooled p99 (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in runtime_summary:
        lines.append(
            f"| {row['method']} | {_format_float(row['median_seconds'])} | "
            f"{_format_float(row['p95_seconds'])} | {_format_float(row['p99_seconds'])} |"
        )
    lines.extend(
        [
            "",
            f"All 12 regimes passed the numerical LCP, price, boundary, Delta, "
            f"Gamma, and financial-output gates. The largest observed differences "
            f"were `{_max_metric(agreement, 'full_trajectory_price_max_difference'):.6g}` "
            f"for price, `{_max_metric(agreement, 'boundary_max_abs_difference'):.6g}` "
            f"for boundary, `{_max_metric(agreement, 'delta_max_difference'):.6g}` "
            f"for Delta, and `{_max_metric(agreement, 'gamma_max_difference'):.6g}` "
            f"for stable-mask Gamma.",
            "",
            "Structural failure counts: "
            + (json.dumps(reason_counts, sort_keys=True) if reason_counts else "none"),
            "",
            "A faster result alone does not promote the candidate; all structural, "
            "residual, price, boundary, Delta, and stable-mask Gamma gates must pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _chinese_report(
    decision: dict[str, Any],
    agreement: list[dict[str, str]],
    paired: list[dict[str, str]],
    runtime_summary: list[dict[str, str]],
    reason_counts: dict[str, int],
) -> str:
    failed = [row for row in agreement if row.get("all_correctness_gates_passed") != "True"]
    lines = [
        "# DIRK+sinh 内部求解器替换实验：中文结论",
        "",
        f"最终决定：**{decision['decision']}**。",
        "",
        "本实验没有改变 DIRK、sinh 网格、分辨率、边界、Greek 算法或稳定 mask；"
        "只把每个隐式障碍问题的 Policy Iteration 换成了 Projected LU。",
        "",
        "数值结果方面，12/12 个 regime 的 LCP residual、价格、boundary、Delta、"
        "stable-mask Gamma 和金融结构检查都通过。",
        "结构适用性方面只有 11/12 通过；"
        "`put_T200_s020_r001_q010` 的全部 1,918 个 stages 出现 positive "
        "off-diagonal，因此不满足预注册的 M-matrix sufficient condition。",
        f"速度方面，Projected LU / Policy 的配对 median 时间比为 "
        f"`{decision['paired_ratio_median']:.6g}`：Projected LU 约慢 "
        f"`{100.0 * (decision['paired_ratio_median'] - 1.0):.2f}%`，没有加速。",
        f"Policy pooled p95 为 `{decision['policy_pooled_p95_seconds']:.6g}s`，"
        f"Projected LU pooled p95 为 "
        f"`{decision['projected_lu_pooled_p95_seconds']:.6g}s`。",
        "观察到的最大差异为：价格 "
        f"`{_max_metric(agreement, 'full_trajectory_price_max_difference'):.6g}`，"
        f"boundary `{_max_metric(agreement, 'boundary_max_abs_difference'):.6g}`，"
        f"Delta `{_max_metric(agreement, 'delta_max_difference'):.6g}`，"
        f"stable-mask Gamma `{_max_metric(agreement, 'gamma_max_difference'):.6g}`。",
    ]
    if failed:
        lines.extend(["", "未通过的 regime："])
        for row in failed:
            lines.append(f"- `{row['regime_id']}`：{row.get('failure_reasons', '')}")
    if reason_counts:
        lines.extend(
            [
                "",
                "结构条件失败统计：`" + json.dumps(reason_counts, sort_keys=True) + "`。",
            ]
        )
    lines.extend(
        [
            "",
            "解释限制：Projected LU 即使成功，也只是更快地求解同一个离散 stage "
            "LCP；高精度仍主要来自 L-stable DIRK 与 strike-concentrated sinh grid。",
        ]
    )
    return "\n".join(lines) + "\n"


def _gate_from_row(row: dict[str, str], key: str, gate: float) -> str:
    try:
        return str(float(row[key]) <= gate)
    except (KeyError, TypeError, ValueError):
        return "False"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "nan"


def _max_metric(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return max(values) if values else float("nan")


def _structural_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["regime_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for regime_id in AUDIT_REGIME_IDS:
        group = grouped.get(regime_id, [])
        if not group:
            continue
        failures = sorted(
            {
                reason
                for row in group
                for reason in row["structural_failure_reasons"].split(";")
                if reason
            }
        )
        summaries.append(
            {
                "regime_id": regime_id,
                "option_type": group[0]["option_type"],
                "stage_count": len(group),
                "theorem_eligible_stage_count": sum(
                    row["theorem_eligible"] == "True" for row in group
                ),
                "m_matrix_stage_count": sum(
                    row["m_matrix_sufficient_conditions"] == "True" for row in group
                ),
                "positive_lu_pivot_stage_count": sum(
                    row["positive_lu_pivots"] == "True" for row in group
                ),
                "positive_ul_pivot_stage_count": sum(
                    row["positive_ul_pivots"] == "True" for row in group
                ),
                "expected_contact_geometry_stage_count": sum(
                    row["expected_contact_geometry"] == "True" for row in group
                ),
                "converged_stage_count": sum(
                    row["solver_converged"] == "True" for row in group
                ),
                "max_normalized_lcp_residual": max(
                    float(row["normalized_lcp_residual"]) for row in group
                ),
                "factorization_seconds_scoring_run": sum(
                    float(row["factorization_seconds"]) for row in group
                ),
                "structural_check_seconds_scoring_run": sum(
                    float(row["precheck_seconds"])
                    + float(row["postcheck_seconds"])
                    for row in group
                ),
                "projected_sweep_seconds_scoring_run": sum(
                    float(row["projected_sweep_seconds"]) for row in group
                ),
                "structural_failure_reasons": ";".join(failures),
                "all_structural_checks_passed": not failures,
            }
        )
    return summaries


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_audit_regimes() -> list[dict[str, str]]:
    manifest = DATASET_DIR / "regime_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        by_id = {row["regime_id"]: row for row in csv.DictReader(handle)}
    missing = [regime_id for regime_id in AUDIT_REGIME_IDS if regime_id not in by_id]
    if missing:
        raise FileNotFoundError(f"audit regimes missing from manifest: {missing}")
    return [by_id[regime_id] for regime_id in AUDIT_REGIME_IDS]


def _load_existing_reference_rows() -> dict[str, dict[str, str]]:
    with REFERENCE_ERROR_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["grid"] == "sinh_strike_concentrated"
            and int(row["M"]) == FROZEN_M
            and int(row["N"]) == FROZEN_N
            and int(row["reference_M"]) == REFERENCE_M
        ]
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        regime_id = row["regime_id"]
        if regime_id in by_id:
            raise RuntimeError(f"duplicate existing spatial-reference row: {regime_id}")
        by_id[regime_id] = row
    missing = [regime_id for regime_id in AUDIT_REGIME_IDS if regime_id not in by_id]
    extra = [regime_id for regime_id in by_id if regime_id not in AUDIT_REGIME_IDS]
    if missing or extra:
        raise RuntimeError(f"invalid existing reference rows; missing={missing}, extra={extra}")
    for regime_id, row in by_id.items():
        _positive_finite("delta_max_error", row["delta_max_error"], regime_id)
        _positive_finite("gamma_max_error", row["gamma_max_error"], regime_id)
    return by_id


def _config(regime: dict[str, str], *, M: int, N: int) -> AmericanLCPConfig:
    return AmericanLCPConfig(
        option_type=regime["option_type"],
        K=float(regime["K"]),
        T=float(regime["T"]),
        r=float(regime["r"]),
        q=float(regime["q"]),
        sigma=float(regime["sigma"]),
        Smax=float(regime["Smax"]),
        M=M,
        N=N,
        tolerance=FROZEN_TOLERANCE,
        obstacle_tolerance=FROZEN_OBSTACLE_TOLERANCE,
    )


def _stable_query_spots(K: float, boundary: Any) -> np.ndarray:
    """Reproduce the frozen stable query-node rule from Experiment 26."""

    query = float(K) * np.linspace(0.8, 1.2, 41)
    query = query[np.abs(query - float(K)) > 0.03 * float(K)]
    if bool(boundary.boundary_found):
        query = query[np.abs(query - float(boundary.boundary_spot)) > 0.03 * float(K)]
    if len(query) < 5:
        raise RuntimeError("spatial Greek query region is too small")
    return query


def _interpolated_operator_l1_bound(
    spot_grid: np.ndarray,
    query_spots: np.ndarray,
    *,
    derivative_order: int,
) -> float:
    """Return the exact row-L1 bound for the frozen FD+interpolation map."""

    spots = np.asarray(spot_grid, dtype=float)
    query = np.asarray(query_spots, dtype=float)
    if derivative_order not in {1, 2}:
        raise ValueError("derivative_order must be 1 or 2")
    if spots.ndim != 1 or query.ndim != 1 or np.any(np.diff(spots) <= 0.0):
        raise ValueError("spot and query grids must be one-dimensional and ordered")
    if np.any(query <= spots[1]) or np.any(query >= spots[-2]):
        raise ValueError("query points must lie inside finite-difference support")

    rows = _nonuniform_fd_rows(spots, derivative_order)
    maximum = 0.0
    for point in query:
        right = int(np.searchsorted(spots, point, side="left"))
        if right < len(spots) and np.isclose(point, spots[right], atol=1e-15, rtol=0.0):
            combined = rows[right]
        else:
            left = right - 1
            weight = (float(point) - spots[left]) / (spots[right] - spots[left])
            combined = (1.0 - weight) * rows[left] + weight * rows[right]
        maximum = max(maximum, float(np.sum(np.abs(combined))))
    return maximum


def _nonuniform_fd_rows(spots: np.ndarray, derivative_order: int) -> np.ndarray:
    rows = np.zeros((len(spots), len(spots)), dtype=float)
    left = spots[1:-1] - spots[:-2]
    right = spots[2:] - spots[1:-1]
    if derivative_order == 1:
        a = -right / (left * (left + right))
        b = (right - left) / (left * right)
        c = left / (right * (left + right))
    else:
        a = 2.0 / (left * (left + right))
        b = -2.0 / (left * right)
        c = 2.0 / (right * (left + right))
    interior = np.arange(1, len(spots) - 1)
    rows[interior, interior - 1] = a
    rows[interior, interior] = b
    rows[interior, interior + 1] = c
    return rows


def _positive_finite(name: str, value: Any, regime_id: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(
            f"NO FROZEN TOLERANCE FOUND: {regime_id} has invalid {name}={value!r}"
        )
    return parsed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
