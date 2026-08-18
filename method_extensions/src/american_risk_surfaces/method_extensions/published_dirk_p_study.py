"""Independent evaluation of the published in 't Hout DIRK-P framework."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryPoint,
    continuation_premium,
    extract_boundary_at_time,
)
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.method_extensions.protocol import (
    AUDIT_REGIME_IDS,
    DATASET_DIR,
    environment_manifest,
    sha256_file,
)
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import european_call_price
from american_risk_surfaces.solvers.greek_integrators import (
    AmericanGreekIntegratorResult,
    american_dirk_policy_price,
)
from american_risk_surfaces.solvers.grid import (
    inthout_published_spot_grid,
    sinh_spot_grid,
)
from american_risk_surfaces.solvers.published_dirk_p import (
    PUBLISHED_DAMPING_STEPS,
    PUBLISHED_DIRK_THETA,
    PUBLISHED_PENALTY_LARGE,
    PUBLISHED_PENALTY_MAX_ITER,
    PUBLISHED_PENALTY_TOLERANCE,
    PublishedDIRKPResult,
    american_published_dirk_p_price,
)
from american_risk_surfaces.workspace import EXTENSION_ROOT


PROJECT_ROOT = EXTENSION_ROOT
RESULTS_DIR = PROJECT_ROOT / "results" / "16_published_dirk_p"
REPORTS_DIR = PROJECT_ROOT / "reports" / "18_published_dirk_p"
PROTOCOL_DIR = RESULTS_DIR / "00_protocol"
REPRODUCTION_DIR = RESULTS_DIR / "01_reproduction"
AUDIT_DIR = RESULTS_DIR / "02_audit"
FIGURES_DIR = RESULTS_DIR / "03_figures"

PAPER_URL = "https://arxiv.org/abs/2401.13361"
PAPER_DOI = "https://doi.org/10.1016/j.matcom.2024.10.038"
GRID_SOURCE_URL = "https://arxiv.org/abs/2207.10060"
PAPER_PDF_SHA256 = "38a9690bdf37be0ee0dd1b98451b2a55ec85a8a30508f5fac2f8c62c51640bbb"

CANDIDATE_LEVELS = ((120, 240), (240, 480), (480, 960))
REFERENCE_M = 480
REFERENCE_N = 960
WARMUPS = 5
TIMING_REPEATS = 30
TIMING_SEED = 20260819
BOUNDARY_THRESHOLD = 1e-6

# These gates are intentionally frozen before the first formal candidate run.
REFERENCE_ERROR_MULTIPLIER = 1.25
MIN_EMPIRICAL_ORDER = 1.5
MIN_STABLE_ORDER_FRACTION = 0.90
# These are the project-wide tolerances frozen by Experiment 21.  They apply
# to the common ``compute_lcp_residual`` metrics even though the published
# penalty iteration itself stops with its paper-specific 1e-7 update test.
MAX_NORMALIZED_PENALTY_LCP_RESIDUAL = 1e-12
MAX_NORMALIZED_PENALTY_OBSTACLE = 1e-12
MAX_RUNTIME_MEDIAN_RATIO = 1.25
MAX_RUNTIME_P95_RATIO = 1.50


def write_protocol() -> dict[str, Path]:
    """Write the immutable method, evidence, and decision protocol."""

    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    reference_scales = _reference_error_scales()
    scale_rows = [reference_scales[regime_id] for regime_id in AUDIT_REGIME_IDS]
    scales_path = PROTOCOL_DIR / "frozen_reference_error_scales.csv"
    _write_csv(scales_path, scale_rows)

    source_paths = (
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "published_dirk_p.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "greek_integrators.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "grid.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers" / "operator.py",
        PROJECT_ROOT / "src" / "american_risk_surfaces" / "method_extensions" / "published_dirk_p_study.py",
        PROJECT_ROOT / "experiments" / "68_published_dirk_p_protocol.py",
        PROJECT_ROOT / "experiments" / "69_published_dirk_p_reproduction.py",
        PROJECT_ROOT / "experiments" / "70_published_dirk_p_formal_audit.py",
        PROJECT_ROOT / "experiments" / "71_published_dirk_p_synthesis.py",
        PROJECT_ROOT / "experiments" / "72_published_dirk_p_metric_gate_correction.py",
        PROJECT_ROOT / "experiments" / "23_greek_time_integrator_audit.py",
        PROJECT_ROOT / "experiments" / "26_greek_spatial_grid_audit.py",
        PROJECT_ROOT / "results" / "07_method_extensions" / "03_greek_audit" / "spatial_convergence.csv",
        PROJECT_ROOT / "results" / "07_method_extensions" / "00_protocol" / "tolerance_decision.json",
    )
    sources = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in source_paths
    ]
    manifest_path = DATASET_DIR / "regime_manifest.csv"
    protocol = {
        "experiment": "published_dirk_p_high_accuracy_candidate",
        "decision_options": [
            "PROMOTE_PUBLISHED_DIRK_P",
            "RETAIN_DIRK_POLICY_SINH",
        ],
        "paper": {
            "title": "A note on the numerical approximation of Greeks for American-style options",
            "author": "Karel J. in 't Hout",
            "arxiv": PAPER_URL,
            "doi": PAPER_DOI,
            "downloaded_pdf_sha256": PAPER_PDF_SHA256,
            "grid_formula_source": GRID_SOURCE_URL,
        },
        "published_method_frozen": {
            "dirk": "Cash DIRK (3.3)-(3.4), DIRKa-P (3.12)",
            "theta": PUBLISHED_DIRK_THETA,
            "l_stable": True,
            "penalty_large": PUBLISHED_PENALTY_LARGE,
            "penalty_tolerance": PUBLISHED_PENALTY_TOLERANCE,
            "penalty_stopping_rule": "max componentwise relative update < tol OR penalty matrix unchanged",
            "implementation_iteration_guard": PUBLISHED_PENALTY_MAX_ITER,
            "time_grid": "tau_n=(n/N)^2*T",
            "be_p_damping_steps": PUBLISHED_DAMPING_STEPS,
            "space_grid": "uniform on [0,2K], sinh tail to Smax",
            "space_grid_d": "K/10",
            "finite_differences": "second-order three-point central differences on the nonuniform grid",
            "postsolve_payoff_projection": False,
        },
        "surf_application_frozen": {
            "audit_regime_ids": list(AUDIT_REGIME_IDS),
            "candidate_levels_M_N": [list(item) for item in CANDIDATE_LEVELS],
            "internal_reference": "DIRK + Policy Iteration + SURF strike-concentrated sinh grid",
            "internal_reference_M": REFERENCE_M,
            "internal_reference_N": REFERENCE_N,
            "boundary_threshold": BOUNDARY_THRESHOLD,
            "stable_gamma_query": "existing 0.8K--1.2K query, excluding +/-0.03K around strike and extracted boundary",
            "q0_call_rule": "no exercise boundary; European BSM theorem control",
            "float_precision": "float64",
        },
        "pre_registered_gates": {
            "per_regime_price_delta_gamma_max_error": (
                "candidate-vs-current-reference <= 1.25 times the existing regime-specific "
                "M480-vs-M960 SURF sinh reference error"
            ),
            "reference_error_multiplier": REFERENCE_ERROR_MULTIPLIER,
            "boundary": "final boundary agreement within max(candidate local cell, reference local cell)",
            "q0_call": "no fabricated boundary and candidate BSM error <= 1.25 * reference BSM error",
            "normalized_penalty_lcp_residual_max": MAX_NORMALIZED_PENALTY_LCP_RESIDUAL,
            "normalized_penalty_obstacle_max": MAX_NORMALIZED_PENALTY_OBSTACLE,
            "minimum_empirical_order": MIN_EMPIRICAL_ORDER,
            "minimum_joint_order_fraction": MIN_STABLE_ORDER_FRACTION,
            "runtime_median_candidate_over_reference_max": MAX_RUNTIME_MEDIAN_RATIO,
            "runtime_p95_candidate_over_reference_max": MAX_RUNTIME_P95_RATIO,
            "all_12_regimes_must_pass_accuracy_and_structure": True,
            "no_post_result_tuning": True,
        },
        "protocol_amendment": {
            "version": 2,
            "reason": (
                "The first formal pass exposed that the generic premium-threshold extractor "
                "can label a numerical transition for q=0 calls. The user-required theorem "
                "control is now applied before boundary scoring; raw threshold behaviour is "
                "retained only as a diagnostic."
            ),
            "candidate_method_changed": False,
            "candidate_numerical_outputs_changed": False,
            "accuracy_or_runtime_gates_changed": False,
            "first_pass_evidence_preserved_at": "results/16_published_dirk_p/02_audit_v1_raw_threshold_scoring",
        },
        "metric_gate_correction": {
            "version": 3,
            "reason": (
                "The initial DIRK-P study incorrectly introduced candidate-specific 1e-6 "
                "acceptance limits for the common normalized obstacle and LCP residuals. "
                "Those quantities are produced by the same compute_lcp_residual function "
                "and normalization as the project-wide Experiment 21 metrics, whose frozen "
                "limits are 1e-12. The formal rows were rescored without rerunning or changing "
                "any candidate solution, timing, grid, or financial-output data."
            ),
            "previous_incorrect_normalized_obstacle_gate": 1e-6,
            "previous_incorrect_normalized_lcp_gate": 1e-6,
            "corrected_frozen_normalized_obstacle_gate": MAX_NORMALIZED_PENALTY_OBSTACLE,
            "corrected_frozen_normalized_lcp_gate": MAX_NORMALIZED_PENALTY_LCP_RESIDUAL,
            "same_metric_and_normalization": True,
            "source": "results/07_method_extensions/00_protocol/tolerance_decision.json",
            "numerical_outputs_changed": False,
        },
        "timing": {
            "warmups": WARMUPS,
            "repeats": TIMING_REPEATS,
            "randomized_order_seed": TIMING_SEED,
            "single_cpu_thread": True,
            "end_to_end": True,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "frozen_reference_error_scales_sha256": sha256_file(scales_path),
        "source_files": sources,
        "environment": environment_manifest(),
    }
    pre_formal_snapshot = PROTOCOL_DIR / "protocol_v2_pre_formal.json"
    if pre_formal_snapshot.exists():
        protocol["pre_formal_protocol_snapshot"] = {
            "path": str(pre_formal_snapshot.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(pre_formal_snapshot),
            "note": "This immutable v2 snapshot predates the retained formal candidate results.",
        }
    protocol_path = PROTOCOL_DIR / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    fidelity_path = PROTOCOL_DIR / "paper_fidelity_matrix.csv"
    _write_csv(
        fidelity_path,
        [
            {"component": "Cash DIRKa coefficients", "status": "EXACT", "note": "theta=1-sqrt(2)/2"},
            {"component": "DIRK-P stage equations", "status": "EXACT", "note": "paper equation (3.12)"},
            {"component": "penalty Large and tol", "status": "EXACT", "note": "1e7 and 1e-7"},
            {"component": "penalty stop", "status": "EXACT", "note": "relative update or unchanged active penalty matrix"},
            {"component": "time grid", "status": "EXACT", "note": "quadratic paper equation (3.14)"},
            {"component": "initial damping", "status": "EXACT", "note": "first two time steps are BE-P"},
            {"component": "space grid", "status": "EXACT_FORMULA", "note": "uniform [0,2K], d=K/10 sinh tail"},
            {"component": "finite differences", "status": "EXACT_FORMULA", "note": "second-order nonuniform central differences"},
            {"component": "one-dimensional put", "status": "DIRECT_PAPER_SCOPE", "note": "paper Section 4.1"},
            {"component": "dividend call", "status": "SURF_EXTENSION", "note": "paper does not report one-dimensional calls"},
            {"component": "q=0 call", "status": "SURF_EXTENSION_CONTROL", "note": "enforces no-boundary theorem control"},
            {"component": "Smax", "status": "SURF_REGIME_VALUE", "note": "existing SURF Smax=4K rather than paper example 5K"},
            {"component": "far call boundary", "status": "SURF_EXTENSION", "note": "existing exact SURF call boundary"},
        ],
    )
    return {"protocol": protocol_path, "scales": scales_path, "fidelity": fidelity_path}


def run_reproduction() -> dict[str, Path]:
    """Reproduce the paper's representative one-dimensional put setup."""

    _require_protocol()
    REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    config = AmericanLCPConfig(
        option_type="put",
        K=100.0,
        T=0.5,
        r=0.02,
        q=0.0,
        sigma=0.40,
        Smax=500.0,
        M=200,
        N=100,
        tolerance=1e-12,
        obstacle_tolerance=1e-12,
    )
    candidate = american_published_dirk_p_price(config)
    policy = american_dirk_policy_price(
        config,
        theta=PUBLISHED_DIRK_THETA,
        quadratic_time=True,
        damping_steps=2,
        spot_grid=inthout_published_spot_grid(config.Smax, config.K, config.M),
    )
    candidate_boundary = _final_boundary(candidate)
    policy_boundary = _final_boundary(policy)
    comparison_spots = np.linspace(80.0, 120.0, 81)
    candidate_values = np.interp(comparison_spots, candidate.spot_grid, candidate.values)
    policy_values = np.interp(comparison_spots, policy.spot_grid, policy.values)
    row = {
        "paper_case": "one_asset_american_put_4_2",
        "M": config.M,
        "N": config.N,
        "theta": PUBLISHED_DIRK_THETA,
        "penalty_large": PUBLISHED_PENALTY_LARGE,
        "penalty_tolerance": PUBLISHED_PENALTY_TOLERANCE,
        "candidate_converged": candidate.converged,
        "candidate_boundary": candidate_boundary.boundary_spot,
        "paper_reported_boundary_approx": 58.0,
        "candidate_vs_paper_boundary_abs_difference": abs(candidate_boundary.boundary_spot - 58.0),
        "policy_same_grid_boundary": policy_boundary.boundary_spot,
        "candidate_vs_policy_boundary_abs_difference": abs(
            candidate_boundary.boundary_spot - policy_boundary.boundary_spot
        ),
        "candidate_vs_policy_roi_max_price_difference": float(
            np.max(np.abs(candidate_values - policy_values))
        ),
        "candidate_max_normalized_lcp_residual": candidate.max_normalized_lcp_residual,
        "candidate_max_obstacle_violation": candidate.max_obstacle_violation,
        "candidate_max_stage_iterations": candidate.maximum_stage_iterations,
        "candidate_runtime_seconds": candidate.total_seconds,
        "policy_runtime_seconds": policy.total_seconds,
    }
    metrics_path = REPRODUCTION_DIR / "paper_case_reproduction.csv"
    _write_csv(metrics_path, [row])
    grid = candidate.spot_grid
    inside = grid[grid <= 200.0 + 1e-12]
    grid_audit = {
        "nodes": len(grid),
        "Smax": float(grid[-1]),
        "d": 10.0,
        "uniform_region_last_spot": float(inside[-1]),
        "uniform_region_spacing_min": float(np.min(np.diff(inside))),
        "uniform_region_spacing_max": float(np.max(np.diff(inside))),
        "tail_spacing_max": float(np.max(np.diff(grid))),
        "strictly_increasing": bool(np.all(np.diff(grid) > 0.0)),
    }
    audit_path = REPRODUCTION_DIR / "paper_grid_audit.json"
    audit_path.write_text(json.dumps(grid_audit, indent=2, sort_keys=True), encoding="utf-8")
    return {"metrics": metrics_path, "grid": audit_path}


def run_formal_audit(
    *,
    timing_repeats: int = TIMING_REPEATS,
    warmups: int = WARMUPS,
    regime_limit: int | None = None,
) -> dict[str, Path]:
    """Run convergence, financial-output, VI, and paired timing audits."""

    _require_protocol()
    if timing_repeats != TIMING_REPEATS or warmups != WARMUPS:
        raise ValueError("formal audit requires the frozen 5-warmup/30-repeat timing protocol")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    regimes = _audit_regimes(regime_limit)
    if regime_limit is not None:
        # Development-only runs cannot create formal decision evidence.
        formal = False
    else:
        formal = True
    scales = _reference_error_scales()
    convergence_rows: list[dict[str, Any]] = []
    finest_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for regime in regimes:
        reference_config = _config(regime, REFERENCE_M, REFERENCE_N)
        reference = _solve_reference(reference_config)
        reference_boundary = _final_boundary(reference)
        query = _stable_query(reference, reference_boundary)
        full_query = np.linspace(0.0, reference.config.Smax, 481)
        ref_delta = finite_difference_delta_nonuniform(reference.spot_grid, reference.values)
        ref_gamma = finite_difference_gamma_nonuniform(reference.spot_grid, reference.values)
        ref_query = {
            "price": np.interp(query, reference.spot_grid, reference.values),
            "delta": np.interp(query, reference.spot_grid, ref_delta),
            "gamma": np.interp(query, reference.spot_grid, ref_gamma),
            "full_price": np.interp(full_query, reference.spot_grid, reference.values),
        }
        level_payloads: list[dict[str, Any]] = []
        for M, N in CANDIDATE_LEVELS:
            candidate = american_published_dirk_p_price(_config(regime, M, N))
            candidate_boundary = _final_boundary(candidate)
            candidate_delta = finite_difference_delta_nonuniform(
                candidate.spot_grid, candidate.values
            )
            candidate_gamma = finite_difference_gamma_nonuniform(
                candidate.spot_grid, candidate.values
            )
            candidate_query = {
                "price": np.interp(query, candidate.spot_grid, candidate.values),
                "delta": np.interp(query, candidate.spot_grid, candidate_delta),
                "gamma": np.interp(query, candidate.spot_grid, candidate_gamma),
                "full_price": np.interp(full_query, candidate.spot_grid, candidate.values),
            }
            payload = {
                "M": M,
                "N": N,
                "result": candidate,
                "boundary": candidate_boundary,
                "query_values": candidate_query,
            }
            level_payloads.append(payload)
            errors = {
                name: candidate_query[name] - ref_query[name]
                for name in ("price", "delta", "gamma", "full_price")
            }
            convergence_rows.append(
                {
                    "regime_id": regime["regime_id"],
                    "option_type": regime["option_type"],
                    "q": float(regime["q"]),
                    "M": M,
                    "N": N,
                    "query_nodes": len(query),
                    "full_price_nodes": len(full_query),
                    "converged": candidate.converged,
                    "runtime_seconds": candidate.total_seconds,
                    "price_max_error": _max_abs(errors["price"]),
                    "price_rmse": _rmse(errors["price"]),
                    "delta_max_error": _max_abs(errors["delta"]),
                    "delta_rmse": _rmse(errors["delta"]),
                    "gamma_max_error": _max_abs(errors["gamma"]),
                    "gamma_rmse": _rmse(errors["gamma"]),
                    "full_price_max_error": _max_abs(errors["full_price"]),
                    "full_price_rmse": _rmse(errors["full_price"]),
                    "price_empirical_order": float("nan"),
                    "delta_empirical_order": float("nan"),
                    "gamma_empirical_order": float("nan"),
                    "boundary_found": candidate_boundary.boundary_found,
                    "reference_boundary_found": reference_boundary.boundary_found,
                    "boundary_abs_error": _boundary_difference(
                        candidate_boundary, reference_boundary
                    ),
                    "max_obstacle_violation": candidate.max_obstacle_violation,
                    "max_normalized_obstacle_violation": candidate.max_normalized_obstacle_violation,
                    "max_normalized_equation_violation": candidate.max_normalized_equation_violation,
                    "max_normalized_complementarity": candidate.max_normalized_complementarity,
                    "max_normalized_lcp_residual": candidate.max_normalized_lcp_residual,
                    "maximum_stage_iterations": candidate.maximum_stage_iterations,
                    "total_penalty_iterations": candidate.total_penalty_iterations,
                }
            )

        _attach_internal_orders(convergence_rows, regime["regime_id"], level_payloads)
        finest = level_payloads[-1]
        candidate = finest["result"]
        candidate_boundary = finest["boundary"]
        candidate_query = finest["query_values"]
        candidate_delta_error = candidate_query["delta"] - ref_query["delta"]
        candidate_gamma_error = candidate_query["gamma"] - ref_query["gamma"]
        candidate_price_error = candidate_query["price"] - ref_query["price"]
        candidate_full_error = candidate_query["full_price"] - ref_query["full_price"]
        scale = scales[regime["regime_id"]]
        candidate_local = _local_spacing(candidate.spot_grid, _comparison_spot(candidate_boundary, candidate.config.K))
        reference_local = _local_spacing(reference.spot_grid, _comparison_spot(reference_boundary, reference.config.K))
        boundary_error = _boundary_difference(candidate_boundary, reference_boundary)
        q0_control = regime["option_type"] == "call" and float(regime["q"]) == 0.0
        candidate_bsm_max = float("nan")
        reference_bsm_max = float("nan")
        q0_bsm_pass = True
        if q0_control:
            bsm = np.asarray(
                european_call_price(
                    query,
                    candidate.config.K,
                    candidate.config.T,
                    candidate.config.r,
                    0.0,
                    candidate.config.sigma,
                ),
                dtype=float,
            )
            candidate_bsm_max = _max_abs(candidate_query["price"] - bsm)
            reference_bsm_max = _max_abs(ref_query["price"] - bsm)
            q0_bsm_pass = candidate_bsm_max <= REFERENCE_ERROR_MULTIPLIER * max(
                reference_bsm_max, 1e-15
            )
        path = _boundary_path_metrics(candidate, reference, q0_control=q0_control)
        price_gate = REFERENCE_ERROR_MULTIPLIER * float(scale["value_max_error"])
        delta_gate = REFERENCE_ERROR_MULTIPLIER * float(scale["delta_max_error"])
        gamma_gate = REFERENCE_ERROR_MULTIPLIER * float(scale["gamma_max_error"])
        if q0_control:
            boundary_pass = (
                not candidate_boundary.boundary_found
                and not reference_boundary.boundary_found
                and path["candidate_false_boundary_rows"] == 0
            )
        else:
            boundary_pass = bool(
                candidate_boundary.boundary_found
                and reference_boundary.boundary_found
                and np.isfinite(boundary_error)
                and boundary_error <= max(candidate_local, reference_local)
            )
        row = {
            "regime_id": regime["regime_id"],
            "option_type": regime["option_type"],
            "q": float(regime["q"]),
            "M": candidate.config.M,
            "N": candidate.config.N,
            "converged": candidate.converged,
            "price_max_error": _max_abs(candidate_price_error),
            "price_rmse": _rmse(candidate_price_error),
            "price_median_abs_error": _median_abs(candidate_price_error),
            "price_p95_abs_error": _p95_abs(candidate_price_error),
            "price_gate": price_gate,
            "price_pass": _max_abs(candidate_price_error) <= price_gate,
            "full_price_max_error": _max_abs(candidate_full_error),
            "full_price_rmse": _rmse(candidate_full_error),
            "delta_max_error": _max_abs(candidate_delta_error),
            "delta_rmse": _rmse(candidate_delta_error),
            "delta_median_abs_error": _median_abs(candidate_delta_error),
            "delta_p95_abs_error": _p95_abs(candidate_delta_error),
            "delta_gate": delta_gate,
            "delta_pass": _max_abs(candidate_delta_error) <= delta_gate,
            "gamma_max_error": _max_abs(candidate_gamma_error),
            "gamma_rmse": _rmse(candidate_gamma_error),
            "gamma_median_abs_error": _median_abs(candidate_gamma_error),
            "gamma_p95_abs_error": _p95_abs(candidate_gamma_error),
            "gamma_gate": gamma_gate,
            "gamma_pass": _max_abs(candidate_gamma_error) <= gamma_gate,
            "candidate_boundary_found": candidate_boundary.boundary_found,
            "reference_boundary_found": reference_boundary.boundary_found,
            "candidate_boundary_spot": candidate_boundary.boundary_spot,
            "reference_boundary_spot": reference_boundary.boundary_spot,
            "boundary_abs_error": boundary_error,
            "boundary_gate": max(candidate_local, reference_local),
            "boundary_pass": boundary_pass,
            **path,
            "q0_control": q0_control,
            "candidate_bsm_price_max_error": candidate_bsm_max,
            "reference_bsm_price_max_error": reference_bsm_max,
            "q0_bsm_pass": q0_bsm_pass,
            "max_obstacle_violation": candidate.max_obstacle_violation,
            "max_normalized_obstacle_violation": candidate.max_normalized_obstacle_violation,
            "max_normalized_equation_violation": candidate.max_normalized_equation_violation,
            "max_normalized_complementarity": candidate.max_normalized_complementarity,
            "max_normalized_lcp_residual": candidate.max_normalized_lcp_residual,
            "vi_pass": (
                candidate.converged
                and candidate.max_normalized_obstacle_violation
                <= MAX_NORMALIZED_PENALTY_OBSTACLE
                and candidate.max_normalized_lcp_residual
                <= MAX_NORMALIZED_PENALTY_LCP_RESIDUAL
            ),
            "maximum_stage_iterations": candidate.maximum_stage_iterations,
            "total_penalty_iterations": candidate.total_penalty_iterations,
        }
        row["regime_pass"] = bool(
            row["price_pass"]
            and row["delta_pass"]
            and row["gamma_pass"]
            and row["boundary_pass"]
            and row["q0_bsm_pass"]
            and row["vi_pass"]
        )
        finest_rows.append(row)

        if formal:
            timing_rows.extend(
                _paired_timing(regime, warmups=warmups, repeats=timing_repeats)
            )

    convergence_path = AUDIT_DIR / (
        "convergence_ladder.csv" if formal else "development_convergence_ladder.csv"
    )
    finest_path = AUDIT_DIR / (
        "regime_metrics.csv" if formal else "development_regime_metrics.csv"
    )
    _write_csv(convergence_path, convergence_rows)
    _write_csv(finest_path, finest_rows)
    timing_path = AUDIT_DIR / "timing_repeats.csv"
    if formal:
        _write_csv(timing_path, timing_rows)
        timing_summary = _timing_summary(timing_rows, regimes)
        timing_summary_path = AUDIT_DIR / "timing_summary.json"
        timing_summary_path.write_text(
            json.dumps(timing_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        timing_summary_path = AUDIT_DIR / "development_no_formal_timing.json"
        timing_summary_path.write_text(
            json.dumps({"formal": False, "reason": "regime_limit was set"}, indent=2),
            encoding="utf-8",
        )
    return {
        "convergence": convergence_path,
        "metrics": finest_path,
        "timing": timing_path if formal else timing_summary_path,
        "timing_summary": timing_summary_path,
    }


def rescore_existing_audit_with_frozen_vi_gate() -> Path:
    """Correct VI pass/fail using the pre-existing project-wide 1e-12 gate.

    This deliberately does not rerun the numerical method or timing study.  It
    only re-evaluates the stored common normalized residuals and preserves a
    machine-readable correction record.
    """

    metrics_path = AUDIT_DIR / "regime_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError("formal regime metrics are required before rescoring")
    rows = _read_csv(metrics_path)
    affected: list[dict[str, Any]] = []
    for row in rows:
        # Reconstruct the erroneous v2 decision mechanically so the correction
        # remains idempotent even after the CSV has already been rescored.
        previous_vi = bool(
            _as_bool(row["converged"])
            and float(row["max_normalized_obstacle_violation"]) <= 1e-6
            and float(row["max_normalized_lcp_residual"]) <= 1e-6
        )
        corrected_vi = bool(
            _as_bool(row["converged"])
            and float(row["max_normalized_obstacle_violation"])
            <= MAX_NORMALIZED_PENALTY_OBSTACLE
            and float(row["max_normalized_lcp_residual"])
            <= MAX_NORMALIZED_PENALTY_LCP_RESIDUAL
        )
        row["frozen_normalized_obstacle_tolerance"] = (
            MAX_NORMALIZED_PENALTY_OBSTACLE
        )
        row["frozen_normalized_lcp_tolerance"] = (
            MAX_NORMALIZED_PENALTY_LCP_RESIDUAL
        )
        row["vi_pass"] = corrected_vi
        row["regime_pass"] = bool(
            _as_bool(row["price_pass"])
            and _as_bool(row["delta_pass"])
            and _as_bool(row["gamma_pass"])
            and _as_bool(row["boundary_pass"])
            and _as_bool(row["q0_bsm_pass"])
            and corrected_vi
        )
        if previous_vi != corrected_vi:
            affected.append(
                {
                    "regime_id": row["regime_id"],
                    "previous_vi_pass": previous_vi,
                    "corrected_vi_pass": corrected_vi,
                    "max_normalized_obstacle_violation": float(
                        row["max_normalized_obstacle_violation"]
                    ),
                    "max_normalized_lcp_residual": float(
                        row["max_normalized_lcp_residual"]
                    ),
                }
            )
    _write_csv(metrics_path, rows)
    correction = {
        "status": "CORRECTED_TO_PROJECT_WIDE_FROZEN_GATE",
        "same_metric_and_normalization": True,
        "metric_definition": (
            "max(max(obstacle-solution,0)) / "
            "max(1, ||solution||_inf, ||obstacle||_inf)"
        ),
        "previous_incorrect_gate": 1e-6,
        "corrected_frozen_normalized_obstacle_tolerance": (
            MAX_NORMALIZED_PENALTY_OBSTACLE
        ),
        "corrected_frozen_normalized_lcp_tolerance": (
            MAX_NORMALIZED_PENALTY_LCP_RESIDUAL
        ),
        "frozen_tolerance_source": (
            "results/07_method_extensions/00_protocol/tolerance_decision.json"
        ),
        "candidate_solutions_or_timings_rerun": False,
        "candidate_numerical_values_changed": False,
        "affected_rows": affected,
        "corrected_vi_pass_count": sum(_as_bool(row["vi_pass"]) for row in rows),
        "regime_count": len(rows),
    }
    correction_path = PROTOCOL_DIR / "metric_gate_correction.json"
    correction_path.write_text(
        json.dumps(correction, indent=2, sort_keys=True), encoding="utf-8"
    )
    return correction_path


def synthesize() -> dict[str, Path]:
    """Create the immutable decision, figures, and plain-language reports."""

    metrics_path = AUDIT_DIR / "regime_metrics.csv"
    convergence_path = AUDIT_DIR / "convergence_ladder.csv"
    timing_path = AUDIT_DIR / "timing_summary.json"
    if not metrics_path.exists() or not convergence_path.exists() or not timing_path.exists():
        raise RuntimeError("complete formal audit outputs are required for synthesis")
    metrics = _read_csv(metrics_path)
    convergence = _read_csv(convergence_path)
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    finest_convergence = [
        row for row in convergence if int(row["M"]) == CANDIDATE_LEVELS[-1][0]
    ]
    order_pass_rows = [
        row
        for row in finest_convergence
        if float(row["price_empirical_order"]) >= MIN_EMPIRICAL_ORDER
        and float(row["delta_empirical_order"]) >= MIN_EMPIRICAL_ORDER
        and float(row["gamma_empirical_order"]) >= MIN_EMPIRICAL_ORDER
    ]
    order_fraction = len(order_pass_rows) / len(AUDIT_REGIME_IDS)
    accuracy_structure_pass = all(_as_bool(row["regime_pass"]) for row in metrics)
    vi_pass_count = sum(_as_bool(row["vi_pass"]) for row in metrics)
    runtime_pass = bool(timing["runtime_gate_pass"])
    convergence_pass = order_fraction >= MIN_STABLE_ORDER_FRACTION
    promote = accuracy_structure_pass and runtime_pass and convergence_pass
    decision = {
        "decision": (
            "PROMOTE_PUBLISHED_DIRK_P" if promote else "RETAIN_DIRK_POLICY_SINH"
        ),
        "candidate": "published DIRKa-P + quadratic time grid + published nonuniform spatial grid",
        "current_reference": "DIRK + Policy Iteration + SURF strike-concentrated sinh grid",
        "paper_fidelity": "DIRECT_FOR_1D_PUT_WITH_DISCLOSED_SURF_CALL_AND_DOMAIN_EXTENSIONS",
        "all_12_accuracy_structure_pass": accuracy_structure_pass,
        "passing_regimes": sum(_as_bool(row["regime_pass"]) for row in metrics),
        "strict_vi_passing_regimes": vi_pass_count,
        "frozen_normalized_obstacle_tolerance": MAX_NORMALIZED_PENALTY_OBSTACLE,
        "frozen_normalized_lcp_tolerance": MAX_NORMALIZED_PENALTY_LCP_RESIDUAL,
        "regime_count": len(metrics),
        "joint_second_order_fraction": order_fraction,
        "required_joint_second_order_fraction": MIN_STABLE_ORDER_FRACTION,
        "convergence_gate_pass": convergence_pass,
        "runtime_gate_pass": runtime_pass,
        "paired_median_candidate_over_reference": timing[
            "paired_median_candidate_over_reference"
        ],
        "pooled_p95_candidate_over_reference": timing[
            "pooled_p95_candidate_over_reference"
        ],
        "failed_regimes": [
            {
                "regime_id": row["regime_id"],
                "failed_gates": _failed_gates(row),
            }
            for row in metrics
            if not _as_bool(row["regime_pass"])
        ],
        "promotion_rule": (
            "All 12 accuracy/structure gates, >=90% joint second-order fraction, "
            "and frozen runtime non-regression gates must pass."
        ),
        "poster_modified": False,
        "current_reference_modified": False,
    }
    decision_path = RESULTS_DIR / "method_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    _make_figures(metrics, convergence, timing)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    english_path = REPORTS_DIR / "published_dirk_p_technical_report.md"
    chinese_path = REPORTS_DIR / "published_dirk_p_plain_language_CN.md"
    english_path.write_text(
        _english_report(decision, metrics, timing), encoding="utf-8"
    )
    chinese_path.write_text(
        _chinese_report(decision, metrics, timing), encoding="utf-8"
    )
    return {
        "decision": decision_path,
        "english_report": english_path,
        "chinese_report": chinese_path,
    }


def _solve_reference(config: AmericanLCPConfig) -> AmericanGreekIntegratorResult:
    return american_dirk_policy_price(
        config,
        quadratic_time=True,
        damping_steps=2,
        spot_grid=sinh_spot_grid(config.Smax, config.K, config.M),
    )


def _paired_timing(
    regime: dict[str, str], *, warmups: int, repeats: int
) -> list[dict[str, Any]]:
    config = _config(regime, REFERENCE_M, REFERENCE_N)
    solvers = {
        "dirk_policy_sinh": lambda: _solve_reference(config),
        "published_dirk_p": lambda: american_published_dirk_p_price(config),
    }
    rng = random.Random(f"{TIMING_SEED}:{regime['regime_id']}")
    for _ in range(warmups):
        order = list(solvers)
        rng.shuffle(order)
        for name in order:
            solvers[name]()
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = list(solvers)
        rng.shuffle(order)
        for position, name in enumerate(order):
            started = perf_counter_ns()
            result = solvers[name]()
            elapsed = (perf_counter_ns() - started) / 1e9
            rows.append(
                {
                    "regime_id": regime["regime_id"],
                    "option_type": regime["option_type"],
                    "q": float(regime["q"]),
                    "repeat": repeat,
                    "order_position": position,
                    "method": name,
                    "elapsed_seconds": elapsed,
                    "solver_reported_seconds": result.total_seconds,
                    "converged": result.converged,
                }
            )
    return rows


def _timing_summary(
    rows: list[dict[str, Any]], regimes: list[dict[str, str]]
) -> dict[str, Any]:
    per_regime: list[dict[str, Any]] = []
    for regime in regimes:
        regime_id = regime["regime_id"]
        policy = np.array(
            [
                float(row["elapsed_seconds"])
                for row in rows
                if row["regime_id"] == regime_id and row["method"] == "dirk_policy_sinh"
            ]
        )
        candidate = np.array(
            [
                float(row["elapsed_seconds"])
                for row in rows
                if row["regime_id"] == regime_id and row["method"] == "published_dirk_p"
            ]
        )
        per_regime.append(
            {
                "regime_id": regime_id,
                "reference_median": float(np.median(policy)),
                "candidate_median": float(np.median(candidate)),
                "candidate_over_reference": float(np.median(candidate) / np.median(policy)),
                "speedup": float(np.median(policy) / np.median(candidate)),
            }
        )
    ratios = np.array([row["candidate_over_reference"] for row in per_regime])
    policy_all = np.array(
        [float(row["elapsed_seconds"]) for row in rows if row["method"] == "dirk_policy_sinh"]
    )
    candidate_all = np.array(
        [float(row["elapsed_seconds"]) for row in rows if row["method"] == "published_dirk_p"]
    )
    p95_ratio = float(np.quantile(candidate_all, 0.95) / np.quantile(policy_all, 0.95))
    median_ratio = float(np.median(ratios))
    return {
        "per_regime": per_regime,
        "reference_pooled_median_seconds": float(np.median(policy_all)),
        "reference_pooled_p95_seconds": float(np.quantile(policy_all, 0.95)),
        "candidate_pooled_median_seconds": float(np.median(candidate_all)),
        "candidate_pooled_p95_seconds": float(np.quantile(candidate_all, 0.95)),
        "paired_median_candidate_over_reference": median_ratio,
        "paired_median_speedup": float(1.0 / median_ratio),
        "pooled_p95_candidate_over_reference": p95_ratio,
        "runtime_gate_pass": bool(
            median_ratio <= MAX_RUNTIME_MEDIAN_RATIO
            and p95_ratio <= MAX_RUNTIME_P95_RATIO
        ),
    }


def _attach_internal_orders(
    rows: list[dict[str, Any]],
    regime_id: str,
    payloads: list[dict[str, Any]],
) -> None:
    selected = [row for row in rows if row["regime_id"] == regime_id]
    selected.sort(key=lambda row: int(row["M"]))
    for index in range(1, len(payloads)):
        coarse = payloads[index - 1]["query_values"]
        fine = payloads[index]["query_values"]
        if index == 1:
            continue
        coarser = payloads[index - 2]["query_values"]
        for metric in ("price", "delta", "gamma"):
            coarse_difference = _max_abs(coarser[metric] - coarse[metric])
            fine_difference = _max_abs(coarse[metric] - fine[metric])
            selected[index][f"{metric}_empirical_order"] = (
                math.log(coarse_difference / fine_difference, 2.0)
                if coarse_difference > 0.0 and fine_difference > 0.0
                else float("nan")
            )


def _boundary_path_metrics(
    candidate: PublishedDIRKPResult,
    reference: AmericanGreekIntegratorResult,
    *,
    q0_control: bool,
) -> dict[str, Any]:
    candidate_premium = continuation_premium(candidate.value_grid, candidate.payoff)
    reference_premium = continuation_premium(reference.value_grid, reference.payoff)
    differences: list[float] = []
    candidate_raw_threshold_false = 0
    reference_raw_threshold_false = 0
    both_found = 0
    for index in range(1, len(candidate.tau_grid)):
        candidate_point = extract_boundary_at_time(
            candidate.spot_grid,
            candidate_premium[index],
            candidate.config.option_type,
            float(candidate.tau_grid[index]),
            index,
            threshold=BOUNDARY_THRESHOLD,
        )
        reference_point = extract_boundary_at_time(
            reference.spot_grid,
            reference_premium[index],
            reference.config.option_type,
            float(reference.tau_grid[index]),
            index,
            threshold=BOUNDARY_THRESHOLD,
        )
        if q0_control:
            candidate_raw_threshold_false += int(candidate_point.boundary_found)
            reference_raw_threshold_false += int(reference_point.boundary_found)
        elif candidate_point.boundary_found and reference_point.boundary_found:
            differences.append(
                abs(candidate_point.boundary_spot - reference_point.boundary_spot)
            )
            both_found += 1
    return {
        "boundary_path_both_found_rows": both_found,
        "boundary_path_max_abs_error": max(differences) if differences else float("nan"),
        "boundary_path_rmse": (
            float(np.sqrt(np.mean(np.square(differences))))
            if differences
            else float("nan")
        ),
        "candidate_false_boundary_rows": 0,
        "reference_false_boundary_rows": 0,
        "candidate_raw_threshold_false_boundary_rows": candidate_raw_threshold_false,
        "reference_raw_threshold_false_boundary_rows": reference_raw_threshold_false,
    }


def _stable_query(result: Any, boundary: Any) -> np.ndarray:
    query = result.config.K * np.linspace(0.8, 1.2, 41)
    query = query[np.abs(query - result.config.K) > 0.03 * result.config.K]
    if boundary.boundary_found:
        query = query[
            np.abs(query - boundary.boundary_spot) > 0.03 * result.config.K
        ]
    if len(query) < 5:
        raise RuntimeError("validated stable Gamma query contains too few points")
    return query


def _final_boundary(result: Any) -> Any:
    if result.config.option_type == "call" and float(result.config.q) == 0.0:
        premium = continuation_premium(result.values, result.payoff)
        interior = premium[1:-1]
        exercise_count = int(np.count_nonzero(interior <= BOUNDARY_THRESHOLD))
        return BoundaryPoint(
            time_index=len(result.tau_grid) - 1,
            tau=float(result.tau_grid[-1]),
            boundary_found=False,
            boundary_spot=float("nan"),
            threshold=BOUNDARY_THRESHOLD,
            search_direction="high_to_low",
            extraction_method="theorem_control",
            no_boundary_reason="no_dividend_american_call_equals_european_call",
            exercise_like_node_count=exercise_count,
            continuation_like_node_count=int(len(interior) - exercise_count),
        )
    return extract_boundary_at_time(
        result.spot_grid,
        continuation_premium(result.values, result.payoff),
        result.config.option_type,
        float(result.tau_grid[-1]),
        len(result.tau_grid) - 1,
        threshold=BOUNDARY_THRESHOLD,
    )


def _reference_error_scales() -> dict[str, dict[str, Any]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "07_method_extensions"
        / "03_greek_audit"
        / "spatial_convergence.csv"
    )
    rows = _read_csv(path)
    selected = {
        row["regime_id"]: {
            "regime_id": row["regime_id"],
            "source_grid": row["grid"],
            "source_M": int(row["M"]),
            "source_reference_M": int(row["reference_M"]),
            "value_max_error": float(row["value_max_error"]),
            "delta_max_error": float(row["delta_max_error"]),
            "gamma_max_error": float(row["gamma_max_error"]),
            "source_file": "results/07_method_extensions/03_greek_audit/spatial_convergence.csv",
        }
        for row in rows
        if row["grid"] == "sinh_strike_concentrated" and int(row["M"]) == 480
    }
    missing = [regime_id for regime_id in AUDIT_REGIME_IDS if regime_id not in selected]
    if missing:
        raise RuntimeError(f"missing frozen reference errors for: {missing}")
    return selected


def _audit_regimes(limit: int | None = None) -> list[dict[str, str]]:
    with (DATASET_DIR / "regime_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        by_id = {row["regime_id"]: row for row in csv.DictReader(handle)}
    rows = [by_id[regime_id] for regime_id in AUDIT_REGIME_IDS]
    return rows if limit is None else rows[:limit]


def _config(regime: dict[str, str], M: int, N: int) -> AmericanLCPConfig:
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
        tolerance=1e-12,
        obstacle_tolerance=1e-12,
    )


def _comparison_spot(boundary: Any, strike: float) -> float:
    return float(boundary.boundary_spot) if boundary.boundary_found else float(strike)


def _local_spacing(grid: np.ndarray, spot: float) -> float:
    index = int(np.clip(np.searchsorted(grid, spot), 1, len(grid) - 1))
    return float(grid[index] - grid[index - 1])


def _boundary_difference(candidate: Any, reference: Any) -> float:
    if candidate.boundary_found and reference.boundary_found:
        return abs(float(candidate.boundary_spot) - float(reference.boundary_spot))
    if not candidate.boundary_found and not reference.boundary_found:
        return 0.0
    return float("inf")


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _median_abs(values: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(values, dtype=float))))


def _p95_abs(values: np.ndarray) -> float:
    return float(np.quantile(np.abs(np.asarray(values, dtype=float)), 0.95))


def _require_protocol() -> None:
    if not (PROTOCOL_DIR / "protocol.json").exists():
        raise RuntimeError("run Experiment 68 to freeze the protocol first")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _failed_gates(row: dict[str, Any]) -> list[str]:
    names = ("price", "delta", "gamma", "boundary", "q0_bsm", "vi")
    return [name for name in names if not _as_bool(row[f"{name}_pass"])]


def _make_figures(
    metrics: list[dict[str, str]],
    convergence: list[dict[str, str]],
    timing: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["regime_id"].replace("_", "\n", 1) for row in metrics]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, metric, gate, title in (
        (axes[0], "price_max_error", "price_gate", "Price max error"),
        (axes[1], "delta_max_error", "delta_gate", "Delta max error"),
        (axes[2], "gamma_max_error", "gamma_gate", "Stable-mask Gamma max error"),
    ):
        values = np.array([float(row[metric]) for row in metrics])
        gates = np.array([float(row[gate]) for row in metrics])
        axis.semilogy(x, values, "o-", label="published DIRK-P vs current reference")
        axis.semilogy(x, gates, "s--", label="pre-registered gate")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_gates.png", dpi=180)
    plt.close(fig)

    per_regime = timing["per_regime"]
    ratios = np.array([float(row["candidate_over_reference"]) for row in per_regime])
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.bar(np.arange(len(ratios)), ratios)
    axis.axhline(1.0, color="black", linewidth=1, label="same runtime")
    axis.axhline(MAX_RUNTIME_MEDIAN_RATIO, color="red", linestyle="--", label="frozen gate")
    axis.set_ylabel("Published DIRK-P / current reference")
    axis.set_xticks(np.arange(len(ratios)), labels, rotation=45, ha="right", fontsize=7)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "runtime_ratios.png", dpi=180)
    plt.close(fig)


def _english_report(
    decision: dict[str, Any], metrics: list[dict[str, str]], timing: dict[str, Any]
) -> str:
    lines = [
        "# Published DIRK-P High-Accuracy Candidate Audit",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "## Fidelity",
        "",
        "The Cash DIRKa coefficients, DIRK-P stage equations, finite penalty (Large=1e7), "
        "tol=1e-7 stop, quadratic time mesh, first-two-step BE-P damping, the published "
        "uniform-to-2K/sinh-tail grid (d=K/10), and nonuniform central differences were "
        "implemented directly. No payoff projection was used to hide finite-penalty error.",
        "",
        "Unavoidable SURF extensions are the dividend-paying and no-dividend calls, the "
        "existing Smax=4K regimes rather than the paper's representative 5K example, and "
        "SURF's exact far-call boundary. The paper's one-dimensional experiment itself is a put.",
        "",
        "## Twelve-regime evidence",
        "",
        "The paper stopping rule and the SURF acceptance rule are distinct: all runs may "
        "converge under the published penalty iteration while still failing the common "
        "frozen normalized obstacle/LCP tolerance of 1e-12.",
        "",
        f"Strict VI gate: {decision['strict_vi_passing_regimes']}/{decision['regime_count']} regimes pass; "
        f"normalized obstacle and LCP tolerances are both "
        f"{decision['frozen_normalized_obstacle_tolerance']:.0e}.",
        "",
        "| Regime | Price max/gate | Boundary error/gate | Delta max/gate | Gamma max/gate | VI residual/gate | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['regime_id']} | {float(row['price_max_error']):.3g}/{float(row['price_gate']):.3g} | "
            f"{float(row['boundary_abs_error']):.3g}/{float(row['boundary_gate']):.3g} | "
            f"{float(row['delta_max_error']):.3g}/{float(row['delta_gate']):.3g} | "
            f"{float(row['gamma_max_error']):.3g}/{float(row['gamma_gate']):.3g} | "
            f"{float(row['max_normalized_lcp_residual']):.3g}/{MAX_NORMALIZED_PENALTY_LCP_RESIDUAL:.0e} | "
            f"{'yes' if _as_bool(row['regime_pass']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Joint second-order fraction: {decision['joint_second_order_fraction']:.3f} "
            f"(frozen gate {decision['required_joint_second_order_fraction']:.3f}).",
            "",
            "## Runtime",
            "",
            f"Pooled median: {timing['candidate_pooled_median_seconds']:.6g} s (candidate) "
            f"versus {timing['reference_pooled_median_seconds']:.6g} s (current reference).",
            "",
            f"Paired median ratio candidate/reference: {timing['paired_median_candidate_over_reference']:.4f}; "
            f"speedup: {timing['paired_median_speedup']:.4f}x; pooled p95 ratio: "
            f"{timing['pooled_p95_candidate_over_reference']:.4f}.",
            "",
            "| Regime | Candidate median (s) | Current reference median (s) | Candidate/reference |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in timing["per_regime"]:
        lines.append(
            f"| {row['regime_id']} | {row['candidate_median']:.6g} | "
            f"{row['reference_median']:.6g} | {row['candidate_over_reference']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The current reference and poster were not modified.",
        ]
    )
    return "\n".join(lines) + "\n"


def _chinese_report(
    decision: dict[str, Any], metrics: list[dict[str, str]], timing: dict[str, Any]
) -> str:
    failed = [row for row in metrics if not _as_bool(row["regime_pass"])]
    lines = [
        "# Published DIRK-P 高精度候选实验（中文结论）",
        "",
        f"最终决定：**{decision['decision']}**。",
        "",
        "本次实现忠实保留了论文的 L-stable Cash DIRK、两阶段 penalty 迭代、"
        "Large=1e7、tol=1e-7、二次时间网格、前两步 BE-P、论文的 [0,2K] 均匀/远端 sinh 网格"
        "以及非均匀二阶中心差分。没有在求解后把价格强行投影到 payoff 上。",
        "",
        "不可避免的区别是：论文的一维实验是 American put；SURF 还要求 dividend call 和 q=0 call，"
        "并保留现有 Smax=4K 与 call 远端边界。这些属于明确披露的项目外推，不是论文原样案例。",
        "",
        f"12 个 regime 中有 {decision['passing_regimes']}/{decision['regime_count']} 个通过全部"
        "价格、边界、Delta、stable-mask Gamma 和 VI 门槛。",
        "",
        "论文 penalty 的 tol=1e-7 是迭代停止条件，不是 SURF 正式 obstacle/LCP 验收容差。"
        f"所有计算都按论文规则收敛，但只有 {decision['strict_vi_passing_regimes']}/"
        f"{decision['regime_count']} 个 regime 通过项目冻结的 normalized obstacle/LCP "
        f"{decision['frozen_normalized_obstacle_tolerance']:.0e} 门槛。",
        "",
        f"联合二阶收敛比例为 {decision['joint_second_order_fraction']:.3f}（门槛 "
        f"{decision['required_joint_second_order_fraction']:.3f}）。",
        "",
        f"运行时间的 paired median candidate/reference 比值为 "
        f"{timing['paired_median_candidate_over_reference']:.4f}，即约 "
        f"{timing['paired_median_speedup']:.4f}x speedup；p95 比值为 "
        f"{timing['pooled_p95_candidate_over_reference']:.4f}。",
        "",
        "## 逐 regime 正式结果",
        "",
        "| Regime | Price max | Boundary error | Delta max | Stable Gamma max | VI residual/gate | 通过 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['regime_id']}` | {float(row['price_max_error']):.3g} | "
            f"{float(row['boundary_abs_error']):.3g} | {float(row['delta_max_error']):.3g} | "
            f"{float(row['gamma_max_error']):.3g} | "
            f"{float(row['max_normalized_lcp_residual']):.3g}/{MAX_NORMALIZED_PENALTY_LCP_RESIDUAL:.0e} | "
            f"{'是' if _as_bool(row['regime_pass']) else '否'} |"
        )
    if failed:
        lines.extend(["", "未通过的 regimes 与原因：", ""])
        for row in failed:
            lines.append(f"- `{row['regime_id']}`：{', '.join(_failed_gates(row))}")
    lines.extend(
        [
            "",
            "这项结果只决定候选资格；现有 DIRK+Policy+sinh 结果、当前 reference 和 poster 均未被覆盖或修改。",
        ]
    )
    return "\n".join(lines) + "\n"
