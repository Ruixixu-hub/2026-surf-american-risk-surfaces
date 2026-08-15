"""Leakage-safe orchestration for Experiments 43--45."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.boundary_aligned_basis.alignment import (
    extract_oracle_boundary_path,
)
from american_risk_surfaces.boundary_aligned_basis.basis import (
    build_oracle_basis_ladder,
    load_oracle_basis,
    save_oracle_basis,
)
from american_risk_surfaces.boundary_aligned_basis.evaluation import evaluate_oracle_basis
from american_risk_surfaces.boundary_aligned_basis.protocol import (
    DIMENSION_LADDER,
    LOCAL_BIN_LADDER,
    REDUCTION_RMSE_LIMIT,
    RESULTS_DIR,
    TOTAL_RMSE_FLOOR,
    assert_oracle_regime_allowed,
    train_snapshot_paths,
)
from american_risk_surfaces.boundary_aligned_basis.types import BoundaryAlignmentConfig
from american_risk_surfaces.reduced_order.metrics import (
    interpolate_reference_surface,
    score_value_trajectory,
)
from american_risk_surfaces.reduced_order.protocol import RBRegime, load_regimes
from american_risk_surfaces.reduced_order.snapshots import (
    boundary_lift_grid,
    trajectory_multipliers,
)
from american_risk_surfaces.reduced_order.types import RBFOMSnapshot
from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid


BASIS_DIR = RESULTS_DIR / "02_basis"
VALIDATION_DIR = RESULTS_DIR / "03_validation"


def build_basis_grid(
    *,
    arms: Iterable[str],
    dimensions: Iterable[int] = DIMENSION_LADDER,
    bin_counts: Iterable[int] = LOCAL_BIN_LADDER,
    canonical_points_by_family: dict[str, int] | None = None,
) -> tuple[list[Path], Path]:
    paths: list[Path] = []
    manifest_path = BASIS_DIR / "basis_manifest.csv"
    manifest: list[dict[str, object]] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
    requested_arms = {str(item).upper() for item in arms}
    manifest = [row for row in manifest if str(row.get("arm", "")).upper() not in requested_arms]
    for family in ("put", "call"):
        for arm in sorted(requested_arms):
            counts = tuple(bin_counts) if arm in {"L", "AL"} else (None,)
            for count in counts:
                started = perf_counter()
                label = "global" if count is None else f"k{count}"
                try:
                    alignment = None
                    if arm in {"A", "AL"}:
                        points = (canonical_points_by_family or {}).get(family)
                        if points is None:
                            raise RuntimeError("aligned arm did not pass Experiment 42")
                        alignment = BoundaryAlignmentConfig(canonical_points=points)
                    ladder = build_oracle_basis_ladder(
                        train_snapshot_paths(family),
                        arm,
                        family,
                        dimensions=dimensions,
                        bin_count=count,
                        alignment_config=alignment,
                    )
                except (RuntimeError, ValueError) as error:
                    manifest.append(
                        {
                            "arm": arm,
                            "option_type": family,
                            "bin_count": count or 1,
                            "dimension": "",
                            "status": "FAILED",
                            "failure_reason": str(error),
                            "construction_seconds": perf_counter() - started,
                        }
                    )
                    continue
                for dimension, artifact in ladder.items():
                    path = BASIS_DIR / arm / family / label / f"basis_{dimension:02d}.npz"
                    save_oracle_basis(artifact, path)
                    paths.append(path)
                    manifest.append(
                        {
                            "arm": arm,
                            "option_type": family,
                            "bin_count": count or 1,
                            "actual_bin_count": len(artifact.bin_labels),
                            "dimension": dimension,
                            "stored_primal_modes": artifact.metadata["stored_primal_modes"],
                            "stored_dual_generators": artifact.metadata["stored_dual_generators"],
                            "artifact_bytes": path.stat().st_size,
                            "status": "COMPLETE",
                            "failure_reason": "",
                            "construction_seconds": artifact.metadata["construction_seconds"],
                            "path": str(path.relative_to(RESULTS_DIR)),
                        }
                    )
    deduplicated = []
    seen = set()
    for row in reversed(manifest):
        key = (str(row["arm"]), str(row["option_type"]), str(row["bin_count"]), str(row["dimension"]))
        if key not in seen:
            deduplicated.append(row)
            seen.add(key)
    manifest = list(reversed(deduplicated))
    _write_csv(manifest_path, manifest)
    _write_greedy_histories(paths)
    return paths, manifest_path


def evaluate_validation_grid(
    artifact_paths: Iterable[Path | str],
    *,
    reference_m: int = 480,
    reference_n: int = 960,
) -> list[dict[str, object]]:
    artifacts = [load_oracle_basis(path) for path in sorted(map(Path, artifact_paths))]
    if not artifacts:
        return []
    rows: list[dict[str, object]] = []
    regimes = load_regimes(splits=("validation",))
    for regime in regimes:
        assert_oracle_regime_allowed(regime.split, regime.option_type, regime.q)
        family_artifacts = [item for item in artifacts if item.option_type == regime.option_type]
        if not family_artifacts:
            continue
        config = regime.config()
        full = american_cn_lcp_price(config, lcp_solver="policy_iteration")
        reference_multiplier, _, _ = trajectory_multipliers(config, full.value_grid)
        snapshot = RBFOMSnapshot(
            regime.regime_id,
            regime.option_type,
            full.spot_grid,
            full.tau_grid,
            full.payoff,
            full.value_grid,
            boundary_lift_grid(config, full.spot_grid, full.tau_grid),
            full.value_grid[:, 1:-1] - boundary_lift_grid(config, full.spot_grid, full.tau_grid)[:, 1:-1],
            reference_multiplier,
            reference_multiplier > 1e-10,
            np.zeros((config.N + 1, 4)),
            {"regime": {"split": "validation", "q": regime.q}},
        )
        boundary, found = extract_oracle_boundary_path(snapshot)
        fine_config = type(config)(
            config.option_type,
            config.K,
            config.T,
            config.r,
            config.q,
            config.sigma,
            config.Smax,
            reference_m,
            reference_n,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        fine = american_dirk_policy_price(
            fine_config,
            quadratic_time=True,
            damping_steps=2,
            spot_grid=sinh_spot_grid(config.Smax, config.K, reference_m),
        )
        high = interpolate_reference_surface(
            fine.value_grid,
            fine.spot_grid,
            fine.tau_grid,
            full.spot_grid,
            full.tau_grid,
        )
        cn_high = score_value_trajectory(
            full.value_grid,
            high,
            full.payoff,
            full.spot_grid,
            full.tau_grid,
            regime.option_type,
        )
        for artifact in family_artifacts:
            alignment = BoundaryAlignmentConfig(canonical_points=len(artifact.metric_grids[0]))
            try:
                result = evaluate_oracle_basis(
                    artifact,
                    config,
                    full.value_grid,
                    alignment_config=alignment,
                    boundary_path=boundary,
                    boundary_found=found,
                )
                high_metrics = score_value_trajectory(
                    result.projected_value_grid,
                    high,
                    full.payoff,
                    full.spot_grid,
                    full.tau_grid,
                    regime.option_type,
                )
                failure = ""
                finite = bool(np.all(np.isfinite(result.projected_value_grid)))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                result = None
                high_metrics = {}
                failure = str(error)
                finite = False
            base: dict[str, object] = {
                "regime_id": regime.regime_id,
                "option_type": regime.option_type,
                "split": regime.split,
                "arm": artifact.arm,
                "dimension": artifact.active_dimension,
                "bin_count": len(artifact.bin_labels),
                "stored_primal_modes": artifact.metadata["stored_primal_modes"],
                "stored_dual_generators": artifact.metadata["stored_dual_generators"],
                "finite_reconstruction": finite,
                "failure_reason": failure,
                "oracle_information_used": ";".join(result.oracle_information_used) if result else "",
                "cn_high_price_rmse": cn_high["price_rmse"],
                "cn_high_delta_rmse": cn_high["delta_rmse"],
                "cn_high_stable_gamma_rmse": cn_high["stable_gamma_rmse"],
            }
            if result is not None:
                base.update({f"reduction_{key}": value for key, value in result.metrics.items()})
                base.update({f"high_{key}": value for key, value in high_metrics.items()})
            rows.append(base)
    return rows


def make_validation_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    decision: dict[str, object] = {"families": {}}
    for family in ("put", "call"):
        family_rows = [row for row in rows if row["option_type"] == family]
        candidates = sorted(
            {
                (str(row["arm"]), int(row["dimension"]), int(row["bin_count"]))
                for row in family_rows
                if str(row["arm"]) in {"A", "L", "AL"}
            }
        )
        summaries = []
        expected_regime_count = sum(
            1
            for regime in load_regimes(splits=("validation",), option_type=family)
            if not (family == "call" and regime.q == 0.0)
        )
        for arm, dimension, bins in candidates:
            selected = [
                row for row in family_rows
                if row["arm"] == arm
                and int(row["dimension"]) == dimension
                and int(row["bin_count"]) == bins
            ]
            unaligned = {
                row["regime_id"]: row for row in family_rows
                if row["arm"] == "U" and int(row["dimension"]) == dimension
            }
            all_finite = len(selected) == expected_regime_count and all(
                _as_bool(row["finite_reconstruction"]) for row in selected
            )
            if not all_finite or not all(row["regime_id"] in unaligned for row in selected):
                passed = False
                summary = {"arm": arm, "dimension": dimension, "bin_count": bins, "passed": False}
            else:
                worst_reduction = max(float(row["reduction_projected_price_rmse"]) for row in selected)
                total_ok = all(
                    float(row["high_price_rmse"])
                    <= max(TOTAL_RMSE_FLOOR, 1.25 * float(row["cn_high_price_rmse"]))
                    for row in selected
                )
                boundary = max(_finite_or_inf(row.get("reduction_boundary_conditional_mae")) for row in selected)
                delta_ratio = max(
                    float(row["high_delta_rmse"]) / max(float(row["cn_high_delta_rmse"]), 1e-15)
                    for row in selected
                )
                gamma_ratio = max(
                    float(row["high_stable_gamma_rmse"])
                    / max(float(row["cn_high_stable_gamma_rmse"]), 1e-15)
                    for row in selected
                )
                obstacle = max(float(row["reduction_projected_obstacle_violation"]) for row in selected)
                f1 = min(float(row["reduction_active_set_f1"]) for row in selected)
                residual_ratio = max(
                    float(row["reduction_projected_full_lcp_residual"])
                    / max(float(unaligned[row["regime_id"]]["reduction_projected_full_lcp_residual"]), 1e-15)
                    for row in selected
                )
                baseline_boundary = max(
                    _finite_or_inf(unaligned[row["regime_id"]].get("reduction_boundary_conditional_mae"))
                    for row in selected
                )
                key_improvement = 1.0 - boundary / max(baseline_boundary, 1e-15)
                passed = bool(
                    worst_reduction <= REDUCTION_RMSE_LIMIT
                    and total_ok
                    and boundary <= 0.066667
                    and delta_ratio <= 1.25
                    and gamma_ratio <= 1.25
                    and obstacle <= 1e-12
                    and f1 >= 0.98
                    and residual_ratio <= 0.50
                    and key_improvement >= 0.50
                )
                summary = {
                    "arm": arm,
                    "dimension": dimension,
                    "bin_count": bins,
                    "stored_primal_modes": selected[0]["stored_primal_modes"],
                    "stored_dual_generators": selected[0]["stored_dual_generators"],
                    "worst_reduction_price_rmse": worst_reduction,
                    "total_error_pass": total_ok,
                    "worst_boundary_mae": boundary,
                    "worst_delta_ratio": delta_ratio,
                    "worst_gamma_ratio": gamma_ratio,
                    "worst_projected_obstacle": obstacle,
                    "minimum_active_f1": f1,
                    "worst_residual_ratio_vs_U": residual_ratio,
                    "boundary_improvement_vs_U": key_improvement,
                    "passed": passed,
                }
            summaries.append(summary)
        passing = [item for item in summaries if item["passed"]]
        if passing:
            chosen = min(
                passing,
                key=lambda item: (
                    item["dimension"],
                    int(item.get("stored_primal_modes", 999)) + int(item.get("stored_dual_generators", 999)),
                    item.get("worst_boundary_mae", float("inf")),
                ),
            )
            status = {
                "A": "GO_ORACLE_ALIGNMENT",
                "L": "GO_ORACLE_LOCALIZATION",
                "AL": "GO_ORACLE_ALIGNED_LOCALIZED",
            }[chosen["arm"]]
        else:
            chosen = None
            status = "STOP_RB_ROUTE"
        decision["families"][family] = {
            "status": status,
            "selected": chosen,
            "configurations": summaries,
        }
    statuses = [decision["families"][family]["status"] for family in ("put", "call")]
    if all(status.startswith("GO_") for status in statuses):
        decision["status"] = "GO_ORACLE_BASIS"
    elif any(status.startswith("GO_") for status in statuses):
        decision["status"] = "PARTIAL_GO"
    else:
        decision["status"] = "STOP_RB_ROUTE"
    decision["heldout_accessed"] = False
    return decision


def _write_greedy_histories(paths: Iterable[Path]) -> None:
    rows: list[dict[str, object]] = []
    for path in paths:
        artifact = load_oracle_basis(path)
        if artifact.active_dimension != 32 and not (
            artifact.metadata.get("actual_bin_count", 1) * artifact.active_dimension >= 120
        ):
            continue
        for kind, key in (("primal", "primal_history"), ("dual", "dual_history")):
            for label, histories in artifact.metadata[key].items():
                for row in histories:
                    rows.append(
                        {
                            "arm": artifact.arm,
                            "option_type": artifact.option_type,
                            "bin_label": label,
                            "kind": kind,
                            **row,
                        }
                    )
    if rows:
        _write_csv(BASIS_DIR / "greedy_decay_history.csv", rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_outputs(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    ladder = VALIDATION_DIR / "oracle_validation_ladder.csv"
    _write_csv(ladder, rows)
    decision = make_validation_decision(rows)
    path = VALIDATION_DIR / "method_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return ladder, path


def _finite_or_inf(value: object) -> float:
    if value is None:
        return float("inf")
    converted = float(value)
    return converted if np.isfinite(converted) else float("inf")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"
