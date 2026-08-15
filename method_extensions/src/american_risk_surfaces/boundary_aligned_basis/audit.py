"""Transformation and extraction audits that gate aligned arms."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from american_risk_surfaces.boundary_aligned_basis.alignment import (
    align_dual_multiplier,
    align_primal_state,
    build_boundary_alignment_map,
    extract_oracle_boundary_path,
    inverse_align_dual_multiplier,
    inverse_align_primal_state,
    pairing_relative_error,
    sanitize_snapshot_multiplier,
)
from american_risk_surfaces.boundary_aligned_basis.types import BoundaryAlignmentConfig
from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
from american_risk_surfaces.reduced_order.snapshots import load_snapshot


def audit_alignment_resolution(
    snapshot_paths: Iterable[Path | str],
    config: BoundaryAlignmentConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    family = ""
    for path in sorted(map(Path, snapshot_paths)):
        snapshot = load_snapshot(path)
        family = snapshot.option_type
        boundaries, found = extract_oracle_boundary_path(
            snapshot, config.boundary_threshold
        )
        try:
            multipliers, source_correction = sanitize_snapshot_multiplier(
                snapshot.multiplier_grid
            )
        except ValueError as error:
            rows.append(
                {
                    "regime_id": snapshot.regime_id,
                    "option_type": family,
                    "canonical_points": config.canonical_points,
                    "status": "FAILED_SOURCE_MULTIPLIER",
                    "failure_reason": str(error),
                    "boundary_found_count": int(np.sum(found[1:])),
                }
            )
            continue
        squared_state = 0.0
        squared_multiplier = 0.0
        state_count = 0
        multiplier_count = 0
        max_pairing = 0.0
        trajectory_original_pairing = 0.0
        trajectory_canonical_pairing = 0.0
        max_boundary_roundtrip = 0.0
        minimum_aligned_multiplier = float("inf")
        maximum_endpoint_error = 0.0
        failed = ""
        for index in range(1, len(snapshot.tau_grid)):
            mapping = build_boundary_alignment_map(
                boundaries[index] if found[index] else None,
                config,
                physical_grid=snapshot.spot_grid,
            )
            try:
                aligned_state = align_primal_state(snapshot.lifted_state_grid[index], mapping)
                aligned_multiplier = align_dual_multiplier(multipliers[index], mapping)
                state_roundtrip = inverse_align_primal_state(aligned_state, mapping)
                multiplier_roundtrip = inverse_align_dual_multiplier(aligned_multiplier, mapping)
            except (ValueError, RuntimeError) as error:
                failed = str(error)
                break
            difference = state_roundtrip - snapshot.lifted_state_grid[index]
            lambda_difference = multiplier_roundtrip - multipliers[index]
            squared_state += float(np.dot(difference, difference))
            squared_multiplier += float(np.dot(lambda_difference, lambda_difference))
            state_count += difference.size
            multiplier_count += lambda_difference.size
            pairing = pairing_relative_error(
                snapshot.lifted_state_grid[index],
                multipliers[index],
                aligned_state,
                aligned_multiplier,
            )
            max_pairing = max(max_pairing, pairing)
            trajectory_original_pairing += float(
                snapshot.lifted_state_grid[index] @ multipliers[index]
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                trajectory_canonical_pairing += float(aligned_state @ aligned_multiplier)
            minimum_aligned_multiplier = min(
                minimum_aligned_multiplier, float(np.min(aligned_multiplier))
            )
            if found[index]:
                canonical_at_boundary = np.interp(
                    boundaries[index], mapping.physical_at_canonical, mapping.canonical_grid
                )
                physical_roundtrip = np.interp(
                    canonical_at_boundary, mapping.canonical_grid, mapping.physical_at_canonical
                )
                max_boundary_roundtrip = max(
                    max_boundary_roundtrip, abs(physical_roundtrip - boundaries[index])
                )
            maximum_endpoint_error = max(
                maximum_endpoint_error,
                abs(mapping.physical_at_canonical[0] - snapshot.spot_grid[0]),
                abs(mapping.physical_at_canonical[-1] - snapshot.spot_grid[-1]),
            )
        surface_rmse = np.sqrt(squared_state / max(state_count, 1))
        multiplier_rmse = np.sqrt(squared_multiplier / max(multiplier_count, 1))
        trajectory_pairing = abs(
            trajectory_canonical_pairing - trajectory_original_pairing
        ) / max(
            abs(trajectory_original_pairing),
            abs(trajectory_canonical_pairing),
            1e-12,
        )
        spacing = float(snapshot.spot_grid[1] - snapshot.spot_grid[0])
        gates = {
            "surface_rmse_pass": surface_rmse <= 2.5e-5,
            "boundary_roundtrip_pass": max_boundary_roundtrip <= 0.25 * spacing,
            "multiplier_nonnegative_pass": minimum_aligned_multiplier >= 0.0,
            "pairing_pass": trajectory_pairing <= config.pairing_tolerance,
            "endpoint_pass": maximum_endpoint_error <= 1e-12,
        }
        rows.append(
            {
                "regime_id": snapshot.regime_id,
                "option_type": family,
                "canonical_points": config.canonical_points,
                "status": "COMPLETE" if not failed else "FAILED_TRANSFORM",
                "failure_reason": failed,
                "boundary_found_count": int(np.sum(found[1:])),
                "no_boundary_count": int(np.sum(~found[1:])),
                "surface_roundtrip_rmse": surface_rmse,
                "multiplier_roundtrip_rmse": multiplier_rmse,
                "max_row_pairing_relative_error": max_pairing,
                "trajectory_pairing_relative_error": trajectory_pairing,
                "boundary_roundtrip_max": max_boundary_roundtrip,
                "minimum_aligned_multiplier": minimum_aligned_multiplier,
                "endpoint_max_error": maximum_endpoint_error,
                "source_multiplier_correction_max": source_correction,
                **gates,
                "all_gates_pass": not failed and all(gates.values()),
            }
        )
    complete = [row for row in rows if row.get("status") == "COMPLETE"]
    all_pass = len(complete) == len(rows) and all(row["all_gates_pass"] for row in complete)
    summary = {
        "option_type": family,
        "config": asdict(config),
        "regime_count": len(rows),
        "complete_count": len(complete),
        "all_gates_pass": bool(all_pass),
        "worst_surface_roundtrip_rmse": max(
            (float(row["surface_roundtrip_rmse"]) for row in complete), default=float("inf")
        ),
        "worst_trajectory_pairing_relative_error": max(
            (float(row["trajectory_pairing_relative_error"]) for row in complete),
            default=float("inf"),
        ),
        "status": "GO_TRANSFORM" if all_pass else "DEFER_INTERPOLATION",
    }
    return rows, summary


def audit_boundary_threshold_sensitivity(
    snapshot_paths: Iterable[Path | str],
    *,
    thresholds: tuple[float, ...] = (5e-7, 1e-6, 2e-6),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    changed = 0
    usable = 0
    family = ""
    for path in sorted(map(Path, snapshot_paths)):
        snapshot = load_snapshot(path)
        family = snapshot.option_type
        premium = snapshot.value_grid - snapshot.payoff[None, :]
        extracted: dict[float, np.ndarray] = {}
        found: dict[float, np.ndarray] = {}
        for threshold in thresholds:
            values = np.full(len(snapshot.tau_grid), np.nan)
            flags = np.zeros(len(snapshot.tau_grid), dtype=bool)
            for index in range(1, len(snapshot.tau_grid)):
                point = extract_boundary_at_time(
                    snapshot.spot_grid,
                    premium[index],
                    family,
                    float(snapshot.tau_grid[index]),
                    index,
                    threshold=threshold,
                )
                if point.boundary_found:
                    values[index] = point.boundary_spot
                    flags[index] = True
            extracted[threshold] = values
            found[threshold] = flags
        baseline = 1e-6
        base_found = found[baseline]
        regime_usable = int(np.sum(base_found))
        regime_changed = 0
        for index in np.flatnonzero(base_found):
            deviations = []
            for threshold in thresholds:
                deviations.append(
                    abs(extracted[threshold][index] - extracted[baseline][index])
                    if found[threshold][index]
                    else float("inf")
                )
            if max(deviations) > 0.5 * float(snapshot.spot_grid[1] - snapshot.spot_grid[0]):
                regime_changed += 1
        usable += regime_usable
        changed += regime_changed
        rows.append(
            {
                "regime_id": snapshot.regime_id,
                "option_type": family,
                "usable_rows": regime_usable,
                "changed_over_half_cell": regime_changed,
                "changed_fraction": regime_changed / max(regime_usable, 1),
            }
        )
    fraction = changed / max(usable, 1)
    summary = {
        "option_type": family,
        "thresholds": list(thresholds),
        "usable_rows": usable,
        "changed_over_half_cell": changed,
        "changed_fraction": fraction,
        "status": "GO_BOUNDARY_EXTRACTION" if fraction <= 0.10 else "DEFER_BOUNDARY_EXTRACTION",
    }
    return rows, summary
