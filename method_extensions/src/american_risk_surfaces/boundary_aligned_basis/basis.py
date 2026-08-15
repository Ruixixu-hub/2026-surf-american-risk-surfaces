"""Train-only global/aligned/localized primal and dual oracle bases."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.boundary_aligned_basis.alignment import (
    align_dual_multiplier,
    align_primal_state,
    build_boundary_alignment_map,
    extract_oracle_boundary_path,
    sanitize_snapshot_multiplier,
)
from american_risk_surfaces.boundary_aligned_basis.metric import (
    WeightedH1Metric,
    g_orthonormalize,
)
from american_risk_surfaces.boundary_aligned_basis.protocol import (
    DIMENSION_LADDER,
    MAX_STORED_DUAL_GENERATORS,
    MAX_STORED_PRIMAL_MODES,
    protocol_hash,
)
from american_risk_surfaces.boundary_aligned_basis.types import (
    BoundaryAlignmentConfig,
    OracleBasisArtifact,
)
from american_risk_surfaces.reduced_order.snapshots import load_snapshot


@dataclass(frozen=True)
class _TrainingRows:
    states: np.ndarray
    multipliers: np.ndarray
    regime_ids: np.ndarray
    time_indices: np.ndarray
    bin_values: np.ndarray
    boundary_found: np.ndarray
    grid: np.ndarray
    source_correction_max: float


def build_oracle_basis_ladder(
    train_manifest: Iterable[Path | str],
    arm: str,
    option_type: str,
    *,
    dimensions: Iterable[int] = DIMENSION_LADDER,
    bin_count: int | None = None,
    alignment_config: BoundaryAlignmentConfig | None = None,
) -> dict[int, OracleBasisArtifact]:
    arm = str(arm).upper()
    family = str(option_type).lower()
    if arm not in {"U", "A", "L", "AL"}:
        raise ValueError("arm must be U, A, L, or AL")
    if family not in {"put", "call"}:
        raise ValueError("option_type must be put or call")
    localized = arm in {"L", "AL"}
    aligned = arm in {"A", "AL"}
    if localized and bin_count not in {2, 4, 8}:
        raise ValueError("localized arms require bin_count in {2,4,8}")
    if not localized and bin_count is not None:
        raise ValueError("global arms do not use bin_count")
    requested = tuple(sorted({int(value) for value in dimensions}))
    if not requested or requested[0] < 1 or requested[-1] > 32:
        raise ValueError("dimensions must be between 1 and 32")
    config = alignment_config or BoundaryAlignmentConfig()
    started = perf_counter()
    rows = _load_training_rows(train_manifest, family, aligned, config)
    assignments, edges, labels = _bin_assignments(rows, family, bin_count if localized else None)
    bin_indices = [np.flatnonzero(assignments == index) for index in range(len(labels))]
    max_budget_dimension = min(
        requested[-1],
        MAX_STORED_PRIMAL_MODES // len(labels),
        MAX_STORED_DUAL_GENERATORS // len(labels),
    )
    primal_max: list[np.ndarray] = []
    dual_max: list[np.ndarray] = []
    primal_history: dict[str, object] = {}
    dual_history: dict[str, object] = {}
    metric = WeightedH1Metric.from_grid(rows.grid)
    for label, indices in zip(labels, bin_indices, strict=True):
        if len(indices) < max_budget_dimension:
            raise RuntimeError(
                f"bin {label} has only {len(indices)} rows for {max_budget_dimension} modes"
            )
        primal, p_history = pod_greedy_rows(
            rows.states[indices], rows.regime_ids[indices], metric, max_budget_dimension
        )
        dual, d_history = angle_greedy_rows(
            rows.multipliers[indices],
            rows.regime_ids[indices],
            rows.time_indices[indices],
            metric,
            max_budget_dimension,
        )
        primal_max.append(primal)
        dual_max.append(dual)
        primal_history[label] = p_history
        dual_history[label] = d_history
    artifacts: dict[int, OracleBasisArtifact] = {}
    for dimension in requested:
        if dimension > max_budget_dimension:
            continue
        if any(base.shape[1] < dimension for base in primal_max):
            continue
        if any(
            base.shape[1] < dimension and base.shape[1] != 0
            for base in dual_max
        ):
            continue
        primal = tuple(base[:, :dimension].copy() for base in primal_max)
        dual = tuple(base[:, :dimension].copy() for base in dual_max)
        total_primal = sum(base.shape[1] for base in primal)
        total_dual = sum(base.shape[1] for base in dual)
        artifacts[dimension] = OracleBasisArtifact(
            arm,
            family,
            primal,
            dual,
            edges,
            labels,
            dimension,
            total_primal + total_dual,
            tuple(rows.grid.copy() for _ in labels),
            {
                "protocol_hash": protocol_hash(),
                "aligned": aligned,
                "localized": localized,
                "requested_bin_count": bin_count,
                "actual_bin_count": len(labels),
                "bin_occupancy": {label: int(len(index)) for label, index in zip(labels, bin_indices)},
                "stored_primal_modes": total_primal,
                "stored_dual_generators": total_dual,
                "construction_seconds": perf_counter() - started,
                "source_multiplier_correction_max": rows.source_correction_max,
                "primal_history": primal_history,
                "dual_history": dual_history,
            },
        )
    return artifacts


def pod_greedy_rows(
    states: np.ndarray,
    regime_ids: np.ndarray,
    metric: WeightedH1Metric,
    max_modes: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    values = np.asarray(states, dtype=float)
    identifiers = np.asarray(regime_ids).astype(str)
    if values.ndim != 2 or values.shape[1] != metric.size:
        raise ValueError("state matrix width must match metric")
    unique = np.unique(identifiers)
    basis = np.empty((metric.size, 0), dtype=float)
    state_energy = metric.norm_squared_rows(values)
    history: list[dict[str, object]] = []
    for iteration in range(int(max_modes)):
        if basis.shape[1]:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                coefficients = values @ metric.apply(basis.T).T
            residual_energy = np.maximum(state_energy - np.sum(coefficients**2, axis=1), 0.0)
        else:
            coefficients = np.empty((len(values), 0))
            residual_energy = state_energy.copy()
        errors = {name: float(np.mean(residual_energy[identifiers == name])) for name in unique}
        maximum = max(errors.values())
        selected = min(
            name for name, value in errors.items() if np.isclose(value, maximum, rtol=1e-13)
        )
        if maximum <= 1e-28:
            break
        mask = identifiers == selected
        residual = values[mask] - coefficients[mask] @ basis.T
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            correlation = residual @ metric.apply(residual).T
        if not np.all(np.isfinite(correlation)):
            raise RuntimeError("non-finite POD residual correlation")
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        eigenvalue = float(eigenvalues[-1])
        if eigenvalue <= 1e-28:
            break
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mode = residual.T @ eigenvectors[:, -1] / np.sqrt(eigenvalue)
        enlarged = g_orthonormalize(np.column_stack((basis, mode)), metric)
        if enlarged.shape[1] == basis.shape[1]:
            break
        basis = enlarged
        history.append(
            {
                "iteration": iteration + 1,
                "regime_id": selected,
                "mean_squared_projection_error": maximum,
                "leading_residual_eigenvalue": eigenvalue,
            }
        )
    return basis, history


def angle_greedy_rows(
    multipliers: np.ndarray,
    regime_ids: np.ndarray,
    time_indices: np.ndarray,
    metric: WeightedH1Metric,
    max_generators: int,
    *,
    zero_tolerance: float = 1e-14,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    values = np.asarray(multipliers, dtype=float)
    if float(np.min(values)) < -1e-14:
        raise ValueError("dual candidate below -1e-14")
    values = np.maximum(values, 0.0)
    inverse = metric.solve(values)
    norms = np.sqrt(np.maximum(np.einsum("ij,ij->i", values, inverse), 0.0))
    keep = norms >= zero_tolerance
    values = values[keep]
    inverse = inverse[keep]
    norms = norms[keep]
    names = np.asarray(regime_ids).astype(str)[keep]
    times = np.asarray(time_indices, dtype=int)[keep]
    if not len(values):
        return np.empty((metric.size, 0)), []
    values = values / norms[:, None]
    inverse = inverse / norms[:, None]
    generators = np.empty((metric.size, 0), dtype=float)
    inverse_generators = np.empty((metric.size, 0), dtype=float)
    history: list[dict[str, object]] = []
    for iteration in range(int(max_generators)):
        if generators.shape[1] == 0:
            scores = norms.copy()
        else:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                gram = generators.T @ inverse_generators
                cross = generators.T @ inverse.T
            coefficients = np.linalg.solve(gram, cross)
            scores = np.sqrt(np.maximum(1.0 - np.sum(cross * coefficients, axis=0), 0.0))
        maximum = float(np.max(scores))
        if maximum <= 1e-12:
            break
        choices = np.flatnonzero(np.isclose(scores, maximum, rtol=1e-13, atol=0.0))
        index = min(choices, key=lambda item: (names[item], int(times[item])))
        proposed = np.column_stack((generators, values[index]))
        proposed_inverse = np.column_stack((inverse_generators, inverse[index]))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cone_gram = proposed.T @ proposed_inverse
        condition = float(np.linalg.cond(cone_gram))
        if not np.isfinite(condition) or condition > 1e12:
            break
        generators = proposed
        inverse_generators = proposed_inverse
        history.append(
            {
                "iteration": iteration + 1,
                "regime_id": names[index],
                "time_index": int(times[index]),
                "selection_score": maximum,
                "original_w_norm": float(norms[index]),
                "cone_gram_condition_number": condition,
            }
        )
    return generators, history


def save_oracle_basis(artifact: OracleBasisArtifact, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    arrays: dict[str, object] = {
        "arm": artifact.arm,
        "option_type": artifact.option_type,
        "bin_edges": artifact.bin_edges,
        "bin_labels_json": json.dumps(artifact.bin_labels),
        "active_dimension": artifact.active_dimension,
        "total_stored_modes": artifact.total_stored_modes,
        "metadata_json": json.dumps(artifact.metadata, sort_keys=True),
    }
    for index, (primal, dual, grid) in enumerate(
        zip(artifact.primal_bases, artifact.dual_generators, artifact.metric_grids, strict=True)
    ):
        arrays[f"primal_{index}"] = primal
        arrays[f"dual_{index}"] = dual
        arrays[f"grid_{index}"] = grid
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def load_oracle_basis(path: Path | str) -> OracleBasisArtifact:
    with np.load(path, allow_pickle=False) as data:
        labels = tuple(json.loads(str(data["bin_labels_json"])))
        return OracleBasisArtifact(
            str(data["arm"]),
            str(data["option_type"]),
            tuple(data[f"primal_{index}"].copy() for index in range(len(labels))),
            tuple(data[f"dual_{index}"].copy() for index in range(len(labels))),
            data["bin_edges"].copy(),
            labels,
            int(data["active_dimension"]),
            int(data["total_stored_modes"]),
            tuple(data[f"grid_{index}"].copy() for index in range(len(labels))),
            json.loads(str(data["metadata_json"])),
        )


def _load_training_rows(
    paths: Iterable[Path | str],
    family: str,
    aligned: bool,
    config: BoundaryAlignmentConfig,
) -> _TrainingRows:
    states: list[np.ndarray] = []
    multipliers: list[np.ndarray] = []
    names: list[str] = []
    times: list[int] = []
    bins: list[float] = []
    found_rows: list[bool] = []
    grid: np.ndarray | None = None
    maximum_correction = 0.0
    for path in sorted(map(Path, paths)):
        snapshot = load_snapshot(path)
        split = snapshot.metadata.get("regime", {}).get("split")
        q = float(snapshot.metadata.get("regime", {}).get("q", 0.0))
        if split != "train" or snapshot.option_type != family:
            raise PermissionError("basis input must be a same-family train snapshot")
        if family == "call" and q == 0.0:
            raise PermissionError("q=0 calls are sealed")
        if aligned:
            try:
                clean_multipliers, correction = sanitize_snapshot_multiplier(
                    snapshot.multiplier_grid
                )
            except ValueError as error:
                raise ValueError(
                    f"{snapshot.regime_id}: {error}; regenerate the FOM snapshot with the "
                    "boundary-alignment sanitization rule before aligned basis construction"
                ) from error
        else:
            # Physical U/L are unaffected by PCHIP.  Preserve the established
            # FOM convention: values within the frozen 1e-12 LCP tolerance are
            # reported and clipped to the dual cone, never shifted.
            clean_multipliers = snapshot.multiplier_grid.copy()
            floor = float(np.min(clean_multipliers))
            if floor < -1e-12:
                raise ValueError(f"{snapshot.regime_id}: multiplier below FOM tolerance")
            correction = max(-floor, 0.0)
            clean_multipliers[clean_multipliers < 0.0] = 0.0
        maximum_correction = max(maximum_correction, correction)
        boundaries, found = extract_oracle_boundary_path(snapshot, config.boundary_threshold)
        for index in range(1, len(snapshot.tau_grid)):
            boundary = boundaries[index] if found[index] else None
            if aligned:
                mapping = build_boundary_alignment_map(
                    boundary, config, physical_grid=snapshot.spot_grid
                )
                states.append(align_primal_state(snapshot.lifted_state_grid[index], mapping))
                multipliers.append(align_dual_multiplier(clean_multipliers[index], mapping))
                grid = mapping.canonical_grid
            else:
                states.append(snapshot.lifted_state_grid[index].copy())
                multipliers.append(clean_multipliers[index].copy())
                grid = snapshot.spot_grid
            names.append(snapshot.regime_id)
            times.append(index)
            found_rows.append(bool(found[index]))
            if not found[index]:
                bins.append(float("nan"))
            elif family == "put":
                bins.append(float(boundaries[index]))
            else:
                bins.append(float((boundaries[index] - 1.0) / 3.0))
    if grid is None:
        raise ValueError("no train snapshots supplied")
    return _TrainingRows(
        np.asarray(states),
        np.asarray(multipliers),
        np.asarray(names),
        np.asarray(times),
        np.asarray(bins),
        np.asarray(found_rows),
        np.asarray(grid),
        maximum_correction,
    )


def _bin_assignments(
    rows: _TrainingRows,
    family: str,
    bin_count: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if bin_count is None:
        return np.zeros(len(rows.states), dtype=int), np.asarray([-np.inf, np.inf]), ("global",)
    finite = rows.bin_values[np.isfinite(rows.bin_values)]
    if not len(finite):
        raise RuntimeError(f"no boundaries available for localized {family} basis")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 2:
        raise RuntimeError("quantile edges collapsed to one value")
    assignments = np.full(len(rows.states), -1, dtype=int)
    assignments[rows.boundary_found] = np.digitize(
        rows.bin_values[rows.boundary_found], edges[1:-1], right=False
    )
    labels = tuple(f"boundary_bin_{index}" for index in range(len(edges) - 1))
    if np.any(~rows.boundary_found):
        assignments[~rows.boundary_found] = len(labels)
        labels = labels + ("no_boundary",)
    return assignments, edges, labels
