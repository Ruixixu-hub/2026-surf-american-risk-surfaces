"""Deterministic primal/dual greedy bases for the algebraic American-option VI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from american_risk_surfaces.reduced_order.protocol import protocol_hash
from american_risk_surfaces.reduced_order.snapshots import load_snapshot
from american_risk_surfaces.reduced_order.types import PrimalDualRBBasis


def weighted_h1_gram(spot_grid: np.ndarray) -> np.ndarray:
    """Return the frozen weighted H1 Gram matrix on zero-boundary states."""

    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1 or len(spots) < 3:
        raise ValueError("spot_grid must contain at least three nodes")
    spacings = np.diff(spots)
    if np.any(spacings <= 0.0) or not np.allclose(spacings, spacings[0], rtol=1e-12, atol=1e-14):
        raise ValueError("weighted_h1_gram requires a strictly increasing uniform grid")
    spacing = float(spacings[0])
    interior_size = len(spots) - 2
    difference = np.zeros((interior_size + 1, interior_size), dtype=float)
    for edge in range(interior_size + 1):
        if edge > 0:
            difference[edge, edge - 1] -= 1.0
        if edge < interior_size:
            difference[edge, edge] += 1.0
    midpoints = 0.5 * (spots[:-1] + spots[1:])
    gram = spacing * np.eye(interior_size)
    # Some macOS BLAS builds expose stale floating-point status flags on the
    # first matrix product in a process. Finiteness is checked explicitly below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram += difference.T @ np.diag(midpoints**2 / spacing) @ difference
    if not np.all(np.isfinite(gram)):
        raise RuntimeError("weighted H1 Gram matrix contains non-finite values")
    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-13):
        raise RuntimeError("weighted H1 Gram matrix is not symmetric")
    if float(np.min(np.linalg.eigvalsh(gram))) <= 0.0:
        raise RuntimeError("weighted H1 Gram matrix is not positive definite")
    return gram


def g_orthonormalize(
    vectors: np.ndarray,
    gram: np.ndarray,
    *,
    relative_tolerance: float = 1e-12,
) -> np.ndarray:
    """Modified Gram-Schmidt with reorthogonalization in the G inner product."""

    candidates = np.asarray(vectors, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    if candidates.ndim != 2 or candidates.shape[0] != gram.shape[0]:
        raise ValueError("vectors must have one row per Gram-matrix row")
    columns: list[np.ndarray] = []
    for index in range(candidates.shape[1]):
        original = candidates[:, index].copy()
        vector = original.copy()
        original_norm = _g_norm(original, gram)
        if original_norm == 0.0:
            continue
        for _ in range(2):
            for basis_vector in columns:
                vector -= basis_vector * float(basis_vector @ gram @ vector)
        norm = _g_norm(vector, gram)
        if norm <= relative_tolerance * original_norm:
            continue
        columns.append(vector / norm)
    if not columns:
        return np.empty((gram.shape[0], 0), dtype=float)
    result = np.column_stack(columns)
    identity = result.T @ gram @ result
    if not np.allclose(identity, np.eye(result.shape[1]), rtol=1e-10, atol=1e-10):
        raise RuntimeError("G-orthonormalization failed")
    return result


def build_primal_dual_basis(
    snapshot_manifest: Iterable[Path | str],
    option_type: str,
    dimension: int,
) -> PrimalDualRBBasis:
    """Build one family-specific POD/angle-greedy basis from train snapshots."""

    return build_primal_dual_basis_ladder(
        snapshot_manifest, option_type, (int(dimension),)
    )[int(dimension)]


@np.errstate(divide="ignore", over="ignore", invalid="ignore")
def build_primal_dual_basis_ladder(
    snapshot_manifest: Iterable[Path | str],
    option_type: str,
    dimensions: Iterable[int],
) -> dict[int, PrimalDualRBBasis]:
    """Build a whole ladder while running the expensive greedy searches once."""

    family = str(option_type).lower()
    if family not in {"put", "call"}:
        raise ValueError("option_type must be put or call")
    requested_dimensions = tuple(sorted({int(value) for value in dimensions}))
    if not requested_dimensions or requested_dimensions[0] < 1:
        raise ValueError("dimensions must contain positive integers")
    maximum_dimension = requested_dimensions[-1]
    snapshots = [load_snapshot(path) for path in sorted(map(Path, snapshot_manifest))]
    snapshots = [item for item in snapshots if item.option_type == family]
    if not snapshots:
        raise ValueError(f"no {family} snapshots supplied")
    if any(item.metadata.get("regime", {}).get("split") != "train" for item in snapshots):
        raise ValueError("basis construction is restricted to train snapshots")
    reference_grid = snapshots[0].spot_grid
    if any(not np.array_equal(item.spot_grid, reference_grid) for item in snapshots[1:]):
        raise ValueError("all snapshots must use the same spot grid")
    gram = weighted_h1_gram(reference_grid)
    primal_all, primal_history = pod_greedy(
        [(item.regime_id, item.lifted_state_grid) for item in snapshots], gram, maximum_dimension
    )
    dual_all, dual_history = angle_greedy(
        [
            (item.regime_id, time_index, item.multiplier_grid[time_index])
            for item in snapshots
            for time_index in range(1, len(item.tau_grid))
        ],
        gram,
        maximum_dimension,
    )
    if primal_all.shape[1] < maximum_dimension:
        raise RuntimeError(f"only {primal_all.shape[1]} independent primal modes available")
    if dual_all.shape[1] < maximum_dimension:
        raise RuntimeError(f"only {dual_all.shape[1]} independent dual generators available")
    result: dict[int, PrimalDualRBBasis] = {}
    for requested in requested_dimensions:
        primal_pod = primal_all[:, :requested]
        dual = dual_all[:, :requested]
        supremizers = np.linalg.solve(gram, dual)
        primal = g_orthonormalize(np.column_stack((primal_pod, supremizers)), gram)
        dual_gram = dual.T @ np.linalg.solve(gram, dual)
        condition_number = float(np.linalg.cond(dual_gram))
        inf_sup = reduced_inf_sup_constant(primal, dual, gram)
        if inf_sup < 1e-8:
            raise RuntimeError(f"reduced inf-sup constant {inf_sup:.3e} is below 1e-8")
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise RuntimeError(f"dual Gram condition number {condition_number:.3e} exceeds 1e12")
        result[requested] = PrimalDualRBBasis(
            option_type=family,
            primal_basis=primal,
            dual_generators=dual,
            gram_matrix=gram,
            primal_dimension=primal.shape[1],
            dual_dimension=dual.shape[1],
            inf_sup_constant=inf_sup,
            condition_number=condition_number,
            metadata={
                "protocol_hash": protocol_hash(),
                "requested_dimension": requested,
                "pod_dimension": primal_pod.shape[1],
                "supremizer_count": requested,
                "snapshot_regime_ids": [item.regime_id for item in snapshots],
                "primal_history": primal_history[:requested],
                "dual_history": dual_history[:requested],
            },
        )
    return result


def pod_greedy(
    trajectories: Iterable[tuple[str, np.ndarray]],
    gram: np.ndarray,
    max_modes: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Strong trajectory POD-greedy using mean squared G projection error."""

    ordered = sorted(
        [(str(regime_id), np.asarray(values, dtype=float)) for regime_id, values in trajectories],
        key=lambda item: item[0],
    )
    if not ordered:
        raise ValueError("trajectories must not be empty")
    width = gram.shape[0]
    if any(values.ndim != 2 or values.shape[1] != width for _, values in ordered):
        raise ValueError("trajectory width must match gram_matrix")
    cholesky = np.linalg.cholesky(gram)
    basis = np.empty((width, 0), dtype=float)
    history: list[dict[str, object]] = []
    for iteration in range(int(max_modes)):
        errors: list[tuple[float, str, np.ndarray]] = []
        for regime_id, values in ordered:
            residual = _projection_residual(values, basis, gram)
            energy = float(np.mean(np.sum((residual @ cholesky) ** 2, axis=1)))
            errors.append((energy, regime_id, residual))
        maximum = max(item[0] for item in errors)
        if not np.isfinite(maximum):
            raise RuntimeError("non-finite POD-greedy projection energy")
        selected = min(
            (item for item in errors if np.isclose(item[0], maximum, rtol=1e-14, atol=0.0)),
            key=lambda item: item[1],
        )
        energy, regime_id, residual = selected
        if energy <= 1e-28:
            break
        _, singular_values, right = np.linalg.svd(residual @ cholesky, full_matrices=False)
        mode = np.linalg.solve(cholesky.T, right[0])
        enlarged = g_orthonormalize(np.column_stack((basis, mode)), gram)
        if enlarged.shape[1] == basis.shape[1]:
            break
        basis = enlarged
        history.append(
            {
                "iteration": iteration + 1,
                "regime_id": regime_id,
                "mean_squared_projection_error": energy,
                "leading_residual_singular_value": float(singular_values[0]),
            }
        )
    return basis, history


def angle_greedy(
    snapshots: Iterable[tuple[str, int, np.ndarray]],
    gram: np.ndarray,
    max_generators: int,
    *,
    zero_tolerance: float = 1e-14,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Angle-greedy cone construction without sign-destroying orthogonalization."""

    candidates: list[tuple[str, int, np.ndarray, float]] = []
    for regime_id, time_index, raw in snapshots:
        vector = np.asarray(raw, dtype=float).copy()
        if vector.ndim != 1 or len(vector) != gram.shape[0]:
            raise ValueError("dual snapshot width must match gram_matrix")
        if float(np.min(vector)) < -1e-12:
            raise ValueError("dual multiplier snapshots must be nonnegative")
        vector[vector < 0.0] = 0.0
        norm = _w_norm(vector, gram)
        if norm >= zero_tolerance:
            candidates.append((str(regime_id), int(time_index), vector / norm, norm))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        return np.empty((gram.shape[0], 0), dtype=float), []
    generators = np.empty((gram.shape[0], 0), dtype=float)
    history: list[dict[str, object]] = []
    candidate_matrix = np.column_stack([item[2] for item in candidates])
    inverse_candidate_matrix = np.linalg.solve(gram, candidate_matrix)
    for iteration in range(int(max_generators)):
        if generators.shape[1] == 0:
            scores = [item[3] for item in candidates]
        else:
            dual_gram = generators.T @ np.linalg.solve(gram, generators)
            cross = generators.T @ inverse_candidate_matrix
            coefficients = np.linalg.solve(dual_gram, cross)
            squared = 1.0 - np.sum(cross * coefficients, axis=0)
            scores = np.sqrt(np.maximum(squared, 0.0)).tolist()
        maximum = max(scores)
        if not np.isfinite(maximum):
            raise RuntimeError("non-finite angle-greedy score")
        if maximum <= 1e-12:
            break
        choices = [
            index
            for index, score in enumerate(scores)
            if np.isclose(score, maximum, rtol=1e-14, atol=0.0)
        ]
        selected_index = min(choices, key=lambda index: (candidates[index][0], candidates[index][1]))
        regime_id, time_index, vector, original_norm = candidates[selected_index]
        generators = np.column_stack((generators, vector))
        history.append(
            {
                "iteration": iteration + 1,
                "regime_id": regime_id,
                "time_index": time_index,
                "selection_score": float(maximum),
                "original_w_norm": float(original_norm),
            }
        )
    if float(np.min(generators)) < -1e-15:
        raise RuntimeError("angle-greedy produced a negative cone generator")
    return generators, history


def reduced_inf_sup_constant(primal: np.ndarray, dual: np.ndarray, gram: np.ndarray) -> float:
    if dual.shape[1] == 0:
        return float("inf")
    dual_gram = dual.T @ np.linalg.solve(gram, dual)
    eigenvalues, eigenvectors = np.linalg.eigh(dual_gram)
    if float(np.min(eigenvalues)) <= 0.0:
        return 0.0
    inverse_root = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    singular_values = np.linalg.svd(primal.T @ dual @ inverse_root, compute_uv=False)
    return float(np.min(singular_values))


def save_basis(basis: PrimalDualRBBasis, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            option_type=basis.option_type,
            primal_basis=basis.primal_basis,
            dual_generators=basis.dual_generators,
            gram_matrix=basis.gram_matrix,
            primal_dimension=basis.primal_dimension,
            dual_dimension=basis.dual_dimension,
            inf_sup_constant=basis.inf_sup_constant,
            condition_number=basis.condition_number,
            metadata_json=json.dumps(basis.metadata, sort_keys=True),
        )
    os.replace(temporary, destination)


def load_basis(path: Path | str) -> PrimalDualRBBasis:
    with np.load(path, allow_pickle=False) as data:
        return PrimalDualRBBasis(
            option_type=str(data["option_type"]),
            primal_basis=data["primal_basis"].copy(),
            dual_generators=data["dual_generators"].copy(),
            gram_matrix=data["gram_matrix"].copy(),
            primal_dimension=int(data["primal_dimension"]),
            dual_dimension=int(data["dual_dimension"]),
            inf_sup_constant=float(data["inf_sup_constant"]),
            condition_number=float(data["condition_number"]),
            metadata=json.loads(str(data["metadata_json"])),
        )


def _projection_residual(values: np.ndarray, basis: np.ndarray, gram: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return values.copy()
    coefficients = values @ gram @ basis
    return values - coefficients @ basis.T


def _g_norm(vector: np.ndarray, gram: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, float(vector @ gram @ vector))))


def _w_norm(vector: np.ndarray, gram: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, float(vector @ np.linalg.solve(gram, vector)))))
