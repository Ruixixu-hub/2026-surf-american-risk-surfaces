"""Leakage-safe POD and oracle boundary-alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from american_risk_surfaces.surrogates import price_premium as stage3


@dataclass(frozen=True)
class PremiumSurfaceDataset:
    regime_ids: np.ndarray
    split_by_regime: np.ndarray
    tau_grid: np.ndarray
    moneyness_grid: np.ndarray
    premium_surfaces: np.ndarray
    boundary_surfaces: np.ndarray
    boundary_near_masks: np.ndarray
    strict_interior_masks: np.ndarray


@dataclass(frozen=True)
class PODBasis:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    train_regime_ids: tuple[str, ...]
    representation: str

    def reconstruct(self, surfaces: np.ndarray, modes: int) -> np.ndarray:
        if modes < 1 or modes > self.components.shape[0]:
            raise ValueError("modes is outside the fitted POD basis.")
        flat = np.asarray(surfaces, dtype=float).reshape(len(surfaces), -1)
        selected = self.components[:modes]
        scores = np.einsum("ij,kj->ik", flat - self.mean, selected, optimize=False)
        reconstructed = self.mean + np.einsum(
            "ik,kj->ij", scores, selected, optimize=False
        )
        return reconstructed.reshape(surfaces.shape)


def load_premium_surface_dataset(
    bundle: stage3.SurrogateDatasetBundle | None = None,
) -> PremiumSurfaceDataset:
    bundle = stage3.load_v1_dataset() if bundle is None else bundle
    regime_count = len(bundle.regime_ids)
    surfaces: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    boundary_masks: list[np.ndarray] = []
    strict_masks: list[np.ndarray] = []
    splits: list[str] = []
    common_tau: np.ndarray | None = None
    common_moneyness: np.ndarray | None = None
    split_index_column = stage3.AUDIT_NUMERIC_INDEX["split_index"]
    moneyness_column = stage3.AUDIT_NUMERIC_INDEX["S_over_K"]

    for regime_index in range(regime_count):
        indices = np.flatnonzero(bundle.regime_index == regime_index)
        tau = bundle.X[indices, stage3.FEATURE_NAMES.index("tau_fraction")]
        moneyness = bundle.audit_numeric[indices, moneyness_column]
        tau_values = np.unique(tau)
        moneyness_values = np.unique(moneyness)
        expected = len(tau_values) * len(moneyness_values)
        if len(indices) != expected:
            raise ValueError("each regime must be a complete tau/moneyness surface")
        if common_tau is None:
            common_tau = tau_values
            common_moneyness = moneyness_values
        elif not np.allclose(tau_values, common_tau) or not np.allclose(
            moneyness_values, common_moneyness
        ):
            raise ValueError("all regimes must share the sampled surface grid")
        shape = (len(tau_values), len(moneyness_values))
        surfaces.append(bundle.y_premium[indices].reshape(shape))
        boundaries.append(bundle.y_boundary[indices].reshape(shape))
        boundary_masks.append(
            bundle.masks[indices, stage3.MASK_INDEX["boundary_near"]].reshape(shape)
        )
        strict_masks.append(
            bundle.masks[indices, stage3.MASK_INDEX["strict_interior"]].reshape(shape)
        )
        split_index = int(bundle.audit_numeric[indices[0], split_index_column])
        splits.append(str(bundle.split_names[split_index]))

    assert common_tau is not None and common_moneyness is not None
    return PremiumSurfaceDataset(
        regime_ids=np.asarray(bundle.regime_ids, dtype=str),
        split_by_regime=np.asarray(splits, dtype=str),
        tau_grid=common_tau,
        moneyness_grid=common_moneyness,
        premium_surfaces=np.asarray(surfaces, dtype=float),
        boundary_surfaces=np.asarray(boundaries, dtype=float),
        boundary_near_masks=np.asarray(boundary_masks, dtype=bool),
        strict_interior_masks=np.asarray(strict_masks, dtype=bool),
    )


def fit_pod_basis(
    surfaces: np.ndarray,
    train_regime_ids: np.ndarray,
    *,
    representation: str,
) -> PODBasis:
    array = np.asarray(surfaces, dtype=float)
    if array.ndim != 3 or len(array) < 2:
        raise ValueError("surfaces must have shape (regime, tau, spot)")
    flat = array.reshape(len(array), -1)
    mean = np.mean(flat, axis=0)
    _, singular_values, right = np.linalg.svd(flat - mean, full_matrices=False)
    return PODBasis(
        mean=mean,
        components=right,
        singular_values=singular_values,
        train_regime_ids=tuple(map(str, train_regime_ids)),
        representation=representation,
    )


def oracle_align_surfaces(
    dataset: PremiumSurfaceDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Align each row to its true boundary; this is a diagnostic, not an online model."""

    x_grid = np.log(dataset.moneyness_grid)
    aligned = np.empty_like(dataset.premium_surfaces)
    shifts = np.zeros(dataset.premium_surfaces.shape[:2], dtype=float)
    for regime in range(len(dataset.premium_surfaces)):
        for time_index in range(len(dataset.tau_grid)):
            boundary_row = dataset.boundary_surfaces[regime, time_index]
            finite = boundary_row[np.isfinite(boundary_row) & (boundary_row > 0.0)]
            shift = float(np.log(np.median(finite))) if finite.size else 0.0
            shifts[regime, time_index] = shift
            aligned[regime, time_index] = np.interp(
                x_grid + shift,
                x_grid,
                dataset.premium_surfaces[regime, time_index],
                left=dataset.premium_surfaces[regime, time_index, 0],
                right=dataset.premium_surfaces[regime, time_index, -1],
            )
    return aligned, shifts


def undo_oracle_alignment(
    aligned_surfaces: np.ndarray,
    shifts: np.ndarray,
    moneyness_grid: np.ndarray,
) -> np.ndarray:
    x_grid = np.log(np.asarray(moneyness_grid, dtype=float))
    aligned = np.asarray(aligned_surfaces, dtype=float)
    restored = np.empty_like(aligned)
    for regime in range(len(aligned)):
        for time_index in range(aligned.shape[1]):
            restored[regime, time_index] = np.interp(
                x_grid - shifts[regime, time_index],
                x_grid,
                aligned[regime, time_index],
                left=aligned[regime, time_index, 0],
                right=aligned[regime, time_index, -1],
            )
    return restored
