"""Memory-efficient weighted-H1 metric used on physical and fine canonical grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class WeightedH1Metric:
    grid: np.ndarray
    diagonal: np.ndarray
    off_diagonal: np.ndarray

    @classmethod
    def from_grid(cls, full_grid: np.ndarray) -> "WeightedH1Metric":
        grid = np.asarray(full_grid, dtype=float)
        if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0.0):
            raise ValueError("metric grid must be strictly increasing")
        spacing = np.diff(grid)
        if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-14):
            raise ValueError("weighted H1 metric requires a uniform grid")
        h = float(spacing[0])
        edge_weights = (0.5 * (grid[:-1] + grid[1:])) ** 2 / h
        diagonal = h + edge_weights[:-1] + edge_weights[1:]
        off = -edge_weights[1:-1]
        metric = cls(grid.copy(), diagonal, off)
        if np.min(metric.eigenvalues()) <= 0.0:
            raise RuntimeError("weighted H1 metric is not positive definite")
        return metric

    @property
    def size(self) -> int:
        return len(self.diagonal)

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.size:
            raise ValueError("last array dimension must match metric size")
        result = array * self.diagonal
        result[..., :-1] += array[..., 1:] * self.off_diagonal
        result[..., 1:] += array[..., :-1] * self.off_diagonal
        return result

    def solve(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        transpose = array.ndim == 2 and array.shape[1] == self.size
        right = array.T if transpose else array
        if right.shape[0] != self.size:
            raise ValueError("first solve dimension must match metric size")
        bands = np.zeros((3, self.size), dtype=float)
        bands[0, 1:] = self.off_diagonal
        bands[1] = self.diagonal
        bands[2, :-1] = self.off_diagonal
        solved = solve_banded((1, 1), bands, right, check_finite=True)
        return solved.T if transpose else solved

    def inner(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(left) @ self.apply(np.asarray(right)).T

    def norm_squared_rows(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.einsum("ij,ij->i", array, self.apply(array))

    def eigenvalues(self) -> np.ndarray:
        # The study only needs a definitive SPD check; the 1,919/3,839 matrices
        # are small enough for the tridiagonal eigensolver.
        from scipy.linalg import eigvalsh_tridiagonal

        return eigvalsh_tridiagonal(self.diagonal, self.off_diagonal)


def g_orthonormalize(
    vectors: np.ndarray,
    metric: WeightedH1Metric,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    candidates = np.asarray(vectors, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    columns: list[np.ndarray] = []
    for candidate in candidates.T:
        vector = candidate.copy()
        original = np.sqrt(max(float(candidate @ metric.apply(candidate)), 0.0))
        if original == 0.0:
            continue
        for _ in range(2):
            for basis_vector in columns:
                vector -= basis_vector * float(basis_vector @ metric.apply(vector))
        norm = np.sqrt(max(float(vector @ metric.apply(vector)), 0.0))
        if norm <= tolerance * original:
            continue
        columns.append(vector / norm)
    if not columns:
        return np.empty((metric.size, 0), dtype=float)
    result = np.column_stack(columns)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        identity = result.T @ metric.apply(result.T).T
    if not np.allclose(identity, np.eye(result.shape[1]), atol=1e-9, rtol=1e-9):
        raise RuntimeError("weighted-H1 orthonormalization failed")
    return result
