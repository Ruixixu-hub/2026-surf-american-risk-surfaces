"""Full-grid continuation-premium POD representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from american_risk_surfaces.basis_operator.protocol import (
    PROJECT_ROOT,
    assert_training_snapshot_allowed,
    protocol_hash,
)
from american_risk_surfaces.basis_operator.types import PremiumPODBasis
from american_risk_surfaces.reduced_order.snapshots import load_snapshot


def premium_vector_from_value(value_grid: np.ndarray, payoff: np.ndarray, K: float = 1.0) -> np.ndarray:
    values = np.asarray(value_grid, dtype=float)
    obstacle = np.asarray(payoff, dtype=float)
    if values.shape != (121, 121) or obstacle.shape != (121,):
        raise ValueError("formal basis data must use the frozen 121x121 full grid")
    premium = (values - obstacle[np.newaxis, :]) / float(K)
    return premium[1:, 1:-1].reshape(-1)


def load_training_matrix(
    snapshot_paths: Iterable[Path | str], option_type: str
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    identifiers: list[str] = []
    tau_grid = None
    m_grid = None
    family = str(option_type).lower()
    for raw_path in sorted(map(Path, snapshot_paths)):
        assert_training_snapshot_allowed(raw_path, family)
        snapshot = load_snapshot(raw_path)
        if snapshot.option_type != family or snapshot.metadata["regime"]["split"] != "train":
            raise PermissionError("snapshot family/split does not match train-only request")
        if family == "call" and float(snapshot.metadata["regime"]["q"]) <= 0.0:
            raise PermissionError("q=0 calls may not be used to fit the dividend-call model")
        rows.append(premium_vector_from_value(snapshot.value_grid, snapshot.payoff))
        identifiers.append(snapshot.regime_id)
        tau_grid = snapshot.tau_grid[1:] / float(snapshot.tau_grid[-1])
        m_grid = snapshot.spot_grid[1:-1] / float(snapshot.metadata["regime"]["K"])
    if not rows:
        raise ValueError("at least one train snapshot is required")
    return np.vstack(rows), tuple(identifiers), np.asarray(tau_grid), np.asarray(m_grid)


def fit_full_grid_premium_basis(
    train_snapshot_manifest: Iterable[Path | str], option_type: str, modes: int
) -> PremiumPODBasis:
    matrix, identifiers, tau_grid, m_grid = load_training_matrix(
        train_snapshot_manifest, option_type
    )
    requested = int(modes)
    if requested < 1 or requested > min(matrix.shape):
        raise ValueError("modes must be positive and no larger than the training rank")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:requested].copy()
    metadata = {
        "protocol_hash": protocol_hash(),
        "vector_order": "tau-major; tau indices 1:121, spot indices 1:120",
        "training_surface_count": len(identifiers),
        "full_singular_value_count": len(singular),
        "retained_energy": float(
            np.sum(singular[:requested] ** 2) / max(np.sum(singular**2), 1e-30)
        ),
    }
    return PremiumPODBasis(
        str(option_type).lower(), mean, components, singular.copy(),
        tau_grid, m_grid, identifiers, metadata,
    )


def project_premium_coefficients(
    basis: PremiumPODBasis, premium_surface: np.ndarray
) -> np.ndarray:
    vector = np.asarray(premium_surface, dtype=float)
    if vector.shape == (121, 121):
        raise ValueError("pass normalized premium, not a value grid; use the 120x119 learned slice")
    flat = vector.reshape(-1)
    if flat.shape != basis.mean_premium.shape:
        raise ValueError("premium surface does not match the basis learned slice")
    return np.einsum("ij,j->i", basis.components, flat - basis.mean_premium, optimize=False)


def reconstruct_premium_vector(basis: PremiumPODBasis, coefficients: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.shape != (basis.components.shape[0],):
        raise ValueError("coefficient vector has the wrong size")
    return basis.mean_premium + np.einsum("i,ij->j", coeff, basis.components, optimize=False)


def save_premium_basis(basis: PremiumPODBasis, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            option_type=basis.option_type,
            mean_premium=basis.mean_premium,
            components=basis.components,
            singular_values=basis.singular_values,
            positive_tau_grid=basis.positive_tau_grid,
            interior_moneyness_grid=basis.interior_moneyness_grid,
            train_regime_ids=np.asarray(basis.train_regime_ids),
            metadata_json=json.dumps(basis.metadata, sort_keys=True),
        )
    return destination


def load_premium_basis(path: Path | str) -> PremiumPODBasis:
    with np.load(path, allow_pickle=False) as data:
        return PremiumPODBasis(
            str(data["option_type"]), data["mean_premium"].copy(),
            data["components"].copy(), data["singular_values"].copy(),
            data["positive_tau_grid"].copy(), data["interior_moneyness_grid"].copy(),
            tuple(map(str, data["train_regime_ids"])),
            json.loads(str(data["metadata_json"])),
        )


def basis_sha256(path: Path | str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
