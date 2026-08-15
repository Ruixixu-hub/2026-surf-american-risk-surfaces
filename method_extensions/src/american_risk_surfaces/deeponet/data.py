"""Train-only full-grid bundle construction for positive-premium DeepONet."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from american_risk_surfaces.deeponet.protocol import (
    PREMIUM_THRESHOLD,
    assert_training_regime_allowed,
    assert_training_snapshot_allowed,
    protocol_hash,
)
from american_risk_surfaces.deeponet.types import DeepONetTrainingBundle
from american_risk_surfaces.reduced_order.protocol import sha256_file
from american_risk_surfaces.reduced_order.snapshots import load_snapshot


def cartesian_coordinate_grid(M: int = 120, N: int = 120) -> np.ndarray:
    """Return positive-time/interior-space coordinates in row-major surface order."""

    if M < 2 or N < 1:
        raise ValueError("M must be at least 2 and N must be positive")
    m = np.linspace(0.0, 4.0, M + 1)[1:-1]
    s = np.linspace(0.0, 1.0, N + 1)[1:]
    ss, mm = np.meshgrid(s, m, indexing="ij")
    return np.column_stack((mm.reshape(-1) / 2.0 - 1.0, 2.0 * ss.reshape(-1) - 1.0))


def build_deeponet_training_bundle(
    train_snapshot_manifest: Iterable[Path | str], option_type: str
) -> DeepONetTrainingBundle:
    family = str(option_type).lower()
    if family not in {"put", "call"}:
        raise ValueError("option_type must be put or call")
    paths = sorted(map(Path, train_snapshot_manifest))
    if not paths:
        raise ValueError("train snapshot manifest is empty")
    features: list[list[float]] = []
    premiums: list[np.ndarray] = []
    values: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    continuations: list[np.ndarray] = []
    identifiers: list[str] = []
    regimes: list[dict[str, object]] = []
    payoff: np.ndarray | None = None
    for path in paths:
        assert_training_snapshot_allowed(path, family)
        snapshot = load_snapshot(path)
        regime = dict(snapshot.metadata["regime"])
        assert_training_regime_allowed(
            str(regime["split"]), snapshot.option_type, float(regime["q"])
        )
        if snapshot.option_type != family:
            raise PermissionError("put and call DeepONet data must remain separate")
        if snapshot.value_grid.shape != (121, 121):
            raise ValueError("formal DeepONet snapshots must use a 121x121 grid")
        current_payoff = np.asarray(snapshot.payoff, dtype=float)
        if payoff is None:
            payoff = current_payoff.copy()
        elif not np.array_equal(payoff, current_payoff):
            raise ValueError("family snapshots must share the frozen normalized payoff")
        premium = (
            np.asarray(snapshot.value_grid[1:, 1:-1], dtype=float)
            - current_payoff[np.newaxis, 1:-1]
        ) / float(regime["K"])
        if float(np.min(premium)) < -1e-12:
            raise RuntimeError(
                f"snapshot {snapshot.regime_id} violates the frozen obstacle tolerance"
            )
        premium[np.abs(premium) < 5e-15] = 0.0
        premium = np.maximum(premium, 0.0)
        continuation = premium > PREMIUM_THRESHOLD
        boundary = _boundary_mask(continuation)
        features.append([
            math.log(float(regime["T"])), float(regime["sigma"]),
            float(regime["r"]), float(regime["q"]),
        ])
        premiums.append(premium)
        values.append(np.asarray(snapshot.value_grid, dtype=float))
        boundaries.append(boundary)
        continuations.append(continuation)
        identifiers.append(snapshot.regime_id)
        regimes.append(regime)
    feature_array = np.asarray(features, dtype=float)
    feature_mean = np.mean(feature_array, axis=0)
    feature_scale = np.std(feature_array, axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    premium_array = np.asarray(premiums, dtype=float)
    value_array = np.asarray(values, dtype=float)
    boundary_array = np.asarray(boundaries, dtype=bool)
    continuation_array = np.asarray(continuations, dtype=bool)
    derivative = np.diff(premium_array, axis=2)
    premium_rms = max(float(np.sqrt(np.mean(premium_array**2))), 1e-12)
    derivative_rms = max(float(np.sqrt(np.mean(derivative**2))), 1e-12)
    class_weights = _balanced_class_weights(continuation_array)
    return DeepONetTrainingBundle(
        family,
        tuple(identifiers),
        feature_array,
        (feature_array - feature_mean) / feature_scale,
        feature_mean,
        feature_scale,
        cartesian_coordinate_grid(),
        premium_array,
        value_array,
        np.asarray(payoff, dtype=float),
        boundary_array,
        continuation_array,
        premium_rms,
        derivative_rms,
        class_weights,
        tuple(regimes),
        {
            "protocol": protocol_hash(),
            "snapshots": sha256_manifest(paths),
        },
    )


def _boundary_mask(continuation: np.ndarray) -> np.ndarray:
    state = np.asarray(continuation, dtype=bool)
    if state.shape != (120, 119):
        raise ValueError("continuation mask must have shape (120,119)")
    mask = np.zeros_like(state)
    transitions = state[:, 1:] != state[:, :-1]
    mask[:, 1:] |= transitions
    mask[:, :-1] |= transitions
    mask[:, 2:] |= transitions[:, :-1]
    mask[:, :-2] |= transitions[:, 1:]
    return mask


def _balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    continuation = np.asarray(labels, dtype=bool)
    positive = float(np.mean(continuation))
    negative = 1.0 - positive
    weights = np.asarray([
        0.5 / max(negative, 1e-15),
        0.5 / max(positive, 1e-15),
    ])
    weights = np.clip(weights, 0.25, 4.0)
    normalizer = negative * weights[0] + positive * weights[1]
    return weights / max(normalizer, 1e-15)


def sha256_manifest(paths: Iterable[Path | str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(map(Path, paths)):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()
