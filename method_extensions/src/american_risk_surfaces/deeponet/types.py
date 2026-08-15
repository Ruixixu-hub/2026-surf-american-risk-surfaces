"""Immutable public artifacts for the positive-premium DeepONet study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


DeepONetArm = Literal["N0", "N1", "N2"]


@dataclass(frozen=True)
class DeepONetTrainingConfig:
    option_type: str
    arm: DeepONetArm
    latent_rank: int
    seed: int
    steps: int
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "auto"
    dtype: str = "float64"
    checkpoint_interval: int = 1000
    time_budget_seconds: float | None = None


@dataclass(frozen=True)
class DeepONetTrainingBundle:
    option_type: str
    regime_ids: tuple[str, ...]
    features: np.ndarray
    features_scaled: np.ndarray
    input_scaler_mean: np.ndarray
    input_scaler_scale: np.ndarray
    coordinate_grid: np.ndarray
    premium_surfaces: np.ndarray
    value_surfaces: np.ndarray
    payoff: np.ndarray
    boundary_mask: np.ndarray
    continuation_mask: np.ndarray
    premium_rms: float
    derivative_rms: float
    class_weights: np.ndarray
    regimes: tuple[dict[str, object], ...]
    hashes: dict[str, str]


@dataclass(frozen=True)
class DeepONetArtifact:
    state_dict: dict[str, Any]
    input_scaler_mean: np.ndarray
    input_scaler_scale: np.ndarray
    config: dict[str, object]
    hashes: dict[str, str]


@dataclass(frozen=True)
class DeepONetPrediction:
    raw_premium_grid: np.ndarray
    projected_premium_grid: np.ndarray
    value_grid: np.ndarray
    timing: dict[str, float]
    control_branch: str
    ad_delta_grid: np.ndarray | None = None
    ad_gamma_grid: np.ndarray | None = None


@dataclass(frozen=True)
class DeepONetTrainingResult:
    status: Literal["COMPLETE", "FAILED", "BUDGET_EXHAUSTED"]
    checkpoint_path: Path
    history_path: Path
    training_seconds: float
    failure_reason: str | None
