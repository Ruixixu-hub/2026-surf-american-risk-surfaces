"""Public immutable artifacts for the positive-premium basis operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class PremiumPODBasis:
    option_type: str
    mean_premium: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    positive_tau_grid: np.ndarray
    interior_moneyness_grid: np.ndarray
    train_regime_ids: tuple[str, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class BasisOperatorTrainingConfig:
    option_type: str
    modes: int
    loss_variant: Literal["coefficient", "structure_aware"]
    seed: int
    steps: int
    batch_size: int = 16
    learning_rate: float = 1e-3
    dtype: str = "float64"
    checkpoint_interval: int = 500


@dataclass(frozen=True)
class BasisOperatorArtifact:
    basis: PremiumPODBasis
    state_dict: dict[str, Any]
    input_scaler_mean: np.ndarray
    input_scaler_scale: np.ndarray
    coefficient_scale: np.ndarray
    config: dict[str, object]
    hashes: dict[str, str]


@dataclass(frozen=True)
class BasisOperatorPrediction:
    raw_premium_grid: np.ndarray
    projected_premium_grid: np.ndarray
    value_grid: np.ndarray
    timing: dict[str, float]
    control_branch: str


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    history_path: Path
    status: Literal["COMPLETE", "FAILED"]
    final_loss: float
    failure_reason: str | None = None
