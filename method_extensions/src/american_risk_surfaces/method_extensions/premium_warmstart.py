"""Checkpointed positive-premium MLP and safe LCP warm-start utilities."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.surrogates import price_premium as stage3


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WARMSTART_DIR = (
    PROJECT_ROOT / "results" / "07_method_extensions" / "02_warmstart"
)
DEFAULT_CHECKPOINT_PATH = DEFAULT_WARMSTART_DIR / "checkpoints" / "positive_premium_mlp.pt"
SUPPORT_MONEYNESS = (0.4, 1.8)

__all__ = (
    "DEFAULT_CHECKPOINT_PATH",
    "PositivePremiumSurfaceModel",
    "SurfacePrediction",
    "GatedSurfaceInitializer",
    "train_positive_premium_checkpoint",
)


@dataclass(frozen=True)
class SurfacePrediction:
    premium_grid: np.ndarray
    value_grid: np.ndarray
    inference_seconds: float


class PositivePremiumSurfaceModel:
    """Loaded Stage 3 network plus explicit scaler and provenance metadata."""

    def __init__(self, model: Any, scaler_mean: np.ndarray, scaler_scale: np.ndarray, metadata: dict[str, Any]):
        self.model = model
        self.scaler_mean = np.asarray(scaler_mean, dtype=float)
        self.scaler_scale = np.asarray(scaler_scale, dtype=float)
        self.metadata = dict(metadata)
        if self.scaler_mean.shape != (len(stage3.FEATURE_NAMES) + 1,):
            raise ValueError("checkpoint scaler mean has the wrong shape.")
        if self.scaler_scale.shape != self.scaler_mean.shape:
            raise ValueError("checkpoint scaler scale has the wrong shape.")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CHECKPOINT_PATH) -> "PositivePremiumSurfaceModel":
        stage3._require_torch()
        checkpoint = stage3.torch.load(Path(path), map_location="cpu", weights_only=False)
        model = stage3._MLP(
            input_dim=int(checkpoint["architecture"]["input_dim"]),
            hidden_units=int(checkpoint["architecture"]["hidden_units"]),
            output_activation=str(checkpoint["architecture"]["output_activation"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(
            model=model,
            scaler_mean=np.asarray(checkpoint["scaler_mean"], dtype=float),
            scaler_scale=np.asarray(checkpoint["scaler_scale"], dtype=float),
            metadata=dict(checkpoint["metadata"]),
        )

    def predict_surface(
        self,
        config: AmericanLCPConfig,
        spot_grid: np.ndarray,
        tau_grid: np.ndarray,
        *,
        batch_size: int = 131072,
    ) -> SurfacePrediction:
        """Predict a normalized-premium surface for every interior grid node."""

        spots = np.asarray(spot_grid, dtype=float)
        taus = np.asarray(tau_grid, dtype=float)
        if spots.ndim != 1 or taus.ndim != 1 or len(spots) < 3 or len(taus) < 1:
            raise ValueError("spot_grid and tau_grid must be one-dimensional valid grids.")
        interior = spots[1:-1]
        if np.any(interior <= 0.0):
            raise ValueError("interior spot nodes must be positive for log-moneyness.")
        payoff_function = call_payoff if config.option_type == "call" else put_payoff
        payoff_over_k = np.asarray(payoff_function(interior, config.K), dtype=float) / config.K
        tau_fraction = np.zeros_like(taus) if config.T == 0.0 else taus / config.T
        row_count = len(taus) * len(interior)
        features = np.empty((row_count, len(stage3.FEATURE_NAMES)), dtype=float)
        features[:, 0] = np.tile(np.log(interior / config.K), len(taus))
        features[:, 1] = np.repeat(tau_fraction, len(interior))
        features[:, 2] = config.r
        features[:, 3] = config.q
        features[:, 4] = config.sigma
        features[:, 5] = config.T
        features[:, 6] = 1.0 if config.option_type == "call" else 0.0
        augmented = np.column_stack([features, np.tile(payoff_over_k, len(taus))])
        scaled = ((augmented - self.scaler_mean) / self.scaler_scale).astype(np.float32)

        started = perf_counter()
        predicted = np.empty(row_count, dtype=float)
        with stage3.torch.no_grad():
            for start in range(0, row_count, batch_size):
                stop = min(start + batch_size, row_count)
                batch = stage3.torch.from_numpy(scaled[start:stop])
                predicted[start:stop] = self.model(batch).detach().cpu().numpy()
        inference_seconds = perf_counter() - started
        premium = config.K * predicted.reshape(len(taus), len(interior))
        payoff = config.K * payoff_over_k
        value = payoff[np.newaxis, :] + premium
        return SurfacePrediction(
            premium_grid=premium,
            value_grid=value,
            inference_seconds=float(inference_seconds),
        )


class GatedSurfaceInitializer:
    """Use neural values only on the evidenced moneyness support."""

    name = "positive_premium_mlp_gated"

    def __init__(
        self,
        predicted_value_grid: np.ndarray,
        interior_spots: np.ndarray,
        K: float,
        *,
        support: tuple[float, float] = SUPPORT_MONEYNESS,
        raw_extrapolation: bool = False,
    ) -> None:
        self.predicted_value_grid = np.asarray(predicted_value_grid, dtype=float)
        spots = np.asarray(interior_spots, dtype=float)
        self.support_mask = (spots / float(K) >= support[0]) & (spots / float(K) <= support[1])
        if raw_extrapolation:
            self.support_mask[:] = True
            self.name = "positive_premium_mlp_raw_extrapolation"
        if self.predicted_value_grid.ndim != 2:
            raise ValueError("predicted_value_grid must be two-dimensional.")
        if self.predicted_value_grid.shape[1] != len(spots):
            raise ValueError("prediction spot dimension must match interior_spots.")

    def __call__(
        self,
        step_index: int,
        _tau: float,
        previous_values: np.ndarray,
        obstacle: np.ndarray,
    ) -> np.ndarray:
        if step_index < 0 or step_index >= self.predicted_value_grid.shape[0]:
            raise ValueError("step_index is outside the predicted time grid.")
        predicted = np.maximum(self.predicted_value_grid[step_index], obstacle)
        return np.where(self.support_mask, predicted, previous_values)


def train_positive_premium_checkpoint(
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    *,
    bundle: stage3.SurrogateDatasetBundle | None = None,
    train_cap: int = stage3.TRAIN_ROW_CAP,
    epochs: int = 10,
    batch_size: int = 8192,
    hidden_units: int = 64,
) -> dict[str, Path]:
    """Deterministically retrain Stage 3 and save a new extension-only artifact."""

    stage3._require_torch()
    stage3._set_random_seed(stage3.RANDOM_SEED)
    if hasattr(stage3.torch, "set_num_threads"):
        stage3.torch.set_num_threads(1)
    bundle = stage3.load_v1_dataset() if bundle is None else bundle
    split_map = stage3.split_masks(bundle)
    train_indices = np.flatnonzero(split_map["train"])
    selected_train = stage3.capped_train_indices(
        train_indices, cap=train_cap, seed=stage3.RANDOM_SEED
    )
    validation_indices = np.flatnonzero(split_map["validation"])
    preprocessor = stage3.fit_preprocessor(bundle, train_indices)
    config = stage3.TrainingConfig(
        seed=stage3.RANDOM_SEED,
        train_cap=train_cap,
        epochs=epochs,
        batch_size=batch_size,
        hidden_units=hidden_units,
    )
    started = perf_counter()
    model, history = stage3._train_positive_premium_model(
        bundle, preprocessor, selected_train, validation_indices, config
    )
    training_seconds = perf_counter() - started
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash = _sha256(bundle.dataset_path) if bundle.dataset_path.exists() else "synthetic"
    metadata = {
        "artifact_role": "lcp_warmstart_only",
        "source_stage": "deterministic_stage3_positive_premium_retraining",
        "dataset_path": str(bundle.dataset_path),
        "dataset_sha256": dataset_hash,
        "feature_names": list(stage3.FEATURE_NAMES),
        "target": "premium_over_K",
        "output_reconstruction": "value=payoff+K*softplus(raw)",
        "support_moneyness": list(SUPPORT_MONEYNESS),
        "seed": stage3.RANDOM_SEED,
        "model_seed": stage3.RANDOM_SEED + 17,
        "train_cap": train_cap,
        "selected_train_rows": int(selected_train.size),
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "training_seconds": training_seconds,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": stage3.torch.__version__,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": {
            "input_dim": len(stage3.FEATURE_NAMES) + 1,
            "hidden_units": hidden_units,
            "output_activation": "softplus",
        },
        "scaler_mean": preprocessor.premium_scaler.mean_.copy(),
        "scaler_scale": preprocessor.premium_scaler.scale_.copy(),
        "metadata": metadata,
        "training_history": history,
    }
    stage3.torch.save(checkpoint, checkpoint_path)
    manifest_path = checkpoint_path.with_suffix(".json")
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["checkpoint_sha256"] = _sha256(checkpoint_path)
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"checkpoint": checkpoint_path, "manifest": manifest_path}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
