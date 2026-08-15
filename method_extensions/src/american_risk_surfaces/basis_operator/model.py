"""Small nonlinear maps from option parameters to POD coefficients."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.basis_operator.basis import (
    basis_sha256,
    load_training_matrix,
    project_premium_coefficients,
)
from american_risk_surfaces.basis_operator.protocol import protocol_hash
from american_risk_surfaces.basis_operator.types import (
    BasisOperatorArtifact,
    BasisOperatorTrainingConfig,
    PremiumPODBasis,
    TrainingResult,
)
from american_risk_surfaces.reduced_order.snapshots import load_snapshot


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - reported clearly at runtime
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for basis-operator training")


class BasisCoefficientNetwork(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        _require_torch()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, output_dim),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs):
        return self.layers(inputs)


def regime_features(snapshot_paths: Iterable[Path | str], option_type: str) -> np.ndarray:
    rows = []
    for path in sorted(map(Path, snapshot_paths)):
        snapshot = load_snapshot(path)
        regime = snapshot.metadata["regime"]
        if regime["split"] != "train" or snapshot.option_type != option_type:
            raise PermissionError("operator features must be paired with train-only family snapshots")
        rows.append([math.log(float(regime["T"])), regime["sigma"], regime["r"], regime["q"]])
    return np.asarray(rows, dtype=float)


def prepare_training_arrays(
    basis: PremiumPODBasis, snapshot_paths: Iterable[Path | str]
) -> dict[str, np.ndarray]:
    paths = sorted(map(Path, snapshot_paths))
    matrix, identifiers, _tau, _spot = load_training_matrix(paths, basis.option_type)
    if identifiers != basis.train_regime_ids:
        raise RuntimeError("training row order does not match basis provenance")
    features = regime_features(paths, basis.option_type)
    input_mean = np.mean(features, axis=0)
    input_scale = np.std(features, axis=0)
    input_scale[input_scale < 1e-12] = 1.0
    coefficients = np.einsum(
        "ni,mi->nm", matrix - basis.mean_premium, basis.components, optimize=False
    )
    coefficient_scale = np.std(coefficients, axis=0)
    coefficient_scale[coefficient_scale < 1e-12] = 1.0
    exercise = matrix <= 1e-6
    boundary = np.zeros_like(exercise, dtype=bool)
    shaped = exercise.reshape(len(matrix), 120, 119)
    shaped_boundary = boundary.reshape(len(matrix), 120, 119)
    transitions = shaped[:, :, 1:] != shaped[:, :, :-1]
    shaped_boundary[:, :, 1:] |= transitions
    shaped_boundary[:, :, :-1] |= transitions
    shaped_boundary[:, :, 2:] |= transitions[:, :, :-1]
    shaped_boundary[:, :, :-2] |= transitions[:, :, 1:]
    premium_rms = max(float(np.sqrt(np.mean(matrix**2))), 1e-12)
    derivative = np.diff(matrix.reshape(len(matrix), 120, 119), axis=2)
    derivative_rms = max(float(np.sqrt(np.mean(derivative**2))), 1e-12)
    return {
        "features": features,
        "features_scaled": (features - input_mean) / input_scale,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "coefficients": coefficients,
        "coefficient_scale": coefficient_scale,
        "coefficients_scaled": coefficients / coefficient_scale,
        "premium": matrix,
        "exercise_mask": exercise,
        "boundary_mask": boundary,
        "premium_rms": np.asarray(premium_rms),
        "derivative_rms": np.asarray(derivative_rms),
    }


def train_basis_coefficient_operator(
    basis: PremiumPODBasis,
    train_manifest: Iterable[Path | str],
    config: BasisOperatorTrainingConfig,
    *,
    output_dir: Path | str,
    basis_path: Path | str | None = None,
) -> TrainingResult:
    _require_torch()
    scoring_marker = Path(__file__).resolve().parents[3] / "results/11_positive_premium_basis_operator/06_heldout/SCORING_COMPLETE_DO_NOT_RETRAIN.json"
    if scoring_marker.exists():
        raise PermissionError("heldout was scored permanently; operator retraining is locked")
    if config.option_type != basis.option_type or config.modes != basis.components.shape[0]:
        raise ValueError("training config and basis do not match")
    if config.dtype != "float64":
        raise ValueError("formal training is frozen to float64")
    if config.loss_variant not in {"coefficient", "structure_aware"}:
        raise ValueError("unknown loss variant")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = prepare_training_arrays(basis, train_manifest)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    dtype = torch.float64
    model = BasisCoefficientNetwork(4, config.modes).to(dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.steps, 1), eta_min=1e-5
    )
    tensors = {
        key: torch.as_tensor(value, dtype=dtype)
        for key, value in arrays.items()
        if key in {"features_scaled", "coefficients_scaled", "coefficient_scale", "premium"}
    }
    exercise = torch.as_tensor(arrays["exercise_mask"], dtype=torch.bool)
    boundary = torch.as_tensor(arrays["boundary_mask"], dtype=torch.bool)
    mean = torch.as_tensor(basis.mean_premium, dtype=dtype)
    components = torch.as_tensor(basis.components, dtype=dtype)
    generator = torch.Generator().manual_seed(config.seed + 1009)
    history: list[dict[str, float | int]] = []
    checkpoint_path = output / "checkpoint.pt"
    started = perf_counter()
    failure_reason = None
    rescued = False
    for step in range(1, config.steps + 1):
        indices = torch.randint(
            0, len(tensors["features_scaled"]), (config.batch_size,), generator=generator
        )
        prediction_scaled = model(tensors["features_scaled"][indices])
        coefficient_loss = torch.mean(
            (prediction_scaled - tensors["coefficients_scaled"][indices]) ** 2
        )
        surface_loss = torch.zeros((), dtype=dtype)
        boundary_loss = torch.zeros((), dtype=dtype)
        derivative_loss = torch.zeros((), dtype=dtype)
        exercise_loss = torch.zeros((), dtype=dtype)
        if config.loss_variant == "structure_aware":
            coefficients = prediction_scaled * tensors["coefficient_scale"]
            raw = mean + coefficients @ components
            projected = torch.relu(raw)
            target = tensors["premium"][indices]
            surface_loss = torch.mean((projected - target) ** 2) / float(arrays["premium_rms"] ** 2)
            batch_boundary = boundary[indices]
            if torch.any(batch_boundary):
                boundary_loss = torch.mean((projected[batch_boundary] - target[batch_boundary]) ** 2) / float(arrays["premium_rms"] ** 2)
            shaped_prediction = projected.reshape(config.batch_size, 120, 119)
            shaped_target = target.reshape(config.batch_size, 120, 119)
            derivative_loss = torch.mean(
                (torch.diff(shaped_prediction, dim=2) - torch.diff(shaped_target, dim=2)) ** 2
            ) / float(arrays["derivative_rms"] ** 2)
            batch_exercise = exercise[indices]
            if torch.any(batch_exercise):
                exercise_loss = torch.mean(raw[batch_exercise] ** 2) / float(arrays["premium_rms"] ** 2)
        loss = coefficient_loss + surface_loss + 4.0 * boundary_loss + 0.1 * derivative_loss + 0.1 * exercise_loss
        if not torch.isfinite(loss):
            if rescued:
                failure_reason = "non-finite loss after one learning-rate rescue"
                break
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
            rescued = True
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "coefficient_loss": float(coefficient_loss.detach()),
                "surface_loss": float(surface_loss.detach()),
                "boundary_loss": float(boundary_loss.detach()),
                "derivative_loss": float(derivative_loss.detach()),
                "exercise_loss": float(exercise_loss.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
        if step % config.checkpoint_interval == 0 or step == config.steps:
            _save_checkpoint(
                checkpoint_path, model, basis, arrays, config, basis_path,
                final_loss=float(loss.detach()), training_seconds=perf_counter() - started,
            )
    history_path = output / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(history[0]) if history else ("step", "loss"))
        writer.writeheader()
        writer.writerows(history)
    if failure_reason is not None:
        (output / "failure.json").write_text(
            json.dumps({"failure_reason": failure_reason}, indent=2), encoding="utf-8"
        )
        return TrainingResult(checkpoint_path, history_path, "FAILED", float("inf"), failure_reason)
    return TrainingResult(checkpoint_path, history_path, "COMPLETE", float(history[-1]["loss"]))


def _save_checkpoint(
    path: Path,
    model,
    basis: PremiumPODBasis,
    arrays: dict[str, np.ndarray],
    config: BasisOperatorTrainingConfig,
    basis_path: Path | str | None,
    *,
    final_loss: float,
    training_seconds: float,
) -> None:
    basis_hash = basis_sha256(basis_path) if basis_path is not None else "embedded-not-file-hashed"
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "basis": basis,
        "input_scaler_mean": arrays["input_mean"],
        "input_scaler_scale": arrays["input_scale"],
        "coefficient_scale": arrays["coefficient_scale"],
        "config": asdict(config),
        "hashes": {"protocol": protocol_hash(), "basis": basis_hash},
        "metadata": {
            "feature_order": ["log(T)", "sigma", "r", "q"],
            "architecture": "3x64 SiLU; zero-initialized final layer",
            "final_loss": final_loss,
            "training_seconds": training_seconds,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_basis_operator_artifact(path: Path | str) -> BasisOperatorArtifact:
    _require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload["hashes"]["protocol"] != protocol_hash():
        raise RuntimeError("checkpoint protocol hash does not match the frozen study")
    return BasisOperatorArtifact(
        payload["basis"], payload["state_dict"],
        np.asarray(payload["input_scaler_mean"], dtype=float),
        np.asarray(payload["input_scaler_scale"], dtype=float),
        np.asarray(payload["coefficient_scale"], dtype=float),
        dict(payload["config"]), dict(payload["hashes"]),
    )


def infer_coefficients(artifact: BasisOperatorArtifact, features: np.ndarray) -> np.ndarray:
    _require_torch()
    model = BasisCoefficientNetwork(4, artifact.basis.components.shape[0]).to(dtype=torch.float64)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    scaled = (np.asarray(features, dtype=float) - artifact.input_scaler_mean) / artifact.input_scaler_scale
    with torch.no_grad():
        standardized = model(torch.as_tensor(scaled, dtype=torch.float64)).cpu().numpy()
    return standardized * artifact.coefficient_scale


def infer_coefficients_numpy(
    artifact: BasisOperatorArtifact, features: np.ndarray
) -> np.ndarray:
    """Dependency-light exported inference exactly matching the PyTorch network."""

    values = (np.asarray(features, dtype=float) - artifact.input_scaler_mean) / artifact.input_scaler_scale
    state = artifact.state_dict
    for layer in (0, 2, 4):
        weight = np.asarray(state[f"layers.{layer}.weight"], dtype=float)
        bias = np.asarray(state[f"layers.{layer}.bias"], dtype=float)
        values = np.einsum("ni,oi->no", values, weight, optimize=False) + bias
        values = values / (1.0 + np.exp(-values))
    weight = np.asarray(state["layers.6.weight"], dtype=float)
    bias = np.asarray(state["layers.6.bias"], dtype=float)
    return (
        np.einsum("ni,oi->no", values, weight, optimize=False) + bias
    ) * artifact.coefficient_scale
