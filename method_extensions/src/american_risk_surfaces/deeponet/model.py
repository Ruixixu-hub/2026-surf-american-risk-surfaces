"""Branch/trunk network, frozen losses, checkpointing, and inference."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from american_risk_surfaces.deeponet.physics import batched_cn_lcp_residual
from american_risk_surfaces.deeponet.protocol import FB_EPSILON, protocol_hash
from american_risk_surfaces.deeponet.types import (
    DeepONetArtifact,
    DeepONetTrainingBundle,
    DeepONetTrainingConfig,
    DeepONetTrainingResult,
)


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for DeepONet")


class PositivePremiumDeepONet(nn.Module if nn is not None else object):
    def __init__(self, latent_rank: int) -> None:
        _require_torch()
        super().__init__()
        rank = int(latent_rank)
        if rank not in {32, 64, 128}:
            raise ValueError("latent_rank must be 32, 64, or 128")
        self.latent_rank = rank
        self.branch = _mlp((4, 128, 128, 128, rank))
        self.trunk = _mlp((2, 128, 128, 128, 128, rank))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.bias)

    def encode_branch(self, features):
        return self.branch(features)

    def encode_trunk(self, coordinates):
        return self.trunk(coordinates)

    def contract(self, branch_latent, trunk_latent):
        if branch_latent.shape[-1] != self.latent_rank:
            raise ValueError("branch latent width does not match model rank")
        if trunk_latent.shape[-1] != self.latent_rank:
            raise ValueError("trunk latent width does not match model rank")
        return branch_latent @ trunk_latent.T / math.sqrt(self.latent_rank) + self.bias

    def forward(self, features, coordinates):
        return self.contract(self.encode_branch(features), self.encode_trunk(coordinates))


def _mlp(widths: tuple[int, ...]):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def count_parameters(model) -> int:
    return int(sum(item.numel() for item in model.parameters()))


def train_positive_premium_deeponet(
    training_bundle: DeepONetTrainingBundle,
    config: DeepONetTrainingConfig,
    *,
    output_dir: Path | str,
    resume: bool = False,
) -> DeepONetTrainingResult:
    _require_torch()
    _validate_training_config(training_bundle, config)
    marker = (
        Path(__file__).resolve().parents[3]
        / "results/12_positive_premium_deeponet/04_heldout/SCORING_COMPLETE_DO_NOT_RETRAIN.json"
    )
    if marker.exists():
        raise PermissionError("heldout was scored permanently; DeepONet retraining is locked")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    history_path = output / "training_history.csv"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = _resolve_device(config.device)
    dtype = torch.float64
    model = PositivePremiumDeepONet(config.latent_rank).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.steps, 1), eta_min=1e-5
    )
    generator = torch.Generator().manual_seed(config.seed + 1709)
    start_step = 0
    previous_training_seconds = 0.0
    rescued = False
    history: list[dict[str, float | int]] = []
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _assert_checkpoint_compatible(payload, training_bundle, config)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if "generator_state" in payload:
            generator.set_state(payload["generator_state"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        start_step = int(payload["step"])
        previous_training_seconds = float(payload.get("training_seconds", 0.0))
        rescued = bool(payload.get("rescued", False))
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = [dict(item) for item in csv.DictReader(handle)]
    tensors = _bundle_tensors(training_bundle, device, dtype)
    started = perf_counter()
    failure_reason = None
    status = "COMPLETE"
    last_loss = float("inf")
    for step in range(start_step + 1, config.steps + 1):
        if (
            config.time_budget_seconds is not None
            and previous_training_seconds + perf_counter() - started >= config.time_budget_seconds
        ):
            status = "BUDGET_EXHAUSTED"
            failure_reason = "formal per-seed wall-clock budget exhausted"
            break
        indices = torch.randint(
            0, len(training_bundle.regime_ids), (config.batch_size,), generator=generator
        ).to(device=device)
        features = tensors["features_scaled"].index_select(0, indices)
        raw = model(features, tensors["coordinates"])
        raw_surface = raw.reshape(config.batch_size, 120, 119)
        projected = torch.relu(raw_surface)
        target = tensors["premium"].index_select(0, indices)
        losses = compute_deeponet_losses(
            raw_surface, projected, target,
            tensors["boundary"].index_select(0, indices),
            tensors["continuation"].index_select(0, indices),
            training_bundle, tensors, indices, config.arm,
        )
        loss = losses["total"]
        if not torch.isfinite(loss):
            if rescued:
                status, failure_reason = "FAILED", "non-finite loss after one learning-rate rescue"
                break
            if checkpoint_path.exists():
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(payload["state_dict"])
                optimizer.load_state_dict(payload["optimizer_state"])
                scheduler.load_state_dict(payload["scheduler_state"])
                if "generator_state" in payload:
                    generator.set_state(payload["generator_state"])
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
            rescued = True
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % 100 == 0 or step == config.steps:
            history.append({
                "step": step,
                "total": last_loss,
                **{key: float(value.detach()) for key, value in losses.items() if key != "total"},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(previous_training_seconds + perf_counter() - started),
            })
            _write_history(history_path, history)
            _atomic_json(output / "heartbeat.json", {
                "status": "RUNNING", "step": step, "target_steps": config.steps,
                "last_loss": last_loss,
                "elapsed_seconds": previous_training_seconds + perf_counter() - started,
                "protocol_hash": training_bundle.hashes["protocol"],
            })
        if step % config.checkpoint_interval == 0 or step == config.steps:
            _save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, training_bundle, config,
                step=step, last_loss=last_loss,
                training_seconds=previous_training_seconds + perf_counter() - started,
                rescued=rescued, generator=generator,
            )
    elapsed = previous_training_seconds + perf_counter() - started
    if status != "COMPLETE":
        _save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, training_bundle, config,
            step=min(config.steps, max(start_step, step - 1)), last_loss=last_loss,
            training_seconds=elapsed, rescued=rescued, generator=generator,
            status=status,
        )
        (output / "failure.json").write_text(
            json.dumps({"status": status, "failure_reason": failure_reason}, indent=2),
            encoding="utf-8",
        )
    _atomic_json(output / "status.json", {
        "status": status, "step": (
            config.steps if status == "COMPLETE"
            else min(config.steps, max(start_step, step - 1))
        ),
        "target_steps": config.steps, "training_seconds": elapsed,
        "failure_reason": failure_reason,
        "protocol_hash": training_bundle.hashes["protocol"],
    })
    return DeepONetTrainingResult(
        status, checkpoint_path, history_path, float(elapsed), failure_reason
    )


def compute_deeponet_losses(
    raw_surface, projected, target, boundary, continuation,
    bundle: DeepONetTrainingBundle, tensors: dict[str, object], indices, arm: str,
) -> dict[str, object]:
    surface = torch.mean((projected - target).square()) / (bundle.premium_rms**2)
    boundary_loss = torch.zeros((), dtype=raw_surface.dtype, device=raw_surface.device)
    derivative = torch.zeros_like(boundary_loss)
    exercise = torch.zeros_like(boundary_loss)
    fb_loss = torch.zeros_like(boundary_loss)
    if arm in {"N1", "N2"}:
        if torch.any(boundary):
            boundary_loss = torch.mean((projected[boundary] - target[boundary]).square()) / (
                bundle.premium_rms**2
            )
        derivative = torch.mean(
            (torch.diff(projected, dim=2) - torch.diff(target, dim=2)).square()
        ) / (bundle.derivative_rms**2)
        logits = raw_surface / 1e-4
        per_node = F.binary_cross_entropy_with_logits(
            logits, continuation.to(dtype=raw_surface.dtype), reduction="none"
        )
        weights = torch.where(
            continuation, tensors["class_weights"][1], tensors["class_weights"][0]
        )
        exercise = torch.mean(per_node * weights)
    if arm == "N2":
        value_grid = reconstruct_torch_value_grid(
            projected, tensors["payoff"], tensors["regimes"], indices,
        )
        selected = tensors["regimes"].index_select(0, indices)
        residual = batched_cn_lcp_residual(
            value_grid, tensors["payoff"], selected[:, 0], selected[:, 1],
            selected[:, 2], selected[:, 3], epsilon=FB_EPSILON,
        )
        fb_loss = torch.mean(residual.fischer_burmeister.square())
    total = surface
    if arm in {"N1", "N2"}:
        total = total + 4.0 * boundary_loss + 0.1 * derivative + 0.1 * exercise
    if arm == "N2":
        total = total + 0.1 * fb_loss
    return {
        "total": total,
        "surface": surface,
        "boundary": boundary_loss,
        "derivative": derivative,
        "exercise": exercise,
        "fb": fb_loss,
    }


def reconstruct_torch_value_grid(projected, payoff, regime_parameters, indices):
    """Insert exact terminal/boundary values while retaining gradients inside."""

    batch = projected.shape[0]
    device, dtype = projected.device, projected.dtype
    values = torch.zeros((batch, 121, 121), dtype=dtype, device=device)
    values[:, 0, :] = payoff.unsqueeze(0)
    values[:, 1:, 1:-1] = payoff[None, None, 1:-1] + projected
    selected = regime_parameters.index_select(0, indices)
    T, r, q, family_code = selected[:, 0], selected[:, 2], selected[:, 3], selected[:, 4]
    fraction = torch.linspace(0.0, 1.0, 121, dtype=dtype, device=device)[None, :]
    tau = T[:, None] * fraction
    put = family_code[:, None] < 0.5
    left = torch.where(put, torch.ones_like(tau), torch.zeros_like(tau))
    call_right = torch.maximum(
        torch.full_like(tau, 3.0),
        4.0 * torch.exp(-q[:, None] * tau) - torch.exp(-r[:, None] * tau),
    )
    right = torch.where(put, torch.zeros_like(tau), call_right)
    values[:, :, 0] = left
    values[:, :, -1] = right
    values[:, 0, :] = payoff.unsqueeze(0)
    return values


def load_deeponet_artifact(path: Path | str) -> DeepONetArtifact:
    _require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload["hashes"]["protocol"] != protocol_hash():
        raise RuntimeError("checkpoint protocol hash does not match the frozen DeepONet study")
    if payload.get("status", "COMPLETE") != "COMPLETE" or int(payload["step"]) != int(payload["config"]["steps"]):
        raise RuntimeError("checkpoint is not a complete formal artifact")
    return DeepONetArtifact(
        dict(payload["state_dict"]),
        np.asarray(payload["input_scaler_mean"], dtype=float),
        np.asarray(payload["input_scaler_scale"], dtype=float),
        dict(payload["config"]), dict(payload["hashes"]),
    )


def model_from_artifact(artifact: DeepONetArtifact, *, device: str | object = "cpu"):
    _require_torch()
    target = torch.device(device)
    model = PositivePremiumDeepONet(int(artifact.config["latent_rank"]))
    model.load_state_dict(artifact.state_dict)
    model.to(device=target, dtype=torch.float64)
    model.eval()
    return model


def infer_deeponet_numpy(
    artifact: DeepONetArtifact,
    scaled_features: np.ndarray,
    scaled_coordinates: np.ndarray,
) -> np.ndarray:
    """Dependency-light exported inference used to audit PyTorch serialization."""

    features = np.asarray(scaled_features, dtype=float)
    coordinates = np.asarray(scaled_coordinates, dtype=float)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("scaled_features must have shape (batch,4)")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("scaled_coordinates must have shape (points,2)")
    branch = _numpy_sequential(artifact.state_dict, "branch", features)
    trunk = _numpy_sequential(artifact.state_dict, "trunk", coordinates)
    bias = float(np.asarray(artifact.state_dict["bias"].detach().cpu()))
    return branch @ trunk.T / math.sqrt(int(artifact.config["latent_rank"])) + bias


def _numpy_sequential(state_dict, prefix, values):
    indices = sorted({
        int(key.split(".")[1]) for key in state_dict
        if key.startswith(f"{prefix}.") and key.endswith(".weight")
    })
    result = np.asarray(values, dtype=float)
    for position, index in enumerate(indices):
        weight = state_dict[f"{prefix}.{index}.weight"].detach().cpu().numpy()
        bias = state_dict[f"{prefix}.{index}.bias"].detach().cpu().numpy()
        # ``np.einsum`` avoids a macOS Accelerate/OpenBLAS warning observed for
        # these small C-contiguous float64 matrices while remaining the same
        # deterministic affine map.
        result = np.einsum("bi,oi->bo", result, weight, optimize=False) + bias
        if position < len(indices) - 1:
            clipped = np.clip(result, -60.0, 60.0)
            result = result / (1.0 + np.exp(-clipped))
    return result


def _bundle_tensors(bundle, device, dtype):
    regime_rows = np.asarray([
        [float(item["T"]), float(item["sigma"]), float(item["r"]), float(item["q"]),
         0.0 if item["option_type"] == "put" else 1.0]
        for item in bundle.regimes
    ])
    return {
        "features_scaled": torch.as_tensor(bundle.features_scaled, dtype=dtype, device=device),
        "coordinates": torch.as_tensor(bundle.coordinate_grid, dtype=dtype, device=device),
        "premium": torch.as_tensor(bundle.premium_surfaces, dtype=dtype, device=device),
        "boundary": torch.as_tensor(bundle.boundary_mask, dtype=torch.bool, device=device),
        "continuation": torch.as_tensor(bundle.continuation_mask, dtype=torch.bool, device=device),
        "payoff": torch.as_tensor(bundle.payoff, dtype=dtype, device=device),
        "class_weights": torch.as_tensor(bundle.class_weights, dtype=dtype, device=device),
        "regimes": torch.as_tensor(regime_rows, dtype=dtype, device=device),
    }


def _validate_training_config(bundle, config):
    if config.option_type != bundle.option_type:
        raise ValueError("training bundle and config option family differ")
    if config.arm not in {"N0", "N1", "N2"}:
        raise ValueError("arm must be N0, N1, or N2")
    if config.latent_rank not in {32, 64, 128}:
        raise ValueError("latent_rank must be 32, 64, or 128")
    if config.dtype != "float64":
        raise ValueError("formal DeepONet training is frozen to float64")
    if config.steps < 1 or config.batch_size < 1 or config.checkpoint_interval < 1:
        raise ValueError("steps, batch_size, and checkpoint_interval must be positive")
    if config.time_budget_seconds is not None and config.time_budget_seconds <= 0.0:
        raise ValueError("time budget must be positive")


def _resolve_device(name: str):
    requested = str(name).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _save_checkpoint(
    path, model, optimizer, scheduler, bundle, config, *, step, last_loss,
    training_seconds, rescued, generator, status="COMPLETE",
):
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "input_scaler_mean": bundle.input_scaler_mean,
        "input_scaler_scale": bundle.input_scaler_scale,
        "config": asdict(config),
        "hashes": dict(bundle.hashes),
        "step": int(step), "last_loss": float(last_loss),
        "training_seconds": float(training_seconds), "rescued": bool(rescued),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "status": status,
        "metadata": {
            "parameter_count": count_parameters(model),
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _assert_checkpoint_compatible(payload, bundle, config):
    if payload.get("status", "COMPLETE") != "COMPLETE":
        raise RuntimeError("a failed or budget-exhausted formal seed may not be resumed")
    if payload["hashes"] != bundle.hashes:
        raise RuntimeError("resume checkpoint data/protocol hash mismatch")
    frozen = dict(payload["config"])
    current = asdict(config)
    for key in current:
        if key != "time_budget_seconds" and frozen.get(key) != current[key]:
            raise RuntimeError(f"resume checkpoint config mismatch: {key}")


def _write_history(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except Exception:
        return "unavailable"
