"""Prediction, continuous VI diagnostics, Greeks, and Arm E hybrid utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.pinn.formulation import (
    PINNProblem,
    fischer_burmeister_numpy,
    spatial_boundary_numpy,
    trial_value,
    value_and_vi_residual,
)
from american_risk_surfaces.pinn.networks import NetworkSpec, build_network
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    AmericanLCPResult,
    assemble_american_cn_lcp_step,
    american_cn_lcp_price,
)
from american_risk_surfaces.solvers.lcp import compute_lcp_residual


@dataclass(frozen=True)
class LoadedPINN:
    model: Any
    problem: PINNProblem
    arm: str
    representation: str
    checkpoint_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PINNSurfacePrediction:
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    value_grid: np.ndarray
    inference_seconds: float
    transfer_seconds: float


@dataclass(frozen=True)
class HybridTiming:
    inference_seconds: float
    transfer_seconds: float
    projection_seconds: float
    lcp_finish_seconds: float
    online_total_seconds: float


class PINNSurfaceInitializer:
    """Projected neural time slices supplied to the shared LCP marcher."""

    name = "pinn_surface_projected"

    def __init__(self, value_grid: np.ndarray) -> None:
        values = np.asarray(value_grid, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
            raise ValueError("value_grid must be a valid time-by-space surface.")
        if not np.all(np.isfinite(values)):
            raise ValueError("value_grid must contain finite values.")
        self.value_grid = values.copy()

    def __call__(
        self,
        step_index: int,
        _tau: float,
        _previous_values: np.ndarray,
        obstacle: np.ndarray,
    ) -> np.ndarray:
        if not 0 <= step_index < self.value_grid.shape[0]:
            raise ValueError("step_index is outside the prediction surface.")
        predicted = self.value_grid[step_index, 1:-1]
        if predicted.shape != obstacle.shape:
            raise ValueError("prediction and obstacle shapes do not match.")
        return np.maximum(predicted, obstacle)


def load_pinn_checkpoint(
    path: Path | str,
    *,
    device: str = "cpu",
) -> LoadedPINN:
    torch = _torch()
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("protocol") != "surf_pinn_cde_v1":
        raise ValueError("checkpoint does not use the SURF PINN C/D/E protocol.")
    spec = NetworkSpec(**payload["network_spec"])
    model = build_network(spec).to(device=torch.device(device), dtype=torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    problem = PINNProblem(**payload["problem"])
    training_config = payload["training_config"]
    arm = str(training_config["arm"])
    representation = (
        "positive_premium"
        if training_config.get("d_variant") == "positive_premium"
        else "smooth_etc"
    )
    return LoadedPINN(
        model=model,
        problem=problem,
        arm=arm,
        representation=representation,
        checkpoint_path=checkpoint_path,
        metadata=payload,
    )


def predict_pinn_surface(
    checkpoint: LoadedPINN | Path | str,
    spot_grid: Any,
    tau_grid: Any,
    *,
    K: float,
    device: str = "cpu",
    batch_size: int = 65536,
) -> PINNSurfacePrediction:
    torch = _torch()
    loaded = checkpoint if isinstance(checkpoint, LoadedPINN) else load_pinn_checkpoint(checkpoint, device=device)
    spots = np.asarray(spot_grid, dtype=float)
    taus = np.asarray(tau_grid, dtype=float)
    if K <= 0.0 or spots.ndim != 1 or taus.ndim != 1:
        raise ValueError("K and one-dimensional grids are required.")
    if len(spots) < 3 or len(taus) < 1 or np.any(spots < 0.0):
        raise ValueError("invalid spot or time grid.")
    moneyness = spots[1:-1] / K
    if np.any(moneyness < loaded.problem.m_min) or np.any(moneyness > loaded.problem.m_max):
        raise ValueError("interior spot grid lies outside the frozen PINN domain.")
    normalized_time = taus / loaded.problem.T
    if np.any(normalized_time < -1e-14) or np.any(normalized_time > 1.0 + 1e-14):
        raise ValueError("tau grid lies outside [0, T].")
    x = np.log(moneyness)
    coordinates = np.column_stack(
        (np.tile(x, len(taus)), np.repeat(normalized_time, len(x)))
    )
    device_object = next(loaded.model.parameters()).device
    started = perf_counter()
    prediction_device = torch.empty(
        len(coordinates), dtype=torch.float64, device=device_object
    )
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            stop = min(start + batch_size, len(coordinates))
            tensor = torch.as_tensor(
                coordinates[start:stop], dtype=torch.float64, device=device_object
            )
            value = trial_value(
                loaded.model,
                loaded.problem,
                tensor[:, 0:1],
                tensor[:, 1:2],
                arm=loaded.arm,
                representation=loaded.representation,
            )
            prediction_device[start:stop] = value.detach().reshape(-1)
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_elapsed = perf_counter() - started
    transfer_started = perf_counter()
    prediction = prediction_device.cpu().numpy()
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    transfer_elapsed = perf_counter() - transfer_started
    value_grid = np.empty((len(taus), len(spots)), dtype=float)
    value_grid[:, 1:-1] = K * prediction.reshape(len(taus), len(x))
    lower, upper = spatial_boundary_numpy(loaded.problem, normalized_time)
    value_grid[:, 0] = K * lower
    value_grid[:, -1] = K * upper
    if spots[0] == 0.0:
        value_grid[:, 0] = K if loaded.problem.option_type == "put" else 0.0
    return PINNSurfacePrediction(
        spot_grid=spots.copy(),
        tau_grid=taus.copy(),
        value_grid=value_grid,
        inference_seconds=float(inference_elapsed),
        transfer_seconds=float(transfer_elapsed),
    )


def evaluate_pinn_vi(
    checkpoint: LoadedPINN | Path | str,
    evaluation_points: Any,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    torch = _torch()
    loaded = checkpoint if isinstance(checkpoint, LoadedPINN) else load_pinn_checkpoint(checkpoint, device=device)
    points = np.asarray(evaluation_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("evaluation_points must be a finite (n, 2) array of (x, s).")
    tensor = torch.as_tensor(
        points,
        dtype=torch.float64,
        device=next(loaded.model.parameters()).device,
    ).requires_grad_(True)
    residual = value_and_vi_residual(
        loaded.model,
        loaded.problem,
        tensor,
        arm=loaded.arm,
        representation=loaded.representation,
        create_graph=False,
    )
    arrays = {
        name: value.detach().cpu().numpy().reshape(-1)
        for name, value in residual.items()
        if name != "coordinates"
    }
    arrays["delta"] = arrays["value_x"] / np.exp(points[:, 0])
    arrays["scaled_gamma"] = (
        arrays["value_xx"] - arrays["value_x"]
    ) / np.exp(2.0 * points[:, 0])
    arrays["obstacle_violation"] = np.maximum(-arrays["obstacle_gap"], 0.0)
    arrays["equation_violation"] = np.maximum(-arrays["equation_gap"], 0.0)
    arrays["fb"] = fischer_burmeister_numpy(
        arrays["normalized_gap"], arrays["normalized_equation"]
    )
    return arrays


def make_pinn_policy_initializer(predicted_surface: Any) -> PINNSurfaceInitializer:
    if isinstance(predicted_surface, PINNSurfacePrediction):
        values = predicted_surface.value_grid
    else:
        values = np.asarray(predicted_surface, dtype=float)
    return PINNSurfaceInitializer(values)


def run_arm_e_hybrid(
    checkpoint: LoadedPINN | Path | str,
    config: AmericanLCPConfig,
    *,
    device: str = "cpu",
) -> tuple[AmericanLCPResult, HybridTiming]:
    loaded = checkpoint if isinstance(checkpoint, LoadedPINN) else load_pinn_checkpoint(checkpoint, device=device)
    spot_grid = np.linspace(0.0, config.Smax, config.M + 1)
    tau_grid = np.linspace(0.0, config.T, config.N + 1)
    started = perf_counter()
    prediction = predict_pinn_surface(
        loaded, spot_grid, tau_grid, K=config.K, device=device
    )
    transferred = np.asarray(prediction.value_grid, dtype=float).copy()
    after_transfer = perf_counter()
    payoff = np.maximum(
        config.K - spot_grid if config.option_type == "put" else spot_grid - config.K,
        0.0,
    )
    projected = np.maximum(transferred, payoff[np.newaxis, :])
    initializer = make_pinn_policy_initializer(projected)
    after_projection = perf_counter()
    result = american_cn_lcp_price(
        config,
        lcp_solver="policy_iteration",
        initializer=initializer,
    )
    finished = perf_counter()
    timing = HybridTiming(
        inference_seconds=prediction.inference_seconds,
        transfer_seconds=prediction.transfer_seconds,
        projection_seconds=after_projection - after_transfer,
        lcp_finish_seconds=result.lcp_finish_seconds,
        online_total_seconds=finished - started,
    )
    return result, timing


def discrete_lcp_audit(
    prediction: PINNSurfacePrediction,
    config: AmericanLCPConfig,
) -> dict[str, float]:
    if prediction.value_grid.shape != (config.N + 1, config.M + 1):
        raise ValueError("prediction grid does not match the classical LCP config.")
    residuals = []
    for step in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(
            config, prediction.value_grid[step - 1], step
        )
        residuals.append(
            compute_lcp_residual(system, prediction.value_grid[step, 1:-1])
        )
    return {
        "max_normalized_lcp_residual": max(item.normalized_lcp_residual for item in residuals),
        "max_normalized_obstacle_violation": max(
            item.normalized_obstacle_violation for item in residuals
        ),
        "max_normalized_equation_violation": max(
            item.normalized_equation_violation for item in residuals
        ),
        "max_normalized_complementarity": max(
            item.normalized_complementarity for item in residuals
        ),
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for PINN evaluation.") from exc
    return torch
