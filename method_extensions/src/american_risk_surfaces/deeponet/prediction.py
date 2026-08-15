"""Continuous-coordinate prediction, exact controls, and Policy initialization."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from american_risk_surfaces.deeponet.model import model_from_artifact, torch
from american_risk_surfaces.deeponet.types import DeepONetArtifact, DeepONetPrediction
from american_risk_surfaces.method_extensions.premium_warmstart import GatedSurfaceInitializer
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import (
    call_payoff,
    european_call_price,
    put_payoff,
)
from american_risk_surfaces.solvers.operator import american_call_boundaries, american_put_boundaries


def predict_deeponet_surface(
    artifact: DeepONetArtifact,
    option_config: AmericanLCPConfig,
    spot_grid: np.ndarray | None = None,
    tau_grid: np.ndarray | None = None,
    *,
    device: str = "cpu",
    compute_ad_greeks: bool = False,
    _allow_q0_neural_ood: bool = False,
    _prepared_model=None,
) -> DeepONetPrediction:
    if (
        option_config.option_type == "call" and option_config.q == 0.0
        and not _allow_q0_neural_ood
    ):
        return predict_q0_call_analytic_control(option_config, spot_grid, tau_grid)
    if artifact.config["option_type"] != option_config.option_type:
        raise ValueError("DeepONet artifact and option family differ")
    if option_config.K != 1.0 or option_config.Smax != 4.0:
        raise ValueError("DeepONet is frozen to K=1 and Smax=4")
    spots = _spot_grid(option_config, spot_grid)
    taus = _tau_grid(option_config, tau_grid)
    started = perf_counter()
    features = np.asarray([
        math.log(option_config.T), option_config.sigma, option_config.r, option_config.q,
    ])
    scaled = (features - artifact.input_scaler_mean) / artifact.input_scaler_scale
    feature_seconds = perf_counter() - started
    coordinate_started = perf_counter()
    coordinates = _coordinates(spots[1:-1], taus[1:], option_config)
    coordinate_seconds = perf_counter() - coordinate_started
    target_device = torch.device(device)
    materialization_started = perf_counter()
    if _prepared_model is None:
        model = model_from_artifact(artifact, device=target_device)
    else:
        model = _prepared_model
        expected_rank = int(artifact.config["latent_rank"])
        if getattr(model, "latent_rank", None) != expected_rank:
            raise ValueError("prepared model rank does not match the artifact")
    model_device = next(model.parameters()).device
    if model_device != target_device:
        raise ValueError("prepared DeepONet model is on the wrong device")
    _synchronize(target_device)
    materialization_seconds = (
        perf_counter() - materialization_started if _prepared_model is None else 0.0
    )
    branch_started = perf_counter()
    branch = model.encode_branch(
        torch.as_tensor(scaled[None, :], dtype=torch.float64, device=target_device)
    )
    _synchronize(target_device)
    branch_seconds = perf_counter() - branch_started
    trunk_input = torch.as_tensor(coordinates, dtype=torch.float64, device=target_device)
    if compute_ad_greeks:
        trunk_input.requires_grad_(True)
    trunk_started = perf_counter()
    trunk = model.encode_trunk(trunk_input)
    _synchronize(target_device)
    trunk_seconds = perf_counter() - trunk_started
    contract_started = perf_counter()
    raw_tensor = model.contract(branch, trunk).reshape(len(taus) - 1, len(spots) - 2)
    _synchronize(target_device)
    contraction_seconds = perf_counter() - contract_started
    projection_started = perf_counter()
    projected_tensor = torch.relu(raw_tensor)
    _synchronize(target_device)
    projection_seconds = perf_counter() - projection_started
    transfer_started = perf_counter()
    raw_interior = raw_tensor.detach().cpu().numpy()
    projected_interior = projected_tensor.detach().cpu().numpy()
    _synchronize(target_device)
    transfer_seconds = perf_counter() - transfer_started
    boundary_started = perf_counter()
    raw, projected, value = _reconstruct_full(
        raw_interior, projected_interior, option_config, spots, taus
    )
    boundary_seconds = perf_counter() - boundary_started
    timing = {
        "feature_scaling_seconds": float(feature_seconds),
        "coordinate_generation_seconds": float(coordinate_seconds),
        "model_materialization_seconds": float(materialization_seconds),
        "branch_seconds": float(branch_seconds),
        "trunk_seconds": float(trunk_seconds),
        "cartesian_contraction_seconds": float(contraction_seconds),
        "hard_projection_seconds": float(projection_seconds),
        "gpu_to_cpu_seconds": float(transfer_seconds),
        "projection_boundary_seconds": float(boundary_seconds),
    }
    timing["prediction_seconds"] = float(sum(timing.values()))
    ad_delta = None
    ad_gamma = None
    if compute_ad_greeks:
        ad_started = perf_counter()
        first = torch.autograd.grad(
            projected_tensor.sum(), trunk_input, create_graph=True, retain_graph=True
        )[0][:, 0]
        second = torch.autograd.grad(first.sum(), trunk_input, retain_graph=False)[0][:, 0]
        _synchronize(target_device)
        timing["ad_greek_seconds"] = float(perf_counter() - ad_started)
        timing["ad_delta_finite_fraction"] = float(torch.isfinite(first).double().mean().detach().cpu())
        timing["ad_gamma_finite_fraction"] = float(torch.isfinite(second).double().mean().detach().cpu())
        payoff_delta = np.zeros(len(spots) - 2, dtype=float)
        if option_config.option_type == "put":
            payoff_delta[spots[1:-1] < option_config.K] = -1.0
        else:
            payoff_delta[spots[1:-1] > option_config.K] = 1.0
        delta_interior = (
            0.5 * first.detach().cpu().numpy().reshape(len(taus) - 1, len(spots) - 2)
            + payoff_delta[None, :]
        )
        gamma_interior = (
            0.25 * second.detach().cpu().numpy().reshape(len(taus) - 1, len(spots) - 2)
            / option_config.K
        )
        ad_delta = np.full((len(taus), len(spots)), np.nan)
        ad_gamma = np.full_like(ad_delta, np.nan)
        ad_delta[1:, 1:-1] = delta_interior
        ad_gamma[1:, 1:-1] = gamma_interior
        timing["prediction_seconds"] += timing["ad_greek_seconds"]
    return DeepONetPrediction(
        raw, projected, value, timing, "positive_premium_deeponet", ad_delta, ad_gamma
    )


def predict_q0_call_analytic_control(
    config: AmericanLCPConfig,
    spot_grid: np.ndarray | None = None,
    tau_grid: np.ndarray | None = None,
) -> DeepONetPrediction:
    if config.option_type != "call" or config.q != 0.0:
        raise ValueError("analytic control is only for no-dividend calls")
    spots = _spot_grid(config, spot_grid)
    taus = _tau_grid(config, tau_grid)
    started = perf_counter()
    payoff = np.asarray(call_payoff(spots, config.K), dtype=float)
    value = np.vstack([
        european_call_price(spots, config.K, float(tau), config.r, 0.0, config.sigma)
        for tau in taus
    ])
    premium = np.maximum((value - payoff[np.newaxis, :]) / config.K, 0.0)
    elapsed = perf_counter() - started
    return DeepONetPrediction(
        premium.copy(), premium, value,
        {"analytic_surface_seconds": elapsed, "prediction_seconds": elapsed},
        "EUROPEAN_BSM_ANALYTIC_Q0_CALL",
    )


def make_deeponet_policy_initializer(prediction) -> GatedSurfaceInitializer:
    values = prediction.value_grid if isinstance(prediction, DeepONetPrediction) else prediction
    array = np.asarray(values, dtype=float)
    if array.shape != (121, 121):
        raise ValueError("strict hybrid initializer requires the frozen 121x121 grid")
    spots = np.linspace(0.0, 4.0, 121)[1:-1]
    return GatedSurfaceInitializer(
        array[:, 1:-1], spots, 1.0, support=(0.0, 4.0), raw_extrapolation=True
    )


def _coordinates(spots, taus, config):
    m_scaled = np.asarray(spots, dtype=float) / (2.0 * config.K) - 1.0
    s_scaled = 2.0 * np.asarray(taus, dtype=float) / config.T - 1.0
    ss, mm = np.meshgrid(s_scaled, m_scaled, indexing="ij")
    return np.column_stack((mm.reshape(-1), ss.reshape(-1)))


def _reconstruct_full(raw_interior, projected_interior, config, spots, taus):
    payoff_fn = put_payoff if config.option_type == "put" else call_payoff
    payoff = np.asarray(payoff_fn(spots, config.K), dtype=float)
    raw = np.zeros((len(taus), len(spots)), dtype=float)
    projected = np.zeros_like(raw)
    raw[1:, 1:-1] = raw_interior
    projected[1:, 1:-1] = projected_interior
    for index, tau in enumerate(taus):
        if config.option_type == "put":
            left, right = american_put_boundaries(config.K, float(tau))
        else:
            left, right = american_call_boundaries(
                config.Smax, config.K, float(tau), config.r, config.q
            )
        raw[index, 0] = (left - payoff[0]) / config.K
        raw[index, -1] = (right - payoff[-1]) / config.K
        projected[index, 0] = max(raw[index, 0], 0.0)
        projected[index, -1] = max(raw[index, -1], 0.0)
    raw[0] = 0.0
    projected[0] = 0.0
    return raw, projected, payoff[np.newaxis, :] + config.K * projected


def _spot_grid(config, supplied):
    spots = np.linspace(0.0, config.Smax, config.M + 1) if supplied is None else np.asarray(supplied, dtype=float)
    if spots.ndim != 1 or len(spots) < 3 or not np.isclose(spots[0], 0.0) or not np.isclose(spots[-1], config.Smax) or np.any(np.diff(spots) <= 0.0):
        raise ValueError("spot_grid must be strictly increasing from 0 to Smax")
    return spots


def _tau_grid(config, supplied):
    taus = np.linspace(0.0, config.T, config.N + 1) if supplied is None else np.asarray(supplied, dtype=float)
    if taus.ndim != 1 or len(taus) < 2 or not np.isclose(taus[0], 0.0) or not np.isclose(taus[-1], config.T) or np.any(np.diff(taus) <= 0.0):
        raise ValueError("tau_grid must be strictly increasing from 0 to T")
    return taus


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
