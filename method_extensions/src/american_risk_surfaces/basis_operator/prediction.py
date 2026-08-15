"""Surface reconstruction, exact controls, and hybrid initialization."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from american_risk_surfaces.basis_operator.basis import reconstruct_premium_vector
from american_risk_surfaces.basis_operator.model import infer_coefficients
from american_risk_surfaces.basis_operator.types import (
    BasisOperatorArtifact,
    BasisOperatorPrediction,
    PremiumPODBasis,
)
from american_risk_surfaces.method_extensions.premium_warmstart import GatedSurfaceInitializer
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import (
    call_payoff,
    european_call_price,
    put_payoff,
)
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.operator import american_call_boundaries, american_put_boundaries


def reconstruct_full_prediction(
    basis: PremiumPODBasis,
    coefficients: np.ndarray,
    config: AmericanLCPConfig,
    *,
    projection: str = "hard",
    coefficient_seconds: float = 0.0,
) -> BasisOperatorPrediction:
    if config.M != 120 or config.N != 120 or config.K != 1.0 or config.Smax != 4.0:
        raise ValueError("basis operator is frozen to K=1,Smax=4,M=N=120")
    if config.option_type != basis.option_type:
        raise ValueError("basis family and option config differ")
    reconstruction_started = perf_counter()
    raw_interior = reconstruct_premium_vector(basis, coefficients).reshape(120, 119)
    reconstruction_seconds = perf_counter() - reconstruction_started
    projection_started = perf_counter()
    if projection == "hard":
        projected_interior = np.maximum(raw_interior, 0.0)
    elif projection == "softplus":
        beta = 1e-4
        projected_interior = np.maximum(
            beta * np.logaddexp(0.0, raw_interior / beta),
            np.finfo(float).tiny,
        )
    elif projection == "raw":
        projected_interior = raw_interior.copy()
    else:
        raise ValueError("projection must be hard, softplus, or raw")
    projection_seconds = perf_counter() - projection_started
    boundary_started = perf_counter()
    spots = uniform_spot_grid(config.Smax, config.M)[0]
    taus = uniform_tau_grid(config.T, config.N)[0]
    payoff_fn = put_payoff if config.option_type == "put" else call_payoff
    payoff = np.asarray(payoff_fn(spots, config.K), dtype=float)
    raw = np.zeros((121, 121), dtype=float)
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
    value = payoff[np.newaxis, :] + config.K * projected
    boundary_seconds = perf_counter() - boundary_started
    return BasisOperatorPrediction(
        raw, projected, value,
        {
            "coefficient_inference_seconds": float(coefficient_seconds),
            "basis_reconstruction_seconds": float(reconstruction_seconds),
            "projection_seconds": float(projection_seconds),
            "boundary_terminal_seconds": float(boundary_seconds),
            "prediction_seconds": float(coefficient_seconds + reconstruction_seconds + projection_seconds + boundary_seconds),
        },
        "basis_operator",
    )


def predict_basis_operator_surface(
    artifact: BasisOperatorArtifact,
    option_config: AmericanLCPConfig,
    *,
    projection: str = "hard",
) -> BasisOperatorPrediction:
    if option_config.option_type == "call" and option_config.q == 0.0:
        return predict_no_dividend_call_control(option_config)
    scaling_started = perf_counter()
    features = np.asarray([[math.log(option_config.T), option_config.sigma, option_config.r, option_config.q]])
    feature_seconds = perf_counter() - scaling_started
    inference_started = perf_counter()
    coefficients = infer_coefficients(artifact, features)[0]
    inference_seconds = perf_counter() - inference_started
    result = reconstruct_full_prediction(
        artifact.basis, coefficients, option_config, projection=projection,
        coefficient_seconds=feature_seconds + inference_seconds,
    )
    result.timing["feature_scaling_seconds"] = feature_seconds
    result.timing["mlp_inference_seconds"] = inference_seconds
    return result


def predict_no_dividend_call_control(config: AmericanLCPConfig) -> BasisOperatorPrediction:
    if config.option_type != "call" or config.q != 0.0:
        raise ValueError("analytic control is only for no-dividend calls")
    started = perf_counter()
    spots = uniform_spot_grid(config.Smax, config.M)[0]
    taus = uniform_tau_grid(config.T, config.N)[0]
    payoff = np.asarray(call_payoff(spots, config.K), dtype=float)
    value = np.vstack([
        european_call_price(spots, config.K, float(tau), config.r, 0.0, config.sigma)
        for tau in taus
    ])
    premium = np.maximum((value - payoff[np.newaxis, :]) / config.K, 0.0)
    elapsed = perf_counter() - started
    return BasisOperatorPrediction(
        premium.copy(), premium, value,
        {"analytic_surface_seconds": elapsed, "prediction_seconds": elapsed},
        "EUROPEAN_BSM_ANALYTIC_Q0_CALL",
    )


def make_basis_operator_policy_initializer(predicted_surface: np.ndarray) -> GatedSurfaceInitializer:
    values = np.asarray(predicted_surface, dtype=float)
    if values.shape != (121, 121):
        raise ValueError("predicted surface must be the frozen 121x121 grid")
    spots = np.linspace(0.0, 4.0, 121)[1:-1]
    return GatedSurfaceInitializer(values[:, 1:-1], spots, 1.0, support=(0.0, 4.0), raw_extrapolation=True)
