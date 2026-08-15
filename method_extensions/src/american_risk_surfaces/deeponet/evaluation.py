"""Common DeepONet price, boundary, Greek, exercise-set, and LCP audits."""

from __future__ import annotations

import numpy as np

from american_risk_surfaces.basis_operator.evaluation import audit_basis_operator_surface
from american_risk_surfaces.deeponet.protocol import FB_EPSILON
from american_risk_surfaces.deeponet.types import DeepONetPrediction
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.lcp import tridiagonal_matvec


def audit_deeponet_surface(
    prediction: DeepONetPrediction,
    option_config: AmericanLCPConfig,
    reference: np.ndarray | None = None,
    *,
    prefix: str = "reduction",
) -> dict[str, float]:
    metrics = audit_basis_operator_surface(
        prediction, option_config, reference_value_grid=reference, prefix=prefix
    )
    fb_values = []
    for step in range(1, option_config.N + 1):
        system = assemble_american_cn_lcp_step(
            option_config, prediction.value_grid[step - 1], step
        )
        values = prediction.value_grid[step, 1:-1]
        matrix_action = tridiagonal_matvec(system, values)
        gap = values - system.obstacle
        equation = matrix_action - system.rhs
        value_scale = max(
            1.0, float(np.max(np.abs(values))), float(np.max(np.abs(system.obstacle)))
        )
        equation_scale = max(
            1.0, float(np.max(np.abs(matrix_action))), float(np.max(np.abs(system.rhs)))
        )
        a, b = gap / value_scale, equation / equation_scale
        fb = a + b - np.sqrt(a * a + b * b + FB_EPSILON**2) + FB_EPSILON
        fb_values.append(fb)
    flattened = np.concatenate(fb_values) if fb_values else np.zeros(1)
    metrics["normalized_fb_rmse"] = float(np.sqrt(np.mean(flattened**2)))
    metrics["normalized_fb_max_abs"] = float(np.max(np.abs(flattened)))
    if reference is not None and prediction.ad_delta_grid is not None:
        truth = np.asarray(reference, dtype=float)
        spots = np.linspace(0.0, option_config.Smax, option_config.M + 1)
        taus = np.linspace(0.0, option_config.T, option_config.N + 1)
        ref_delta = np.gradient(truth, spots, axis=1, edge_order=2)
        ref_gamma = np.gradient(ref_delta, spots, axis=1, edge_order=2)
        delta_mask, gamma_mask = _approved_ad_masks(
            truth, spots, taus, option_config
        )
        delta_error = prediction.ad_delta_grid - ref_delta
        gamma_error = prediction.ad_gamma_grid - ref_gamma
        metrics[f"{prefix}_ad_delta_rmse"] = float(
            np.sqrt(np.mean(delta_error[delta_mask] ** 2))
        )
        metrics[f"{prefix}_ad_stable_gamma_rmse"] = float(
            np.sqrt(np.mean(gamma_error[gamma_mask] ** 2))
        )
    return metrics


def _approved_ad_masks(reference, spots, taus, config):
    payoff = (
        np.maximum(config.K - spots, 0.0)
        if config.option_type == "put"
        else np.maximum(spots - config.K, 0.0)
    )
    premium = reference - payoff[None, :]
    boundary = np.zeros_like(reference, dtype=bool)
    state = premium <= 1e-6
    transitions = state[:, 1:] != state[:, :-1]
    boundary[:, 1:] |= transitions
    boundary[:, :-1] |= transitions
    spacing = float(spots[1] - spots[0])
    strike = np.abs(spots - config.K) <= 2.0 * spacing
    maturity = taus <= 0.1 * config.T
    finite = np.isfinite(reference)
    mask = finite & ~boundary & ~strike[None, :] & ~maturity[:, None]
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask, mask.copy()
