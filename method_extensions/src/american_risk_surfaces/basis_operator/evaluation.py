"""Common accuracy, exercise-set, Greek, and LCP audits."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from american_risk_surfaces.basis_operator.protocol import PREMIUM_THRESHOLD
from american_risk_surfaces.basis_operator.types import BasisOperatorPrediction
from american_risk_surfaces.reduced_order.metrics import score_value_trajectory
from american_risk_surfaces.reduced_order.snapshots import trajectory_multipliers
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig


def audit_basis_operator_surface(
    prediction: BasisOperatorPrediction,
    option_config: AmericanLCPConfig,
    *,
    reference_value_grid: np.ndarray | None = None,
    prefix: str = "reduction",
) -> dict[str, float]:
    started = perf_counter()
    spots = np.linspace(0.0, option_config.Smax, option_config.M + 1)
    taus = np.linspace(0.0, option_config.T, option_config.N + 1)
    if option_config.option_type == "put":
        payoff = np.maximum(option_config.K - spots, 0.0)
    else:
        payoff = np.maximum(spots - option_config.K, 0.0)
    multipliers, active, residual_rows = trajectory_multipliers(
        option_config, prediction.value_grid
    )
    metrics = {
        "normalized_obstacle_violation_max": float(np.max(residual_rows[:, 0])),
        "normalized_equation_violation_max": float(np.max(residual_rows[:, 1])),
        "normalized_complementarity_max": float(np.max(residual_rows[:, 2])),
        "normalized_full_lcp_residual_max": float(np.max(residual_rows[:, 3])),
        "normalized_full_lcp_residual_p95": float(np.quantile(residual_rows[1:, 3], 0.95)),
        "raw_negative_premium_max": float(np.max(np.maximum(-prediction.raw_premium_grid, 0.0))),
        "projected_obstacle_violation": float(
            np.max(np.maximum(payoff[np.newaxis, :] - prediction.value_grid, 0.0))
        ),
        "monotonicity_violation_rate": monotonicity_violation_rate(
            prediction.value_grid, option_config.option_type
        ),
        "convexity_violation_rate": float(np.mean(np.diff(prediction.value_grid, n=2, axis=1) < -1e-10)),
        "audit_seconds": perf_counter() - started,
    }
    if reference_value_grid is not None:
        reference = np.asarray(reference_value_grid, dtype=float)
        scored = score_value_trajectory(
            prediction.value_grid, reference, payoff, spots, taus,
            option_config.option_type,
        )
        metrics.update({f"{prefix}_{key}": value for key, value in scored.items()})
        predicted_exercise = prediction.projected_premium_grid <= PREMIUM_THRESHOLD
        reference_exercise = (reference - payoff[np.newaxis, :]) / option_config.K <= PREMIUM_THRESHOLD
        metrics.update(exercise_set_metrics(predicted_exercise, reference_exercise, prefix=prefix))
    return metrics


def exercise_set_metrics(
    prediction: np.ndarray, reference: np.ndarray, *, prefix: str
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(reference, dtype=bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    return {
        f"{prefix}_exercise_precision": float(precision),
        f"{prefix}_exercise_recall": float(recall),
        f"{prefix}_exercise_f1": float(f1),
    }


def monotonicity_violation_rate(values: np.ndarray, option_type: str) -> float:
    derivative = np.diff(np.asarray(values, dtype=float), axis=1)
    violations = derivative > 1e-10 if option_type == "put" else derivative < -1e-10
    return float(np.mean(violations))
