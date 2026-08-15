"""Per-regime accuracy, boundary, active-set, and Greek metrics for RB-VI."""

from __future__ import annotations

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time


def score_value_trajectory(
    prediction: np.ndarray,
    reference: np.ndarray,
    payoff: np.ndarray,
    spot_grid: np.ndarray,
    tau_grid: np.ndarray,
    option_type: str,
    *,
    predicted_multiplier: np.ndarray | None = None,
    reference_multiplier: np.ndarray | None = None,
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)
    truth = np.asarray(reference, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    spots = np.asarray(spot_grid, dtype=float)
    taus = np.asarray(tau_grid, dtype=float)
    if predicted.shape != truth.shape or predicted.shape != (len(taus), len(spots)):
        raise ValueError("prediction and reference must match the full grid")
    error = predicted - truth
    metrics = _error_metrics(error, prefix="price")
    metrics["price_relative_l2"] = float(
        np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-15)
    )
    premium = truth - payoff[np.newaxis, :]
    spacing = float(spots[1] - spots[0])
    boundary = _boundary_series(truth, payoff, spots, taus, option_type)
    boundary_mask = np.zeros_like(truth, dtype=bool)
    for time_index, location in enumerate(boundary):
        if np.isfinite(location):
            boundary_mask[time_index] = np.abs(spots - location) <= spacing
    region_masks = {
        "exercise": premium <= 1e-6,
        "continuation": premium > 1e-6,
        "boundary_near": boundary_mask,
        "strike_near": np.broadcast_to(np.abs(spots - 1.0) <= 2.0 * spacing, truth.shape),
        "maturity_near": np.broadcast_to(taus[:, None] <= 0.1 * max(taus[-1], 1e-15), truth.shape),
    }
    region_masks["strict_interior"] = ~(
        region_masks["boundary_near"]
        | region_masks["strike_near"]
        | region_masks["maturity_near"]
    )
    for region, mask in region_masks.items():
        selected = error[mask]
        if selected.size:
            metrics.update(_error_metrics(selected, prefix=f"{region}_price"))
    predicted_boundary = _boundary_series(predicted, payoff, spots, taus, option_type)
    found_truth = np.isfinite(boundary)
    found_predicted = np.isfinite(predicted_boundary)
    both = found_truth & found_predicted
    metrics.update(
        {
            "boundary_found_rate": float(np.mean(found_predicted[1:])),
            "boundary_missed_count": float(np.sum(found_truth & ~found_predicted)),
            "boundary_false_count": float(np.sum(~found_truth & found_predicted)),
            "boundary_conditional_mae": (
                float(np.mean(np.abs(predicted_boundary[both] - boundary[both])))
                if np.any(both)
                else float("nan")
            ),
            "boundary_conditional_rmse": (
                float(np.sqrt(np.mean((predicted_boundary[both] - boundary[both]) ** 2)))
                if np.any(both)
                else float("nan")
            ),
            "boundary_shape_error": _boundary_shape_error(predicted_boundary, boundary),
        }
    )
    delta_prediction = np.gradient(predicted, spots, axis=1, edge_order=2)
    delta_reference = np.gradient(truth, spots, axis=1, edge_order=2)
    gamma_prediction = np.gradient(delta_prediction, spots, axis=1, edge_order=2)
    gamma_reference = np.gradient(delta_reference, spots, axis=1, edge_order=2)
    stable_gamma = region_masks["strict_interior"].copy()
    stable_gamma[:, :2] = False
    stable_gamma[:, -2:] = False
    metrics["delta_rmse"] = float(np.sqrt(np.mean((delta_prediction - delta_reference) ** 2)))
    metrics["stable_gamma_rmse"] = float(
        np.sqrt(np.mean((gamma_prediction[stable_gamma] - gamma_reference[stable_gamma]) ** 2))
    )
    if predicted_multiplier is not None and reference_multiplier is not None:
        predicted_lambda = np.asarray(predicted_multiplier, dtype=float)
        reference_lambda = np.asarray(reference_multiplier, dtype=float)
        if predicted_lambda.shape != reference_lambda.shape:
            raise ValueError("multiplier shapes differ")
        metrics["multiplier_rmse"] = float(
            np.sqrt(np.mean((predicted_lambda - reference_lambda) ** 2))
        )
        predicted_active = predicted_lambda > 1e-10
        reference_active = reference_lambda > 1e-10
        true_positive = int(np.sum(predicted_active & reference_active))
        false_positive = int(np.sum(predicted_active & ~reference_active))
        false_negative = int(np.sum(~predicted_active & reference_active))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        metrics["active_set_precision"] = float(precision)
        metrics["active_set_recall"] = float(recall)
        metrics["active_set_f1"] = float(2.0 * precision * recall / max(precision + recall, 1e-15))
    return metrics


def interpolate_reference_surface(
    source_values: np.ndarray,
    source_spots: np.ndarray,
    source_taus: np.ndarray,
    target_spots: np.ndarray,
    target_taus: np.ndarray,
) -> np.ndarray:
    """Deterministic tensor interpolation from a fine nonuniform reference grid."""

    spatial = np.vstack(
        [np.interp(target_spots, source_spots, row) for row in np.asarray(source_values)]
    )
    return np.column_stack(
        [np.interp(target_taus, source_taus, spatial[:, column]) for column in range(len(target_spots))]
    )


def _boundary_series(
    values: np.ndarray,
    payoff: np.ndarray,
    spots: np.ndarray,
    taus: np.ndarray,
    option_type: str,
) -> np.ndarray:
    premium = values - payoff[np.newaxis, :]
    result = np.full(len(taus), np.nan)
    for index, tau in enumerate(taus):
        point = extract_boundary_at_time(spots, premium[index], option_type, float(tau), index)
        if point.boundary_found:
            result[index] = point.boundary_spot
    return result


def _boundary_shape_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    common = np.isfinite(prediction) & np.isfinite(reference)
    indices = np.flatnonzero(common)
    if len(indices) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(prediction[indices] - reference[indices]) ** 2)))


def _error_metrics(error: np.ndarray, *, prefix: str) -> dict[str, float]:
    flat = np.asarray(error, dtype=float).reshape(-1)
    return {
        f"{prefix}_mae": float(np.mean(np.abs(flat))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(flat**2))),
        f"{prefix}_max_abs_error": float(np.max(np.abs(flat))),
    }
