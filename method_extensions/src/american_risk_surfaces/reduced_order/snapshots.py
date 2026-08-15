"""Full-grid Policy-Iteration snapshots, multipliers, and boundary lifts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from american_risk_surfaces.reduced_order.protocol import RBRegime, protocol_hash
from american_risk_surfaces.reduced_order.types import RBFOMSnapshot
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.lcp import compute_lcp_residual, tridiagonal_matvec
from american_risk_surfaces.solvers.operator import american_call_boundaries, american_put_boundaries


def boundary_lift_grid(config: AmericanLCPConfig, spot_grid: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    spots = np.asarray(spot_grid, dtype=float)
    taus = np.asarray(tau_grid, dtype=float)
    fraction = spots / config.Smax
    result = np.empty((len(taus), len(spots)), dtype=float)
    for index, tau in enumerate(taus):
        if config.option_type == "put":
            left, right = american_put_boundaries(config.K, float(tau))
        elif config.option_type == "call":
            left, right = american_call_boundaries(
                config.Smax, config.K, float(tau), config.r, config.q
            )
        else:
            raise ValueError("option_type must be put or call")
        result[index] = (1.0 - fraction) * left + fraction * right
    return result


def generate_fom_snapshot(regime: RBRegime, output_path: Path | str) -> RBFOMSnapshot:
    if regime.split != "train":
        raise ValueError("basis snapshots may only be generated from train regimes")
    started = perf_counter()
    config = regime.config()
    result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    if not result.converged:
        raise RuntimeError(f"Policy Iteration failed for {regime.regime_id}")
    lift = boundary_lift_grid(config, result.spot_grid, result.tau_grid)
    state = result.value_grid[:, 1:-1] - lift[:, 1:-1]
    multipliers = np.zeros_like(state)
    active = np.zeros_like(state, dtype=bool)
    residual_rows = np.zeros((config.N + 1, 4), dtype=float)
    max_reassembly_difference = 0.0
    for step in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(config, result.value_grid[step - 1], step)
        values = result.value_grid[step, 1:-1]
        multiplier = tridiagonal_matvec(system, values) - system.rhs
        multiplier[np.abs(multiplier) < 5e-15] = 0.0
        multipliers[step] = multiplier
        gap = values - system.obstacle
        active[step] = gap <= 1e-10 * max(1.0, float(np.max(np.abs(values))))
        residual = compute_lcp_residual(system, values)
        residual_rows[step] = (
            residual.normalized_obstacle_violation,
            residual.normalized_equation_violation,
            residual.normalized_complementarity,
            residual.normalized_lcp_residual,
        )
        max_reassembly_difference = max(
            max_reassembly_difference,
            float(np.max(np.abs(values - result.lcp_results[step - 1].solution))),
        )
    if float(np.max(np.maximum(-multipliers, 0.0))) > 1e-12:
        raise RuntimeError("snapshot multiplier violates nonnegativity")
    if float(np.max(residual_rows[:, 3])) > 1e-12:
        raise RuntimeError("snapshot does not meet frozen LCP tolerance")
    payoff_fn = put_payoff if config.option_type == "put" else call_payoff
    payoff = np.asarray(payoff_fn(result.spot_grid, config.K), dtype=float)
    snapshot = RBFOMSnapshot(
        regime.regime_id,
        regime.option_type,
        result.spot_grid,
        result.tau_grid,
        payoff,
        result.value_grid,
        lift,
        state,
        multipliers,
        active,
        residual_rows,
        {
            "regime": asdict(regime),
            "protocol_hash": protocol_hash(),
            "solver": "CN+Policy Iteration previous-slice",
            "generation_seconds": perf_counter() - started,
            "max_reassembly_difference": max_reassembly_difference,
        },
    )
    save_snapshot(snapshot, output_path)
    return snapshot


def trajectory_multipliers(
    config: AmericanLCPConfig, value_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassemble multipliers, active sets, and shared residuals for any trajectory."""

    values = np.asarray(value_grid, dtype=float)
    if values.shape != (config.N + 1, config.M + 1):
        raise ValueError("value_grid shape must be (N + 1, M + 1)")
    multipliers = np.zeros((config.N + 1, config.M - 1), dtype=float)
    active = np.zeros_like(multipliers, dtype=bool)
    residual_rows = np.zeros((config.N + 1, 4), dtype=float)
    for step in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(config, values[step - 1], step)
        interior = values[step, 1:-1]
        multiplier = tridiagonal_matvec(system, interior) - system.rhs
        multiplier[np.abs(multiplier) < 5e-15] = 0.0
        multipliers[step] = multiplier
        active[step] = interior - system.obstacle <= 1e-10 * max(
            1.0, float(np.max(np.abs(interior)))
        )
        residual = compute_lcp_residual(system, interior)
        residual_rows[step] = (
            residual.normalized_obstacle_violation,
            residual.normalized_equation_violation,
            residual.normalized_complementarity,
            residual.normalized_lcp_residual,
        )
    return multipliers, active, residual_rows


def save_snapshot(snapshot: RBFOMSnapshot, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            regime_id=snapshot.regime_id,
            option_type=snapshot.option_type,
            spot_grid=snapshot.spot_grid,
            tau_grid=snapshot.tau_grid,
            payoff=snapshot.payoff,
            value_grid=snapshot.value_grid,
            boundary_lift_grid=snapshot.boundary_lift_grid,
            lifted_state_grid=snapshot.lifted_state_grid,
            multiplier_grid=snapshot.multiplier_grid,
            active_set_grid=snapshot.active_set_grid,
            residual_by_time=snapshot.residual_by_time,
            metadata_json=json.dumps(snapshot.metadata, sort_keys=True),
        )
    os.replace(temporary, destination)


def load_snapshot(path: Path | str) -> RBFOMSnapshot:
    with np.load(path, allow_pickle=False) as data:
        return RBFOMSnapshot(
            str(data["regime_id"]),
            str(data["option_type"]),
            data["spot_grid"].copy(),
            data["tau_grid"].copy(),
            data["payoff"].copy(),
            data["value_grid"].copy(),
            data["boundary_lift_grid"].copy(),
            data["lifted_state_grid"].copy(),
            data["multiplier_grid"].copy(),
            data["active_set_grid"].astype(bool),
            data["residual_by_time"].copy(),
            json.loads(str(data["metadata_json"])),
        )
