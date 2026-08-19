"""Common Crank-Nicolson marcher for residual-controlled American LCP solvers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.lcp import (
    LCPSolveResult,
    TridiagonalLCP,
    psor_lcp_solve_residual,
)
from american_risk_surfaces.solvers.cn_psor import AmericanCNPSORResult, PSORResult
from american_risk_surfaces.solvers.operator import (
    american_call_boundaries,
    american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
)
from american_risk_surfaces.solvers.penalty_newton import penalty_newton_lcp_solve
from american_risk_surfaces.solvers.policy_iteration import policy_iteration_lcp_solve
from american_risk_surfaces.solvers.projected_lu import (
    factorize_projected_lu,
    projected_lu_lcp_solve,
)


LCPSolverName = Literal[
    "psor",
    "policy_iteration",
    "penalty_newton",
    "projected_lu_single",
    "projected_lu_double",
]
InitialGuessProvider = Callable[[int, float, np.ndarray, np.ndarray], np.ndarray]

__all__ = (
    "AmericanLCPConfig",
    "AmericanLCPResult",
    "InitialGuessProvider",
    "assemble_american_cn_lcp_step",
    "american_cn_lcp_price",
    "as_legacy_cn_psor_result",
)


@dataclass(frozen=True)
class AmericanLCPConfig:
    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    omega: float = 1.2
    tolerance: float = 1e-10
    obstacle_tolerance: float = 1e-12
    max_iter: int = 10000
    penalty: float = 1e10
    penalty_newton_max_iter: int = 100


@dataclass(frozen=True)
class AmericanLCPResult:
    config: AmericanLCPConfig
    solver: str
    initializer: str
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    payoff: np.ndarray
    value_grid: np.ndarray
    values: np.ndarray
    lcp_results: tuple[LCPSolveResult, ...]
    converged: bool
    max_obstacle_violation: float
    initialization_seconds: float
    solver_setup_seconds: float
    lcp_finish_seconds: float
    total_seconds: float


def assemble_american_cn_lcp_step(
    config: AmericanLCPConfig,
    previous_values: np.ndarray,
    step_index: int,
) -> TridiagonalLCP:
    """Assemble one CN LCP using an externally supplied previous time slice.

    This read-only helper lets surrogate and PINN predictions use the exact same
    discrete sign convention and residual implementation as the classical
    marcher. ``step_index`` is one-based and identifies the new time slice.
    """

    validated = _validated_config(config)
    if not 1 <= step_index <= validated.N:
        raise ValueError("step_index must be between 1 and N inclusive.")
    values = np.asarray(previous_values, dtype=float)
    if values.shape != (validated.M + 1,) or not np.all(np.isfinite(values)):
        raise ValueError("previous_values must be a finite array with M + 1 entries.")
    spot_grid, dS = uniform_spot_grid(validated.Smax, validated.M)
    tau_grid, dtau = uniform_tau_grid(validated.T, validated.N)
    payoff_function, boundary_function = _option_helpers(validated)
    payoff = np.asarray(payoff_function(spot_grid, validated.K), dtype=float)
    coefficients = black_scholes_operator_coefficients(
        spot_grid,
        dS=dS,
        r=validated.r,
        q=validated.q,
        sigma=validated.sigma,
    )
    half_step = 0.5 * dtau
    old_tau = float(tau_grid[step_index - 1])
    new_tau = float(tau_grid[step_index])
    old_lower, old_upper = boundary_function(old_tau)
    new_lower, new_upper = boundary_function(new_tau)
    bounded = values.copy()
    bounded[0] = old_lower
    bounded[-1] = old_upper
    rhs = bounded[1:-1] + half_step * apply_black_scholes_operator(bounded, coefficients)
    rhs[0] += half_step * coefficients.lower[0] * new_lower
    rhs[-1] += half_step * coefficients.upper[-1] * new_upper
    return TridiagonalLCP(
        lower=-half_step * coefficients.lower[1:],
        diagonal=1.0 - half_step * coefficients.diagonal,
        upper=-half_step * coefficients.upper[:-1],
        rhs=rhs,
        obstacle=payoff[1:-1],
    )


def as_legacy_cn_psor_result(result: AmericanLCPResult) -> AmericanCNPSORResult:
    """Adapt a generic result for the existing boundary/Greek diagnostic APIs."""

    if not isinstance(result, AmericanLCPResult):
        raise ValueError("result must be an AmericanLCPResult.")
    config = result.config
    step_results = tuple(
        PSORResult(
            solution=step.solution.copy(),
            converged=step.converged,
            iterations=step.iterations,
            final_update=step.residual.normalized_lcp_residual,
            tolerance=step.tolerance,
            omega=config.omega if step.method == "psor" else 1.0,
            max_iter=step.max_iter,
        )
        for step in result.lcp_results
    )
    return AmericanCNPSORResult(
        option_type=config.option_type,
        K=config.K,
        T=config.T,
        r=config.r,
        q=config.q,
        sigma=config.sigma,
        Smax=config.Smax,
        M=config.M,
        N=config.N,
        spot_grid=result.spot_grid.copy(),
        tau_grid=result.tau_grid.copy(),
        payoff=result.payoff.copy(),
        value_grid=result.value_grid.copy(),
        values=result.values.copy(),
        psor_results=step_results,
        converged=result.converged,
        max_obstacle_violation=result.max_obstacle_violation,
    )


def american_cn_lcp_price(
    config: AmericanLCPConfig,
    *,
    lcp_solver: LCPSolverName = "policy_iteration",
    initializer: str | InitialGuessProvider = "previous_slice",
) -> AmericanLCPResult:
    """Price an American option using one shared CN discretization and LCP solver."""

    validated = _validated_config(config)
    projected_lu = lcp_solver in {"projected_lu_single", "projected_lu_double"}
    if projected_lu:
        if callable(initializer):
            raise ValueError("Projected LU is direct and does not accept a custom initializer.")
        if initializer != "previous_slice":
            raise ValueError("Projected LU does not accept an initializer.")
        solver_function = None
        provider = None
        initializer_name = "none_direct"
    else:
        solver_function = _solver_function(lcp_solver)
        provider, initializer_name = _initializer(initializer)
    started = perf_counter()

    spot_grid, dS = uniform_spot_grid(validated.Smax, validated.M)
    tau_grid, dtau = uniform_tau_grid(validated.T, validated.N)
    payoff_function, boundary_function = _option_helpers(validated)
    payoff = np.asarray(payoff_function(spot_grid, validated.K), dtype=float)
    values = payoff.copy()
    value_grid = np.empty((len(tau_grid), len(spot_grid)), dtype=float)
    value_grid[0] = values
    solve_results: list[LCPSolveResult] = []
    initialization_seconds = 0.0
    solver_setup_seconds = 0.0

    if dtau > 0.0:
        coefficients = black_scholes_operator_coefficients(
            spot_grid,
            dS=dS,
            r=validated.r,
            q=validated.q,
            sigma=validated.sigma,
        )
        half_step = 0.5 * dtau
        lhs_lower = -half_step * coefficients.lower[1:]
        lhs_diagonal = 1.0 - half_step * coefficients.diagonal
        lhs_upper = -half_step * coefficients.upper[:-1]
        interior_obstacle = payoff[1:-1]
        projected_factorization = None
        projected_mode = None
        if projected_lu:
            template = TridiagonalLCP(
                lower=lhs_lower,
                diagonal=lhs_diagonal,
                upper=lhs_upper,
                rhs=np.zeros_like(interior_obstacle),
                obstacle=interior_obstacle,
            )
            setup_started = perf_counter()
            if lcp_solver == "projected_lu_double":
                directions = ("lu", "ul")
                projected_mode = "double"
            elif validated.option_type == "put":
                directions = ("ul",)
                projected_mode = "single_put"
            else:
                directions = ("lu",)
                projected_mode = "single_call"
            projected_factorization = factorize_projected_lu(
                template, directions=directions
            )
            solver_setup_seconds = perf_counter() - setup_started

        for step in range(validated.N):
            old_tau = float(tau_grid[step])
            new_tau = float(tau_grid[step + 1])
            old_lower, old_upper = boundary_function(old_tau)
            new_lower, new_upper = boundary_function(new_tau)
            values[0] = old_lower
            values[-1] = old_upper
            rhs = values[1:-1] + half_step * apply_black_scholes_operator(values, coefficients)
            rhs[0] += half_step * coefficients.lower[0] * new_lower
            rhs[-1] += half_step * coefficients.upper[-1] * new_upper
            system = TridiagonalLCP(
                lower=lhs_lower,
                diagonal=lhs_diagonal,
                upper=lhs_upper,
                rhs=rhs,
                obstacle=interior_obstacle,
            )

            if projected_lu:
                assert projected_factorization is not None
                assert projected_mode is not None
                solve_result = projected_lu_lcp_solve(
                    system,
                    projected_factorization,
                    mode=projected_mode,
                    tolerance=validated.tolerance,
                    obstacle_tolerance=validated.obstacle_tolerance,
                )
            else:
                assert provider is not None
                assert solver_function is not None
                init_started = perf_counter()
                initial = provider(
                    step + 1, new_tau, values[1:-1].copy(), interior_obstacle
                )
                initial = np.maximum(np.asarray(initial, dtype=float), interior_obstacle)
                initialization_seconds += perf_counter() - init_started
                if initial.shape != interior_obstacle.shape:
                    raise ValueError(
                        "initializer must return one value for each interior spot node."
                    )

                kwargs: dict[str, Any] = {
                    "initial": initial,
                    "tolerance": validated.tolerance,
                    "obstacle_tolerance": validated.obstacle_tolerance,
                    "max_iter": validated.max_iter,
                }
                if lcp_solver == "psor":
                    kwargs["omega"] = validated.omega
                elif lcp_solver == "penalty_newton":
                    kwargs["penalty"] = validated.penalty
                    kwargs["max_iter"] = validated.penalty_newton_max_iter
                solve_result = solver_function(system, **kwargs)
            solve_results.append(solve_result)

            next_values = np.empty_like(values)
            next_values[0] = new_lower
            next_values[-1] = new_upper
            next_values[1:-1] = solve_result.solution
            values = next_values
            value_grid[step + 1] = values
    else:
        value_grid[:] = payoff

    total_seconds = perf_counter() - started
    finish_seconds = float(sum(result.elapsed_seconds for result in solve_results))
    max_obstacle = float(np.max(np.maximum(payoff[np.newaxis, :] - value_grid, 0.0)))
    return AmericanLCPResult(
        config=validated,
        solver=lcp_solver,
        initializer=initializer_name,
        spot_grid=spot_grid,
        tau_grid=tau_grid,
        payoff=payoff,
        value_grid=value_grid,
        values=values.copy(),
        lcp_results=tuple(solve_results),
        converged=all(result.converged for result in solve_results),
        max_obstacle_violation=max_obstacle,
        initialization_seconds=float(initialization_seconds),
        solver_setup_seconds=float(solver_setup_seconds),
        lcp_finish_seconds=finish_seconds,
        total_seconds=float(total_seconds),
    )


def _solver_function(name: str) -> Callable[..., LCPSolveResult]:
    if name == "psor":
        return psor_lcp_solve_residual
    if name == "policy_iteration":
        return policy_iteration_lcp_solve
    if name == "penalty_newton":
        return penalty_newton_lcp_solve
    raise ValueError(
        "lcp_solver must be 'psor', 'policy_iteration', 'penalty_newton', "
        "'projected_lu_single', or 'projected_lu_double'."
    )


def _initializer(
    initializer: str | InitialGuessProvider,
) -> tuple[InitialGuessProvider, str]:
    if initializer == "previous_slice":
        return (lambda _step, _tau, previous, _obstacle: previous), "previous_slice"
    if callable(initializer):
        return initializer, getattr(initializer, "name", getattr(initializer, "__name__", "custom"))
    raise ValueError("initializer must be 'previous_slice' or a callable.")


def _validated_config(config: AmericanLCPConfig) -> AmericanLCPConfig:
    if not isinstance(config, AmericanLCPConfig):
        raise ValueError("config must be an AmericanLCPConfig.")
    option = str(config.option_type).lower()
    if option not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    if config.K <= 0.0 or config.Smax <= 0.0:
        raise ValueError("K and Smax must be positive.")
    if config.T < 0.0 or config.sigma < 0.0:
        raise ValueError("T and sigma must be nonnegative.")
    if isinstance(config.M, bool) or not isinstance(config.M, int) or config.M < 2:
        raise ValueError("M must be an integer of at least 2.")
    if isinstance(config.N, bool) or not isinstance(config.N, int) or config.N < 1:
        raise ValueError("N must be a positive integer.")
    if not 0.0 < float(config.omega) < 2.0:
        raise ValueError("omega must satisfy 0 < omega < 2.")
    if config.tolerance <= 0.0 or config.obstacle_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive.")
    if not np.isfinite(config.penalty) or config.penalty <= 0.0:
        raise ValueError("penalty must be positive and finite.")
    if isinstance(config.max_iter, bool) or not isinstance(config.max_iter, int) or config.max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if (
        isinstance(config.penalty_newton_max_iter, bool)
        or not isinstance(config.penalty_newton_max_iter, int)
        or config.penalty_newton_max_iter < 1
    ):
        raise ValueError("penalty_newton_max_iter must be a positive integer.")
    return AmericanLCPConfig(
        option_type=option,
        K=float(config.K),
        T=float(config.T),
        r=float(config.r),
        q=float(config.q),
        sigma=float(config.sigma),
        Smax=float(config.Smax),
        M=config.M,
        N=config.N,
        omega=float(config.omega),
        tolerance=float(config.tolerance),
        obstacle_tolerance=float(config.obstacle_tolerance),
        max_iter=config.max_iter,
        penalty=float(config.penalty),
        penalty_newton_max_iter=config.penalty_newton_max_iter,
    )


def _option_helpers(
    config: AmericanLCPConfig,
) -> tuple[Callable[..., Any], Callable[[float], tuple[float, float]]]:
    if config.option_type == "call":
        return call_payoff, lambda tau: american_call_boundaries(
            Smax=config.Smax,
            K=config.K,
            tau=tau,
            r=config.r,
            q=config.q,
        )
    return put_payoff, lambda tau: american_put_boundaries(K=config.K, tau=tau)
