"""Mathematical formulation for the SURF American-option PINN arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


OptionType = Literal["put", "call"]
PINNArm = Literal["C", "D"]


@dataclass(frozen=True)
class PINNProblem:
    """One label-free American-option variational-inequality problem."""

    regime_id: str
    option_type: OptionType
    T: float
    r: float
    q: float
    sigma: float
    m_min: float = 1e-4
    m_max: float = 4.0

    def __post_init__(self) -> None:
        option = str(self.option_type).lower()
        if option not in {"put", "call"}:
            raise ValueError("option_type must be 'put' or 'call'.")
        if not self.regime_id:
            raise ValueError("regime_id must not be empty.")
        scalars = (self.T, self.r, self.q, self.sigma, self.m_min, self.m_max)
        if not all(np.isfinite(float(value)) for value in scalars):
            raise ValueError("problem parameters must be finite.")
        if self.T <= 0.0:
            raise ValueError("T must be positive for normalized-time PINNs.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative.")
        if self.m_min <= 0.0 or self.m_max <= self.m_min:
            raise ValueError("moneyness bounds must satisfy 0 < m_min < m_max.")
        object.__setattr__(self, "option_type", option)

    @property
    def x_min(self) -> float:
        return float(np.log(self.m_min))

    @property
    def x_max(self) -> float:
        return float(np.log(self.m_max))

    @property
    def value_scale(self) -> float:
        maximum_payoff = 1.0 if self.option_type == "put" else self.m_max - 1.0
        return float(max(1.0, maximum_payoff))

    @property
    def equation_scale(self) -> float:
        return float(
            self.value_scale * (1.0 + self.T * (self.r + self.q + self.sigma**2))
        )


def payoff_numpy(problem: PINNProblem, x: Any) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    moneyness = np.exp(values)
    if problem.option_type == "put":
        return np.maximum(1.0 - moneyness, 0.0)
    return np.maximum(moneyness - 1.0, 0.0)


def payoff_torch(problem: PINNProblem, x: Any) -> Any:
    torch = _torch()
    moneyness = torch.exp(x)
    if problem.option_type == "put":
        return torch.clamp(1.0 - moneyness, min=0.0)
    return torch.clamp(moneyness - 1.0, min=0.0)


def spatial_boundary_numpy(problem: PINNProblem, s: Any) -> tuple[np.ndarray, np.ndarray]:
    normalized_time = np.asarray(s, dtype=float)
    tau = problem.T * normalized_time
    if problem.option_type == "put":
        return (
            np.full_like(normalized_time, 1.0 - problem.m_min),
            np.zeros_like(normalized_time),
        )
    lower = np.zeros_like(normalized_time)
    upper = np.maximum(
        problem.m_max - 1.0,
        problem.m_max * np.exp(-problem.q * tau) - np.exp(-problem.r * tau),
    )
    return lower, upper


def spatial_boundary_torch(problem: PINNProblem, s: Any) -> tuple[Any, Any]:
    torch = _torch()
    tau = problem.T * s
    if problem.option_type == "put":
        return torch.full_like(s, 1.0 - problem.m_min), torch.zeros_like(s)
    lower = torch.zeros_like(s)
    intrinsic = torch.full_like(s, problem.m_max - 1.0)
    asymptotic = problem.m_max * torch.exp(-problem.q * tau) - torch.exp(
        -problem.r * tau
    )
    return lower, torch.maximum(intrinsic, asymptotic)


def exact_terminal_lift(problem: PINNProblem, x: Any, s: Any) -> Any:
    """ETCNN-style singularity-aware lift in normalized variables.

    The explicit zero-time branch is essential: it both enforces the payoff and
    avoids evaluating a zero denominator in the Gaussian argument.
    """

    torch = _torch()
    tau = problem.T * s
    safe_tau = torch.clamp(tau, min=torch.finfo(x.dtype).eps)
    root_tau = torch.sqrt(safe_tau)
    moneyness = torch.exp(x)
    drift = x + (problem.r - problem.q) * safe_tau
    sign = -1.0 if problem.option_type == "put" else 1.0
    d0 = sign * drift / (problem.sigma * root_tau + torch.finfo(x.dtype).eps)
    normal_cdf = 0.5 * (1.0 + torch.erf(d0 / np.sqrt(2.0)))
    normal_pdf = torch.exp(-0.5 * d0.square()) / np.sqrt(2.0 * np.pi)
    discounted_spot = moneyness * torch.exp(-problem.q * safe_tau)
    discounted_strike = torch.exp(-problem.r * safe_tau)
    if problem.option_type == "put":
        leading = normal_cdf * (discounted_strike - discounted_spot)
    else:
        leading = normal_cdf * (discounted_spot - discounted_strike)
    singular = (
        0.5
        * problem.sigma
        * root_tau
        * normal_pdf
        * (discounted_spot + discounted_strike)
    )
    lift = leading + singular
    return torch.where(tau <= 0.0, payoff_torch(problem, x), lift)


def smooth_fischer_burmeister(a: Any, b: Any, epsilon: float = 1e-12) -> Any:
    torch = _torch()
    eps = torch.as_tensor(epsilon, dtype=a.dtype, device=a.device)
    return a + b - torch.sqrt(a.square() + b.square() + eps.square()) + eps


def fischer_burmeister_numpy(a: Any, b: Any) -> np.ndarray:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    return first + second - np.sqrt(first**2 + second**2)


def normalized_network_inputs(problem: PINNProblem, x: Any, s: Any) -> Any:
    torch = _torch()
    scaled_x = 2.0 * (x - problem.x_min) / (problem.x_max - problem.x_min) - 1.0
    scaled_s = 2.0 * s - 1.0
    return torch.cat((scaled_x, scaled_s), dim=1)


def trial_value(
    model: Any,
    problem: PINNProblem,
    x: Any,
    s: Any,
    *,
    arm: PINNArm,
    representation: str = "smooth_etc",
) -> Any:
    raw = model(normalized_network_inputs(problem, x, s))
    if arm == "C":
        return raw
    if arm != "D":
        raise ValueError("arm must be 'C' or 'D'.")
    if representation == "smooth_etc":
        return exact_terminal_lift(problem, x, s) + s * raw
    if representation == "positive_premium":
        return payoff_torch(problem, x) + s * _torch().nn.functional.softplus(raw)
    raise ValueError("unknown Arm D representation.")


def value_and_vi_residual(
    model: Any,
    problem: PINNProblem,
    coordinates: Any,
    *,
    arm: PINNArm,
    representation: str = "smooth_etc",
    create_graph: bool = True,
) -> dict[str, Any]:
    """Evaluate value, derivatives, and consistently signed VI residuals."""

    torch = _torch()
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2) with columns (x, s).")
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)
    x = coordinates[:, 0:1]
    s = coordinates[:, 1:2]
    value = trial_value(
        model, problem, x, s, arm=arm, representation=representation
    )
    gradient = torch.autograd.grad(
        value,
        coordinates,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]
    value_x = gradient[:, 0:1]
    value_s = gradient[:, 1:2]
    second_gradient = torch.autograd.grad(
        value_x,
        coordinates,
        grad_outputs=torch.ones_like(value_x),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    value_xx = second_gradient[:, 0:1]
    operator = (
        0.5 * problem.sigma**2 * value_xx
        + (problem.r - problem.q - 0.5 * problem.sigma**2) * value_x
        - problem.r * value
    )
    equation_gap = value_s - problem.T * operator
    obstacle_gap = value - payoff_torch(problem, x)
    normalized_gap = obstacle_gap / problem.value_scale
    normalized_equation = equation_gap / problem.equation_scale
    return {
        "coordinates": coordinates,
        "value": value,
        "value_x": value_x,
        "value_xx": value_xx,
        "value_s": value_s,
        "operator": operator,
        "obstacle_gap": obstacle_gap,
        "equation_gap": equation_gap,
        "normalized_gap": normalized_gap,
        "normalized_equation": normalized_equation,
        "complementarity": obstacle_gap * equation_gap,
        "fb": fischer_burmeister_torch(normalized_gap, normalized_equation),
    }


def fischer_burmeister_torch(a: Any, b: Any) -> Any:
    return a + b - _torch().sqrt(a.square() + b.square())


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("PyTorch is required for PINN operations.") from exc
    return torch
