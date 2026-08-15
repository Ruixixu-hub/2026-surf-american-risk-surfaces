"""Differentiable CN-LCP residuals shared by the N2 DeepONet loss."""

from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TorchLCPResidual:
    obstacle_gap: object
    equation_gap: object
    normalized_obstacle_gap: object
    normalized_equation_gap: object
    fischer_burmeister: object
    value_scale: object
    equation_scale: object


def smooth_fischer_burmeister(a, b, epsilon: float = 1e-12):
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable LCP residuals")
    eps = torch.as_tensor(float(epsilon), dtype=a.dtype, device=a.device)
    return a + b - torch.sqrt(a.square() + b.square() + eps.square()) + eps


def batched_cn_lcp_residual(
    value_grid,
    payoff,
    T,
    sigma,
    r,
    q,
    *,
    epsilon: float = 1e-12,
) -> TorchLCPResidual:
    """Match ``assemble_american_cn_lcp_step`` for batched 121x121 trajectories."""

    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable LCP residuals")
    if value_grid.ndim != 3 or tuple(value_grid.shape[1:]) != (121, 121):
        raise ValueError("value_grid must have shape (batch,121,121)")
    batch = value_grid.shape[0]
    for name, item in (("T", T), ("sigma", sigma), ("r", r), ("q", q)):
        if item.ndim != 1 or item.shape[0] != batch:
            raise ValueError(f"{name} must have one entry per trajectory")
    obstacle = payoff
    if obstacle.ndim == 1:
        obstacle = obstacle.unsqueeze(0).expand(batch, -1)
    if tuple(obstacle.shape) != (batch, 121):
        raise ValueError("payoff must have shape (121,) or (batch,121)")
    dtype, device = value_grid.dtype, value_grid.device
    spots = torch.linspace(0.0, 4.0, 121, dtype=dtype, device=device)[1:-1]
    dS = torch.as_tensor(4.0 / 120.0, dtype=dtype, device=device)
    diffusion = 0.5 * sigma[:, None].square() * spots[None, :].square() / dS.square()
    drift = 0.5 * (r - q)[:, None] * spots[None, :] / dS
    lower = diffusion - drift
    diagonal = -2.0 * diffusion - r[:, None]
    upper = diffusion + drift
    half = 0.5 * T / 120.0
    old = value_grid[:, :-1, :]
    new = value_grid[:, 1:, :]
    old_operator = (
        lower[:, None, :] * old[:, :, :-2]
        + diagonal[:, None, :] * old[:, :, 1:-1]
        + upper[:, None, :] * old[:, :, 2:]
    )
    rhs = old[:, :, 1:-1] + half[:, None, None] * old_operator
    rhs_adjustment = torch.zeros_like(rhs)
    rhs_adjustment[:, :, 0] = half[:, None] * lower[:, None, 0] * new[:, :, 0]
    rhs_adjustment[:, :, -1] = half[:, None] * upper[:, None, -1] * new[:, :, -1]
    rhs = rhs + rhs_adjustment
    diagonal_a = 1.0 - half[:, None] * diagonal
    new_interior = new[:, :, 1:-1]
    zeros = torch.zeros((batch, 120, 1), dtype=dtype, device=device)
    lower_action = torch.cat(
        (
            zeros,
            -half[:, None, None] * lower[:, None, 1:] * new_interior[:, :, :-1],
        ),
        dim=2,
    )
    upper_action = torch.cat(
        (
            -half[:, None, None] * upper[:, None, :-1] * new_interior[:, :, 1:],
            zeros,
        ),
        dim=2,
    )
    matrix_action = diagonal_a[:, None, :] * new_interior + lower_action + upper_action
    gap = new[:, :, 1:-1] - obstacle[:, None, 1:-1]
    equation = matrix_action - rhs
    value_scale = torch.maximum(
        torch.ones((batch, 120, 1), dtype=dtype, device=device),
        torch.maximum(
            new[:, :, 1:-1].abs().amax(dim=2, keepdim=True),
            obstacle[:, None, 1:-1].abs().amax(dim=2, keepdim=True),
        ),
    ).detach()
    equation_scale = torch.maximum(
        torch.ones((batch, 120, 1), dtype=dtype, device=device),
        torch.maximum(
            matrix_action.abs().amax(dim=2, keepdim=True),
            rhs.abs().amax(dim=2, keepdim=True),
        ),
    ).detach()
    normalized_gap = gap / value_scale
    normalized_equation = equation / equation_scale
    fb = smooth_fischer_burmeister(normalized_gap, normalized_equation, epsilon)
    return TorchLCPResidual(
        gap, equation, normalized_gap, normalized_equation, fb,
        value_scale, equation_scale,
    )
