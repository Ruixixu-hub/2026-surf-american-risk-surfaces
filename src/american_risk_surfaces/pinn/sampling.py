"""Deterministic global, focused, curriculum, and adaptive PINN sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.pinn.formulation import PINNProblem


@dataclass(frozen=True)
class SamplingBatch:
    interior: Any
    terminal_x: Any
    boundary_s: Any
    component_counts: dict[str, int]


class PINNSampler:
    """A seeded sampler whose adaptive state never uses reference labels."""

    def __init__(
        self,
        problem: PINNProblem,
        *,
        seed: int,
        pool_size: int = 65536,
        candidate_size: int = 16384,
        device: str = "cpu",
        dtype: Any = None,
    ) -> None:
        torch = _torch()
        self.problem = problem
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype is None else dtype
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.global_pool = self._sobol(pool_size, self.seed)
        self.candidate_pool = self._sobol(candidate_size, self.seed + 991)
        adaptive_size = max(1, int(round(0.20 * pool_size)))
        self.adaptive_pool = self.global_pool[:adaptive_size].clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "torch_generator_state": self.generator.get_state(),
            "numpy_generator_state": self.numpy_rng.bit_generator.state,
            "adaptive_pool": self.adaptive_pool.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # ``torch.load(..., map_location="cuda")`` also moves this CPU RNG
        # byte state to CUDA. CPU Generators only accept a CPU ByteTensor, so
        # normalize it here before restoring interrupted Windows CUDA runs.
        generator_state = _torch().as_tensor(state["torch_generator_state"]).cpu()
        if generator_state.dtype != _torch().uint8:
            generator_state = generator_state.to(dtype=_torch().uint8)
        self.generator.set_state(generator_state.contiguous())
        self.numpy_rng.bit_generator.state = state["numpy_generator_state"]
        adaptive = _torch().as_tensor(state["adaptive_pool"], dtype=self.dtype).cpu()
        if adaptive.ndim != 2 or adaptive.shape[1] != 2:
            raise ValueError("checkpoint adaptive sampler state has the wrong shape.")
        self.adaptive_pool = adaptive.clone()

    def curriculum_upper(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0
        fraction = step / total_steps
        if fraction <= 0.20:
            return 0.10
        if fraction <= 0.40:
            return 0.25
        if fraction <= 0.60:
            return 0.50
        return 1.0

    def sample(
        self,
        batch_size: int,
        boundary_batch_size: int,
        *,
        mode: str,
        time_upper: float = 1.0,
    ) -> SamplingBatch:
        torch = _torch()
        if batch_size < 4 or boundary_batch_size < 2:
            raise ValueError("batch sizes are too small.")
        if not 0.0 < time_upper <= 1.0:
            raise ValueError("time_upper must be in (0, 1].")
        if mode == "global":
            interior = self._draw_pool(self.global_pool, batch_size)
            counts = {"global": batch_size, "maturity": 0, "strike": 0, "adaptive": 0}
        elif mode in {"mixture", "adaptive"}:
            n_global = int(round(0.40 * batch_size))
            n_maturity = int(round(0.20 * batch_size))
            n_strike = int(round(0.20 * batch_size))
            n_adaptive = batch_size - n_global - n_maturity - n_strike
            global_points = self._draw_pool(self.global_pool, n_global)
            maturity = self._focused_maturity(n_maturity)
            strike = self._focused_strike(n_strike)
            source = self.adaptive_pool if mode == "adaptive" else self.global_pool
            adaptive = self._draw_pool(source, n_adaptive)
            interior = torch.cat((global_points, maturity, strike, adaptive), dim=0)
            counts = {
                "global": n_global,
                "maturity": n_maturity,
                "strike": n_strike,
                "adaptive": n_adaptive,
            }
        else:
            raise ValueError("mode must be 'global', 'mixture', or 'adaptive'.")
        interior = interior.clone()
        interior[:, 1] *= time_upper
        terminal_indices = torch.randint(
            len(self.global_pool), (boundary_batch_size,), generator=self.generator
        )
        terminal_x = self.global_pool[terminal_indices, 0:1]
        boundary_s = torch.rand(
            (boundary_batch_size, 1), generator=self.generator, dtype=self.dtype
        ) * time_upper
        return SamplingBatch(
            interior=interior.to(self.device),
            terminal_x=terminal_x.to(self.device),
            boundary_s=boundary_s.to(self.device),
            component_counts=counts,
        )

    def refresh_adaptive(
        self,
        residual_evaluator: Callable[[Any], Any],
        *,
        replace_fraction: float = 0.5,
        x_bins: int = 32,
        s_bins: int = 16,
        time_upper: float = 1.0,
    ) -> dict[str, float]:
        torch = _torch()
        if not 0.0 < replace_fraction <= 1.0:
            raise ValueError("replace_fraction must be in (0, 1].")
        candidates = self.candidate_pool.clone()
        candidates[:, 1] *= time_upper
        residual = residual_evaluator(candidates.to(self.device))
        scores = torch.as_tensor(residual).detach().abs().reshape(-1).cpu().numpy()
        points = candidates.cpu().numpy()
        x_index = np.minimum(
            x_bins - 1,
            np.floor((points[:, 0] - self.problem.x_min) / (self.problem.x_max - self.problem.x_min) * x_bins).astype(int),
        )
        s_index = np.minimum(
            s_bins - 1,
            np.floor(points[:, 1] / max(time_upper, np.finfo(float).eps) * s_bins).astype(int),
        )
        selected: list[int] = []
        for cell in np.unique(x_index * s_bins + s_index):
            members = np.flatnonzero(x_index * s_bins + s_index == cell)
            selected.append(int(members[np.argmax(scores[members])]))
        target = max(1, int(round(replace_fraction * len(self.adaptive_pool))))
        selected = sorted(selected, key=lambda index: scores[index], reverse=True)[:target]
        if len(selected) < target:
            ranked = np.argsort(-scores)
            seen = set(selected)
            selected.extend(int(index) for index in ranked if int(index) not in seen)
            selected = selected[:target]
        replacement = candidates[torch.as_tensor(selected, dtype=torch.long)]
        keep = len(self.adaptive_pool) - len(replacement)
        self.adaptive_pool = torch.cat((self.adaptive_pool[:keep], replacement), dim=0)
        occupied = len(np.unique(x_index[selected] * s_bins + s_index[selected]))
        return {
            "selected": float(len(selected)),
            "occupied_bins": float(occupied),
            "max_residual": float(np.max(scores)),
            "median_selected_residual": float(np.median(scores[selected])),
        }

    def _sobol(self, size: int, seed: int) -> Any:
        torch = _torch()
        if size < 1:
            raise ValueError("pool sizes must be positive.")
        unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(size)
        unit = unit.to(dtype=self.dtype)
        unit[:, 0] = self.problem.x_min + unit[:, 0] * (
            self.problem.x_max - self.problem.x_min
        )
        return unit

    def _draw_pool(self, pool: Any, count: int) -> Any:
        torch = _torch()
        indices = torch.randint(len(pool), (count,), generator=self.generator)
        return pool[indices]

    def _focused_maturity(self, count: int) -> Any:
        torch = _torch()
        x = self._draw_pool(self.global_pool, count)[:, 0:1]
        s = torch.as_tensor(
            self.numpy_rng.beta(0.5, 2.0, size=(count, 1)), dtype=self.dtype
        )
        return torch.cat((x, s), dim=1)

    def _focused_strike(self, count: int) -> Any:
        torch = _torch()
        x = torch.randn((count, 1), generator=self.generator, dtype=self.dtype) * 0.15
        x = torch.clamp(x, self.problem.x_min, self.problem.x_max)
        s = torch.rand((count, 1), generator=self.generator, dtype=self.dtype)
        return torch.cat((x, s), dim=1)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for PINN sampling.") from exc
    return torch
