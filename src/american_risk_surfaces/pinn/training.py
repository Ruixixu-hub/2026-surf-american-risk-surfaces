"""Checkpointed single-regime training for SURF PINN Arms C and D."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from american_risk_surfaces.pinn.formulation import (
    PINNArm,
    PINNProblem,
    payoff_torch,
    smooth_fischer_burmeister,
    spatial_boundary_torch,
    trial_value,
    value_and_vi_residual,
)
from american_risk_surfaces.pinn.networks import NetworkSpec, build_network, count_parameters
from american_risk_surfaces.pinn.sampling import PINNSampler, SamplingBatch


RunStatus = Literal["COMPLETE", "FAILED", "BUDGET_EXHAUSTED"]


@dataclass(frozen=True)
class PINNTrainingConfig:
    arm: PINNArm
    seed: int
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: str = "float64"
    adam_steps: int = 40000
    lbfgs_max_evaluations: int = 2000
    interior_batch_size: int = 4096
    boundary_batch_size: int = 512
    checkpoint_interval: int = 2000
    gradient_log_interval: int = 200
    adaptive_interval: int = 2000
    pool_size: int = 65536
    candidate_size: int = 16384
    max_seconds: float = 3600.0
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-5
    d_variant: Literal[
        "etc_soft",
        "etc_fb_global",
        "etc_fb_mixture",
        "etc_fb_adaptive",
        "positive_premium",
    ] = "etc_fb_adaptive"
    network_spec: NetworkSpec = NetworkSpec()

    def __post_init__(self) -> None:
        if self.arm not in {"C", "D"}:
            raise ValueError("arm must be 'C' or 'D'.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.dtype != "float64":
            raise ValueError("the frozen PINN protocol requires float64.")
        integers = (
            self.adam_steps,
            self.lbfgs_max_evaluations,
            self.interior_batch_size,
            self.boundary_batch_size,
            self.checkpoint_interval,
            self.gradient_log_interval,
            self.adaptive_interval,
            self.pool_size,
            self.candidate_size,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in integers):
            raise ValueError("training counts must be nonnegative integers.")
        if min(self.interior_batch_size, self.boundary_batch_size, self.pool_size) < 2:
            raise ValueError("batch and pool sizes are too small.")
        if self.max_seconds <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("time and learning-rate controls must be positive.")


@dataclass(frozen=True)
class PINNRunResult:
    status: RunStatus
    checkpoint_path: Path
    training_seconds: float
    best_physics_monitor_score: float
    adam_steps_completed: int
    lbfgs_evaluations: int
    failure_reason: str | None


def train_single_regime_pinn(
    problem: PINNProblem,
    config: PINNTrainingConfig,
    *,
    output_dir: Path | str,
    resume: bool = False,
) -> PINNRunResult:
    """Train one physics-only model without importing any reference labels."""

    torch = _torch()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{config.arm}_{problem.regime_id}_seed{config.seed}"
    latest_path = checkpoint_dir / f"{run_name}_latest.pt"
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    history_path = output / f"{run_name}_history.csv"
    status_path = output / f"{run_name}_status.json"
    heartbeat_path = output / f"{run_name}_heartbeat.json"
    config_hash = training_config_hash(problem, config)
    compatible_resume_hashes = {
        config_hash,
        _legacy_training_config_hash(problem, config, max_seconds=3600.0),
    }

    device = _resolve_device(config.device)
    _set_seed(config.seed)
    model = build_network(config.network_spec).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.adam_steps),
        eta_min=config.final_learning_rate,
    )
    sampler = PINNSampler(
        problem,
        seed=config.seed,
        pool_size=config.pool_size,
        candidate_size=config.candidate_size,
        device=str(device),
        dtype=torch.float64,
    )
    monitor = sampler.sample(
        min(2048, config.interior_batch_size),
        min(256, config.boundary_batch_size),
        mode="global",
    )
    started_step = 0
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    recovered_once = False
    prior_training_seconds = 0.0

    if resume and status_path.exists():
        terminal = json.loads(status_path.read_text(encoding="utf-8"))
        if terminal.get("config_hash") not in compatible_resume_hashes:
            raise ValueError("terminal status config hash does not match the requested run.")
        terminal_status = terminal.get("status")
        budget_can_resume = (
            terminal_status == "BUDGET_EXHAUSTED"
            and float(terminal.get("training_seconds", 0.0)) < config.max_seconds
        )
        if terminal_status in {"COMPLETE", "FAILED", "BUDGET_EXHAUSTED"} and not budget_can_resume:
            return PINNRunResult(
                status=terminal["status"],
                checkpoint_path=Path(terminal["checkpoint_path"]),
                training_seconds=float(terminal["training_seconds"]),
                best_physics_monitor_score=float(terminal["best_physics_monitor_score"]),
                adam_steps_completed=int(terminal["adam_steps_completed"]),
                lbfgs_evaluations=int(terminal["lbfgs_evaluations"]),
                failure_reason=terminal.get("failure_reason"),
            )

    if resume and latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("config_hash") not in compatible_resume_hashes:
            raise ValueError("checkpoint config hash does not match the requested run.")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        started_step = int(saved["adam_step"])
        best_score = float(saved["best_physics_monitor_score"])
        recovered_once = bool(saved.get("recovered_once", False))
        prior_training_seconds = float(saved.get("cumulative_training_seconds", 0.0))
        if "sampler_state" in saved:
            sampler.load_state_dict(saved["sampler_state"])
        history = list(saved.get("history", []))

    start_time = perf_counter()
    status: RunStatus = "COMPLETE"
    failure_reason: str | None = None
    completed_step = started_step
    lbfgs_evaluations = 0
    _write_heartbeat(
        heartbeat_path,
        problem=problem,
        config=config,
        config_hash=config_hash,
        phase="ADAM",
        step=started_step,
        cumulative_seconds=prior_training_seconds,
    )

    try:
        for step in range(started_step + 1, config.adam_steps + 1):
            if prior_training_seconds + perf_counter() - start_time >= config.max_seconds:
                status = "BUDGET_EXHAUSTED"
                failure_reason = "one-seed wall-clock budget exhausted during Adam"
                break
            time_upper = sampler.curriculum_upper(step, config.adam_steps) if config.arm == "D" else 1.0
            batch = sampler.sample(
                config.interior_batch_size,
                config.boundary_batch_size,
                mode=_sampling_mode(config),
                time_upper=time_upper,
            )
            optimizer.zero_grad(set_to_none=True)
            terms = _loss_terms(model, problem, config, batch)
            total = sum(terms.values())
            if not bool(torch.isfinite(total)):
                if recovered_once or not latest_path.exists():
                    raise FloatingPointError("non-finite PINN loss")
                saved = torch.load(latest_path, map_location=device, weights_only=False)
                model.load_state_dict(saved["state_dict"])
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1
                recovered_once = True
                continue
            gradient_norms = {}
            if config.gradient_log_interval and step % config.gradient_log_interval == 0:
                gradient_norms = _gradient_norms(model, terms)
            total.backward()
            optimizer.step()
            scheduler.step()
            completed_step = step

            adaptive_row: dict[str, float] = {}
            if (
                config.arm == "D"
                and config.d_variant in {"etc_fb_adaptive", "positive_premium"}
                and config.adaptive_interval
                and step % config.adaptive_interval == 0
            ):
                adaptive_row = sampler.refresh_adaptive(
                    lambda points: _adaptive_residual(model, problem, config, points),
                    time_upper=time_upper,
                )

            if step == 1 or step % max(1, config.gradient_log_interval) == 0:
                history.append(
                    {
                        "phase": "adam",
                        "step": step,
                        "total_loss": float(total.detach().cpu()),
                        **{name: float(value.detach().cpu()) for name, value in terms.items()},
                        **{f"grad_{name}": value for name, value in gradient_norms.items()},
                        **{f"adaptive_{name}": value for name, value in adaptive_row.items()},
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "time_upper": time_upper,
                        "elapsed_seconds": perf_counter() - start_time,
                    }
                )

            if step % config.checkpoint_interval == 0 or step == config.adam_steps:
                monitor_score = _monitor_score(model, problem, config, monitor)
                best_score = min(best_score, monitor_score)
                payload = _checkpoint_payload(
                    problem,
                    config,
                    config_hash,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_score,
                    recovered_once,
                    sampler,
                    history,
                    prior_training_seconds + perf_counter() - start_time,
                )
                _atomic_torch_save(payload, latest_path)
                if monitor_score <= best_score:
                    _atomic_torch_save(payload, best_path)
                _write_history(history_path, history)
                _write_heartbeat(
                    heartbeat_path,
                    problem=problem,
                    config=config,
                    config_hash=config_hash,
                    phase="ADAM",
                    step=step,
                    cumulative_seconds=prior_training_seconds + perf_counter() - start_time,
                    monitor_score=monitor_score,
                )
                cumulative = prior_training_seconds + perf_counter() - start_time
                progress = step / max(config.adam_steps, 1)
                eta = cumulative / progress - cumulative if progress > 0.0 else float("inf")
                print(
                    f"  [Adam {step}/{config.adam_steps} | {100.0 * progress:5.1f}%] "
                    f"monitor={monitor_score:.6e} elapsed={_format_duration(cumulative)} "
                    f"eta={_format_duration(eta)}",
                    flush=True,
                )

        if status == "COMPLETE" and config.lbfgs_max_evaluations > 0:
            remaining = config.max_seconds - (
                prior_training_seconds + perf_counter() - start_time
            )
            if remaining <= 0.0:
                status = "BUDGET_EXHAUSTED"
                failure_reason = "one-seed wall-clock budget exhausted before L-BFGS"
            else:
                fixed_batch = sampler.sample(
                    config.interior_batch_size,
                    config.boundary_batch_size,
                    mode=_sampling_mode(config),
                )
                lbfgs = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=config.lbfgs_max_evaluations,
                    max_eval=config.lbfgs_max_evaluations,
                    history_size=50,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-9,
                    tolerance_change=1e-12,
                )

                def closure() -> Any:
                    nonlocal lbfgs_evaluations, status, failure_reason
                    if prior_training_seconds + perf_counter() - start_time >= config.max_seconds:
                        status = "BUDGET_EXHAUSTED"
                        failure_reason = "one-seed wall-clock budget exhausted during L-BFGS"
                        raise _BudgetExhausted
                    lbfgs.zero_grad(set_to_none=True)
                    terms = _loss_terms(model, problem, config, fixed_batch)
                    loss = sum(terms.values())
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("non-finite L-BFGS loss")
                    loss.backward()
                    lbfgs_evaluations += 1
                    return loss

                try:
                    lbfgs.step(closure)
                except _BudgetExhausted:
                    pass
                monitor_score = _monitor_score(model, problem, config, monitor)
                best_score = min(best_score, monitor_score)
                history.append(
                    {
                        "phase": "lbfgs",
                        "step": lbfgs_evaluations,
                        "total_loss": monitor_score,
                        "elapsed_seconds": perf_counter() - start_time,
                    }
                )
                print(
                    f"  [L-BFGS {lbfgs_evaluations}/{config.lbfgs_max_evaluations}] "
                    f"monitor={monitor_score:.6e}",
                    flush=True,
                )
                payload = _checkpoint_payload(
                    problem,
                    config,
                    config_hash,
                    model,
                    optimizer,
                    scheduler,
                    completed_step,
                    best_score,
                    recovered_once,
                    sampler,
                    history,
                    prior_training_seconds + perf_counter() - start_time,
                )
                _atomic_torch_save(payload, latest_path)
                if monitor_score <= best_score:
                    _atomic_torch_save(payload, best_path)
                _write_history(history_path, history)
                _write_heartbeat(
                    heartbeat_path,
                    problem=problem,
                    config=config,
                    config_hash=config_hash,
                    phase="LBFGS",
                    step=lbfgs_evaluations,
                    cumulative_seconds=prior_training_seconds + perf_counter() - start_time,
                    monitor_score=monitor_score,
                )
    except Exception as exc:
        status = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"

    elapsed = prior_training_seconds + perf_counter() - start_time
    selected_checkpoint = best_path if best_path.exists() else latest_path
    status_payload = {
        "status": status,
        "failure_reason": failure_reason,
        "regime_id": problem.regime_id,
        "arm": config.arm,
        "d_variant": config.d_variant if config.arm == "D" else None,
        "seed": config.seed,
        "config_hash": config_hash,
        "checkpoint_path": str(selected_checkpoint),
        "training_seconds": elapsed,
        "best_physics_monitor_score": best_score,
        "adam_steps_completed": completed_step,
        "lbfgs_evaluations": lbfgs_evaluations,
        "device": str(device),
    }
    _atomic_json(status_path, status_payload)
    _write_heartbeat(
        heartbeat_path,
        problem=problem,
        config=config,
        config_hash=config_hash,
        phase=status,
        step=completed_step,
        cumulative_seconds=elapsed,
        monitor_score=best_score,
        failure_reason=failure_reason,
    )
    return PINNRunResult(
        status=status,
        checkpoint_path=selected_checkpoint,
        training_seconds=float(elapsed),
        best_physics_monitor_score=float(best_score),
        adam_steps_completed=completed_step,
        lbfgs_evaluations=lbfgs_evaluations,
        failure_reason=failure_reason,
    )


def training_config_hash(problem: PINNProblem, config: PINNTrainingConfig) -> str:
    """Hash optimization semantics while excluding the operational wall-clock guard."""

    config_payload = {**asdict(config), "network_spec": config.network_spec.to_dict()}
    config_payload.pop("max_seconds", None)
    payload = {
        "problem": asdict(problem),
        "config": config_payload,
        "protocol": "surf_pinn_cde_v1",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _legacy_training_config_hash(
    problem: PINNProblem,
    config: PINNTrainingConfig,
    *,
    max_seconds: float,
) -> str:
    """Reproduce pre-parallel hashes so one-hour checkpoints remain resumable."""

    config_payload = {**asdict(config), "network_spec": config.network_spec.to_dict()}
    config_payload["max_seconds"] = float(max_seconds)
    payload = {
        "problem": asdict(problem),
        "config": config_payload,
        "protocol": "surf_pinn_cde_v1",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _loss_terms(
    model: Any,
    problem: PINNProblem,
    config: PINNTrainingConfig,
    batch: SamplingBatch,
) -> dict[str, Any]:
    torch = _torch()
    interior = batch.interior.detach().clone().requires_grad_(True)
    residual = value_and_vi_residual(
        model,
        problem,
        interior,
        arm=config.arm,
        representation=_representation(config),
    )
    normalized_gap = residual["normalized_gap"]
    normalized_equation = residual["normalized_equation"]
    soft_objective = config.arm == "C" or config.d_variant == "etc_soft"
    if soft_objective:
        terms = {
            "obstacle": torch.relu(-normalized_gap).square().mean(),
            "equation": torch.relu(-normalized_equation).square().mean(),
            "complementarity": (normalized_gap * normalized_equation).square().mean(),
        }
    else:
        terms = {
            "fb": smooth_fischer_burmeister(
                normalized_gap, normalized_equation, epsilon=1e-12
            ).square().mean()
        }

    s_boundary = batch.boundary_s
    left_x = torch.full_like(s_boundary, problem.x_min)
    right_x = torch.full_like(s_boundary, problem.x_max)
    expected_left, expected_right = spatial_boundary_torch(problem, s_boundary)
    predicted_left = trial_value(
        model,
        problem,
        left_x,
        s_boundary,
        arm=config.arm,
        representation=_representation(config),
    )
    predicted_right = trial_value(
        model,
        problem,
        right_x,
        s_boundary,
        arm=config.arm,
        representation=_representation(config),
    )
    terms["boundary"] = (
        (predicted_left - expected_left).square().mean()
        + (predicted_right - expected_right).square().mean()
    ) / problem.value_scale**2
    if config.arm == "C":
        terminal_s = torch.zeros_like(batch.terminal_x)
        predicted_terminal = trial_value(
            model, problem, batch.terminal_x, terminal_s, arm="C"
        )
        terms["terminal"] = (
            (predicted_terminal - payoff_torch(problem, batch.terminal_x)).square().mean()
            / problem.value_scale**2
        )
    return terms


def _adaptive_residual(
    model: Any,
    problem: PINNProblem,
    config: PINNTrainingConfig,
    points: Any,
) -> Any:
    torch = _torch()
    values = []
    for start in range(0, len(points), 2048):
        batch = points[start : start + 2048].detach().clone().requires_grad_(True)
        residual = value_and_vi_residual(
            model,
            problem,
            batch,
            arm=config.arm,
            representation=_representation(config),
            create_graph=False,
        )
        values.append(
            smooth_fischer_burmeister(
                residual["normalized_gap"], residual["normalized_equation"]
            ).detach()
        )
    return torch.cat(values, dim=0)


def _monitor_score(
    model: Any,
    problem: PINNProblem,
    config: PINNTrainingConfig,
    monitor: SamplingBatch,
) -> float:
    terms = _loss_terms(model, problem, config, monitor)
    return float(sum(terms.values()).detach().cpu())


def _gradient_norms(model: Any, terms: dict[str, Any]) -> dict[str, float]:
    torch = _torch()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    result: dict[str, float] = {}
    for name, term in terms.items():
        gradients = torch.autograd.grad(
            term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = sum(
            gradient.detach().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        result[name] = float(torch.sqrt(squared).cpu()) if not isinstance(squared, int) else 0.0
    return result


def _sampling_mode(config: PINNTrainingConfig) -> str:
    if config.arm == "C" or config.d_variant in {"etc_soft", "etc_fb_global"}:
        return "global"
    if config.d_variant == "etc_fb_mixture":
        return "mixture"
    return "adaptive"


def _representation(config: PINNTrainingConfig) -> str:
    return "positive_premium" if config.d_variant == "positive_premium" else "smooth_etc"


def _checkpoint_payload(
    problem: PINNProblem,
    config: PINNTrainingConfig,
    config_hash: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    adam_step: int,
    best_score: float,
    recovered_once: bool,
    sampler: PINNSampler,
    history: list[dict[str, Any]],
    cumulative_training_seconds: float,
) -> dict[str, Any]:
    torch = _torch()
    return {
        "protocol": "surf_pinn_cde_v1",
        "problem": asdict(problem),
        "training_config": {**asdict(config), "network_spec": config.network_spec.to_dict()},
        "network_spec": config.network_spec.to_dict(),
        "config_hash": config_hash,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "adam_step": adam_step,
        "best_physics_monitor_score": best_score,
        "recovered_once": recovered_once,
        "sampler_state": sampler.state_dict(),
        "history": history,
        "cumulative_training_seconds": cumulative_training_seconds,
        "parameter_count": count_parameters(model),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def _resolve_device(requested: str) -> Any:
    torch = _torch()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    torch = _torch()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_heartbeat(
    path: Path,
    *,
    problem: PINNProblem,
    config: PINNTrainingConfig,
    config_hash: str,
    phase: str,
    step: int,
    cumulative_seconds: float,
    monitor_score: float | None = None,
    failure_reason: str | None = None,
) -> None:
    """Atomically expose progress for Windows shard supervision."""

    _atomic_json(
        path,
        {
            "protocol": "surf_pinn_cde_v1",
            "regime_id": problem.regime_id,
            "arm": config.arm,
            "seed": config.seed,
            "phase": phase,
            "step": int(step),
            "cumulative_seconds": float(cumulative_seconds),
            "physics_monitor_score": monitor_score,
            "failure_reason": failure_reason,
            "config_hash": config_hash,
        },
    )


class _BudgetExhausted(Exception):
    pass


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for PINN training.") from exc
    return torch
