"""High-accuracy DIRK/Policy reference cache for formal PINN scoring."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.pinn.protocol import RegimeRecord, load_regime_records
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid


def generate_reference_cache(
    output_dir: Path | str,
    *,
    splits: Iterable[str],
    spatial_steps: int = 480,
    time_steps: int = 960,
    regime_limit: int | None = None,
    regime_ids: Iterable[str] | None = None,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_names = tuple(splits)
    selected_ids = None if regime_ids is None else tuple(regime_ids)
    records = load_regime_records(splits=split_names, regime_ids=selected_ids)
    if regime_limit is not None:
        records = records[:regime_limit]
    paths = []
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        path = output / f"{record.regime_id}.npz"
        if path.exists():
            paths.append(path)
            print(
                f"[reference {index}/{len(records)} | {100.0 * index / max(len(records), 1):5.1f}%] "
                f"CACHED {record.regime_id}",
                flush=True,
            )
            continue
        print(
            f"[reference {index}/{len(records)} | {100.0 * (index - 1) / max(len(records), 1):5.1f}%] "
            f"SOLVING {record.regime_id} (M={spatial_steps}, N={time_steps})",
            flush=True,
        )
        arrays = _solve_reference(record, spatial_steps, time_steps)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
        paths.append(path)
        elapsed = perf_counter() - started
        remaining = elapsed / index * (len(records) - index)
        print(
            f"[reference {index}/{len(records)} | {100.0 * index / max(len(records), 1):5.1f}%] "
            f"COMPLETE elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )
    manifest = {
        "method": "DIRK+Policy Iteration+sinh strike-concentrated grid",
        "spatial_steps": spatial_steps,
        "time_steps": time_steps,
        "splits": list(split_names),
        "requested_regime_ids": None if selected_ids is None else list(selected_ids),
        "regimes": len(paths),
        "lcp_tolerance": 1e-12,
        "files": [path.name for path in paths],
    }
    (output / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def interpolate_reference(
    reference_path: Path | str,
    coordinates: np.ndarray,
) -> dict[str, np.ndarray]:
    reference = np.load(reference_path)
    query = np.asarray(coordinates, dtype=float)
    m = np.exp(query[:, 0])
    s = query[:, 1]
    result = {}
    for name in ("value_over_k", "delta", "scaled_gamma"):
        result[name] = _interpolate_surface(
            reference["moneyness_grid"],
            reference["normalized_time_grid"],
            reference[name],
            m,
            s,
        )
    source_boundary = reference["boundary_over_k"]
    finite_boundary = np.isfinite(source_boundary)
    if np.count_nonzero(finite_boundary) >= 2:
        result["boundary_over_k"] = np.interp(
            s,
            reference["normalized_time_grid"][finite_boundary],
            source_boundary[finite_boundary],
            left=np.nan,
            right=np.nan,
        )
    else:
        result["boundary_over_k"] = np.full(len(s), np.nan, dtype=float)
    return result


def interpolate_grid_surface(
    source_m: np.ndarray,
    source_s: np.ndarray,
    surface: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Interpolate a normalized surface at ``(log-moneyness, normalized-time)``."""

    query = np.asarray(coordinates, dtype=float)
    return _interpolate_surface(
        np.asarray(source_m, dtype=float),
        np.asarray(source_s, dtype=float),
        np.asarray(surface, dtype=float),
        np.exp(query[:, 0]),
        query[:, 1],
    )


def _solve_reference(record: RegimeRecord, spatial_steps: int, time_steps: int) -> dict[str, np.ndarray]:
    config = AmericanLCPConfig(
        record.option_type,
        record.K,
        record.T,
        record.r,
        record.q,
        record.sigma,
        record.Smax,
        spatial_steps,
        time_steps,
        tolerance=1e-12,
        obstacle_tolerance=1e-12,
    )
    grid = sinh_spot_grid(config.Smax, config.K, spatial_steps)
    result = american_dirk_policy_price(config, spot_grid=grid)
    if not result.converged:
        raise RuntimeError(f"high-accuracy reference failed for {record.regime_id}")
    delta = np.vstack(
        [finite_difference_delta_nonuniform(result.spot_grid, row) for row in result.value_grid]
    )
    gamma = np.vstack(
        [finite_difference_gamma_nonuniform(result.spot_grid, row) for row in result.value_grid]
    )
    payoff_function = call_payoff if record.option_type == "call" else put_payoff
    payoff = np.asarray(payoff_function(result.spot_grid, record.K), dtype=float)
    premium = result.value_grid - payoff[np.newaxis, :]
    boundaries = np.full(len(result.tau_grid), np.nan, dtype=float)
    for index, tau in enumerate(result.tau_grid):
        point = extract_boundary_at_time(
            result.spot_grid,
            premium[index],
            record.option_type,
            float(tau),
            index,
            threshold=1e-6,
        )
        if point.boundary_found:
            boundaries[index] = point.boundary_spot / record.K
    return {
        "moneyness_grid": result.spot_grid / record.K,
        "normalized_time_grid": result.tau_grid / record.T,
        "value_over_k": result.value_grid / record.K,
        "delta": delta,
        "scaled_gamma": record.K * gamma,
        "boundary_over_k": boundaries,
    }


def _interpolate_surface(
    source_m: np.ndarray,
    source_s: np.ndarray,
    surface: np.ndarray,
    query_m: np.ndarray,
    query_s: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(query_m), dtype=float)
    upper_index = np.searchsorted(source_s, query_s, side="right")
    upper_index = np.clip(upper_index, 1, len(source_s) - 1)
    lower_index = upper_index - 1
    denominator = source_s[upper_index] - source_s[lower_index]
    weight = np.divide(
        query_s - source_s[lower_index],
        denominator,
        out=np.zeros_like(query_s),
        where=denominator > 0.0,
    )
    for index in range(len(query_m)):
        lower_value = np.interp(query_m[index], source_m, surface[lower_index[index]])
        upper_value = np.interp(query_m[index], source_m, surface[upper_index[index]])
        output[index] = lower_value + weight[index] * (upper_value - lower_value)
    return output
