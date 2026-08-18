"""Label-free orchestration for SURF PINN training jobs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.pinn.evaluation import predict_pinn_surface
from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.protocol import SEEDS, RegimeRecord, build_job_manifest
from american_risk_surfaces.pinn.training import (
    PINNTrainingConfig,
    train_single_regime_pinn,
    training_config_hash,
)


def run_training_jobs(
    *,
    arms: Iterable[str],
    splits: Iterable[str],
    output_dir: Path | str,
    device: str,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = False,
    seeds: Iterable[int] = SEEDS,
    regime_ids: Iterable[str] | None = None,
    adam_steps: int = 40000,
    lbfgs_max_evaluations: int = 2000,
    max_seconds: float = 3600.0,
    d_variant: str = "etc_fb_adaptive",
    network_spec: NetworkSpec = NetworkSpec(),
) -> list[dict[str, Any]]:
    """Run jobs using only regime parameters and physics; no label path is imported."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = build_job_manifest(
        arms=arms,
        splits=splits,
        seeds=seeds,
        regime_ids=regime_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    job_path = output / f"job_manifest_shard_{shard_index:03d}_of_{shard_count:03d}.json"
    job_path.write_text(json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    study_started = perf_counter()
    for job_index, job in enumerate(jobs, start=1):
        record = RegimeRecord(**job["parameters"])
        print(
            f"[job {job_index}/{len(jobs)} | "
            f"{100.0 * (job_index - 1) / max(len(jobs), 1):5.1f}%] "
            f"START arm={job['arm']} split={record.split} "
            f"regime={record.regime_id} seed={job['seed']}",
            flush=True,
        )
        config = PINNTrainingConfig(
            arm=job["arm"],
            seed=job["seed"],
            device=device,
            adam_steps=adam_steps,
            lbfgs_max_evaluations=lbfgs_max_evaluations,
            max_seconds=max_seconds,
            d_variant=d_variant,
            network_spec=network_spec,
            checkpoint_interval=min(2000, max(1, adam_steps)),
            gradient_log_interval=min(200, max(1, adam_steps)),
            adaptive_interval=min(2000, max(1, adam_steps)),
        )
        run_dir = output / job["arm"] / record.split / record.regime_id / f"seed_{job['seed']}"
        result = train_single_regime_pinn(
            record.problem(), config, output_dir=run_dir, resume=resume
        )
        prediction_path = run_dir / "prediction_surface.npz"
        if result.status == "COMPLETE" and not prediction_path.exists():
            prediction = predict_pinn_surface(
                result.checkpoint_path,
                np.linspace(0.0, record.Smax, record.M + 1),
                np.linspace(0.0, record.T, record.N + 1),
                K=record.K,
                device=device,
            )
            temporary = prediction_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    spot_grid=prediction.spot_grid,
                    tau_grid=prediction.tau_grid,
                    value_grid=prediction.value_grid,
                    inference_seconds=prediction.inference_seconds,
                    transfer_seconds=prediction.transfer_seconds,
                )
            os.replace(temporary, prediction_path)
        row = {
            **{key: job[key] for key in ("arm", "split", "regime_id", "seed")},
            **asdict(result),
            "variant": d_variant if job["arm"] == "D" else "soft_lcp",
            "network_spec": json.dumps(network_spec.to_dict(), sort_keys=True),
            "prediction_path": str(prediction_path) if prediction_path.exists() else "",
            "config_hash": training_config_hash(record.problem(), config),
        }
        row["checkpoint_path"] = str(row["checkpoint_path"])
        rows.append(row)
        _write_csv(
            output
            / f"training_status_shard_{shard_index:03d}_of_{shard_count:03d}.csv",
            rows,
        )
        elapsed = perf_counter() - study_started
        remaining = elapsed / job_index * (len(jobs) - job_index)
        print(
            f"[job {job_index}/{len(jobs)} | "
            f"{100.0 * job_index / max(len(jobs), 1):5.1f}%] "
            f"{result.status} training={_format_duration(result.training_seconds)} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
