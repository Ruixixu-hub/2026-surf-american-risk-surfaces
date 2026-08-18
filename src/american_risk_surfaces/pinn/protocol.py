"""Frozen manifests, label-free jobs, and reproducibility controls for PINN C/D/E."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.pinn.formulation import PINNProblem


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "results" / "04_surrogate_dataset" / "v1_small_grid"
RESULTS_DIR = PROJECT_ROOT / "results" / "08_pinn_gap"
REPORTS_DIR = PROJECT_ROOT / "reports" / "10_pinn_gap"
REGIME_MANIFEST = DATASET_DIR / "regime_manifest.csv"
SPLIT_MANIFEST = DATASET_DIR / "split_assignment.csv"
DATASET_PATH = DATASET_DIR / "dataset_v1_small_grid.npz"
SEEDS = (17, 29, 43, 71, 101)
HELDOUT_SPLITS = ("test", "stress_holdout")
DEVELOPMENT_REGIME_IDS = (
    "put_T025_s020_r001_q000",
    "put_T200_s060_r010_q006",
    "call_T025_s020_r001_q003",
    "call_T200_s060_r010_q006",
    "put_T050_s020_r005_q000",
    "put_T050_s060_r005_q006",
    "call_T050_s020_r005_q010",
    "call_T050_s060_r005_q006",
)


@dataclass(frozen=True)
class RegimeRecord:
    regime_id: str
    option_type: str
    T: float
    sigma: float
    r: float
    q: float
    K: float
    Smax: float
    M: int
    N: int
    split: str

    def problem(self) -> PINNProblem:
        return PINNProblem(
            regime_id=self.regime_id,
            option_type=self.option_type,
            T=self.T,
            r=self.r,
            q=self.q,
            sigma=self.sigma,
        )


def load_regime_records(
    *,
    splits: Iterable[str] | None = None,
    regime_ids: Iterable[str] | None = None,
) -> list[RegimeRecord]:
    """Load parameter metadata only; this function never opens the label NPZ."""

    allowed_splits = None if splits is None else set(splits)
    allowed_ids = None if regime_ids is None else set(regime_ids)
    with REGIME_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        RegimeRecord(
            regime_id=row["regime_id"],
            option_type=row["option_type"],
            T=float(row["T"]),
            sigma=float(row["sigma"]),
            r=float(row["r"]),
            q=float(row["q"]),
            K=float(row["K"]),
            Smax=float(row["Smax"]),
            M=int(row["M"]),
            N=int(row["N"]),
            split=row["split"],
        )
        for row in rows
        if (allowed_splits is None or row["split"] in allowed_splits)
        and (allowed_ids is None or row["regime_id"] in allowed_ids)
    ]
    return sorted(records, key=lambda item: item.regime_id)


def build_job_manifest(
    *,
    arms: Iterable[str],
    splits: Iterable[str],
    seeds: Iterable[int] = SEEDS,
    regime_ids: Iterable[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard controls must satisfy 0 <= index < count.")
    jobs = [
        {
            "arm": arm,
            "split": record.split,
            "regime_id": record.regime_id,
            "seed": int(seed),
            "parameters": asdict(record),
        }
        for arm in sorted(set(arms))
        for record in load_regime_records(splits=splits, regime_ids=regime_ids)
        for seed in sorted(set(int(value) for value in seeds))
    ]
    jobs.sort(key=lambda row: (row["arm"], row["split"], row["regime_id"], row["seed"]))
    return [row for index, row in enumerate(jobs) if index % shard_count == shard_index]


def write_protocol_manifest(output_dir: Path | str = RESULTS_DIR / "00_protocol") -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = (REGIME_MANIFEST, SPLIT_MANIFEST, DATASET_PATH)
    input_rows = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in inputs
    ]
    split_counts: dict[str, int] = {}
    for record in load_regime_records():
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    protocol = {
        "protocol_version": "surf_pinn_cde_v1",
        "coordinates": {"m": "S/K", "x": "log(S/K)", "s": "tau/T", "u": "U/K"},
        "domain": {"m_min": 1e-4, "m_max": 4.0, "s_min": 0.0, "s_max": 1.0},
        "operator": "0.5*sigma^2*u_xx + (r-q-0.5*sigma^2)*u_x - r*u",
        "equation_gap": "rho=u_s-T*L_x(u)",
        "vi_sign": "g=u-phi>=0; rho>=0; g*rho=0",
        "lcp_tolerance": 1e-12,
        "seeds": list(SEEDS),
        "split_counts": split_counts,
        "development_regime_ids": list(DEVELOPMENT_REGIME_IDS),
        "heldout_policy": "train physics without labels; score once after checkpoints freeze",
        "frozen_inputs": input_rows,
        "environment": environment_manifest(),
    }
    manifest_path = output / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    checksum_path = output / "frozen_input_checksums.csv"
    with checksum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(input_rows)
    label_free_path = output / "label_free_regimes.csv"
    records = load_regime_records()
    with label_free_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return {
        "manifest": manifest_path,
        "checksums": checksum_path,
        "label_free_regimes": label_free_path,
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    except Exception:
        torch_version = "unavailable"
        cuda_version = None
        cuda_available = False
        gpu = "unavailable"
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch_version,
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu": gpu,
    }
