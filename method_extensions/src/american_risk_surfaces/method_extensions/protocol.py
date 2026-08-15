"""Frozen protocol and reproducibility utilities for method extensions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.workspace import EXTENSION_ROOT, frozen_input, portable_path


PROJECT_ROOT = EXTENSION_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "07_method_extensions" / "00_protocol"
DATASET_DIR = frozen_input("results/04_surrogate_dataset/v1_small_grid")
STAGE3_DIR = frozen_input("results/05_surrogate_models/price_premium")

AUDIT_REGIME_IDS = (
    "put_T100_s020_r005_q003",
    "call_T100_s020_r005_q006",
    "call_T100_s020_r005_q000",
    "put_T025_s020_r001_q000",
    "call_T025_s020_r001_q000",
    "put_T200_s060_r010_q010",
    "call_T200_s060_r010_q010",
    "put_T200_s020_r001_q010",
    "call_T200_s020_r010_q010",
    "put_T050_s060_r005_q003",
    "call_T050_s060_r005_q010",
    "call_T100_s060_r001_q006",
)

FROZEN_INPUTS = (
    DATASET_DIR / "dataset_v1_small_grid.npz",
    DATASET_DIR / "regime_manifest.csv",
    DATASET_DIR / "split_assignment.csv",
    DATASET_DIR / "schema_snapshot.csv",
    STAGE3_DIR / "model_run_manifest.csv",
    STAGE3_DIR / "surrogate_metrics_by_split.csv",
    STAGE3_DIR / "obstacle_violation_summary.csv",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def environment_manifest() -> dict[str, Any]:
    versions: dict[str, str] = {"numpy": np.__version__}
    distribution_names = {
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "matplotlib": "matplotlib",
    }
    for package, distribution in distribution_names.items():
        try:
            versions[package] = metadata.version(distribution)
        except Exception:
            versions[package] = "unavailable"
    return {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "float_precision": "float64",
        "versions": versions,
    }


def frozen_input_rows(paths: Iterable[Path] = FROZEN_INPUTS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        rows.append(
            {
                "path": portable_path(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "missing",
            }
        )
    return rows


def write_protocol_manifest(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_rows = frozen_input_rows()
    checksums_path = output / "frozen_input_checksums.csv"
    with checksums_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "exists", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(input_rows)

    protocol = {
        "protocol_version": "method_extensions_v1",
        "dataset_scope": "frozen_288_regime_v1_small_grid",
        "split_policy": "immutable_regime_level_splits",
        "gamma_status": "BLOCKED",
        "candidate_lcp_tolerances": [1e-8, 1e-10, 1e-12],
        "target_normalized_lcp_tolerance": 1e-10,
        "target_normalized_obstacle_tolerance": 1e-12,
        "audit_regime_ids": list(AUDIT_REGIME_IDS),
        "frozen_inputs": input_rows,
        "environment": environment_manifest(),
    }
    manifest_path = output / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest_path, "checksums": checksums_path}
