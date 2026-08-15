"""Frozen protocol and leakage rules for Experiments 42--45."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from american_risk_surfaces.reduced_order.protocol import (
    PROJECT_ROOT,
    REGIME_MANIFEST,
    SPLIT_MANIFEST,
    load_regimes,
    sha256_file,
)
from american_risk_surfaces.workspace import populated_input_directory, portable_path


RESULTS_DIR = PROJECT_ROOT / "results" / "10_boundary_aligned_localized_basis"
REPORTS_DIR = PROJECT_ROOT / "reports" / "12_boundary_aligned_localized_basis"
SOURCE_SNAPSHOT_DIR = populated_input_directory(
    "results/09_reduced_basis_vi/01_snapshots/train_only", "*/*.npz"
)
DIMENSION_LADDER = (4, 8, 12, 16, 24, 32)
LOCAL_BIN_LADDER = (2, 4, 8)
BOUNDARY_THRESHOLD = 1e-6
CANONICAL_POINT_LADDER = (1921, 3841)
MAX_STORED_PRIMAL_MODES = 128
MAX_STORED_DUAL_GENERATORS = 128
REDUCTION_RMSE_LIMIT = 4.94989e-4
TOTAL_RMSE_FLOOR = 0.002474946


def train_snapshot_paths(option_type: str) -> list[Path]:
    family = str(option_type).lower()
    paths = sorted((SOURCE_SNAPSHOT_DIR / family).glob("*.npz"))
    expected = {
        item.regime_id for item in load_regimes(splits=("train",), option_type=family)
    }
    actual = {path.stem for path in paths}
    if actual != expected:
        raise RuntimeError(
            f"train snapshot mismatch for {family}: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    return paths


def assert_oracle_regime_allowed(split: str, option_type: str, q: float) -> None:
    if split not in {"train", "validation"}:
        raise PermissionError("oracle basis code is statically restricted to train/validation")
    if option_type == "call" and q == 0.0:
        raise PermissionError("no-dividend calls remain sealed in this experiment")


def protocol_core_payload() -> dict[str, object]:
    snapshots = train_snapshot_paths("put") + train_snapshot_paths("call")
    return {
        "protocol": "surf_oracle_boundary_aligned_localized_basis_v1",
        "question": "moving-boundary representability falsification; not an online solver",
        "allowed_splits": ["train", "validation"],
        "sealed": ["test", "stress_holdout", "q=0 call"],
        "source_snapshot_count": len(snapshots),
        "source_snapshot_hashes": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in snapshots
        ],
        "frozen_input_hashes": {
            "regime_manifest": sha256_file(REGIME_MANIFEST),
            "split_manifest": sha256_file(SPLIT_MANIFEST),
        },
        "boundary_threshold": BOUNDARY_THRESHOLD,
        "canonical_point_ladder": list(CANONICAL_POINT_LADDER),
        "interpolation": "PCHIP",
        "arms": ["U", "A", "L", "AL"],
        "dimension_ladder": list(DIMENSION_LADDER),
        "local_bin_ladder": list(LOCAL_BIN_LADDER),
        "budgets": {
            "active_dimension": 32,
            "stored_primal_modes": MAX_STORED_PRIMAL_MODES,
            "stored_dual_generators": MAX_STORED_DUAL_GENERATORS,
        },
        "lcp_tolerance": 1e-12,
    }


def protocol_hash() -> str:
    raw = json.dumps(protocol_core_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_protocol() -> Path:
    output = RESULTS_DIR / "00_protocol"
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unavailable"
    payload = protocol_core_payload()
    payload["protocol_hash"] = protocol_hash()
    payload["environment"] = {
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    path = output / "protocol_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
