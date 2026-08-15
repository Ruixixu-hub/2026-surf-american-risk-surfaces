"""Frozen data, split, and benchmark protocol for Experiments 46--51."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from functools import lru_cache
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


RESULTS_DIR = PROJECT_ROOT / "results" / "11_positive_premium_basis_operator"
REPORTS_DIR = PROJECT_ROOT / "reports" / "13_positive_premium_basis_operator"
SOURCE_SNAPSHOT_DIR = populated_input_directory(
    "results/09_reduced_basis_vi/01_snapshots/train_only", "*/*.npz"
)
DIMENSION_LADDER = (4, 8, 12, 16, 24, 32)
SEEDS = (17, 29, 43, 71, 101)
REDUCTION_RMSE_LIMIT = 4.94989e-4
TOTAL_RMSE_FLOOR = 0.002474946
BOUNDARY_LIMIT = 0.066667
BOUNDARY_CLASSICAL_LIMIT = 0.033333
PREMIUM_THRESHOLD = 1e-6
PROTOCOL_NAME = "surf_positive_premium_pod_basis_operator_v1"


def train_snapshot_paths(option_type: str) -> list[Path]:
    family = str(option_type).lower()
    if family not in {"put", "call"}:
        raise ValueError("option_type must be put or call")
    paths = sorted((SOURCE_SNAPSHOT_DIR / family).glob("*.npz"))
    expected = {
        item.regime_id
        for item in load_regimes(splits=("train",), option_type=family)
        if family == "put" or item.q > 0.0
    }
    actual = {path.stem for path in paths}
    if actual != expected:
        raise RuntimeError(
            f"train snapshot mismatch for {family}: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    return paths


def assert_training_snapshot_allowed(path: Path | str, option_type: str) -> None:
    source = Path(path).resolve()
    allowed = SOURCE_SNAPSHOT_DIR.resolve() / str(option_type).lower()
    if source.parent != allowed or source.suffix != ".npz":
        raise PermissionError("basis/operator training accepts train-only snapshot paths")


def assert_mapping_regime_allowed(split: str, option_type: str, q: float) -> None:
    if split not in {"train", "validation"}:
        raise PermissionError("mapping development is restricted to train/validation")
    if option_type == "call" and q <= 0.0:
        raise PermissionError("q=0 calls use the analytic control and are not training labels")


@lru_cache(maxsize=1)
def protocol_core_payload() -> dict[str, object]:
    paths = train_snapshot_paths("put") + train_snapshot_paths("call")
    split_counts = {
        split: len(load_regimes(splits=(split,)))
        for split in ("train", "validation", "test", "stress_holdout")
    }
    family_counts = {
        family: len(train_snapshot_paths(family)) for family in ("put", "call")
    }
    return {
        "protocol": PROTOCOL_NAME,
        "method": "positive-premium mean-centered Euclidean POD basis operator",
        "grid": {"K": 1.0, "Smax": 4.0, "M": 120, "N": 120},
        "learned_slice": {"positive_time": [1, 120], "interior_spot": [1, 119], "size": 14280},
        "premium": "(U-Phi)/K; hard max(raw,0) for the formal output",
        "families": "separate put and dividend-call basis/network",
        "q0_call": "European BSM analytic control; raw network is heldout OOD diagnostic only",
        "dimension_ladder": list(DIMENSION_LADDER),
        "seeds": list(SEEDS),
        "split_counts": split_counts,
        "train_family_counts": family_counts,
        "benchmarks": {
            "historical": "CN+PSOR",
            "strong": "CN+Policy Iteration",
            "high_reference": "DIRK+Policy+sinh M=480,N=960",
        },
        "thresholds": {
            "reduction_rmse": REDUCTION_RMSE_LIMIT,
            "total_rmse_floor": TOTAL_RMSE_FLOOR,
            "boundary": BOUNDARY_LIMIT,
            "premium_exercise": PREMIUM_THRESHOLD,
            "lcp": 1e-12,
        },
        "frozen_inputs": {
            "regime_manifest": sha256_file(REGIME_MANIFEST),
            "split_manifest": sha256_file(SPLIT_MANIFEST),
            "snapshot_count": len(paths),
            "snapshot_hashes": [
                {"path": portable_path(path), "sha256": sha256_file(path)}
                for path in paths
            ],
        },
    }


@lru_cache(maxsize=1)
def protocol_hash() -> str:
    raw = json.dumps(protocol_core_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_protocol() -> Path:
    output = RESULTS_DIR / "00_protocol"
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        commit = "unavailable"
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unavailable"
    payload = protocol_core_payload()
    payload["protocol_hash"] = protocol_hash()
    payload["environment"] = {
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
        "platform": platform.platform(),
    }
    path = output / "protocol_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
