"""Frozen data, split, architecture, and gate protocol for Experiments 52--57."""

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
from american_risk_surfaces.workspace import (
    frozen_input,
    populated_input_directory,
    portable_path,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "12_positive_premium_deeponet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "14_positive_premium_deeponet"
SOURCE_SNAPSHOT_DIR = populated_input_directory(
    "results/09_reduced_basis_vi/01_snapshots/train_only", "*/*.npz"
)
BASIS_RESULTS_DIR = frozen_input("results/11_positive_premium_basis_operator")
VALIDATION_CACHE_DIR = BASIS_RESULTS_DIR / "02_validation_cache"
BASIS_VALIDATION_METRICS = (
    BASIS_RESULTS_DIR / "05_five_seed_validation" / "five_seed_validation_metrics.csv"
)

LATENT_RANKS = (32, 64, 128)
ARMS = ("N0", "N1", "N2")
SEEDS = (17, 29, 43, 71, 101)
DEVELOPMENT_STEPS = 6000
FORMAL_STEPS = 12000
REDUCTION_RMSE_LIMIT = 4.94989e-4
TOTAL_RMSE_FLOOR = 0.002474946
BOUNDARY_LIMIT = 0.066667
BOUNDARY_CLASSICAL_LIMIT = 0.033333
PREMIUM_THRESHOLD = 1e-6
LCP_TOLERANCE = 1e-12
FB_EPSILON = 1e-12
PROTOCOL_NAME = "surf_positive_premium_deeponet_v1"


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
        raise PermissionError("DeepONet training accepts train-only family snapshots")


def assert_training_regime_allowed(split: str, option_type: str, q: float) -> None:
    if split != "train":
        raise PermissionError("DeepONet training is restricted to train regimes")
    if option_type == "call" and q <= 0.0:
        raise PermissionError("q=0 calls use the analytic control branch")


@lru_cache(maxsize=1)
def protocol_core_payload() -> dict[str, object]:
    paths = train_snapshot_paths("put") + train_snapshot_paths("call")
    implementation_paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    experiment_paths = [
        PROJECT_ROOT / "experiments" / f"{number}_{name}.py"
        for number, name in (
            (52, "deeponet_protocol_and_data_audit"),
            (53, "deeponet_development"),
            (54, "deeponet_five_seed_validation"),
            (55, "deeponet_heldout_prediction_and_scoring"),
            (56, "deeponet_hybrid_and_runtime"),
            (57, "deeponet_synthesis"),
        )
    ]
    required = [REGIME_MANIFEST, SPLIT_MANIFEST, BASIS_VALIDATION_METRICS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required frozen inputs are missing: {missing}")
    return {
        "protocol": PROTOCOL_NAME,
        "method": "parameter-conditioned positive-premium DeepONet",
        "grid": {"K": 1.0, "Smax": 4.0, "M": 120, "N": 120},
        "learned_slice": {
            "positive_time": [1, 120], "interior_spot": [1, 119], "size": 14280,
        },
        "features": ["log(T)", "sigma", "r", "q"],
        "coordinates": ["m=S/K mapped to [-1,1]", "s=tau/T mapped to [-1,1]"],
        "architecture": {
            "branch": "4-128-128-128-p SiLU",
            "trunk": "2-128-128-128-128-p SiLU",
            "contraction": "c + sum(branch*trunk)/sqrt(p)",
            "latent_ranks": list(LATENT_RANKS),
        },
        "arms": {
            "N0": "normalized projected-premium surface loss",
            "N1": "N0 plus boundary, spatial derivative, and balanced exercise BCE",
            "N2": "N1 plus differentiable discrete CN-LCP Fischer-Burmeister loss",
            "H": "selected DeepONet projected initializer plus strict Policy Iteration",
        },
        "training": {
            "regime_batch": 8,
            "development_steps": DEVELOPMENT_STEPS,
            "formal_steps": FORMAL_STEPS,
            "seeds": list(SEEDS),
            "optimizer": "AdamW lr=1e-3 weight_decay=1e-6 cosine to 1e-5",
            "dtype": "float64",
            "formal_seed_budget_seconds": 3600,
        },
        "q0_call": "European BSM analytic control; neural extrapolation OOD-only after scoring",
        "split_counts": {
            split: len(load_regimes(splits=(split,)))
            for split in ("train", "validation", "test", "stress_holdout")
        },
        "train_family_counts": {
            family: len(train_snapshot_paths(family)) for family in ("put", "call")
        },
        "benchmarks": {
            "historical": "CN+PSOR",
            "strengthened_1": "CN+Policy Iteration",
            "strengthened_2": "CN+Projected LU",
            "learned_validation_only": "frozen 12-mode P2 basis operator",
            "high_reference": "DIRK+Policy+sinh M=480,N=960",
        },
        "thresholds": {
            "reduction_rmse": REDUCTION_RMSE_LIMIT,
            "total_rmse_floor": TOTAL_RMSE_FLOOR,
            "boundary": BOUNDARY_LIMIT,
            "boundary_classical": BOUNDARY_CLASSICAL_LIMIT,
            "exercise_f1": 0.98,
            "delta_gamma_ratio": 1.25,
            "lcp_oracle_ratio": 1.05,
            "obstacle": 1e-12,
            "lcp_solver": LCP_TOLERANCE,
        },
        "frozen_inputs": {
            "regime_manifest": sha256_file(REGIME_MANIFEST),
            "split_manifest": sha256_file(SPLIT_MANIFEST),
            "basis_validation_metrics": sha256_file(BASIS_VALIDATION_METRICS),
            "snapshot_hashes": [
                {"path": portable_path(path), "sha256": sha256_file(path)}
                for path in paths
            ],
            "implementation_hashes": [
                {
                    "path": portable_path(path),
                    "sha256": sha256_file(path),
                }
                for path in [*implementation_paths, *experiment_paths]
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
        dirty_paths = subprocess.run(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
    except Exception:
        dirty_paths = ["unavailable"]
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unavailable"
    payload = protocol_core_payload()
    payload["protocol_hash"] = protocol_hash()
    payload["environment"] = {
        "git_commit": commit,
        "git_worktree_clean": not bool(dirty_paths),
        "git_dirty_entry_count": len(dirty_paths),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
        "platform": platform.platform(),
    }
    path = output / "protocol_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
