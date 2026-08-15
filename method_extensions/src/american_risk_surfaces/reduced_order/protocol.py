"""Frozen manifests and leakage-safe regime metadata for the RB-VI study."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.workspace import EXTENSION_ROOT, frozen_input, portable_path


PROJECT_ROOT = EXTENSION_ROOT
DATASET_DIR = frozen_input("results/04_surrogate_dataset/v1_small_grid")
RESULTS_DIR = PROJECT_ROOT / "results" / "09_reduced_basis_vi"
REPORTS_DIR = PROJECT_ROOT / "reports" / "11_reduced_basis_vi"
REGIME_MANIFEST = DATASET_DIR / "regime_manifest.csv"
SPLIT_MANIFEST = DATASET_DIR / "split_assignment.csv"
DATASET_PATH = DATASET_DIR / "dataset_v1_small_grid.npz"
DIMENSION_LADDER = (4, 8, 12, 16, 24, 32)
LABEL_FLOOR = 0.0019799570496789242
REDUCTION_RMSE_LIMIT = 0.25 * LABEL_FLOOR
TOTAL_ERROR_FLOOR = 1.25 * LABEL_FLOOR


@dataclass(frozen=True)
class RBRegime:
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

    def config(self) -> AmericanLCPConfig:
        return AmericanLCPConfig(
            self.option_type,
            self.K,
            self.T,
            self.r,
            self.q,
            self.sigma,
            self.Smax,
            self.M,
            self.N,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )


def load_regimes(
    *, splits: Iterable[str] | None = None, option_type: str | None = None
) -> list[RBRegime]:
    allowed = None if splits is None else set(splits)
    with REGIME_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    regimes = [
        RBRegime(
            row["regime_id"],
            row["option_type"],
            float(row["T"]),
            float(row["sigma"]),
            float(row["r"]),
            float(row["q"]),
            float(row["K"]),
            float(row["Smax"]),
            int(row["M"]),
            int(row["N"]),
            row["split"],
        )
        for row in rows
        if (allowed is None or row["split"] in allowed)
        and (option_type is None or row["option_type"] == option_type)
    ]
    return sorted(regimes, key=lambda item: item.regime_id)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_payload() -> dict[str, object]:
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
    payload["environment"] = {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    payload["protocol_hash"] = _payload_hash(protocol_core_payload())
    return payload


def protocol_core_payload() -> dict[str, object]:
    """Return only immutable mathematical/data inputs used for artifact identity."""

    counts = {
        split: len(load_regimes(splits=(split,)))
        for split in ("train", "validation", "test", "stress_holdout")
    }
    return {
        "protocol": "surf_fd_lcp_primal_dual_rb_vi_v1",
        "fom": "CN+Policy Iteration previous-slice",
        "grid": {"K": 1.0, "Smax": 4.0, "M": 120, "N": 120},
        "lcp_tolerance": 1e-12,
        "obstacle_tolerance": 1e-12,
        "dimension_ladder": list(DIMENSION_LADDER),
        "label_floor": LABEL_FLOOR,
        "reduction_rmse_limit": REDUCTION_RMSE_LIMIT,
        "total_error_floor": TOTAL_ERROR_FLOOR,
        "split_counts": counts,
        "basis_policy": "train-only; separate put/call bases",
        "frozen_inputs": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in (REGIME_MANIFEST, SPLIT_MANIFEST, DATASET_PATH)
        ],
    }


def write_protocol(output_dir: Path | str = RESULTS_DIR / "00_protocol") -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    path = output / "protocol_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rows = [asdict(item) for item in load_regimes()]
    with (output / "regime_manifest_frozen.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def protocol_hash() -> str:
    return _payload_hash(protocol_core_payload())


def _payload_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
