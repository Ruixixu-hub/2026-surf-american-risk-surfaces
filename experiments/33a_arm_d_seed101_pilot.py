"""One-seed Arm D held-out pilot for a declared limited-compute study.

This is deliberately isolated from the registered five-seed held-out study.  The
seed is selected from validation before held-out labels are opened, and the
output records that selection so the pilot cannot be mistaken for an unbiased
five-seed result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.protocol import (
    HELDOUT_SPLITS,
    RESULTS_DIR,
    SEEDS,
    build_job_manifest,
)
from american_risk_surfaces.pinn.reference import generate_reference_cache
from american_risk_surfaces.pinn.scoring import (
    score_checkpoints_once,
    score_classical_baselines,
)
from american_risk_surfaces.pinn.study import run_training_jobs


PILOT_ROOT = RESULTS_DIR / "04_heldout_pilots" / "arm_d_seed101"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "reference", "score", "all"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of the frozen seeds: {SEEDS}")

    frozen_path = RESULTS_DIR / "03_validation_gates" / "frozen_pinn_configuration.json"
    if not frozen_path.exists():
        raise RuntimeError("validation has not frozen a GO Arm D configuration")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    output = RESULTS_DIR / "04_heldout_pilots" / f"arm_d_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    _write_pilot_protocol(output, frozen_path, frozen, args.seed)

    if args.phase in {"train", "all"}:
        _train(output, frozen, args.seed, args.device, args.resume)
    if args.phase in {"reference", "all"}:
        _reference(output)
    if args.phase in {"score", "all"}:
        _score(output, args.seed, args.device)


def _train(
    output: Path,
    frozen: dict[str, Any],
    seed: int,
    device: str,
    resume: bool,
) -> None:
    rows = run_training_jobs(
        arms=("D",),
        splits=HELDOUT_SPLITS,
        output_dir=output / "training",
        device=device,
        resume=resume,
        seeds=(seed,),
        adam_steps=int(frozen["adam_steps"]),
        lbfgs_max_evaluations=int(frozen["lbfgs_max_evaluations"]),
        d_variant=str(frozen["d_variant"]),
        network_spec=_network_spec(str(frozen["architecture"])),
    )
    print(json.dumps({"pilot": "Arm D single seed", "seed": seed, "jobs": len(rows)}, indent=2))


def _reference(output: Path) -> None:
    paths = generate_reference_cache(
        output / "high_accuracy_reference",
        splits=HELDOUT_SPLITS,
        spatial_steps=480,
        time_steps=960,
    )
    print(json.dumps({"reference_regimes": len(paths)}, indent=2))


def _score(output: Path, seed: int, device: str) -> None:
    marker = output / "SCORING_HAS_STARTED.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            raise RuntimeError("pilot scoring is already complete; rescoring is forbidden")

    statuses = _all_status_rows(output / "training")
    expected = build_job_manifest(arms=("D",), splits=HELDOUT_SPLITS, seeds=(seed,))
    expected_keys = {
        (row["arm"], row["split"], row["regime_id"], str(row["seed"])) for row in expected
    }
    actual_keys = {
        (row["arm"], row["split"], row["regime_id"], str(row["seed"])) for row in statuses
    }
    if actual_keys != expected_keys or len(statuses) != len(actual_keys):
        raise RuntimeError(
            f"pilot requires all 67 unique Arm D jobs before scoring; found {len(actual_keys)}"
        )
    noncomplete = [row for row in statuses if row.get("status") != "COMPLETE"]
    if noncomplete:
        raise RuntimeError(f"pilot has {len(noncomplete)} non-COMPLETE jobs; scoring is blocked")

    reference_dir = output / "high_accuracy_reference"
    if len(list(reference_dir.glob("*.npz"))) != 67:
        raise RuntimeError("pilot scoring requires all 67 high-accuracy references")
    marker.write_text(
        json.dumps(
            {
                "status": "IN_PROGRESS",
                "seed": seed,
                "terminal_jobs": 67,
                "scope": "VALIDATION_SELECTED_SINGLE_SEED_PILOT",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pinn_rows = score_checkpoints_once(
        statuses,
        output_dir=output / "scoring" / "pinn",
        device=device,
        reference_dir=reference_dir,
    )
    classical_rows = score_classical_baselines(
        splits=HELDOUT_SPLITS,
        reference_dir=reference_dir,
        output_dir=output / "scoring" / "classical",
    )
    summary = _pilot_summary(pinn_rows, classical_rows, statuses, seed)
    (output / "single_seed_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "seed": seed,
                "terminal_jobs": 67,
                "scope": "VALIDATION_SELECTED_SINGLE_SEED_PILOT",
                "warning": "Not the registered five-seed held-out study.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _pilot_summary(
    pinn_rows: Iterable[dict[str, Any]],
    classical_rows: Iterable[dict[str, Any]],
    statuses: Iterable[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    all_rows = [row for row in [*pinn_rows, *classical_rows] if row["region"] == "all"]
    metrics: dict[str, dict[str, float]] = {}
    for arm in ("A", "B", "D"):
        selected = [row for row in all_rows if row["arm"] == arm]
        metrics[arm] = {
            field: float(median(float(row[field]) for row in selected))
            for field in ("rmse", "mae", "max_abs_error", "delta_rmse", "scaled_gamma_rmse")
        }
        if arm in {"A", "B"}:
            metrics[arm]["runtime_seconds"] = float(
                median(float(row["runtime_seconds"]) for row in selected)
            )
        else:
            metrics[arm]["vi_p95"] = float(median(float(row["vi_p95"]) for row in selected))
            metrics[arm]["boundary_found_rate"] = float(
                median(float(row["boundary_found_rate"]) for row in selected)
            )
    status_list = list(statuses)
    metrics["D"]["training_seconds"] = float(
        median(float(row["training_seconds"]) for row in status_list)
    )
    return {
        "status": "COMPLETE_SINGLE_SEED_PILOT",
        "evidence_scope": "67 test/stress regimes; Arm D only; one validation-selected seed",
        "seed": seed,
        "selection_disclosure": (
            "Seed 101 was selected before held-out scoring because it had the lowest Arm D "
            "validation median price RMSE. This is not a random-seed robustness estimate."
        ),
        "metrics": metrics,
        "price_rmse_ratio_D_over_B": metrics["D"]["rmse"] / metrics["B"]["rmse"],
        "warning": "Do not describe this pilot as the registered five-seed held-out experiment.",
    }


def _all_status_rows(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("training_status_shard_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _write_pilot_protocol(
    output: Path,
    frozen_path: Path,
    frozen: dict[str, Any],
    seed: int,
) -> None:
    payload = {
        "protocol": "surf_pinn_cde_v1_single_seed_pilot",
        "evidence_scope": "PILOT_NOT_FORMAL_FIVE_SEED_HELDOUT",
        "arms": ["D"],
        "seed": seed,
        "regimes": 67,
        "splits": list(HELDOUT_SPLITS),
        "selection_basis": (
            "Validation-selected seed: seed 101 had the lowest Arm D median price RMSE "
            "among the five registered validation seeds."
        ),
        "compute_budget_reason": "One-day GPU availability",
        "frozen_configuration": frozen,
        "frozen_configuration_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "scoring_rule": "Score once after all 67 jobs finish; never switch seed after scoring.",
    }
    path = output / "pilot_protocol_amendment.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("pilot protocol amendment changed; refusing to continue")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _network_spec(name: str) -> NetworkSpec:
    specifications = {
        "resnet_4x2x50": NetworkSpec("resnet", width=50, blocks=4, layers_per_block=2),
        "resnet_4x2x64": NetworkSpec("resnet", width=64, blocks=4, layers_per_block=2),
        "mlp_4x64": NetworkSpec("mlp", width=64, hidden_layers=4),
        "mlp_6x64": NetworkSpec("mlp", width=64, hidden_layers=6),
    }
    try:
        return specifications[name]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen architecture: {name}") from exc


if __name__ == "__main__":
    main()
