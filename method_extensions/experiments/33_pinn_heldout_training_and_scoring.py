"""Experiment 33: separated label-free held-out training and one-time scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.protocol import (
    HELDOUT_SPLITS,
    RESULTS_DIR,
    build_job_manifest,
)
from american_risk_surfaces.pinn.reference import generate_reference_cache
from american_risk_surfaces.pinn.scoring import (
    decide_arm_d,
    score_checkpoints_once,
    score_classical_baselines,
)
from american_risk_surfaces.pinn.study import run_training_jobs


OUTPUT = RESULTS_DIR / "04_heldout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reference", "train", "score"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "reference":
        paths = generate_reference_cache(
            OUTPUT / "high_accuracy_reference",
            splits=HELDOUT_SPLITS,
            spatial_steps=480,
            time_steps=960,
        )
        print(json.dumps({"reference_regimes": len(paths)}, indent=2))
        return
    if args.phase == "train":
        frozen_path = RESULTS_DIR / "03_validation_gates" / "frozen_pinn_configuration.json"
        if not frozen_path.exists():
            raise RuntimeError(
                "validation gates have not frozen a GO configuration; held-out training is blocked"
            )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        rows = run_training_jobs(
            arms=("C", "D"),
            splits=HELDOUT_SPLITS,
            output_dir=OUTPUT / "training",
            device=args.device,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=args.resume,
            adam_steps=int(frozen["adam_steps"]),
            lbfgs_max_evaluations=int(frozen["lbfgs_max_evaluations"]),
            d_variant=str(frozen["d_variant"]),
            network_spec=_network_spec(str(frozen["architecture"])),
        )
        print(json.dumps({"jobs_in_shard": len(rows)}, indent=2))
        return
    marker = OUTPUT / "SCORING_HAS_STARTED.json"
    marker_state = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if marker_state and marker_state.get("status") == "COMPLETE":
        raise RuntimeError("held-out scoring is complete; the one-time lock forbids rescoring")
    rows = _all_status_rows(OUTPUT / "training")
    terminal = [row for row in rows if row["status"] in {"COMPLETE", "FAILED", "BUDGET_EXHAUSTED"}]
    aggregate = _validate_status_aggregate(terminal)
    (OUTPUT / "heldout_aggregate_manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "status": "IN_PROGRESS",
                "terminal_jobs": 670,
                "rule": "no retraining after held-out scoring; interrupted scoring may resume",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reference_dir = OUTPUT / "high_accuracy_reference"
    reference_files = list(reference_dir.glob("*.npz"))
    if len(reference_files) != 67:
        raise RuntimeError(
            f"formal scoring requires 67 DIRK/Policy reference files, found {len(reference_files)}"
        )
    metrics = score_checkpoints_once(
        terminal,
        output_dir=OUTPUT / "scoring",
        device=args.device,
        reference_dir=reference_dir,
    )
    classical_metrics = score_classical_baselines(
        splits=HELDOUT_SPLITS,
        reference_dir=reference_dir,
        output_dir=OUTPUT / "scoring",
    )
    decision = decide_arm_d(metrics, terminal)
    (OUTPUT / "arm_d_heldout_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "terminal_jobs": 670,
                "scored_complete_checkpoints": len(
                    [row for row in terminal if row["status"] == "COMPLETE"]
                ),
                "classical_metric_rows": len(classical_metrics),
                "rule": "no retraining or rescoring after held-out results are visible",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


def _all_status_rows(directory: Path) -> list[dict[str, str]]:
    rows = []
    for path in directory.glob("training_status_shard_*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _validate_status_aggregate(rows: list[dict[str, str]]) -> dict[str, object]:
    expected = build_job_manifest(arms=("C", "D"), splits=HELDOUT_SPLITS)
    expected_keys = {
        (row["arm"], row["split"], row["regime_id"], str(row["seed"])) for row in expected
    }
    actual_keys = {
        (row["arm"], row["split"], row["regime_id"], str(row["seed"])) for row in rows
    }
    if len(rows) != len(actual_keys):
        raise RuntimeError("duplicate held-out status rows detected")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"held-out job manifest mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    missing_hash = [row for row in rows if not row.get("config_hash")]
    missing_prediction = [
        row
        for row in rows
        if row["status"] == "COMPLETE"
        and (not row.get("prediction_path") or not Path(row["prediction_path"]).exists())
    ]
    if missing_hash or missing_prediction:
        raise RuntimeError(
            "aggregate has missing config hashes or COMPLETE jobs without predictions"
        )
    architecture_specs = sorted({row.get("network_spec", "") for row in rows})
    if len(architecture_specs) != 1:
        raise RuntimeError("C and D held-out jobs do not share one frozen architecture")
    return {
        "protocol": "surf_pinn_cde_v1",
        "expected_jobs": len(expected_keys),
        "terminal_jobs": len(rows),
        "complete_jobs": sum(row["status"] == "COMPLETE" for row in rows),
        "failed_jobs": sum(row["status"] == "FAILED" for row in rows),
        "budget_exhausted_jobs": sum(row["status"] == "BUDGET_EXHAUSTED" for row in rows),
        "unique_config_hashes": len({row["config_hash"] for row in rows}),
        "network_spec": architecture_specs[0],
        "prediction_files_verified": sum(row["status"] == "COMPLETE" for row in rows),
    }


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
