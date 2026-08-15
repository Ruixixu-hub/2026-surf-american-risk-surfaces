"""Experiment 32: one-way validation scoring and pre-registered C/D gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from american_risk_surfaces.pinn.protocol import (
    DEVELOPMENT_REGIME_IDS,
    RESULTS_DIR,
    load_regime_records,
)
from american_risk_surfaces.pinn.reference import generate_reference_cache
from american_risk_surfaces.pinn.scoring import (
    decide_arm_c_architecture,
    decide_arm_d,
    decide_arm_d_ablations,
    score_checkpoints_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-c-dir", type=Path, default=RESULTS_DIR / "01_arm_c" / "validation")
    parser.add_argument("--arm-d-dir", type=Path, default=RESULTS_DIR / "02_arm_d" / "validation" / "etc_fb_adaptive")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablation-dir", type=Path, default=RESULTS_DIR / "02_arm_d" / "ablation")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--generate-reference", action="store_true")
    parser.add_argument("--architecture-only", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=RESULTS_DIR / "01_arm_c" / "sensitivity",
    )
    args = parser.parse_args()
    reference_dir = args.reference_dir
    if args.generate_reference:
        reference_dir = reference_dir or RESULTS_DIR / "03_validation_gates" / "high_accuracy_reference"
        validation_ids = tuple(
            record.regime_id for record in load_regime_records(splits=("validation",))
        )
        required_ids = tuple(sorted(set(DEVELOPMENT_REGIME_IDS) | set(validation_ids)))
        generate_reference_cache(
            reference_dir,
            splits=("train", "validation"),
            spatial_steps=480,
            time_steps=960,
            regime_ids=required_ids,
        )
    output = RESULTS_DIR / "03_validation_gates"
    if args.architecture_only:
        sensitivity_rows = _status_rows((args.sensitivity_dir,))
        sensitivity_metrics = score_checkpoints_once(
            sensitivity_rows,
            output_dir=output / "architecture_sensitivity",
            device=args.device,
            reference_dir=reference_dir,
        )
        architecture_decision = decide_arm_c_architecture(sensitivity_metrics)
        (output / "arm_c_architecture_decision.json").write_text(
            json.dumps(architecture_decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(architecture_decision, indent=2, sort_keys=True))
        return
    if args.ablation_only:
        ablation_rows = _status_rows((args.ablation_dir,))
        ablation_metrics = score_checkpoints_once(
            ablation_rows,
            output_dir=output / "ablations",
            device=args.device,
            reference_dir=reference_dir,
        )
        ablation_decision = decide_arm_d_ablations(ablation_metrics)
        (output / "arm_d_ablation_decision.json").write_text(
            json.dumps(ablation_decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(ablation_decision, indent=2, sort_keys=True))
        return
    rows = _status_rows((args.arm_c_dir, args.arm_d_dir))
    metrics = score_checkpoints_once(
        rows, output_dir=output, device=args.device, reference_dir=reference_dir
    )
    decision = decide_arm_d(metrics, rows)
    decision["completed_checkpoints"] = len(rows)
    path = output / "arm_d_validation_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    ablation_decision = {
        "adaptive_sampling": {"status": "DEFER"},
        "positive_premium": {"status": "DEFER"},
    }
    ablation_rows = _status_rows((args.ablation_dir,)) if args.ablation_dir.exists() else []
    if ablation_rows:
        ablation_metrics = score_checkpoints_once(
            ablation_rows,
            output_dir=output / "ablations",
            device=args.device,
            reference_dir=reference_dir,
        )
        ablation_decision = decide_arm_d_ablations(ablation_metrics)
        (output / "arm_d_ablation_decision.json").write_text(
            json.dumps(ablation_decision, indent=2, sort_keys=True), encoding="utf-8"
        )
    architecture_decision = {
        "status": "KEEP_ANCHOR",
        "selected_architecture": "resnet_4x2x50",
        "reason": "no completed sensitivity study; retain frozen anchor",
    }
    sensitivity_rows = (
        _status_rows((args.sensitivity_dir,)) if args.sensitivity_dir.exists() else []
    )
    if sensitivity_rows:
        sensitivity_metrics = score_checkpoints_once(
            sensitivity_rows,
            output_dir=output / "architecture_sensitivity",
            device=args.device,
            reference_dir=reference_dir,
        )
        architecture_decision = decide_arm_c_architecture(sensitivity_metrics)
        (output / "arm_c_architecture_decision.json").write_text(
            json.dumps(architecture_decision, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if decision.get("status") == "GO":
        variant = "etc_fb_mixture"
        if ablation_decision["adaptive_sampling"].get("status") == "GO":
            variant = "etc_fb_adaptive"
        if ablation_decision["positive_premium"].get("status") == "GO":
            variant = "positive_premium"
        frozen = {
            "protocol": "surf_pinn_cde_v1",
            "architecture": architecture_decision.get(
                "selected_architecture", "resnet_4x2x50"
            ),
            "d_variant": variant,
            "adam_steps": 40000,
            "lbfgs_max_evaluations": 2000,
            "seed_list": [17, 29, 43, 71, 101],
            "source_decisions": {
                "arm_d_validation": decision,
                "arm_c_architecture": architecture_decision,
                "arm_d_ablations": ablation_decision,
            },
        }
        serialized = json.dumps(frozen, indent=2, sort_keys=True)
        frozen_path = output / "frozen_pinn_configuration.json"
        frozen_path.write_text(serialized, encoding="utf-8")
        (output / "frozen_pinn_configuration.sha256").write_text(
            hashlib.sha256(serialized.encode("utf-8")).hexdigest() + "\n",
            encoding="utf-8",
        )
    print(json.dumps(decision, indent=2, sort_keys=True))


def _status_rows(directories: tuple[Path, ...]) -> list[dict[str, object]]:
    rows = []
    for directory in directories:
        for path in directory.rglob("training_status_shard_*.csv"):
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    unique = {
        (
            row["arm"],
            row.get("variant", "unknown"),
            row.get("network_spec", "unknown"),
            row["regime_id"],
            row["seed"],
        ): row
        for row in rows
    }
    return list(unique.values())


if __name__ == "__main__":
    main()
