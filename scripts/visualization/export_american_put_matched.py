#!/usr/bin/env python3
"""Export compact pointwise visualization data from completed PINN artifacts.

This utility performs inference and interpolation only. It does not train a
model, solve a new reference problem, or invoke held-out scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
from american_risk_surfaces.pinn.evaluation import (
    evaluate_pinn_vi,
    load_pinn_checkpoint,
    predict_pinn_surface,
)
from american_risk_surfaces.pinn.reference import interpolate_reference


REGIME_ID = "put_T100_s060_r005_q010"
OPTION_TYPE = "put"
K = 1.0
T = 1.0
R = 0.05
Q = 0.10
SIGMA = 0.60
SMAX = 4.0
BOUNDARY_THRESHOLD = 1e-6
PAYOFF_TOLERANCE = 1e-6
REFERENCE_SOURCE = "DIRK_policy_sinh_M480_N960"
EXPORT_STATEMENT = (
    "Visualization export from existing completed artifacts; not a new score, "
    "training run, or reference solve."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-c-checkpoint", type=Path, required=True)
    parser.add_argument("--arm-c-status", type=Path, required=True)
    parser.add_argument("--arm-d-checkpoint", type=Path, required=True)
    parser.add_argument("--arm-d-status", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def common_grids() -> tuple[np.ndarray, np.ndarray]:
    spot = np.linspace(0.5, 1.5, 401, dtype=float)
    base_tau = np.linspace(0.01, 1.0, 161, dtype=float)
    tau = np.unique(np.concatenate((base_tau, np.array([0.1, 0.5, 1.0]))))
    return spot, tau


def evaluation_coordinates(spot: np.ndarray, tau: np.ndarray) -> np.ndarray:
    spot_mesh, tau_mesh = np.meshgrid(spot, tau)
    return np.column_stack((np.log(spot_mesh.reshape(-1)), tau_mesh.reshape(-1)))


def evaluate_model_fields(
    checkpoint: Path,
    spot: np.ndarray,
    tau: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[Any, dict[str, np.ndarray]]:
    loaded = load_pinn_checkpoint(checkpoint, device=device)
    coordinates = evaluation_coordinates(spot, tau)
    chunks: dict[str, list[np.ndarray]] = {
        "V_over_K": [],
        "Delta": [],
        "scaled_Gamma": [],
    }
    for start in range(0, len(coordinates), batch_size):
        stop = min(start + batch_size, len(coordinates))
        evaluated = evaluate_pinn_vi(loaded, coordinates[start:stop], device=device)
        chunks["V_over_K"].append(evaluated["value"])
        chunks["Delta"].append(evaluated["delta"])
        chunks["scaled_Gamma"].append(evaluated["scaled_gamma"])
    fields = {
        name: np.concatenate(parts).reshape(len(tau), len(spot))
        for name, parts in chunks.items()
    }
    return loaded, fields


def evaluate_reference_fields(
    reference: Path,
    spot: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    coordinates = evaluation_coordinates(spot, tau)
    interpolated = interpolate_reference(reference, coordinates)
    return {
        "V_over_K": interpolated["value_over_k"].reshape(len(tau), len(spot)),
        "Delta": interpolated["delta"].reshape(len(tau), len(spot)),
        "scaled_Gamma": interpolated["scaled_gamma"].reshape(len(tau), len(spot)),
    }


def model_boundaries(loaded: Any, tau: np.ndarray, *, device: str) -> list[dict[str, Any]]:
    boundary_spot = np.linspace(0.0, SMAX, 121, dtype=float)
    boundary_tau = np.concatenate((np.array([0.0]), tau))
    prediction = predict_pinn_surface(
        loaded,
        boundary_spot,
        boundary_tau * T,
        K=K,
        device=device,
    )
    payoff = np.maximum(K - boundary_spot, 0.0)
    premium = prediction.value_grid - payoff[np.newaxis, :]
    rows: list[dict[str, Any]] = []
    for full_index, tau_fraction in enumerate(boundary_tau[1:], start=1):
        point = extract_boundary_at_time(
            boundary_spot,
            premium[full_index],
            OPTION_TYPE,
            float(tau_fraction * T),
            full_index,
            threshold=BOUNDARY_THRESHOLD,
        )
        rows.append(
            {
                "tau_over_T": float(tau_fraction),
                "boundary_found": bool(point.boundary_found),
                "S_star_over_K": (
                    float(point.boundary_spot / K) if point.boundary_found else None
                ),
                "threshold": float(point.threshold),
                "extraction_method": point.extraction_method,
                "no_boundary_reason": point.no_boundary_reason,
                "exercise_like_node_count": point.exercise_like_node_count,
                "continuation_like_node_count": point.continuation_like_node_count,
            }
        )
    return rows


def reference_boundaries(reference: Path, tau: np.ndarray) -> list[dict[str, Any]]:
    coordinates = np.column_stack((np.zeros(len(tau), dtype=float), tau))
    interpolated = interpolate_reference(reference, coordinates)["boundary_over_k"]
    rows: list[dict[str, Any]] = []
    for tau_fraction, boundary in zip(tau, interpolated):
        found = bool(np.isfinite(boundary))
        rows.append(
            {
                "tau_over_T": float(tau_fraction),
                "boundary_found": found,
                "S_star_over_K": float(boundary) if found else None,
                "threshold": BOUNDARY_THRESHOLD,
                "extraction_method": "stored_reference_boundary_linear_time_interpolation",
                "no_boundary_reason": "" if found else "stored_reference_boundary_unavailable",
                "exercise_like_node_count": None,
                "continuation_like_node_count": None,
            }
        )
    return rows


def write_surface(
    path: Path,
    spot: np.ndarray,
    tau: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    payoff = np.maximum(1.0 - spot, 0.0)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            S_over_K=spot,
            tau_over_T=tau,
            V_over_K=fields["V_over_K"],
            Delta=fields["Delta"],
            scaled_Gamma=fields["scaled_Gamma"],
            payoff_over_K=payoff,
        )


def write_slice_csvs(
    directory: Path,
    spot: np.ndarray,
    tau: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    payoff = np.maximum(1.0 - spot, 0.0)
    filenames = {1.0: "price_slice_tau_1.csv", 0.5: "price_slice_tau_05.csv", 0.1: "price_slice_tau_01.csv"}
    for target_tau, filename in filenames.items():
        indices = np.flatnonzero(tau == target_tau)
        if len(indices) != 1:
            raise RuntimeError(f"common surface grid does not contain tau={target_tau}")
        index = int(indices[0])
        with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "S_over_K",
                    "tau_over_T",
                    "V_over_K",
                    "Delta",
                    "scaled_Gamma",
                    "payoff_over_K",
                    "continuation_premium_over_K",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            for column, moneyness in enumerate(spot):
                value = float(fields["V_over_K"][index, column])
                writer.writerow(
                    {
                        "S_over_K": format(float(moneyness), ".17g"),
                        "tau_over_T": format(target_tau, ".17g"),
                        "V_over_K": format(value, ".17g"),
                        "Delta": format(float(fields["Delta"][index, column]), ".17g"),
                        "scaled_Gamma": format(
                            float(fields["scaled_Gamma"][index, column]), ".17g"
                        ),
                        "payoff_over_K": format(float(payoff[column]), ".17g"),
                        "continuation_premium_over_K": format(
                            value - float(payoff[column]), ".17g"
                        ),
                    }
                )


def write_boundary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "tau_over_T",
        "boundary_found",
        "S_star_over_K",
        "threshold",
        "extraction_method",
        "no_boundary_reason",
        "exercise_like_node_count",
        "continuation_like_node_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            for name in ("tau_over_T", "S_star_over_K", "threshold"):
                rendered[name] = "" if row[name] is None else format(float(row[name]), ".17g")
            rendered["boundary_found"] = "true" if row["boundary_found"] else "false"
            for name in ("exercise_like_node_count", "continuation_like_node_count"):
                rendered[name] = "" if row[name] is None else str(int(row[name]))
            writer.writerow(rendered)


def payoff_validation(fields: dict[str, np.ndarray], spot: np.ndarray) -> dict[str, Any]:
    payoff = np.maximum(1.0 - spot, 0.0)[np.newaxis, :]
    gap = fields["V_over_K"] - payoff
    return {
        "minimum_V_minus_payoff": float(np.min(gap)),
        "negative_gap_count": int(np.count_nonzero(gap < 0.0)),
        "violation_count_beyond_tolerance": int(np.count_nonzero(gap < -PAYOFF_TOLERANCE)),
        "tolerance": PAYOFF_TOLERANCE,
        "values_were_clipped_or_projected": False,
    }


def package_validation(
    directory: Path,
    spot: np.ndarray,
    tau: np.ndarray,
    fields: dict[str, np.ndarray],
    boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    finite_fields = {
        name: bool(np.all(np.isfinite(values))) for name, values in fields.items()
    }
    if not all(finite_fields.values()):
        raise RuntimeError(f"non-finite surface field in {directory.name}: {finite_fields}")
    found_boundaries = [
        float(row["S_star_over_K"])
        for row in boundaries
        if row["boundary_found"]
    ]
    if not np.all(np.isfinite(found_boundaries)):
        raise RuntimeError(f"non-finite found boundary in {directory.name}")

    surface = np.load(directory / "surface.npz")
    for name in surface.files:
        array = surface[name]
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"non-finite exported array {directory.name}/{name}")
    tau_one_index = int(np.flatnonzero(surface["tau_over_T"] == 1.0)[0])
    with (directory / "price_slice_tau_1.csv").open(newline="", encoding="utf-8") as handle:
        slice_rows = list(csv.DictReader(handle))
    csv_spot = np.array([float(row["S_over_K"]) for row in slice_rows])
    csv_value = np.array([float(row["V_over_K"]) for row in slice_rows])
    coordinate_error = float(np.max(np.abs(csv_spot - surface["S_over_K"])))
    slice_error = float(np.max(np.abs(csv_value - surface["V_over_K"][tau_one_index])))
    if coordinate_error > 0.0 or slice_error > 0.0:
        raise RuntimeError(f"tau=1 slice mismatch in {directory.name}")
    return {
        "all_surface_fields_finite": finite_fields,
        "all_npz_numeric_arrays_finite": True,
        "boundary_found_count": len(found_boundaries),
        "boundary_missing_count": len(boundaries) - len(found_boundaries),
        "all_found_boundary_coordinates_finite": True,
        "tau_1_surface_vs_csv_max_coordinate_error": coordinate_error,
        "tau_1_surface_vs_csv_max_value_error": slice_error,
        "payoff_check": payoff_validation(fields, spot),
    }


def base_metadata(source_commit: str, spot: np.ndarray, tau: np.ndarray) -> dict[str, Any]:
    return {
        "regime_id": REGIME_ID,
        "option_type": OPTION_TYPE,
        "K": K,
        "T": T,
        "r": R,
        "q": Q,
        "sigma": SIGMA,
        "Smax": SMAX,
        "normalization": {
            "S_over_K": "S / K",
            "tau_over_T": "tau / T",
            "V_over_K": "V / K",
            "scaled_Gamma": "K * Gamma",
        },
        "source_git_commit": source_commit,
        "reference_source": REFERENCE_SOURCE,
        "grid_definition": {
            "surface": {
                "S_over_K_min": float(spot[0]),
                "S_over_K_max": float(spot[-1]),
                "S_points": len(spot),
                "S_spacing": "equally_spaced",
                "tau_over_T_min": float(tau[0]),
                "tau_over_T_max": float(tau[-1]),
                "tau_points": len(tau),
                "tau_construction": (
                    "sorted unique union of 161 equally spaced points over [0.01, 1.0] "
                    "and exact slice levels {0.1, 0.5, 1.0}"
                ),
            },
            "price_slices": {
                "S_points": len(spot),
                "S_over_K_range": [float(spot[0]), float(spot[-1])],
                "tau_over_T": [1.0, 0.5, 0.1],
            },
            "model_boundary_extraction": {
                "full_S_over_K_range": [0.0, 4.0],
                "S_points": 121,
                "tau_points_including_internal_tau_zero_row": len(tau) + 1,
                "exported_positive_tau_points": len(tau),
                "threshold": BOUNDARY_THRESHOLD,
                "method": "extract_boundary_at_time / linear_threshold_crossing",
            },
        },
        "statement": EXPORT_STATEMENT,
    }


def status_record(path: Path, expected_arm: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE",
        "arm": expected_arm,
        "seed": 101,
        "regime_id": REGIME_ID,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"status mismatch for {path}: {name}={payload.get(name)!r}")
    return payload


def reference_equivalence(primary: Path, comparison: Path | None) -> dict[str, Any]:
    if comparison is None:
        return {"comparison_reference_available": False}
    left = np.load(primary)
    right = np.load(comparison)
    if set(left.files) != set(right.files):
        raise RuntimeError("reference artifacts have different field sets")
    maximum_differences: dict[str, float] = {}
    for name in left.files:
        if left[name].shape != right[name].shape:
            raise RuntimeError(f"reference shape mismatch for {name}")
        maximum_differences[name] = float(np.nanmax(np.abs(left[name] - right[name])))
    return {
        "comparison_reference_available": True,
        "comparison_reference_sha256": sha256(comparison),
        "maximum_absolute_array_differences": maximum_differences,
        "numerically_equivalent_within_1e-9": bool(
            max(maximum_differences.values()) <= 1e-9
        ),
    }


def export_package(
    directory: Path,
    *,
    arm: str,
    variant: str,
    checkpoint: Path | None,
    checkpoint_status: dict[str, Any] | None,
    checkpoint_hash: str | None,
    reference: Path,
    source_commit: str,
    spot: np.ndarray,
    tau: np.ndarray,
    fields: dict[str, np.ndarray],
    boundaries: list[dict[str, Any]],
    loaded: Any | None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    write_surface(directory / "surface.npz", spot, tau, fields)
    write_slice_csvs(directory, spot, tau, fields)
    write_boundary_csv(directory / "free_boundary.csv", boundaries)
    validation = package_validation(directory, spot, tau, fields, boundaries)
    metadata = base_metadata(source_commit, spot, tau)
    metadata.update(
        {
            "model": "high_accuracy_reference" if arm == "reference" else "PINN",
            "arm": arm,
            "model_or_arm": arm,
            "variant": variant,
            "seed": None if arm == "reference" else 101,
            "checkpoint_path": (
                None if checkpoint_status is None else checkpoint_status["checkpoint_path"]
            ),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_config_hash": (
                None if checkpoint_status is None else checkpoint_status.get("config_hash")
            ),
            "checkpoint_environment": None if loaded is None else loaded.metadata.get("environment"),
            "greek_source": (
                "stored high-accuracy finite-difference fields interpolated to common grid"
                if arm == "reference"
                else "checkpoint automatic differentiation via evaluate_pinn_vi"
            ),
            "reference_artifact_sha256": sha256(reference),
            "validation": validation,
        }
    )
    write_json(directory / "metadata.json", metadata)
    return validation


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {args.output_dir}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    for path in (
        args.arm_c_checkpoint,
        args.arm_c_status,
        args.arm_d_checkpoint,
        args.arm_d_status,
        args.reference,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    c_status = status_record(args.arm_c_status, "C")
    d_status = status_record(args.arm_d_status, "D")
    spot, tau = common_grids()
    reference_fields = evaluate_reference_fields(args.reference, spot, tau)
    c_loaded, c_fields = evaluate_model_fields(
        args.arm_c_checkpoint,
        spot,
        tau,
        device=args.device,
        batch_size=args.batch_size,
    )
    d_loaded, d_fields = evaluate_model_fields(
        args.arm_d_checkpoint,
        spot,
        tau,
        device=args.device,
        batch_size=args.batch_size,
    )
    c_boundaries = model_boundaries(c_loaded, tau, device=args.device)
    d_boundaries = model_boundaries(d_loaded, tau, device=args.device)
    ref_boundaries = reference_boundaries(args.reference, tau)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    validations = {
        "reference": export_package(
            args.output_dir / "reference",
            arm="reference",
            variant=REFERENCE_SOURCE,
            checkpoint=None,
            checkpoint_status=None,
            checkpoint_hash=None,
            reference=args.reference,
            source_commit=args.source_git_commit,
            spot=spot,
            tau=tau,
            fields=reference_fields,
            boundaries=ref_boundaries,
            loaded=None,
        ),
        "arm_c_seed101": export_package(
            args.output_dir / "arm_c_seed101",
            arm="C",
            variant="soft_lcp",
            checkpoint=args.arm_c_checkpoint,
            checkpoint_status=c_status,
            checkpoint_hash=sha256(args.arm_c_checkpoint),
            reference=args.reference,
            source_commit=args.source_git_commit,
            spot=spot,
            tau=tau,
            fields=c_fields,
            boundaries=c_boundaries,
            loaded=c_loaded,
        ),
        "arm_d_seed101": export_package(
            args.output_dir / "arm_d_seed101",
            arm="D",
            variant="etc_fb_mixture",
            checkpoint=args.arm_d_checkpoint,
            checkpoint_status=d_status,
            checkpoint_hash=sha256(args.arm_d_checkpoint),
            reference=args.reference,
            source_commit=args.source_git_commit,
            spot=spot,
            tau=tau,
            fields=d_fields,
            boundaries=d_boundaries,
            loaded=d_loaded,
        ),
    }

    coordinate_checks: dict[str, bool] = {}
    reference_surface = np.load(args.output_dir / "reference" / "surface.npz")
    for name in ("arm_c_seed101", "arm_d_seed101"):
        surface = np.load(args.output_dir / name / "surface.npz")
        coordinate_checks[f"{name}_S_matches_reference"] = bool(
            np.array_equal(surface["S_over_K"], reference_surface["S_over_K"])
        )
        coordinate_checks[f"{name}_tau_matches_reference"] = bool(
            np.array_equal(surface["tau_over_T"], reference_surface["tau_over_T"])
        )
    if not all(coordinate_checks.values()):
        raise RuntimeError(f"cross-package coordinate mismatch: {coordinate_checks}")

    equivalence = reference_equivalence(args.reference, args.comparison_reference)
    if equivalence.get("comparison_reference_available") and not equivalence[
        "numerically_equivalent_within_1e-9"
    ]:
        raise RuntimeError("the independently stored reference artifacts are not equivalent")

    validation_report = {
        "regime_id": REGIME_ID,
        "source_git_commit": args.source_git_commit,
        "statement": EXPORT_STATEMENT,
        "surface_grid_shape_tau_by_S": [len(tau), len(spot)],
        "coordinate_checks": coordinate_checks,
        "package_checks": validations,
        "reference_artifact_equivalence": equivalence,
        "canonical_experiment_files_modified": False,
    }
    write_json(args.output_dir / "validation_report.json", validation_report)
    (args.output_dir / "README.md").write_text(
        "# Matched American-put visualization export\n\n"
        f"Regime: `{REGIME_ID}`. This compact package contains direct pointwise "
        "Arm C/D seed-101 predictions and the stored high-accuracy reference on "
        "a common plotting grid. No model was retrained, no held-out score was "
        "rerun, and neural values were not clipped or projected to the payoff.\n\n"
        "The source artifacts remain outside Git. `validation_report.json` records "
        "finite-array, coordinate, tau=1 slice, payoff, boundary, and independent "
        "reference-equivalence checks.\n",
        encoding="utf-8",
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            relative = path.relative_to(args.output_dir).as_posix()
            artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        args.output_dir / "checksums.json",
        {
            "algorithm": "SHA-256",
            "artifacts": artifacts,
            "checksums_json_self_hash": "excluded because a manifest cannot contain its own stable hash",
        },
    )
    print(json.dumps(validation_report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
