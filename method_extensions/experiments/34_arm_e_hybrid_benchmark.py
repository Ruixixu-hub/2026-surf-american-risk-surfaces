"""Experiment 34: strict Arm E PINN-to-Policy-Iteration benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from american_risk_surfaces.pinn.evaluation import load_pinn_checkpoint, run_arm_e_hybrid
from american_risk_surfaces.pinn.protocol import RESULTS_DIR, load_regime_records
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig, american_cn_lcp_price


OUTPUT = RESULTS_DIR / "05_arm_e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--allow-without-d-go", action="store_true")
    args = parser.parse_args()
    decision_path = RESULTS_DIR / "04_heldout" / "arm_d_heldout_decision.json"
    if not args.allow_without_d_go:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("status") != "GO":
            raise RuntimeError("Arm D did not pass its held-out gate; Arm E remains blocked")
    statuses = _completed_d_statuses(RESULTS_DIR / "04_heldout" / "training")
    records = {record.regime_id: record for record in load_regime_records(splits=("test", "stress_holdout"))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime_rows = []
    accuracy_rows = []
    rng = np.random.default_rng(20260812)
    for row in statuses:
        record = records[row["regime_id"]]
        config = AmericanLCPConfig(
            record.option_type,
            record.K,
            record.T,
            record.r,
            record.q,
            record.sigma,
            record.Smax,
            record.M,
            record.N,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        load_started = perf_counter()
        loaded = load_pinn_checkpoint(row["checkpoint_path"], device=args.device)
        load_seconds = perf_counter() - load_started
        for _ in range(args.warmups):
            american_cn_lcp_price(config, lcp_solver="policy_iteration")
            run_arm_e_hybrid(loaded, config, device=args.device)
        first_e = None
        first_b = None
        for repeat in range(args.repeats):
            order = ["B", "E"]
            rng.shuffle(order)
            for arm in order:
                if arm == "B":
                    started = perf_counter()
                    result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
                    total = perf_counter() - started
                    first_b = result if first_b is None else first_b
                    timing = {
                        "inference_seconds": 0.0,
                        "transfer_seconds": 0.0,
                        "projection_seconds": 0.0,
                        "lcp_finish_seconds": result.lcp_finish_seconds,
                        "online_total_seconds": total,
                    }
                else:
                    result, hybrid_timing = run_arm_e_hybrid(loaded, config, device=args.device)
                    first_e = result if first_e is None else first_e
                    timing = hybrid_timing.__dict__
                runtime_rows.append(
                    {
                        "arm": arm,
                        "regime_id": record.regime_id,
                        "split": record.split,
                        "seed": row["seed"],
                        "repeat": repeat,
                        "checkpoint_load_seconds": load_seconds if arm == "E" else 0.0,
                        "training_seconds": float(row["training_seconds"]) if arm == "E" else 0.0,
                        "first_query_total_seconds": (
                            load_seconds + float(timing["online_total_seconds"])
                            if arm == "E"
                            else float(timing["online_total_seconds"])
                        ),
                        **timing,
                        "converged": result.converged,
                        "max_normalized_lcp_residual": max(
                            step.residual.normalized_lcp_residual for step in result.lcp_results
                        ),
                    }
                )
        assert first_e is not None and first_b is not None
        accuracy_rows.append(
            {
                "regime_id": record.regime_id,
                "split": record.split,
                "seed": row["seed"],
                "max_abs_e_minus_b": float(np.max(np.abs(first_e.value_grid - first_b.value_grid))),
                "e_converged": first_e.converged,
                "e_max_obstacle_violation": first_e.max_obstacle_violation,
                "e_total_iterations": sum(step.iterations for step in first_e.lcp_results),
                "b_total_iterations": sum(step.iterations for step in first_b.lcp_results),
            }
        )
        _write_csv(OUTPUT / "runtime_samples.csv", runtime_rows)
        _write_csv(OUTPUT / "accuracy_and_iterations.csv", accuracy_rows)
    decision = _decision(runtime_rows, accuracy_rows)
    (OUTPUT / "arm_e_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


def _completed_d_statuses(directory: Path) -> list[dict[str, str]]:
    rows = []
    for path in directory.glob("training_status_shard_*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    unique = {(row["regime_id"], row["seed"]): row for row in rows if row["arm"] == "D" and row["status"] == "COMPLETE"}
    return list(unique.values())


def _decision(runtime_rows: list[dict[str, object]], accuracy_rows: list[dict[str, object]]) -> dict[str, object]:
    b = np.array([float(row["online_total_seconds"]) for row in runtime_rows if row["arm"] == "B"])
    e = np.array([float(row["online_total_seconds"]) for row in runtime_rows if row["arm"] == "E"])
    all_tolerance = all(float(row["max_normalized_lcp_residual"]) <= 1e-12 for row in runtime_rows if row["arm"] == "E")
    identical = max(float(row["max_abs_e_minus_b"]) for row in accuracy_rows) <= 1e-9
    median_pass = float(np.median(e)) < float(np.median(b))
    p95_pass = float(np.quantile(e, 0.95)) <= float(np.quantile(b, 0.95))
    go = all_tolerance and identical and median_pass and p95_pass
    saved = float(np.median(b) - np.median(e))
    median_training = float(np.median([float(row["training_seconds"]) for row in runtime_rows if row["arm"] == "E"]))
    median_loading = float(np.median([float(row["checkpoint_load_seconds"]) for row in runtime_rows if row["arm"] == "E"]))
    break_even = (median_training + median_loading) / saved if saved > 0.0 else float("inf")
    return {
        "status": "GO" if go else "STOP",
        "all_e_runs_reach_1e-12": all_tolerance,
        "strict_solution_matches_b": identical,
        "median_b_seconds": float(np.median(b)),
        "median_e_seconds": float(np.median(e)),
        "p95_b_seconds": float(np.quantile(b, 0.95)),
        "p95_e_seconds": float(np.quantile(e, 0.95)),
        "p99_b_seconds": float(np.quantile(b, 0.99)),
        "p99_e_seconds": float(np.quantile(e, 0.99)),
        "median_training_seconds": median_training,
        "median_checkpoint_loading_seconds": median_loading,
        "median_first_query_e_seconds": float(
            np.median(
                [
                    float(row["first_query_total_seconds"])
                    for row in runtime_rows
                    if row["arm"] == "E"
                ]
            )
        ),
        "break_even_queries": break_even,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
