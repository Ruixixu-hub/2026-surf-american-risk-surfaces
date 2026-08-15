"""Experiment 41: final RB-VI gate, break-even analysis, figures, and report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.reduced_order.protocol import REPORTS_DIR, RESULTS_DIR
from american_risk_surfaces.reduced_order.study import BASIS_DIR, HELDOUT_DIR, VALIDATION_DIR


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def synthesize() -> dict[str, object]:
    frozen = json.loads((VALIDATION_DIR / "frozen_rb_config.json").read_text(encoding="utf-8"))
    if not any(
        frozen["families"][family]["selected_dimension"] is not None
        for family in ("put", "call")
    ):
        return _synthesize_validation_stop(frozen)
    metrics = _read_csv(HELDOUT_DIR / "heldout_metrics.csv")
    runtimes = _read_csv(HELDOUT_DIR / "runtime_samples.csv")
    snapshots = _read_csv(RESULTS_DIR / "01_snapshots" / "snapshot_manifest.csv")
    bases = _read_csv(BASIS_DIR / "basis_manifest.csv")
    decision: dict[str, object] = {"families": {}}
    for family in ("put", "call"):
        dimension = frozen["families"][family]["selected_dimension"]
        if dimension is None:
            decision["families"][family] = {
                "status": "STOP_ACCURACY",
                "reason": "no validation basis passed",
            }
            continue
        projected = [
            row for row in metrics
            if row["option_type"] == family and row["method"] == "RB_VI_PROJECTED"
        ]
        policies = {
            row["regime_id"]: row for row in metrics
            if row["option_type"] == family and row["method"] == "CN_POLICY"
        }
        accuracy = bool(projected) and all(
            float(row["price_rmse"])
            <= max(0.002474946, 1.25 * float(policies[row["regime_id"]]["price_rmse"]))
            for row in projected
        )
        structure = bool(projected) and all(
            float(row["obstacle_violation"]) <= 1e-12 for row in projected
        )
        rb_times = np.asarray([
            float(row["elapsed_seconds"]) for row in runtimes
            if row["option_type"] == family and row["method"] == "RB_VI"
        ])
        policy_times = np.asarray([
            float(row["elapsed_seconds"]) for row in runtimes
            if row["option_type"] == family and row["method"] == "CN_POLICY"
        ])
        rb_median = float(np.median(rb_times))
        policy_median = float(np.median(policy_times))
        speedup = 1.0 - rb_median / policy_median
        p95_pass = float(np.quantile(rb_times, 0.95)) <= float(np.quantile(policy_times, 0.95))
        snapshot_cost = sum(
            float(row["elapsed_seconds"]) for row in snapshots if row["option_type"] == family
        )
        basis_cost = sum(
            float(row["construction_seconds"])
            for row in bases if row["option_type"] == family
        )
        validation_rows = [
            row for row in _read_csv(VALIDATION_DIR / "validation_ladder.csv")
            if row["option_type"] == family
        ]
        validation_cost = sum(
            float(row["timing_total_seconds"])
            for row in validation_rows
        )
        unique_validation = {}
        for row in validation_rows:
            unique_validation[row["regime_id"]] = (
                float(row["fom_generation_seconds"])
                + float(row["high_reference_generation_seconds"])
            )
        validation_cost += sum(unique_validation.values())
        offline_cost = snapshot_cost + basis_cost + validation_cost
        saving = policy_median - rb_median
        break_even = offline_cost / saving if saving > 0.0 else float("inf")
        coverage_values = [
            row["estimator_covers_reduction_error"].lower() == "true"
            for row in projected
        ]
        coverage = float(np.mean(coverage_values)) if coverage_values else 0.0
        if not accuracy:
            status = "STOP_ACCURACY"
        elif not structure:
            status = "STOP_STABILITY"
        elif speedup < 0.20 or not p95_pass or break_even > 10000:
            status = "STOP_ONLINE_VALUE"
        else:
            status = "GO_RB_VI"
        decision["families"][family] = {
            "status": status,
            "dimension": dimension,
            "accuracy_pass": accuracy,
            "structure_pass": structure,
            "median_speedup_fraction": speedup,
            "rb_median_seconds": rb_median,
            "policy_median_seconds": policy_median,
            "rb_p95_seconds": float(np.quantile(rb_times, 0.95)),
            "policy_p95_seconds": float(np.quantile(policy_times, 0.95)),
            "offline_seconds": offline_cost,
            "break_even_queries": break_even,
            "estimator_coverage": coverage,
        }
    family_statuses = [decision["families"][family]["status"] for family in ("put", "call")]
    if family_statuses == ["GO_RB_VI", "GO_RB_VI"]:
        decision["status"] = "GO_RB_VI"
    elif "GO_RB_VI" in family_statuses:
        decision["status"] = "PARTIAL_GO"
    elif "STOP_STABILITY" in family_statuses:
        decision["status"] = "STOP_STABILITY"
    elif "STOP_ACCURACY" in family_statuses:
        decision["status"] = "STOP_ACCURACY"
    else:
        decision["status"] = "STOP_ONLINE_VALUE"
    coefficient_path = RESULTS_DIR.parent / "07_method_extensions" / "05_pod_coefficient" / "coefficient_decision.json"
    decision["stopped_polynomial_pod_comparator"] = json.loads(
        coefficient_path.read_text(encoding="utf-8")
    ) if coefficient_path.exists() else {"status": "NOT_AVAILABLE"}
    output = RESULTS_DIR / "06_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "method_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_rows = [
        {"option_type": family, **decision["families"][family]}
        for family in ("put", "call")
    ]
    with (output / "method_decision_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary_rows for key in row}))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_report(decision)
    _write_pareto(metrics, runtimes, output)
    return decision


def _synthesize_validation_stop(frozen: dict[str, object]) -> dict[str, object]:
    validation = _read_csv(VALIDATION_DIR / "validation_ladder.csv")
    basis_rows = _read_csv(BASIS_DIR / "basis_manifest.csv")
    decision: dict[str, object] = {
        "status": "STOP_ACCURACY",
        "heldout_scoring_performed": False,
        "reason": "Neither option family passed the pre-registered validation accuracy/structure gate.",
        "families": {},
    }
    for family in ("put", "call"):
        stable_dimensions = sorted(
            {
                int(row["requested_dimension"])
                for row in validation
                if row["option_type"] == family and row["converged"].lower() == "true"
            }
        )
        summaries = []
        for dimension in stable_dimensions:
            rows = [
                row for row in validation
                if row["option_type"] == family and int(row["requested_dimension"]) == dimension
            ]
            finite_boundary = [
                float(row["projected_reduction_boundary_conditional_mae"])
                for row in rows
                if row.get("projected_reduction_boundary_conditional_mae")
                and np.isfinite(float(row["projected_reduction_boundary_conditional_mae"]))
            ]
            summaries.append(
                {
                    "dimension": dimension,
                    "worst_reduction_price_rmse": max(
                        float(row["projected_reduction_price_rmse"]) for row in rows
                    ),
                    "worst_boundary_mae": max(finite_boundary) if finite_boundary else float("inf"),
                    "worst_delta_ratio_vs_cn": max(
                        float(row["projected_high_delta_rmse"])
                        / max(float(row["cn_high_delta_rmse"]), 1e-15)
                        for row in rows
                    ),
                    "worst_gamma_ratio_vs_cn": max(
                        float(row["projected_high_stable_gamma_rmse"])
                        / max(float(row["cn_high_stable_gamma_rmse"]), 1e-15)
                        for row in rows
                    ),
                }
            )
        best = min(summaries, key=lambda row: row["worst_reduction_price_rmse"])
        stability_failures = [
            {
                "dimension": int(row["requested_dimension"]),
                "reason": row["failure_reason"],
            }
            for row in basis_rows
            if row["option_type"] == family and row["status"] != "COMPLETE"
        ]
        decision["families"][family] = {
            "status": "STOP_ACCURACY",
            "best_stable_dimension_by_price": best["dimension"],
            "worst_reduction_price_rmse": best["worst_reduction_price_rmse"],
            "worst_boundary_mae": best["worst_boundary_mae"],
            "worst_delta_ratio_vs_cn": best["worst_delta_ratio_vs_cn"],
            "worst_gamma_ratio_vs_cn": best["worst_gamma_ratio_vs_cn"],
            "stability_failures": stability_failures,
            "ladder": summaries,
        }
    output = RESULTS_DIR / "06_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "method_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = [
        {
            "option_type": family,
            **{key: value for key, value in decision["families"][family].items() if key not in {"ladder", "stability_failures"}},
        }
        for family in ("put", "call")
    ]
    with (output / "method_decision_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    put = decision["families"]["put"]
    call = decision["families"]["call"]
    report = (
        "# SURF Primal/Dual Reduced-Basis VI Report\n\n"
        "Final decision: **STOP_ACCURACY** at validation.\n\n"
        "No test/stress high-accuracy labels were opened, and Experiment 40 was not run, because neither option family passed the frozen validation gate.\n\n"
        "## Plain result\n\n"
        f"The best stable put basis by price used {put['best_stable_dimension_by_price']} dual generators. "
        f"Its worst reduction RMSE was {put['worst_reduction_price_rmse']:.6g}, but its worst boundary error was "
        f"{put['worst_boundary_mae']:.6g}; Delta and stable-mask Gamma reached "
        f"{put['worst_delta_ratio_vs_cn']:.3f}x and {put['worst_gamma_ratio_vs_cn']:.3f}x the CN+Policy reference errors.\n\n"
        f"The best stable call basis by price used {call['best_stable_dimension_by_price']} dual generators. "
        f"Its worst reduction RMSE was {call['worst_reduction_price_rmse']:.6g}, boundary error was "
        f"{call['worst_boundary_mae']:.6g}, and its Delta/Gamma error ratios were "
        f"{call['worst_delta_ratio_vs_cn']:.3f}x/{call['worst_gamma_ratio_vs_cn']:.3f}x.\n\n"
        "Put dimension 32 was rejected by the stability gate rather than regularized: the dual cone became numerically linearly dependent.\n\n"
        "## Decision\n\n"
        "The low price error confirms the value trajectories are compressible, but the global primal/dual cone does not preserve the moving exercise boundary and Greeks well enough. The next justified branch is a boundary-aligned/localized basis or a positive-premium basis operator/DeepONet; this RB-VI model must not be presented as an online winner.\n"
    )
    (REPORTS_DIR / "rb_vi_report.md").write_text(report, encoding="utf-8")
    _write_validation_figures(validation, output)
    _write_validation_diagnostics(validation, decision, output)
    return decision


def _write_validation_figures(validation: list[dict[str, str]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in ("put", "call"):
        dimensions = []
        price = []
        boundary = []
        for dimension in (4, 8, 12, 16, 24, 32):
            rows = [
                row for row in validation
                if row["option_type"] == family
                and int(row["requested_dimension"]) == dimension
                and row["converged"].lower() == "true"
            ]
            if not rows:
                continue
            dimensions.append(dimension)
            price.append(max(float(row["projected_reduction_price_rmse"]) for row in rows))
            values = [
                float(row["projected_reduction_boundary_conditional_mae"])
                for row in rows
                if np.isfinite(float(row["projected_reduction_boundary_conditional_mae"]))
            ]
            boundary.append(max(values) if values else np.nan)
        axes[0].plot(dimensions, price, marker="o", label=family)
        axes[1].plot(dimensions, boundary, marker="o", label=family)
    axes[0].axhline(0.25 * 0.0019799570496789242, color="black", linestyle="--", label="price gate")
    axes[1].axhline(4.0 / 120.0, color="black", linestyle="--", label="one-cell gate")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Worst validation price reduction error")
    axes[1].set_title("Worst conditional boundary error")
    for axis in axes:
        axis.set_xlabel("Dual basis dimension")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "validation_ladder_gates.png", dpi=180)
    plt.close(figure)


def _write_validation_diagnostics(
    validation: list[dict[str, str]], decision: dict[str, object], output: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
    from american_risk_surfaces.reduced_order import (
        assemble_affine_rb_operator,
        load_basis,
        load_regimes,
        solve_reduced_american_vi,
        trajectory_multipliers,
    )
    from american_risk_surfaces.solvers.american_lcp import american_cn_lcp_price

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    cases = {}
    for row_index, family in enumerate(("put", "call")):
        dimension = int(decision["families"][family]["best_stable_dimension_by_price"])
        basis = load_basis(BASIS_DIR / family / f"basis_{dimension:02d}.npz")
        spots = np.linspace(0.0, 4.0, basis.primal_basis.shape[0] + 2)[1:-1]
        axes[row_index, 0].plot(spots, basis.primal_basis[:, :4])
        axes[row_index, 1].plot(spots, basis.dual_generators[:, :4])
        axes[row_index, 0].set_ylabel(family)
        axes[row_index, 0].set_title(f"{family}: first primal modes")
        axes[row_index, 1].set_title(f"{family}: first nonnegative dual generators")
        selected = [
            row for row in validation
            if row["option_type"] == family
            and int(row["requested_dimension"]) == dimension
        ]
        worst = max(selected, key=lambda row: float(row["projected_reduction_price_rmse"]))
        regime = next(
            item for item in load_regimes(splits=("validation",))
            if item.regime_id == worst["regime_id"]
        )
        full = american_cn_lcp_price(regime.config(), lcp_solver="policy_iteration")
        rb = solve_reduced_american_vi(assemble_affine_rb_operator(basis), regime.config())
        reference_multiplier, _, _ = trajectory_multipliers(regime.config(), full.value_grid)
        cases[family] = (regime, full, rb, reference_multiplier)
    for axis in axes[-1]:
        axis.set_xlabel("Spot S")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "basis_dual_generators.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, family in zip(axes, ("put", "call")):
        regime, full, rb, _ = cases[family]
        image = axis.imshow(
            np.abs(rb.projected_value_grid - full.value_grid),
            origin="lower",
            aspect="auto",
            extent=(0.0, 4.0, 0.0, regime.T),
        )
        axis.set_title(f"Worst validation {family}: |RB - CN+Policy|")
        axis.set_xlabel("Spot S")
        axis.set_ylabel("Time to maturity")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "worst_case_surfaces.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, family in enumerate(("put", "call")):
        regime, full, rb, reference_multiplier = cases[family]
        reference_boundary = []
        rb_boundary = []
        for index, tau in enumerate(full.tau_grid):
            reference_point = extract_boundary_at_time(
                full.spot_grid,
                full.value_grid[index] - full.payoff,
                family,
                float(tau),
                index,
            )
            rb_point = extract_boundary_at_time(
                rb.spot_grid,
                rb.projected_value_grid[index] - full.payoff,
                family,
                float(tau),
                index,
            )
            reference_boundary.append(reference_point.boundary_spot if reference_point.boundary_found else np.nan)
            rb_boundary.append(rb_point.boundary_spot if rb_point.boundary_found else np.nan)
        axes[row_index, 0].plot(full.tau_grid, reference_boundary, label="CN+Policy")
        axes[row_index, 0].plot(full.tau_grid, rb_boundary, label="RB projected")
        axes[row_index, 0].set_title(f"{family} boundary overlay")
        axes[row_index, 0].set_ylabel("Boundary spot")
        axes[row_index, 1].plot(full.spot_grid[1:-1], reference_multiplier[-1], label="FOM multiplier")
        axes[row_index, 1].plot(full.spot_grid[1:-1], rb.reconstructed_multiplier_grid[-1], label="RB multiplier")
        axes[row_index, 1].set_title(f"{family} final active-set multiplier")
        for axis in axes[row_index]:
            axis.legend()
            axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time to maturity")
    axes[-1, 1].set_xlabel("Spot S")
    figure.tight_layout()
    figure.savefig(output / "boundary_active_set_overlay.png", dpi=180)
    plt.close(figure)


def _write_report(decision: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SURF Primal/Dual Reduced-Basis VI Report",
        "",
        f"Final decision: **{decision['status']}**",
        "",
        "The bases were built only from the 202 training regimes. Put and call dimensions were selected on validation before held-out scoring.",
        "",
        "| Family | Status | Dimension | Median speedup | Break-even queries | Estimator coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("put", "call"):
        item = decision["families"][family]
        lines.append(
            f"| {family} | {item['status']} | {item.get('dimension')} | "
            f"{item.get('median_speedup_fraction', float('nan')):.2%} | "
            f"{item.get('break_even_queries', float('nan')):.1f} | "
            f"{item.get('estimator_coverage', float('nan')):.2%} |"
        )
    lines.extend(
        [
            "",
            "The reported online RB time includes affine assembly, PDAS, reconstruction, obstacle projection, and the full-grid LCP residual audit. The estimator is empirical and discretely calibrated; it is not claimed as a rigorous continuous error bound.",
            "",
            "The stopped polynomial POD coefficient model remains a negative comparator, not a successful RB-VI implementation.",
        ]
    )
    (REPORTS_DIR / "rb_vi_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pareto(metrics: list[dict[str, str]], runtimes: list[dict[str, str]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = ("CN_PSOR", "CN_POLICY", "RB_VI_PROJECTED")
    timing_name = {"CN_PSOR": "CN_PSOR", "CN_POLICY": "CN_POLICY", "RB_VI_PROJECTED": "RB_VI"}
    figure, axis = plt.subplots(figsize=(7, 5))
    for method in methods:
        errors = [float(row["price_rmse"]) for row in metrics if row["method"] == method]
        times = [
            float(row["elapsed_seconds"])
            for row in runtimes if row["method"] == timing_name[method]
        ]
        if errors and times:
            axis.scatter(np.median(times), np.median(errors), label=method, s=55)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Median end-to-end seconds")
    axis.set_ylabel("Median price RMSE vs high reference")
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "price_runtime_pareto.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    print(json.dumps(synthesize(), indent=2, sort_keys=True))
