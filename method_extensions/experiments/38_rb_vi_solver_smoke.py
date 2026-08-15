"""Experiment 38: synthetic PDAS and four-family RB solver smoke checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.reduced_order import (
    assemble_affine_rb_operator,
    load_basis,
    solve_reduced_american_vi,
    solve_reduced_mixed_lcp,
)
from american_risk_surfaces.reduced_order.protocol import RESULTS_DIR, load_regimes
from american_risk_surfaces.reduced_order.study import BASIS_DIR


def main() -> None:
    toy = solve_reduced_mixed_lcp(
        np.eye(2), np.eye(2), np.array([0.0, 2.0]), np.array([1.0, 1.0])
    )
    if not toy.converged or not np.allclose(toy.alpha, [1.0, 2.0]):
        raise RuntimeError("synthetic mixed LCP reproduction failed")
    train = load_regimes(splits=("train",))
    stress = load_regimes(splits=("stress_holdout",))
    all_regimes = load_regimes()
    selected = [
        next(regime for regime in train if regime.option_type == "put"),
        next(regime for regime in stress if regime.option_type == "put"),
        next(regime for regime in train if regime.option_type == "call" and regime.q > 0.0),
        next(regime for regime in all_regimes if regime.option_type == "call" and regime.q == 0.0),
    ]
    artifacts = {
        family: assemble_affine_rb_operator(load_basis(BASIS_DIR / family / "basis_04.npz"))
        for family in ("put", "call")
    }
    rows = []
    for regime in selected:
        result = solve_reduced_american_vi(artifacts[regime.option_type], regime.config())
        rows.append(
            {
                "regime_id": regime.regime_id,
                "option_type": regime.option_type,
                "q": regime.q,
                "converged": result.converged,
                "failure_reason": result.failure_reason or "",
                "max_iterations": max(result.iterations_by_time, default=0),
                "reduced_residual_max": result.reduced_residual_max,
                "full_lcp_residual_max": result.full_lcp_residual_max,
                "boundaries_finite": bool(
                    np.all(np.isfinite(result.raw_value_grid[:, [0, -1]]))
                ),
            }
        )
    output = RESULTS_DIR / "03_smoke"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "solver_smoke.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "status": "GO_VALIDATION" if all(row["converged"] and row["boundaries_finite"] for row in rows) else "STOP_STABILITY",
        "synthetic_reproduction": True,
        "cases": rows,
    }
    (output / "smoke_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
