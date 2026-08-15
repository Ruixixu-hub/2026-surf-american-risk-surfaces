"""Experiment 29: freeze and audit the SURF PINN mathematical protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.pinn.protocol import RESULTS_DIR, write_protocol_manifest
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.lcp import compute_lcp_residual


def run_audit(output_dir: Path | str = RESULTS_DIR / "00_protocol") -> dict[str, object]:
    output = Path(output_dir)
    paths = write_protocol_manifest(output)
    config = AmericanLCPConfig(
        "put", 1.0, 0.25, 0.01, 0.0, 0.2, 4.0, 48, 24, tolerance=1e-12
    )
    result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    residuals = []
    for step in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(config, result.value_grid[step - 1], step)
        residuals.append(compute_lcp_residual(system, result.value_grid[step, 1:-1]))
    audit = {
        "status": "PASS" if result.converged else "FAIL",
        "sign_convention": "g=u-phi>=0; rho=u_s-T*L_x(u)>=0; g*rho=0",
        "classical_control_converged": result.converged,
        "max_discrete_normalized_lcp_residual": max(
            item.normalized_lcp_residual for item in residuals
        ),
        "max_discrete_normalized_obstacle_violation": max(
            item.normalized_obstacle_violation for item in residuals
        ),
        "finite_surface": bool(np.all(np.isfinite(result.value_grid))),
        "protocol_paths": {key: str(value) for key, value in paths.items()},
    }
    if audit["max_discrete_normalized_lcp_residual"] > 1e-12:
        audit["status"] = "FAIL"
    audit_path = output / "operator_sign_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError("PINN operator/LCP sign audit failed; training is blocked.")
    return audit


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
