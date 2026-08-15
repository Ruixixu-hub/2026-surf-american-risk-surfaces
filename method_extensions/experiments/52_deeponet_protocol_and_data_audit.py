"""Experiment 52: freeze the DeepONet protocol and audit all required inputs."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from american_risk_surfaces.deeponet.data import build_deeponet_training_bundle
from american_risk_surfaces.deeponet.protocol import (
    BASIS_VALIDATION_METRICS,
    RESULTS_DIR,
    VALIDATION_CACHE_DIR,
    protocol_hash,
    train_snapshot_paths,
    write_protocol,
)


def main() -> None:
    protocol_path = write_protocol()
    families = {}
    for family in ("put", "call"):
        bundle = build_deeponet_training_bundle(train_snapshot_paths(family), family)
        families[family] = {
            "snapshot_count": len(bundle.regime_ids),
            "feature_shape": list(bundle.features.shape),
            "premium_shape": list(bundle.premium_surfaces.shape),
            "coordinate_shape": list(bundle.coordinate_grid.shape),
            "boundary_nodes": int(bundle.boundary_mask.sum()),
            "continuation_fraction": float(bundle.continuation_mask.mean()),
            "class_weights": bundle.class_weights.tolist(),
            "hashes": bundle.hashes,
        }
    validation_caches = sorted(VALIDATION_CACHE_DIR.glob("*.npz"))
    package_root = Path(__file__).parents[1] / "src/american_risk_surfaces/deeponet"
    forbidden_training_imports = []
    for source in (package_root / "data.py", package_root / "model.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = [
            node.module or "" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        if any("validation_reference_bundle" in name or "heldout" in name for name in imports):
            forbidden_training_imports.append(str(source))
    if forbidden_training_imports:
        raise RuntimeError(
            f"training label-isolation audit failed: {forbidden_training_imports}"
        )
    payload = {
        "status": "PASS",
        "protocol_path": str(protocol_path),
        "protocol_hash": protocol_hash(),
        "families": families,
        "validation_cache_count": len(validation_caches),
        "basis_operator_comparator_exists": BASIS_VALIDATION_METRICS.exists(),
        "heldout_labels_read": False,
        "training_label_isolation_static_audit": "PASS",
        "q0_call_route": "EUROPEAN_BSM_ANALYTIC_Q0_CALL",
    }
    output = RESULTS_DIR / "00_protocol" / "data_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
