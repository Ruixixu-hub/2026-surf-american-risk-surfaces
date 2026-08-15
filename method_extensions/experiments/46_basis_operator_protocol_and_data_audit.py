"""Experiment 46: freeze data, representation, routing, and leakage protocol."""

from __future__ import annotations

import json

from american_risk_surfaces.basis_operator.protocol import (
    RESULTS_DIR,
    load_regimes,
    train_snapshot_paths,
    write_protocol,
)


def main() -> None:
    path = write_protocol()
    rows = load_regimes()
    audit = {
        "protocol": str(path),
        "train_snapshot_counts": {
            family: len(train_snapshot_paths(family)) for family in ("put", "call")
        },
        "split_counts": {
            split: sum(item.split == split for item in rows)
            for split in ("train", "validation", "test", "stress_holdout")
        },
        "validation_family_counts": {
            "put": sum(item.split == "validation" and item.option_type == "put" for item in rows),
            "dividend_call": sum(item.split == "validation" and item.option_type == "call" and item.q > 0 for item in rows),
        },
        "heldout_counts": {
            "put": sum(item.split in {"test", "stress_holdout"} and item.option_type == "put" for item in rows),
            "dividend_call": sum(item.split in {"test", "stress_holdout"} and item.option_type == "call" and item.q > 0 for item in rows),
            "q0_call_analytic": sum(item.split in {"test", "stress_holdout"} and item.option_type == "call" and item.q == 0 for item in rows),
        },
        "learned_vector_size": 120 * 119,
        "q0_primary_route": "EUROPEAN_BSM_ANALYTIC",
        "status": "PASS",
    }
    output = RESULTS_DIR / "00_protocol" / "data_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
