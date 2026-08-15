"""Experiment 50: heldout prediction/scoring behind the permanent validation gate."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.basis_operator.heldout import (
    run_heldout_predictions,
    score_heldout_predictions,
    run_q0_raw_ood_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict", "score", "q0-ood"))
    arguments = parser.parse_args()
    if arguments.command == "predict":
        result = run_heldout_predictions()
    elif arguments.command == "score":
        result = score_heldout_predictions()
    else:
        result = run_q0_raw_ood_diagnostic()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
