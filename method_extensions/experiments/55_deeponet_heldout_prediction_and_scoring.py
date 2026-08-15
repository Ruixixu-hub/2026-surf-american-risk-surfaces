"""Experiment 55: permanently gated heldout prediction, scoring, and q=0 OOD."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.deeponet.heldout import (
    run_heldout_predictions,
    run_q0_raw_ood_diagnostic,
    score_heldout_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict-heldout", "score-heldout", "q0-ood"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()
    if arguments.command == "predict-heldout":
        result = run_heldout_predictions(device=arguments.device)
    elif arguments.command == "score-heldout":
        result = score_heldout_predictions()
    else:
        result = run_q0_raw_ood_diagnostic(device=arguments.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
