"""Experiment 57: layered decisions and Chinese plain-language report."""

from __future__ import annotations

import json

from american_risk_surfaces.deeponet.report import synthesize_deeponet


if __name__ == "__main__":
    print(json.dumps(synthesize_deeponet(), indent=2, sort_keys=True))
