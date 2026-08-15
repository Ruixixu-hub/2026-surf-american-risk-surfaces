"""Experiment 51: layered decision and Chinese plain-language report."""

from __future__ import annotations

import json

from american_risk_surfaces.basis_operator.report import synthesize_basis_operator


if __name__ == "__main__":
    print(json.dumps(synthesize_basis_operator(), indent=2, sort_keys=True))
