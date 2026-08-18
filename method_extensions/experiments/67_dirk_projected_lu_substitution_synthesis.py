"""Experiment 67: regenerate reports from the immutable substitution evidence."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.dirk_projected_lu_study import (
    synthesize_substitution_audit,
)


if __name__ == "__main__":
    outputs = synthesize_substitution_audit()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))

