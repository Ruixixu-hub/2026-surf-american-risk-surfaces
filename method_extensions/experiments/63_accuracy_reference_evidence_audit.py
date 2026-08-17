"""Experiment 63: audit existing CN/Rannacher/DIRK and uniform/sinh evidence."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.poster_unified_study import (
    audit_accuracy_reference_evidence,
)


if __name__ == "__main__":
    result = audit_accuracy_reference_evidence()
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))
