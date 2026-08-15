"""Experiment 58: freeze Projected-LU protocol and audit eligibility."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from american_risk_surfaces.method_extensions.projected_lu_study import (
    run_protocol_and_eligibility,
)


if __name__ == "__main__":
    result = run_protocol_and_eligibility()
    print(json.dumps(result["decision_data"], indent=2, sort_keys=True))
