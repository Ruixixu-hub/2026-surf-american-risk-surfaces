"""Experiment 59: validate single/double sweeps and freeze one mode."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from american_risk_surfaces.method_extensions.projected_lu_study import (
    run_validation_and_freeze,
)


if __name__ == "__main__":
    result = run_validation_and_freeze()
    print(json.dumps(result["frozen_data"], indent=2, sort_keys=True))
