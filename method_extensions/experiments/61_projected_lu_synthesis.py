"""Experiment 61: synthesize Projected-LU decisions and reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "surf-matplotlib-cache")
)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from american_risk_surfaces.method_extensions.projected_lu_study import (
    synthesize_projected_lu,
)


if __name__ == "__main__":
    outputs = synthesize_projected_lu()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
