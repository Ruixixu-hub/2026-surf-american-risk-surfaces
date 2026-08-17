"""Experiment 64: create poster-ready evidence tables, figures, and reports."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from american_risk_surfaces.method_extensions.poster_unified_study import (
    synthesize_poster_comparison,
)


if __name__ == "__main__":
    outputs = synthesize_poster_comparison()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
