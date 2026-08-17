"""Focused tests for the poster-only evidence layer."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from american_risk_surfaces.method_extensions.poster_unified_study import (
    BENCHMARK_ROLES,
    audit_accuracy_reference_evidence,
)


class PosterUnifiedStudyTests(unittest.TestCase):
    def test_benchmark_labels_use_basic_not_historical(self) -> None:
        self.assertEqual(
            "Basic / Original Classical Benchmark", BENCHMARK_ROLES["psor"]
        )
        self.assertNotIn("Historical", BENCHMARK_ROLES["psor"])

    def test_existing_accuracy_reference_evidence_passes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = audit_accuracy_reference_evidence(Path(directory))
            decision = result["decision_data"]
            self.assertEqual("REUSE_EXISTING_REFERENCE_EVIDENCE", decision["status"])
            self.assertEqual(
                "dirk_lstable_quadratic", decision["selected_time_integrator"]
            )
            self.assertEqual(
                "sinh_strike_concentrated", decision["selected_spatial_grid"]
            )


if __name__ == "__main__":
    unittest.main()
