"""Ticket 12: tests for solver validation synthesis and artifact audit."""

from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "10_solver_validation_synthesis.py"
TABLE_DIR = PROJECT_ROOT / "results" / "01_solver_validation" / "tables"

EVIDENCE_CSV = TABLE_DIR / "ticket_12_solver_validation_evidence_summary.csv"
GATE_CSV = TABLE_DIR / "ticket_12_validation_gate_decision.csv"
AUDIT_CSV = TABLE_DIR / "ticket_12_artifact_audit.csv"

EVIDENCE_COLUMNS = [
    "ticket",
    "validation_area",
    "source_artifact",
    "key_metric",
    "metric_value",
    "status",
    "limitation",
]

GATE_COLUMNS = [
    "decision",
    "allowed_values",
    "basis",
    "next_recommended_stage",
    "required_limitations",
    "blocked_until_later",
    "artifact_audit_status",
    "evidence_status",
]

AUDIT_COLUMNS = [
    "artifact_group",
    "ticket",
    "artifact_type",
    "path",
    "required",
    "exists",
    "status",
    "notes",
]

ALLOWED_DECISIONS = {
    "PASS_SOLVER_VALIDATION_WITH_LIMITATIONS",
    "PASS_TO_REFERENCE_INTEGRATION_AND_RESEARCH_STAGE",
    "REVIEW_REQUIRED_BEFORE_DOWNSTREAM_STAGE",
}


def load_synthesis_module():
    spec = importlib.util.spec_from_file_location("ticket12_synthesis", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestSolverValidationSynthesis(unittest.TestCase):
    def test_expected_artifact_audit_reports_present_and_missing_files(self) -> None:
        module = load_synthesis_module()
        expected = [
            {
                "artifact_group": "synthetic",
                "ticket": "known",
                "artifact_type": "file",
                "path": "reports/03_solver/solver_validation_plan.md",
                "required": True,
                "notes": "known present file",
            },
            {
                "artifact_group": "synthetic",
                "ticket": "missing",
                "artifact_type": "file",
                "path": "reports/03_solver/no_such_ticket_12_file.md",
                "required": True,
                "notes": "known missing file",
            },
        ]

        rows = module.audit_artifacts(expected, PROJECT_ROOT)

        by_ticket = {row["ticket"]: row for row in rows}
        self.assertEqual("PRESENT", by_ticket["known"]["status"])
        self.assertEqual("True", by_ticket["known"]["exists"])
        self.assertEqual("MISSING", by_ticket["missing"]["status"])
        self.assertEqual("False", by_ticket["missing"]["exists"])

    def test_synthetic_missing_artifact_is_reported_not_fabricated(self) -> None:
        module = load_synthesis_module()
        rows = module.audit_artifacts(
            [
                {
                    "artifact_group": "synthetic",
                    "ticket": "synthetic_missing",
                    "artifact_type": "csv",
                    "path": "results/01_solver_validation/tables/not_real_ticket_12.csv",
                    "required": True,
                    "notes": "must remain missing",
                }
            ],
            PROJECT_ROOT,
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("MISSING", rows[0]["status"])
        self.assertFalse((PROJECT_ROOT / rows[0]["path"]).exists())

    def test_evidence_summary_csv_has_expected_columns(self) -> None:
        module = load_synthesis_module()
        module.main()
        with EVIDENCE_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(EVIDENCE_COLUMNS, reader.fieldnames)
            rows = list(reader)
        self.assertGreaterEqual(len(rows), 12)
        self.assertTrue(any(row["ticket"] == "Ticket 03" for row in rows))
        self.assertTrue(any(row["ticket"] == "Ticket 10A" for row in rows))

    def test_gate_decision_csv_has_expected_columns(self) -> None:
        module = load_synthesis_module()
        module.main()
        with GATE_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(GATE_COLUMNS, reader.fieldnames)
            rows = list(reader)
        self.assertEqual(1, len(rows))

    def test_gate_decision_value_is_allowed(self) -> None:
        module = load_synthesis_module()
        _, gate_rows, _ = module.main()

        self.assertEqual(1, len(gate_rows))
        self.assertIn(gate_rows[0]["decision"], ALLOWED_DECISIONS)

    def test_script_runs_and_writes_expected_csv_artifacts(self) -> None:
        module = load_synthesis_module()
        evidence_rows, gate_rows, audit_rows = module.main()

        self.assertTrue(EVIDENCE_CSV.exists())
        self.assertTrue(GATE_CSV.exists())
        self.assertTrue(AUDIT_CSV.exists())
        self.assertGreater(len(evidence_rows), 0)
        self.assertEqual(1, len(gate_rows))
        self.assertGreater(len(audit_rows), 20)

    def test_no_new_pricing_solver_api_is_introduced(self) -> None:
        module = load_synthesis_module()
        public_functions = [
            name
            for name, value in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
        ]

        forbidden_fragments = ("crank", "psor", "price", "solve", "payoff")
        for name in public_functions:
            self.assertFalse(
                any(fragment in name.lower() for fragment in forbidden_fragments),
                msg=f"Unexpected pricing/solver-like public API: {name}",
            )

        ticket12_solver_files = list((PROJECT_ROOT / "src" / "american_risk_surfaces" / "solvers").glob("*ticket_12*"))
        self.assertEqual([], ticket12_solver_files)

    def test_no_stress_dataset_neural_or_label_outputs_are_introduced(self) -> None:
        module = load_synthesis_module()
        names = [name for name in dir(module) if not name.startswith("_")]
        paths = [path.as_posix() for path in PROJECT_ROOT.rglob("*ticket_12*")]
        combined = "\n".join(names + paths).lower()

        for forbidden in ("stress", "dataset", "neural", "label"):
            self.assertNotIn(forbidden, combined)

    def test_new_python_files_include_ticket_12_docstrings(self) -> None:
        module = load_synthesis_module()
        self.assertIn("Ticket 12", module.__doc__ or "")
        self.assertIn("Ticket 12", Path(__file__).read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
