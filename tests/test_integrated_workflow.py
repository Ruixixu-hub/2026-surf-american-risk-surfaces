"""Stage 6: integrated workflow and claim synthesis."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "18_integrated_workflow.py"
STAGE6_DOCSTRING = '"""Stage 6: integrated workflow and claim synthesis."""'


class IntegratedWorkflowTests(unittest.TestCase):
    def test_required_input_csvs_parse(self) -> None:
        stage6 = _load_stage6()
        for path in stage6.REQUIRED_INPUT_CSVS:
            rows = stage6.read_csv_rows(PROJECT_ROOT / path)
            self.assertGreater(len(rows), 0, path)

    def test_component_readiness_includes_required_components(self) -> None:
        stage6 = _load_stage6()
        rows = stage6.build_component_readiness_matrix(PROJECT_ROOT)
        components = {row["component"] for row in rows}
        for component in (
            "CN/PSOR solver benchmark",
            "v1 small-grid dataset",
            "price/premium surrogate",
            "boundary diagnostic component",
            "Delta diagnostic component",
            "Gamma component",
        ):
            self.assertIn(component, components)
        gamma = next(row for row in rows if row["component"] == "Gamma component")
        self.assertEqual("BLOCKED", gamma["status"])

    def test_claim_and_blocked_matrices_have_expected_claim_types(self) -> None:
        stage6 = _load_stage6()
        claims = stage6.build_claim_evidence_matrix()
        blocked = stage6.build_blocked_claims_matrix()
        self.assertEqual(stage6.CLAIM_EVIDENCE_FIELDNAMES, list(claims[0]))
        self.assertEqual(stage6.BLOCKED_CLAIMS_FIELDNAMES, list(blocked[0]))
        self.assertTrue(any(row["status"] == "SUPPORTED_WITH_LIMITATIONS" for row in claims))
        self.assertTrue(any("Gamma" in row["blocked_claim"] for row in blocked))
        self.assertTrue(any("production" in row["blocked_claim"].lower() for row in blocked))

    def test_final_readiness_decision_is_allowed(self) -> None:
        stage6 = _load_stage6()
        rows = stage6.build_component_readiness_matrix(PROJECT_ROOT)
        decision = stage6.stage6_review_decision(rows, stage6.build_claim_evidence_matrix(), stage6.build_blocked_claims_matrix())
        self.assertIn(
            decision,
            {"READY_FOR_FINAL_PAPER_DRAFTING", "REVIEW_REQUIRED_BEFORE_FINAL_PAPER"},
        )

    def test_temporary_run_writes_expected_outputs_without_real_report_path(self) -> None:
        stage6 = _load_stage6()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            report_dir = temp_root / "reports"
            results_dir = temp_root / "results"
            result = stage6.run_integrated_workflow(
                project_root=PROJECT_ROOT,
                report_dir=report_dir,
                results_dir=results_dir,
                create_figures=True,
            )
            self.assertEqual(report_dir / "integrated_workflow_report.tex", result.report_tex_path)
            self.assertTrue(result.report_tex_path.exists())
            self.assertTrue(result.final_paper_outline_path.exists())
            for path in result.report_csv_paths + result.result_csv_paths:
                self.assertTrue(path.exists(), path)
                self.assertGreater(len(_read_csv(path)), 0, path)
            for path in result.figure_paths:
                self.assertTrue(path.exists(), path)
            self.assertNotEqual(
                PROJECT_ROOT / "reports" / "07_integrated" / "integrated_workflow_report.tex",
                result.report_tex_path,
            )

    def test_no_model_training_or_solver_code_is_referenced(self) -> None:
        source = EXPERIMENT_PATH.read_text()
        forbidden_tokens = (
            "import torch",
            "from torch",
            "american_risk_surfaces.solvers",
            "run_surrogate_experiment",
            "run_boundary_diagnostics_experiment",
            "run_delta_diagnostics_experiment",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source)

    def test_new_python_files_include_stage_6_docstring(self) -> None:
        for path in (
            EXPERIMENT_PATH,
            PROJECT_ROOT / "tests" / "test_integrated_workflow.py",
        ):
            self.assertTrue(path.exists(), path)
            self.assertIn(STAGE6_DOCSTRING, path.read_text())


def _load_stage6():
    spec = importlib.util.spec_from_file_location("stage6_integrated_workflow", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load Stage 6 experiment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
