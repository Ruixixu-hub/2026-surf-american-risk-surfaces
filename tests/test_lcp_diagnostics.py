"""Ticket 08 tests for obstacle and complementarity diagnostics."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "05_obstacle_complementarity_diagnostics.py"
SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_08_lcp_diagnostics_summary.csv"
)
BY_STEP_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_08_lcp_diagnostics_by_step.csv"
)
OBSTACLE_FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_08_lcp_obstacle_violation_by_step.png"
)
EQUATION_FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_08_lcp_equation_violation_by_step.png"
)
COMPLEMENTARITY_FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_08_lcp_complementarity_by_step.png"
)

EXPECTED_SUMMARY_COLUMNS = [
    "case_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "value_gap_tolerance",
    "equation_gap_tolerance",
    "complementarity_tolerance",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_value_gap",
    "max_obstacle_violation",
    "min_equation_gap",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "mean_max_abs_complementarity_product",
    "max_exercise_like_node_count",
    "max_continuation_like_node_count",
    "max_ambiguous_node_count",
    "status",
    "obstacle_figure_created",
    "equation_figure_created",
    "complementarity_figure_created",
]

EXPECTED_BY_STEP_COLUMNS = [
    "case_name",
    "option_type",
    "time_step",
    "tau",
    "psor_iterations",
    "psor_final_update",
    "min_value_gap",
    "max_obstacle_violation",
    "min_equation_gap",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "mean_abs_complementarity_product",
    "exercise_like_node_count",
    "continuation_like_node_count",
    "ambiguous_node_count",
]


class LCPDiagnosticsTests(unittest.TestCase):
    def test_gap_arrays_match_manual_small_example(self):
        from american_risk_surfaces.diagnostics.lcp import compute_lcp_gap_arrays

        values = np.array([1.0, 0.5, 2.0])
        payoff = np.array([0.75, 0.6, 1.5])
        matrix_action = np.array([1.2, 0.4, 2.5])
        rhs = np.array([1.0, 0.45, 2.0])

        gaps = compute_lcp_gap_arrays(values, payoff, matrix_action, rhs)

        npt.assert_allclose(gaps.value_gap, [0.25, -0.1, 0.5])
        npt.assert_allclose(gaps.obstacle_violation, [0.0, 0.1, 0.0])
        npt.assert_allclose(gaps.equation_gap, [0.2, -0.05, 0.5])
        npt.assert_allclose(gaps.equation_violation, [0.0, 0.05, 0.0])
        npt.assert_allclose(gaps.complementarity_product, [0.05, 0.005, 0.25])

    def test_toy_lcp_exercise_and_continuation_nodes(self):
        from american_risk_surfaces.diagnostics.lcp import (
            ReconstructedLCPStep,
            compute_lcp_gap_arrays,
            summarize_lcp_step,
        )
        from american_risk_surfaces.solvers.cn_psor import PSORResult

        values = np.array([1.0, 1.2])
        payoff = np.array([1.0, 0.5])
        matrix_action = np.array([1.3, 0.8])
        rhs = np.array([1.0, 0.8])
        gaps = compute_lcp_gap_arrays(values, payoff, matrix_action, rhs)
        step = ReconstructedLCPStep(
            step_index=1,
            tau=0.1,
            interior_spots=np.array([0.9, 1.1]),
            values=values,
            payoff=payoff,
            matrix_action=matrix_action,
            rhs=rhs,
            gaps=gaps,
        )
        psor_result = PSORResult(
            solution=values,
            converged=True,
            iterations=4,
            final_update=1e-9,
            tolerance=1e-8,
            omega=1.2,
            max_iter=10000,
        )

        row = summarize_lcp_step(step, psor_result)

        self.assertEqual(row.exercise_like_node_count, 1)
        self.assertEqual(row.continuation_like_node_count, 1)
        self.assertEqual(row.ambiguous_node_count, 0)
        self.assertEqual(row.psor_iterations, 4)

    def test_boundary_nodes_are_excluded_from_reconstructed_lcp_arrays(self):
        from american_risk_surfaces.diagnostics.lcp import reconstruct_lcp_step

        result = _small_american_put_result()
        step = reconstruct_lcp_step(result, 1)

        expected_interior_length = result.M - 1
        self.assertEqual(len(step.values), expected_interior_length)
        self.assertEqual(len(step.payoff), expected_interior_length)
        self.assertEqual(len(step.matrix_action), expected_interior_length)
        self.assertEqual(len(step.rhs), expected_interior_length)
        self.assertEqual(len(step.interior_spots), expected_interior_length)

    def test_american_put_diagnostics_are_small_and_finite(self):
        from american_risk_surfaces.diagnostics.lcp import diagnose_lcp_result

        diagnostics = diagnose_lcp_result(_small_american_put_result(), "put_smoke")
        summary = diagnostics.summary

        self.assertEqual(summary.status, "PASS")
        self.assertTrue(summary.all_psor_steps_converged)
        self.assertLessEqual(summary.max_obstacle_violation, 1e-8)
        self.assertLessEqual(summary.max_equation_violation, 1e-6)
        self.assertLessEqual(summary.max_abs_complementarity_product, 1e-6)
        self.assertEqual(len(diagnostics.step_rows), summary.psor_step_count)
        self.assertTrue(np.isfinite(summary.max_abs_complementarity_product))

    def test_dividend_call_diagnostics_are_small_and_finite(self):
        from american_risk_surfaces.diagnostics.lcp import diagnose_lcp_result

        result = american_crank_nicolson_psor_price(
            option_type="call",
            K=1.0,
            T=1.0,
            r=0.05,
            q=0.08,
            sigma=0.2,
            Smax=4.0,
            M=40,
            N=40,
        )
        diagnostics = diagnose_lcp_result(result, "dividend_call_smoke")
        summary = diagnostics.summary

        self.assertEqual(summary.status, "PASS")
        self.assertTrue(summary.all_psor_steps_converged)
        self.assertLessEqual(summary.max_obstacle_violation, 1e-8)
        self.assertLessEqual(summary.max_equation_violation, 1e-6)
        self.assertLessEqual(summary.max_abs_complementarity_product, 1e-6)
        self.assertTrue(np.isfinite(summary.max_equation_violation))

    def test_invalid_inputs_raise_value_error(self):
        from american_risk_surfaces.diagnostics.lcp import (
            compute_lcp_gap_arrays,
            diagnose_lcp_result,
            reconstruct_lcp_step,
        )

        with self.assertRaises(ValueError):
            compute_lcp_gap_arrays([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            reconstruct_lcp_step(_small_american_put_result(), 0)
        with self.assertRaises(ValueError):
            reconstruct_lcp_step(_small_american_put_result(), 999)
        with self.assertRaises(ValueError):
            diagnose_lcp_result(object(), "bad")
        with self.assertRaises(ValueError):
            diagnose_lcp_result(_small_american_put_result(), "bad_tol", value_gap_tolerance=-1)


class LCPDiagnosticsExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket08_module()
        cls.summary_rows, cls.by_step_rows, cls.metadata = cls.module.main()

    def test_experiment_writes_expected_csv_artifacts(self):
        self.assertEqual(len(self.summary_rows), 2)
        self.assertEqual(len(self.by_step_rows), 240)
        self.assertTrue(SUMMARY_CSV.exists())
        self.assertTrue(BY_STEP_CSV.exists())
        self.assertEqual(self.metadata["summary_csv"], str(SUMMARY_CSV))
        self.assertEqual(self.metadata["by_step_csv"], str(BY_STEP_CSV))

    def test_csv_tables_have_expected_columns(self):
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SUMMARY_COLUMNS)
            summary_rows = list(reader)

        with BY_STEP_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_BY_STEP_COLUMNS)
            by_step_rows = list(reader)

        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(len(by_step_rows), 240)
        self.assertTrue(all(row["status"] == "PASS" for row in summary_rows))

    def test_figures_created_when_matplotlib_available(self):
        if (
            self.metadata["obstacle_figure_created"] != "True"
            or self.metadata["equation_figure_created"] != "True"
            or self.metadata["complementarity_figure_created"] != "True"
        ):
            self.skipTest("matplotlib unavailable in this runtime")

        self.assertTrue(OBSTACLE_FIGURE_PATH.exists())
        self.assertTrue(EQUATION_FIGURE_PATH.exists())
        self.assertTrue(COMPLEMENTARITY_FIGURE_PATH.exists())
        self.assertGreater(OBSTACLE_FIGURE_PATH.stat().st_size, 0)
        self.assertGreater(EQUATION_FIGURE_PATH.stat().st_size, 0)
        self.assertGreater(COMPLEMENTARITY_FIGURE_PATH.stat().st_size, 0)

    def test_no_forbidden_boundary_greek_stress_dataset_or_neural_files_or_api(self):
        forbidden_fragments = ("boundary", "extract", "greek", "stress", "dataset", "neural")
        ticket08_paths = [
            path
            for path in PROJECT_ROOT.rglob("*ticket_08*")
            if ".git" not in path.parts and "__pycache__" not in path.parts
        ]
        public_names = [
            name
            for name, value in vars(self.module).items()
            if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
        ]

        for path in ticket08_paths:
            with self.subTest(path=path):
                lowered = path.name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

        for name in public_names:
            with self.subTest(public_name=name):
                lowered = name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

    def test_new_python_files_include_ticket_08_docstrings(self):
        import american_risk_surfaces.diagnostics.lcp as lcp

        self.assertIn("Ticket 08", inspect.getdoc(lcp))
        self.assertIn("Ticket 08", inspect.getdoc(self.module))
        test_doc = Path(__file__).read_text(encoding="utf-8")
        self.assertIn("Ticket 08", test_doc)


def _small_american_put_result():
    return american_crank_nicolson_psor_price(
        option_type="put",
        K=1.0,
        T=1.0,
        r=0.05,
        q=0.02,
        sigma=0.2,
        Smax=4.0,
        M=40,
        N=40,
    )


def _load_ticket08_module():
    spec = importlib.util.spec_from_file_location(
        "ticket08_obstacle_complementarity_diagnostics", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
