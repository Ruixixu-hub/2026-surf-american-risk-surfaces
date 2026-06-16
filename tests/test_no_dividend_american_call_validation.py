"""Ticket 06 tests for the no-dividend American call validation experiment."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "03_no_dividend_american_call_validation.py"
VALIDATION_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_06_no_dividend_american_call_validation.csv"
)
SELECTED_SPOTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_06_no_dividend_american_call_selected_spots.csv"
)
VALUE_FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_06_no_dividend_american_call_value_comparison.png"
)
ERROR_FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_06_no_dividend_american_call_theorem_error.png"
)


EXPECTED_VALIDATION_COLUMNS = [
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
    "target_lower_moneyness",
    "target_upper_moneyness",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_obstacle_gap",
    "max_obstacle_violation",
    "min_american_minus_european",
    "max_american_minus_european",
    "max_abs_american_european_error",
    "rmse_american_european_error",
    "max_error_spot",
    "medium_to_fine_selected_spot_max_abs_diff",
    "value_figure_created",
    "error_figure_created",
]

EXPECTED_SELECTED_COLUMNS = [
    "case_name",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "M",
    "N",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "payoff",
    "european_call",
    "american_call",
    "american_minus_european",
    "american_minus_payoff",
]


class NoDividendAmericanCallValidationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket06_module()
        cls.validation_rows, cls.selected_rows, cls.metadata = cls.module.main()

    def test_experiment_script_runs_and_writes_expected_artifacts(self):
        self.assertEqual(len(self.validation_rows), 2)
        self.assertEqual(len(self.selected_rows), 10)
        self.assertTrue(VALIDATION_CSV.exists())
        self.assertTrue(SELECTED_SPOTS_CSV.exists())
        self.assertEqual(self.metadata["validation_csv"], str(VALIDATION_CSV))
        self.assertEqual(self.metadata["selected_spots_csv"], str(SELECTED_SPOTS_CSV))

    def test_validation_table_has_expected_columns(self):
        with VALIDATION_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_VALIDATION_COLUMNS)
            rows = list(reader)

        self.assertEqual(len(rows), len(self.validation_rows))

    def test_selected_spots_table_has_expected_columns(self):
        with SELECTED_SPOTS_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SELECTED_COLUMNS)
            rows = list(reader)

        self.assertEqual(len(rows), len(self.selected_rows))

    def test_all_validation_cases_use_q_zero(self):
        for row in self.validation_rows + self.selected_rows:
            with self.subTest(case_name=row["case_name"]):
                self.assertAlmostEqual(float(row["q"]), 0.0)

    def test_american_call_values_dominate_payoff(self):
        min_obstacle_gap = min(float(row["min_obstacle_gap"]) for row in self.validation_rows)
        max_obstacle_violation = max(
            float(row["max_obstacle_violation"]) for row in self.validation_rows
        )
        selected_gaps = [float(row["american_minus_payoff"]) for row in self.selected_rows]

        self.assertGreaterEqual(min_obstacle_gap, -1e-8)
        self.assertLessEqual(max_obstacle_violation, 1e-8)
        self.assertGreaterEqual(min(selected_gaps), -1e-8)

    def test_american_call_values_are_close_to_european_calls(self):
        max_errors = [
            float(row["max_abs_american_european_error"]) for row in self.validation_rows
        ]
        rmses = [float(row["rmse_american_european_error"]) for row in self.validation_rows]

        self.assertLessEqual(max(max_errors), 1e-3)
        self.assertLessEqual(max(rmses), 5e-4)

    def test_no_false_early_exercise_premium_is_small(self):
        max_american_minus_european = max(
            float(row["max_american_minus_european"]) for row in self.validation_rows
        )
        selected_premiums = [
            float(row["american_minus_european"]) for row in self.selected_rows
        ]

        self.assertLessEqual(max_american_minus_european, 1e-3)
        self.assertLessEqual(max(selected_premiums), 1e-3)

    def test_psor_convergence_summary_for_default_case(self):
        for row in self.validation_rows:
            with self.subTest(case_name=row["case_name"]):
                self.assertEqual(row["all_psor_steps_converged"], "True")
                self.assertEqual(int(row["psor_step_count"]), int(row["N"]))
                self.assertGreater(int(row["max_psor_iterations"]), 0)
                self.assertGreater(float(row["mean_psor_iterations"]), 0.0)
                self.assertLessEqual(float(row["max_final_update"]), 1e-8)

    def test_medium_to_fine_error_is_small_and_improves(self):
        rows_by_case = {row["case_name"]: row for row in self.validation_rows}
        medium_error = float(rows_by_case["medium"]["max_abs_american_european_error"])
        fine_error = float(rows_by_case["fine"]["max_abs_american_european_error"])
        selected_difference = float(
            rows_by_case["fine"]["medium_to_fine_selected_spot_max_abs_diff"]
        )

        self.assertLessEqual(fine_error, medium_error + 1e-12)
        self.assertGreaterEqual(selected_difference, 0.0)
        self.assertLess(selected_difference, 0.012)

    def test_nearest_spot_index_is_predictable(self):
        spot_grid = np.array([0.0, 0.5, 1.0, 1.5])

        self.assertEqual(self.module.nearest_spot_index(spot_grid, 0.74), 1)
        self.assertEqual(self.module.nearest_spot_index(spot_grid, 0.76), 2)
        self.assertEqual(self.module.nearest_spot_index(spot_grid, 2.0), 3)

    def test_figures_created_when_matplotlib_available(self):
        if (
            self.metadata["value_figure_created"] != "True"
            or self.metadata["error_figure_created"] != "True"
        ):
            self.skipTest("matplotlib unavailable in this runtime")

        self.assertTrue(VALUE_FIGURE_PATH.exists())
        self.assertTrue(ERROR_FIGURE_PATH.exists())
        self.assertGreater(VALUE_FIGURE_PATH.stat().st_size, 0)
        self.assertGreater(ERROR_FIGURE_PATH.stat().st_size, 0)
        self.assertEqual(self.metadata["value_figure_path"], str(VALUE_FIGURE_PATH))
        self.assertEqual(self.metadata["error_figure_path"], str(ERROR_FIGURE_PATH))

    def test_no_forbidden_boundary_greek_stress_dataset_or_neural_files_or_api(self):
        forbidden_fragments = ("boundary", "extract", "greek", "stress", "dataset", "neural")
        ticket06_paths = [
            path
            for path in PROJECT_ROOT.rglob("*ticket_06*")
            if ".git" not in path.parts and "__pycache__" not in path.parts
        ]
        public_names = [
            name
            for name, value in vars(self.module).items()
            if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
        ]

        for path in ticket06_paths:
            with self.subTest(path=path):
                lowered = path.name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

        for name in public_names:
            with self.subTest(public_name=name):
                lowered = name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

    def test_new_python_files_include_ticket_06_docstrings(self):
        self.assertIn("Ticket 06", inspect.getdoc(self.module))

        test_doc = Path(__file__).read_text(encoding="utf-8")
        self.assertIn("Ticket 06", test_doc)


def _load_ticket06_module():
    spec = importlib.util.spec_from_file_location(
        "ticket06_no_dividend_american_call_validation", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
