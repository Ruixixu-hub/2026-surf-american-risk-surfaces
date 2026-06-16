"""Ticket 05 tests for the American put validation experiment."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "02_american_put_validation.py"
VALIDATION_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_05_american_put_validation.csv"
)
SELECTED_SPOTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_05_american_put_selected_spots.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_05_american_put_value_vs_payoff.png"
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
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "min_obstacle_gap",
    "max_obstacle_violation",
    "min_american_minus_european",
    "max_american_minus_european",
    "medium_to_fine_reference",
    "figure_created",
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
    "european_put",
    "american_put",
    "american_minus_european",
    "american_minus_payoff",
]


class AmericanPutValidationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket05_module()
        cls.validation_rows, cls.selected_rows, cls.metadata = cls.module.main()

    def test_experiment_script_runs_and_writes_expected_artifacts(self):
        self.assertGreaterEqual(len(self.validation_rows), 2)
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

    def test_american_put_values_dominate_payoff(self):
        min_obstacle_gap = min(float(row["min_obstacle_gap"]) for row in self.validation_rows)
        max_obstacle_violation = max(
            float(row["max_obstacle_violation"]) for row in self.validation_rows
        )

        self.assertGreaterEqual(min_obstacle_gap, -1e-8)
        self.assertLessEqual(max_obstacle_violation, 1e-8)

    def test_american_put_values_dominate_european_put(self):
        min_american_minus_european = min(
            float(row["min_american_minus_european"]) for row in self.validation_rows
        )
        selected_differences = [
            float(row["american_minus_european"]) for row in self.selected_rows
        ]

        self.assertGreaterEqual(min_american_minus_european, -1e-6)
        self.assertGreaterEqual(min(selected_differences), -1e-6)

    def test_psor_convergence_summary_for_default_case(self):
        for row in self.validation_rows:
            with self.subTest(case_name=row["case_name"]):
                self.assertEqual(row["all_psor_steps_converged"], "True")
                self.assertEqual(int(row["psor_step_count"]), int(row["N"]))
                self.assertGreater(int(row["max_psor_iterations"]), 0)
                self.assertGreater(float(row["mean_psor_iterations"]), 0.0)
                self.assertLessEqual(float(row["max_final_update"]), 1e-8)

    def test_nearest_spot_index_is_predictable(self):
        spot_grid = np.array([0.0, 0.5, 1.0, 1.5])

        self.assertEqual(self.module.nearest_spot_index(spot_grid, 0.74), 1)
        self.assertEqual(self.module.nearest_spot_index(spot_grid, 0.76), 2)
        self.assertEqual(self.module.nearest_spot_index(spot_grid, 2.0), 3)

    def test_value_figure_created_when_matplotlib_available(self):
        if self.metadata["figure_created"] != "True":
            self.skipTest("matplotlib unavailable in this runtime")

        self.assertTrue(FIGURE_PATH.exists())
        self.assertGreater(FIGURE_PATH.stat().st_size, 0)
        self.assertEqual(self.metadata["figure_path"], str(FIGURE_PATH))

    def test_no_forbidden_boundary_greek_stress_dataset_or_neural_files_created(self):
        forbidden_fragments = ("boundary", "extract", "greek", "stress", "dataset", "neural")
        ticket05_paths = [
            path
            for path in PROJECT_ROOT.rglob("*ticket_05*")
            if ".git" not in path.parts and "__pycache__" not in path.parts
        ]
        public_names = [
            name
            for name, value in vars(self.module).items()
            if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
        ]

        for path in ticket05_paths:
            with self.subTest(path=path):
                lowered = path.name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

        for name in public_names:
            with self.subTest(public_name=name):
                lowered = name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))

    def test_new_python_files_include_ticket_05_docstrings(self):
        self.assertIn("Ticket 05", inspect.getdoc(self.module))

        test_doc = Path(__file__).read_text(encoding="utf-8")
        self.assertIn("Ticket 05", test_doc)


def _load_ticket05_module():
    spec = importlib.util.spec_from_file_location("ticket05_american_put_validation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
