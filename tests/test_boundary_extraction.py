"""Ticket 09 tests for continuation premium and boundary extraction."""

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
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "06_boundary_extraction.py"
BOUNDARY_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "diagnostics"
    / "boundary.py"
)
SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_09_boundary_extraction_summary.csv"
)
BY_TIME_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_09_boundary_by_time.csv"
)
SELECTED_TIMES_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_09_boundary_selected_times.csv"
)
PUT_BOUNDARY_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_09_american_put_boundary_curve.png"
)
CALL_BOUNDARY_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_09_dividend_call_boundary_curve.png"
)
PREMIUM_PROFILE_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_09_continuation_premium_profiles.png"
)
BOUNDARY_STATUS_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_09_boundary_found_status.png"
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
    "threshold",
    "search_direction",
    "total_time_rows",
    "positive_tau_rows",
    "found_boundary_count",
    "no_boundary_count",
    "maturity_ambiguous_count",
    "all_continuation_count",
    "all_exercise_count",
    "expected_exercise_side_absent_count",
    "no_clean_transition_count",
    "insufficient_interior_nodes_count",
    "first_boundary_tau",
    "last_boundary_tau",
    "min_boundary_spot",
    "max_boundary_spot",
    "status",
    "put_boundary_figure_created",
    "call_boundary_figure_created",
    "premium_profile_figure_created",
    "boundary_status_figure_created",
]

EXPECTED_BY_TIME_COLUMNS = [
    "case_name",
    "option_type",
    "time_index",
    "tau",
    "boundary_found",
    "boundary_spot",
    "threshold",
    "search_direction",
    "extraction_method",
    "no_boundary_reason",
    "exercise_like_node_count",
    "continuation_like_node_count",
]

EXPECTED_SELECTED_TIME_COLUMNS = [
    "case_name",
    "option_type",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "spot",
    "moneyness",
    "value",
    "payoff",
    "premium",
    "threshold",
    "premium_class",
]


class BoundaryExtractionUnitTests(unittest.TestCase):
    def test_continuation_premium_subtracts_payoff(self):
        from american_risk_surfaces.diagnostics.boundary import continuation_premium

        values = np.array([[1.0, 2.5, 4.0], [1.2, 2.0, 4.6]])
        payoff = np.array([0.5, 2.0, 4.0])

        premium = continuation_premium(values, payoff)

        npt.assert_allclose(premium, [[0.5, 0.5, 0.0], [0.7, 0.0, 0.6]])

    def test_linear_interpolation_known_crossing(self):
        from american_risk_surfaces.diagnostics.boundary import (
            linear_interpolate_threshold_crossing,
        )

        crossing = linear_interpolate_threshold_crossing(1.0, 0.0, 2.0, 0.2, 0.05)

        self.assertAlmostEqual(crossing, 1.25)

    def test_synthetic_put_boundary_uses_low_spot_transition(self):
        from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time

        point = extract_boundary_at_time(
            spot_grid=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            premium_row=np.array([0.0, 0.0, 0.1, 0.2, 0.3]),
            option_type="put",
            tau=0.5,
            time_index=10,
            threshold=0.01,
        )

        self.assertTrue(point.boundary_found)
        self.assertEqual(point.search_direction, "low_to_high")
        self.assertEqual(point.extraction_method, "linear_threshold_crossing")
        self.assertAlmostEqual(point.boundary_spot, 0.55)

    def test_synthetic_dividend_call_boundary_uses_high_spot_transition(self):
        from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time

        point = extract_boundary_at_time(
            spot_grid=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            premium_row=np.array([0.4, 0.3, 0.2, 0.0, 0.0]),
            option_type="call",
            tau=0.5,
            time_index=10,
            threshold=0.01,
        )

        self.assertTrue(point.boundary_found)
        self.assertEqual(point.search_direction, "high_to_low")
        self.assertEqual(point.extraction_method, "linear_threshold_crossing")
        self.assertAlmostEqual(point.boundary_spot, 1.475)

    def test_no_boundary_metadata_is_explicit(self):
        from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time

        spot_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        all_continuation = extract_boundary_at_time(
            spot_grid, np.full(5, 0.2), "put", tau=0.5, time_index=1, threshold=0.01
        )
        all_exercise = extract_boundary_at_time(
            spot_grid, np.zeros(5), "call", tau=0.5, time_index=1, threshold=0.01
        )
        no_clean_transition = extract_boundary_at_time(
            spot_grid,
            np.array([0.0, 0.0, 0.2, 0.0, 0.2]),
            "put",
            tau=0.5,
            time_index=1,
            threshold=0.01,
        )

        self.assertFalse(all_continuation.boundary_found)
        self.assertEqual(all_continuation.no_boundary_reason, "all_continuation_like")
        self.assertFalse(all_exercise.boundary_found)
        self.assertEqual(all_exercise.no_boundary_reason, "all_exercise_like")
        self.assertFalse(no_clean_transition.boundary_found)
        self.assertEqual(no_clean_transition.no_boundary_reason, "no_clean_transition")

    def test_real_american_put_curve_finds_positive_tau_boundaries(self):
        from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve

        curve = extract_boundary_curve(_small_american_put_result(), "put_smoke")
        found_rows = [
            point for point in curve.points if point.tau > 0.0 and point.boundary_found
        ]

        self.assertGreater(len(found_rows), 0)
        self.assertEqual(curve.option_type, "put")
        self.assertGreaterEqual(curve.premium_grid.min(), -1e-8)

    def test_real_dividend_call_curve_finds_positive_tau_boundaries(self):
        from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve

        curve = extract_boundary_curve(_small_dividend_call_result(), "dividend_call_smoke")
        found_rows = [
            point for point in curve.points if point.tau > 0.0 and point.boundary_found
        ]

        self.assertGreater(len(found_rows), 0)
        self.assertEqual(curve.option_type, "call")
        self.assertGreaterEqual(curve.premium_grid.min(), -1e-8)

    def test_no_dividend_call_control_does_not_force_boundary(self):
        from american_risk_surfaces.diagnostics.boundary import (
            extract_boundary_curve,
            summarize_boundary_curve,
        )

        curve = extract_boundary_curve(_small_no_dividend_call_result(), "no_dividend_control")
        summary = summarize_boundary_curve(curve)

        self.assertEqual(summary.found_boundary_count, 0)
        self.assertEqual(summary.status, "NO_BOUNDARY_FOUND")

    def test_invalid_inputs_raise_value_error(self):
        from american_risk_surfaces.diagnostics.boundary import (
            continuation_premium,
            extract_boundary_at_time,
            extract_boundary_curve,
            linear_interpolate_threshold_crossing,
        )

        with self.assertRaises(ValueError):
            continuation_premium(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            linear_interpolate_threshold_crossing(1.0, 0.1, 1.0, 0.2, 0.05)
        with self.assertRaises(ValueError):
            extract_boundary_at_time([0.0, 1.0], [0.0, 0.1], "put", 0.5, 1)
        with self.assertRaises(ValueError):
            extract_boundary_at_time([0.0, 1.0, 2.0], [0.0, 0.1, 0.2], "bad", 0.5, 1)
        with self.assertRaises(ValueError):
            extract_boundary_at_time([0.0, 1.0, 2.0], [0.0, 0.1, 0.2], "put", 0.5, 1, threshold=-1)
        with self.assertRaises(ValueError):
            extract_boundary_curve(object(), "bad")


class BoundaryExtractionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket09_module()
        (
            cls.summary_rows,
            cls.by_time_rows,
            cls.selected_time_rows,
            cls.metadata,
        ) = cls.module.main()

    def test_experiment_writes_expected_csv_artifacts(self):
        self.assertEqual(len(self.summary_rows), 3)
        self.assertEqual(len(self.by_time_rows), 363)
        self.assertEqual(len(self.selected_time_rows), 1089)
        self.assertTrue(SUMMARY_CSV.exists())
        self.assertTrue(BY_TIME_CSV.exists())
        self.assertTrue(SELECTED_TIMES_CSV.exists())
        self.assertEqual(self.metadata["summary_csv"], str(SUMMARY_CSV))
        self.assertEqual(self.metadata["by_time_csv"], str(BY_TIME_CSV))
        self.assertEqual(self.metadata["selected_times_csv"], str(SELECTED_TIMES_CSV))

    def test_csv_tables_have_expected_columns(self):
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SUMMARY_COLUMNS)
            summary_rows = list(reader)

        with BY_TIME_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_BY_TIME_COLUMNS)
            by_time_rows = list(reader)

        with SELECTED_TIMES_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SELECTED_TIME_COLUMNS)
            selected_rows = list(reader)

        self.assertEqual(len(summary_rows), 3)
        self.assertEqual(len(by_time_rows), 363)
        self.assertEqual(len(selected_rows), 1089)

    def test_summary_contains_expected_boundary_behaviour(self):
        rows = {row["case_name"]: row for row in self.summary_rows}

        self.assertGreater(int(rows["american_put_medium"]["found_boundary_count"]), 0)
        self.assertGreater(int(rows["dividend_call_medium"]["found_boundary_count"]), 0)
        self.assertEqual(int(rows["no_dividend_call_control"]["found_boundary_count"]), 0)
        self.assertEqual(rows["no_dividend_call_control"]["status"], "NO_BOUNDARY_FOUND")

    def test_by_time_rows_record_maturity_ambiguity(self):
        maturity_rows = [
            row for row in self.by_time_rows if int(row["time_index"]) == 0
        ]

        self.assertEqual(len(maturity_rows), 3)
        self.assertTrue(all(row["no_boundary_reason"] == "maturity_row_ambiguous" for row in maturity_rows))
        self.assertTrue(all(row["boundary_found"] == "False" for row in maturity_rows))

    def test_selected_time_rows_record_premium_classes(self):
        classes = {row["premium_class"] for row in self.selected_time_rows}

        self.assertIn("exercise_like", classes)
        self.assertIn("continuation_like", classes)

    def test_figures_created_when_matplotlib_available(self):
        self.assertTrue(PUT_BOUNDARY_FIGURE.exists())
        self.assertTrue(CALL_BOUNDARY_FIGURE.exists())
        self.assertTrue(PREMIUM_PROFILE_FIGURE.exists())
        self.assertTrue(BOUNDARY_STATUS_FIGURE.exists())
        self.assertGreater(PUT_BOUNDARY_FIGURE.stat().st_size, 0)
        self.assertGreater(CALL_BOUNDARY_FIGURE.stat().st_size, 0)
        self.assertGreater(PREMIUM_PROFILE_FIGURE.stat().st_size, 0)
        self.assertGreater(BOUNDARY_STATUS_FIGURE.stat().st_size, 0)
        self.assertEqual(self.metadata["put_boundary_figure_path"], str(PUT_BOUNDARY_FIGURE))
        self.assertEqual(self.metadata["call_boundary_figure_path"], str(CALL_BOUNDARY_FIGURE))
        self.assertEqual(self.metadata["premium_profile_figure_path"], str(PREMIUM_PROFILE_FIGURE))
        self.assertEqual(self.metadata["boundary_status_figure_path"], str(BOUNDARY_STATUS_FIGURE))

    def test_no_forbidden_greek_stress_dataset_or_neural_files_or_api(self):
        import american_risk_surfaces.diagnostics.boundary as boundary

        forbidden = ("greek", "delta", "gamma", "stress", "dataset", "neural")
        public_names = getattr(boundary, "__all__", ())
        for name in public_names:
            lowered = name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), name)

        new_paths = [BOUNDARY_MODULE_PATH, SCRIPT_PATH]
        for path in new_paths:
            lowered = path.name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), str(path))

    def test_new_python_files_include_ticket_09_docstrings(self):
        import american_risk_surfaces.diagnostics.boundary as boundary

        self.assertIn("Ticket 09", inspect.getdoc(boundary))
        self.assertIn("Ticket 09", inspect.getdoc(self.module))


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


def _small_dividend_call_result():
    return american_crank_nicolson_psor_price(
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


def _small_no_dividend_call_result():
    return american_crank_nicolson_psor_price(
        option_type="call",
        K=1.0,
        T=1.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        Smax=4.0,
        M=40,
        N=40,
    )


def _load_ticket09_module():
    spec = importlib.util.spec_from_file_location("ticket09_boundary_extraction", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
