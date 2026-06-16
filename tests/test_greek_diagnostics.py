"""Ticket 10 tests for Delta and Gamma diagnostics."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.diagnostics.boundary import BoundaryCurve, BoundaryPoint
from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "07_greek_diagnostics.py"
GREEKS_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "diagnostics"
    / "greeks.py"
)
SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10_greek_diagnostics_summary.csv"
)
SELECTED_PROFILES_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10_greek_selected_profiles.csv"
)
BY_TIME_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10_greek_by_time.csv"
)
PUT_PROFILE_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10_american_put_delta_gamma_profiles.png"
)
CALL_PROFILE_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10_dividend_call_delta_gamma_profiles.png"
)
GAMMA_BOUNDARY_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10_gamma_boundary_diagnostic.png"
)
GAMMA_BY_TIME_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10_gamma_by_time.png"
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
    "kink_band_steps",
    "boundary_band_steps",
    "delta_bound_tolerance",
    "gamma_negative_tolerance",
    "finite_delta_count",
    "finite_gamma_count",
    "nonfinite_delta_count",
    "nonfinite_gamma_count",
    "min_delta",
    "max_delta",
    "min_gamma",
    "max_gamma",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "strict_delta_lower_violation_count",
    "strict_delta_upper_violation_count",
    "strict_negative_gamma_count",
    "status",
    "put_profile_figure_created",
    "call_profile_figure_created",
    "gamma_boundary_figure_created",
    "gamma_by_time_figure_created",
]

EXPECTED_SELECTED_COLUMNS = [
    "case_name",
    "option_type",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "delta",
    "gamma",
    "boundary_near",
    "kink_near",
    "maturity_row",
    "strict_interior",
]

EXPECTED_BY_TIME_COLUMNS = [
    "case_name",
    "option_type",
    "time_index",
    "tau",
    "finite_delta_count",
    "finite_gamma_count",
    "boundary_near_node_count",
    "kink_near_node_count",
    "strict_interior_count",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "min_delta_strict",
    "max_delta_strict",
    "strict_delta_lower_violation_count",
    "strict_delta_upper_violation_count",
    "strict_negative_gamma_count",
    "warning_flag",
]


class GreekDiagnosticsUnitTests(unittest.TestCase):
    def test_delta_gamma_constant_function(self):
        from american_risk_surfaces.diagnostics.greeks import (
            finite_difference_delta,
            finite_difference_gamma,
        )

        spot_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.full_like(spot_grid, 2.5)

        delta = finite_difference_delta(spot_grid, values)
        gamma = finite_difference_gamma(spot_grid, values)

        self.assertTrue(np.isnan(delta[0]))
        self.assertTrue(np.isnan(delta[-1]))
        self.assertTrue(np.isnan(gamma[0]))
        self.assertTrue(np.isnan(gamma[-1]))
        npt.assert_allclose(delta[1:-1], 0.0)
        npt.assert_allclose(gamma[1:-1], 0.0)

    def test_delta_gamma_linear_function(self):
        from american_risk_surfaces.diagnostics.greeks import (
            finite_difference_delta,
            finite_difference_gamma,
        )

        spot_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = spot_grid.copy()

        delta = finite_difference_delta(spot_grid, values)
        gamma = finite_difference_gamma(spot_grid, values)

        npt.assert_allclose(delta[1:-1], 1.0)
        npt.assert_allclose(gamma[1:-1], 0.0)

    def test_delta_gamma_quadratic_function(self):
        from american_risk_surfaces.diagnostics.greeks import (
            finite_difference_delta,
            finite_difference_gamma,
        )

        spot_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = spot_grid**2

        delta = finite_difference_delta(spot_grid, values)
        gamma = finite_difference_gamma(spot_grid, values)

        npt.assert_allclose(delta[1:-1], 2.0 * spot_grid[1:-1])
        npt.assert_allclose(gamma[1:-1], 2.0)

    def test_delta_gamma_shapes_and_nan_boundaries(self):
        from american_risk_surfaces.diagnostics.greeks import (
            finite_difference_delta,
            finite_difference_gamma,
        )

        spot_grid = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.vstack([spot_grid, spot_grid**2])

        delta = finite_difference_delta(spot_grid, values)
        gamma = finite_difference_gamma(spot_grid, values)

        self.assertEqual(delta.shape, values.shape)
        self.assertEqual(gamma.shape, values.shape)
        self.assertTrue(np.all(np.isnan(delta[:, 0])))
        self.assertTrue(np.all(np.isnan(delta[:, -1])))
        self.assertTrue(np.all(np.isnan(gamma[:, 0])))
        self.assertTrue(np.all(np.isnan(gamma[:, -1])))
        npt.assert_allclose(delta[0, 1:-1], 1.0)
        npt.assert_allclose(gamma[1, 1:-1], 2.0)

    def test_invalid_spot_and_value_inputs_raise_value_error(self):
        from american_risk_surfaces.diagnostics.greeks import (
            diagnose_greek_result,
            finite_difference_delta,
            finite_difference_gamma,
        )

        with self.assertRaises(ValueError):
            finite_difference_delta(np.array([[0.0, 1.0, 2.0]]), np.array([0.0, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            finite_difference_delta(np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            finite_difference_gamma(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            finite_difference_gamma(np.array([0.0, 1.0, 2.0]), np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            diagnose_greek_result(object(), "bad")
        with self.assertRaises(ValueError):
            diagnose_greek_result(_small_american_put_result(), "")
        with self.assertRaises(ValueError):
            diagnose_greek_result(_small_american_put_result(), "bad", kink_band_steps=-1)

    def test_boundary_near_mask_marks_nodes_near_synthetic_boundary(self):
        from american_risk_surfaces.diagnostics.greeks import boundary_near_mask

        curve = _synthetic_boundary_curve(boundary_spots={1: 1.5})
        mask = boundary_near_mask(curve, curve.spot_grid, curve.tau_grid, boundary_band_steps=1)

        self.assertEqual(mask.shape, curve.value_grid.shape)
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[1, 2])
        self.assertTrue(mask[1, 3])
        self.assertFalse(mask[2].any())

    def test_boundary_near_mask_empty_when_no_boundary_found(self):
        from american_risk_surfaces.diagnostics.greeks import boundary_near_mask

        curve = _synthetic_boundary_curve(boundary_spots={})
        mask = boundary_near_mask(curve, curve.spot_grid, curve.tau_grid)

        self.assertFalse(mask.any())

    def test_payoff_kink_and_maturity_masks(self):
        from american_risk_surfaces.diagnostics.greeks import (
            maturity_row_mask,
            payoff_kink_mask,
        )

        spot_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        tau_grid = np.array([0.0, 0.5, 1.0])

        kink_mask = payoff_kink_mask(spot_grid, tau_grid, K=1.0, kink_band_steps=1)
        maturity_mask = maturity_row_mask(spot_grid, tau_grid)

        self.assertEqual(kink_mask.shape, (3, 5))
        self.assertTrue(np.all(kink_mask[:, 1:4]))
        self.assertFalse(kink_mask[:, 0].any())
        self.assertFalse(kink_mask[:, -1].any())
        self.assertTrue(maturity_mask[0].all())
        self.assertFalse(maturity_mask[1:].any())

    def test_real_american_put_greek_diagnostics_produce_finite_values(self):
        from american_risk_surfaces.diagnostics.greeks import diagnose_greek_result

        diagnostics = diagnose_greek_result(_small_american_put_result(), "put_smoke")
        summary = diagnostics.summary

        self.assertEqual(summary.option_type, "put")
        self.assertGreater(summary.finite_delta_count, 0)
        self.assertGreater(summary.finite_gamma_count, 0)
        self.assertGreater(summary.strict_interior_node_count, 0)
        self.assertTrue(np.isfinite(summary.max_abs_gamma))
        self.assertEqual(diagnostics.arrays.delta.shape, diagnostics.result_shape)
        self.assertEqual(len(diagnostics.by_time_rows), len(diagnostics.tau_grid))

    def test_real_dividend_call_greek_diagnostics_produce_finite_values(self):
        from american_risk_surfaces.diagnostics.greeks import diagnose_greek_result

        diagnostics = diagnose_greek_result(_small_dividend_call_result(), "call_smoke")
        summary = diagnostics.summary

        self.assertEqual(summary.option_type, "call")
        self.assertGreater(summary.finite_delta_count, 0)
        self.assertGreater(summary.finite_gamma_count, 0)
        self.assertGreater(summary.strict_interior_node_count, 0)
        self.assertTrue(np.isfinite(summary.max_abs_gamma_strict))


class GreekDiagnosticsExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket10_module()
        (
            cls.summary_rows,
            cls.selected_rows,
            cls.by_time_rows,
            cls.metadata,
        ) = cls.module.main()

    def test_experiment_writes_expected_csv_artifacts(self):
        self.assertEqual(len(self.summary_rows), 2)
        self.assertEqual(len(self.selected_rows), 36)
        self.assertEqual(len(self.by_time_rows), 242)
        self.assertTrue(SUMMARY_CSV.exists())
        self.assertTrue(SELECTED_PROFILES_CSV.exists())
        self.assertTrue(BY_TIME_CSV.exists())
        self.assertEqual(self.metadata["summary_csv"], str(SUMMARY_CSV))
        self.assertEqual(self.metadata["selected_profiles_csv"], str(SELECTED_PROFILES_CSV))
        self.assertEqual(self.metadata["by_time_csv"], str(BY_TIME_CSV))

    def test_csv_tables_have_expected_columns(self):
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SUMMARY_COLUMNS)
            summary_rows = list(reader)

        with SELECTED_PROFILES_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SELECTED_COLUMNS)
            selected_rows = list(reader)

        with BY_TIME_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_BY_TIME_COLUMNS)
            by_time_rows = list(reader)

        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(len(selected_rows), 36)
        self.assertEqual(len(by_time_rows), 242)

    def test_summary_records_caution_masks_and_status(self):
        rows = {row["case_name"]: row for row in self.summary_rows}

        for case_name in ("american_put_medium", "dividend_call_medium"):
            row = rows[case_name]
            self.assertGreater(int(row["finite_delta_count"]), 0)
            self.assertGreater(int(row["finite_gamma_count"]), 0)
            self.assertGreater(int(row["strict_interior_node_count"]), 0)
            self.assertGreaterEqual(int(row["boundary_near_node_count"]), 0)
            self.assertGreater(int(row["kink_near_node_count"]), 0)
            self.assertIn(row["status"], {"PASS_WITH_CAUTIONS", "REVIEW"})

    def test_selected_profiles_record_masks(self):
        strict_values = {row["strict_interior"] for row in self.selected_rows}

        self.assertIn("True", strict_values)
        self.assertIn("False", strict_values)

    def test_figures_created_when_matplotlib_available(self):
        self.assertTrue(PUT_PROFILE_FIGURE.exists())
        self.assertTrue(CALL_PROFILE_FIGURE.exists())
        self.assertTrue(GAMMA_BOUNDARY_FIGURE.exists())
        self.assertTrue(GAMMA_BY_TIME_FIGURE.exists())
        self.assertGreater(PUT_PROFILE_FIGURE.stat().st_size, 0)
        self.assertGreater(CALL_PROFILE_FIGURE.stat().st_size, 0)
        self.assertGreater(GAMMA_BOUNDARY_FIGURE.stat().st_size, 0)
        self.assertGreater(GAMMA_BY_TIME_FIGURE.stat().st_size, 0)
        self.assertEqual(self.metadata["put_profile_figure_path"], str(PUT_PROFILE_FIGURE))
        self.assertEqual(self.metadata["call_profile_figure_path"], str(CALL_PROFILE_FIGURE))
        self.assertEqual(self.metadata["gamma_boundary_figure_path"], str(GAMMA_BOUNDARY_FIGURE))
        self.assertEqual(self.metadata["gamma_by_time_figure_path"], str(GAMMA_BY_TIME_FIGURE))

    def test_no_forbidden_rannacher_stress_dataset_or_neural_files_or_api(self):
        import american_risk_surfaces.diagnostics.greeks as greeks

        forbidden = ("rannacher", "smooth", "stress", "dataset", "neural")
        public_names = getattr(greeks, "__all__", ())
        for name in public_names:
            lowered = name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), name)

        new_paths = [GREEKS_MODULE_PATH, SCRIPT_PATH]
        for path in new_paths:
            lowered = path.name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), str(path))

    def test_new_python_files_include_ticket_10_docstrings(self):
        import american_risk_surfaces.diagnostics.greeks as greeks

        self.assertIn("Ticket 10", inspect.getdoc(greeks))
        self.assertIn("Ticket 10", inspect.getdoc(self.module))


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


def _synthetic_boundary_curve(boundary_spots):
    spot_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tau_grid = np.array([0.0, 0.5, 1.0])
    points = []
    for index, tau in enumerate(tau_grid):
        found = index in boundary_spots
        points.append(
            BoundaryPoint(
                time_index=index,
                tau=float(tau),
                boundary_found=found,
                boundary_spot=float(boundary_spots[index]) if found else float("nan"),
                threshold=1e-6,
                search_direction="low_to_high",
                extraction_method="linear_threshold_crossing" if found else "none",
                no_boundary_reason="" if found else "all_continuation_like",
                exercise_like_node_count=1 if found else 0,
                continuation_like_node_count=2,
            )
        )
    payoff = np.maximum(1.0 - spot_grid, 0.0)
    value_grid = np.tile(payoff, (len(tau_grid), 1))
    return BoundaryCurve(
        case_name="synthetic",
        option_type="put",
        K=1.0,
        T=1.0,
        r=0.05,
        q=0.02,
        sigma=0.2,
        Smax=2.0,
        M=4,
        N=2,
        threshold=1e-6,
        spot_grid=spot_grid,
        tau_grid=tau_grid,
        payoff=payoff,
        value_grid=value_grid,
        premium_grid=value_grid - payoff,
        points=tuple(points),
    )


def _load_ticket10_module():
    spec = importlib.util.spec_from_file_location("ticket10_greek_diagnostics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
