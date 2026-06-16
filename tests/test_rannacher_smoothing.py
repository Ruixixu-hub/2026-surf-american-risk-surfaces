"""Ticket 10A tests for Rannacher smoothing comparison."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve
from american_risk_surfaces.diagnostics.greeks import diagnose_greek_result
from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price
from american_risk_surfaces.solvers.grid import uniform_spot_grid
from american_risk_surfaces.solvers.operator import (
    american_put_boundaries,
    black_scholes_operator_coefficients,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "08_rannacher_smoothing_comparison.py"
RANNACHER_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "solvers"
    / "rannacher.py"
)
SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10a_rannacher_comparison_summary.csv"
)
SELECTED_SPOTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10a_rannacher_selected_spots.csv"
)
GREEK_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10a_rannacher_greek_summary.csv"
)
BOUNDARY_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_10a_rannacher_boundary_summary.csv"
)
VALUE_PROFILE_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_value_profiles.png"
)
PRICE_DIFF_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_price_difference_profiles.png"
)
GAMMA_PROFILE_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_gamma_profiles.png"
)
GAMMA_FULL_STRICT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_gamma_full_vs_strict.png"
)
BOUNDARY_COMPARISON_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_boundary_curve_comparison.png"
)
PSOR_ITERATIONS_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_10a_psor_iterations.png"
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
    "rannacher_substeps",
    "rannacher_substep_size",
    "baseline_converged",
    "rannacher_converged",
    "baseline_max_obstacle_violation",
    "rannacher_max_obstacle_violation",
    "max_abs_price_difference",
    "rmse_price_difference",
    "mean_price_difference",
    "max_price_difference_spot",
    "gate_recommendation",
    "value_figure_created",
    "price_difference_figure_created",
    "gamma_profile_figure_created",
    "gamma_full_strict_figure_created",
    "boundary_figure_created",
    "psor_figure_created",
]

EXPECTED_SELECTED_COLUMNS = [
    "case_name",
    "option_type",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "payoff",
    "baseline_value",
    "rannacher_value",
    "rannacher_minus_baseline",
]

EXPECTED_GREEK_COLUMNS = [
    "case_name",
    "option_type",
    "baseline_max_abs_gamma",
    "rannacher_max_abs_gamma",
    "baseline_max_abs_gamma_strict",
    "rannacher_max_abs_gamma_strict",
    "baseline_boundary_near_node_count",
    "rannacher_boundary_near_node_count",
    "baseline_strict_negative_gamma_count",
    "rannacher_strict_negative_gamma_count",
    "strict_gamma_change",
    "full_gamma_change",
]

EXPECTED_BOUNDARY_COLUMNS = [
    "case_name",
    "option_type",
    "baseline_found_boundary_count",
    "rannacher_found_boundary_count",
    "matched_boundary_count",
    "max_abs_boundary_shift",
    "mean_abs_boundary_shift",
    "first_baseline_boundary_tau",
    "first_rannacher_boundary_tau",
    "boundary_status",
]


class RannacherSolverTests(unittest.TestCase):
    def test_backward_euler_psor_step_respects_obstacle_and_metadata(self):
        from american_risk_surfaces.solvers.rannacher import backward_euler_psor_step

        spot_grid, dS = uniform_spot_grid(2.0, 20)
        payoff = np.maximum(1.0 - spot_grid, 0.0)
        values = payoff + 0.02
        coefficients = black_scholes_operator_coefficients(
            spot_grid, dS=dS, r=0.05, q=0.02, sigma=0.2
        )
        lower_boundary, upper_boundary = american_put_boundaries(1.0, 0.025)

        next_values, psor_result = backward_euler_psor_step(
            values,
            payoff,
            coefficients,
            dtau=0.025,
            new_lower_boundary=lower_boundary,
            new_upper_boundary=upper_boundary,
        )

        self.assertEqual(next_values.shape, values.shape)
        self.assertTrue(psor_result.converged)
        self.assertGreater(psor_result.iterations, 0)
        self.assertLessEqual(np.max(np.maximum(payoff - next_values, 0.0)), 1e-8)
        self.assertEqual(next_values[0], lower_boundary)
        self.assertEqual(next_values[-1], upper_boundary)

    def test_rannacher_result_shape_and_metadata_match_baseline_grid(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        baseline = _small_baseline_put_result()
        smoothed = rannacher_crank_nicolson_psor_price(
            option_type="put",
            K=1.0,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=4.0,
            M=40,
            N=40,
            rannacher_substeps=2,
        )

        self.assertTrue(smoothed.metadata.enabled)
        self.assertEqual(smoothed.metadata.rannacher_substeps, 2)
        self.assertEqual(smoothed.result.value_grid.shape, baseline.value_grid.shape)
        npt.assert_allclose(smoothed.result.spot_grid, baseline.spot_grid)
        npt.assert_allclose(smoothed.result.tau_grid, baseline.tau_grid)
        npt.assert_allclose(smoothed.result.payoff, baseline.payoff)
        self.assertEqual(len(smoothed.metadata.rannacher_psor_results), 2)
        self.assertEqual(len(smoothed.metadata.cn_psor_results), 39)

    def test_zero_rannacher_substeps_delegates_to_baseline(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        baseline = _small_baseline_put_result()
        smoothed = rannacher_crank_nicolson_psor_price(
            option_type="put",
            K=1.0,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=4.0,
            M=40,
            N=40,
            rannacher_substeps=0,
        )

        self.assertFalse(smoothed.metadata.enabled)
        npt.assert_allclose(smoothed.result.value_grid, baseline.value_grid)
        self.assertEqual(len(smoothed.metadata.rannacher_psor_results), 0)
        self.assertEqual(len(smoothed.metadata.cn_psor_results), len(baseline.psor_results))

    def test_smoothed_put_and_dividend_call_respect_obstacle(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        for option_type, q in (("put", 0.02), ("call", 0.08)):
            result = rannacher_crank_nicolson_psor_price(
                option_type=option_type,
                K=1.0,
                T=1.0,
                r=0.05,
                q=q,
                sigma=0.2,
                Smax=4.0,
                M=40,
                N=40,
            ).result

            self.assertTrue(result.converged)
            self.assertLessEqual(result.max_obstacle_violation, 1e-8)
            self.assertLessEqual(
                float(np.max(np.maximum(result.payoff[np.newaxis, :] - result.value_grid, 0.0))),
                1e-8,
            )

    def test_smoothed_and_baseline_prices_are_finite_and_close(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        baseline = _small_baseline_put_result()
        smoothed = rannacher_crank_nicolson_psor_price(
            option_type="put",
            K=1.0,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=4.0,
            M=40,
            N=40,
        ).result

        difference = smoothed.values - baseline.values
        self.assertTrue(np.all(np.isfinite(difference)))
        self.assertLess(float(np.max(np.abs(difference))), 0.02)

    def test_boundary_extraction_runs_for_baseline_and_smoothed_results(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        baseline_curve = extract_boundary_curve(_small_baseline_put_result(), "baseline")
        smoothed_curve = extract_boundary_curve(
            rannacher_crank_nicolson_psor_price(
                option_type="put",
                K=1.0,
                T=1.0,
                r=0.05,
                q=0.02,
                sigma=0.2,
                Smax=4.0,
                M=40,
                N=40,
            ).result,
            "smoothed",
        )

        self.assertGreater(sum(point.boundary_found for point in baseline_curve.points), 0)
        self.assertGreater(sum(point.boundary_found for point in smoothed_curve.points), 0)

    def test_greek_diagnostics_runs_for_baseline_and_smoothed_results(self):
        from american_risk_surfaces.solvers.rannacher import (
            rannacher_crank_nicolson_psor_price,
        )

        baseline_greeks = diagnose_greek_result(_small_baseline_put_result(), "baseline")
        smoothed_greeks = diagnose_greek_result(
            rannacher_crank_nicolson_psor_price(
                option_type="put",
                K=1.0,
                T=1.0,
                r=0.05,
                q=0.02,
                sigma=0.2,
                Smax=4.0,
                M=40,
                N=40,
            ).result,
            "smoothed",
        )

        self.assertGreater(baseline_greeks.summary.finite_gamma_count, 0)
        self.assertGreater(smoothed_greeks.summary.finite_gamma_count, 0)
        self.assertTrue(np.isfinite(smoothed_greeks.summary.max_abs_gamma_strict))

    def test_invalid_inputs_raise_value_error(self):
        from american_risk_surfaces.solvers.rannacher import (
            backward_euler_psor_step,
            rannacher_crank_nicolson_psor_price,
        )

        spot_grid, dS = uniform_spot_grid(2.0, 20)
        payoff = np.maximum(1.0 - spot_grid, 0.0)
        coefficients = black_scholes_operator_coefficients(
            spot_grid, dS=dS, r=0.05, q=0.02, sigma=0.2
        )
        with self.assertRaises(ValueError):
            backward_euler_psor_step(payoff[:-1], payoff, coefficients, 0.01, 1.0, 0.0)
        with self.assertRaises(ValueError):
            backward_euler_psor_step(payoff, payoff, coefficients, -0.01, 1.0, 0.0)
        with self.assertRaises(ValueError):
            rannacher_crank_nicolson_psor_price("bad", 1.0, 1.0, 0.05, 0.02, 0.2, 4.0, 40, 40)
        with self.assertRaises(ValueError):
            rannacher_crank_nicolson_psor_price("put", 1.0, 1.0, 0.05, 0.02, -0.2, 4.0, 40, 40)
        with self.assertRaises(ValueError):
            rannacher_crank_nicolson_psor_price("put", 1.0, 1.0, 0.05, 0.02, 0.2, 4.0, 40, 40, rannacher_substeps=-1)


class RannacherExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket10a_module()
        (
            cls.summary_rows,
            cls.selected_rows,
            cls.greek_rows,
            cls.boundary_rows,
            cls.metadata,
        ) = cls.module.main()

    def test_experiment_writes_expected_csv_artifacts(self):
        self.assertEqual(len(self.summary_rows), 2)
        self.assertEqual(len(self.selected_rows), 12)
        self.assertEqual(len(self.greek_rows), 2)
        self.assertEqual(len(self.boundary_rows), 2)
        self.assertTrue(SUMMARY_CSV.exists())
        self.assertTrue(SELECTED_SPOTS_CSV.exists())
        self.assertTrue(GREEK_SUMMARY_CSV.exists())
        self.assertTrue(BOUNDARY_SUMMARY_CSV.exists())
        self.assertEqual(self.metadata["summary_csv"], str(SUMMARY_CSV))
        self.assertEqual(self.metadata["selected_spots_csv"], str(SELECTED_SPOTS_CSV))
        self.assertEqual(self.metadata["greek_summary_csv"], str(GREEK_SUMMARY_CSV))
        self.assertEqual(self.metadata["boundary_summary_csv"], str(BOUNDARY_SUMMARY_CSV))

    def test_csv_tables_have_expected_columns(self):
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SUMMARY_COLUMNS)
            summary_rows = list(reader)
        with SELECTED_SPOTS_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_SELECTED_COLUMNS)
            selected_rows = list(reader)
        with GREEK_SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_GREEK_COLUMNS)
            greek_rows = list(reader)
        with BOUNDARY_SUMMARY_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, EXPECTED_BOUNDARY_COLUMNS)
            boundary_rows = list(reader)

        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(len(selected_rows), 12)
        self.assertEqual(len(greek_rows), 2)
        self.assertEqual(len(boundary_rows), 2)

    def test_summary_contains_gate_recommendations(self):
        allowed = {
            "KEEP_BASELINE_FOR_NOW",
            "USE_RANNACHER_FOR_GREEK_DIAGNOSTICS",
            "INCONCLUSIVE_NEEDS_GRID_SENSITIVITY",
        }
        for row in self.summary_rows:
            self.assertIn(row["gate_recommendation"], allowed)
            self.assertEqual(row["rannacher_substeps"], "2")
            self.assertEqual(row["baseline_converged"], "True")
            self.assertEqual(row["rannacher_converged"], "True")

    def test_figures_created_when_matplotlib_available(self):
        figures = [
            VALUE_PROFILE_FIGURE,
            PRICE_DIFF_FIGURE,
            GAMMA_PROFILE_FIGURE,
            GAMMA_FULL_STRICT_FIGURE,
            BOUNDARY_COMPARISON_FIGURE,
            PSOR_ITERATIONS_FIGURE,
        ]
        for figure in figures:
            self.assertTrue(figure.exists(), figure)
            self.assertGreater(figure.stat().st_size, 0)

        self.assertEqual(self.metadata["value_figure_path"], str(VALUE_PROFILE_FIGURE))
        self.assertEqual(self.metadata["price_difference_figure_path"], str(PRICE_DIFF_FIGURE))
        self.assertEqual(self.metadata["gamma_profile_figure_path"], str(GAMMA_PROFILE_FIGURE))
        self.assertEqual(self.metadata["gamma_full_strict_figure_path"], str(GAMMA_FULL_STRICT_FIGURE))
        self.assertEqual(self.metadata["boundary_figure_path"], str(BOUNDARY_COMPARISON_FIGURE))
        self.assertEqual(self.metadata["psor_figure_path"], str(PSOR_ITERATIONS_FIGURE))

    def test_no_forbidden_stress_dataset_neural_or_label_files_or_api(self):
        import american_risk_surfaces.solvers.rannacher as rannacher

        forbidden = ("stress", "dataset", "neural", "label")
        public_names = getattr(rannacher, "__all__", ())
        for name in public_names:
            lowered = name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), name)

        new_paths = [RANNACHER_MODULE_PATH, SCRIPT_PATH]
        for path in new_paths:
            lowered = path.name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), str(path))

    def test_new_python_files_include_ticket_10a_docstrings(self):
        import american_risk_surfaces.solvers.rannacher as rannacher

        self.assertIn("Ticket 10A", inspect.getdoc(rannacher))
        self.assertIn("Ticket 10A", inspect.getdoc(self.module))


def _small_baseline_put_result():
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


def _load_ticket10a_module():
    spec = importlib.util.spec_from_file_location("ticket10a_rannacher", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
