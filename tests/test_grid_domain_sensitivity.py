"""Ticket 11 tests for grid and domain sensitivity diagnostics."""

import csv
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "09_grid_domain_sensitivity.py"
SENSITIVITY_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "diagnostics"
    / "sensitivity.py"
)
GRID_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_11_grid_sensitivity_summary.csv"
)
DOMAIN_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_11_domain_sensitivity_summary.csv"
)
SELECTED_SPOT_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_11_selected_spot_sensitivity.csv"
)
BOUNDARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_11_boundary_sensitivity.csv"
)
DIAGNOSTIC_CSV = (
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "tables"
    / "ticket_11_diagnostic_sensitivity.csv"
)

FIGURE_PATHS = [
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_price_vs_grid_size.png",
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_price_error_vs_reference.png",
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_boundary_grid_comparison.png",
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_domain_cutoff_comparison.png",
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_lcp_diagnostic_stability.png",
    PROJECT_ROOT
    / "results"
    / "01_solver_validation"
    / "figures"
    / "ticket_11_gamma_stability.png",
]

SUMMARY_COLUMNS = [
    "sensitivity_type",
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
    "dS",
    "dtau",
    "reference_case_name",
    "reference_Smax",
    "reference_M",
    "reference_N",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_abs_selected_price_difference",
    "rmse_selected_price_difference",
    "boundary_found_count",
    "max_abs_boundary_shift",
    "mean_abs_boundary_shift",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "runtime_seconds",
    "price_vs_grid_figure_created",
    "price_error_figure_created",
    "boundary_figure_created",
    "domain_figure_created",
    "lcp_figure_created",
    "gamma_figure_created",
]

SELECTED_COLUMNS = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "reference_case_name",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "reference_nearest_spot",
    "reference_actual_moneyness",
    "reference_value",
    "difference_vs_reference",
    "abs_difference_vs_reference",
    "relative_difference_vs_reference",
]

BOUNDARY_COLUMNS = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "reference_case_name",
    "target_tau_fraction",
    "nearest_tau",
    "boundary_found",
    "boundary_spot",
    "reference_nearest_tau",
    "reference_boundary_found",
    "reference_boundary_spot",
    "boundary_shift",
    "abs_boundary_shift",
    "boundary_status",
]

DIAGNOSTIC_COLUMNS = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "Smax",
    "M",
    "N",
    "dS",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "strict_negative_gamma_count",
    "runtime_seconds",
]


class SensitivityConfigurationTests(unittest.TestCase):
    def test_grid_sensitivity_cases_have_expected_configurations(self):
        from american_risk_surfaces.diagnostics.sensitivity import grid_sensitivity_cases

        cases = grid_sensitivity_cases()
        grouped = _group_cases(cases)
        self.assertEqual(set(grouped), {"american_put", "dividend_call"})
        for family_cases in grouped.values():
            self.assertEqual([(case.Smax, case.M, case.N) for case in family_cases], [
                (4.0, 80, 80),
                (4.0, 120, 120),
                (4.0, 180, 180),
            ])
            self.assertTrue(all(case.sensitivity_type == "grid" for case in family_cases))

    def test_domain_sensitivity_cases_keep_dS_comparable(self):
        from american_risk_surfaces.diagnostics.sensitivity import domain_sensitivity_cases

        cases = domain_sensitivity_cases()
        grouped = _group_cases(cases)
        self.assertEqual(set(grouped), {"american_put", "dividend_call"})
        for family_cases in grouped.values():
            self.assertEqual([(case.Smax, case.M, case.N) for case in family_cases], [
                (4.0, 120, 120),
                (5.0, 150, 120),
                (6.0, 180, 120),
            ])
            spacings = [case.Smax / case.M for case in family_cases]
            self.assertLess(max(spacings) - min(spacings), 1e-14)
            self.assertTrue(all(case.sensitivity_type == "domain" for case in family_cases))

    def test_nearest_spot_index_is_predictable(self):
        from american_risk_surfaces.diagnostics.sensitivity import nearest_spot_index

        grid = np.array([0.0, 0.5, 1.0, 1.5])
        self.assertEqual(nearest_spot_index(grid, 0.76), 2)
        self.assertEqual(nearest_spot_index(grid, 0.74), 1)
        self.assertEqual(nearest_spot_index(grid, 1.5), 3)

    def test_selected_spot_reference_difference_on_synthetic_runs(self):
        from american_risk_surfaces.diagnostics.sensitivity import (
            SensitivityCase,
            SensitivityRunResult,
            selected_spot_rows,
        )

        case = SensitivityCase("grid", "synthetic", "put", 1.0, 1.0, 0.05, 0.02, 0.2, 2.0, 4, 4)
        reference_case = SensitivityCase(
            "grid", "synthetic_ref", "put", 1.0, 1.0, 0.05, 0.02, 0.2, 2.0, 4, 4
        )
        run = SensitivityRunResult(
            case=case,
            result=_fake_result([0.0, 0.5, 1.0, 1.5, 2.0], [1.0, 0.5, 0.2, 0.1, 0.0]),
            boundary_curve=None,
            boundary_summary=None,
            lcp_diagnostics=None,
            greek_diagnostics=None,
            runtime_seconds=0.0,
            dS=0.5,
            dtau=0.25,
        )
        reference = SensitivityRunResult(
            case=reference_case,
            result=_fake_result([0.0, 0.5, 1.0, 1.5, 2.0], [1.0, 0.5, 0.25, 0.1, 0.0]),
            boundary_curve=None,
            boundary_summary=None,
            lcp_diagnostics=None,
            greek_diagnostics=None,
            runtime_seconds=0.0,
            dS=0.5,
            dtau=0.25,
        )

        rows = selected_spot_rows(run, reference, selected_moneyness=(1.0,))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].difference_vs_reference, -0.05)
        self.assertAlmostEqual(rows[0].abs_difference_vs_reference, 0.05)


class SensitivityRunTests(unittest.TestCase):
    def test_small_solver_sensitivity_run_completes_and_reports_convergence(self):
        from american_risk_surfaces.diagnostics.sensitivity import (
            SensitivityCase,
            run_sensitivity_case,
        )

        case = SensitivityCase("grid", "small_put", "put", 1.0, 0.5, 0.05, 0.02, 0.2, 3.0, 30, 30)
        run = run_sensitivity_case(case)
        self.assertTrue(run.result.converged)
        self.assertEqual(run.result.value_grid.shape, (31, 31))
        self.assertTrue(np.all(np.isfinite(run.result.values)))
        self.assertTrue(np.isfinite(run.runtime_seconds))

    def test_boundary_sensitivity_rows_are_produced(self):
        from american_risk_surfaces.diagnostics.sensitivity import (
            SensitivityCase,
            boundary_shift_rows,
            run_sensitivity_case,
        )

        run = run_sensitivity_case(
            SensitivityCase("grid", "small_put", "put", 1.0, 0.5, 0.05, 0.02, 0.2, 3.0, 30, 30)
        )
        reference = run_sensitivity_case(
            SensitivityCase("grid", "small_put_ref", "put", 1.0, 0.5, 0.05, 0.02, 0.2, 3.0, 40, 40)
        )
        rows = boundary_shift_rows(run, reference, selected_tau_fractions=(0.5, 1.0))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.boundary_status in {"matched", "unmatched"} for row in rows))

    def test_lcp_diagnostic_sensitivity_is_finite(self):
        from american_risk_surfaces.diagnostics.sensitivity import (
            SensitivityCase,
            diagnostic_row,
            run_sensitivity_case,
        )

        run = run_sensitivity_case(
            SensitivityCase("grid", "small_call", "call", 1.0, 0.5, 0.05, 0.08, 0.2, 3.0, 30, 30)
        )
        row = diagnostic_row(run)
        self.assertTrue(np.isfinite(row.max_obstacle_violation))
        self.assertTrue(np.isfinite(row.max_equation_violation))
        self.assertTrue(np.isfinite(row.max_abs_complementarity_product))

    def test_greek_diagnostic_sensitivity_is_finite_and_not_labels(self):
        from american_risk_surfaces.diagnostics.sensitivity import (
            SensitivityCase,
            diagnostic_row,
            run_sensitivity_case,
        )

        run = run_sensitivity_case(
            SensitivityCase("grid", "small_put", "put", 1.0, 0.5, 0.05, 0.02, 0.2, 3.0, 30, 30)
        )
        row = diagnostic_row(run)
        self.assertTrue(np.isfinite(row.max_abs_gamma))
        self.assertTrue(np.isfinite(row.max_abs_gamma_strict))
        self.assertGreaterEqual(row.boundary_near_node_count, 0)


class SensitivityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")
        cls.module = _load_ticket11_module()
        (
            cls.grid_rows,
            cls.domain_rows,
            cls.selected_rows,
            cls.boundary_rows,
            cls.diagnostic_rows,
            cls.metadata,
        ) = cls.module.main()

    def test_experiment_writes_expected_csv_artifacts(self):
        self.assertEqual(len(self.grid_rows), 6)
        self.assertEqual(len(self.domain_rows), 6)
        self.assertEqual(len(self.selected_rows), 72)
        self.assertEqual(len(self.boundary_rows), 36)
        self.assertEqual(len(self.diagnostic_rows), 12)
        for path in [GRID_SUMMARY_CSV, DOMAIN_SUMMARY_CSV, SELECTED_SPOT_CSV, BOUNDARY_CSV, DIAGNOSTIC_CSV]:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0)

    def test_csv_tables_have_expected_columns(self):
        expected = [
            (GRID_SUMMARY_CSV, SUMMARY_COLUMNS),
            (DOMAIN_SUMMARY_CSV, SUMMARY_COLUMNS),
            (SELECTED_SPOT_CSV, SELECTED_COLUMNS),
            (BOUNDARY_CSV, BOUNDARY_COLUMNS),
            (DIAGNOSTIC_CSV, DIAGNOSTIC_COLUMNS),
        ]
        for path, columns in expected:
            with path.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertEqual(reader.fieldnames, columns)
                self.assertGreater(len(list(reader)), 0)

    def test_figures_created_when_matplotlib_available(self):
        for figure_path in FIGURE_PATHS:
            self.assertTrue(figure_path.exists(), figure_path)
            self.assertGreater(figure_path.stat().st_size, 0)

    def test_no_forbidden_stress_dataset_neural_or_label_files_or_api(self):
        import american_risk_surfaces.diagnostics.sensitivity as sensitivity

        forbidden = ("stress", "dataset", "neural", "label")
        for name in getattr(sensitivity, "__all__", ()):
            lowered = name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), name)
        for path in [SENSITIVITY_MODULE_PATH, SCRIPT_PATH]:
            lowered = path.name.lower()
            self.assertFalse(any(fragment in lowered for fragment in forbidden), str(path))

    def test_new_python_files_include_ticket_11_docstrings(self):
        import american_risk_surfaces.diagnostics.sensitivity as sensitivity

        self.assertIn("Ticket 11", inspect.getdoc(sensitivity))
        self.assertIn("Ticket 11", inspect.getdoc(self.module))


def _group_cases(cases):
    grouped = {}
    for case in cases:
        family = "american_put" if case.option_type == "put" else "dividend_call"
        grouped.setdefault(family, []).append(case)
    return grouped


def _fake_result(spot_grid, values):
    return SimpleNamespace(
        K=1.0,
        option_type="put",
        spot_grid=np.asarray(spot_grid, dtype=float),
        values=np.asarray(values, dtype=float),
        payoff=np.maximum(1.0 - np.asarray(spot_grid, dtype=float), 0.0),
    )


def _load_ticket11_module():
    spec = importlib.util.spec_from_file_location("ticket11_sensitivity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
