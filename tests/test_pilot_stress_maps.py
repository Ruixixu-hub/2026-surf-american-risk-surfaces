"""Pilot 01 tests for controlled pilot stress maps."""

from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "11_pilot_stress_maps.py"
SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "downstream"
    / "pilot_stress_maps.py"
)

RUN_COLUMNS = [
    "case_name",
    "case_family",
    "variation_name",
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
    "solver_name",
    "solver_variant",
    "premium_threshold",
    "interpretation_lower_moneyness",
    "interpretation_upper_moneyness",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "boundary_found_count",
    "boundary_status",
    "greek_status",
    "acceptance_status",
    "downstream_use_status",
]

DIAGNOSTIC_COLUMNS = [
    "case_name",
    "option_type",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "finite_delta_count",
    "finite_gamma_count",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "greek_status",
    "downstream_use_status",
]

BOUNDARY_COLUMNS = [
    "case_name",
    "option_type",
    "threshold",
    "search_direction",
    "found_boundary_count",
    "no_boundary_count",
    "maturity_ambiguous_count",
    "all_continuation_count",
    "all_exercise_count",
    "expected_exercise_side_absent_count",
    "no_clean_transition_count",
    "min_boundary_spot",
    "max_boundary_spot",
    "status",
    "downstream_use_status",
]

SLICE_COLUMNS = [
    "case_name",
    "option_type",
    "target_tau_fraction",
    "nearest_tau",
    "time_index",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "payoff",
    "continuation_premium",
    "exercise_indicator",
    "boundary_found_at_time",
    "boundary_spot_at_time",
    "boundary_distance",
    "delta",
    "gamma",
    "boundary_near",
    "kink_near",
    "maturity_row",
    "strict_interior",
    "downstream_use_status",
]

MANIFEST_COLUMNS = [
    "output_id",
    "output_type",
    "case_name",
    "case_family",
    "variation_name",
    "path",
    "created",
    "description",
    "solver_name",
    "solver_variant",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "premium_threshold",
    "interpretation_lower_moneyness",
    "interpretation_upper_moneyness",
    "contains_greek_diagnostics",
    "contains_boundary_overlay",
    "downstream_use_status",
    "review_status",
]


class PilotCaseTests(unittest.TestCase):
    def test_pilot_cases_match_approved_small_scope(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import pilot_cases

        cases = pilot_cases(include_higher_grid_checks=True)
        self.assertEqual(len(cases), 11)
        by_name = {case.case_name: case for case in cases}

        self.assertEqual(by_name["american_put_base"].option_type, "put")
        self.assertEqual((by_name["american_put_base"].M, by_name["american_put_base"].N), (120, 120))
        self.assertEqual(by_name["american_put_high_grid_check"].M, 180)
        self.assertEqual(by_name["dividend_call_high_grid_check"].N, 180)
        self.assertEqual(by_name["american_put_sigma_040"].sigma, 0.40)
        self.assertEqual(by_name["american_put_r_003"].r, 0.03)
        self.assertEqual(by_name["american_put_q_000"].q, 0.00)
        self.assertEqual(by_name["dividend_call_q_003"].q, 0.03)
        self.assertEqual(by_name["dividend_call_q_010"].q, 0.10)
        self.assertTrue(all(case.K == 1.0 for case in cases))
        self.assertTrue(all(case.Smax == 4.0 for case in cases))

    def test_default_solver_metadata_is_baseline_not_rannacher(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import (
            SOLVER_NAME,
            SOLVER_VARIANT,
        )

        self.assertEqual(SOLVER_NAME, "american_crank_nicolson_psor_price")
        self.assertEqual(SOLVER_VARIANT, "baseline_cn_psor")
        self.assertNotIn("rannacher", SOLVER_VARIANT.lower())


class PilotRunTests(unittest.TestCase):
    def test_small_smoke_pilot_run_completes_and_has_required_metadata(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import (
            PilotCase,
            run_pilot_case,
            run_summary_row,
        )

        case = PilotCase(
            case_name="small_put",
            case_family="american_put",
            variation_name="small",
            option_type="put",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=3.0,
            M=24,
            N=24,
        )
        artifacts = run_pilot_case(case)
        row = run_summary_row(artifacts)

        self.assertTrue(artifacts.result.converged)
        self.assertEqual(artifacts.result.value_grid.shape, (25, 25))
        self.assertTrue(set(RUN_COLUMNS).issubset(row))
        self.assertEqual(row["solver_variant"], "baseline_cn_psor")
        self.assertEqual(row["downstream_use_status"], "pilot_diagnostic_only")

    def test_continuation_premium_equals_value_minus_payoff(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import continuation_premium_grid

        result = _fake_result(
            value_grid=np.array([[1.0, 0.5, 0.1], [1.1, 0.6, 0.2]]),
            payoff=np.array([1.0, 0.4, 0.0]),
        )
        premium = continuation_premium_grid(result)
        np.testing.assert_allclose(premium, result.value_grid - result.payoff)

    def test_exercise_indicator_respects_threshold(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import exercise_indicator

        premium = np.array([[0.0, 1e-7, 2e-6]])
        indicator = exercise_indicator(premium, threshold=1e-6)
        np.testing.assert_array_equal(indicator, np.array([[1, 1, 0]]))

    def test_boundary_and_lcp_diagnostics_are_recorded(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import (
            PilotCase,
            boundary_summary_row,
            diagnostic_summary_row,
            run_pilot_case,
        )

        case = PilotCase(
            case_name="small_dividend_call",
            case_family="dividend_call",
            variation_name="small",
            option_type="call",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.08,
            sigma=0.2,
            Smax=3.0,
            M=24,
            N=24,
        )
        artifacts = run_pilot_case(case)
        boundary_row = boundary_summary_row(artifacts)
        diagnostic_row = diagnostic_summary_row(artifacts)

        self.assertTrue(set(BOUNDARY_COLUMNS).issubset(boundary_row))
        self.assertTrue(set(DIAGNOSTIC_COLUMNS).issubset(diagnostic_row))
        self.assertGreaterEqual(int(boundary_row["found_boundary_count"]), 0)
        self.assertTrue(np.isfinite(float(diagnostic_row["max_equation_violation"])))
        self.assertTrue(np.isfinite(float(diagnostic_row["max_abs_complementarity_product"])))

    def test_selected_slice_rows_have_stable_columns(self):
        from american_risk_surfaces.downstream.pilot_stress_maps import (
            PilotCase,
            run_pilot_case,
            selected_slice_rows,
        )

        case = PilotCase(
            case_name="small_slice_put",
            case_family="american_put",
            variation_name="small",
            option_type="put",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=3.0,
            M=24,
            N=24,
        )
        artifacts = run_pilot_case(case)
        rows = selected_slice_rows(artifacts, selected_tau_fractions=(0.5,), selected_moneyness=(1.0,))

        self.assertEqual(len(rows), 1)
        self.assertTrue(set(SLICE_COLUMNS).issubset(rows[0]))
        self.assertAlmostEqual(
            float(rows[0]["continuation_premium"]),
            float(rows[0]["value"]) - float(rows[0]["payoff"]),
        )


class PilotExperimentTests(unittest.TestCase):
    def test_experiment_writes_manifest_and_csv_artifacts(self):
        module = _load_experiment_module()
        from american_risk_surfaces.downstream.pilot_stress_maps import PilotCase

        case = PilotCase(
            case_name="tiny_put_artifact",
            case_family="american_put",
            variation_name="tiny",
            option_type="put",
            K=1.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=3.0,
            M=18,
            N=18,
        )
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_rows, diagnostic_rows, boundary_rows, slice_rows, manifest_rows, metadata = (
                module.main(
                    cases=(case,),
                    table_dir=root / "tables",
                    figure_dir=root / "figures",
                    create_figures=False,
                )
            )
            self.assertEqual(len(run_rows), 1)
            self.assertEqual(len(diagnostic_rows), 1)
            self.assertEqual(len(boundary_rows), 1)
            self.assertGreater(len(slice_rows), 0)
            self.assertGreaterEqual(len(manifest_rows), 5)
            self.assertTrue(Path(metadata["run_summary_csv"]).exists())
            self.assertTrue(Path(metadata["output_manifest_csv"]).exists())

            with Path(metadata["output_manifest_csv"]).open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertTrue(set(MANIFEST_COLUMNS).issubset(reader.fieldnames or []))

    def test_figures_created_when_matplotlib_available(self):
        module = _load_experiment_module()
        if module._load_pyplot() is None:
            self.skipTest("matplotlib is unavailable")

        from american_risk_surfaces.downstream.pilot_stress_maps import PilotCase

        put_case = PilotCase(
            case_name="tiny_put_figure",
            case_family="american_put",
            variation_name="base",
            option_type="put",
            K=1.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=3.0,
            M=18,
            N=18,
            is_base_case=True,
        )
        call_case = PilotCase(
            case_name="tiny_call_figure",
            case_family="dividend_call",
            variation_name="base",
            option_type="call",
            K=1.0,
            T=0.25,
            r=0.05,
            q=0.08,
            sigma=0.2,
            Smax=3.0,
            M=18,
            N=18,
            is_base_case=True,
        )
        with TemporaryDirectory() as tmpdir:
            _, _, _, _, manifest_rows, metadata = module.main(
                cases=(put_case, call_case),
                table_dir=Path(tmpdir) / "tables",
                figure_dir=Path(tmpdir) / "figures",
                create_figures=True,
            )
            figure_paths = [
                Path(row["path"])
                for row in manifest_rows
                if row["output_type"] == "figure" and row["created"] == "True"
            ]
            self.assertGreater(len(figure_paths), 0)
            self.assertTrue(all(path.exists() for path in figure_paths))
            self.assertEqual(metadata["value_heatmap_figure_created"], "True")


class PilotScopeTests(unittest.TestCase):
    def test_no_forbidden_dataset_neural_surrogate_training_or_label_api(self):
        from american_risk_surfaces.downstream import pilot_stress_maps

        forbidden = ("dataset", "neural", "surrogate", "training", "ml_label")
        public_names = [
            name for name, _ in inspect.getmembers(pilot_stress_maps) if not name.startswith("_")
        ]
        lowered = " ".join(public_names).lower()
        for word in forbidden:
            self.assertNotIn(word, lowered)

        forbidden_paths = [
            path
            for path in PROJECT_ROOT.rglob("*")
            if "pilot_01" in path.name.lower()
            and any(word in path.name.lower() for word in forbidden)
        ]
        self.assertEqual(forbidden_paths, [])

    def test_new_python_files_include_pilot_01_docstrings(self):
        expected = "Pilot 01: controlled pilot stress maps for validated American CN/PSOR solver."
        for path in (SOURCE_PATH, SCRIPT_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertIn(expected, text)


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location("pilot_stress_map_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_result(value_grid: np.ndarray, payoff: np.ndarray):
    class FakeResult:
        pass

    result = FakeResult()
    result.value_grid = np.asarray(value_grid, dtype=float)
    result.payoff = np.asarray(payoff, dtype=float)
    return result
