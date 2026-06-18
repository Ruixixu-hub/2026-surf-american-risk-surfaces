"""Quantitative Risk Analytics tests for small controlled expansion after Pilot 01."""

from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "12_quantitative_risk_analytics.py"
SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "downstream"
    / "risk_analytics.py"
)

EXPECTED_DOCSTRING = (
    "Quantitative Risk Analytics: small controlled expansion after Pilot 01 stress maps."
)

RUN_COLUMNS = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
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
    "runtime_seconds",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "boundary_found_count",
    "boundary_status",
    "min_boundary_spot",
    "max_boundary_spot",
    "max_continuation_premium",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "acceptance_status",
    "downstream_use_status",
]

BOUNDARY_COLUMNS = [
    "case_name",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "target_tau_fraction",
    "nearest_tau",
    "time_index",
    "boundary_found",
    "boundary_spot",
    "boundary_moneyness",
    "no_boundary_reason",
    "max_continuation_premium_at_tau",
    "mean_continuation_premium_at_tau_interpretation_region",
    "downstream_use_status",
]

GREEK_COLUMNS = [
    "case_name",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "finite_delta_count",
    "finite_gamma_count",
    "nonfinite_delta_count",
    "nonfinite_gamma_count",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "strict_negative_gamma_count",
    "greek_status",
    "downstream_use_status",
]

LCP_COLUMNS = [
    "case_name",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "lcp_status",
    "max_obstacle_violation",
    "min_value_gap",
    "max_equation_violation",
    "min_equation_gap",
    "max_abs_complementarity_product",
    "mean_max_abs_complementarity_product",
    "downstream_use_status",
]

RUNTIME_COLUMNS = [
    "case_name",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "runtime_seconds",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "all_psor_steps_converged",
    "downstream_use_status",
]

MANIFEST_COLUMNS = [
    "output_id",
    "output_type",
    "path",
    "created",
    "description",
    "solver_name",
    "solver_variant",
    "case_count",
    "contains_boundary_metrics",
    "contains_greek_metrics",
    "contains_lcp_metrics",
    "downstream_use_status",
    "review_status",
]


class RiskAnalyticsCaseTests(unittest.TestCase):
    def test_case_construction_uses_approved_small_sweeps(self):
        from american_risk_surfaces.downstream.risk_analytics import (
            risk_analytics_cases,
        )

        cases = risk_analytics_cases(include_q_sigma_heatmap=True)
        regular = [case for case in cases if not case.is_heatmap_case]
        heatmap = [case for case in cases if case.is_heatmap_case]
        self.assertEqual(len(regular), 19)
        self.assertEqual(len(heatmap), 12)

        put_vol = [
            case.stress_value
            for case in regular
            if case.case_family == "american_put"
            and case.sweep_name == "put_volatility_sweep"
        ]
        call_q = [
            case.stress_value
            for case in regular
            if case.case_family == "dividend_call"
            and case.sweep_name == "call_dividend_yield_sweep"
        ]
        self.assertEqual(put_vol, [0.20, 0.40, 0.60])
        self.assertEqual(call_q, [0.03, 0.08, 0.10, 0.14])
        self.assertTrue(all(case.K == 1.0 for case in cases))
        self.assertTrue(all(case.Smax == 4.0 for case in cases))
        self.assertTrue(all((case.M, case.N) == (120, 120) for case in cases))

    def test_solver_metadata_is_baseline_and_not_rannacher(self):
        from american_risk_surfaces.downstream.risk_analytics import (
            SOLVER_NAME,
            SOLVER_VARIANT,
        )

        self.assertEqual(SOLVER_NAME, "american_crank_nicolson_psor_price")
        self.assertEqual(SOLVER_VARIANT, "baseline_cn_psor")
        self.assertNotIn("rannacher", SOLVER_VARIANT.lower())

    def test_direction_check_helper_classifies_monotone_data(self):
        from american_risk_surfaces.downstream.risk_analytics import monotone_direction

        self.assertEqual(monotone_direction([3.0, 2.0, 1.0]), "decreasing")
        self.assertEqual(monotone_direction([1.0, 2.0, 3.0]), "increasing")
        self.assertEqual(monotone_direction([1.0, 1.0, 1.0]), "flat")
        self.assertEqual(monotone_direction([1.0, 3.0, 2.0]), "mixed")


class RiskAnalyticsRunTests(unittest.TestCase):
    def test_small_smoke_run_produces_stable_rows(self):
        from american_risk_surfaces.downstream.risk_analytics import (
            RiskAnalyticsCase,
            boundary_metric_rows,
            greek_metric_row,
            lcp_metric_row,
            run_risk_analytics_case,
            run_summary_row,
            runtime_iteration_row,
        )

        case = RiskAnalyticsCase(
            case_name="small_put_sigma",
            case_family="american_put",
            sweep_name="put_volatility_sweep",
            stress_parameter="sigma",
            stress_value=0.20,
            option_type="put",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.02,
            sigma=0.20,
            Smax=3.0,
            M=24,
            N=24,
        )
        run = run_risk_analytics_case(case)
        self.assertEqual(run.solver_variant, "baseline_cn_psor")
        self.assertTrue(np.isfinite(run.runtime_seconds))
        self.assertGreaterEqual(run.boundary_summary.found_boundary_count, 0)

        run_row = run_summary_row(run)
        boundary_rows = boundary_metric_rows(run, selected_tau_fractions=(0.5, 1.0))
        greek_row = greek_metric_row(run)
        lcp_row = lcp_metric_row(run)
        runtime_row = runtime_iteration_row(run)

        self.assertTrue(set(RUN_COLUMNS).issubset(run_row))
        self.assertEqual(len(boundary_rows), 2)
        self.assertTrue(set(BOUNDARY_COLUMNS).issubset(boundary_rows[0]))
        self.assertTrue(set(GREEK_COLUMNS).issubset(greek_row))
        self.assertTrue(set(LCP_COLUMNS).issubset(lcp_row))
        self.assertTrue(set(RUNTIME_COLUMNS).issubset(runtime_row))
        self.assertEqual(run_row["downstream_use_status"], "analytics_diagnostic_only")

    def test_batch_runner_caches_duplicate_parameter_solves(self):
        from american_risk_surfaces.downstream.risk_analytics import (
            RiskAnalyticsCase,
            run_risk_analytics_cases,
        )

        first = RiskAnalyticsCase(
            "base_in_first_sweep",
            "american_put",
            "put_volatility_sweep",
            "sigma",
            0.20,
            "put",
            1.0,
            0.25,
            0.05,
            0.02,
            0.20,
            3.0,
            18,
            18,
        )
        second = RiskAnalyticsCase(
            "base_in_second_sweep",
            "american_put",
            "put_rate_sweep",
            "r",
            0.05,
            "put",
            1.0,
            0.25,
            0.05,
            0.02,
            0.20,
            3.0,
            18,
            18,
        )
        runs = run_risk_analytics_cases((first, second), cache_duplicate_parameters=True)
        self.assertEqual(len(runs), 2)
        self.assertIs(runs[0].result, runs[1].result)
        self.assertEqual(runs[0].case.case_name, "base_in_first_sweep")
        self.assertEqual(runs[1].case.case_name, "base_in_second_sweep")


class RiskAnalyticsExperimentTests(unittest.TestCase):
    def test_experiment_writes_expected_csv_artifacts(self):
        module = _load_experiment_module()
        from american_risk_surfaces.downstream.risk_analytics import RiskAnalyticsCase

        case = RiskAnalyticsCase(
            "tiny_call_artifact",
            "dividend_call",
            "call_dividend_yield_sweep",
            "q",
            0.08,
            "call",
            1.0,
            0.25,
            0.05,
            0.08,
            0.20,
            3.0,
            18,
            18,
        )
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (
                run_rows,
                boundary_rows,
                greek_rows,
                lcp_rows,
                runtime_rows,
                manifest_rows,
                metadata,
            ) = module.main(
                cases=(case,),
                table_dir=root / "tables",
                figure_dir=root / "figures",
                create_figures=False,
                include_q_sigma_heatmap=False,
            )

            self.assertEqual(len(run_rows), 1)
            self.assertGreater(len(boundary_rows), 0)
            self.assertEqual(len(greek_rows), 1)
            self.assertEqual(len(lcp_rows), 1)
            self.assertEqual(len(runtime_rows), 1)
            self.assertGreaterEqual(len(manifest_rows), 6)
            self.assertTrue(Path(metadata["run_summary_csv"]).exists())
            self.assertTrue(Path(metadata["output_manifest_csv"]).exists())
            with Path(metadata["run_summary_csv"]).open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertTrue(set(RUN_COLUMNS).issubset(reader.fieldnames or []))
            with Path(metadata["output_manifest_csv"]).open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertTrue(set(MANIFEST_COLUMNS).issubset(reader.fieldnames or []))

    def test_figures_created_when_matplotlib_available(self):
        module = _load_experiment_module()
        if module._load_pyplot() is None:
            self.skipTest("matplotlib is unavailable")

        from american_risk_surfaces.downstream.risk_analytics import RiskAnalyticsCase

        cases = (
            RiskAnalyticsCase(
                "tiny_put_sigma_020",
                "american_put",
                "put_volatility_sweep",
                "sigma",
                0.20,
                "put",
                1.0,
                0.25,
                0.05,
                0.02,
                0.20,
                3.0,
                18,
                18,
            ),
            RiskAnalyticsCase(
                "tiny_put_sigma_040",
                "american_put",
                "put_volatility_sweep",
                "sigma",
                0.40,
                "put",
                1.0,
                0.25,
                0.05,
                0.02,
                0.40,
                3.0,
                18,
                18,
            ),
            RiskAnalyticsCase(
                "tiny_call_q_003",
                "dividend_call",
                "call_dividend_yield_sweep",
                "q",
                0.03,
                "call",
                1.0,
                0.25,
                0.05,
                0.03,
                0.20,
                3.0,
                18,
                18,
            ),
            RiskAnalyticsCase(
                "tiny_call_q_010",
                "dividend_call",
                "call_dividend_yield_sweep",
                "q",
                0.10,
                "call",
                1.0,
                0.25,
                0.05,
                0.10,
                0.20,
                3.0,
                18,
                18,
            ),
        )
        with TemporaryDirectory() as tmpdir:
            *_, manifest_rows, metadata = module.main(
                cases=cases,
                table_dir=Path(tmpdir) / "tables",
                figure_dir=Path(tmpdir) / "figures",
                create_figures=True,
                include_q_sigma_heatmap=False,
            )
            figure_paths = [
                Path(row["path"])
                for row in manifest_rows
                if row["output_type"] == "figure" and row["created"] == "True"
            ]
            self.assertGreater(len(figure_paths), 0)
            self.assertTrue(all(path.exists() for path in figure_paths))
            self.assertEqual(metadata["put_boundary_vs_volatility_figure_created"], "True")


class RiskAnalyticsScopeTests(unittest.TestCase):
    def test_no_forbidden_files_or_public_api_are_introduced(self):
        from american_risk_surfaces.downstream import risk_analytics

        forbidden_api = ("dataset", "neural", "surrogate", "training", "ml_label", "model")
        public_names = [
            name for name, _ in inspect.getmembers(risk_analytics) if not name.startswith("_")
        ]
        lowered_names = " ".join(public_names).lower()
        for word in forbidden_api:
            self.assertNotIn(word, lowered_names)

        forbidden_paths = [
            path
            for path in PROJECT_ROOT.rglob("*")
            if "risk_analytics" in path.name.lower()
            and any(word in path.name.lower() for word in forbidden_api)
        ]
        self.assertEqual(forbidden_paths, [])

    def test_new_python_files_include_required_docstrings(self):
        for path in (SOURCE_PATH, SCRIPT_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertIn(EXPECTED_DOCSTRING, text)


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location("risk_analytics_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
