"""v0 Dry-Run Dataset tests for the eight-regime generation gate."""

from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "american_risk_surfaces"
    / "data"
    / "dry_run_dataset.py"
)
INIT_PATH = PROJECT_ROOT / "src" / "american_risk_surfaces" / "data" / "__init__.py"
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "13_v0_dry_run_dataset.py"
EXPECTED_DOCSTRING = "v0 Dry-Run Dataset: eight-regime surrogate dataset generation test."

NPZ_KEYS = {
    "X",
    "y_value",
    "y_payoff",
    "y_premium",
    "y_exercise_indicator",
    "y_boundary",
    "y_delta",
    "y_scaled_gamma",
    "masks",
    "regime_index",
    "feature_names",
    "label_names",
    "mask_names",
    "audit_numeric",
    "audit_numeric_names",
    "regime_ids",
    "dry_run_ids",
    "split_names",
}

REGIME_MANIFEST_COLUMNS = {
    "dry_run_id",
    "regime_id",
    "option_type",
    "T",
    "sigma",
    "r",
    "q",
    "K",
    "Smax",
    "M",
    "N",
    "dS",
    "dtau",
    "split",
    "solver_name",
    "solver_variant",
    "runtime_seconds",
    "accepted_sample_rows",
    "acceptance_status",
    "downstream_use_status",
}

DIAGNOSTIC_COLUMNS = {
    "dry_run_id",
    "regime_id",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "boundary_found_count",
    "boundary_threshold",
    "finite_delta_count",
    "finite_gamma_count",
    "max_abs_gamma_strict",
    "runtime_seconds",
    "acceptance_status",
    "downstream_use_status",
}


class DryRunPlanTests(unittest.TestCase):
    def test_loads_exactly_the_approved_eight_regimes(self):
        from american_risk_surfaces.data.dry_run_dataset import load_dry_run_plan

        regimes = load_dry_run_plan()

        self.assertEqual(len(regimes), 8)
        self.assertEqual(
            [regime.dry_run_id for regime in regimes],
            [f"dry_{index:02d}" for index in range(1, 9)],
        )
        self.assertEqual(regimes[0].regime_id, "put_T100_s020_r005_q003")
        self.assertEqual(regimes[-1].regime_id, "call_T100_s020_r005_q000")
        self.assertTrue(all(regime.K == 1.0 for regime in regimes))
        self.assertTrue(all(regime.Smax == 4.0 for regime in regimes))
        self.assertTrue(all((regime.M, regime.N) == (120, 120) for regime in regimes))
        self.assertTrue(all(regime.solver_variant == "baseline_cn_psor" for regime in regimes))

    def test_full_288_regime_plan_is_only_validation_context(self):
        from american_risk_surfaces.data.dry_run_dataset import (
            expected_full_regime_count,
            load_dry_run_plan,
        )

        self.assertEqual(expected_full_regime_count(), 288)
        self.assertEqual(len(load_dry_run_plan()), 8)

    def test_feature_label_mask_and_audit_names_are_stable(self):
        from american_risk_surfaces.data.dry_run_dataset import (
            AUDIT_NUMERIC_NAMES,
            FEATURE_NAMES,
            LABEL_NAMES,
            MASK_NAMES,
        )

        self.assertEqual(
            FEATURE_NAMES,
            (
                "log_moneyness",
                "tau_fraction",
                "r",
                "q",
                "sigma",
                "T",
                "is_call",
            ),
        )
        self.assertIn("value_over_K", LABEL_NAMES)
        self.assertIn("scaled_gamma", LABEL_NAMES)
        self.assertEqual(
            MASK_NAMES,
            (
                "payoff_kink_near",
                "boundary_near",
                "maturity_row",
                "strict_interior",
                "gamma_allowed_mask",
                "delta_allowed_mask",
                "exercise_region",
                "continuation_region",
            ),
        )
        self.assertIn("S_over_K", AUDIT_NUMERIC_NAMES)
        self.assertIn("tau", AUDIT_NUMERIC_NAMES)


class DryRunGenerationTests(unittest.TestCase):
    def test_small_smoke_run_completes_and_uses_baseline_solver(self):
        from american_risk_surfaces.data.dry_run_dataset import (
            DryRunRegime,
            run_dry_run_regime,
        )

        regime = DryRunRegime(
            dry_run_id="dry_smoke",
            regime_id="put_smoke",
            option_type="put",
            T=0.5,
            sigma=0.2,
            r=0.05,
            q=0.03,
            K=1.0,
            Smax=3.0,
            M=24,
            N=24,
            split="train",
            split_reason="unit smoke",
            solver_variant="baseline_cn_psor",
            reason_selected="unit smoke",
            expected_check="unit smoke",
            required_manual_review="yes",
        )

        artifacts = run_dry_run_regime(regime)

        self.assertEqual(artifacts.solver_name, "american_crank_nicolson_psor_price")
        self.assertEqual(artifacts.solver_variant, "baseline_cn_psor")
        self.assertNotIn("rannacher", artifacts.solver_variant.lower())
        self.assertTrue(artifacts.result.converged)
        self.assertTrue(np.isfinite(artifacts.lcp_diagnostics.summary.max_equation_violation))

    def test_sample_rows_have_premium_masks_and_expected_shapes(self):
        from american_risk_surfaces.data.dry_run_dataset import (
            DryRunRegime,
            build_dataset_arrays,
            sample_regime_rows,
            run_dry_run_regime,
        )

        regime = DryRunRegime(
            dry_run_id="dry_smoke",
            regime_id="call_smoke",
            option_type="call",
            T=0.5,
            sigma=0.2,
            r=0.05,
            q=0.08,
            K=1.0,
            Smax=3.0,
            M=24,
            N=24,
            split="train",
            split_reason="unit smoke",
            solver_variant="baseline_cn_psor",
            reason_selected="unit smoke",
            expected_check="unit smoke",
            required_manual_review="yes",
        )
        artifacts = run_dry_run_regime(regime)
        rows = sample_regime_rows(artifacts)
        arrays = build_dataset_arrays(rows)

        self.assertGreater(len(rows), 0)
        self.assertEqual(arrays["X"].shape[0], arrays["masks"].shape[0])
        self.assertEqual(arrays["X"].shape[1], 7)
        self.assertEqual(arrays["masks"].shape[1], 8)
        np.testing.assert_allclose(arrays["y_premium"], arrays["y_value"] - arrays["y_payoff"])

    def test_failed_regime_handling_marks_review_required(self):
        from american_risk_surfaces.data.dry_run_dataset import evaluate_acceptance

        status, reason = evaluate_acceptance(
            all_psor_steps_converged=False,
            max_obstacle_violation=0.0,
            max_equation_violation=0.0,
            max_abs_complementarity_product=0.0,
            metadata_complete=True,
        )

        self.assertEqual(status, "review_required")
        self.assertIn("psor_not_converged", reason)


class DryRunOutputTests(unittest.TestCase):
    def test_generation_writes_npz_and_manifest_csvs(self):
        from american_risk_surfaces.data.dry_run_dataset import (
            DryRunRegime,
            generate_v0_dry_run_dataset,
        )

        regimes = (
            DryRunRegime(
                dry_run_id="dry_smoke",
                regime_id="put_smoke",
                option_type="put",
                T=0.5,
                sigma=0.2,
                r=0.05,
                q=0.03,
                K=1.0,
                Smax=3.0,
                M=24,
                N=24,
                split="train",
                split_reason="unit smoke",
                solver_variant="baseline_cn_psor",
                reason_selected="unit smoke",
                expected_check="unit smoke",
                required_manual_review="yes",
            ),
        )
        with TemporaryDirectory() as tmpdir:
            package = generate_v0_dry_run_dataset(
                output_dir=Path(tmpdir),
                regimes=regimes,
                create_figures=False,
            )

            self.assertTrue(package.npz_path.exists())
            with np.load(package.npz_path, allow_pickle=True) as data:
                self.assertTrue(NPZ_KEYS.issubset(data.files))
                self.assertEqual(data["X"].shape[0], data["masks"].shape[0])

            for path in (
                package.regime_manifest_path,
                package.diagnostic_summary_path,
                package.split_assignment_path,
                package.schema_snapshot_path,
                package.output_manifest_path,
            ):
                self.assertTrue(path.exists())
                with path.open(newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertGreater(len(rows), 0)

            with package.regime_manifest_path.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertTrue(REGIME_MANIFEST_COLUMNS.issubset(row))
            self.assertEqual(row["downstream_use_status"], "v0_dry_run_only")

            with package.diagnostic_summary_path.open(newline="") as file:
                diagnostic_row = next(csv.DictReader(file))
            self.assertTrue(DIAGNOSTIC_COLUMNS.issubset(diagnostic_row))

            with package.output_manifest_path.open(newline="") as file:
                manifest_rows = list(csv.DictReader(file))
            self.assertTrue(
                any(
                    row["output_id"] == "output_manifest" and row["created"] == "True"
                    for row in manifest_rows
                )
            )

    def test_experiment_script_runs_with_reduced_regime_set(self):
        spec = importlib.util.spec_from_file_location("v0_dry_run_experiment", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        from american_risk_surfaces.data.dry_run_dataset import DryRunRegime

        regime = DryRunRegime(
            dry_run_id="dry_smoke",
            regime_id="put_smoke",
            option_type="put",
            T=0.5,
            sigma=0.2,
            r=0.05,
            q=0.03,
            K=1.0,
            Smax=3.0,
            M=24,
            N=24,
            split="train",
            split_reason="unit smoke",
            solver_variant="baseline_cn_psor",
            reason_selected="unit smoke",
            expected_check="unit smoke",
            required_manual_review="yes",
        )

        with TemporaryDirectory() as tmpdir:
            package = module.main(
                output_dir=Path(tmpdir),
                regimes=(regime,),
                create_figures=False,
            )
        self.assertTrue(package.review_decision in {"READY_FOR_V1_SMALL_GRID_PLANNING", "REVIEW_REQUIRED_BEFORE_V1"})


class DryRunScopeTests(unittest.TestCase):
    def test_no_forbidden_model_training_or_pt_api_is_introduced(self):
        import american_risk_surfaces.data.dry_run_dataset as dry_run_dataset

        public_names = {
            name.lower()
            for name, _ in inspect.getmembers(dry_run_dataset)
            if not name.startswith("_")
        }
        forbidden = ("neural", "model", "training", "fit", "torch", "pt_file")
        for name in public_names:
            self.assertFalse(
                any(token in name for token in forbidden),
                f"forbidden public API name found: {name}",
            )

    def test_new_python_files_include_required_docstrings(self):
        self.assertEqual(SOURCE_PATH.read_text().splitlines()[0].strip('"'), EXPECTED_DOCSTRING)
        self.assertEqual(SCRIPT_PATH.read_text().splitlines()[0].strip('"'), EXPECTED_DOCSTRING)
        self.assertEqual(INIT_PATH.read_text().splitlines()[0].strip('"'), EXPECTED_DOCSTRING)


if __name__ == "__main__":
    unittest.main()
