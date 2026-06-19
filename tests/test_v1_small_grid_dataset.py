"""v1 Small-Grid Dataset: 288-regime surrogate dataset generation and QA."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from american_risk_surfaces.data import small_grid_dataset as v1


class V1SmallGridDatasetTests(unittest.TestCase):
    def test_plan_loads_exactly_288_regimes_with_expected_splits(self) -> None:
        regimes = v1.load_small_grid_plan()
        self.assertEqual(288, len(regimes))
        counts = v1.split_counts(regimes)
        self.assertEqual(
            {"train": 202, "validation": 19, "test": 43, "stress_holdout": 24},
            counts,
        )
        self.assertTrue(all(regime.solver_variant == "baseline_cn_psor" for regime in regimes))

    def test_feature_label_mask_and_audit_names_are_stable(self) -> None:
        self.assertEqual(
            (
                "log_moneyness",
                "tau_fraction",
                "r",
                "q",
                "sigma",
                "T",
                "is_call",
            ),
            v1.FEATURE_NAMES,
        )
        self.assertEqual(
            (
                "value_over_K",
                "payoff_over_K",
                "premium_over_K",
                "exercise_indicator",
                "boundary_spot_over_K",
                "delta",
                "scaled_gamma",
            ),
            v1.LABEL_NAMES,
        )
        self.assertEqual(8, len(v1.MASK_NAMES))
        self.assertIn("split_index", v1.AUDIT_NUMERIC_NAMES)

    def test_reduced_smoke_generation_writes_expected_npz_and_manifests(self) -> None:
        regimes = tuple(
            replace(regime, M=18, N=18)
            for regime in v1.load_small_grid_plan()[:2]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package = v1.generate_v1_small_grid_dataset(
                output_dir=Path(tmpdir),
                regimes=regimes,
                create_figures=False,
                include_higher_grid_confirmation=False,
            )
            self.assertTrue(package.npz_path.exists())
            self.assertEqual(2, package.total_regime_count)
            self.assertGreater(package.accepted_row_count, 0)
            self.assertIn(
                package.review_decision,
                {"READY_FOR_PRICE_SURROGATE_PLANNING", "REVIEW_REQUIRED_BEFORE_SURROGATE"},
            )
            with np.load(package.npz_path) as data:
                self.assertEqual(v1.EXPECTED_NPZ_KEYS, tuple(data.files))
                self.assertEqual((package.accepted_row_count, 7), data["X"].shape)
                self.assertEqual((package.accepted_row_count, 8), data["masks"].shape)
                np.testing.assert_allclose(
                    data["y_premium"],
                    data["y_value"] - data["y_payoff"],
                    atol=1e-10,
                )
            for path in (
                package.regime_manifest_path,
                package.diagnostic_summary_path,
                package.split_assignment_path,
                package.schema_snapshot_path,
                package.output_manifest_path,
                package.quality_summary_path,
                package.higher_grid_confirmation_path,
            ):
                self.assertTrue(path.exists(), path)
                self.assertGreater(len(_read_csv(path)), 0, path)

    def test_acceptance_helper_marks_failed_regime_review_required(self) -> None:
        status, reason = v1.evaluate_acceptance(
            all_psor_steps_converged=False,
            max_obstacle_violation=0.0,
            max_equation_violation=0.0,
            max_abs_complementarity_product=0.0,
            metadata_complete=True,
        )
        self.assertEqual("review_required", status)
        self.assertIn("psor_not_converged", reason)

    def test_manifest_columns_are_stable(self) -> None:
        self.assertIn("regime_id", v1.REGIME_MANIFEST_FIELDNAMES)
        self.assertIn("downstream_use_status", v1.REGIME_MANIFEST_FIELDNAMES)
        self.assertIn("max_abs_gamma_strict", v1.DIAGNOSTIC_SUMMARY_FIELDNAMES)
        self.assertIn("created", v1.OUTPUT_MANIFEST_FIELDNAMES)
        self.assertIn("review_decision", v1.QUALITY_SUMMARY_FIELDNAMES)

    def test_experiment_entrypoint_uses_v1_generator(self) -> None:
        experiment = Path("experiments/14_v1_small_grid_dataset.py")
        spec = importlib.util.spec_from_file_location("v1_experiment", experiment)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_no_neural_model_training_or_pt_api_is_introduced(self) -> None:
        public_names = set(v1.__all__)
        forbidden_fragments = ("neural", "training", "model_file", "pt_file")
        for name in public_names:
            lowered = name.lower()
            self.assertFalse(
                any(fragment in lowered for fragment in forbidden_fragments),
                name,
            )
        self.assertEqual("baseline_cn_psor", v1.SOLVER_VARIANT)

    def test_new_python_files_include_required_docstring(self) -> None:
        expected = '"""v1 Small-Grid Dataset: 288-regime surrogate dataset generation and QA."""'
        for path in (
            Path("src/american_risk_surfaces/data/small_grid_dataset.py"),
            Path("experiments/14_v1_small_grid_dataset.py"),
            Path("tests/test_v1_small_grid_dataset.py"),
        ):
            self.assertTrue(path.exists(), path)
            self.assertIn(expected, path.read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return rows


if __name__ == "__main__":
    unittest.main()
