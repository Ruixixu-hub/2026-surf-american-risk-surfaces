"""Stage 3: price and positive-premium surrogate comparison."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from american_risk_surfaces.surrogates import price_premium as stage3


class PricePremiumSurrogateTests(unittest.TestCase):
    def test_v1_dataset_loads_expected_keys(self) -> None:
        bundle = stage3.load_v1_dataset()
        self.assertEqual(stage3.EXPECTED_DATASET_KEYS, tuple(bundle.arrays.files))
        self.assertEqual(7, bundle.X.shape[1])
        self.assertEqual(bundle.X.shape[0], bundle.y_value.shape[0])
        self.assertEqual(bundle.X.shape[0], bundle.masks.shape[0])

    def test_split_masks_are_disjoint_and_named(self) -> None:
        bundle = _synthetic_bundle()
        masks = stage3.split_masks(bundle)
        self.assertEqual(("train", "validation", "test", "stress_holdout"), tuple(masks))
        total = sum(mask.astype(int) for mask in masks.values())
        np.testing.assert_array_equal(total, np.ones(bundle.X.shape[0], dtype=int))

    def test_scaler_is_fit_on_train_only(self) -> None:
        bundle = _synthetic_bundle()
        train_mask = stage3.split_masks(bundle)["train"]
        train_indices = np.flatnonzero(train_mask)
        preprocessor = stage3.fit_preprocessor(bundle, train_indices, model_name="direct_value_mlp")
        transformed_train = preprocessor.transform_direct(bundle.X[train_indices])
        np.testing.assert_allclose(transformed_train.mean(axis=0), np.zeros(7), atol=1e-12)

    def test_capped_train_sampling_is_deterministic(self) -> None:
        train_indices = np.arange(1000)
        first = stage3.capped_train_indices(train_indices, cap=25, seed=stage3.RANDOM_SEED)
        second = stage3.capped_train_indices(train_indices, cap=25, seed=stage3.RANDOM_SEED)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(25, len(first))
        self.assertTrue(np.all(first[:-1] <= first[1:]))

    def test_direct_and_premium_smoke_training(self) -> None:
        bundle = _synthetic_bundle(row_count=160)
        output_dir = Path(tempfile.mkdtemp())
        report_path = output_dir / "synthetic_stage3_report.tex"
        result = stage3.run_surrogate_experiment(
            bundle=bundle,
            output_dir=output_dir,
            report_tex_path=report_path,
            train_cap=80,
            epochs=2,
            batch_size=32,
            create_figures=False,
        )
        self.assertIn("direct_value_mlp", result.predictions)
        self.assertIn("positive_premium_mlp", result.predictions)
        premium_prediction = result.predictions["positive_premium_mlp"]
        np.testing.assert_allclose(
            premium_prediction.predicted_value,
            bundle.y_payoff + premium_prediction.predicted_premium,
            atol=1e-7,
        )
        self.assertEqual(report_path, result.report_tex_path)
        self.assertTrue(report_path.exists())

    def test_relative_error_and_violation_tolerances_are_fixed(self) -> None:
        self.assertEqual(1e-4, stage3.RELATIVE_ERROR_DENOMINATOR_FLOOR)
        self.assertEqual(1e-4, stage3.NEAR_ZERO_RATE_TOLERANCE)

    def test_metric_rows_have_stable_columns(self) -> None:
        bundle = _synthetic_bundle()
        prediction = stage3.ModelPrediction(
            model_name="direct_value_mlp",
            predicted_value=bundle.y_value.copy(),
            predicted_premium=bundle.y_premium.copy(),
        )
        rows = stage3.metrics_by_split_rows(bundle, {"direct_value_mlp": prediction})
        self.assertTrue(rows)
        self.assertEqual(stage3.METRICS_BY_SPLIT_FIELDNAMES, list(rows[0]))
        region_rows = stage3.metrics_by_region_rows(bundle, {"direct_value_mlp": prediction})
        self.assertEqual(stage3.METRICS_BY_REGION_FIELDNAMES, list(region_rows[0]))

    def test_obstacle_violation_metric_detects_negative_premium(self) -> None:
        bundle = _synthetic_bundle()
        prediction = stage3.ModelPrediction(
            model_name="bad_model",
            predicted_value=bundle.y_payoff - 0.01,
            predicted_premium=np.full(bundle.y_payoff.shape, -0.01),
        )
        rows = stage3.obstacle_violation_rows(bundle, {"bad_model": prediction})
        self.assertGreater(float(rows[0]["obstacle_violation_rate"]), 0.0)
        self.assertGreater(float(rows[0]["negative_premium_rate"]), 0.0)

    def test_forbidden_targets_and_heads_are_not_public_api(self) -> None:
        for target in ("delta", "scaled_gamma", "boundary_spot_over_K"):
            with self.assertRaises(ValueError):
                stage3.validate_training_target(target)
        public = " ".join(stage3.__all__).lower()
        self.assertNotIn("boundary_head", public)
        self.assertNotIn("delta_head", public)

    def test_experiment_writes_csvs_without_model_weights(self) -> None:
        bundle = _synthetic_bundle(row_count=160)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = output_dir / "synthetic_stage3_report.tex"
            result = stage3.run_surrogate_experiment(
                bundle=bundle,
                output_dir=output_dir,
                report_tex_path=report_path,
                train_cap=80,
                epochs=2,
                batch_size=32,
                create_figures=False,
            )
            for path in (
                result.metrics_by_split_path,
                result.metrics_by_region_path,
                result.obstacle_summary_path,
                result.prediction_sample_audit_path,
                result.model_run_manifest_path,
            ):
                self.assertTrue(path.exists(), path)
                self.assertGreater(len(_read_csv(path)), 0)
            self.assertEqual(report_path, result.report_tex_path)
            self.assertTrue(report_path.exists())
            self.assertFalse(list(Path(tmpdir).rglob("*.pt")))

    def test_new_python_files_include_stage_3_docstring(self) -> None:
        expected = '"""Stage 3: price and positive-premium surrogate comparison."""'
        for path in (
            Path("src/american_risk_surfaces/surrogates/price_premium.py"),
            Path("experiments/15_price_premium_surrogate.py"),
            Path("tests/test_price_premium_surrogate.py"),
        ):
            self.assertTrue(path.exists(), path)
            self.assertIn(expected, path.read_text())


def _synthetic_bundle(row_count: int = 80) -> stage3.SurrogateDatasetBundle:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(row_count, 7)).astype(float)
    payoff = np.maximum(0.0, 0.8 - np.linspace(0.4, 1.8, row_count))
    premium = 0.02 + 0.01 * rng.random(row_count)
    value = payoff + premium
    masks = np.zeros((row_count, 8), dtype=bool)
    masks[:, stage3.MASK_INDEX["strict_interior"]] = True
    masks[::5, stage3.MASK_INDEX["boundary_near"]] = True
    split_index = np.repeat(np.arange(4), row_count // 4)
    if len(split_index) < row_count:
        split_index = np.concatenate([split_index, np.full(row_count - len(split_index), 3)])
    audit = np.zeros((row_count, 11), dtype=float)
    audit[:, stage3.AUDIT_NUMERIC_INDEX["split_index"]] = split_index[:row_count]
    return stage3.SurrogateDatasetBundle(
        dataset_path=Path("synthetic.npz"),
        X=X,
        y_value=value,
        y_payoff=payoff,
        y_premium=premium,
        y_exercise_indicator=(premium <= 1e-6).astype(float),
        y_boundary=np.full(row_count, np.nan),
        y_delta=np.zeros(row_count),
        y_scaled_gamma=np.zeros(row_count),
        masks=masks,
        regime_index=np.zeros(row_count, dtype=int),
        feature_names=np.array(stage3.FEATURE_NAMES),
        label_names=np.array(stage3.LABEL_NAMES),
        mask_names=np.array(stage3.MASK_NAMES),
        audit_numeric=audit,
        audit_numeric_names=np.array(stage3.AUDIT_NUMERIC_NAMES),
        regime_ids=np.array(["synthetic_regime"]),
        split_names=np.array(stage3.SPLIT_NAMES),
        arrays=_SyntheticFiles(),
    )


class _SyntheticFiles:
    files = stage3.EXPECTED_DATASET_KEYS


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
