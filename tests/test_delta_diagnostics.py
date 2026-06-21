"""Stage 5: Delta diagnostic and supervised Delta-head comparison."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from american_risk_surfaces.surrogates import delta as stage5
from american_risk_surfaces.surrogates import price_premium as stage3


class DeltaDiagnosticsTests(unittest.TestCase):
    def test_v1_dataset_has_delta_and_delta_allowed_mask(self) -> None:
        bundle = stage3.load_v1_dataset()
        self.assertEqual(bundle.X.shape[0], bundle.y_delta.shape[0])
        self.assertIn("delta_allowed_mask", stage3.MASK_INDEX)
        allowed = stage5.delta_allowed_mask(bundle)
        self.assertEqual(bundle.X.shape[0], allowed.shape[0])
        self.assertGreater(int(np.count_nonzero(allowed)), 0)
        self.assertTrue(np.all(np.isfinite(bundle.y_delta[allowed])))

    def test_delta_training_rows_use_train_split_and_allowed_mask(self) -> None:
        bundle = _synthetic_bundle()
        rows = stage5.delta_training_indices(bundle)
        train_mask = stage3.split_masks(bundle)["train"]
        expected = np.flatnonzero(train_mask & bundle.masks[:, stage3.MASK_INDEX["delta_allowed_mask"]])
        np.testing.assert_array_equal(expected, rows)

    def test_gamma_targets_are_rejected(self) -> None:
        for target in ("scaled_gamma", "gamma", "y_scaled_gamma", "boundary_spot_over_K"):
            with self.assertRaises(ValueError):
                stage5.validate_delta_training_target(target)

    def test_log_moneyness_finite_difference_matches_smooth_function(self) -> None:
        log_m = np.array([-0.2, 0.0, 0.4])
        features = np.column_stack(
            [
                log_m,
                np.full(3, 0.5),
                np.full(3, 0.05),
                np.full(3, 0.02),
                np.full(3, 0.2),
                np.ones(3),
                np.zeros(3),
            ]
        )

        def smooth_value(X: np.ndarray) -> np.ndarray:  # noqa: N803
            moneyness = np.exp(X[:, stage3.FEATURE_NAMES.index("log_moneyness")])
            return moneyness**2

        delta = stage5.central_log_moneyness_delta(features, smooth_value, step=1e-5)
        expected = 2.0 * np.exp(log_m)
        np.testing.assert_allclose(expected, delta, rtol=1e-5, atol=1e-5)

    def test_supervised_delta_head_trains_on_tiny_synthetic_data(self) -> None:
        bundle = _synthetic_bundle(row_count=160)
        prediction, history = stage5.train_supervised_delta_head(
            bundle,
            train_cap=80,
            epochs=2,
            batch_size=32,
        )
        self.assertEqual(bundle.X.shape[0], len(prediction.predicted_delta))
        self.assertGreater(len(history), 0)
        self.assertTrue(np.all(np.isfinite(prediction.predicted_delta)))
        calls = bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5
        self.assertTrue(np.all(prediction.predicted_delta[calls] >= -1e-12))
        self.assertTrue(np.all(prediction.predicted_delta[calls] <= 1.0 + 1e-12))
        self.assertTrue(np.all(prediction.predicted_delta[~calls] >= -1.0 - 1e-12))
        self.assertTrue(np.all(prediction.predicted_delta[~calls] <= 1e-12))

    def test_delta_bounds_violation_metric_detects_bad_predictions(self) -> None:
        bundle = _synthetic_bundle()
        bad = stage5.DeltaPrediction(
            method_name="bad_delta",
            predicted_delta=np.where(bundle.X[:, stage3.FEATURE_NAMES.index("is_call")] >= 0.5, -0.2, 0.2),
        )
        rows = stage5.delta_bounds_violation_rows(bundle, {"bad_delta": bad})
        self.assertEqual(stage5.DELTA_BOUNDS_FIELDNAMES, list(rows[0]))
        self.assertTrue(any(float(row["bounds_violation_rate"]) > 0.0 for row in rows))
        self.assertTrue(any(float(row["sign_violation_rate"]) > 0.0 for row in rows))

    def test_metric_csv_schemas_are_stable(self) -> None:
        bundle = _synthetic_bundle()
        prediction = stage5.DeltaPrediction(
            method_name="synthetic_delta",
            predicted_delta=bundle.y_delta.copy(),
        )
        predictions = {"synthetic_delta": prediction}
        self.assertEqual(stage5.DELTA_METRICS_BY_SPLIT_FIELDNAMES, list(stage5.delta_metrics_by_split_rows(bundle, predictions)[0]))
        self.assertEqual(
            stage5.DELTA_METRICS_BY_OPTION_TYPE_FIELDNAMES,
            list(stage5.delta_metrics_by_option_type_rows(bundle, predictions)[0]),
        )
        self.assertEqual(stage5.DELTA_METRICS_BY_REGION_FIELDNAMES, list(stage5.delta_metrics_by_region_rows(bundle, predictions)[0]))
        self.assertEqual(stage5.DELTA_CURVE_SAMPLE_AUDIT_FIELDNAMES, list(stage5.delta_curve_sample_audit_rows(bundle, predictions)[0]))

    def test_report_path_is_configurable_for_synthetic_runs(self) -> None:
        bundle = _synthetic_bundle()
        prediction = stage5.DeltaPrediction(
            method_name="synthetic_delta",
            predicted_delta=bundle.y_delta.copy(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = output_dir / "synthetic_delta_report.tex"
            result = stage5.write_delta_outputs(
                output_dir=output_dir,
                report_tex_path=report_path,
                bundle=bundle,
                predictions={"synthetic_delta": prediction},
                training_histories={"synthetic_delta": [{"epoch": 1.0, "train_loss": 0.0, "validation_loss": 0.0}]},
                create_figures=False,
            )
            self.assertEqual(report_path, result.report_tex_path)
            self.assertTrue(report_path.exists())
            for path in (
                result.metrics_by_split_path,
                result.metrics_by_option_type_path,
                result.metrics_by_region_path,
                result.bounds_summary_path,
                result.curve_sample_audit_path,
                result.model_manifest_path,
            ):
                self.assertTrue(path.exists(), path)
                self.assertGreater(len(_read_csv(path)), 0)
            self.assertFalse(list(output_dir.rglob("*.pt")))

    def test_no_gamma_head_or_gamma_target_api(self) -> None:
        public = " ".join(stage5.__all__).lower()
        self.assertNotIn("gamma_head", public)
        self.assertNotIn("train_gamma", public)

    def test_new_python_files_include_stage_5_docstring(self) -> None:
        expected = '"""Stage 5: Delta diagnostic and supervised Delta-head comparison."""'
        for path in (
            Path("src/american_risk_surfaces/surrogates/delta.py"),
            Path("experiments/17_delta_diagnostics.py"),
            Path("tests/test_delta_diagnostics.py"),
        ):
            self.assertTrue(path.exists(), path)
            self.assertIn(expected, path.read_text())


def _synthetic_bundle(row_count: int = 80) -> stage3.SurrogateDatasetBundle:
    rng = np.random.default_rng(456)
    log_m = np.linspace(np.log(0.4), np.log(1.8), row_count)
    is_call = np.resize(np.array([0.0, 1.0]), row_count)
    X = np.column_stack(
        [
            log_m,
            np.linspace(0.0, 1.0, row_count),
            np.full(row_count, 0.05),
            np.where(is_call >= 0.5, 0.08, 0.02),
            np.full(row_count, 0.2),
            np.ones(row_count),
            is_call,
        ]
    ).astype(float)
    moneyness = np.exp(log_m)
    payoff = np.where(is_call >= 0.5, np.maximum(moneyness - 1.0, 0.0), np.maximum(1.0 - moneyness, 0.0))
    premium = 0.02 + 0.01 * rng.random(row_count)
    value = payoff + premium
    delta = np.where(is_call >= 0.5, np.clip(0.1 + 0.6 * moneyness / 1.8, 0.0, 1.0), -np.clip(1.1 - 0.6 * moneyness / 1.8, 0.0, 1.0))
    masks = np.zeros((row_count, len(stage3.MASK_NAMES)), dtype=bool)
    masks[:, stage3.MASK_INDEX["strict_interior"]] = True
    masks[:, stage3.MASK_INDEX["delta_allowed_mask"]] = True
    masks[:, stage3.MASK_INDEX["gamma_allowed_mask"]] = True
    masks[::7, stage3.MASK_INDEX["boundary_near"]] = True
    masks[::11, stage3.MASK_INDEX["payoff_kink_near"]] = True
    masks[::13, stage3.MASK_INDEX["maturity_row"]] = True
    split_index = np.resize(np.arange(4), row_count)
    audit = np.zeros((row_count, len(stage3.AUDIT_NUMERIC_NAMES)), dtype=float)
    audit[:, stage3.AUDIT_NUMERIC_INDEX["S_over_K"]] = moneyness
    audit[:, stage3.AUDIT_NUMERIC_INDEX["tau"]] = X[:, stage3.FEATURE_NAMES.index("tau_fraction")]
    audit[:, stage3.AUDIT_NUMERIC_INDEX["split_index"]] = split_index
    return stage3.SurrogateDatasetBundle(
        dataset_path=Path("synthetic_delta.npz"),
        X=X,
        y_value=value,
        y_payoff=payoff,
        y_premium=premium,
        y_exercise_indicator=(premium <= 1e-6).astype(float),
        y_boundary=np.full(row_count, np.nan),
        y_delta=delta,
        y_scaled_gamma=np.zeros(row_count),
        masks=masks,
        regime_index=np.zeros(row_count, dtype=int),
        feature_names=np.array(stage3.FEATURE_NAMES),
        label_names=np.array(stage3.LABEL_NAMES),
        mask_names=np.array(stage3.MASK_NAMES),
        audit_numeric=audit,
        audit_numeric_names=np.array(stage3.AUDIT_NUMERIC_NAMES),
        regime_ids=np.array(["synthetic_delta_regime"]),
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
