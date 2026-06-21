"""Stage 4: boundary diagnostic and boundary-head comparison."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from american_risk_surfaces.surrogates import boundary as stage4
from american_risk_surfaces.surrogates import price_premium as stage3


class BoundaryDiagnosticsTests(unittest.TestCase):
    def test_v1_dataset_loads_for_stage_4(self) -> None:
        bundle = stage3.load_v1_dataset()
        self.assertEqual(stage3.EXPECTED_DATASET_KEYS, tuple(bundle.arrays.files))
        self.assertEqual(bundle.X.shape[0], bundle.y_boundary.shape[0])

    def test_boundary_target_rows_are_one_per_regime_time(self) -> None:
        bundle = stage3.load_v1_dataset()
        targets = stage4.build_boundary_target_table(bundle)
        self.assertEqual(288 * 121, targets.row_count)
        self.assertEqual(targets.row_count, len(targets.curve_row_indices))
        self.assertTrue(np.all([len(indices) == 43 for indices in targets.curve_row_indices[:10]]))
        self.assertGreater(int(np.count_nonzero(targets.reference_found)), 0)
        self.assertGreater(int(np.count_nonzero(~targets.reference_found)), 0)

    def test_premium_implied_boundary_extraction_on_synthetic_curves(self) -> None:
        put = stage4.extract_premium_implied_boundary_for_curve(
            np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            np.array([0.0, 0.02, 0.08, 0.10, 0.20]),
            option_type="put",
            tau=0.5,
            time_index=1,
            threshold=0.05,
        )
        self.assertTrue(put.boundary_found)
        self.assertAlmostEqual(0.75, put.boundary_spot)
        call = stage4.extract_premium_implied_boundary_for_curve(
            np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            np.array([0.20, 0.10, 0.08, 0.02, 0.00]),
            option_type="call",
            tau=0.5,
            time_index=1,
            threshold=0.05,
        )
        self.assertTrue(call.boundary_found)
        self.assertAlmostEqual(1.25, call.boundary_spot)
        missing = stage4.extract_premium_implied_boundary_for_curve(
            np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            np.full(5, 0.2),
            option_type="put",
            tau=0.5,
            time_index=1,
            threshold=0.05,
        )
        self.assertFalse(missing.boundary_found)
        self.assertEqual("all_continuation_like", missing.no_boundary_reason)

    def test_boundary_metric_rows_are_computed_on_known_values(self) -> None:
        targets = _synthetic_targets()
        prediction = stage4.BoundaryPrediction(
            method_name="synthetic_method",
            predicted_boundary=np.array([0.7, 1.2, np.nan, 0.9]),
            predicted_found=np.array([True, True, False, True]),
            no_boundary_reason=np.array(["", "", "all_continuation_like", ""]),
        )
        rows = stage4.boundary_metrics_by_split_rows(targets, {"synthetic_method": prediction})
        self.assertEqual(stage4.BOUNDARY_METRICS_BY_SPLIT_FIELDNAMES, list(rows[0]))
        train = next(row for row in rows if row["split"] == "train")
        self.assertAlmostEqual(0.1, float(train["boundary_mae"]))
        self.assertAlmostEqual(0.1, float(train["boundary_rmse"]))
        self.assertAlmostEqual(1.0, float(train["boundary_found_agreement_rate"]))

    def test_direct_boundary_head_trains_on_tiny_synthetic_data(self) -> None:
        targets = _synthetic_targets(row_count=80)
        prediction = stage4.train_direct_boundary_head(targets)
        self.assertEqual(targets.row_count, len(prediction.predicted_boundary))
        self.assertTrue(np.any(prediction.predicted_found))

    def test_forbidden_delta_gamma_targets_and_heads_are_absent(self) -> None:
        for target in ("delta", "scaled_gamma"):
            with self.assertRaises(ValueError):
                stage4.validate_boundary_training_target(target)
        public = " ".join(stage4.__all__).lower()
        self.assertNotIn("delta_head", public)
        self.assertNotIn("gamma_head", public)

    def test_output_csv_schemas_are_stable(self) -> None:
        targets = _synthetic_targets()
        prediction = stage4.BoundaryPrediction(
            method_name="synthetic_method",
            predicted_boundary=targets.reference_boundary.copy(),
            predicted_found=targets.reference_found.copy(),
            no_boundary_reason=np.array([""] * targets.row_count),
        )
        self.assertEqual(
            stage4.BOUNDARY_METRICS_BY_OPTION_TYPE_FIELDNAMES,
            list(stage4.boundary_metrics_by_option_type_rows(targets, {"synthetic_method": prediction})[0]),
        )
        self.assertEqual(
            stage4.BOUNDARY_METRICS_BY_REGIME_FIELDNAMES,
            list(stage4.boundary_metrics_by_regime_rows(targets, {"synthetic_method": prediction})[0]),
        )
        self.assertEqual(
            stage4.BOUNDARY_CURVE_SAMPLE_AUDIT_FIELDNAMES,
            list(stage4.boundary_curve_sample_audit_rows(targets, {"synthetic_method": prediction})[0]),
        )

    def test_rmse_plot_metric_preserves_nan(self) -> None:
        rows = [
            {
                "method_name": "premium_implied_boundary",
                "split": "validation",
                "boundary_rmse": float("nan"),
            }
        ]
        self.assertTrue(
            np.isnan(
                stage4._metric_value(  # noqa: SLF001 - regression test for plotting behavior.
                    rows,
                    "premium_implied_boundary",
                    "validation",
                    "boundary_rmse",
                )
            )
        )

    def test_report_path_is_configurable_for_synthetic_runs(self) -> None:
        targets = _synthetic_targets()
        prediction = stage4.BoundaryPrediction(
            method_name="synthetic_method",
            predicted_boundary=targets.reference_boundary.copy(),
            predicted_found=targets.reference_found.copy(),
            no_boundary_reason=np.array([""] * targets.row_count),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = output_dir / "synthetic_boundary_report.tex"
            result = stage4.write_boundary_outputs(
                output_dir=output_dir,
                report_tex_path=report_path,
                targets=targets,
                predictions={"synthetic_method": prediction},
                create_figures=False,
            )
            self.assertEqual(report_path, result.report_tex_path)
            self.assertTrue(report_path.exists())
            for path in (
                result.metrics_by_split_path,
                result.metrics_by_option_type_path,
                result.metrics_by_regime_path,
                result.curve_sample_audit_path,
                result.model_manifest_path,
            ):
                self.assertTrue(path.exists())
                self.assertGreater(len(_read_csv(path)), 0)
            self.assertFalse(list(output_dir.rglob("*.pt")))

    def test_new_python_files_include_stage_4_docstring(self) -> None:
        expected = '"""Stage 4: boundary diagnostic and boundary-head comparison."""'
        for path in (
            Path("src/american_risk_surfaces/surrogates/boundary.py"),
            Path("experiments/16_boundary_diagnostics.py"),
            Path("tests/test_boundary_diagnostics.py"),
        ):
            self.assertTrue(path.exists(), path)
            self.assertIn(expected, path.read_text())


def _synthetic_targets(row_count: int = 4) -> stage4.BoundaryTargetTable:
    rng = np.random.default_rng(321)
    splits = np.resize(np.array(["train", "validation", "test", "stress_holdout"]), row_count)
    option_types = np.resize(np.array(["put", "call", "call", "put"]), row_count)
    reference_found = np.resize(np.array([True, True, False, True]), row_count)
    reference_boundary = np.where(reference_found, 0.8 + 0.1 * rng.normal(size=row_count), np.nan)
    reference_boundary = reference_boundary.astype(float)
    reference_boundary[0] = 0.8
    if row_count > 1:
        reference_boundary[1] = 1.1
    if row_count > 3:
        reference_boundary[3] = 0.8
    q = np.where(option_types == "call", 0.08, 0.02)
    if row_count > 2:
        q[2] = 0.0
    return stage4.BoundaryTargetTable(
        row_count=row_count,
        target_id=np.arange(row_count),
        regime_index=np.arange(row_count),
        regime_id=np.array([f"synthetic_{i:03d}" for i in range(row_count)]),
        split=splits,
        option_type=option_types,
        T=np.ones(row_count),
        sigma=np.full(row_count, 0.2),
        r=np.full(row_count, 0.05),
        q=q,
        tau=np.linspace(0.0, 1.0, row_count),
        tau_fraction=np.linspace(0.0, 1.0, row_count),
        is_call=(option_types == "call").astype(float),
        reference_boundary=reference_boundary,
        reference_found=reference_found,
        reference_in_sample_window=np.isfinite(reference_boundary)
        & (reference_boundary >= 0.4)
        & (reference_boundary <= 1.8),
        is_no_dividend_call_control=(option_types == "call") & (q == 0.0),
        curve_row_indices=tuple(np.array([i], dtype=int) for i in range(row_count)),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
