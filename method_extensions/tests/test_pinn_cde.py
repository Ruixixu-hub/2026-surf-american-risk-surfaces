"""Tests for the SURF Arm C/D/E PINN implementation and leakage controls."""

from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.pinn import (
    NetworkSpec,
    PINNProblem,
    PINNTrainingConfig,
    discrete_lcp_audit,
    load_pinn_checkpoint,
    predict_pinn_surface,
    run_arm_e_hybrid,
    train_single_regime_pinn,
)
from american_risk_surfaces.pinn.formulation import (
    exact_terminal_lift,
    fischer_burmeister_numpy,
    payoff_numpy,
    smooth_fischer_burmeister,
    spatial_boundary_numpy,
    value_and_vi_residual,
)
from american_risk_surfaces.pinn.protocol import (
    HELDOUT_SPLITS,
    SEEDS,
    build_job_manifest,
    load_regime_records,
)
from american_risk_surfaces.pinn.sampling import PINNSampler
from american_risk_surfaces.pinn.reference import generate_reference_cache
from american_risk_surfaces.pinn.scoring import score_classical_baselines
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.lcp import compute_lcp_residual


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class PINNFormulationTests(unittest.TestCase):
    def test_payoffs_boundaries_and_scaling(self) -> None:
        put = PINNProblem("put", "put", 1.0, 0.05, 0.03, 0.2)
        call = PINNProblem("call", "call", 1.0, 0.05, 0.03, 0.2)
        x = np.log(np.array([0.5, 1.0, 2.0]))
        npt.assert_allclose(payoff_numpy(put, x), [0.5, 0.0, 0.0])
        npt.assert_allclose(payoff_numpy(call, x), [0.0, 0.0, 1.0])
        put_left, put_right = spatial_boundary_numpy(put, np.array([0.0, 1.0]))
        npt.assert_allclose(put_left, 1.0 - put.m_min)
        npt.assert_allclose(put_right, 0.0)
        call_left, call_right = spatial_boundary_numpy(call, np.array([0.0, 1.0]))
        npt.assert_allclose(call_left, 0.0)
        self.assertTrue(np.all(call_right >= call.m_max - 1.0))
        self.assertEqual(call.value_scale, 3.0)

    def test_exact_terminal_is_payoff_for_put_and_call(self) -> None:
        x = torch.linspace(np.log(1e-4), np.log(4.0), 101, dtype=torch.float64).reshape(-1, 1)
        zero = torch.zeros_like(x)
        for option in ("put", "call"):
            problem = PINNProblem(option, option, 1.0, 0.05, 0.03, 0.2)
            actual = exact_terminal_lift(problem, x, zero).detach().numpy().reshape(-1)
            npt.assert_allclose(actual, payoff_numpy(problem, x.numpy().reshape(-1)), atol=1e-12)

    def test_q_zero_put_matches_published_expansion(self) -> None:
        problem = PINNProblem("paper_put", "put", 1.0, 0.02, 0.0, 0.25)
        x = torch.tensor([[-0.2], [0.0], [0.3]], dtype=torch.float64)
        s = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float64)
        tau = s.numpy().reshape(-1)
        log_m = x.numpy().reshape(-1)
        d0 = -(log_m + problem.r * tau) / (problem.sigma * np.sqrt(tau))
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(d0 / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * d0**2) / np.sqrt(2.0 * np.pi)
        m = np.exp(log_m)
        expected = cdf * (np.exp(-problem.r * tau) - m) + (
            0.5 * problem.sigma * np.sqrt(tau) * pdf * (np.exp(-problem.r * tau) + m)
        )
        actual = exact_terminal_lift(problem, x, s).detach().numpy().reshape(-1)
        npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_autodiff_operator_has_frozen_forward_sign(self) -> None:
        problem = PINNProblem("analytic", "put", 2.0, 0.05, 0.01, 0.3)

        class AnalyticModel(torch.nn.Module):
            def forward(self, normalized: torch.Tensor) -> torch.Tensor:
                x = problem.x_min + 0.5 * (normalized[:, 0:1] + 1.0) * (
                    problem.x_max - problem.x_min
                )
                s = 0.5 * (normalized[:, 1:2] + 1.0)
                return x.square() + 3.0 * s + 4.0

        points = torch.tensor([[-0.3, 0.4], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)
        result = value_and_vi_residual(AnalyticModel(), problem, points, arm="C")
        x = points.detach().numpy()[:, 0]
        s = points.detach().numpy()[:, 1]
        u = x**2 + 3.0 * s + 4.0
        operator = 0.5 * problem.sigma**2 * 2.0 + (
            problem.r - problem.q - 0.5 * problem.sigma**2
        ) * 2.0 * x - problem.r * u
        expected = 3.0 - problem.T * operator
        npt.assert_allclose(result["equation_gap"].detach().numpy().reshape(-1), expected)

    def test_fischer_burmeister_and_origin_gradient(self) -> None:
        npt.assert_allclose(fischer_burmeister_numpy([1.0, 0.0], [0.0, 2.0]), 0.0)
        self.assertNotEqual(float(fischer_burmeister_numpy([-1.0], [0.0])[0]), 0.0)
        a = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
        loss = smooth_fischer_burmeister(a, b).square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())


@unittest.skipIf(torch is None, "PyTorch is required")
class PINNSamplingAndTrainingTests(unittest.TestCase):
    def test_sampler_is_deterministic_and_has_registered_mixture(self) -> None:
        problem = PINNProblem("sample", "put", 1.0, 0.05, 0.0, 0.2)
        first = PINNSampler(problem, seed=17, pool_size=128, candidate_size=64)
        second = PINNSampler(problem, seed=17, pool_size=128, candidate_size=64)
        batch_a = first.sample(100, 10, mode="adaptive", time_upper=0.25)
        batch_b = second.sample(100, 10, mode="adaptive", time_upper=0.25)
        npt.assert_allclose(batch_a.interior.numpy(), batch_b.interior.numpy())
        self.assertEqual(
            batch_a.component_counts,
            {"global": 40, "maturity": 20, "strike": 20, "adaptive": 20},
        )
        self.assertLessEqual(float(batch_a.interior[:, 1].max()), 0.25)

    def test_sampler_resume_accepts_mapped_rng_state(self) -> None:
        problem = PINNProblem("resume", "put", 1.0, 0.05, 0.0, 0.2)
        source = PINNSampler(problem, seed=17, pool_size=128, candidate_size=64)
        source.sample(40, 8, mode="adaptive")
        state = source.state_dict()
        # Simulate the tensor-device mapping performed by checkpoint loading.
        mapped = dict(state)
        mapped["torch_generator_state"] = state["torch_generator_state"].clone()
        restored = PINNSampler(problem, seed=17, pool_size=128, candidate_size=64)
        restored.load_state_dict(mapped)
        expected = source.sample(40, 8, mode="adaptive")
        actual = restored.sample(40, 8, mode="adaptive")
        npt.assert_allclose(actual.interior.numpy(), expected.interior.numpy())

    def test_tiny_training_checkpoint_resume_and_prediction(self) -> None:
        problem = PINNProblem("tiny", "put", 0.25, 0.01, 0.0, 0.2)
        config = PINNTrainingConfig(
            arm="D",
            seed=17,
            device="cpu",
            adam_steps=2,
            lbfgs_max_evaluations=0,
            interior_batch_size=32,
            boundary_batch_size=8,
            checkpoint_interval=1,
            gradient_log_interval=1,
            adaptive_interval=1,
            pool_size=64,
            candidate_size=32,
            max_seconds=30.0,
            network_spec=NetworkSpec(width=8, blocks=1, layers_per_block=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = train_single_regime_pinn(problem, config, output_dir=directory)
            self.assertEqual(first.status, "COMPLETE")
            resumed = train_single_regime_pinn(problem, config, output_dir=directory, resume=True)
            self.assertEqual(resumed.status, "COMPLETE")
            loaded = load_pinn_checkpoint(resumed.checkpoint_path)
            surface = predict_pinn_surface(
                loaded,
                np.linspace(0.0, 4.0, 13),
                np.linspace(0.0, 0.25, 9),
                K=1.0,
            )
            self.assertEqual(surface.value_grid.shape, (9, 13))
            npt.assert_allclose(surface.value_grid[:, 0], 1.0)
            self.assertGreaterEqual(surface.transfer_seconds, 0.0)
            heartbeat = next(Path(directory).glob("*_heartbeat.json"))
            self.assertIn('"phase": "COMPLETE"', heartbeat.read_text(encoding="utf-8"))

    def test_label_free_training_modules_do_not_open_reference_bundle(self) -> None:
        from american_risk_surfaces.pinn import study, training

        source = inspect.getsource(training) + inspect.getsource(study)
        self.assertNotIn("dataset_v1_small_grid.npz", source)
        self.assertNotIn("y_value", source)
        self.assertNotIn("np.load", source)


class PINNProtocolAndHybridTests(unittest.TestCase):
    def test_frozen_split_and_job_counts(self) -> None:
        records = load_regime_records()
        counts = {split: sum(record.split == split for record in records) for split in {r.split for r in records}}
        self.assertEqual(counts, {"train": 202, "validation": 19, "test": 43, "stress_holdout": 24})
        jobs = build_job_manifest(arms=("C", "D"), splits=HELDOUT_SPLITS, seeds=SEEDS)
        self.assertEqual(len(jobs), 670)
        shard_a = build_job_manifest(
            arms=("C", "D"), splits=HELDOUT_SPLITS, seeds=SEEDS, shard_index=0, shard_count=2
        )
        shard_b = build_job_manifest(
            arms=("C", "D"), splits=HELDOUT_SPLITS, seeds=SEEDS, shard_index=1, shard_count=2
        )
        self.assertEqual(len(shard_a) + len(shard_b), 670)
        self.assertFalse({(j["arm"], j["regime_id"], j["seed"]) for j in shard_a} & {(j["arm"], j["regime_id"], j["seed"]) for j in shard_b})

    def test_shared_step_assembly_matches_classical_solution_residual(self) -> None:
        config = AmericanLCPConfig("put", 1.0, 0.25, 0.01, 0.0, 0.2, 4.0, 24, 12, tolerance=1e-12)
        result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
        for step in range(1, config.N + 1):
            system = assemble_american_cn_lcp_step(config, result.value_grid[step - 1], step)
            residual = compute_lcp_residual(system, result.value_grid[step, 1:-1])
            self.assertLessEqual(residual.normalized_lcp_residual, 1e-12)

    def test_small_high_reference_and_both_classical_baselines_score(self) -> None:
        record = load_regime_records(splits=("validation",))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = generate_reference_cache(
                root / "reference",
                splits=("validation",),
                spatial_steps=24,
                time_steps=32,
                regime_ids=(record.regime_id,),
            )
            self.assertEqual(len(paths), 1)
            rows = score_classical_baselines(
                splits=("validation",),
                regime_ids=(record.regime_id,),
                reference_dir=root / "reference",
                output_dir=root / "scoring",
            )
            all_rows = [row for row in rows if row["region"] == "all"]
            self.assertEqual({row["arm"] for row in all_rows}, {"A", "B"})
            self.assertTrue(all(np.isfinite(row["rmse"]) for row in all_rows))

    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_arm_e_projects_and_finishes_to_same_strict_solution(self) -> None:
        problem = PINNProblem("hybrid", "put", 0.25, 0.01, 0.0, 0.2)
        train_config = PINNTrainingConfig(
            arm="D",
            seed=17,
            device="cpu",
            adam_steps=1,
            lbfgs_max_evaluations=0,
            interior_batch_size=16,
            boundary_batch_size=4,
            checkpoint_interval=1,
            gradient_log_interval=1,
            adaptive_interval=1,
            pool_size=32,
            candidate_size=16,
            max_seconds=30.0,
            network_spec=NetworkSpec(width=8, blocks=1, layers_per_block=2),
        )
        classical_config = AmericanLCPConfig(
            "put", 1.0, 0.25, 0.01, 0.0, 0.2, 4.0, 12, 8, tolerance=1e-12
        )
        with tempfile.TemporaryDirectory() as directory:
            trained = train_single_regime_pinn(problem, train_config, output_dir=directory)
            hybrid, _ = run_arm_e_hybrid(trained.checkpoint_path, classical_config)
            baseline = american_cn_lcp_price(classical_config, lcp_solver="policy_iteration")
            self.assertTrue(hybrid.converged)
            self.assertLessEqual(hybrid.max_obstacle_violation, 1e-12)
            npt.assert_allclose(hybrid.value_grid, baseline.value_grid, atol=1e-11)
            prediction = predict_pinn_surface(
                trained.checkpoint_path,
                baseline.spot_grid,
                baseline.tau_grid,
                K=1.0,
            )
            audit = discrete_lcp_audit(prediction, classical_config)
            self.assertIn("max_normalized_lcp_residual", audit)


if __name__ == "__main__":
    unittest.main()
