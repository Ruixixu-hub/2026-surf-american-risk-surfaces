from __future__ import annotations

import ast
from dataclasses import asdict

import numpy as np
import pytest

from american_risk_surfaces.deeponet.data import (
    _balanced_class_weights,
    _boundary_mask,
    build_deeponet_training_bundle,
    cartesian_coordinate_grid,
)
from american_risk_surfaces.deeponet.evaluation import audit_deeponet_surface
from american_risk_surfaces.deeponet.heldout import run_q0_raw_ood_diagnostic
from american_risk_surfaces.deeponet.model import (
    PositivePremiumDeepONet,
    _assert_checkpoint_compatible,
    count_parameters,
    infer_deeponet_numpy,
)
from american_risk_surfaces.deeponet.physics import (
    batched_cn_lcp_residual,
    smooth_fischer_burmeister,
)
from american_risk_surfaces.deeponet.prediction import (
    make_deeponet_policy_initializer,
    predict_deeponet_surface,
    predict_q0_call_analytic_control,
)
from american_risk_surfaces.deeponet.protocol import (
    RESULTS_DIR,
    assert_training_snapshot_allowed,
    train_snapshot_paths,
)
from american_risk_surfaces.deeponet.types import DeepONetArtifact
from american_risk_surfaces.reduced_order.protocol import load_regimes
from american_risk_surfaces.reduced_order.snapshots import load_snapshot
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.black_scholes import european_call_price
from american_risk_surfaces.solvers.lcp import compute_lcp_residual, tridiagonal_matvec


torch = pytest.importorskip("torch")


def test_branch_trunk_shape_and_cartesian_contraction_match_pointwise() -> None:
    torch.manual_seed(11)
    model = PositivePremiumDeepONet(32).to(dtype=torch.float64)
    branch_input = torch.randn((3, 4), dtype=torch.float64)
    trunk_input = torch.randn((17, 2), dtype=torch.float64)
    branch = model.encode_branch(branch_input)
    trunk = model.encode_trunk(trunk_input)
    cartesian = model(branch_input, trunk_input)
    pointwise = torch.stack([
        torch.sum(branch[row] * trunk[column]) / np.sqrt(32.0) + model.bias
        for row in range(3) for column in range(17)
    ]).reshape(3, 17)
    assert branch.shape == (3, 32)
    assert trunk.shape == (17, 32)
    assert torch.allclose(cartesian, pointwise, atol=1e-12)
    assert count_parameters(model) == sum(item.numel() for item in model.parameters())


def test_exported_numpy_and_pytorch_inference_match() -> None:
    torch.manual_seed(19)
    model = PositivePremiumDeepONet(32).to(dtype=torch.float64)
    features = np.asarray([[0.2, -0.4, 0.3, 0.1], [-0.1, 0.7, -0.2, 0.5]])
    coordinates = np.asarray([[-0.7, -0.5], [0.0, 0.3], [0.9, 1.0]])
    artifact = DeepONetArtifact(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        np.zeros(4), np.ones(4),
        {"option_type": "put", "latent_rank": 32}, {"protocol": "unit"},
    )
    expected = model(
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(coordinates, dtype=torch.float64),
    ).detach().numpy()
    assert np.allclose(
        infer_deeponet_numpy(artifact, features, coordinates), expected, atol=2e-13
    )


def test_resume_rejects_config_hash_and_terminal_status_mismatch() -> None:
    bundle = build_deeponet_training_bundle(train_snapshot_paths("put"), "put")
    from american_risk_surfaces.deeponet.types import DeepONetTrainingConfig

    config = DeepONetTrainingConfig("put", "N0", 32, 17, 10)
    payload = {"status": "COMPLETE", "hashes": bundle.hashes, "config": asdict(config)}
    _assert_checkpoint_compatible(payload, bundle, config)
    changed = DeepONetTrainingConfig("put", "N0", 64, 17, 10)
    with pytest.raises(RuntimeError, match="config mismatch"):
        _assert_checkpoint_compatible(payload, bundle, changed)
    payload["status"] = "BUDGET_EXHAUSTED"
    with pytest.raises(RuntimeError, match="may not be resumed"):
        _assert_checkpoint_compatible(payload, bundle, config)


def test_prepared_model_must_match_artifact_rank() -> None:
    artifact = DeepONetArtifact(
        PositivePremiumDeepONet(32).state_dict(), np.zeros(4), np.ones(4),
        {"option_type": "put", "latent_rank": 32}, {"protocol": "unit"},
    )
    regime = load_regimes(splits=("train",), option_type="put")[0]
    with pytest.raises(ValueError, match="rank"):
        predict_deeponet_surface(
            artifact, regime.config(), _prepared_model=PositivePremiumDeepONet(64)
        )


def test_coordinate_order_and_train_only_family_bundle() -> None:
    coordinates = cartesian_coordinate_grid()
    assert coordinates.shape == (14280, 2)
    assert np.allclose(coordinates[0], [-0.9833333333333333, -0.9833333333333333])
    assert np.allclose(coordinates[-1], [0.9833333333333334, 1.0])
    put_paths = train_snapshot_paths("put")
    bundle = build_deeponet_training_bundle(put_paths, "put")
    assert len(bundle.regime_ids) == 117
    assert bundle.premium_surfaces.shape == (117, 120, 119)
    assert bundle.boundary_mask.shape == bundle.premium_surfaces.shape
    assert np.isclose(np.mean(np.where(
        bundle.continuation_mask, bundle.class_weights[1], bundle.class_weights[0]
    )), 1.0)
    assert_training_snapshot_allowed(put_paths[0], "put")
    with pytest.raises(PermissionError):
        assert_training_snapshot_allowed(put_paths[0], "call")


def test_boundary_mask_and_class_weights_are_deterministic() -> None:
    continuation = np.zeros((120, 119), dtype=bool)
    continuation[:, 40:] = True
    first = _boundary_mask(continuation)
    second = _boundary_mask(continuation.copy())
    assert np.array_equal(first, second)
    assert np.all(first[:, 38:42])
    labels = np.stack((continuation, ~continuation))
    assert np.array_equal(_balanced_class_weights(labels), _balanced_class_weights(labels))


def test_smooth_fb_zero_on_complementarity_and_finite_at_origin() -> None:
    a = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    fb = smooth_fischer_burmeister(a, b)
    assert torch.max(torch.abs(fb)) <= 1e-12
    fb.sum().backward()
    assert torch.all(torch.isfinite(a.grad))
    assert torch.all(torch.isfinite(b.grad))
    violated = smooth_fischer_burmeister(
        torch.tensor([-1.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    assert torch.abs(violated).item() > 0.1


@pytest.mark.parametrize("family", ["put", "call"])
def test_torch_lcp_matches_public_numpy_assembly(family: str) -> None:
    candidates = [
        item for item in load_regimes(splits=("train",), option_type=family)
        if family == "put" or item.q > 0.0
    ]
    regime = candidates[0]
    snapshot = load_snapshot(train_snapshot_paths(family)[0])
    values = torch.as_tensor(snapshot.value_grid[None, :, :], dtype=torch.float64)
    payoff = torch.as_tensor(snapshot.payoff, dtype=torch.float64)
    residual = batched_cn_lcp_residual(
        values, payoff,
        torch.tensor([regime.T], dtype=torch.float64),
        torch.tensor([regime.sigma], dtype=torch.float64),
        torch.tensor([regime.r], dtype=torch.float64),
        torch.tensor([regime.q], dtype=torch.float64),
    )
    for step in (1, 60, 120):
        system = assemble_american_cn_lcp_step(regime.config(), snapshot.value_grid[step - 1], step)
        interior = snapshot.value_grid[step, 1:-1]
        expected_equation = tridiagonal_matvec(system, interior) - system.rhs
        expected_gap = interior - system.obstacle
        assert np.allclose(residual.equation_gap[0, step - 1].detach(), expected_equation, atol=2e-13)
        assert np.allclose(residual.obstacle_gap[0, step - 1].detach(), expected_gap, atol=2e-13)
        assert compute_lcp_residual(system, interior).normalized_lcp_residual <= 1e-12


def test_q0_analytic_control_is_pointwise_bsm_and_exact_terminal() -> None:
    config = AmericanLCPConfig("call", 1.0, 0.5, 0.05, 0.0, 0.3, 4.0, 120, 120)
    prediction = predict_q0_call_analytic_control(config)
    spots = np.linspace(0.0, 4.0, 121)
    assert np.array_equal(prediction.value_grid[0], np.maximum(spots - 1.0, 0.0))
    for index in (0, 1, 60, 120):
        tau = config.T * index / 120
        expected = european_call_price(spots, 1.0, tau, 0.05, 0.0, 0.3)
        assert np.allclose(prediction.value_grid[index], expected, atol=1e-13)
    assert np.all(prediction.projected_premium_grid >= 0.0)


def test_hard_positive_initializer_reaches_same_strict_policy_solution() -> None:
    config = AmericanLCPConfig(
        "put", 1.0, 0.25, 0.05, 0.0, 0.2, 4.0, 120, 120,
        tolerance=1e-12, obstacle_tolerance=1e-12,
    )
    reference = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    perturbation = 1e-3 * np.sin(np.pi * reference.spot_grid / 4.0)[None, :]
    prediction = np.maximum(reference.value_grid + perturbation, reference.payoff[None, :])
    initializer = make_deeponet_policy_initializer(prediction)
    hybrid = american_cn_lcp_price(config, lcp_solver="policy_iteration", initializer=initializer)
    assert hybrid.converged
    assert np.max(np.abs(hybrid.value_grid - reference.value_grid)) <= 1e-12


def test_heldout_module_does_not_import_reference_bundle() -> None:
    path = (
        __import__("pathlib").Path(__file__).parents[1]
        / "src/american_risk_surfaces/deeponet/heldout.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "validation_reference_bundle" not in imported


def test_q0_raw_ood_cannot_run_before_scoring_marker() -> None:
    marker = RESULTS_DIR / "04_heldout/SCORING_COMPLETE_DO_NOT_RETRAIN.json"
    if marker.exists():
        pytest.skip("real scoring marker exists; do not mutate it")
    with pytest.raises(PermissionError):
        run_q0_raw_ood_diagnostic(device="cpu")
