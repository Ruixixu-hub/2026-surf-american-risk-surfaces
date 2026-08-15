from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from american_risk_surfaces.basis_operator.basis import (
    load_premium_basis,
    project_premium_coefficients,
    reconstruct_premium_vector,
)
from american_risk_surfaces.basis_operator.heldout import run_q0_raw_ood_diagnostic
from american_risk_surfaces.basis_operator.model import (
    BasisCoefficientNetwork,
    infer_coefficients,
    infer_coefficients_numpy,
)
from american_risk_surfaces.basis_operator.prediction import (
    make_basis_operator_policy_initializer,
    predict_no_dividend_call_control,
    reconstruct_full_prediction,
)
from american_risk_surfaces.basis_operator.protocol import (
    RESULTS_DIR,
    assert_mapping_regime_allowed,
    assert_training_snapshot_allowed,
    train_snapshot_paths,
)
from american_risk_surfaces.basis_operator.types import BasisOperatorArtifact
from american_risk_surfaces.reduced_order.protocol import load_regimes
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig, american_cn_lcp_price
from american_risk_surfaces.solvers.black_scholes import european_call_price
from american_risk_surfaces.workspace import frozen_input


def _basis(family: str = "put", modes: int = 8):
    return load_premium_basis(
        frozen_input(
            Path("results/11_positive_premium_basis_operator/01_pod_basis")
            / family
            / f"premium_basis_{modes:02d}.npz"
        )
    )


def test_basis_is_train_only_family_separated_and_orthonormal() -> None:
    put = _basis("put")
    call = _basis("call")
    assert len(put.train_regime_ids) == 117
    assert len(call.train_regime_ids) == 85
    assert set(put.train_regime_ids).isdisjoint(call.train_regime_ids)
    assert np.allclose(put.components @ put.components.T, np.eye(8), atol=1e-10)
    path = train_snapshot_paths("put")[0]
    assert_training_snapshot_allowed(path, "put")
    with pytest.raises(PermissionError):
        assert_training_snapshot_allowed(path, "call")
    with pytest.raises(PermissionError):
        assert_mapping_regime_allowed("test", "put", 0.0)


def test_projection_reconstruction_and_exact_terminal_boundaries() -> None:
    basis = _basis("put")
    coefficients = np.linspace(-0.01, 0.01, 8)
    vector = reconstruct_premium_vector(basis, coefficients)
    recovered = project_premium_coefficients(basis, vector)
    assert np.allclose(recovered, coefficients, atol=1e-11)
    regime = load_regimes(splits=("validation",), option_type="put")[0]
    prediction = reconstruct_full_prediction(basis, coefficients, regime.config())
    assert np.all(prediction.projected_premium_grid >= 0.0)
    assert np.all(prediction.projected_premium_grid[0] == 0.0)
    assert np.all(prediction.value_grid[:, 0] == 1.0)
    assert np.all(prediction.value_grid[:, -1] == 0.0)


def test_hard_projection_allows_zero_and_softplus_is_strictly_positive_inside() -> None:
    basis = _basis("put")
    coefficients = np.full(8, -1e3)
    regime = load_regimes(splits=("validation",), option_type="put")[0]
    hard = reconstruct_full_prediction(basis, coefficients, regime.config(), projection="hard")
    soft = reconstruct_full_prediction(basis, coefficients, regime.config(), projection="softplus")
    assert np.any(hard.projected_premium_grid[1:, 1:-1] == 0.0)
    assert np.all(soft.projected_premium_grid[1:, 1:-1] > 0.0)


def test_numpy_and_torch_inference_match() -> None:
    torch = pytest.importorskip("torch")
    basis = _basis("put")
    torch.manual_seed(123)
    model = BasisCoefficientNetwork(4, 8).to(dtype=torch.float64)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    artifact = BasisOperatorArtifact(
        basis,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        np.asarray([0.0, 0.3, 0.05, 0.05]),
        np.asarray([1.0, 0.2, 0.02, 0.03]),
        np.linspace(0.5, 1.5, 8),
        {"option_type": "put", "modes": 8},
        {"protocol": "unit"},
    )
    features = np.asarray([[0.0, 0.4, 0.05, 0.03], [-0.2, 0.2, 0.04, 0.06]])
    assert np.allclose(
        infer_coefficients(artifact, features),
        infer_coefficients_numpy(artifact, features),
        atol=1e-12,
    )


def test_q0_call_analytic_control_is_pointwise_bsm() -> None:
    config = AmericanLCPConfig("call", 1.0, 0.5, 0.05, 0.0, 0.3, 4.0, 120, 120)
    prediction = predict_no_dividend_call_control(config)
    spots = np.linspace(0.0, 4.0, 121)
    for index in (0, 1, 60, 120):
        tau = config.T * index / 120
        expected = european_call_price(spots, 1.0, tau, 0.05, 0.0, 0.3)
        assert np.allclose(prediction.value_grid[index], expected, atol=1e-13)
    assert prediction.control_branch == "EUROPEAN_BSM_ANALYTIC_Q0_CALL"


def test_hybrid_initializer_reaches_same_strict_policy_solution() -> None:
    config = AmericanLCPConfig(
        "put", 1.0, 0.25, 0.05, 0.0, 0.2, 4.0, 120, 120,
        tolerance=1e-12, obstacle_tolerance=1e-12,
    )
    reference = american_cn_lcp_price(config, lcp_solver="policy_iteration")
    perturbation = 1e-3 * np.sin(np.pi * reference.spot_grid / config.Smax)[None, :]
    prediction = np.maximum(reference.value_grid + perturbation, reference.payoff[None, :])
    initializer = make_basis_operator_policy_initializer(prediction)
    hybrid = american_cn_lcp_price(
        config, lcp_solver="policy_iteration", initializer=initializer
    )
    assert hybrid.converged
    assert np.max(np.abs(hybrid.value_grid - reference.value_grid)) <= 1e-12
    assert max(item.residual.normalized_lcp_residual for item in hybrid.lcp_results) <= 1e-12


def test_q0_raw_ood_cannot_run_before_main_scoring_marker() -> None:
    marker = RESULTS_DIR / "06_heldout/SCORING_COMPLETE_DO_NOT_RETRAIN.json"
    if marker.exists():
        pytest.skip("real heldout marker exists; do not mutate it in a unit test")
    with pytest.raises(PermissionError):
        run_q0_raw_ood_diagnostic()
