"""Positive-premium POD basis operator for American-option risk surfaces."""

from american_risk_surfaces.basis_operator.basis import (
    fit_full_grid_premium_basis,
    project_premium_coefficients,
)
from american_risk_surfaces.basis_operator.evaluation import audit_basis_operator_surface
from american_risk_surfaces.basis_operator.model import train_basis_coefficient_operator
from american_risk_surfaces.basis_operator.prediction import (
    make_basis_operator_policy_initializer,
    predict_basis_operator_surface,
    predict_no_dividend_call_control,
)
from american_risk_surfaces.basis_operator.types import (
    BasisOperatorArtifact,
    BasisOperatorPrediction,
    BasisOperatorTrainingConfig,
    PremiumPODBasis,
)

__all__ = (
    "PremiumPODBasis",
    "BasisOperatorTrainingConfig",
    "BasisOperatorArtifact",
    "BasisOperatorPrediction",
    "fit_full_grid_premium_basis",
    "project_premium_coefficients",
    "train_basis_coefficient_operator",
    "predict_basis_operator_surface",
    "predict_no_dividend_call_control",
    "make_basis_operator_policy_initializer",
    "audit_basis_operator_surface",
)
