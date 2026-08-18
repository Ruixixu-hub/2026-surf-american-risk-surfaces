"""SURF physics-informed American-option solver experiments."""

from american_risk_surfaces.pinn.evaluation import (
    HybridTiming,
    LoadedPINN,
    PINNSurfacePrediction,
    discrete_lcp_audit,
    evaluate_pinn_vi,
    load_pinn_checkpoint,
    make_pinn_policy_initializer,
    predict_pinn_surface,
    run_arm_e_hybrid,
)
from american_risk_surfaces.pinn.formulation import PINNProblem
from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.training import (
    PINNRunResult,
    PINNTrainingConfig,
    train_single_regime_pinn,
)

__all__ = (
    "HybridTiming",
    "LoadedPINN",
    "NetworkSpec",
    "PINNProblem",
    "PINNRunResult",
    "PINNSurfacePrediction",
    "PINNTrainingConfig",
    "discrete_lcp_audit",
    "evaluate_pinn_vi",
    "load_pinn_checkpoint",
    "make_pinn_policy_initializer",
    "predict_pinn_surface",
    "run_arm_e_hybrid",
    "train_single_regime_pinn",
)
