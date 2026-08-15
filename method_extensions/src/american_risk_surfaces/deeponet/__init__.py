"""Parameter-conditioned positive-premium DeepONet for American risk surfaces."""

from american_risk_surfaces.deeponet.data import build_deeponet_training_bundle
from american_risk_surfaces.deeponet.evaluation import audit_deeponet_surface
from american_risk_surfaces.deeponet.model import (
    PositivePremiumDeepONet,
    infer_deeponet_numpy,
    load_deeponet_artifact,
    train_positive_premium_deeponet,
)
from american_risk_surfaces.deeponet.prediction import (
    make_deeponet_policy_initializer,
    predict_deeponet_surface,
    predict_q0_call_analytic_control,
)
from american_risk_surfaces.deeponet.types import (
    DeepONetArtifact,
    DeepONetPrediction,
    DeepONetTrainingBundle,
    DeepONetTrainingConfig,
    DeepONetTrainingResult,
)

__all__ = (
    "DeepONetTrainingConfig",
    "DeepONetTrainingBundle",
    "DeepONetArtifact",
    "DeepONetPrediction",
    "DeepONetTrainingResult",
    "PositivePremiumDeepONet",
    "infer_deeponet_numpy",
    "build_deeponet_training_bundle",
    "train_positive_premium_deeponet",
    "load_deeponet_artifact",
    "predict_deeponet_surface",
    "predict_q0_call_analytic_control",
    "make_deeponet_policy_initializer",
    "audit_deeponet_surface",
)
