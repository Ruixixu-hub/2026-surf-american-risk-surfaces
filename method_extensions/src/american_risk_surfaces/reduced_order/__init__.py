"""Primal/dual reduced-basis variational inequalities for American options."""

from american_risk_surfaces.reduced_order.basis import (
    angle_greedy,
    build_primal_dual_basis,
    build_primal_dual_basis_ladder,
    g_orthonormalize,
    load_basis,
    pod_greedy,
    reduced_inf_sup_constant,
    save_basis,
    weighted_h1_gram,
)
from american_risk_surfaces.reduced_order.protocol import RBRegime, load_regimes
from american_risk_surfaces.reduced_order.snapshots import (
    boundary_lift_grid,
    generate_fom_snapshot,
    load_snapshot,
    trajectory_multipliers,
)
from american_risk_surfaces.reduced_order.solver import (
    ReducedPDASResult,
    assemble_affine_rb_operator,
    assemble_affine_reduced_step,
    audit_rb_trajectory,
    direct_reduced_step,
    solve_reduced_american_vi,
    solve_reduced_mixed_lcp,
)
from american_risk_surfaces.reduced_order.types import (
    AffineRBOperator,
    PrimalDualRBBasis,
    RBBasisArtifact,
    RBFOMSnapshot,
    RBVISolveResult,
)

__all__ = (
    "RBRegime",
    "RBFOMSnapshot",
    "PrimalDualRBBasis",
    "RBBasisArtifact",
    "RBVISolveResult",
    "AffineRBOperator",
    "load_regimes",
    "boundary_lift_grid",
    "generate_fom_snapshot",
    "load_snapshot",
    "trajectory_multipliers",
    "weighted_h1_gram",
    "g_orthonormalize",
    "pod_greedy",
    "angle_greedy",
    "reduced_inf_sup_constant",
    "build_primal_dual_basis",
    "build_primal_dual_basis_ladder",
    "save_basis",
    "load_basis",
    "ReducedPDASResult",
    "assemble_affine_rb_operator",
    "assemble_affine_reduced_step",
    "solve_reduced_mixed_lcp",
    "solve_reduced_american_vi",
    "audit_rb_trajectory",
    "direct_reduced_step",
)
