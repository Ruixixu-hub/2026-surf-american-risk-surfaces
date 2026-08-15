"""Oracle boundary-aligned and localized basis falsification tools."""

from american_risk_surfaces.boundary_aligned_basis.alignment import (
    align_dual_multiplier,
    align_primal_state,
    build_boundary_alignment_map,
    extract_oracle_boundary_path,
    inverse_align_dual_multiplier,
    inverse_align_primal_state,
    pairing_relative_error,
    sanitize_snapshot_multiplier,
)
from american_risk_surfaces.boundary_aligned_basis.types import (
    BoundaryAlignmentConfig,
    BoundaryAlignmentMap,
    OracleBasisArtifact,
    OracleFalsificationResult,
)
from american_risk_surfaces.boundary_aligned_basis.basis import (
    angle_greedy_rows,
    build_oracle_basis_ladder,
    load_oracle_basis,
    pod_greedy_rows,
    save_oracle_basis,
)
from american_risk_surfaces.boundary_aligned_basis.audit import (
    audit_alignment_resolution,
    audit_boundary_threshold_sensitivity,
)
from american_risk_surfaces.boundary_aligned_basis.evaluation import evaluate_oracle_basis

__all__ = (
    "BoundaryAlignmentConfig",
    "BoundaryAlignmentMap",
    "OracleBasisArtifact",
    "OracleFalsificationResult",
    "extract_oracle_boundary_path",
    "build_boundary_alignment_map",
    "align_primal_state",
    "align_dual_multiplier",
    "inverse_align_primal_state",
    "inverse_align_dual_multiplier",
    "pairing_relative_error",
    "sanitize_snapshot_multiplier",
    "build_oracle_basis_ladder",
    "pod_greedy_rows",
    "angle_greedy_rows",
    "save_oracle_basis",
    "load_oracle_basis",
    "audit_alignment_resolution",
    "audit_boundary_threshold_sensitivity",
    "evaluate_oracle_basis",
)
