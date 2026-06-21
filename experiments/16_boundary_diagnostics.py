"""Stage 4: boundary diagnostic and boundary-head comparison."""

from __future__ import annotations

from pathlib import Path

from american_risk_surfaces.surrogates.boundary import (
    DEFAULT_OUTPUT_DIR,
    run_boundary_diagnostics_experiment,
)


def main() -> object:
    """Run the approved Stage 4 boundary diagnostic experiment."""

    result = run_boundary_diagnostics_experiment(
        output_dir=DEFAULT_OUTPUT_DIR,
        epochs=10,
        batch_size=8192,
        create_figures=True,
    )
    print(f"Stage 4 review decision: {result.review_decision}")
    print(f"Boundary outputs written to: {Path(result.output_dir)}")
    print(f"Boundary report written to: {result.report_tex_path}")
    return result


if __name__ == "__main__":
    main()
