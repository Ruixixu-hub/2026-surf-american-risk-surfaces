"""Stage 5: Delta diagnostic and supervised Delta-head comparison."""

from __future__ import annotations

from pathlib import Path

from american_risk_surfaces.surrogates.delta import (
    DEFAULT_OUTPUT_DIR,
    run_delta_diagnostics_experiment,
)


def main():
    result = run_delta_diagnostics_experiment(
        output_dir=DEFAULT_OUTPUT_DIR,
        epochs=10,
        batch_size=8192,
        create_figures=True,
    )
    print(f"Stage 5 review decision: {result.review_decision}")
    print(f"Delta outputs written to: {Path(result.output_dir)}")
    print(f"Delta report written to: {result.report_tex_path}")
    return result


if __name__ == "__main__":
    main()
