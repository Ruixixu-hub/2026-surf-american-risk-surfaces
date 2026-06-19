"""Stage 3: price and positive-premium surrogate comparison."""

from __future__ import annotations

from pathlib import Path

from american_risk_surfaces.surrogates.price_premium import (
    DEFAULT_OUTPUT_DIR,
    TRAIN_ROW_CAP,
    run_surrogate_experiment,
)


def main() -> object:
    """Run the approved Stage 3 price/premium surrogate experiment."""

    result = run_surrogate_experiment(
        output_dir=DEFAULT_OUTPUT_DIR,
        train_cap=TRAIN_ROW_CAP,
        epochs=10,
        batch_size=8192,
        create_figures=True,
    )
    print(f"Stage 3 review decision: {result.review_decision}")
    print(f"Metrics written to: {Path(result.output_dir)}")
    print(f"Report written to: {result.report_tex_path}")
    return result


if __name__ == "__main__":
    main()
