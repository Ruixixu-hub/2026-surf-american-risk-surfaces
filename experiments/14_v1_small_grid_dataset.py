"""v1 Small-Grid Dataset: 288-regime surrogate dataset generation and QA."""

from __future__ import annotations

import os
from pathlib import Path

from american_risk_surfaces.data.small_grid_dataset import (
    DEFAULT_OUTPUT_DIR,
    SmallGridDatasetPackage,
    SmallGridRegime,
    generate_v1_small_grid_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")


def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    regimes: tuple[SmallGridRegime, ...] | None = None,
    create_figures: bool = True,
    include_higher_grid_confirmation: bool = True,
) -> SmallGridDatasetPackage:
    """Run the approved v1 small-grid dataset generation and QA pass."""

    return generate_v1_small_grid_dataset(
        output_dir=Path(output_dir),
        regimes=regimes,
        create_figures=create_figures,
        include_higher_grid_confirmation=include_higher_grid_confirmation,
    )


if __name__ == "__main__":
    result = main()
    print(f"review_decision={result.review_decision}")
    print(f"accepted_row_count={result.accepted_row_count}")
    print(f"total_regime_count={result.total_regime_count}")
    print(f"output_dir={result.output_dir}")
