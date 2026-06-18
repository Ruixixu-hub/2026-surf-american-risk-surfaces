"""v0 Dry-Run Dataset: eight-regime surrogate dataset generation test."""

from __future__ import annotations

import os
from pathlib import Path

from american_risk_surfaces.data.dry_run_dataset import (
    DEFAULT_OUTPUT_DIR,
    DryRunDatasetPackage,
    DryRunRegime,
    generate_v0_dry_run_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")


def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    regimes: tuple[DryRunRegime, ...] | None = None,
    create_figures: bool = True,
) -> DryRunDatasetPackage:
    """Run the v0 dry-run dataset generation test."""

    package = generate_v0_dry_run_dataset(
        output_dir=Path(output_dir),
        regimes=regimes,
        create_figures=create_figures,
    )
    return package


if __name__ == "__main__":
    result = main()
    print(f"review_decision={result.review_decision}")
    print(f"accepted_row_count={result.accepted_row_count}")
    print(f"output_dir={result.output_dir}")
