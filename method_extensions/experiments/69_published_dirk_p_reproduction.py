"""Reproduce the paper's representative one-dimensional American put case."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.published_dirk_p_study import run_reproduction


def main() -> None:
    print(json.dumps({key: str(value) for key, value in run_reproduction().items()}, indent=2))


if __name__ == "__main__":
    main()
