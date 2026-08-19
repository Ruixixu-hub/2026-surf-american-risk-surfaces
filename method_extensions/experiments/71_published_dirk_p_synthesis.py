"""Synthesize the published DIRK-P decision and reports."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.published_dirk_p_study import synthesize


def main() -> None:
    print(json.dumps({key: str(value) for key, value in synthesize().items()}, indent=2))


if __name__ == "__main__":
    main()
