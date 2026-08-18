"""Freeze the published DIRK-P method and decision protocol."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.published_dirk_p_study import write_protocol


def main() -> None:
    print(json.dumps({key: str(value) for key, value in write_protocol().items()}, indent=2))


if __name__ == "__main__":
    main()
