"""Correct DIRK-P VI scoring to the project-wide frozen 1e-12 gate."""

from __future__ import annotations

import json

from american_risk_surfaces.method_extensions.published_dirk_p_study import (
    rescore_existing_audit_with_frozen_vi_gate,
    synthesize,
    write_protocol,
)


def main() -> None:
    protocol = write_protocol()
    correction = rescore_existing_audit_with_frozen_vi_gate()
    reports = synthesize()
    print(
        json.dumps(
            {
                "protocol": {key: str(value) for key, value in protocol.items()},
                "correction": str(correction),
                "reports": {key: str(value) for key, value in reports.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
