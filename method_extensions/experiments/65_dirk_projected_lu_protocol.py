"""Experiment 65: freeze DIRK+sinh solver-substitution gates before candidate runs."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.method_extensions.dirk_projected_lu_study import (
    freeze_substitution_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    outputs = freeze_substitution_protocol(allow_existing=args.allow_existing)
    print(json.dumps(outputs["protocol_data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
