"""Run the frozen twelve-regime published DIRK-P formal audit."""

from __future__ import annotations

import argparse
import json

from american_risk_surfaces.method_extensions.published_dirk_p_study import run_formal_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-limit", type=int)
    args = parser.parse_args()
    outputs = run_formal_audit(regime_limit=args.regime_limit)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
