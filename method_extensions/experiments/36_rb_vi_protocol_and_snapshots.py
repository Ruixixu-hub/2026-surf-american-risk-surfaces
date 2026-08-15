"""Experiment 36: freeze RB-VI protocol and generate train-only FOM snapshots."""

from __future__ import annotations

import argparse

from american_risk_surfaces.reduced_order.protocol import write_protocol
from american_risk_surfaces.reduced_order.study import generate_train_snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--option-type", choices=("put", "call"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    protocol = write_protocol()
    manifest = generate_train_snapshots(
        option_type=arguments.option_type,
        limit=arguments.limit,
        resume=not arguments.no_resume,
    )
    print(f"protocol={protocol}")
    print(f"snapshot_manifest={manifest}")


if __name__ == "__main__":
    main()
