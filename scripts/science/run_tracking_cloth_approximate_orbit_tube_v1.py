#!/usr/bin/env python3
"""Prospective approximate-orbit tube calibration on Tracking Cloth self-collisions.

The 27 A2 cotton/denim/wool Self-collisions recordings were retained as a
header-only support-negative cohort. This study calibrates on 18 complete
recordings and opens nine deterministic target recordings only after a
content-addressed calibration seal exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prob4d._tracking_cloth_approximate_orbit_calibration import _calibrate
from prob4d._tracking_cloth_approximate_orbit_io import _load_protocol
from prob4d._tracking_cloth_approximate_orbit_target import _evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("calibrate", "evaluate"):
        child = subparsers.add_parser(name)
        child.add_argument("--dataset-root", required=True)
        child.add_argument("--protocol", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--source-revision", required=True)
        if name == "evaluate":
            child.add_argument("--calibration", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol = _load_protocol(Path(args.protocol).resolve())
    if args.mode == "calibrate":
        return _calibrate(args, protocol)
    return _evaluate(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
