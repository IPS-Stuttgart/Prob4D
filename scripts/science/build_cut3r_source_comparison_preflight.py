#!/usr/bin/env python3
"""Build an outcome-blind retained CUT3R source-comparison preflight.

The command verifies the exact source inputs and frozen CUT3R provider surface
without decoding RGB frames, running inference, or touching target outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prob4d._atomic_file import atomic_write_bytes
from prob4d._cut3r_source_preflight_runtime import build_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    if args.output.is_symlink():
        raise ValueError("preflight output must not be a symbolic link")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        atomic_write_bytes(output, encoded, overwrite=False)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to replace different retained preflight bytes: {output}"
            ) from None
    print(
        json.dumps(
            {
                "artifact_id": report["artifact_id"],
                "decision": report["decision"],
                "resolved_case_count": report["resolved_case_count"],
                "resolved_group_count": report["resolved_group_count"],
                "failure_count": len(report["failures"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "source-comparison-preflight-ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
