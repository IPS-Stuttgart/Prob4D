#!/usr/bin/env python3
"""Validate retained CUT3R source-comparison cases and shard custody."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prob4d.cut3r_source_comparison_verifier import (
    validate_case_artifact,
    validate_shard_artifact,
    write_custody_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case_parser = subparsers.add_parser("case", help="validate one retained case")
    case_parser.add_argument("--case-root", type=Path, required=True)
    case_parser.add_argument("--expected-plan-id")
    case_parser.add_argument("--require-success", action="store_true")

    shard_parser = subparsers.add_parser(
        "shard",
        help="validate a shard report and every referenced case",
    )
    shard_parser.add_argument("--output-root", type=Path, required=True)
    shard_parser.add_argument("--report", type=Path, required=True)
    shard_parser.add_argument("--receipt", type=Path, required=True)
    shard_parser.add_argument("--expected-plan-id")
    shard_parser.add_argument(
        "--allow-technical-failures",
        action="store_true",
        help="retain a valid negative custody receipt instead of requiring all cases to succeed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "case":
        manifest = validate_case_artifact(
            args.case_root,
            expected_plan_id=args.expected_plan_id,
            require_success=args.require_success,
        )
        print(
            json.dumps(
                {
                    "artifact_id": manifest["artifact_id"],
                    "case_id": manifest["case_id"],
                    "plan_id": manifest["plan_id"],
                    "status": manifest["status"],
                },
                sort_keys=True,
            )
        )
        return 0

    receipt = validate_shard_artifact(
        args.output_root,
        args.report,
        expected_plan_id=args.expected_plan_id,
        require_success=not args.allow_technical_failures,
    )
    write_custody_receipt(args.receipt, receipt)
    print(
        json.dumps(
            {
                "case_count": receipt["case_count"],
                "decision": receipt["decision"],
                "receipt_id": receipt["receipt_id"],
                "retained_technical_failure_count": receipt[
                    "retained_technical_failure_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
