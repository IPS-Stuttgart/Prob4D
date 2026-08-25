#!/usr/bin/env python3
"""Build the outcome-blind recurrent CUT3R source-comparison execution plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prob4d.cut3r_source_comparison_plan import (
    build_execution_plan,
    save_execution_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_execution_plan(
        repository=args.repository,
        preflight_path=args.preflight,
        cut3r_checkout=args.cut3r_checkout,
        checkpoint=args.checkpoint,
    )
    save_execution_plan(args.output, plan)
    print(
        json.dumps(
            {
                "case_count": len(plan["cases"]),
                "decision": plan["decision"],
                "plan_id": plan["plan_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
