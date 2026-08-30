#!/usr/bin/env python3
"""Build or verify CUT3R's native RoPE kernel and emit an immutable receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prob4d.cut3r_runtime_contract import require_compiled_cut3r_rope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run CUT3R's pinned curope setup.py before verification.",
    )
    parser.add_argument(
        "--expected-artifact-id",
        help="Fail unless the verified receipt matches this frozen artifact ID.",
    )
    return parser


def _write_no_clobber(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise


def main() -> int:
    args = _parser().parse_args()
    checkout = args.cut3r_checkout.expanduser().resolve(strict=True)
    curope_root = checkout / "src/croco/models/curope"

    if args.build:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=curope_root,
            check=True,
        )

    receipt = require_compiled_cut3r_rope(checkout)
    if (
        args.expected_artifact_id is not None
        and receipt["artifact_id"] != args.expected_artifact_id
    ):
        raise SystemExit("verified CUT3R runtime receipt differs from the frozen artifact ID")
    _write_no_clobber(args.output, receipt)
    print(receipt["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
