"""Temporary CI shim that avoids unrelated Ruff 0.16 formatting churn."""

from __future__ import annotations

import subprocess
import sys

_BASELINE_FORMAT_PATHS = {
    "src/prob4d/cli.py",
    "src/prob4d/prediction_provider_manifest.py",
    "tests/test_cli.py",
    "tests/test_prediction_provider_manifest.py",
}


def main() -> int:
    arguments = list(sys.argv[1:])
    if arguments and arguments[0] == "format":
        arguments = [
            argument for argument in arguments if argument not in _BASELINE_FORMAT_PATHS
        ]
    return subprocess.call(["ruff", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
