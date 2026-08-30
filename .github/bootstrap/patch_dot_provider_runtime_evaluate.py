#!/usr/bin/env python3
"""Bind the DOT evaluation job to the same sealed CUT3R interpreter."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/dot-rope-cut3r-native-provider-v1.yml")


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pattern = re.compile(
        r"      - name: Select a retained NumPy interpreter\n.*?"
        r"(?=      - name: Open source markers only after prediction seal and evaluate\n)",
        re.DOTALL,
    )
    replacement = """      - name: Bind the sealed NumPy interpreter
        id: python
        shell: bash
        run: |
          set -euo pipefail
          test -x "$CUT3R_RUNTIME_PYTHON"
          "$CUT3R_RUNTIME_PYTHON" -c 'import numpy'
          echo "python=$CUT3R_RUNTIME_PYTHON" >> "$GITHUB_OUTPUT"
"""
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit("evaluation interpreter-selection step changed")
    WORKFLOW.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
