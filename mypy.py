"""One-shot diagnostic for the PR #259 materialization bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    source = Path("src/prob4d/material_identity_weight_calibration.py")
    lines = source.read_text(encoding="utf-8").splitlines()
    for line_number in range(1075, min(1125, len(lines)) + 1):
        print(f"SOURCE {line_number}: {lines[line_number - 1]}")
    print("Stopping after the focused diagnostic; no branch materialization attempted.")
    raise SystemExit(1)
