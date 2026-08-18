"""One-shot import/type-alias diagnostic for the PR #259 bootstrap."""

from __future__ import annotations

from pathlib import Path


if __name__ == "__main__":
    source = Path("src/prob4d/material_identity_weight_calibration.py")
    lines = source.read_text(encoding="utf-8").splitlines()
    for line_number in range(1, min(100, len(lines)) + 1):
        print(f"SOURCE {line_number}: {lines[line_number - 1]}")
    print("Stopping after the focused import diagnostic; no materialization attempted.")
    raise SystemExit(1)
