"""Run the sliding DLO segment audit against an exact official checkout."""

from __future__ import annotations

from pathlib import Path

import audit_deform_dlo45_observability_v1 as base

base.EXPECTED_ROOT = Path("external/DEFORM/data_set")

import audit_deform_dlo45_local_segments_v2 as audit  # noqa: E402


if __name__ == "__main__":
    audit.main()
