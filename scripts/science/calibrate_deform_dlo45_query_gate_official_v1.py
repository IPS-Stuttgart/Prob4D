"""Run the DLO query-gate source calibration on an exact official checkout."""

from __future__ import annotations

from pathlib import Path

import audit_deform_dlo45_observability_v1 as base

base.EXPECTED_ROOT = Path("external/DEFORM/data_set")

import calibrate_deform_dlo45_query_gate_v1 as calibration  # noqa: E402


if __name__ == "__main__":
    calibration.main()
