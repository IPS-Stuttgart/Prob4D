"""Run the held-out DLO query-observability evaluation on exact official data."""

from __future__ import annotations

from pathlib import Path

import audit_deform_dlo45_observability_v1 as base

base.EXPECTED_ROOT = Path("external/DEFORM/data_set")

import evaluate_deform_dlo45_query_observability_v1 as evaluation  # noqa: E402


if __name__ == "__main__":
    evaluation.main()
