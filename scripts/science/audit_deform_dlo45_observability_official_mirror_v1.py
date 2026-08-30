"""Run the DLO4/DLO5 source audit against an exact official checkout.

The scientific implementation is shared with the local-mirror audit. This thin
entry point only binds the separately checked-out public repository path before
any dataset file is opened.
"""

from __future__ import annotations

from pathlib import Path

import audit_deform_dlo45_observability_v1 as audit

OFFICIAL_CHECKOUT_DATASET_ROOT = Path("external/DEFORM/data_set")


def main() -> None:
    audit.EXPECTED_ROOT = OFFICIAL_CHECKOUT_DATASET_ROOT
    audit.main()


if __name__ == "__main__":
    main()
