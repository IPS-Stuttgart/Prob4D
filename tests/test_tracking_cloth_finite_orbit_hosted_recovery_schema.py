from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "protocols/tracking-cloth-finite-orbit-hosted-recovery-v1.schema.json"


def test_hosted_recovery_schema_freezes_the_operational_boundary() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert schema["additionalProperties"] is False
    assert properties["superseded_run_id"]["const"] == 33361712662
    assert properties["opens_collision_outcomes"]["const"] is True
    assert properties["target_side_retuning_allowed"]["const"] is False
    assert properties["raw_data_publication_authorized"]["const"] is False
    assert properties["zenodo_record"]["const"] == "14644526"
    assert properties["zenodo_md5"]["const"] == "b4868b702f8a42b2ea1069d0f1a3b8f6"
