from __future__ import annotations

import json
from pathlib import Path

from prob4d.information_contract_sealed import (
    evaluate_sealed_information_contract,
    generate_sealed_smoke,
)


def _rewrite_case_ids(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        case["case_id"] = f"dataset/sequence/{case['case_id']}"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_hierarchical_case_ids_resolve_joined_payloads(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")
    _rewrite_case_ids(challenge)
    _rewrite_case_ids(submission)

    result = evaluate_sealed_information_contract(challenge, submission)

    assert result["aggregate"]["contract"]["all_cases_pass"] is True
    assert all(
        case["case_id"].startswith("dataset/sequence/")
        for case in result["cases"]
    )
