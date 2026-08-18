from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from prob4d.provider_portfolio import (
    PROVIDER_PORTFOLIO_CLAIM_BOUNDARY,
    PROVIDER_PORTFOLIO_SCHEMA,
    PROVIDER_PORTFOLIO_VERSION,
    PROVIDER_STAGES,
    PROVIDER_STAGES_V1,
    build_provider_portfolio,
    load_provider_portfolio,
    main,
    provider_portfolio_summary,
    validate_provider_portfolio,
    write_provider_portfolio,
)


def _digest(character: str) -> str:
    return character * 64


def _gates(
    *,
    passed: int,
    in_progress: int | None = None,
    failed: int | None = None,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, stage in enumerate(PROVIDER_STAGES):
        if index < passed:
            result[stage] = {
                "decision": "passed",
                "evidence_id": _digest(chr(ord("a") + index)),
            }
        elif index == in_progress:
            result[stage] = {"decision": "in-progress", "evidence_id": None}
        elif index == failed:
            result[stage] = {
                "decision": "failed",
                "evidence_id": _digest("f"),
            }
        else:
            result[stage] = {"decision": "not-started", "evidence_id": None}
    return result


def _entry(
    provider_id: str,
    *,
    role: str,
    status: str,
    passed: int,
    in_progress: int | None = None,
    failed: int | None = None,
    point_authorized: bool = False,
) -> dict[str, object]:
    return {
        "provider_id": provider_id,
        "provider_family": provider_id.split("-")[0],
        "role": role,
        "status": status,
        "gates": _gates(
            passed=passed,
            in_progress=in_progress,
            failed=failed,
        ),
        "point_covariance_development_authorized": point_authorized,
        "metadata": {"owner": "provider-readiness"},
    }


def _spec() -> dict[str, object]:
    return {
        "entries": [
            _entry(
                "cut3r-recurrent-v1",
                role="primary",
                status="active",
                passed=3,
                in_progress=3,
            ),
            _entry(
                "motioncrafter-v2",
                role="alternative",
                status="active",
                passed=0,
                in_progress=0,
            ),
            _entry(
                "vggt-baseline-v1",
                role="parked",
                status="parked",
                passed=1,
            ),
        ],
        "metadata": {"split": "source-only", "target_outcomes_opened": False},
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _as_legacy_v1(portfolio: dict[str, object]) -> dict[str, object]:
    legacy = deepcopy(portfolio)
    legacy["schema_version"] = 1
    legacy["policy"]["ordered_stages"] = list(PROVIDER_STAGES_V1)
    for entry in legacy["entries"]:
        del entry["gates"]["linearization-closure"]
    unsigned = deepcopy(legacy)
    unsigned.pop("portfolio_id")
    legacy["portfolio_id"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return legacy


def test_provider_portfolio_is_deterministic_and_order_invariant() -> None:
    first = build_provider_portfolio(_spec())
    reversed_spec = _spec()
    reversed_spec["entries"] = list(reversed(reversed_spec["entries"]))
    second = build_provider_portfolio(reversed_spec)

    assert first == second
    assert first["schema"] == PROVIDER_PORTFOLIO_SCHEMA
    assert first["schema_version"] == PROVIDER_PORTFOLIO_VERSION == 2
    assert first["policy"]["ordered_stages"] == list(PROVIDER_STAGES)
    assert first["claim_boundary"] == PROVIDER_PORTFOLIO_CLAIM_BOUNDARY
    assert [entry["provider_id"] for entry in first["entries"]] == [
        "cut3r-recurrent-v1",
        "motioncrafter-v2",
        "vggt-baseline-v1",
    ]
    assert validate_provider_portfolio(first) == first

    summary = provider_portfolio_summary(first)
    assert summary["schema_version"] == 2
    assert summary["entry_count"] == 3
    assert summary["status_counts"]["active"] == 2
    assert summary["active"] == [
        {
            "provider_id": "cut3r-recurrent-v1",
            "role": "primary",
            "stage": "gauge-dependence",
        },
        {
            "provider_id": "motioncrafter-v2",
            "role": "alternative",
            "stage": "support",
        },
    ]


def test_provider_portfolio_v1_artifacts_remain_readable_without_implied_closure() -> None:
    current = build_provider_portfolio(
        {
            "entries": [
                _entry(
                    "legacy-provider-v1",
                    role="primary",
                    status="active",
                    passed=5,
                    in_progress=5,
                    point_authorized=True,
                )
            ],
            "metadata": {"target_outcomes_opened": False},
        }
    )
    legacy = _as_legacy_v1(current)

    validated = validate_provider_portfolio(legacy)
    assert validated == legacy
    assert validated["schema_version"] == 1
    assert "linearization-closure" not in validated["entries"][0]["gates"]
    summary = provider_portfolio_summary(validated)
    assert summary["schema_version"] == 1
    assert summary["active"][0]["stage"] == "conditional-covariance"


def test_v2_portfolio_requires_explicit_linearization_closure_gate() -> None:
    spec = _spec()
    del spec["entries"][0]["gates"]["linearization-closure"]
    with pytest.raises(ValueError, match="linearization-closure"):
        build_provider_portfolio(spec)


def test_provider_portfolio_enforces_active_work_in_progress_budget() -> None:
    spec = _spec()
    spec["entries"].append(
        _entry(
            "second-primary-v1",
            role="primary",
            status="active",
            passed=0,
            in_progress=0,
        )
    )
    with pytest.raises(ValueError, match="active-primary budget"):
        build_provider_portfolio(spec)

    without_primary = _spec()
    without_primary["entries"] = [without_primary["entries"][1]]
    with pytest.raises(ValueError, match="alternative requires"):
        build_provider_portfolio(without_primary)


def test_provider_portfolio_enforces_ordered_scientific_gates() -> None:
    spec = _spec()
    entry = spec["entries"][0]
    entry["gates"]["identity"] = {
        "decision": "not-started",
        "evidence_id": None,
    }
    with pytest.raises(ValueError, match="pass every earlier gate"):
        build_provider_portfolio(spec)

    rejected = {
        "entries": [
            _entry(
                "negative-provider-v1",
                role="primary",
                status="rejected",
                passed=1,
                failed=1,
            )
        ],
        "metadata": {},
    }
    artifact = build_provider_portfolio(rejected)
    assert artifact["entries"][0]["gates"]["means"]["decision"] == "failed"


def test_point_covariance_work_requires_localization_authorization() -> None:
    unauthorized = {
        "entries": [
            _entry(
                "provider-v1",
                role="primary",
                status="active",
                passed=5,
                in_progress=5,
                point_authorized=False,
            )
        ],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="localization authorization"):
        build_provider_portfolio(unauthorized)

    authorized = deepcopy(unauthorized)
    authorized["entries"][0]["point_covariance_development_authorized"] = True
    artifact = build_provider_portfolio(authorized)
    assert artifact["entries"][0]["status"] == "active"


def test_point_covariance_authorization_requires_linearization_closure() -> None:
    premature = {
        "entries": [
            _entry(
                "provider-v1",
                role="primary",
                status="active",
                passed=4,
                in_progress=4,
                point_authorized=True,
            )
        ],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="linearization-closure"):
        build_provider_portfolio(premature)


def test_portfolio_rejects_coercive_types_and_tampering() -> None:
    coercive = _spec()
    coercive["entries"][0]["point_covariance_development_authorized"] = 1
    with pytest.raises(ValueError, match="must be a Boolean"):
        build_provider_portfolio(coercive)

    artifact = build_provider_portfolio(_spec())
    tampered = deepcopy(artifact)
    tampered["entries"][0]["provider_family"] = "changed"
    with pytest.raises(ValueError, match="portfolio_id"):
        validate_provider_portfolio(tampered)


def test_portfolio_round_trip_no_clobber_and_cli(tmp_path: Path, capsys) -> None:
    artifact = build_provider_portfolio(_spec())
    output = tmp_path / "portfolio.json"
    assert write_provider_portfolio(output, artifact) == artifact
    assert write_provider_portfolio(output, artifact) == artifact
    assert load_provider_portfolio(output) == artifact

    different = build_provider_portfolio(
        {
            "entries": [
                _entry(
                    "other-v1",
                    role="primary",
                    status="active",
                    passed=0,
                    in_progress=0,
                )
            ],
            "metadata": {},
        }
    )
    with pytest.raises(FileExistsError, match="different provider portfolio"):
        write_provider_portfolio(output, different)

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    cli_output = tmp_path / "cli-portfolio.json"
    assert main(["build", str(spec_path), "--output", str(cli_output)]) == 0
    built_id = capsys.readouterr().out.strip()
    assert built_id == load_provider_portfolio(cli_output)["portfolio_id"]

    assert main(["verify", str(cli_output)]) == 0
    assert capsys.readouterr().out.strip() == built_id

    assert main(["summarize", str(cli_output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["portfolio_id"] == built_id
    assert summary["schema_version"] == 2


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_provider_portfolio(path)
