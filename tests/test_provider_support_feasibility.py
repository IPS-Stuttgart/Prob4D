from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prob4d.provider_support_feasibility import (
    PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY,
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
    load_provider_support_feasibility,
    load_provider_support_feasibility_request,
    main,
    write_provider_support_feasibility,
    write_provider_support_feasibility_request,
)

SOURCE_REVISION = "a" * 40
PROVIDER_REVISION = "b" * 40
DIGESTS = {
    "model": "1" * 64,
    "loader": "2" * 64,
    "cohort": "3" * 64,
    "lock": "4" * 64,
    "intrinsics": "5" * 64,
    "extrinsics": "6" * 64,
    "anchor": "7" * 64,
}


def _stream(
    group_id: str = "group-0",
    stream_id: str = "camera-0",
    **changes: object,
) -> ProviderSupportStreamV1:
    values: dict[str, object] = {
        "group_id": group_id,
        "stream_id": stream_id,
        "causal_frame_start": 0,
        "causal_frame_stop_exclusive": 4,
        "required_frame_ids": (0, 1, 2, 3),
        "available_frame_ids": (0, 1, 2, 3),
        "geometry_supported_frame_ids": (0, 1, 2, 3),
        "minimum_geometry_support_fraction": 1.0,
        "intrinsics_required": True,
        "intrinsics_id": DIGESTS["intrinsics"],
        "extrinsics_required": True,
        "extrinsics_id": DIGESTS["extrinsics"],
        "metric_anchor_required": True,
        "metric_anchor_id": DIGESTS["anchor"],
        "technical_failure_code": None,
        "metadata": {"source": "released-camera-metadata"},
    }
    values.update(changes)
    return ProviderSupportStreamV1(**values)  # type: ignore[arg-type]


def _request(
    streams: tuple[ProviderSupportStreamV1, ...] | None = None,
    **changes: object,
) -> ProviderSupportFeasibilityRequestV1:
    values: dict[str, object] = {
        "protocol_id": "provider-support-feasibility-test-v1",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": SOURCE_REVISION,
        "provider_family": "external-4d-provider",
        "provider_repository": "example/provider",
        "provider_revision": PROVIDER_REVISION,
        "model_set_id": DIGESTS["model"],
        "loader_id": DIGESTS["loader"],
        "cohort_binding_id": DIGESTS["cohort"],
        "promotion_lock_id": DIGESTS["lock"],
        "coordinate_semantics": "metric-world-frame",
        "admission_rule": "all-streams",
        "minimum_supported_fraction": 1.0,
        "permitted_technical_exclusion_codes": (),
        "maximum_technical_exclusions": 0,
        "prediction_payloads_opened": False,
        "residuals_used": False,
        "target_outcomes_used": False,
        "streams": streams or (_stream(),),
        "metadata": {"phase": "pre-residual"},
    }
    values.update(changes)
    return ProviderSupportFeasibilityRequestV1(**values)  # type: ignore[arg-type]


def test_all_streams_support_passes_and_is_content_addressed() -> None:
    request = _request(
        (
            _stream("group-1", "camera-1"),
            _stream("group-0", "camera-0"),
        )
    )
    result = evaluate_provider_support_feasibility(request)

    assert request.streams[0].key == ("group-0", "camera-0")
    assert result.support_feasible
    assert result.decision_reason == "support-feasible"
    assert result.support_fraction == 1.0
    assert result.supported_stream_count == 2
    assert result.excluded_stream_count == 0
    assert len(request.request_id) == 64
    assert len(result.provider_support_feasibility_id) == 64
    assert (
        result.to_dict()["claim_boundary"]
        == PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY
    )


def test_missing_support_fails_with_specific_reasons() -> None:
    stream = _stream(
        available_frame_ids=(0, 1, 2),
        geometry_supported_frame_ids=(0, 1),
        intrinsics_id=None,
        metric_anchor_id=None,
    )
    result = evaluate_provider_support_feasibility(_request((stream,)))

    assert not result.support_feasible
    assert result.decision_reason == "support-threshold-not-met"
    assert result.support_fraction == 0.0
    assert result.stream_results[0].reason_codes == (
        "insufficient-geometry-support",
        "missing-intrinsics",
        "missing-metric-anchor",
        "missing-required-frames",
    )


def test_causal_prefix_rejects_future_or_unavailable_geometry_frames() -> None:
    with pytest.raises(ValueError, match="crosses the frozen causal prefix"):
        _stream(available_frame_ids=(0, 1, 2, 4))
    with pytest.raises(ValueError, match="subset of available_frame_ids"):
        _stream(
            available_frame_ids=(0, 1, 2),
            geometry_supported_frame_ids=(0, 1, 2, 3),
        )


def test_minimum_fraction_rule_is_frozen_and_deterministic() -> None:
    supported = _stream("group-0", "camera-0")
    unsupported = _stream(
        "group-1",
        "camera-0",
        geometry_supported_frame_ids=(0,),
    )
    request = _request(
        (supported, unsupported),
        admission_rule="minimum-stream-fraction",
        minimum_supported_fraction=0.5,
    )
    result = evaluate_provider_support_feasibility(request)

    assert result.support_feasible
    assert result.support_fraction == 0.5

    with pytest.raises(ValueError, match="all-streams admission"):
        replace(request, admission_rule="all-streams")


def test_declared_technical_exclusion_obeys_budget() -> None:
    excluded = _stream(
        "group-1",
        technical_failure_code="released-file-unreadable",
        available_frame_ids=(),
        geometry_supported_frame_ids=(),
    )
    supported = _stream("group-0")
    passing = _request(
        (supported, excluded),
        permitted_technical_exclusion_codes=("released-file-unreadable",),
        maximum_technical_exclusions=1,
    )
    result = evaluate_provider_support_feasibility(passing)

    assert result.support_feasible
    assert result.excluded_stream_count == 1
    assert result.evaluable_stream_count == 1
    assert result.stream_results[1].excluded_from_admission
    assert "permitted-technical-exclusion" in result.stream_results[1].reason_codes

    over_budget = replace(passing, maximum_technical_exclusions=0)
    failed = evaluate_provider_support_feasibility(over_budget)
    assert not failed.support_feasible
    assert failed.decision_reason == "technical-exclusion-budget-exceeded"
    assert failed.excluded_stream_count == 0
    assert failed.stream_results[1].reason_codes[0] == (
        "insufficient-geometry-support"
    )
    assert "technical-exclusion-budget-exceeded" in (
        failed.stream_results[1].reason_codes
    )


def test_unpermitted_technical_failure_is_not_removed_from_denominator() -> None:
    failed_stream = _stream(
        technical_failure_code="unexpected-runtime-error",
    )
    result = evaluate_provider_support_feasibility(_request((failed_stream,)))

    assert not result.support_feasible
    assert result.excluded_stream_count == 0
    assert result.evaluable_stream_count == 1
    assert result.stream_results[0].reason_codes == (
        "unpermitted-technical-failure",
    )


@pytest.mark.parametrize(
    "field_name",
    ["prediction_payloads_opened", "residuals_used", "target_outcomes_used"],
)
def test_request_rejects_post_support_information(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be false"):
        _request(**{field_name: True})


def test_request_and_result_round_trip_and_detect_tampering(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_provider_support_feasibility_request(request_path, request)
    loaded_request = load_provider_support_feasibility_request(request_path)
    assert loaded_request == request

    result = evaluate_provider_support_feasibility(loaded_request)
    write_provider_support_feasibility(result_path, result)
    assert load_provider_support_feasibility(result_path) == result

    with pytest.raises(FileExistsError):
        write_provider_support_feasibility(result_path, result)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["support_feasible"] = False
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="does not replay|identity mismatch",
    ):
        load_provider_support_feasibility(result_path)


def test_identity_changes_when_support_metadata_changes() -> None:
    original = _request()
    changed = replace(
        original,
        streams=(
            replace(
                original.streams[0],
                geometry_supported_frame_ids=(0, 1, 2),
            ),
        ),
    )
    assert changed.request_id != original.request_id
    assert (
        evaluate_provider_support_feasibility(changed)
        .provider_support_feasibility_id
        != evaluate_provider_support_feasibility(original)
        .provider_support_feasibility_id
    )


def test_cli_evaluates_and_replays_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_provider_support_feasibility_request(request_path, _request())

    assert (
        main(
            [
                "evaluate",
                "--request",
                str(request_path),
                "--output",
                str(result_path),
                "--compact",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["support_feasible"] is True
    assert summary["support_fraction"] == 1.0

    assert main(["verify", "--artifact", str(result_path), "--compact"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == summary


def test_cli_returns_nonzero_for_support_negative(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _request(
        (
            _stream(
                geometry_supported_frame_ids=(0,),
            ),
        )
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_provider_support_feasibility_request(request_path, request)

    assert (
        main(
            [
                "evaluate",
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["support_feasible"] is False
