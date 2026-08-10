from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prob4d.provider_support_design import (
    PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY,
    ProviderSupportDesignCandidateV1,
    ProviderSupportDesignRequestV1,
    evaluate_provider_support_design,
    load_provider_support_design,
    load_provider_support_design_request,
    main,
    write_provider_support_design,
    write_provider_support_design_request,
    write_selected_provider_support_feasibility,
    write_selected_provider_support_request,
)
from prob4d.provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    load_provider_support_feasibility,
    load_provider_support_feasibility_request,
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
ROSTER = (
    ("group-0", "camera-0"),
    ("group-0", "camera-1"),
    ("group-1", "camera-0"),
    ("group-1", "camera-1"),
)


def _stream(
    group_id: str,
    stream_id: str,
    *,
    frame_count: int = 4,
    causal_span: int | None = None,
    supported: bool = True,
    **changes: object,
) -> ProviderSupportStreamV1:
    span = frame_count if causal_span is None else causal_span
    required = tuple(range(frame_count))
    values: dict[str, object] = {
        "group_id": group_id,
        "stream_id": stream_id,
        "causal_frame_start": 0,
        "causal_frame_stop_exclusive": span,
        "required_frame_ids": required,
        "available_frame_ids": required,
        "geometry_supported_frame_ids": required if supported else (),
        "minimum_geometry_support_fraction": 1.0,
        "intrinsics_required": True,
        "intrinsics_id": DIGESTS["intrinsics"],
        "extrinsics_required": True,
        "extrinsics_id": DIGESTS["extrinsics"],
        "metric_anchor_required": True,
        "metric_anchor_id": DIGESTS["anchor"],
        "technical_failure_code": None,
        "metadata": {"source": "geometry-only"},
    }
    values.update(changes)
    return ProviderSupportStreamV1(**values)  # type: ignore[arg-type]


def _feasibility_request(
    candidate_id: str,
    supported_keys: set[tuple[str, str]],
    *,
    frame_count: int = 4,
    causal_span: int | None = None,
    admission_rule: str = "minimum-stream-fraction",
    minimum_supported_fraction: float = 0.5,
    streams: tuple[ProviderSupportStreamV1, ...] | None = None,
    **changes: object,
) -> ProviderSupportFeasibilityRequestV1:
    values: dict[str, object] = {
        "protocol_id": f"provider-support-{candidate_id}",
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
        "admission_rule": admission_rule,
        "minimum_supported_fraction": minimum_supported_fraction,
        "permitted_technical_exclusion_codes": (),
        "maximum_technical_exclusions": 0,
        "prediction_payloads_opened": False,
        "residuals_used": False,
        "target_outcomes_used": False,
        "streams": streams
        or tuple(
            _stream(
                group_id,
                stream_id,
                frame_count=frame_count,
                causal_span=causal_span,
                supported=(group_id, stream_id) in supported_keys,
            )
            for group_id, stream_id in ROSTER
        ),
        "metadata": {"candidate": candidate_id},
    }
    values.update(changes)
    return ProviderSupportFeasibilityRequestV1(**values)  # type: ignore[arg-type]


def _candidate(
    candidate_id: str,
    supported_keys: set[tuple[str, str]],
    **changes: object,
) -> ProviderSupportDesignCandidateV1:
    request_changes = dict(changes)
    metadata = request_changes.pop("candidate_metadata", {"frozen": True})
    return ProviderSupportDesignCandidateV1(
        candidate_id=candidate_id,
        feasibility_request=_feasibility_request(
            candidate_id,
            supported_keys,
            **request_changes,
        ),
        metadata=metadata,  # type: ignore[arg-type]
    )


def _design_request(
    candidates: tuple[ProviderSupportDesignCandidateV1, ...],
    **changes: object,
) -> ProviderSupportDesignRequestV1:
    values: dict[str, object] = {
        "protocol_id": "provider-support-design-test-v1",
        "candidates": candidates,
        "prediction_payloads_opened": False,
        "residuals_used": False,
        "target_outcomes_used": False,
        "metadata": {"phase": "pre-residual"},
    }
    values.update(changes)
    return ProviderSupportDesignRequestV1(**values)  # type: ignore[arg-type]


def test_selects_support_feasible_candidate_before_negative_candidate() -> None:
    negative = _candidate("negative", {ROSTER[0]})
    feasible = _candidate("feasible", {ROSTER[0], ROSTER[2]})
    result = evaluate_provider_support_design(_design_request((negative, feasible)))

    assert result.support_design_feasible
    assert result.selected_candidate_id == "feasible"
    assert result.feasible_candidate_count == 1
    assert result.selected_support_feasibility.support_feasible
    assert result.selected_support_request.request_id == (
        result.selected_support_request_id
    )
    assert result.to_dict()["claim_boundary"] == (
        PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY
    )


def test_maximin_group_support_precedes_other_ranking_terms() -> None:
    imbalanced = _candidate(
        "imbalanced",
        {ROSTER[0], ROSTER[1]},
        minimum_supported_fraction=0.25,
    )
    balanced = _candidate(
        "balanced",
        {ROSTER[0], ROSTER[2]},
        minimum_supported_fraction=0.25,
    )
    result = evaluate_provider_support_design(
        _design_request((imbalanced, balanced))
    )

    assert result.selected_candidate_id == "balanced"
    assert result.selected_evaluation.minimum_group_support_fraction == 0.5
    other = next(
        item
        for item in result.candidate_evaluations
        if item.candidate_id == "imbalanced"
    )
    assert other.minimum_group_support_fraction == 0.0
    assert other.support_feasibility.supported_stream_count == 2


def test_supported_frame_cells_precede_shorter_causal_span() -> None:
    supported = {ROSTER[0], ROSTER[2]}
    longer = _candidate(
        "longer-more-cells",
        supported,
        frame_count=6,
        causal_span=6,
    )
    shorter = _candidate(
        "shorter-fewer-cells",
        supported,
        frame_count=4,
        causal_span=4,
    )
    result = evaluate_provider_support_design(_design_request((shorter, longer)))

    assert result.selected_candidate_id == "longer-more-cells"
    assert result.selected_evaluation.supported_required_frame_count == 12


def test_shorter_span_and_candidate_id_are_deterministic_tiebreakers() -> None:
    supported = {ROSTER[0], ROSTER[2]}
    loose = _candidate("loose", supported, frame_count=4, causal_span=8)
    tight_z = _candidate("tight-z", supported, frame_count=4, causal_span=4)
    tight_a = _candidate("tight-a", supported, frame_count=4, causal_span=4)
    result = evaluate_provider_support_design(
        _design_request((tight_z, loose, tight_a))
    )

    assert result.selected_candidate_id == "tight-a"
    assert result.selected_evaluation.maximum_causal_span_frames == 4


def test_candidate_set_requires_common_identity_roster_and_requirements() -> None:
    first = _candidate("first", {ROSTER[0], ROSTER[2]})
    changed_model = _candidate(
        "changed-model",
        {ROSTER[0], ROSTER[2]},
        model_set_id="8" * 64,
    )
    with pytest.raises(ValueError, match="share provider, cohort"):
        _design_request((first, changed_model))

    reduced_roster = _candidate(
        "reduced-roster",
        {ROSTER[0]},
        streams=first.feasibility_request.streams[:-1],
    )
    with pytest.raises(ValueError, match="exact stream roster"):
        _design_request((first, reduced_roster))

    changed_threshold_streams = tuple(
        replace(stream, minimum_geometry_support_fraction=0.5)
        if stream.key == ROSTER[0]
        else stream
        for stream in first.feasibility_request.streams
    )
    changed_threshold = _candidate(
        "changed-threshold",
        {ROSTER[0], ROSTER[2]},
        streams=changed_threshold_streams,
    )
    with pytest.raises(ValueError, match="must not change support thresholds"):
        _design_request((first, changed_threshold))


@pytest.mark.parametrize(
    "field_name",
    ["prediction_payloads_opened", "residuals_used", "target_outcomes_used"],
)
def test_design_request_rejects_late_information(field_name: str) -> None:
    candidate = _candidate("candidate", {ROSTER[0], ROSTER[2]})
    with pytest.raises(ValueError, match=f"{field_name} must be false"):
        _design_request((candidate,), **{field_name: True})


def test_all_negative_candidates_retain_best_valid_negative() -> None:
    one_group = _candidate("one-group", {ROSTER[0]})
    no_groups = _candidate("no-groups", set())
    result = evaluate_provider_support_design(
        _design_request((no_groups, one_group))
    )

    assert not result.support_design_feasible
    assert result.decision_reason == "no-support-feasible-candidate"
    assert result.selected_candidate_id == "one-group"
    assert result.feasible_candidate_count == 0


def test_round_trip_selected_artifacts_and_tamper_detection(tmp_path: Path) -> None:
    request = _design_request(
        (
            _candidate("candidate-b", {ROSTER[0]}),
            _candidate("candidate-a", {ROSTER[0], ROSTER[2]}),
        )
    )
    request_path = tmp_path / "design-request.json"
    result_path = tmp_path / "design-result.json"
    selected_request_path = tmp_path / "selected-request.json"
    selected_result_path = tmp_path / "selected-result.json"
    write_provider_support_design_request(request_path, request)
    loaded_request = load_provider_support_design_request(request_path)
    assert loaded_request == request

    result = evaluate_provider_support_design(loaded_request)
    write_selected_provider_support_request(selected_request_path, result)
    write_selected_provider_support_feasibility(selected_result_path, result)
    write_provider_support_design(result_path, result)

    assert load_provider_support_design(result_path) == result
    assert (
        load_provider_support_feasibility_request(selected_request_path)
        == result.selected_support_request
    )
    assert (
        load_provider_support_feasibility(selected_result_path)
        == result.selected_support_feasibility
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["selected_candidate_id"] = "candidate-b"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch|does not replay"):
        load_provider_support_design(result_path)


def test_cli_selects_writes_replays_and_returns_negative_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing_request_path = tmp_path / "passing-request.json"
    passing_result_path = tmp_path / "passing-result.json"
    selected_request_path = tmp_path / "selected-request.json"
    selected_result_path = tmp_path / "selected-result.json"
    passing = _design_request(
        (_candidate("candidate", {ROSTER[0], ROSTER[2]}),)
    )
    write_provider_support_design_request(passing_request_path, passing)

    assert (
        main(
            [
                "select",
                "--request",
                str(passing_request_path),
                "--output",
                str(passing_result_path),
                "--selected-request-output",
                str(selected_request_path),
                "--selected-feasibility-output",
                str(selected_result_path),
                "--compact",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["support_design_feasible"] is True
    assert summary["selected_candidate_id"] == "candidate"
    assert main(["verify", "--artifact", str(passing_result_path)]) == 0
    assert json.loads(capsys.readouterr().out)["selected_candidate_id"] == (
        "candidate"
    )

    negative_request_path = tmp_path / "negative-request.json"
    negative_result_path = tmp_path / "negative-result.json"
    negative = _design_request((_candidate("negative", set()),))
    write_provider_support_design_request(negative_request_path, negative)
    assert (
        main(
            [
                "select",
                "--request",
                str(negative_request_path),
                "--output",
                str(negative_result_path),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["support_design_feasible"] is False


def test_cli_rejects_colliding_or_existing_outputs(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "output.json"
    request = _design_request(
        (_candidate("candidate", {ROSTER[0], ROSTER[2]}),)
    )
    write_provider_support_design_request(request_path, request)

    with pytest.raises(ValueError, match="output paths must be distinct"):
        main(
            [
                "select",
                "--request",
                str(request_path),
                "--output",
                str(output_path),
                "--selected-request-output",
                str(output_path),
            ]
        )

    output_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(
            [
                "select",
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]
        )
