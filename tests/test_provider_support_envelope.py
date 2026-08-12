from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.provider_support_envelope import (
    PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY,
    derive_provider_support_envelope,
    load_provider_support_envelope,
    write_provider_support_envelope,
)
from prob4d.provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
)

DIGESTS = {
    name: character * 64
    for name, character in {
        "model": "1",
        "loader": "2",
        "cohort": "3",
        "lock": "4",
        "intrinsics": "5",
        "extrinsics": "6",
        "anchor": "7",
    }.items()
}


def _stream(**changes: object) -> ProviderSupportStreamV1:
    values: dict[str, object] = {
        "group_id": "object-a",
        "stream_id": "camera-0",
        "causal_frame_start": 0,
        "causal_frame_stop_exclusive": 6,
        "required_frame_ids": (0, 1, 2, 3, 4, 5),
        "available_frame_ids": (0, 1, 2, 3, 4, 5),
        "geometry_supported_frame_ids": (0, 1, 2, 3, 4, 5),
        "minimum_geometry_support_fraction": 1.0,
        "intrinsics_required": True,
        "intrinsics_id": DIGESTS["intrinsics"],
        "extrinsics_required": True,
        "extrinsics_id": DIGESTS["extrinsics"],
        "metric_anchor_required": True,
        "metric_anchor_id": DIGESTS["anchor"],
        "technical_failure_code": None,
        "metadata": {},
    }
    values.update(changes)
    return ProviderSupportStreamV1(**values)  # type: ignore[arg-type]


def _request(stream: ProviderSupportStreamV1) -> ProviderSupportFeasibilityRequestV1:
    return ProviderSupportFeasibilityRequestV1(
        protocol_id="support-envelope-test-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        provider_family="external-provider",
        provider_repository="example/provider",
        provider_revision="b" * 40,
        model_set_id=DIGESTS["model"],
        loader_id=DIGESTS["loader"],
        cohort_binding_id=DIGESTS["cohort"],
        promotion_lock_id=DIGESTS["lock"],
        coordinate_semantics="metric-world-frame",
        admission_rule="all-streams",
        minimum_supported_fraction=1.0,
        permitted_technical_exclusion_codes=(),
        maximum_technical_exclusions=0,
        prediction_payloads_opened=False,
        residuals_used=False,
        target_outcomes_used=False,
        streams=(stream,),
    )


def test_envelope_reports_all_contiguous_supported_intervals() -> None:
    request = _request(
        _stream(
            available_frame_ids=(0, 1, 2, 4, 5),
            geometry_supported_frame_ids=(0, 1, 4, 5),
        )
    )

    envelope = derive_provider_support_envelope(request)
    stream = envelope.streams[0]

    assert stream.admissible_intervals == ((0, 2), (4, 6))
    assert stream.admissible_frame_count == 4
    assert stream.earliest_admissible_frame == 0
    assert stream.latest_admissible_frame_exclusive == 6
    assert stream.maximum_contiguous_frame_count == 2
    assert stream.required_admissible_frame_count == 4
    assert stream.required_support_fraction == pytest.approx(4 / 6)
    assert not stream.feasibility_supported
    assert envelope.total_admissible_frame_count == 4


def test_missing_static_support_or_technical_failure_empties_envelope() -> None:
    missing_anchor = derive_provider_support_envelope(
        _request(_stream(metric_anchor_id=None))
    ).streams[0]
    assert not missing_anchor.static_support_complete
    assert missing_anchor.admissible_intervals == ()

    technical = derive_provider_support_envelope(
        _request(_stream(technical_failure_code="camera-unreadable"))
    ).streams[0]
    assert technical.technical_failure_code == "camera-unreadable"
    assert technical.admissible_frame_count == 0


def test_envelope_is_replayable_and_does_not_mutate_request(tmp_path: Path) -> None:
    request = _request(_stream())
    before = request.to_dict()
    envelope = derive_provider_support_envelope(
        request,
        metadata={"stage": "pre-residual"},
    )
    path = tmp_path / "envelope.json"
    write_provider_support_envelope(path, envelope)

    loaded = load_provider_support_envelope(path)
    assert loaded.to_dict() == envelope.to_dict()
    assert request.to_dict() == before
    assert loaded.to_dict()["claim_boundary"] == PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["total_admissible_frame_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch|derived fields changed"):
        load_provider_support_envelope(path)
