from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d.observation_timestamp_lineage import (
    TIMESTAMP_UNCERTAINTY_SEMANTICS,
    ObservationTimestampLineageV1,
    load_observation_timestamp_lineage,
    validate_timestamp_lineage_for_bundle,
    write_observation_timestamp_lineage,
)


def _lineage(**overrides: Any) -> ObservationTimestampLineageV1:
    values: dict[str, Any] = {
        "sequence_id": "sequence-1",
        "case_id": "case-1",
        "stream_id": "camera-stream-1",
        "source_revision": "a" * 40,
        "source_artifact_sha256": "b" * 64,
        "causal_frame_stop": 8,
        "clock_domain": "camera-hardware-clock",
        "time_scale": "device-monotonic",
        "timestamp_source": "camera-hardware-packet",
        "factor_ids": ("factor-0", "factor-1", "factor-2"),
        "frame_indices": np.asarray([0, 1, 1], dtype=np.int64),
        "timestamps_ns": np.asarray([100, 200, 210], dtype=np.int64),
        "conditional_timestamp_std_ns": np.asarray([1.0, 2.0, 2.0]),
        "shared_clock_offset_prior_artifact_id": "c" * 64,
        "metadata": {"capture": {"serial": "camera-1"}},
    }
    values.update(overrides)
    return ObservationTimestampLineageV1(**values)


@dataclass(frozen=True)
class _Factor:
    factor_id: str
    frame_index: int


@dataclass(frozen=True)
class _Bundle:
    sequence_id: str = "sequence-1"
    case_id: str | None = "case-1"
    stream_id: str | None = "camera-stream-1"
    source_revision: str = "a" * 40
    causal_frame_stop: int = 8
    factors: tuple[_Factor, ...] = (
        _Factor("factor-0", 0),
        _Factor("factor-1", 1),
        _Factor("factor-2", 1),
    )


def test_lineage_is_content_addressed_and_deterministic() -> None:
    first = _lineage()
    second = _lineage()

    assert first.artifact_id == second.artifact_id
    assert first.to_record()["artifact_id"] == first.artifact_id
    assert first.timestamp_uncertainty_semantics == TIMESTAMP_UNCERTAINTY_SEMANTICS


def test_arrays_are_defensive_and_irreversibly_read_only() -> None:
    frames = np.asarray([0, 1, 1], dtype=np.int64)
    timestamps = np.asarray([100, 200, 210], dtype=np.int64)
    standard_deviations = np.asarray([1.0, 2.0, 2.0])
    lineage = _lineage(
        frame_indices=frames,
        timestamps_ns=timestamps,
        conditional_timestamp_std_ns=standard_deviations,
    )

    frames[0] = 7
    timestamps[0] = 999
    standard_deviations[0] = 999.0
    np.testing.assert_array_equal(lineage.frame_indices, [0, 1, 1])
    np.testing.assert_array_equal(lineage.timestamps_ns, [100, 200, 210])
    np.testing.assert_allclose(
        lineage.conditional_timestamp_std_ns,
        [1.0, 2.0, 2.0],
    )

    for value in (
        lineage.frame_indices,
        lineage.timestamps_ns,
        lineage.conditional_timestamp_std_ns,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_round_trip_and_idempotent_publication(tmp_path: Path) -> None:
    lineage = _lineage()
    path = tmp_path / "timestamp-lineage.json"

    write_observation_timestamp_lineage(lineage, path)
    write_observation_timestamp_lineage(lineage, path)
    loaded = load_observation_timestamp_lineage(path)

    assert loaded.artifact_id == lineage.artifact_id
    assert loaded.identity_record() == lineage.identity_record()


def test_publication_refuses_different_content(tmp_path: Path) -> None:
    path = tmp_path / "timestamp-lineage.json"
    write_observation_timestamp_lineage(_lineage(), path)

    with pytest.raises(ValueError, match="different content"):
        write_observation_timestamp_lineage(
            _lineage(timestamps_ns=np.asarray([100, 200, 211])),
            path,
        )


def test_load_rejects_duplicate_keys_and_tampered_identity(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        load_observation_timestamp_lineage(duplicate)

    tampered = _lineage().to_record()
    tampered["artifact_id"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_observation_timestamp_lineage(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "conditional_timestamp_std_ns",
            np.asarray([1.0, -1.0, 1.0]),
            "finite and nonnegative",
        ),
        (
            "conditional_timestamp_std_ns",
            np.asarray([1.0, np.nan, 1.0]),
            "finite and nonnegative",
        ),
        (
            "timestamp_uncertainty_semantics",
            "marginal-including-shared-offset",
            "semantics changed",
        ),
        (
            "timestamps_ns",
            np.asarray([100.0, 200.0, 210.0]),
            "must contain integers",
        ),
        (
            "frame_indices",
            np.asarray([0, 1, 8], dtype=np.int64),
            "causal frame stop",
        ),
    ],
)
def test_invalid_timing_contracts_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _lineage(**{field: value})


def test_shared_clock_offset_prior_is_optional_but_not_folded_into_jitter() -> None:
    lineage = _lineage(shared_clock_offset_prior_artifact_id=None)

    assert lineage.shared_clock_offset_prior_artifact_id is None
    assert lineage.timestamp_uncertainty_semantics == (
        "conditional-jitter-excludes-shared-clock-offset"
    )


def test_bundle_binding_requires_exact_factor_order_and_causal_identity() -> None:
    lineage = _lineage()
    validate_timestamp_lineage_for_bundle(lineage, _Bundle())

    reordered = _Bundle(
        factors=(
            _Factor("factor-1", 1),
            _Factor("factor-0", 0),
            _Factor("factor-2", 1),
        )
    )
    with pytest.raises(ValueError, match="factor order"):
        validate_timestamp_lineage_for_bundle(lineage, reordered)

    wrong_revision = _Bundle(source_revision="d" * 40)
    with pytest.raises(ValueError, match="source_revision"):
        validate_timestamp_lineage_for_bundle(lineage, wrong_revision)


def test_bundle_defaults_for_case_and_stream_match_bundle_semantics() -> None:
    lineage = _lineage(
        case_id="sequence-1",
        stream_id="sequence-1",
    )
    bundle = _Bundle(case_id=None, stream_id=None)

    validate_timestamp_lineage_for_bundle(lineage, bundle)


def test_from_record_rejects_string_factor_ids() -> None:
    record = _lineage().to_record()
    record["factor_ids"] = "factor-0"

    with pytest.raises(ValueError, match="must be a sequence"):
        ObservationTimestampLineageV1.from_record(record)
