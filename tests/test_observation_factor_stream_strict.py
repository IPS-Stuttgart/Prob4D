from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import prob4d.observation_factor_stream as stream_module
from prob4d.observation_factor_stream import (
    ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1,
    load_observation_factor_stream,
    write_observation_factor_stream,
)


def _update(
    *,
    index: int = 0,
    start: int = 0,
    stop: int = 2,
    previous: str | None = None,
    suffix: str = "0",
) -> ObservationFactorStreamUpdateV1:
    return ObservationFactorStreamUpdateV1(
        update_index=index,
        admitted_frame_start=start,
        causal_frame_stop=stop,
        bundle_manifest_path=f"update-{suffix}/factors.json",
        bundle_manifest_sha256=suffix * 64,
        bundle_payload_sha256=(str((int(suffix) + 1) % 10)) * 64,
        bundle_sequence_id="sequence-a",
        case_id="case-a",
        stream_id="stream-a",
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        factor_count=1,
        observation_count=2,
        persistent_identity_count=2,
        observation_identity_sha256=(str((int(suffix) + 2) % 10)) * 64,
        gauge_ids=("window-0",),
        previous_update_id=previous,
    )


def _stream(*, two_updates: bool = False) -> ObservationFactorStreamV1:
    first = _update()
    updates = [first]
    if two_updates:
        updates.append(
            _update(
                index=1,
                start=2,
                stop=4,
                previous=first.update_id,
                suffix="3",
            )
        )
    return ObservationFactorStreamV1(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="stream-a",
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        updates=tuple(updates),
        metadata={"protocol": "strict-stream-test"},
    )


def test_roundtrip_preserves_existing_identity(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    stream = _stream()
    write_observation_factor_stream(stream, path)
    loaded = load_observation_factor_stream(path, validate_bundles=False)
    assert loaded == stream
    assert loaded.artifact_id == stream.artifact_id


def test_persistence_allows_only_idempotent_or_append_only_rewrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream.json"
    first = _stream()
    extended = _stream(two_updates=True)

    write_observation_factor_stream(first, path)
    write_observation_factor_stream(first, path)
    write_observation_factor_stream(extended, path)
    assert load_observation_factor_stream(path, validate_bundles=False) == extended

    with pytest.raises(ValueError, match="cannot roll back"):
        write_observation_factor_stream(first, path)

    forked_first = replace(
        first.updates[0],
        bundle_manifest_sha256="9" * 64,
        update_id=None,
    )
    forked = replace(first, updates=(forked_first,), artifact_id=None)
    fork_path = tmp_path / "fork.json"
    write_observation_factor_stream(first, fork_path)
    with pytest.raises(ValueError, match="cannot fork"):
        write_observation_factor_stream(forked, fork_path)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    write_observation_factor_stream(_stream(), path)
    text = path.read_text(encoding="utf-8").replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_observation_factor_stream(path, validate_bundles=False)


def test_rejects_nonfinite_json_metadata(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    write_observation_factor_stream(_stream(), path)
    text = path.read_text(encoding="utf-8").replace(
        '"protocol": "strict-stream-test"',
        '"score": NaN',
        1,
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_observation_factor_stream(path, validate_bundles=False)


Mutator = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda record: record.__setitem__("schema_version", "1"),
            "schema_version must be an integer",
        ),
        (
            lambda record: record["updates"][0].__setitem__("update_index", True),
            "update_index must be an integer",
        ),
        (
            lambda record: record["updates"][0].__setitem__("gauge_ids", [1]),
            "gauge_ids must contain nonempty strings",
        ),
        (
            lambda record: record.__setitem__("sequence_id", 7),
            "sequence_id must be a nonempty string",
        ),
    ],
)
def test_rejects_coercion_dependent_values(
    tmp_path: Path,
    mutator: Mutator,
    message: str,
) -> None:
    path = tmp_path / "stream.json"
    write_observation_factor_stream(_stream(), path)
    record = json.loads(path.read_text(encoding="utf-8"))
    mutator(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_observation_factor_stream(path, validate_bundles=False)


def test_rejects_unknown_stream_or_update_fields(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    write_observation_factor_stream(_stream(), path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["unexpected"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="stream manifest fields changed"):
        load_observation_factor_stream(path, validate_bundles=False)

    update_path = tmp_path / "stream-update.json"
    write_observation_factor_stream(_stream(), update_path)
    record = json.loads(update_path.read_text(encoding="utf-8"))
    record["updates"][0]["unexpected"] = 1
    update_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="stream update fields changed"):
        load_observation_factor_stream(update_path, validate_bundles=False)


def test_writer_lock_rejects_a_concurrent_library_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream.json"
    lock = path.with_name(f".{path.name}.lock")
    lock.write_text("competing writer\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already being written"):
        write_observation_factor_stream(_stream(), path)

    assert not path.exists()
    assert lock.read_text(encoding="utf-8") == "competing writer\n"


def test_first_publication_never_overwrites_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "stream.json"
    original_link = stream_module.os.link

    def racing_link(source: object, destination: object) -> None:
        Path(destination).write_text(
            "competing writer\n",
            encoding="utf-8",
        )
        original_link(source, destination)

    monkeypatch.setattr(stream_module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        write_observation_factor_stream(_stream(), path)

    assert path.read_text(encoding="utf-8") == "competing writer\n"
