from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import prob4d.observation_contract as observation_contract_module
from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)


def _artifact() -> ObservationBeliefExportV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefExportV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=int),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


@pytest.mark.parametrize("value", (True, 12.0, np.float64(12.0)))
def test_producer_rejects_noninteger_causal_cutoff(value: object) -> None:
    with pytest.raises(ValueError, match="causal_frame_stop must be an integer"):
        replace(_artifact(), causal_frame_stop=cast(int, value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("case_id", 7, "case_id and stream_id must be nonempty"),
        ("stream_id", b"stream", "case_id and stream_id must be nonempty"),
        ("view_names", (7,), "view_names must contain nonempty names"),
        ("window_names", ("window0", 7), "window_names must contain"),
        ("factor_names", (7,), "factor_names must be nonempty"),
        ("source_repository", 7, "source repository and revision"),
        ("source_revision", b"revision", "source repository and revision"),
        ("source_artifact_sha256", 7, "lowercase SHA-256"),
    ),
)
def test_producer_rejects_lossy_descriptor_aliases(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_artifact(), **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "declared_frame_ids",
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "group_ids",
    ),
)
@pytest.mark.parametrize("dtype", (np.float64, np.bool_))
def test_producer_rejects_noninteger_identity_arrays(
    field: str,
    dtype: type[np.generic],
) -> None:
    source = _artifact()
    values = np.asarray(getattr(source, field), dtype=dtype)
    with pytest.raises(ValueError, match=f"{field} must contain integers"):
        replace(source, **{field: values})


def test_producer_rejects_unsigned_identity_overflow() -> None:
    source = _artifact()
    overflow = np.asarray([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(ValueError, match="outside int64 range"):
        replace(source, group_ids=overflow)


def test_producer_canonicalizes_descriptor_and_identity_storage() -> None:
    source = _artifact()
    artifact = replace(
        source,
        causal_frame_stop=np.int64(source.causal_frame_stop),
        view_names=list(source.view_names),
        window_names=list(source.window_names),
        factor_names=list(source.factor_names),
        frame_ids=np.asarray(source.frame_ids, dtype=np.int16),
    )

    assert type(artifact.causal_frame_stop) is int
    assert type(artifact.view_names) is tuple
    assert type(artifact.window_names) is tuple
    assert type(artifact.factor_names) is tuple
    assert artifact.frame_ids.dtype.str == "<i8"
    assert artifact.artifact_id == source.artifact_id

    for name, values in artifact.arrays().items():
        assert not values.flags.writeable, name
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_atomic_writer_does_not_clobber_retained_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "observation.npz"
    target.write_bytes(b"retained evidence")

    def fail_after_partial_write(handle: Any, **payload: object) -> None:
        del payload
        handle.write(b"partial replacement")
        handle.flush()
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(
        observation_contract_module.np,
        "savez_compressed",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        save_observation_belief_export(target, _artifact())

    assert target.read_bytes() == b"retained evidence"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_atomic_writer_refuses_to_replace_symlink(tmp_path: Path) -> None:
    retained = tmp_path / "retained.npz"
    retained.write_bytes(b"retained evidence")
    target = tmp_path / "observation.npz"
    target.symlink_to(retained)

    with pytest.raises(ValueError, match="refusing to replace symlink"):
        save_observation_belief_export(target, _artifact())

    assert retained.read_bytes() == b"retained evidence"
