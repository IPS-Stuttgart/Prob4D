from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
    MotionCrafterAdapter,
    MotionCrafterRunConfig,
    motioncrafter_seed_for_call,
    validate_motioncrafter_seed_schedule,
)


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


def test_motioncrafter_config_validates_window_geometry() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MotionCrafterRunConfig(
            upstream_root=Path("upstream"),
            video_path=Path("video.mp4"),
            output_directory=Path("output"),
            window_size=25,
            overlap=25,
        )

    with pytest.raises(ValueError, match="divisible"):
        MotionCrafterRunConfig(
            upstream_root=Path("upstream"),
            video_path=Path("video.mp4"),
            output_directory=Path("output"),
            height=321,
        )


def test_motioncrafter_config_validates_source_frame_selection() -> None:
    base = {
        "upstream_root": Path("upstream"),
        "video_path": Path("video.mp4"),
        "output_directory": Path("output"),
    }
    with pytest.raises(ValueError, match="frame_start"):
        MotionCrafterRunConfig(**base, frame_start=-1)
    with pytest.raises(ValueError, match="frame_stop"):
        MotionCrafterRunConfig(**base, frame_start=10, frame_stop=10)
    with pytest.raises(ValueError, match="frame_stride"):
        MotionCrafterRunConfig(**base, frame_stride=0)
    with pytest.raises(ValueError, match="seed must"):
        MotionCrafterRunConfig(**base, seed=-1)
    with pytest.raises(ValueError, match="seed_policy"):
        MotionCrafterRunConfig(**base, seed_policy="implicit")  # type: ignore[arg-type]


def test_motioncrafter_seed_policy_is_deterministic_and_call_sensitive() -> None:
    assert (
        motioncrafter_seed_for_call(
            42,
            policy=MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
            call_id="window-a",
        )
        == 42
    )

    first = motioncrafter_seed_for_call(
        42,
        policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
        call_id="window-a",
    )
    assert first == motioncrafter_seed_for_call(
        42,
        policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
        call_id="window-a",
    )
    assert first != motioncrafter_seed_for_call(
        42,
        policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
        call_id="window-b",
    )
    assert first != motioncrafter_seed_for_call(
        43,
        policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
        call_id="window-a",
    )
    assert 0 <= first < 2**32


def test_motioncrafter_run_records_source_bound_seed_schedule(tmp_path: Path) -> None:
    output = tmp_path / "output"
    config = MotionCrafterRunConfig(
        upstream_root=tmp_path / "upstream",
        video_path=tmp_path / "video.mp4",
        output_directory=output,
        window_size=25,
        overlap=8,
        seed=71,
        seed_policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    )
    adapter = object.__new__(MotionCrafterAdapter)
    adapter.config = config

    def fake_read_video(_: Path | None = None) -> np.ndarray:
        return np.zeros((30, 3, 64, 64), dtype=np.float32)

    adapter.read_video = fake_read_video

    def fake_infer(
        frames: np.ndarray,
        *,
        window_size: int,
        overlap: int,
        seed: int,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        del window_size, overlap, seed
        count = int(frames.shape[0])
        points = np.zeros((count, 2, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        valid = np.ones((count, 2, 2), dtype=bool)
        return _FakeTensor(points), _FakeTensor(valid)

    adapter.infer = fake_infer

    def fake_upstream_commit() -> str:
        return "a" * 40

    adapter._upstream_commit = fake_upstream_commit

    manifest_path = adapter.run()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule = manifest["stochastic_seed_schedule"]
    validation = validate_motioncrafter_seed_schedule(manifest)

    assert validation["call_count"] == 4
    assert validation["source"] == "manifest"
    assert schedule["schema"] == MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA
    assert schedule["policy"] == MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
    assert schedule["root_seed"] == 71
    assert manifest["config"]["seed_policy"] == MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
    calls = schedule["calls"]
    assert [record["product"] for record in calls[:2]] == [
        "disjoint_baseline",
        "latent_linear_baseline",
    ]
    assert len(calls) == 4
    assert len({record["call_id"] for record in calls}) == len(calls)
    assert len({record["effective_seed"] for record in calls}) == len(calls)
    for record in calls:
        assert record["effective_seed"] == motioncrafter_seed_for_call(
            71,
            policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
            call_id=record["call_id"],
        )

    tampered = json.loads(json.dumps(manifest))
    tampered["stochastic_seed_schedule"]["calls"][2]["effective_seed"] += 1
    with pytest.raises(ValueError, match="invalid effective seed"):
        validate_motioncrafter_seed_schedule(tampered)

    reordered = json.loads(json.dumps(manifest))
    reordered_calls = reordered["stochastic_seed_schedule"]["calls"]
    reordered_calls[0], reordered_calls[1] = reordered_calls[1], reordered_calls[0]
    with pytest.raises(ValueError, match="inconsistent call_id"):
        validate_motioncrafter_seed_schedule(reordered)


def test_derived_seed_policy_fails_closed_without_a_schedule() -> None:
    manifest = {
        "config": {
            "seed": 42,
            "seed_policy": MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
        }
    }
    with pytest.raises(ValueError, match="lacks a seed schedule"):
        validate_motioncrafter_seed_schedule(manifest)
