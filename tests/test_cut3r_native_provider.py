from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from prob4d import cut3r_native_provider as provider
from prob4d import cut3r_native_runtime as native
from prob4d.data import PredictionWindow
from prob4d.prediction_cli import main
from prob4d.prediction_provider_manifest import verify_prediction_provider_manifest


class FakeRuntime:
    identity = {"cut3r_revision": "a" * 40, "checkpoint_sha256": "b" * 64}

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.seen: list[bytes] = []
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def step(self, path: Path) -> dict[str, np.ndarray]:
        if self.index == self.fail_at:
            raise RuntimeError("synthetic provider failure")
        self.seen.append(path.read_bytes())
        self.index += 1
        points = np.ones((2, 3, 3), dtype=np.float32)
        points[..., 0] = self.index * 0.1
        return {
            "points": points,
            "confidence": np.full((2, 3), 2, dtype=np.float32),
            "pose": np.eye(4),
            "intrinsics": np.eye(3),
        }


def _frames(root: Path, count: int = 3) -> Path:
    root.mkdir()
    for i in range(count):
        (root / f"{i:06d}.png").write_bytes(f"synthetic frame {i}".encode())
    return root


def _run(tmp_path: Path, *, runtime: FakeRuntime | None = None, stop: int = 2) -> dict[str, Any]:
    return provider.run_cut3r_native(
        tmp_path / "frames",
        tmp_path / "out",
        runtime_factory=lambda: runtime or FakeRuntime(),
        sequence_id="synthetic",
        frame_start=0,
        frame_stop=stop,
    )


def test_native_export_is_causal_direct_and_loadable(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames")
    (frames / "000002.png").unlink()
    (frames / "000002.png").symlink_to("unreadable-future")
    runtime = FakeRuntime()
    receipt = _run(tmp_path, runtime=runtime)
    assert receipt["status"] == "success"
    assert receipt["frames_completed"] == 2
    assert runtime.seen == [b"synthetic frame 0", b"synthetic frame 1"]
    assert not (tmp_path / "out/staging/decoded").exists()
    unsigned = dict(receipt)
    assert unsigned.pop("artifact_id") == provider.content_id(unsigned)
    output = tmp_path / "out/prediction/provider.json"
    manifest, _ = verify_prediction_provider_manifest(output)
    assert manifest.coordinate_semantics == "sequence-local-sim3"
    assert manifest.flow_semantics == "absent"
    assert manifest.metadata["confidence_is_support_not_reliability"]
    payload = manifest.payloads[0]
    assert [row.source_frame_stop_exclusive for row in payload.frame_lineage] == [1, 2]
    window = PredictionWindow.from_npz(output.parent / payload.path, dense_storage_dtype="float32")
    assert window.point_map.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(window.point_map[0, ..., 0], np.full((2, 3), 0.1, np.float32))
    assert window.ray_directions is not None


def test_failure_never_publishes_manifest_and_cannot_clobber(tmp_path: Path) -> None:
    _frames(tmp_path / "frames")
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        _run(tmp_path, runtime=FakeRuntime(fail_at=1))
    receipt = json.loads((tmp_path / "out/run.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["frames_completed"] == 1
    assert receipt["stage"] == "provider-inference"
    assert not receipt["prediction_published"]
    assert not (tmp_path / "out/prediction").exists()
    assert not (tmp_path / "out/staging/decoded").exists()
    before = (tmp_path / "out/run.json").read_bytes()
    with pytest.raises(FileExistsError):
        _run(tmp_path)
    assert (tmp_path / "out/run.json").read_bytes() == before


def test_runtime_failure_occurs_before_any_frame_staging(tmp_path: Path) -> None:
    def failed() -> FakeRuntime:
        raise RuntimeError("runtime unavailable")

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        provider.run_cut3r_native(
            tmp_path / "not-opened",
            tmp_path / "out",
            runtime_factory=failed,
            sequence_id="synthetic",
            frame_start=0,
            frame_stop=1,
        )
    receipt = json.loads((tmp_path / "out/run.json").read_text())
    assert receipt["stage"] == "runtime-initialization"
    assert receipt["frames_completed"] == 0
    assert "input" not in receipt


@pytest.mark.parametrize("stop", [0, -1, True, 4097])
def test_invalid_interval_precedes_runtime(tmp_path: Path, stop: int) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path, stop=stop)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extension", [".tiff", "../a", "png"])
def test_invalid_extension(tmp_path: Path, extension: str) -> None:
    with pytest.raises(ValueError, match="extension"):
        provider.run_cut3r_native(
            tmp_path / "none",
            tmp_path / "out",
            runtime_factory=FakeRuntime,
            sequence_id="x",
            frame_start=0,
            frame_stop=1,
            extension=extension,
        )


def test_missing_selected_frame_fails_closed(tmp_path: Path) -> None:
    _frames(tmp_path / "frames", count=1)
    with pytest.raises(FileNotFoundError):
        _run(tmp_path)
    assert not (tmp_path / "out/prediction").exists()


def test_symlinked_selected_frame_rejected(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames")
    (frames / "000000.png").unlink()
    (frames / "000000.png").symlink_to(frames / "000001.png")
    with pytest.raises(ValueError, match="symbolic"):
        _run(tmp_path)


def test_frame_mutation_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _frames(tmp_path / "frames")
    copy = shutil.copyfile

    def corrupt(source: Path, destination: Path) -> None:
        copy(source, destination)
        destination.write_bytes(b"changed")

    monkeypatch.setattr(shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="changed"):
        _run(tmp_path)


def test_video_emission_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"synthetic video")
    calls: list[list[str]] = []

    def ffmpeg(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        destination = Path(command[-1]).parent
        for i in range(2):
            (destination / f"{i:06d}.png").write_bytes(b"frame")

    monkeypatch.setattr(subprocess, "run", ffmpeg)
    provider.run_cut3r_native(
        video,
        tmp_path / "out",
        runtime_factory=FakeRuntime,
        sequence_id="video",
        frame_start=3,
        frame_stop=5,
        video=True,
    )
    command = calls[0]
    assert command[command.index("-frames:v") + 1] == "2"
    assert command[command.index("-vsync") + 1] == "0"
    assert "-fps_mode" not in command  # supported by FFmpeg 4.4 as well as newer releases
    assert command[command.index("-vf") + 1] == "select=between(n\\,3\\,4)"
    assert "-nostdin" in command


def test_video_error_preserves_decoder_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"synthetic")

    def failed(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command, stderr=b"invalid synthetic video")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="invalid synthetic video"):
        provider.run_cut3r_native(
            video,
            tmp_path / "out",
            runtime_factory=FakeRuntime,
            sequence_id="test",
            frame_start=0,
            frame_stop=1,
            video=True,
        )
    receipt = json.loads((tmp_path / "out/run.json").read_text())
    assert receipt["stage"] == "prefix-staging"
    assert "invalid synthetic video" in receipt["error"]
    assert not (tmp_path / "out/prediction").exists()


@pytest.mark.parametrize("mutation", ["nan", "shape", "pose", "focal", "empty"])
def test_bad_prediction_never_published(tmp_path: Path, mutation: str) -> None:
    frames = _frames(tmp_path / "frames")

    class BadRuntime(FakeRuntime):
        def step(self, path: Path) -> dict[str, np.ndarray]:
            value = super().step(path)
            if mutation == "nan":
                value["points"][0, 0, 0] = np.nan
            elif mutation == "shape":
                value["confidence"] = np.ones((1, 1))
            elif mutation == "pose":
                value["pose"][0, 0] = 2
            elif mutation == "focal":
                value["intrinsics"][0, 0] = -1
            else:
                value["confidence"][:] = 0
            return value

    with pytest.raises(ValueError):
        provider.run_cut3r_native(
            frames,
            tmp_path / "out",
            runtime_factory=BadRuntime,
            sequence_id="x",
            frame_start=0,
            frame_stop=1,
        )
    assert not (tmp_path / "out/prediction").exists()


def test_cli_help_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from prob4d.prediction_cli import main; "
            "main(['--help']); assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run-cut3r" in result.stdout


def test_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prob4d import _cut3r_native_cli as cli

    calls = []

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"status": "success", "artifact_id": "a" * 64, "frames_completed": 2}

    monkeypatch.setattr(cli, "run_cut3r_native", run)
    assert (
        main(
            [
                "run-cut3r",
                "--frames",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--sequence-id",
                "test",
                "--frame-stop",
                "2",
                "--cut3r-checkout",
                str(tmp_path),
                "--checkpoint",
                "trusted.pth",
                "--checkpoint-sha256",
                "b" * 64,
            ]
        )
        == 0
    )
    assert calls[0][1]["frame_stop"] == 2
    assert calls[0][1]["video"] is False


def test_native_runtime_validates_before_optional_imports(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"not actual weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        native.Cut3RNativeRuntime(tmp_path, checkpoint, checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="image_size"):
        native.Cut3RNativeRuntime(tmp_path, checkpoint, checkpoint_sha256="a" * 64, image_size=128)


def test_checkout_verification_rejects_dirty_and_wrong_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "src/dust3r/model.py"
    model.parent.mkdir(parents=True)
    model.write_text("# synthetic source\n")
    status = b""

    def git(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        data = native.SUPPORTED_CUT3R_REVISION.encode() if command[3] == "rev-parse" else status
        return subprocess.CompletedProcess(command, 0, stdout=data)

    monkeypatch.setattr(subprocess, "run", git)
    assert "src/dust3r/model.py" in native.verify_checkout(
        tmp_path, native.SUPPORTED_CUT3R_REVISION
    )
    status = b" M src/dust3r/model.py"
    with pytest.raises(ValueError, match="dirty"):
        native.verify_checkout(tmp_path, native.SUPPORTED_CUT3R_REVISION)
    with pytest.raises(ValueError, match="unsupported"):
        native.verify_checkout(tmp_path, "a" * 40)


def test_rgb_step_carries_state_and_encodes_only_current_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        shape = (1, 2, 3)

        def __init__(self, value: int) -> None:
            self.value = value

        def clone(self) -> Tensor:
            return Tensor(self.value)

        def expand(self, *args: int) -> Tensor:
            return self

    encoded: list[int] = []
    indices: list[int] = []

    class Model:
        pose_retriever = SimpleNamespace(mem=Tensor(0))

        def _encode_image(self, img: int, shape: Any) -> tuple[Any, ...]:
            encoded.append(img)
            return [Tensor(img)], Tensor(0), None

        def _init_state(self, feature: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
            return Tensor(0), pos

        def _forward_decoder_step(
            self,
            views: list[Any],
            i: int,
            feat: Tensor,
            pos: Any,
            shape: Any,
            initial: Tensor,
            initial_mem: Tensor,
            state: Tensor,
            state_pos: Any,
            memory: Tensor,
        ) -> tuple[dict[str, np.ndarray], tuple[Tensor, Tensor]]:
            assert len(views) == 2  # no prefix-sized container or re-encoding
            indices.append(i)
            updated = Tensor(state.value + feat.value)
            return {"points": np.array(updated.value)}, (updated, memory)

    runtime = native.Cut3RNativeRuntime.__new__(native.Cut3RNativeRuntime)
    runtime.torch = SimpleNamespace(
        inference_mode=nullcontext,
        autocast=lambda **kwargs: nullcontext(),
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None),
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)),
    )
    runtime.seed = 42
    runtime.model = Model()
    monkeypatch.setattr(
        runtime, "prepare_view", lambda p: {"img": int(p.name), "true_shape": (2, 3)}
    )
    monkeypatch.setattr(runtime, "decode_prediction", lambda prediction: prediction)
    runtime.reset()
    assert runtime.step(tmp_path / "1")["points"] == 1
    assert runtime.step(tmp_path / "2")["points"] == 3
    assert runtime.step(tmp_path / "3")["points"] == 6
    assert encoded == [1, 2, 3]
    assert indices == [0, 1, 1]
    runtime.reset()
    assert runtime.step(tmp_path / "1")["points"] == 1
    assert runtime.frame_count == 1


def test_build_variant_is_exact_and_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relative = "src/croco/models/curope/kernels.cu"
    member = tmp_path / relative
    member.parent.mkdir(parents=True)
    member.write_bytes(b"synthetic known kernel")
    model = tmp_path / "src/dust3r/model.py"
    model.parent.mkdir()
    model.write_text("# model\n")
    monkeypatch.setattr(
        native,
        "NATIVE_BUILD_COMPATIBILITY_SHA256",
        {
            relative: native.file_sha256(member),
        },
    )

    def git(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if command[3] == "rev-parse":
            data = native.SUPPORTED_CUT3R_REVISION.encode()
        elif command[3] == "status":
            data = f" M {relative}".encode()
        else:
            data = relative.encode() + b"\0"
        return subprocess.CompletedProcess(command, 0, stdout=data)

    monkeypatch.setattr(subprocess, "run", git)
    with pytest.raises(ValueError, match="dirty"):
        native.verify_checkout(tmp_path, native.SUPPORTED_CUT3R_REVISION)
    native.verify_checkout(
        tmp_path, native.SUPPORTED_CUT3R_REVISION, allow_native_build_compatibility=True
    )
    member.write_bytes(b"arbitrary different kernel")
    with pytest.raises(ValueError, match="dirty"):
        native.verify_checkout(
            tmp_path, native.SUPPORTED_CUT3R_REVISION, allow_native_build_compatibility=True
        )
