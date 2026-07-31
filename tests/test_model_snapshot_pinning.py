from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.calibration_compatibility import (
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1,
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V2,
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V3,
    load_prediction_calibration_target,
    motioncrafter_model_identifier,
)
from prob4d.motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MotionCrafterRunConfig,
)


def _historical_manifest(*, derived: bool = False) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "determ",
        "unet_path": "TencentARC/MotionCrafter",
        "vae_path": "TencentARC/MotionCrafter",
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "decode_chunk_size": 25,
        "low_memory_usage": False,
        "seed": 42,
        "frame_stride": 1,
        "height": 320,
        "width": 640,
        "window_size": 25,
        "overlap": 8,
    }
    if derived:
        config["seed_policy"] = MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
    return {
        "format_version": 1,
        "motioncrafter_commit": "a" * 40,
        "config": config,
    }


def _snapshot_manifest() -> dict[str, object]:
    manifest = _historical_manifest()
    config = manifest["config"]
    assert isinstance(config, dict)
    config.update(
        {
            "seed_policy": MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
            "base_model_path": "stabilityai/stable-video-diffusion-img2vid-xt",
            "unet_revision": "1" * 40,
            "vae_revision": "2" * 40,
            "base_model_revision": "3" * 40,
        }
    )
    return manifest


def test_run_config_validates_exact_snapshot_revisions() -> None:
    base = {
        "upstream_root": Path("upstream"),
        "video_path": Path("video.mp4"),
        "output_directory": Path("output"),
    }
    with pytest.raises(ValueError, match="unet_revision"):
        MotionCrafterRunConfig(**base, unet_revision="main")
    with pytest.raises(ValueError, match="base_model_path"):
        MotionCrafterRunConfig(**base, base_model_path="  ")

    config = MotionCrafterRunConfig(
        **base,
        unet_revision="1" * 40,
        vae_revision="2" * 40,
        base_model_revision="3" * 64,
    )
    assert config.unet_revision == "1" * 40
    assert config.base_model_revision == "3" * 64


def test_snapshot_identifier_binds_all_three_model_revisions() -> None:
    manifest = _snapshot_manifest()
    identifier = motioncrafter_model_identifier(manifest)
    assert identifier.startswith(f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V3}:")

    for key in ("unet_revision", "vae_revision", "base_model_revision"):
        changed = json.loads(json.dumps(manifest))
        changed["config"][key] = "4" * 40
        assert motioncrafter_model_identifier(changed) != identifier


def test_snapshot_identifier_fails_closed_on_partial_or_mutable_revisions() -> None:
    missing = _snapshot_manifest()
    config = missing["config"]
    assert isinstance(config, dict)
    del config["base_model_revision"]
    with pytest.raises(ValueError, match="missing model-identifier settings"):
        motioncrafter_model_identifier(missing)

    mutable = _snapshot_manifest()
    config = mutable["config"]
    assert isinstance(config, dict)
    config["unet_revision"] = "main"
    with pytest.raises(ValueError, match="requires exact model revisions"):
        motioncrafter_model_identifier(mutable)


def test_historical_identifier_bytes_remain_versioned_compatibility_surfaces() -> None:
    assert motioncrafter_model_identifier(_historical_manifest()).startswith(
        f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1}:"
    )
    assert motioncrafter_model_identifier(
        _historical_manifest(derived=True)
    ).startswith(f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V2}:")


def test_claim_bearing_target_requires_pinned_new_model_config(tmp_path: Path) -> None:
    manifest = _snapshot_manifest()
    config = manifest["config"]
    assert isinstance(config, dict)
    config["unet_revision"] = None
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="requires exact model revisions"):
        load_prediction_calibration_target(path)

    config["unet_revision"] = "1" * 40
    path.write_text(json.dumps(manifest), encoding="utf-8")
    target = load_prediction_calibration_target(path)
    assert target.model_identifier.startswith(
        f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V3}:"
    )


def test_adapter_passes_exact_revisions_to_every_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from prob4d.motioncrafter import MotionCrafterAdapter

    upstream = tmp_path / "upstream"
    (upstream / "motioncrafter").mkdir(parents=True)
    calls: dict[str, tuple[str, dict[str, object]]] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object):
            calls[cls.__name__] = (path, dict(kwargs))
            return cls()

        def requires_grad_(self, _: bool):
            return self

        def to(self, *args: object, **kwargs: object):
            del args, kwargs
            return self

    class FakeGeometryVAE(FakeModel):
        pass

    class FakeUNet(FakeModel):
        pass

    class FakePipeline(FakeModel):
        def enable_xformers_memory_efficient_attention(self) -> None:
            return None

        def enable_attention_slicing(self) -> None:
            return None

    torch = types.ModuleType("torch")
    torch.__path__ = []  # type: ignore[attr-defined]
    torch.float32 = object()  # type: ignore[attr-defined]
    torch.float16 = object()  # type: ignore[attr-defined]
    functional = types.ModuleType("torch.nn.functional")
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__path__ = []  # type: ignore[attr-defined]
    torch_nn.functional = functional  # type: ignore[attr-defined]
    torch.nn = torch_nn  # type: ignore[attr-defined]

    decord = types.ModuleType("decord")
    decord.VideoReader = object  # type: ignore[attr-defined]
    decord.cpu = lambda _: None  # type: ignore[attr-defined]
    diffusers = types.ModuleType("diffusers")
    diffusers.__path__ = []  # type: ignore[attr-defined]
    diffusers.AutoencoderKL = FakeModel  # type: ignore[attr-defined]
    training_utils = types.ModuleType("diffusers.training_utils")
    training_utils.set_seed = lambda _: None  # type: ignore[attr-defined]
    upstream_module = types.ModuleType("motioncrafter")
    upstream_module.MotionCrafterDetermPipeline = FakePipeline  # type: ignore[attr-defined]
    upstream_module.MotionCrafterDiffPipeline = FakePipeline  # type: ignore[attr-defined]
    upstream_module.UNetSpatioTemporalConditionModelVid2vid = FakeUNet  # type: ignore[attr-defined]
    upstream_module.UnifyAutoencoderKL = FakeGeometryVAE  # type: ignore[attr-defined]

    for name, module in {
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.nn.functional": functional,
        "decord": decord,
        "diffusers": diffusers,
        "diffusers.training_utils": training_utils,
        "motioncrafter": upstream_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    config = MotionCrafterRunConfig(
        upstream_root=upstream,
        video_path=tmp_path / "video.mp4",
        output_directory=tmp_path / "output",
        unet_path="org/unet",
        vae_path="org/vae",
        base_model_path="org/base",
        unet_revision="1" * 40,
        vae_revision="2" * 40,
        base_model_revision="3" * 40,
    )
    MotionCrafterAdapter(config)

    assert calls["FakeGeometryVAE"][0] == "org/vae"
    assert calls["FakeGeometryVAE"][1]["revision"] == "2" * 40
    assert calls["FakeUNet"][0] == "org/unet"
    assert calls["FakeUNet"][1]["revision"] == "1" * 40
    assert calls["FakePipeline"][0] == "org/base"
    assert calls["FakePipeline"][1]["revision"] == "3" * 40
