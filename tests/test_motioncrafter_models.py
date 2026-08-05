from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from prob4d.calibration_compatibility import motioncrafter_model_identifier
from prob4d.motioncrafter import MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
from prob4d.motioncrafter_models import (
    DEFAULT_IMAGE_VAE,
    MOTIONCRAFTER_MODEL_SET_SCHEMA,
    PinnedMotionCrafterModelSet,
    PinnedMotionCrafterRunConfig,
    _pinned_image_vae_proxy,
)
from prob4d.motioncrafter_safe import main as motioncrafter_main


def _local_model_roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    unet = tmp_path / "unet"
    vae = tmp_path / "vae"
    image_vae = tmp_path / "image_vae"
    base = tmp_path / "base"
    (unet / "unet_determ").mkdir(parents=True)
    (unet / "unet_determ" / "config.json").write_text(
        '{"kind":"unet"}',
        encoding="utf-8",
    )
    (unet / "weights.bin").write_bytes(b"unet weights")
    (vae / "geometry_motion_vae").mkdir(parents=True)
    (vae / "geometry_motion_vae" / "config.json").write_text(
        '{"kind":"vae"}',
        encoding="utf-8",
    )
    (image_vae / "vae").mkdir(parents=True)
    (image_vae / "vae" / "config.json").write_text(
        '{"kind":"image-vae"}',
        encoding="utf-8",
    )
    base.mkdir()
    (base / "model_index.json").write_text(
        '{"kind":"base"}',
        encoding="utf-8",
    )
    (base / "model.bin").write_bytes(b"base weights")
    return unet, vae, image_vae, base


def _build_config(
    model_set: PinnedMotionCrafterModelSet,
    tmp_path: Path,
) -> PinnedMotionCrafterRunConfig:
    return model_set.build_config(
        upstream_root=tmp_path / "MotionCrafter",
        video_path=tmp_path / "input.mp4",
        output_directory=tmp_path / "output",
        window_size=25,
        overlap=8,
        seed=11,
        seed_policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    )


def _manifest_config(config: PinnedMotionCrafterRunConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def test_local_model_set_is_path_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first_roots = _local_model_roots(tmp_path / "first")
    second_roots = _local_model_roots(tmp_path / "second")

    first = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference=first_roots[0],
        unet_revision=None,
        vae_reference=first_roots[1],
        vae_revision=None,
        image_vae_reference=first_roots[2],
        image_vae_revision=None,
        base_pipeline_reference=first_roots[3],
        base_pipeline_revision=None,
    )
    second = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference=second_roots[0],
        unet_revision=None,
        vae_reference=second_roots[1],
        vae_revision=None,
        image_vae_reference=second_roots[2],
        image_vae_revision=None,
        base_pipeline_reference=second_roots[3],
        base_pipeline_revision=None,
    )

    assert first.set_sha256 == second.set_sha256
    assert json.loads(first.manifest_json)["schema"] == MOTIONCRAFTER_MODEL_SET_SCHEMA

    (second_roots[3] / "model.bin").write_bytes(b"changed base weights")
    changed = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference=second_roots[0],
        unet_revision=None,
        vae_reference=second_roots[1],
        vae_revision=None,
        image_vae_reference=second_roots[2],
        image_vae_revision=None,
        base_pipeline_reference=second_roots[3],
        base_pipeline_revision=None,
    )
    assert changed.set_sha256 != first.set_sha256


def test_remote_model_sources_require_exact_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lacks an exact remote revision"):
        PinnedMotionCrafterModelSet.inspect(
            model_type="determ",
            unet_reference="TencentARC/MotionCrafter",
            unet_revision=None,
            vae_reference="TencentARC/MotionCrafter",
            vae_revision="a" * 40,
            image_vae_reference=DEFAULT_IMAGE_VAE,
            image_vae_revision="b" * 40,
            base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
            base_pipeline_revision="c" * 40,
        )

    with pytest.raises(ValueError, match="40- or 64-character"):
        PinnedMotionCrafterModelSet.inspect(
            model_type="determ",
            unet_reference="TencentARC/MotionCrafter",
            unet_revision="main",
            vae_reference="TencentARC/MotionCrafter",
            vae_revision="a" * 40,
            image_vae_reference=DEFAULT_IMAGE_VAE,
            image_vae_revision="b" * 40,
            base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
            base_pipeline_revision="c" * 40,
        )


def test_pinned_model_set_binds_all_nested_models_into_calibration_identity(
    tmp_path: Path,
) -> None:
    first = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference="TencentARC/MotionCrafter",
        unet_revision="a" * 40,
        vae_reference="TencentARC/MotionCrafter",
        vae_revision="b" * 40,
        image_vae_reference=DEFAULT_IMAGE_VAE,
        image_vae_revision="c" * 40,
        base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
        base_pipeline_revision="d" * 40,
    )
    changed_base = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference="TencentARC/MotionCrafter",
        unet_revision="a" * 40,
        vae_reference="TencentARC/MotionCrafter",
        vae_revision="b" * 40,
        image_vae_reference=DEFAULT_IMAGE_VAE,
        image_vae_revision="c" * 40,
        base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
        base_pipeline_revision="e" * 40,
    )
    changed_image_vae = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference="TencentARC/MotionCrafter",
        unet_revision="a" * 40,
        vae_reference="TencentARC/MotionCrafter",
        vae_revision="b" * 40,
        image_vae_reference=DEFAULT_IMAGE_VAE,
        image_vae_revision="f" * 40,
        base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
        base_pipeline_revision="d" * 40,
    )
    first_config = _build_config(first, tmp_path)
    changed_config = _build_config(changed_base, tmp_path)

    first_manifest = {
        "format_version": 1,
        "config": _manifest_config(first_config),
    }
    changed_manifest = {
        "format_version": 1,
        "config": _manifest_config(changed_config),
    }
    assert first_config.unet_path.startswith(
        f"{MOTIONCRAFTER_MODEL_SET_SCHEMA}:"
    )
    assert motioncrafter_model_identifier(first_manifest) != (
        motioncrafter_model_identifier(changed_manifest)
    )
    assert changed_image_vae.set_sha256 != first.set_sha256


def test_adapter_factory_rejects_another_model_set_before_gpu_imports(
    tmp_path: Path,
) -> None:
    first_roots = _local_model_roots(tmp_path / "first")
    second_roots = _local_model_roots(tmp_path / "second")
    (second_roots[0] / "weights.bin").write_bytes(b"different unet weights")
    first = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference=first_roots[0],
        unet_revision=None,
        vae_reference=first_roots[1],
        vae_revision=None,
        image_vae_reference=first_roots[2],
        image_vae_revision=None,
        base_pipeline_reference=first_roots[3],
        base_pipeline_revision=None,
    )
    second = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference=second_roots[0],
        unet_revision=None,
        vae_reference=second_roots[1],
        vae_revision=None,
        image_vae_reference=second_roots[2],
        image_vae_revision=None,
        base_pipeline_reference=second_roots[3],
        base_pipeline_revision=None,
    )
    config = _build_config(second, tmp_path)

    with pytest.raises(ValueError, match="adapter/config identity mismatch"):
        first.adapter_factory()(config)


def test_nested_image_vae_load_is_rebound_to_the_pinned_source() -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeAutoencoder:
        @staticmethod
        def from_pretrained(reference: object, **kwargs: object) -> object:
            calls.append((reference, kwargs))
            return object()

    source = PinnedMotionCrafterModelSet.inspect(
        model_type="determ",
        unet_reference="TencentARC/MotionCrafter",
        unet_revision="a" * 40,
        vae_reference="TencentARC/MotionCrafter",
        vae_revision="b" * 40,
        image_vae_reference=DEFAULT_IMAGE_VAE,
        image_vae_revision="c" * 40,
        base_pipeline_reference="stabilityai/stable-video-diffusion-img2vid-xt",
        base_pipeline_revision="d" * 40,
    ).image_vae
    proxy = _pinned_image_vae_proxy(
        source=source,
        original_autoencoder_class=FakeAutoencoder,
        cache_directory="/frozen/cache",
    )

    proxy.from_pretrained(
        DEFAULT_IMAGE_VAE,
        cache_dir="cache",
        subfolder="vae",
        revision=None,
        variant="fp16",
    )
    assert calls == [
        (
            DEFAULT_IMAGE_VAE,
            {
                "cache_dir": "/frozen/cache",
                "subfolder": "vae",
                "variant": "fp16",
                "revision": "c" * 40,
            },
        )
    ]

    with pytest.raises(ValueError, match="nested image-VAE load"):
        proxy.from_pretrained(
            DEFAULT_IMAGE_VAE,
            cache_dir="another-cache",
            subfolder="vae",
            revision=None,
            variant="fp16",
        )


def test_safe_cli_fails_closed_on_mutable_default_model_refs(
    tmp_path: Path,
    capsys: Any,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    with pytest.raises(SystemExit) as error:
        motioncrafter_main(
            [
                str(video),
                "--upstream-root",
                str(tmp_path / "MotionCrafter"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert error.value.code == 2
    assert "exact remote revision" in capsys.readouterr().err
