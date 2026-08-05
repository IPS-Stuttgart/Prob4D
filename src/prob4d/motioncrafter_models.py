"""Immutable model-source binding for MotionCrafter inference."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from .motioncrafter import MotionCrafterAdapter, MotionCrafterRunConfig

MOTIONCRAFTER_MODEL_SOURCE_SCHEMA: Final = "prob4d.motioncrafter-model-source.v1"
MOTIONCRAFTER_MODEL_SET_SCHEMA: Final = "prob4d.motioncrafter-model-set.v2"
DEFAULT_BASE_PIPELINE: Final = "stabilityai/stable-video-diffusion-img2vid-xt"
DEFAULT_IMAGE_VAE: Final = "stable-diffusion-v1-5/stable-diffusion-v1-5"

ModelSourceKind = Literal["local_snapshot", "huggingface_revision"]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read model snapshot member {path}") from error
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_revision(value: object, *, name: str) -> str:
    revision = str(value)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"{name} must be an exact lowercase 40- or 64-character revision"
        )
    return revision


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _module_descriptor() -> dict[str, object]:
    path = Path(__file__).resolve()
    return {
        "module": "prob4d.motioncrafter_models",
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _local_tree_descriptor(
    root: Path,
    *,
    role: str,
    required_members: tuple[str, ...],
) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{role} model source is not a local directory: {root}")
    for relative in required_members:
        required = root / relative
        if not required.exists():
            raise ValueError(
                f"{role} local snapshot lacks required member {relative!r}"
            )

    members: list[dict[str, object]] = []
    for candidate in sorted(
        root.rglob("*"),
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        relative = candidate.relative_to(root)
        if ".git" in relative.parts:
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(
                f"{role} local snapshot contains a non-regular member: {relative}"
            )
        members.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
        )
    if not members:
        raise ValueError(f"{role} local snapshot contains no regular files")

    portable = {
        "schema": MOTIONCRAFTER_MODEL_SOURCE_SCHEMA,
        "role": role,
        "kind": "local_snapshot",
        "members": members,
    }
    return {
        "tree_sha256": hashlib.sha256(_canonical_json(portable)).hexdigest(),
        "file_count": len(members),
        "bytes": sum(int(member["bytes"]) for member in members),
    }


@dataclass(frozen=True)
class PinnedModelSource:
    """One local content-addressed snapshot or exact remote revision."""

    role: str
    kind: ModelSourceKind
    runtime_reference: str
    revision: str | None = None
    tree_sha256: str | None = None
    file_count: int | None = None
    bytes: int | None = None

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        reference = str(self.runtime_reference).strip()
        if not role or not reference:
            raise ValueError("model-source role and reference must be nonempty")
        if self.kind == "local_snapshot":
            if self.revision is not None:
                raise ValueError("local model snapshots cannot declare a remote revision")
            tree_sha = _require_sha256(
                self.tree_sha256,
                name=f"{role} local snapshot tree_sha256",
            )
            file_count = _positive_integer(
                self.file_count,
                name=f"{role} local snapshot file_count",
            )
            byte_count = _positive_integer(
                self.bytes,
                name=f"{role} local snapshot bytes",
            )
            object.__setattr__(self, "tree_sha256", tree_sha)
            object.__setattr__(self, "file_count", file_count)
            object.__setattr__(self, "bytes", byte_count)
        elif self.kind == "huggingface_revision":
            object.__setattr__(
                self,
                "revision",
                _require_revision(
                    self.revision,
                    name=f"{role} Hugging Face revision",
                ),
            )
            if any(
                value is not None
                for value in (self.tree_sha256, self.file_count, self.bytes)
            ):
                raise ValueError(
                    "remote model revisions cannot carry local-tree statistics"
                )
        else:
            raise ValueError(f"unsupported model-source kind {self.kind!r}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "runtime_reference", reference)

    @classmethod
    def inspect(
        cls,
        reference: str | Path,
        *,
        role: str,
        revision: str | None,
        required_members: tuple[str, ...] = (),
    ) -> PinnedModelSource:
        """Resolve a local snapshot or require an exact remote revision."""

        candidate = Path(reference).expanduser()
        if candidate.exists():
            if revision is not None:
                raise ValueError(
                    f"{role} is local; do not combine a local snapshot with --{role}-revision"
                )
            descriptor = _local_tree_descriptor(
                candidate,
                role=role,
                required_members=required_members,
            )
            return cls(
                role=role,
                kind="local_snapshot",
                runtime_reference=str(candidate.resolve()),
                tree_sha256=str(descriptor["tree_sha256"]),
                file_count=int(descriptor["file_count"]),
                bytes=int(descriptor["bytes"]),
            )
        if revision is None:
            raise ValueError(
                f"{role} source {str(reference)!r} is not a local snapshot and lacks "
                "an exact remote revision"
            )
        return cls(
            role=role,
            kind="huggingface_revision",
            runtime_reference=str(reference),
            revision=revision,
        )

    def portable_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": MOTIONCRAFTER_MODEL_SOURCE_SCHEMA,
            "role": self.role,
            "kind": self.kind,
        }
        if self.kind == "local_snapshot":
            record.update(
                tree_sha256=self.tree_sha256,
                file_count=self.file_count,
                bytes=self.bytes,
            )
        else:
            record.update(
                repository=self.runtime_reference,
                revision=self.revision,
            )
        return record

    def from_pretrained_arguments(self) -> tuple[str, dict[str, object]]:
        kwargs: dict[str, object] = {}
        if self.kind == "huggingface_revision":
            kwargs["revision"] = self.revision
        return self.runtime_reference, kwargs


@dataclass(frozen=True)
class PinnedMotionCrafterRunConfig(MotionCrafterRunConfig):
    """MotionCrafter configuration with a portable immutable model-set identity."""

    base_pipeline_path: str = ""
    model_source_schema: str = MOTIONCRAFTER_MODEL_SET_SCHEMA
    model_source_set_sha256: str = ""
    model_source_manifest_json: str = ""
    model_loader_module_sha256: str = ""
    model_loader_module_bytes: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model_source_schema != MOTIONCRAFTER_MODEL_SET_SCHEMA:
            raise ValueError("unsupported MotionCrafter model-set schema")
        set_sha = _require_sha256(
            self.model_source_set_sha256,
            name="MotionCrafter model-set SHA-256",
        )
        loader_sha = _require_sha256(
            self.model_loader_module_sha256,
            name="MotionCrafter model-loader module SHA-256",
        )
        loader_bytes = _positive_integer(
            self.model_loader_module_bytes,
            name="MotionCrafter model-loader module bytes",
        )
        try:
            manifest = json.loads(self.model_source_manifest_json)
        except json.JSONDecodeError as error:
            raise ValueError("MotionCrafter model-source manifest is invalid JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("MotionCrafter model-source manifest must be an object")
        if manifest.get("schema") != MOTIONCRAFTER_MODEL_SET_SCHEMA:
            raise ValueError("MotionCrafter model-source manifest schema changed")
        if manifest.get("model_type") != self.model_type:
            raise ValueError("MotionCrafter model-source manifest model_type changed")
        loader = manifest.get("loader_module")
        if not isinstance(loader, Mapping):
            raise ValueError("MotionCrafter model-source manifest lacks loader metadata")
        if (
            loader.get("sha256") != loader_sha
            or loader.get("bytes") != loader_bytes
        ):
            raise ValueError("MotionCrafter model-loader identity changed")
        if hashlib.sha256(_canonical_json(manifest)).hexdigest() != set_sha:
            raise ValueError("MotionCrafter model-set digest mismatch")
        identity = f"{MOTIONCRAFTER_MODEL_SET_SCHEMA}:{set_sha}"
        if self.unet_path != f"{identity}#unet":
            raise ValueError("MotionCrafter UNet identity changed")
        if self.vae_path != f"{identity}#geometry-motion-vae":
            raise ValueError("MotionCrafter VAE identity changed")
        if self.base_pipeline_path != f"{identity}#base-video-pipeline":
            raise ValueError("MotionCrafter base-pipeline identity changed")
        object.__setattr__(self, "model_source_set_sha256", set_sha)
        object.__setattr__(self, "model_loader_module_sha256", loader_sha)
        object.__setattr__(self, "model_loader_module_bytes", loader_bytes)
        object.__setattr__(
            self,
            "model_source_manifest_json",
            _canonical_json(manifest).decode("utf-8"),
        )


@dataclass(frozen=True)
class PinnedMotionCrafterModelSet:
    """The four immutable model sources used by one MotionCrafter run."""

    model_type: str
    unet: PinnedModelSource
    vae: PinnedModelSource
    image_vae: PinnedModelSource
    base_pipeline: PinnedModelSource
    loader_module_sha256: str
    loader_module_bytes: int
    set_sha256: str
    manifest_json: str

    @classmethod
    def inspect(
        cls,
        *,
        model_type: str,
        unet_reference: str | Path,
        unet_revision: str | None,
        vae_reference: str | Path,
        vae_revision: str | None,
        image_vae_reference: str | Path,
        image_vae_revision: str | None,
        base_pipeline_reference: str | Path,
        base_pipeline_revision: str | None,
    ) -> PinnedMotionCrafterModelSet:
        """Inspect all model sources and derive one portable set identity."""

        if model_type not in {"determ", "diff"}:
            raise ValueError("model_type must be 'determ' or 'diff'")
        unet = PinnedModelSource.inspect(
            unet_reference,
            role="unet",
            revision=unet_revision,
            required_members=(f"unet_{model_type}",),
        )
        vae = PinnedModelSource.inspect(
            vae_reference,
            role="vae",
            revision=vae_revision,
            required_members=("geometry_motion_vae",),
        )
        image_vae = PinnedModelSource.inspect(
            image_vae_reference,
            role="image-vae",
            revision=image_vae_revision,
            required_members=("vae/config.json",),
        )
        base = PinnedModelSource.inspect(
            base_pipeline_reference,
            role="base-pipeline",
            revision=base_pipeline_revision,
            required_members=("model_index.json",),
        )
        loader = _module_descriptor()
        manifest = {
            "schema": MOTIONCRAFTER_MODEL_SET_SCHEMA,
            "model_type": model_type,
            "sources": {
                "unet": unet.portable_record(),
                "vae": vae.portable_record(),
                "image_vae": image_vae.portable_record(),
                "base_pipeline": base.portable_record(),
            },
            "loader_module": loader,
        }
        manifest_json = _canonical_json(manifest).decode("utf-8")
        set_sha = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        return cls(
            model_type=model_type,
            unet=unet,
            vae=vae,
            image_vae=image_vae,
            base_pipeline=base,
            loader_module_sha256=str(loader["sha256"]),
            loader_module_bytes=int(loader["bytes"]),
            set_sha256=set_sha,
            manifest_json=manifest_json,
        )

    def build_config(self, **kwargs: Any) -> PinnedMotionCrafterRunConfig:
        """Create the portable config consumed by the crash-safe runner."""

        identity = f"{MOTIONCRAFTER_MODEL_SET_SCHEMA}:{self.set_sha256}"
        forbidden = {"model_type", "unet_path", "vae_path", "base_pipeline_path"} & set(
            kwargs
        )
        if forbidden:
            raise ValueError(
                "pinned model identities are derived, not caller supplied: "
                + ", ".join(sorted(forbidden))
            )
        return PinnedMotionCrafterRunConfig(
            **kwargs,
            model_type=self.model_type,
            unet_path=f"{identity}#unet",
            vae_path=f"{identity}#geometry-motion-vae",
            base_pipeline_path=f"{identity}#base-video-pipeline",
            model_source_set_sha256=self.set_sha256,
            model_source_manifest_json=self.manifest_json,
            model_loader_module_sha256=self.loader_module_sha256,
            model_loader_module_bytes=self.loader_module_bytes,
        )

    def adapter_factory(
        self,
    ) -> Callable[[MotionCrafterRunConfig], MotionCrafterAdapter]:
        """Return a factory that refuses a config from another model set."""

        expected = self.set_sha256

        def factory(config: MotionCrafterRunConfig) -> MotionCrafterAdapter:
            if not isinstance(config, PinnedMotionCrafterRunConfig):
                raise ValueError("pinned model adapter requires a pinned run config")
            if config.model_source_set_sha256 != expected:
                raise ValueError("pinned model adapter/config identity mismatch")
            return PinnedMotionCrafterAdapter(config, model_set=self)

        return factory


def _pinned_image_vae_proxy(
    *,
    source: PinnedModelSource,
    original_autoencoder_class: Any,
    cache_directory: str,
) -> type:
    """Replace MotionCrafter's hidden mutable image-VAE load with a pinned one."""

    class PinnedImageVaeProxy:
        @staticmethod
        def from_pretrained(reference: object, **kwargs: Any) -> Any:
            if str(reference) != DEFAULT_IMAGE_VAE:
                raise ValueError("MotionCrafter changed its nested image-VAE reference")
            expected = {
                "cache_dir": "cache",
                "subfolder": "vae",
                "revision": None,
                "variant": "fp16",
            }
            if kwargs != expected:
                raise ValueError("MotionCrafter changed its nested image-VAE load")
            runtime_reference, pinned_kwargs = source.from_pretrained_arguments()
            return original_autoencoder_class.from_pretrained(
                runtime_reference,
                cache_dir=cache_directory,
                subfolder="vae",
                variant="fp16",
                **pinned_kwargs,
            )

    return PinnedImageVaeProxy


class PinnedMotionCrafterAdapter(MotionCrafterAdapter):
    """MotionCrafter adapter loading only the model sources bound above."""

    def __init__(
        self,
        config: PinnedMotionCrafterRunConfig,
        *,
        model_set: PinnedMotionCrafterModelSet,
    ) -> None:
        if model_set.set_sha256 != config.model_source_set_sha256:
            raise ValueError("model-set identity differs from the pinned run config")
        self.config = config
        upstream_root = config.upstream_root.resolve()
        if not (upstream_root / "motioncrafter").is_dir():
            raise ValueError(f"{upstream_root} is not a MotionCrafter checkout")
        sys.path.insert(0, str(upstream_root))

        try:
            import torch
            import torch.nn.functional as functional
            from decord import VideoReader, cpu
            from diffusers import AutoencoderKL
            from diffusers.training_utils import set_seed
            from motioncrafter import (
                MotionCrafterDetermPipeline,
                MotionCrafterDiffPipeline,
                UNetSpatioTemporalConditionModelVid2vid,
                UnifyAutoencoderKL,
                geometry_motion_vae,
            )
        except ImportError as error:
            raise RuntimeError(
                "Run prob4d-motioncrafter inside the upstream MotionCrafter environment"
            ) from error

        self.torch = torch
        self.functional = functional
        self.VideoReader = VideoReader
        self.cpu = cpu
        self.set_seed = set_seed

        vae_reference, vae_kwargs = model_set.vae.from_pretrained_arguments()
        original_image_vae_class = geometry_motion_vae.AutoencoderKL
        geometry_motion_vae.AutoencoderKL = _pinned_image_vae_proxy(
            source=model_set.image_vae,
            original_autoencoder_class=original_image_vae_class,
            cache_directory=config.cache_directory,
        )
        try:
            self.geometry_motion_vae = (
                UnifyAutoencoderKL.from_pretrained(
                    vae_reference,
                    subfolder="geometry_motion_vae",
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    cache_dir=config.cache_directory,
                    **vae_kwargs,
                )
                .requires_grad_(False)
                .to("cuda", dtype=torch.float32)
            )
        finally:
            geometry_motion_vae.AutoencoderKL = original_image_vae_class

        unet_reference, unet_kwargs = model_set.unet.from_pretrained_arguments()
        unet = (
            UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
                unet_reference,
                subfolder=(
                    "unet_diff" if config.model_type == "diff" else "unet_determ"
                ),
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir=config.cache_directory,
                **unet_kwargs,
            )
            .requires_grad_(False)
            .to("cuda", dtype=torch.float16)
        )
        pipeline_class = (
            MotionCrafterDiffPipeline
            if config.model_type == "diff"
            else MotionCrafterDetermPipeline
        )
        base_reference, base_kwargs = (
            model_set.base_pipeline.from_pretrained_arguments()
        )
        self.pipeline = pipeline_class.from_pretrained(
            base_reference,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.cache_directory,
            **base_kwargs,
        ).to("cuda")
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipeline.enable_attention_slicing()
        self.video_vae_class = AutoencoderKL


__all__ = [
    "DEFAULT_BASE_PIPELINE",
    "DEFAULT_IMAGE_VAE",
    "MOTIONCRAFTER_MODEL_SET_SCHEMA",
    "MOTIONCRAFTER_MODEL_SOURCE_SCHEMA",
    "PinnedModelSource",
    "PinnedMotionCrafterAdapter",
    "PinnedMotionCrafterModelSet",
    "PinnedMotionCrafterRunConfig",
]
