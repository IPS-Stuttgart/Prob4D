"""Executable provider-neutral prediction loading and conservative fusion.

The provider manifest already records exact prediction bytes, causal source
lineage, coordinate semantics, stochastic members, and shared-dependence groups.
This module turns that portable description into an executable runtime without
silently restoring MotionCrafter-specific assumptions.

Causal loading is deliberately two-stage: the manifest is validated before any
prediction payload is opened, then only payloads whose declared source interval
is admitted by the requested cutoff are hashed, decoded, and validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_json_number,
    require_mapping,
    require_sha256,
)
from .data import PredictionWindow
from .fusion import DEFAULT_FUSION_TILE_SIZE, FusedSequence, FusionMethod, fuse_windows
from .io import save_fused_prediction
from .prediction_provider_manifest import (
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    load_prediction_provider_manifest,
)
from .sim3 import Sim3
from .uncertainty import StructuredCovariance

FloatArray: TypeAlias = NDArray[np.float64]

PROVIDER_RUNTIME_GAUGE_SCHEMA: Final = "prob4d.provider-runtime-gauges"
PROVIDER_RUNTIME_GAUGE_VERSION: Final = 1
_GAUGE_CONFIG_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "manifest_artifact_id",
        "coordinate_semantics",
        "gauges",
    }
)
_SEQUENCE_GAUGE_KEY: Final = "__sequence__"
_CAMERA_TO_WORLD_KEY: Final = "__camera_to_world__"


def _materialized_payload_id(
    descriptor: PredictionPayloadDescriptorV1,
) -> str:
    payload_id = descriptor.payload_id
    if payload_id is None:
        raise ValueError("provider runtime payload ID is missing")
    return payload_id


def _materialized_manifest_id(manifest: PredictionProviderManifestV1) -> str:
    artifact_id = manifest.artifact_id
    if artifact_id is None:
        raise ValueError("provider runtime manifest artifact ID is missing")
    return artifact_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read prediction payload {path.name!r}") from error
    return digest.hexdigest()


def _safe_member(root: Path, relative_path: str, *, name: str) -> Path:
    if "\\" in relative_path:
        raise ValueError(f"{name} must be a POSIX relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    resolved_root = root.resolve()
    current = resolved_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the manifest directory") from error
    return candidate


def _validate_payload(
    manifest_root: Path,
    descriptor: PredictionPayloadDescriptorV1,
) -> PredictionWindow:
    member = _safe_member(
        manifest_root,
        descriptor.path,
        name=f"prediction payload {descriptor.window_id!r} path",
    )
    if not member.is_file():
        raise ValueError(f"prediction payload {descriptor.path!r} is missing")
    stat = member.stat()
    if stat.st_size != descriptor.byte_count:
        raise ValueError(f"prediction payload byte count mismatch for {descriptor.path!r}")
    if _sha256_file(member) != descriptor.sha256:
        raise ValueError(f"prediction payload SHA-256 mismatch for {descriptor.path!r}")
    window = PredictionWindow.from_npz(
        member,
        dense_storage_dtype=descriptor.dense_storage_dtype,
    )
    if window.window_id != descriptor.window_id:
        raise ValueError("prediction payload window identity changed")
    if tuple(int(value) for value in window.frame_indices) != descriptor.output_frame_ids:
        raise ValueError("prediction payload output-frame identities changed")
    if (window.scene_flow is not None) != descriptor.has_scene_flow:
        raise ValueError("prediction payload scene-flow declaration changed")
    if (window.ray_directions is not None) != descriptor.has_ray_directions:
        raise ValueError("prediction payload ray declaration changed")
    if window.dense_storage_dtype != descriptor.dense_storage_dtype:
        raise ValueError("prediction payload dense storage dtype changed")
    return window


def _selected_descriptors(
    manifest: PredictionProviderManifestV1,
    *,
    causal_frame_stop: int | None,
    payload_ids: Sequence[str] | None,
    window_ids: Sequence[str] | None,
) -> tuple[PredictionPayloadDescriptorV1, ...]:
    if payload_ids is not None and window_ids is not None:
        raise ValueError("select payload_ids or window_ids, not both")
    admitted = manifest.payloads
    if causal_frame_stop is not None:
        cutoff = require_exact_integer(
            causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        admitted = manifest.admitted_payloads(cutoff)
    admitted_payload_ids = {_materialized_payload_id(item) for item in admitted}
    admitted_window_ids = {item.window_id for item in admitted}

    if payload_ids is not None:
        requested = tuple(payload_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("payload_ids must be nonempty and unique")
        for index, payload_id in enumerate(requested):
            require_sha256(payload_id, name=f"payload_ids[{index}]")
        unknown = set(requested) - {_materialized_payload_id(item) for item in manifest.payloads}
        if unknown:
            raise ValueError(f"unknown prediction payload IDs: {sorted(unknown)}")
        inadmissible = set(requested) - admitted_payload_ids
        if inadmissible:
            raise ValueError(
                f"selected prediction payloads cross the causal cutoff: {sorted(inadmissible)}"
            )
        requested_set = set(requested)
        selected = tuple(
            item for item in manifest.payloads if _materialized_payload_id(item) in requested_set
        )
    elif window_ids is not None:
        requested_windows = tuple(window_ids)
        if not requested_windows or len(set(requested_windows)) != len(requested_windows):
            raise ValueError("window_ids must be nonempty and unique")
        if any(type(value) is not str or not value for value in requested_windows):
            raise ValueError("window_ids must contain nonempty genuine strings")
        unknown_windows = set(requested_windows) - {item.window_id for item in manifest.payloads}
        if unknown_windows:
            raise ValueError(f"unknown prediction window IDs: {sorted(unknown_windows)}")
        inadmissible_windows = set(requested_windows) - admitted_window_ids
        if inadmissible_windows:
            raise ValueError(
                "selected prediction windows cross the causal cutoff: "
                f"{sorted(inadmissible_windows)}"
            )
        requested_window_set = set(requested_windows)
        selected = tuple(
            item for item in manifest.payloads if item.window_id in requested_window_set
        )
    else:
        selected = tuple(admitted)

    if not selected:
        raise ValueError("no prediction payload is admitted by the runtime selection")
    return selected


def _dependent_alternative_pairs(
    descriptors: Sequence[PredictionPayloadDescriptorV1],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for first_index, first in enumerate(descriptors):
        for second in descriptors[first_index + 1 :]:
            if first.product_role == second.product_role == "independent-window":
                continue
            if first.view_id != second.view_id:
                continue
            if first.output_frame_ids != second.output_frame_ids:
                continue
            if first.stochastic_member_id != second.stochastic_member_id:
                continue
            if not set(first.dependence_group_ids).intersection(second.dependence_group_ids):
                continue
            pairs.append((first.window_id, second.window_id))
    return tuple(pairs)


@dataclass(frozen=True)
class ValidatedPredictionProvider:
    """One causally selected and byte-verified provider execution input."""

    manifest_path: Path
    manifest: PredictionProviderManifestV1
    descriptors: tuple[PredictionPayloadDescriptorV1, ...]
    windows: tuple[PredictionWindow, ...]
    causal_frame_stop: int | None
    dependent_alternatives_admitted: bool = False

    def __post_init__(self) -> None:
        path = Path(self.manifest_path).resolve()
        descriptors = tuple(self.descriptors)
        windows = tuple(self.windows)
        if not descriptors or len(descriptors) != len(windows):
            raise ValueError("provider runtime requires matching nonempty descriptors/windows")
        for descriptor, window in zip(descriptors, windows, strict=True):
            if descriptor.window_id != window.window_id:
                raise ValueError("provider runtime descriptor/window identity mismatch")
        object.__setattr__(self, "manifest_path", path)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "windows", windows)

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(item.window_id for item in self.descriptors)

    @property
    def payload_ids(self) -> tuple[str, ...]:
        return tuple(_materialized_payload_id(item) for item in self.descriptors)

    @property
    def dependence_group_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    group
                    for descriptor in self.descriptors
                    for group in descriptor.dependence_group_ids
                }
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "manifest_artifact_id": _materialized_manifest_id(self.manifest),
            "manifest_path": str(self.manifest_path),
            "sequence_id": self.manifest.sequence_id,
            "provider_family": self.manifest.provider_family,
            "provider_revision": self.manifest.provider_revision,
            "provider_run_id": self.manifest.provider_run_id,
            "model_set_id": self.manifest.model_set_id,
            "coordinate_semantics": self.manifest.coordinate_semantics,
            "causal_frame_stop": self.causal_frame_stop,
            "selected_payload_ids": list(self.payload_ids),
            "selected_window_ids": list(self.window_ids),
            "selected_payload_count": len(self.descriptors),
            "selected_output_frame_count": sum(
                len(item.frame_lineage) for item in self.descriptors
            ),
            "dependence_group_ids": list(self.dependence_group_ids),
            "dependent_alternatives_admitted": self.dependent_alternatives_admitted,
            "future_prediction_payloads_opened": 0,
        }


def load_prediction_provider_runtime(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int | None = None,
    payload_ids: Sequence[str] | None = None,
    window_ids: Sequence[str] | None = None,
    require_scene_flow: bool = False,
    allow_dependent_alternatives: bool = False,
) -> ValidatedPredictionProvider:
    """Validate a provider manifest and open only the selected causal payloads."""

    path = Path(manifest_path).resolve()
    manifest = load_prediction_provider_manifest(path)
    selected = _selected_descriptors(
        manifest,
        causal_frame_stop=causal_frame_stop,
        payload_ids=payload_ids,
        window_ids=window_ids,
    )
    alternatives = _dependent_alternative_pairs(selected)
    if alternatives and not allow_dependent_alternatives:
        formatted = ", ".join(f"{first}/{second}" for first, second in alternatives)
        raise ValueError(
            "selected payloads are dependent alternative constructions; select one "
            f"payload per alternative set or opt into exploratory fusion: {formatted}"
        )
    windows = tuple(_validate_payload(path.parent, item) for item in selected)
    if require_scene_flow and any(window.scene_flow is None for window in windows):
        missing = [window.window_id for window in windows if window.scene_flow is None]
        raise ValueError(f"selected provider payloads have no scene flow: {missing}")
    return ValidatedPredictionProvider(
        manifest_path=path,
        manifest=manifest,
        descriptors=selected,
        windows=windows,
        causal_frame_stop=causal_frame_stop,
        dependent_alternatives_admitted=bool(alternatives),
    )


def resolve_provider_gauges(
    provider: ValidatedPredictionProvider,
    *,
    per_payload_gauges: Mapping[str, Sim3] | None = None,
    sequence_gauge: Sim3 | None = None,
    camera_to_world: Sim3 | None = None,
    allow_unanchored_sequence_gauge: bool = False,
) -> dict[str, Sim3]:
    """Resolve coordinate semantics into one explicit transform per payload."""

    semantics = provider.manifest.coordinate_semantics
    expected = set(provider.window_ids)
    if semantics == "window-local-sim3":
        if sequence_gauge is not None or camera_to_world is not None:
            raise ValueError("window-local providers require per-payload gauges only")
        if per_payload_gauges is None:
            raise ValueError("window-local providers require one gauge per selected payload")
        supplied = set(per_payload_gauges)
        if supplied != expected:
            raise ValueError(
                "per-payload gauge IDs changed; "
                f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
            )
        gauges = dict(per_payload_gauges)
    elif semantics == "sequence-local-sim3":
        if per_payload_gauges is not None or camera_to_world is not None:
            raise ValueError("sequence-local providers require one shared sequence gauge")
        if sequence_gauge is None:
            if not allow_unanchored_sequence_gauge:
                raise ValueError(
                    "sequence-local providers require a metric sequence gauge or an "
                    "explicit exploratory unanchored acknowledgement"
                )
            sequence_gauge = Sim3.identity()
        gauges = {window_id: sequence_gauge for window_id in provider.window_ids}
    elif semantics == "camera-local-metric":
        if len({item.view_id for item in provider.descriptors}) != 1:
            raise ValueError("camera-local metric execution currently requires one selected view")
        if per_payload_gauges is not None or sequence_gauge is not None:
            raise ValueError("camera-local metric providers require camera_to_world only")
        if camera_to_world is None:
            raise ValueError("camera-local metric providers require camera_to_world")
        if not np.isclose(camera_to_world.scale, 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("camera_to_world must be rigid with unit metric scale")
        gauges = {window_id: camera_to_world for window_id in provider.window_ids}
    elif semantics == "metric-world":
        if any(
            value is not None for value in (per_payload_gauges, sequence_gauge, camera_to_world)
        ):
            raise ValueError("metric-world providers must not be transformed again")
        gauges = {window_id: Sim3.identity() for window_id in provider.window_ids}
    else:  # pragma: no cover - closed by the manifest contract.
        raise ValueError(f"unsupported provider coordinate semantics {semantics!r}")
    if any(not isinstance(value, Sim3) for value in gauges.values()):
        raise TypeError("provider gauges must contain Sim3 values")
    return gauges


def fuse_prediction_provider(
    provider: ValidatedPredictionProvider,
    point_uncertainties: Mapping[str, StructuredCovariance],
    *,
    method: FusionMethod = "covariance_intersection",
    flow_uncertainties: Mapping[str, StructuredCovariance] | None = None,
    gauge_covariances: Mapping[str, FloatArray] | None = None,
    per_payload_gauges: Mapping[str, Sim3] | None = None,
    sequence_gauge: Sim3 | None = None,
    camera_to_world: Sim3 | None = None,
    allow_unanchored_sequence_gauge: bool = False,
    fusion_tile_size: int = DEFAULT_FUSION_TILE_SIZE,
) -> FusedSequence:
    """Fuse selected provider payloads through the existing dense fusion core."""

    gauges = resolve_provider_gauges(
        provider,
        per_payload_gauges=per_payload_gauges,
        sequence_gauge=sequence_gauge,
        camera_to_world=camera_to_world,
        allow_unanchored_sequence_gauge=allow_unanchored_sequence_gauge,
    )
    point = dict(point_uncertainties)
    flow = None if flow_uncertainties is None else dict(flow_uncertainties)
    gauge_covariance = None if gauge_covariances is None else dict(gauge_covariances)
    return fuse_windows(
        list(provider.windows),
        gauges,
        point,
        method=method,
        flow_uncertainties=flow,
        gauge_covariances=gauge_covariance,
        fusion_tile_size=fusion_tile_size,
    )


def _isotropic_uncertainty(
    window: PredictionWindow,
    *,
    standard_deviation: float,
) -> StructuredCovariance:
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("exploratory standard deviation must be finite and positive")
    rays = np.asarray(window.rays(dtype=np.float64), dtype=np.float64)
    norms = np.linalg.norm(rays, axis=-1)
    rays[norms <= np.finfo(np.float64).eps] = np.asarray([0.0, 0.0, 1.0])
    variance = np.full(window.shape, standard_deviation**2, dtype=np.float64)
    return StructuredCovariance(
        ray_directions=rays,
        parallel_variance=variance,
        lateral_variance=variance,
    )


def _sim3_vector(value: object, *, name: str) -> Sim3:
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{name} must be a JSON array with seven numbers")
    vector = np.asarray(
        [require_json_number(item, name=f"{name}[{index}]") for index, item in enumerate(value)],
        dtype=np.float64,
    )
    return Sim3.from_vector(vector)


def load_provider_runtime_gauges(
    path: str | Path,
    provider: ValidatedPredictionProvider,
) -> tuple[dict[str, Sim3] | None, Sim3 | None, Sim3 | None, str]:
    """Load an exact manifest-bound gauge configuration for exploratory fusion."""

    source = Path(path).resolve()
    record = load_json_object(source, name="provider runtime gauge configuration")
    require_exact_fields(record, _GAUGE_CONFIG_FIELDS, name="gauge configuration")
    if record["schema"] != PROVIDER_RUNTIME_GAUGE_SCHEMA:
        raise ValueError("unsupported provider runtime gauge schema")
    schema_version = require_exact_integer(
        record["schema_version"],
        name="gauge configuration schema_version",
        minimum=1,
    )
    if schema_version != PROVIDER_RUNTIME_GAUGE_VERSION:
        raise ValueError("unsupported provider runtime gauge version")
    if require_sha256(
        record["manifest_artifact_id"],
        name="manifest_artifact_id",
    ) != _materialized_manifest_id(provider.manifest):
        raise ValueError("gauge configuration targets another provider manifest")
    if record["coordinate_semantics"] != provider.manifest.coordinate_semantics:
        raise ValueError("gauge configuration coordinate semantics changed")
    raw_gauges = require_mapping(record["gauges"], name="gauge configuration gauges")
    if any(type(key) is not str or not key for key in raw_gauges):
        raise ValueError("gauge configuration keys must be nonempty genuine strings")
    gauges = {
        key: _sim3_vector(value, name=f"gauges[{key!r}]") for key, value in raw_gauges.items()
    }
    semantics = provider.manifest.coordinate_semantics
    per_payload: dict[str, Sim3] | None = None
    sequence: Sim3 | None = None
    camera: Sim3 | None = None
    if semantics == "window-local-sim3":
        per_payload = gauges
    elif semantics == "sequence-local-sim3":
        if set(gauges) != {_SEQUENCE_GAUGE_KEY}:
            raise ValueError("sequence-local gauge configuration requires __sequence__")
        sequence = gauges[_SEQUENCE_GAUGE_KEY]
    elif semantics == "camera-local-metric":
        if set(gauges) != {_CAMERA_TO_WORLD_KEY}:
            raise ValueError("camera-local gauge configuration requires __camera_to_world__")
        camera = gauges[_CAMERA_TO_WORLD_KEY]
    elif semantics == "metric-world":
        if gauges:
            raise ValueError("metric-world gauge configuration must contain no gauges")
    return per_payload, sequence, camera, _sha256_file(source)


def fuse_provider_exploratory(
    provider: ValidatedPredictionProvider,
    *,
    point_standard_deviation_m: float,
    method: FusionMethod = "covariance_intersection",
    flow_standard_deviation_m: float | None = None,
    per_payload_gauges: Mapping[str, Sim3] | None = None,
    sequence_gauge: Sim3 | None = None,
    camera_to_world: Sim3 | None = None,
    allow_unanchored_sequence_gauge: bool = False,
    fusion_tile_size: int = DEFAULT_FUSION_TILE_SIZE,
) -> FusedSequence:
    """Run a labelled fixed-isotropic exploratory provider fusion baseline."""

    point_uncertainties = {
        window.window_id: _isotropic_uncertainty(
            window,
            standard_deviation=point_standard_deviation_m,
        )
        for window in provider.windows
    }
    flow_uncertainties = None
    if any(window.scene_flow is not None for window in provider.windows):
        selected_flow_std = (
            point_standard_deviation_m
            if flow_standard_deviation_m is None
            else flow_standard_deviation_m
        )
        flow_uncertainties = {
            window.window_id: _isotropic_uncertainty(
                window,
                standard_deviation=selected_flow_std,
            )
            for window in provider.windows
            if window.scene_flow is not None
        }
    return fuse_prediction_provider(
        provider,
        point_uncertainties,
        method=method,
        flow_uncertainties=flow_uncertainties,
        per_payload_gauges=per_payload_gauges,
        sequence_gauge=sequence_gauge,
        camera_to_world=camera_to_world,
        allow_unanchored_sequence_gauge=allow_unanchored_sequence_gauge,
        fusion_tile_size=fusion_tile_size,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction runtime",
        description="execute causally selected provider-neutral prediction payloads",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="open only selected causal payloads and print their execution summary",
    )
    fusion_parser = subparsers.add_parser(
        "fuse-exploratory",
        help="run a labelled fixed-isotropic exploratory fusion baseline",
    )
    for command_parser in (inspect_parser, fusion_parser):
        command_parser.add_argument("manifest")
        command_parser.add_argument("--causal-frame-stop", type=int)
        command_parser.add_argument("--payload-id", action="append")
        command_parser.add_argument("--window-id", action="append")
        command_parser.add_argument("--require-scene-flow", action="store_true")
        command_parser.add_argument(
            "--allow-dependent-alternatives",
            action="store_true",
            help="permit exploratory CI across explicitly dependent alternatives",
        )
    fusion_parser.add_argument("output")
    fusion_parser.add_argument(
        "--point-standard-deviation-m",
        type=float,
        required=True,
    )
    fusion_parser.add_argument("--flow-standard-deviation-m", type=float)
    fusion_parser.add_argument(
        "--method",
        choices=("uniform", "precision", "covariance_intersection"),
        default="covariance_intersection",
    )
    fusion_parser.add_argument("--gauge-config")
    fusion_parser.add_argument(
        "--allow-unanchored-sequence-gauge",
        action="store_true",
    )
    fusion_parser.add_argument(
        "--fusion-tile-size",
        type=int,
        default=DEFAULT_FUSION_TILE_SIZE,
    )
    fusion_parser.add_argument("--compressed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    provider = load_prediction_provider_runtime(
        arguments.manifest,
        causal_frame_stop=arguments.causal_frame_stop,
        payload_ids=arguments.payload_id,
        window_ids=arguments.window_id,
        require_scene_flow=arguments.require_scene_flow,
        allow_dependent_alternatives=arguments.allow_dependent_alternatives,
    )
    if arguments.command == "inspect":
        print(json.dumps(provider.summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "fuse-exploratory":
        per_payload = None
        sequence = None
        camera = None
        gauge_config_sha256 = None
        if arguments.gauge_config is not None:
            per_payload, sequence, camera, gauge_config_sha256 = load_provider_runtime_gauges(
                arguments.gauge_config, provider
            )
        sequence_result = fuse_provider_exploratory(
            provider,
            point_standard_deviation_m=arguments.point_standard_deviation_m,
            method=arguments.method,
            flow_standard_deviation_m=arguments.flow_standard_deviation_m,
            per_payload_gauges=per_payload,
            sequence_gauge=sequence,
            camera_to_world=camera,
            allow_unanchored_sequence_gauge=(arguments.allow_unanchored_sequence_gauge),
            fusion_tile_size=arguments.fusion_tile_size,
        )
        save_fused_prediction(
            arguments.output,
            sequence_result,
            method_id=(
                f"provider-runtime-exploratory:{provider.manifest.provider_family}:"
                f"{arguments.method}"
            ),
            fusion_method=arguments.method,
            metadata={
                **provider.summary(),
                "claim_bearing": False,
                "uncertainty_semantics": "fixed-isotropic-exploratory-v1",
                "point_standard_deviation_m": arguments.point_standard_deviation_m,
                "flow_standard_deviation_m": arguments.flow_standard_deviation_m,
                "gauge_config_sha256": gauge_config_sha256,
            },
            compressed=arguments.compressed,
        )
        print(
            json.dumps(
                {
                    **provider.summary(),
                    "output": str(Path(arguments.output).resolve()),
                    "fusion_method": arguments.method,
                    "claim_bearing": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    parser.error("unsupported provider runtime command")
    return 2


__all__ = [
    "PROVIDER_RUNTIME_GAUGE_SCHEMA",
    "PROVIDER_RUNTIME_GAUGE_VERSION",
    "ValidatedPredictionProvider",
    "fuse_prediction_provider",
    "fuse_provider_exploratory",
    "load_prediction_provider_runtime",
    "load_provider_runtime_gauges",
    "main",
    "resolve_provider_gauges",
]


if __name__ == "__main__":
    raise SystemExit(main())
