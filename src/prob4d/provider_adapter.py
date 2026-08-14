"""Typed adapter SDK for materializing provider-neutral prediction manifests.

The adapter boundary intentionally stops at canonical :class:`PredictionWindow`
values. Provider inference, model loading, checkpoint management, and native
cache formats remain outside Prob4D; this module owns deterministic conversion,
causal-lineage validation, no-clobber payload publication, and construction of
``PredictionProviderManifestV1`` artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from ._atomic_file import atomic_write_text, publish_temporary_file
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
    require_string_sequence,
)
from .data import PredictionWindow
from .prediction_provider_manifest import (
    COORDINATE_SEMANTICS,
    FLOW_SEMANTICS,
    POINT_SEMANTICS,
    PRODUCT_ROLES,
    RAY_SEMANTICS,
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)

PROVIDER_ADAPTER_IDENTITY_SCHEMA: Final = "prob4d.provider-adapter-identity"
PROVIDER_ADAPTER_REQUEST_SCHEMA: Final = "prob4d.provider-adapter-request"
PROVIDER_ADAPTER_VERSION: Final = 1
PROVIDER_ADAPTER_CLAIM_BOUNDARY: Final = (
    "This adapter contract establishes deterministic conversion into canonical "
    "provider-neutral prediction payloads, declared causal lineage, dependence "
    "semantics, and exact provider/model/loader identities. It does not establish "
    "provider accuracy, uncertainty calibration, statistical independence, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of "
    "the art."
)

_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "adapter_name",
        "adapter_version",
        "adapter_implementation_id",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "provider_run_id",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "uses_truth",
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
        "metadata",
        "claim_boundary",
        "provider_adapter_identity_id",
    }
)
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "sequence_id",
        "causal_frame_stop",
        "input_family_id",
        "input_snapshot_id",
        "target_payloads_opened",
        "target_outcomes_opened",
        "metadata",
        "claim_boundary",
        "provider_adapter_request_id",
    }
)
_RESERVED_METADATA_FIELDS: Final = frozenset(
    {
        "source_adapter",
        "provider_adapter_identity_id",
        "provider_adapter_request_id",
        "provider_adapter_contract_version",
        "provider_adapter_input_family_id",
        "provider_adapter_input_snapshot_id",
        "provider_adapter_causal_frame_stop",
        "provider_adapter_request_metadata",
        "uses_truth",
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _repository(value: object, *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result.count("/") != 1 or result.startswith("/") or result.endswith("/"):
        raise ValueError(f"{name} must use canonical owner/name form")
    return result


def _choice(value: object, choices: Sequence[str], *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    path = require_exact_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def _resolved_output_member(root: Path, relative_path: str, *, name: str) -> Path:
    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the manifest directory") from error
    return candidate


def _relative_member(path: Path, *, root: Path, name: str) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve(strict=False)
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} must lie inside the manifest directory") from error
    return _safe_relative_path(relative.as_posix(), name=name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_arrays_equal(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None:
        return first is second
    return np.array_equal(first, second)


def _windows_equal(first: PredictionWindow, second: PredictionWindow) -> bool:
    return (
        first.window_id == second.window_id
        and first.dense_storage_dtype == second.dense_storage_dtype
        and np.array_equal(first.frame_indices, second.frame_indices)
        and np.array_equal(first.point_map, second.point_map)
        and np.array_equal(first.valid_mask, second.valid_mask)
        and _optional_arrays_equal(first.scene_flow, second.scene_flow)
        and _optional_arrays_equal(first.deform_mask, second.deform_mask)
        and _optional_arrays_equal(first.ray_directions, second.ray_directions)
    )


def _write_window_atomically(path: Path, window: PredictionWindow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("canonical adapter payload destination must not be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        window.to_npz(temporary)
        try:
            publish_temporary_file(temporary, path, overwrite=False)
        except FileExistsError:
            try:
                existing = PredictionWindow.from_npz(
                    path,
                    dense_storage_dtype=window.dense_storage_dtype,
                )
            except (KeyError, OSError, ValueError) as error:
                raise ValueError(
                    f"existing adapter payload {path.name!r} is not a canonical window"
                ) from error
            if not _windows_equal(existing, window):
                raise ValueError(
                    f"refusing to replace different adapter payload {path.name!r}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ProviderAdapterIdentityV1:
    """Exact provider, model, loader, and adapter implementation identity."""

    adapter_name: str
    adapter_version: int
    adapter_implementation_id: str
    provider_family: str
    provider_repository: str
    provider_revision: str
    provider_run_id: str
    model_set_id: str
    loader_id: str
    coordinate_semantics: str
    point_semantics: str
    flow_semantics: str
    ray_semantics: str
    uses_truth: bool = False
    uses_target_outcomes: bool = False
    uses_downstream_physical_innovation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_adapter_identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adapter_name",
            require_exact_string(self.adapter_name, name="adapter_name"),
        )
        object.__setattr__(
            self,
            "adapter_version",
            require_exact_integer(
                self.adapter_version,
                name="adapter_version",
                minimum=1,
            ),
        )
        for name in (
            "adapter_implementation_id",
            "provider_run_id",
            "model_set_id",
            "loader_id",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "provider_family",
            require_exact_string(self.provider_family, name="provider_family"),
        )
        object.__setattr__(
            self,
            "provider_repository",
            _repository(self.provider_repository, name="provider_repository"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            require_revision(self.provider_revision, name="provider_revision"),
        )
        object.__setattr__(
            self,
            "coordinate_semantics",
            _choice(
                self.coordinate_semantics,
                COORDINATE_SEMANTICS,
                name="coordinate_semantics",
            ),
        )
        object.__setattr__(
            self,
            "point_semantics",
            _choice(self.point_semantics, POINT_SEMANTICS, name="point_semantics"),
        )
        object.__setattr__(
            self,
            "flow_semantics",
            _choice(self.flow_semantics, FLOW_SEMANTICS, name="flow_semantics"),
        )
        object.__setattr__(
            self,
            "ray_semantics",
            _choice(self.ray_semantics, RAY_SEMANTICS, name="ray_semantics"),
        )
        for name in (
            "uses_truth",
            "uses_target_outcomes",
            "uses_downstream_physical_innovation",
        ):
            value = _strict_bool(getattr(self, name), name=name)
            if value:
                raise ValueError(f"provider adapters must declare {name}=false")
            object.__setattr__(self, name, value)
        metadata = require_finite_json_mapping(
            self.metadata,
            name="provider-adapter identity metadata",
        )
        conflicting = sorted(_RESERVED_METADATA_FIELDS.intersection(metadata))
        if conflicting:
            raise ValueError(
                "provider-adapter identity metadata uses reserved fields: "
                + ", ".join(conflicting)
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                metadata,
                name="provider-adapter identity metadata",
            ),
        )
        object.__setattr__(
            self,
            "provider_adapter_identity_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_ADAPTER_IDENTITY_SCHEMA,
            "schema_version": PROVIDER_ADAPTER_VERSION,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "adapter_implementation_id": self.adapter_implementation_id,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "provider_run_id": self.provider_run_id,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "point_semantics": self.point_semantics,
            "flow_semantics": self.flow_semantics,
            "ray_semantics": self.ray_semantics,
            "uses_truth": self.uses_truth,
            "uses_target_outcomes": self.uses_target_outcomes,
            "uses_downstream_physical_innovation": (
                self.uses_downstream_physical_innovation
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_ADAPTER_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["provider_adapter_identity_id"] = self.provider_adapter_identity_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderAdapterIdentityV1:
        mapping = require_mapping(value, name="provider-adapter identity")
        require_exact_fields(mapping, _IDENTITY_FIELDS, name="provider-adapter identity")
        if mapping["schema"] != PROVIDER_ADAPTER_IDENTITY_SCHEMA:
            raise ValueError("provider-adapter identity schema changed")
        if mapping["schema_version"] != PROVIDER_ADAPTER_VERSION:
            raise ValueError("provider-adapter identity version changed")
        if mapping["claim_boundary"] != PROVIDER_ADAPTER_CLAIM_BOUNDARY:
            raise ValueError("provider-adapter identity claim boundary changed")
        result = cls(
            adapter_name=mapping["adapter_name"],
            adapter_version=mapping["adapter_version"],
            adapter_implementation_id=mapping["adapter_implementation_id"],
            provider_family=mapping["provider_family"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            provider_run_id=mapping["provider_run_id"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            coordinate_semantics=mapping["coordinate_semantics"],
            point_semantics=mapping["point_semantics"],
            flow_semantics=mapping["flow_semantics"],
            ray_semantics=mapping["ray_semantics"],
            uses_truth=mapping["uses_truth"],
            uses_target_outcomes=mapping["uses_target_outcomes"],
            uses_downstream_physical_innovation=mapping[
                "uses_downstream_physical_innovation"
            ],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="provider-adapter identity metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-adapter identity derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class ProviderAdapterRequestV1:
    """Target-closed request for one exact native input snapshot and cutoff."""

    sequence_id: str
    causal_frame_stop: int
    input_family_id: str
    input_snapshot_id: str
    target_payloads_opened: bool = False
    target_outcomes_opened: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_adapter_request_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            require_exact_string(self.sequence_id, name="sequence_id"),
        )
        object.__setattr__(
            self,
            "causal_frame_stop",
            require_exact_integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        for name in ("input_family_id", "input_snapshot_id"):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        for name in ("target_payloads_opened", "target_outcomes_opened"):
            value = _strict_bool(getattr(self, name), name=name)
            if value:
                raise ValueError("provider-adapter requests require unopened target data")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                require_finite_json_mapping(
                    self.metadata,
                    name="provider-adapter request metadata",
                ),
                name="provider-adapter request metadata",
            ),
        )
        object.__setattr__(
            self,
            "provider_adapter_request_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_ADAPTER_REQUEST_SCHEMA,
            "schema_version": PROVIDER_ADAPTER_VERSION,
            "sequence_id": self.sequence_id,
            "causal_frame_stop": self.causal_frame_stop,
            "input_family_id": self.input_family_id,
            "input_snapshot_id": self.input_snapshot_id,
            "target_payloads_opened": self.target_payloads_opened,
            "target_outcomes_opened": self.target_outcomes_opened,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_ADAPTER_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["provider_adapter_request_id"] = self.provider_adapter_request_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderAdapterRequestV1:
        mapping = require_mapping(value, name="provider-adapter request")
        require_exact_fields(mapping, _REQUEST_FIELDS, name="provider-adapter request")
        if mapping["schema"] != PROVIDER_ADAPTER_REQUEST_SCHEMA:
            raise ValueError("provider-adapter request schema changed")
        if mapping["schema_version"] != PROVIDER_ADAPTER_VERSION:
            raise ValueError("provider-adapter request version changed")
        if mapping["claim_boundary"] != PROVIDER_ADAPTER_CLAIM_BOUNDARY:
            raise ValueError("provider-adapter request claim boundary changed")
        result = cls(
            sequence_id=mapping["sequence_id"],
            causal_frame_stop=mapping["causal_frame_stop"],
            input_family_id=mapping["input_family_id"],
            input_snapshot_id=mapping["input_snapshot_id"],
            target_payloads_opened=mapping["target_payloads_opened"],
            target_outcomes_opened=mapping["target_outcomes_opened"],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="provider-adapter request metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-adapter request derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class ProviderAdapterWindowV1:
    """One canonical window plus provider-neutral semantics and output path."""

    window: PredictionWindow
    relative_path: str
    product_role: str
    view_id: str
    stochastic_member_id: str
    dependence_group_ids: tuple[str, ...]
    frame_lineage: tuple[PredictionFrameLineageV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.window, PredictionWindow):
            raise TypeError("window must be a PredictionWindow")
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(self.relative_path, name="adapter payload path"),
        )
        object.__setattr__(
            self,
            "product_role",
            _choice(self.product_role, PRODUCT_ROLES, name="product_role"),
        )
        object.__setattr__(
            self,
            "view_id",
            require_exact_string(self.view_id, name="view_id"),
        )
        object.__setattr__(
            self,
            "stochastic_member_id",
            require_exact_string(
                self.stochastic_member_id,
                name="stochastic_member_id",
            ),
        )
        if type(self.dependence_group_ids) is not tuple:
            raise TypeError("dependence_group_ids must be a canonical tuple")
        groups = require_string_sequence(
            self.dependence_group_ids,
            name="dependence_group_ids",
        )
        if len(groups) != len(set(groups)):
            raise ValueError("dependence_group_ids must be unique")
        object.__setattr__(self, "dependence_group_ids", groups)
        if type(self.frame_lineage) is not tuple or not self.frame_lineage:
            raise TypeError("frame_lineage must be a nonempty canonical tuple")
        lineage = tuple(self.frame_lineage)
        if any(not isinstance(item, PredictionFrameLineageV1) for item in lineage):
            raise TypeError("frame_lineage must contain PredictionFrameLineageV1 values")
        if tuple(item.output_frame_id for item in lineage) != tuple(
            int(value) for value in self.window.frame_indices
        ):
            raise ValueError("adapter frame lineage differs from window frame identities")
        object.__setattr__(self, "frame_lineage", lineage)


@runtime_checkable
class PredictionProviderAdapterV1(Protocol):
    """Structural protocol implemented by trusted provider conversion adapters."""

    @property
    def identity(self) -> ProviderAdapterIdentityV1:
        """Return the complete immutable adapter/provider identity."""
        ...

    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> Sequence[ProviderAdapterWindowV1]:
        """Produce canonical windows for the exact causal request."""
        ...


@dataclass(frozen=True, slots=True)
class StaticPredictionProviderAdapterV1:
    """In-memory adapter useful for parity tests and already-decoded providers."""

    identity: ProviderAdapterIdentityV1
    windows: tuple[ProviderAdapterWindowV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ProviderAdapterIdentityV1):
            raise TypeError("identity must be ProviderAdapterIdentityV1")
        if type(self.windows) is not tuple or not self.windows:
            raise ValueError("windows must be a nonempty canonical tuple")
        if any(not isinstance(item, ProviderAdapterWindowV1) for item in self.windows):
            raise TypeError("windows must contain ProviderAdapterWindowV1 values")

    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> tuple[ProviderAdapterWindowV1, ...]:
        if not isinstance(request, ProviderAdapterRequestV1):
            raise TypeError("request must be ProviderAdapterRequestV1")
        return self.windows


def write_provider_adapter_request(
    path: str | Path,
    request: ProviderAdapterRequestV1,
    *,
    overwrite: bool = False,
) -> None:
    payload = json.dumps(
        request.to_dict(),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_provider_adapter_request(path: str | Path) -> ProviderAdapterRequestV1:
    return ProviderAdapterRequestV1.from_dict(
        load_json_object(path, name="provider-adapter request")
    )


def materialize_provider_adapter(
    adapter: PredictionProviderAdapterV1,
    request: ProviderAdapterRequestV1,
    output_manifest_path: str | Path,
) -> PredictionProviderManifestV1:
    """Materialize one adapter invocation into the canonical neutral manifest.

    Payload and manifest publication are no-clobber and idempotent. Adapter output
    order is canonicalized, while each payload retains its exact local window ID,
    path, view, stochastic member, dependence groups, and per-frame causal source
    interval.
    """

    if not isinstance(request, ProviderAdapterRequestV1):
        raise TypeError("request must be ProviderAdapterRequestV1")
    identity = getattr(adapter, "identity", None)
    if not isinstance(identity, ProviderAdapterIdentityV1):
        raise TypeError("adapter.identity must be ProviderAdapterIdentityV1")
    producer = getattr(adapter, "produce", None)
    if not callable(producer):
        raise TypeError("adapter must provide a callable produce(request) method")

    produced = tuple(producer(request))
    if not produced:
        raise ValueError("provider adapter produced no canonical windows")
    if any(not isinstance(item, ProviderAdapterWindowV1) for item in produced):
        raise TypeError("adapter output must contain ProviderAdapterWindowV1 values")
    windows = tuple(
        sorted(
            produced,
            key=lambda item: (
                item.view_id,
                item.frame_lineage[0].output_frame_id,
                item.window.window_id,
                item.stochastic_member_id,
                item.relative_path,
            ),
        )
    )
    paths = [item.relative_path for item in windows]
    if len(paths) != len(set(paths)):
        raise ValueError("adapter payload paths must be unique")
    window_ids = [item.window.window_id for item in windows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("adapter window IDs must be unique")
    for item in windows:
        if any(
            lineage.source_frame_stop_exclusive > request.causal_frame_stop
            for lineage in item.frame_lineage
        ):
            raise ValueError("adapter output crosses the requested causal frame boundary")

    manifest_path = Path(output_manifest_path)
    if manifest_path.is_symlink():
        raise ValueError("adapter output manifest must not be a symbolic link")
    manifest_root = manifest_path.parent.resolve()
    payloads: list[PredictionPayloadDescriptorV1] = []
    for index, item in enumerate(windows):
        destination = _resolved_output_member(
            manifest_root,
            item.relative_path,
            name=f"adapter payload {index} path",
        )
        _write_window_atomically(destination, item.window)
        relative_path = _relative_member(
            destination,
            root=manifest_root,
            name=f"adapter payload {index} output path",
        )
        payloads.append(
            PredictionPayloadDescriptorV1(
                product_role=item.product_role,
                window_id=item.window.window_id,
                path=relative_path,
                sha256=_file_sha256(destination),
                byte_count=int(destination.stat().st_size),
                view_id=item.view_id,
                stochastic_member_id=item.stochastic_member_id,
                dependence_group_ids=item.dependence_group_ids,
                dense_storage_dtype=item.window.dense_storage_dtype,
                has_scene_flow=item.window.scene_flow is not None,
                has_ray_directions=item.window.ray_directions is not None,
                frame_lineage=item.frame_lineage,
            )
        )

    manifest_metadata = dict(plain_json(identity.metadata))
    manifest_metadata.update(
        {
            "source_adapter": "prob4d-provider-adapter-v1",
            "provider_adapter_identity_id": identity.provider_adapter_identity_id,
            "provider_adapter_request_id": request.provider_adapter_request_id,
            "provider_adapter_contract_version": PROVIDER_ADAPTER_VERSION,
            "provider_adapter_input_family_id": request.input_family_id,
            "provider_adapter_input_snapshot_id": request.input_snapshot_id,
            "provider_adapter_causal_frame_stop": request.causal_frame_stop,
            "provider_adapter_request_metadata": plain_json(request.metadata),
            "uses_truth": False,
            "uses_target_outcomes": False,
            "uses_downstream_physical_innovation": False,
        }
    )
    manifest = PredictionProviderManifestV1(
        sequence_id=request.sequence_id,
        provider_family=identity.provider_family,
        provider_repository=identity.provider_repository,
        provider_revision=identity.provider_revision,
        provider_run_id=identity.provider_run_id,
        model_set_id=identity.model_set_id,
        loader_id=identity.loader_id,
        coordinate_semantics=identity.coordinate_semantics,
        point_semantics=identity.point_semantics,
        flow_semantics=identity.flow_semantics,
        ray_semantics=identity.ray_semantics,
        payloads=tuple(payloads),
        metadata=manifest_metadata,
    )
    save_prediction_provider_manifest(manifest_path, manifest)
    verified, report = verify_prediction_provider_manifest(
        manifest_path,
        causal_frame_stop=request.causal_frame_stop,
    )
    if report["admitted_payload_count"] != len(payloads):
        raise ValueError("adapter manifest contains payloads outside the causal request")
    return verified


__all__ = [
    "PROVIDER_ADAPTER_CLAIM_BOUNDARY",
    "PROVIDER_ADAPTER_VERSION",
    "PredictionProviderAdapterV1",
    "ProviderAdapterIdentityV1",
    "ProviderAdapterRequestV1",
    "ProviderAdapterWindowV1",
    "StaticPredictionProviderAdapterV1",
    "load_provider_adapter_request",
    "materialize_provider_adapter",
    "write_provider_adapter_request",
]
