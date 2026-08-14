"""Executable conformance suite for trusted provider-adapter implementations."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)
from .prediction_provider_manifest import PredictionProviderManifestV1
from .provider_adapter import (
    PROVIDER_ADAPTER_CLAIM_BOUNDARY,
    PROVIDER_ADAPTER_VERSION,
    PredictionProviderAdapterV1,
    ProviderAdapterIdentityV1,
    ProviderAdapterRequestV1,
    StaticPredictionProviderAdapterV1,
    load_provider_adapter_request,
    materialize_provider_adapter,
    write_provider_adapter_request,
)

PROVIDER_ADAPTER_CONFORMANCE_SCHEMA: Final = "prob4d.provider-adapter-conformance"
PROVIDER_ADAPTER_CONFORMANCE_VERSION: Final = 1
PROVIDER_ADAPTER_CONFORMANCE_CLAIM_BOUNDARY: Final = (
    PROVIDER_ADAPTER_CLAIM_BOUNDARY
    + " This conformance artifact additionally checks exact repeatability, "
    "adapter-output order invariance, and causal-prefix invariance on the supplied "
    "deterministic fixture. It remains interoperability evidence rather than "
    "scientific provider evidence."
)

_RESULT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "adapter_identity",
        "base_request",
        "future_request",
        "base_manifest",
        "repeat_manifest",
        "permuted_manifest",
        "future_manifest",
        "observed_adapter_identity_ids",
        "checks",
        "reason_codes",
        "conformance_pass",
        "metadata",
        "claim_boundary",
        "provider_adapter_conformance_id",
    }
)
_CHECK_FIELDS: Final = frozenset(
    {
        "manifest_roundtrip",
        "adapter_identity_stable",
        "deterministic_repeat",
        "adapter_output_order_invariant",
        "causal_prefix_invariant",
        "future_check_performed",
    }
)


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _manifest_matches(
    manifest: PredictionProviderManifestV1,
    identity: ProviderAdapterIdentityV1,
    request: ProviderAdapterRequestV1,
) -> bool:
    metadata = manifest.metadata
    return (
        manifest.sequence_id == request.sequence_id
        and manifest.provider_family == identity.provider_family
        and manifest.provider_repository == identity.provider_repository
        and manifest.provider_revision == identity.provider_revision
        and manifest.provider_run_id == identity.provider_run_id
        and manifest.model_set_id == identity.model_set_id
        and manifest.loader_id == identity.loader_id
        and manifest.coordinate_semantics == identity.coordinate_semantics
        and manifest.point_semantics == identity.point_semantics
        and manifest.flow_semantics == identity.flow_semantics
        and manifest.ray_semantics == identity.ray_semantics
        and metadata.get("source_adapter") == "prob4d-provider-adapter-v1"
        and metadata.get("provider_adapter_identity_id")
        == identity.provider_adapter_identity_id
        and metadata.get("provider_adapter_request_id")
        == request.provider_adapter_request_id
        and metadata.get("provider_adapter_contract_version")
        == PROVIDER_ADAPTER_VERSION
        and metadata.get("provider_adapter_input_family_id") == request.input_family_id
        and metadata.get("provider_adapter_input_snapshot_id")
        == request.input_snapshot_id
        and metadata.get("provider_adapter_causal_frame_stop")
        == request.causal_frame_stop
        and plain_json(metadata.get("provider_adapter_request_metadata"))
        == plain_json(request.metadata)
        and metadata.get("uses_truth") is False
        and metadata.get("uses_target_outcomes") is False
        and metadata.get("uses_downstream_physical_innovation") is False
        and all(
            payload.is_causally_admitted(request.causal_frame_stop)
            for payload in manifest.payloads
        )
    )


def _payload_ids(manifest: PredictionProviderManifestV1) -> tuple[str, ...]:
    result: list[str] = []
    for payload in manifest.payloads:
        if payload.payload_id is None:
            raise ValueError("conformance manifest contains an unmaterialized payload ID")
        result.append(payload.payload_id)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProviderAdapterConformanceV1:
    """Self-contained replay of one deterministic adapter conformance fixture."""

    adapter_identity: ProviderAdapterIdentityV1
    base_request: ProviderAdapterRequestV1
    future_request: ProviderAdapterRequestV1
    base_manifest: PredictionProviderManifestV1
    repeat_manifest: PredictionProviderManifestV1
    permuted_manifest: PredictionProviderManifestV1
    future_manifest: PredictionProviderManifestV1
    observed_adapter_identity_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checks: Mapping[str, bool] = field(init=False)
    reason_codes: tuple[str, ...] = field(init=False)
    conformance_pass: bool = field(init=False)
    provider_adapter_conformance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_identity, ProviderAdapterIdentityV1):
            raise TypeError("adapter_identity must be ProviderAdapterIdentityV1")
        if not isinstance(self.base_request, ProviderAdapterRequestV1):
            raise TypeError("base_request must be ProviderAdapterRequestV1")
        if not isinstance(self.future_request, ProviderAdapterRequestV1):
            raise TypeError("future_request must be ProviderAdapterRequestV1")
        manifests = (
            self.base_manifest,
            self.repeat_manifest,
            self.permuted_manifest,
            self.future_manifest,
        )
        if any(not isinstance(item, PredictionProviderManifestV1) for item in manifests):
            raise TypeError("all conformance manifests must be provider manifests")
        if self.future_request.sequence_id != self.base_request.sequence_id:
            raise ValueError("base and future requests must use the same sequence")
        if self.future_request.input_family_id != self.base_request.input_family_id:
            raise ValueError("base and future requests must use one input family")
        if self.future_request.causal_frame_stop <= self.base_request.causal_frame_stop:
            raise ValueError("future_request must extend the causal cutoff")
        if self.future_request.input_snapshot_id == self.base_request.input_snapshot_id:
            raise ValueError("future_request must bind an extended input snapshot")
        if plain_json(self.future_request.metadata) != plain_json(self.base_request.metadata):
            raise ValueError("base and future requests must use the same configuration")
        if type(self.observed_adapter_identity_ids) is not tuple:
            raise TypeError("observed_adapter_identity_ids must be a canonical tuple")
        observed = tuple(
            require_sha256(item, name="observed_adapter_identity_id")
            for item in self.observed_adapter_identity_ids
        )
        if not observed:
            raise ValueError("observed_adapter_identity_ids must not be empty")
        object.__setattr__(self, "observed_adapter_identity_ids", observed)

        base_manifests = manifests[:3]
        manifest_roundtrip = all(
            _manifest_matches(item, self.adapter_identity, self.base_request)
            for item in base_manifests
        ) and _manifest_matches(
            self.future_manifest,
            self.adapter_identity,
            self.future_request,
        )
        identity_stable = (
            len(set(observed)) == 1
            and observed[0] == self.adapter_identity.provider_adapter_identity_id
        )
        base_ids = _payload_ids(self.base_manifest)
        repeat_ids = _payload_ids(self.repeat_manifest)
        permuted_ids = _payload_ids(self.permuted_manifest)
        future_prefix = tuple(
            payload.payload_id
            for payload in self.future_manifest.admitted_payloads(
                self.base_request.causal_frame_stop
            )
        )
        checks = {
            "manifest_roundtrip": manifest_roundtrip,
            "adapter_identity_stable": identity_stable,
            "deterministic_repeat": (
                self.base_manifest.artifact_id == self.repeat_manifest.artifact_id
                and base_ids == repeat_ids
            ),
            "adapter_output_order_invariant": (
                self.base_manifest.artifact_id == self.permuted_manifest.artifact_id
                and base_ids == permuted_ids
            ),
            "causal_prefix_invariant": base_ids == future_prefix,
            "future_check_performed": True,
        }
        reasons = tuple(
            sorted(name.replace("_", "-") + "-failed" for name, passed in checks.items() if not passed)
        )
        object.__setattr__(
            self,
            "checks",
            frozen_finite_json_mapping(checks, name="adapter conformance checks"),
        )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "conformance_pass", not reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                require_finite_json_mapping(
                    self.metadata,
                    name="adapter conformance metadata",
                ),
                name="adapter conformance metadata",
            ),
        )
        object.__setattr__(
            self,
            "provider_adapter_conformance_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_ADAPTER_CONFORMANCE_SCHEMA,
            "schema_version": PROVIDER_ADAPTER_CONFORMANCE_VERSION,
            "adapter_identity": self.adapter_identity.to_dict(),
            "base_request": self.base_request.to_dict(),
            "future_request": self.future_request.to_dict(),
            "base_manifest": self.base_manifest.to_record(),
            "repeat_manifest": self.repeat_manifest.to_record(),
            "permuted_manifest": self.permuted_manifest.to_record(),
            "future_manifest": self.future_manifest.to_record(),
            "observed_adapter_identity_ids": list(self.observed_adapter_identity_ids),
            "checks": plain_json(self.checks),
            "reason_codes": list(self.reason_codes),
            "conformance_pass": self.conformance_pass,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_ADAPTER_CONFORMANCE_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["provider_adapter_conformance_id"] = (
            self.provider_adapter_conformance_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderAdapterConformanceV1:
        mapping = require_mapping(value, name="provider-adapter conformance")
        require_exact_fields(mapping, _RESULT_FIELDS, name="provider-adapter conformance")
        if mapping["schema"] != PROVIDER_ADAPTER_CONFORMANCE_SCHEMA:
            raise ValueError("provider-adapter conformance schema changed")
        if mapping["schema_version"] != PROVIDER_ADAPTER_CONFORMANCE_VERSION:
            raise ValueError("provider-adapter conformance version changed")
        if mapping["claim_boundary"] != PROVIDER_ADAPTER_CONFORMANCE_CLAIM_BOUNDARY:
            raise ValueError("provider-adapter conformance claim boundary changed")
        checks = require_mapping(mapping["checks"], name="adapter conformance checks")
        require_exact_fields(checks, _CHECK_FIELDS, name="adapter conformance checks")
        for name, item in checks.items():
            _strict_bool(item, name=f"adapter conformance check {name}")
        observed = mapping["observed_adapter_identity_ids"]
        if not isinstance(observed, list):
            raise ValueError("observed_adapter_identity_ids must be a JSON array")
        result = cls(
            adapter_identity=ProviderAdapterIdentityV1.from_dict(
                mapping["adapter_identity"]
            ),
            base_request=ProviderAdapterRequestV1.from_dict(mapping["base_request"]),
            future_request=ProviderAdapterRequestV1.from_dict(
                mapping["future_request"]
            ),
            base_manifest=PredictionProviderManifestV1.from_record(
                mapping["base_manifest"]
            ),
            repeat_manifest=PredictionProviderManifestV1.from_record(
                mapping["repeat_manifest"]
            ),
            permuted_manifest=PredictionProviderManifestV1.from_record(
                mapping["permuted_manifest"]
            ),
            future_manifest=PredictionProviderManifestV1.from_record(
                mapping["future_manifest"]
            ),
            observed_adapter_identity_ids=tuple(observed),
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="adapter conformance metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-adapter conformance replay changed")
        return result


def write_provider_adapter_conformance(
    path: str | Path,
    result: ProviderAdapterConformanceV1,
    *,
    overwrite: bool = False,
) -> None:
    payload = json.dumps(result.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_provider_adapter_conformance(path: str | Path) -> ProviderAdapterConformanceV1:
    return ProviderAdapterConformanceV1.from_dict(
        load_json_object(path, name="provider-adapter conformance")
    )


def run_provider_adapter_conformance(
    adapter: PredictionProviderAdapterV1,
    base_request: ProviderAdapterRequestV1,
    future_request: ProviderAdapterRequestV1,
    output_directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProviderAdapterConformanceV1:
    """Run deterministic repeat, permutation, and future-prefix checks."""

    root = Path(output_directory)
    identity = getattr(adapter, "identity", None)
    producer = getattr(adapter, "produce", None)
    if not isinstance(identity, ProviderAdapterIdentityV1) or not callable(producer):
        raise TypeError("adapter must implement PredictionProviderAdapterV1")
    observed = [identity.provider_adapter_identity_id]

    def produce(request: ProviderAdapterRequestV1) -> tuple[object, ...]:
        windows = tuple(producer(request))
        current = getattr(adapter, "identity", None)
        if not isinstance(current, ProviderAdapterIdentityV1):
            raise TypeError("adapter.identity changed to an unsupported value")
        observed.append(current.provider_adapter_identity_id)
        return windows

    first = produce(base_request)
    repeat = produce(base_request)
    future = produce(future_request)
    base_manifest = materialize_provider_adapter(
        StaticPredictionProviderAdapterV1(identity=identity, windows=first),
        base_request,
        root / "base/provider-neutral.json",
    )
    repeat_manifest = materialize_provider_adapter(
        StaticPredictionProviderAdapterV1(identity=identity, windows=repeat),
        base_request,
        root / "repeat/provider-neutral.json",
    )
    permuted_manifest = materialize_provider_adapter(
        StaticPredictionProviderAdapterV1(
            identity=identity,
            windows=tuple(reversed(first)),
        ),
        base_request,
        root / "permuted/provider-neutral.json",
    )
    future_manifest = materialize_provider_adapter(
        StaticPredictionProviderAdapterV1(identity=identity, windows=future),
        future_request,
        root / "future/provider-neutral.json",
    )
    return ProviderAdapterConformanceV1(
        adapter_identity=identity,
        base_request=base_request,
        future_request=future_request,
        base_manifest=base_manifest,
        repeat_manifest=repeat_manifest,
        permuted_manifest=permuted_manifest,
        future_manifest=future_manifest,
        observed_adapter_identity_ids=tuple(observed),
        metadata={} if metadata is None else metadata,
    )


def _load_adapter(specification: str) -> PredictionProviderAdapterV1:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("adapter must use module:factory form")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if factory is None:
        raise ValueError(f"adapter factory {specification!r} does not exist")
    adapter = factory() if callable(factory) else factory
    identity = getattr(adapter, "identity", None)
    if not isinstance(identity, ProviderAdapterIdentityV1) or not callable(
        getattr(adapter, "produce", None)
    ):
        raise TypeError("adapter factory must return a PredictionProviderAdapterV1")
    return adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-request")
    build.add_argument("--sequence-id", required=True)
    build.add_argument("--causal-frame-stop", type=int, required=True)
    build.add_argument("--input-family-id", required=True)
    build.add_argument("--input-snapshot-id", required=True)
    build.add_argument("--metadata-json", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    verify_request = subparsers.add_parser("verify-request")
    verify_request.add_argument("--artifact", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("adapter")
    run.add_argument("--base-request", type=Path, required=True)
    run.add_argument("--future-request", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-request":
        metadata = (
            {}
            if arguments.metadata_json is None
            else load_json_object(
                arguments.metadata_json,
                name="provider-adapter request metadata",
            )
        )
        request = ProviderAdapterRequestV1(
            sequence_id=arguments.sequence_id,
            causal_frame_stop=arguments.causal_frame_stop,
            input_family_id=arguments.input_family_id,
            input_snapshot_id=arguments.input_snapshot_id,
            metadata=metadata,
        )
        write_provider_adapter_request(
            arguments.output,
            request,
            overwrite=arguments.overwrite,
        )
        print(json.dumps({"provider_adapter_request_id": request.provider_adapter_request_id}, sort_keys=True))
        return 0
    if arguments.command == "verify-request":
        request = load_provider_adapter_request(arguments.artifact)
        print(json.dumps({"provider_adapter_request_id": request.provider_adapter_request_id}, sort_keys=True))
        return 0
    if arguments.command == "run":
        result = run_provider_adapter_conformance(
            _load_adapter(arguments.adapter),
            load_provider_adapter_request(arguments.base_request),
            load_provider_adapter_request(arguments.future_request),
            arguments.output_dir,
            metadata={"adapter_factory": arguments.adapter},
        )
        write_provider_adapter_conformance(
            arguments.report,
            result,
            overwrite=arguments.overwrite,
        )
    else:
        result = load_provider_adapter_conformance(arguments.artifact)
    print(
        json.dumps(
            {
                "conformance_id": result.provider_adapter_conformance_id,
                "conformance_pass": result.conformance_pass,
                "reason_codes": list(result.reason_codes),
            },
            sort_keys=True,
        )
    )
    return 0 if result.conformance_pass else 2


__all__ = [
    "PROVIDER_ADAPTER_CONFORMANCE_CLAIM_BOUNDARY",
    "PROVIDER_ADAPTER_CONFORMANCE_VERSION",
    "ProviderAdapterConformanceV1",
    "load_provider_adapter_conformance",
    "main",
    "run_provider_adapter_conformance",
    "write_provider_adapter_conformance",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
