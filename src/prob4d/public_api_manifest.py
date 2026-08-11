"""Build and verify a content-addressed inventory of supported Python surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from . import __all__ as ROOT_PUBLIC_EXPORTS
from ._version import __version__
from .api import v1 as api_v1
from .api import v2 as api_v2
from .project_identity import PROB4D_PROJECT_ID

PUBLIC_API_MANIFEST_SCHEMA: Final = "prob4d.public-api-manifest"
PUBLIC_API_MANIFEST_VERSION: Final = 1
ROOT_COMPATIBILITY_SURFACE_VERSION: Final = 1
PUBLIC_API_MANIFEST_CLAIM_BOUNDARY: Final = (
    "This manifest records installed Python compatibility surfaces and their exact "
    "export inventories. It is interoperability evidence only; it does not establish "
    "provider accuracy, uncertainty calibration, physical-query improvement, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")


def _manifest_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _surface_exports(value: Sequence[str], *, name: str) -> list[str]:
    exports = list(value)
    if any(type(item) is not str or not item for item in exports):
        raise ValueError(f"{name} exports must be nonempty exact strings")
    if len(exports) != len(set(exports)):
        raise ValueError(f"{name} exports must be unique")
    return sorted(exports)


def build_public_api_manifest() -> dict[str, Any]:
    """Return the canonical manifest for the executing Prob4D installation."""

    payload: dict[str, Any] = {
        "schema": PUBLIC_API_MANIFEST_SCHEMA,
        "schema_version": PUBLIC_API_MANIFEST_VERSION,
        "package": {
            "name": "prob4d",
            "version": __version__,
            "project_id": PROB4D_PROJECT_ID,
        },
        "surfaces": {
            "compatibility_root": {
                "module": "prob4d",
                "surface_version": ROOT_COMPATIBILITY_SURFACE_VERSION,
                "loading": "lazy-compatibility-shim-v1",
                "exports": _surface_exports(
                    ROOT_PUBLIC_EXPORTS,
                    name="compatibility_root",
                ),
            },
            "api_v1": {
                "module": "prob4d.api.v1",
                "api_version": api_v1.API_VERSION,
                "provider_api_version": api_v1.PROVIDER_API_VERSION,
                "exports": _surface_exports(api_v1.__all__, name="api_v1"),
            },
            "api_v2": {
                "module": "prob4d.api.v2",
                "api_version": api_v2.API_VERSION,
                "provider_api_version": api_v2.PROVIDER_API_VERSION,
                "provider_factor_api_version": api_v2.PROVIDER_FACTOR_API_VERSION,
                "exports": _surface_exports(api_v2.__all__, name="api_v2"),
            },
        },
        "claim_boundary": PUBLIC_API_MANIFST_CLAIM_BOUNDARY,
    }
    payload["manifest_id"] = _manifest_id(payload)
    return validate_public_api_manifest(payload)


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string without surrounding whitespace")
    return cast(str, value)


def _strict_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return cast(int, value)


def _validate_exports(value: Any, *, name: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    exports = _surface_exports(value, name=name)
    if list(value) != exports:
        raise ValueError(f"{name} must be sorted canonically")
    for export in exports:
        if export.startswith("_") and export != "__version__":
            raise ValueError(f"{name} contains a private export: {export}")
    return exports


def validate_public_api_manifest(value: Any) -> dict[str, Any]:
    """Deeply validate a public-API manifest and return canonical plain data."""

    payload = _strict_mapping(value, name="manifest")
    _exact_keys(
        payload,
        {
            "schema",
            "schema_version",
            "package",
            "surfaces",
            "claim_boundary",
            "manifest_id",
        },
        name="manifest",
    )
    if _strict_string(payload["schema"], name="schema") != PUBLIC_API_MANIFEST_SCHEMA:
        raise ValueError("unsupported public-API manifest schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version")
        != PUBLIC_API_MANIFEST_VERSION
    ):
        raise ValueError("unsupported public-API manifest schema version")
    if (
        _strict_string(payload["claim_boundary"], name="claim_boundary")
        != PUBLIC_API_MANIFST_CLAIM_BOUNDARY
    ):
        raise ValueError("public-API manifest claim boundary is not canonical")

    package = _strict_mapping(payload["package"], name="package")
    _exact_keys(package, {"name", "version", "project_id"}, name="package")
    if _strict_string(package["name"], name="package.name") != "prob4d":
        raise ValueError("public-API manifest package name must be 'prob4d'")
    _strict_string(package["version"], name="package.version")
    if (
        _strict_string(package["project_id"], name="package.project_id")
        != PROB4D_PROJECT_ID
    ):
        raise ValueError("public-API manifest project identity is not canonical")

    surfaces = _strict_mapping(payload["surfaces"], name="surfaces")
    _exact_keys(
        surfaces,
        {"compatibility_root", "api_v1", "api_v2"},
        name="surfaces",
    )

    root = _strict_mapping(surfaces["compatibility_root"], name="compatibility_root")
    _exact_keys(
        root,
        {"module", "surface_version", "loading", "exports"},
        name="compatibility_root",
    )
    if _strict_string(root["module"], name="compatibility_root.module") != "prob4d":
        raise ValueError("compatibility-root module is not canonical")
    if (
        _strict_integer(
            root["surface_version"],
            name="compatibility_root.surface_version",
        )
        != ROOT_COMPATIBILITY_SURFACE_VERSION
    ):
        raise ValueError("compatibility-root surface version is not supported")
    if (
        _strict_string(root["loading"], name="compatibility_root.loading")
        != "lazy-compatibility-shim-v1"
    ):
        raise ValueError("compatibility-root loading mode is not canonical")
    _validate_exports(root["exports"], name="compatibility_root.exports")

    api1 = _strict_mapping(surfaces["api_v1"], name="api_v1")
    _exact_keys(
        api1,
        {"module", "api_version", "provider_api_version", "exports"},
        name="api_v1",
    )
    if _strict_string(api1["module"], name="api_v1.module") != "prob4d.api.v1":
        raise ValueError("api_v1 module is not canonical")
    if _strict_integer(api1["api_version"], name="api_v1.api_version") != 1:
        raise ValueError("api_v1 API version is not canonical")
    if (
        _strict_integer(api1["provider_api_version"], name="api_v1.provider_api_version")
        != 1
    ):
        raise ValueError("api_v1 provider API version is not canonical")
    _validate_exports(api1["exports"], name="api_v1.exports")

    api2 = _strict_mapping(surfaces["api_v2"], name="api_v2")
    _exact_keys(
        api2,
        {
            "module",
            "api_version",
            "provider_api_version",
            "provider_factor_api_version",
            "exports",
        },
        name="api_v2",
    )
    if _strict_string(api2["module"], name="api_v2.module") != "prob4d.api.v2":
        raise ValueError("api_v2 module is not canonical")
    if _strict_integer(api2["api_version"], name="api_v2.api_version") != 2:
        raise ValueError("api_v2 API version is not canonical")
    if (
        _strict_integer(api2["provider_api_version"], name="api_v2.provider_api_version")
        != 2
    ):
        raise ValueError("api_v2 provider API version is not canonical")
    if (
        _strict_integer(
            api2["provider_factor_api_version"],
            name="api_v2.provider_factor_api_version",
        )
        != 2
    ):
        raise ValueError("api_v2 provider-factor API version is not canonical")
    _validate_exports(api2["exports"], name="api_v2.exports")

    manifest_id = _strict_string(payload["manifest_id"], name="manifest_id")
    if len(manifest_id) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_id
    ):
        raise ValueError("manifest_id must be a lowercase SHA-256 digest")
    unsigned = dict(payload)
    unsigned.pop("manifest_id")
    if manifest_id != _manifest_id(unsigned):
        raise ValueEError("manifest_id does not match the canonical manifest content")
    canonical: dict[str, Any] = json.loads(_canonical_json(payload))
    return canonical


def load_public_api_manifest(path: str | Path) -> dict[str, Any]:
    """Load strict JSON and validate a public-API manifest."""

    def reject_constant(value: str) -> None:
        raise ValueEError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    return validate_public_api_manifest(payload)


def write_public_api_manifest(
    path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one manifest without replacing different existing bytes."""

    payload = (
        build_public_api_manifest()
        if manifest is None
        else validate_public_api_manifest(manifest)
    )
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("public-API manifest destination must not be a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload)
    if destination.exists():
        existing = load_public_api_manifest(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different public-API manifest: {destination}"
            )
        return existing
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("print", help="print the executing installation's manifest")
    show.set_defaults(handler=_print_command)

    build = subparsers.add_parser("build", help="write the executing installation's manifest")
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="verify a persisted manifest")
    verify.add_argument("manifest")
    verify.add_argument(
        "--require-current",
        action="store_true",
        help="also require equality with the executing installation",
    )
    verify.set_defaults(handler=_verify_command)
    return parser


def _print_command(arguments: argparse.Namespace) -> int:
    del arguments
    print(_canonical_json(build_public_api_manifest()).decode("utf-8"), end="")
    return 0


def _build_command(arguments: argparse.Namespace) -> int:
    manifest = write_public_api_manifest(arguments.output)
    print(manifest["manifest_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    manifest = load_public_api_manifest(arguments.manifest)
    if arguments.require_current and manifest != build_public_api_manifest():
        raise ValueError("persisted public-API manifest does not match this installation")
    print(manifest["manifest_id"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public-API manifest command."""

    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


__all__ = [
    "PUBLIC_API_MANIFEST_CLAIM_BOUNDARY",
    "PUBLIC_API_MANIFEST_SCHEMA",
    "PUBLIC_API_MANIFEST_VERSION",
    "ROOT_COMPATIBILITY_SURFACE_VERSION",
    "build_public_api_manifest",
    "load_public_api_manifest",
    "main",
    "validate_public_api_manifest",
    "write_public_api_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
