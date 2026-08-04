"""Strict current-schema loading for observation-factor bundles."""

from __future__ import annotations

from pathlib import Path

from ._observation_factor_bundle import ObservationFactorBundle
from ._observation_factor_io import (
    load_observation_factor_bundle as _load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from ._observation_factor_manifest_validation import (
    validate_observation_factor_manifest_types,
)
from ._strict_json import load_json_object


def load_observation_factor_bundle(
    manifest_path: str | Path,
) -> ObservationFactorBundle:
    """Load a bundle after fail-closed portable-JSON type validation."""

    record = load_json_object(
        manifest_path,
        name="observation-factor manifest",
    )
    validate_observation_factor_manifest_types(record)
    return _load_observation_factor_bundle(manifest_path)


__all__ = [
    "load_observation_factor_bundle",
    "write_observation_factor_bundle",
]
