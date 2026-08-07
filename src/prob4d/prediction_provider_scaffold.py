"""No-clobber scaffold for external provider-neutral prediction imports."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .prediction_provider_import import (
    PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA,
    PREDICTION_PROVIDER_IMPORT_SPEC_VERSION,
)
from .prediction_provider_manifest import SOURCE_DEPENDENCY_SEMANTICS

PREDICTION_PROVIDER_SCAFFOLD_SPECIFICATION: Final = "provider-import.json"
PREDICTION_PROVIDER_SCAFFOLD_README: Final = "README.md"


def _scaffold_specification() -> dict[str, object]:
    return {
        "schema": PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA,
        "schema_version": PREDICTION_PROVIDER_IMPORT_SPEC_VERSION,
        "sequence_id": "REPLACE_WITH_SEQUENCE_ID",
        "provider_family": "REPLACE_WITH_PROVIDER_FAMILY",
        "provider_repository": "REPLACE_WITH_OWNER_SLASH_REPOSITORY",
        "provider_revision": "REPLACE_WITH_LOWERCASE_40_OR_64_HEX_REVISION",
        "provider_run_id": "REPLACE_WITH_LOWERCASE_SHA256",
        "model_set_id": "REPLACE_WITH_LOWERCASE_SHA256",
        "loader_id": "REPLACE_WITH_LOWERCASE_SHA256",
        "coordinate_semantics": "window-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "forward-point-displacement",
        "ray_semantics": "absent",
        "source_dependency_semantics": SOURCE_DEPENDENCY_SEMANTICS,
        "payloads": [
            {
                "product_role": "independent-window",
                "window_id": "window_0000",
                "path": "windows/window_0000.npz",
                "view_id": "camera-0",
                "stochastic_member_id": "REPLACE_WITH_STOCHASTIC_MEMBER_ID",
                "dependence_group_ids": [
                    "model-set:REPLACE_WITH_MODEL_SET_ID",
                    "input-video:REPLACE_WITH_INPUT_VIDEO_SHA256",
                    "stochastic-member:REPLACE_WITH_MEMBER_ID",
                ],
                "frame_lineage": [
                    {
                        "output_frame_id": 0,
                        "source_frame_start": 0,
                        "source_frame_stop_exclusive": 1,
                        "contributor_ids": ["window_0000"],
                    }
                ],
            }
        ],
        "metadata": {
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "adapter_version": "REPLACE_WITH_ADAPTER_VERSION",
        },
    }


def _scaffold_readme() -> str:
    return """# Prob4D generic provider import scaffold

This directory is intentionally **not importable yet**. Replace every
`REPLACE_WITH_...` value in `provider-import.json`, copy canonical versioned
`PredictionWindow` archives below `windows/`, and declare one exact causal source
interval for every output frame.

The specification must describe the bytes that were actually executed:

- exact provider source revision;
- content identities for the provider run, model set, and loader;
- coordinate, point, flow, and ray semantics;
- stochastic-member and shared-dependency groups; and
- complete per-output source-frame lineage.

Then run:

```bash
prob4d prediction import-generic \\
  provider-import.json \\
  provider-neutral.json

prob4d prediction validate provider-neutral.json
```

Prob4D derives payload hashes, byte counts, dense precision, and optional-field
presence from the exact NPZ bytes. The importer rejects traversal, symbolic links,
duplicate JSON keys, non-finite values, schema drift, malformed identities,
changed source bytes, and lineage that disagrees with payload frame identities.

A valid neutral manifest proves provenance and causal admission only. It does not
establish provider accuracy, calibration, independence, downstream physical-query
benefit, or Causal4D intervention benefit.
"""


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def scaffold_prediction_provider_import(
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Create an intentionally incomplete, no-clobber provider import scaffold."""

    destination_input = Path(output_directory)
    if destination_input.is_symlink():
        raise ValueError("prediction-provider scaffold destination is a symbolic link")
    destination = destination_input.resolve()
    if destination.exists():
        raise FileExistsError(
            "prediction-provider scaffold destination already exists; refusing to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    windows = destination / "windows"
    specification = destination / PREDICTION_PROVIDER_SCAFFOLD_SPECIFICATION
    readme = destination / PREDICTION_PROVIDER_SCAFFOLD_README
    try:
        windows.mkdir()
        _write_new_text(
            specification,
            json.dumps(_scaffold_specification(), indent=2, allow_nan=False) + "\n",
        )
        _write_new_text(readme, _scaffold_readme())
        _fsync_directory(windows)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        specification.unlink(missing_ok=True)
        readme.unlink(missing_ok=True)
        try:
            windows.rmdir()
        except OSError:
            pass
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return specification, readme


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a no-clobber generic-provider import scaffold with explicit "
            "provenance and causal-lineage placeholders."
        )
    )
    parser.add_argument("output_directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    specification, readme = scaffold_prediction_provider_import(arguments.output_directory)
    print(
        json.dumps(
            {
                "output_directory": str(specification.parent),
                "specification": str(specification),
                "readme": str(readme),
                "ready_for_import": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "PREDICTION_PROVIDER_SCAFFOLD_README",
    "PREDICTION_PROVIDER_SCAFFOLD_SPECIFICATION",
    "main",
    "scaffold_prediction_provider_import",
]


if __name__ == "__main__":
    raise SystemExit(main())
