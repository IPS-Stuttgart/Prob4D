"""Export a Prob4D observation belief with an explicit causal stream contract."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from .causal_stream_contract import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    bind_causal_stream_contract_v2,
)
from .observation_contract import save_observation_belief_export
from .observation_export import main as legacy_main
from .observation_validation import load_observation_belief_export
from .provider_v1 import load_metric_gauge_anchor


def _paths(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--metric-gauge-anchor", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parsed, _ = parser.parse_known_args(argv)
    return parsed


def _temporary_path(parent: Path, *, prefix: str, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=prefix,
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _replace_option_value(
    arguments: list[str],
    option: str,
    replacement: str,
) -> list[str]:
    result = list(arguments)
    for index, argument in enumerate(result):
        if argument == option:
            if index + 1 >= len(result):
                raise ValueError(f"{option} requires a value")
            result[index + 1] = replacement
            return result
        if argument.startswith(f"{option}="):
            result[index] = f"{option}={replacement}"
            return result
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_observation(path: Path, artifact) -> None:
    temporary = _temporary_path(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    try:
        save_observation_belief_export(temporary, artifact)
        restored = load_observation_belief_export(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise RuntimeError("observation artifact changed during serialization")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Run the established exporter and bind stream contract v2 when admissible."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(argument in {"-h", "--help"} for argument in arguments):
        return legacy_main(arguments)
    paths = _paths(arguments)
    temporary_output = _temporary_path(
        paths.output_npz.parent,
        prefix=f".{paths.output_npz.name}.legacy.",
        suffix=".npz",
    )
    temporary_summary = None
    legacy_arguments = list(arguments)
    legacy_arguments[1] = str(temporary_output)
    if paths.summary_json is not None:
        temporary_summary = _temporary_path(
            paths.summary_json.parent,
            prefix=f".{paths.summary_json.name}.legacy.",
            suffix=".json",
        )
        legacy_arguments = _replace_option_value(
            legacy_arguments,
            "--summary-json",
            str(temporary_summary),
        )

    try:
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = legacy_main(legacy_arguments)
        if status != 0:
            output = captured.getvalue()
            if output:
                print(output, end="")
            return int(status)

        summary = json.loads(captured.getvalue())
        artifact = load_observation_belief_export(temporary_output)
        anchor = load_metric_gauge_anchor(paths.metric_gauge_anchor)
        if artifact.metadata.get("gauge_mode") == "sequential":
            artifact = bind_causal_stream_contract_v2(
                artifact,
                metric_anchor=anchor,
            )
            summary["prob4d_causal_stream_contract_version"] = (
                PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
            )
            summary["metric_gauge_anchor_id"] = anchor.artifact_id
            summary["metric_anchor_covariance_treatment"] = (
                anchor.covariance_treatment
            )
            summary["gauge_posterior"] = artifact.metadata["gauge_posterior"]

        _atomic_write_observation(paths.output_npz, artifact)
        summary.update(artifact.summary())
        summary["output"] = str(paths.output_npz.resolve())
        if paths.summary_json is not None:
            _atomic_write_json(paths.summary_json, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
