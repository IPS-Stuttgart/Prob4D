"""Export a Prob4D observation belief with an explicit causal stream contract."""

from __future__ import annotations

import argparse
import io
import json
import sys
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the established exporter and bind stream contract v2 when admissible."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(argument in {"-h", "--help"} for argument in arguments):
        return legacy_main(arguments)
    paths = _paths(arguments)
    captured = io.StringIO()
    with redirect_stdout(captured):
        status = legacy_main(arguments)
    if status != 0:
        output = captured.getvalue()
        if output:
            print(output, end="")
        return int(status)

    summary = json.loads(captured.getvalue())
    artifact = load_observation_belief_export(paths.output_npz)
    anchor = load_metric_gauge_anchor(paths.metric_gauge_anchor)
    if artifact.metadata.get("gauge_mode") == "sequential":
        artifact = bind_causal_stream_contract_v2(
            artifact,
            metric_anchor=anchor,
        )
        save_observation_belief_export(paths.output_npz, artifact)
        summary.update(artifact.summary())
        summary["prob4d_causal_stream_contract_version"] = (
            PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        )
        summary["metric_gauge_anchor_id"] = anchor.artifact_id
        summary["gauge_posterior"] = artifact.metadata["gauge_posterior"]
        if paths.summary_json is not None:
            _write_json(paths.summary_json, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
