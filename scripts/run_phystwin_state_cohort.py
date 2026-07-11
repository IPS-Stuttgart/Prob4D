"""Run a locked PhysTwin state-forecast cohort shard on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_logged(command: list[str], log_path: Path, *, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--released-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motioncrafter-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    arguments = parser.parse_args(argv)
    if arguments.shard_count < 1 or not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")

    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    input_contract = protocol["input"]
    cases = sorted(protocol["case_contracts"].items())
    selected = [
        item
        for index, item in enumerate(cases)
        if index % arguments.shard_count == arguments.shard_index
    ]
    environment = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(repository_root / "src")
    environment["HF_HOME"] = str(arguments.cache_dir)

    for case_index, (case_name, contract) in enumerate(selected, start=1):
        frame_start = int(contract["frame_start"])
        frame_stop = int(contract["frame_stop"])
        fit_end = int(contract["fit_end_frame"])
        case_directory = arguments.data_root / case_name
        released_directory = arguments.released_root / case_name
        corrected_path = arguments.corrected_root / case_name / "trajectory.pkl"
        run_directory = arguments.output_root / (
            f"{case_name}_cam0_f{frame_start}_{frame_stop}_determ_confirmation_v1"
        )
        manifest_path = run_directory / "predictions.json"
        print(
            f"[{case_index}/{len(selected)}] {case_name} frames {frame_start}:{frame_stop}",
            flush=True,
        )
        if not manifest_path.exists():
            run_logged(
                [
                    str(arguments.python),
                    "-m",
                    "prob4d.motioncrafter",
                    str(case_directory / "color" / "0.mp4"),
                    "--upstream-root",
                    str(arguments.motioncrafter_root),
                    "--output-dir",
                    str(run_directory),
                    "--cache-dir",
                    str(arguments.cache_dir),
                    "--model-type",
                    str(input_contract["model_type"]),
                    "--seed",
                    str(input_contract["seed"]),
                    "--frame-start",
                    str(frame_start),
                    "--frame-stop",
                    str(frame_stop),
                ],
                run_directory.with_suffix(".inference.log"),
                environment=environment,
            )

        common_arguments = [
            "--fit-end-frame",
            str(fit_end),
            "--physics-trajectory",
            str(released_directory / "inference.pkl"),
            "--corrected-trajectory",
            str(corrected_path),
            "--final-data",
            str(released_directory / "final_data.pkl"),
        ]
        state_output = run_directory / "state_forecast.json"
        if not state_output.exists():
            run_logged(
                [
                    str(arguments.python),
                    "-m",
                    "prob4d.phystwin_state",
                    str(manifest_path),
                    str(case_directory),
                    str(state_output),
                    "--product",
                    str(input_contract["prediction_product"]),
                    *common_arguments,
                ],
                run_directory.with_suffix(".state.log"),
                environment=environment,
            )

        experiment_output = run_directory / "experiment_zero.json"
        if not experiment_output.exists():
            run_logged(
                [
                    str(arguments.python),
                    "-m",
                    "prob4d.phystwin_experiment",
                    str(manifest_path),
                    str(case_directory),
                    str(experiment_output),
                    "--product",
                    str(input_contract["prediction_product"]),
                    *common_arguments,
                ],
                run_directory.with_suffix(".experiment.log"),
                environment=environment,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
