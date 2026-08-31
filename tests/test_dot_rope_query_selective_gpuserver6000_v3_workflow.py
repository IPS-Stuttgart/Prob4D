from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-rope-query-selective-heldout-gpuserver6000-v3.yml"
REQUEST = (
    "protocols/execution_requests/"
    "dot_rope_query_selective_heldout_gpuserver6000_v3.json"
)
PROTOCOL = "protocols/dot-rope-query-selective-heldout-v1.json"


def _load() -> tuple[dict, str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    assert isinstance(value, dict)
    return value, text


def test_workflow_retains_frozen_protocol_and_one_file_trigger() -> None:
    value, text = _load()
    triggers = value.get("on", value.get(True))
    assert triggers["push"] == {"branches": ["main"], "paths": [REQUEST]}
    assert value["env"]["PROTOCOL_PATH"] == PROTOCOL
    assert value["permissions"] == {"contents": "read"}
    assert value["concurrency"]["cancel-in-progress"] is False
    assert 'if [[ ${#changed[@]} -ne 1 || "${changed[0]}" != "$REQUEST_PATH" ]]' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text


def test_provider_only_moves_execution_lane_to_gpuserver6000() -> None:
    value, text = _load()
    provider = value["jobs"]["provider"]
    assert provider["runs-on"] == ["self-hosted", "gpuserver6000"]
    assert provider["environment"] == "trusted-self-hosted-validation"
    assert provider["permissions"] == {"contents": "read"}
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert (
        value["env"]["RUNTIME_ROOT"]
        == "/home/github-runner/.cache/prob4d/dot-r11-r30-cut3r-gpuserver6000-v1"
    )
    assert (
        value["env"]["ARCHIVE_ROOT"]
        == "/home/github-runner/.cache/prob4d/dot-r11-r30-archives-v29"
    )
    assert "prob4d.dot-cut3r-gpuserver6000-runtime-prewarm" in text
    assert "gpuserver4090" not in text
    assert "workstation1" not in text


def test_provider_and_archives_keep_exact_frozen_identities() -> None:
    value, text = _load()
    env = value["env"]
    assert env["CUT3R_REVISION"] == "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
    assert (
        env["CUT3R_CHECKPOINT_SHA256"]
        == "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
    )
    assert env["R11_ARCHIVE"] == "R11-20.zip"
    assert env["R11_MD5"] == "23ce3e7067465d3edabe20b4c7cfa388"
    assert env["R21_ARCHIVE"] == "R21-30.zip"
    assert env["R21_MD5"] == "8aee77f79d1aff6e1f3fd21886b251a0"
    assert "doi:10.13021/ORC2020/XXLVXM" in text
    assert '"reserved_sequences": "R31-R70"' in text


def test_prediction_first_information_order_is_explicit() -> None:
    _, text = _load()
    provider_index = text.index("  provider:")
    seal_index = text.index("  seal:")
    evaluate_index = text.index("  evaluate:")
    assert provider_index < seal_index < evaluate_index
    provider_text = text[provider_index:seal_index]
    assert "Predict R11-R30 from normal-view images only" in provider_text
    assert "two_dimensional_markers_opened" in provider_text
    assert "three_dimensional_markers_opened" in provider_text
    assert "Seal factors and query decisions from 2-D markers only" in text
    assert "Open R11-R30 3-D outcomes and score frozen predictions once" in text


def test_strong_prerequisite_and_terminal_classes_are_unchanged() -> None:
    _, text = _load()
    assert 'verified["decision"] != "heldout-strong-positive"' in text
    for decision in (
        "query-selective-strong-positive",
        "query-selective-bounded-positive",
        "query-selective-mixed-negative-or-insufficient-support",
        "technical-failure",
    ):
        assert decision in text
    assert "run_dot_rope_query_selective_heldout.py" in text
    assert "verify_dot_rope_cut3r_heldout_result.py" in text


def test_no_target_side_optimization_or_downstream_execution() -> None:
    _, text = _load()
    lowered = text.lower()
    for forbidden in (
        "optuna",
        "--tune",
        "scripts/run_bayesian_phystwin",
        "scripts/run_causal4d",
    ):
        assert forbidden not in lowered
    assert (
        "R31-R70 remained unopened. No target-side tuning, BayesianPhysTwin, "
        "or Causal4D execution was authorized."
    ) in text


def test_hosted_archive_materialization_is_checksum_bound_and_nonextracting() -> None:
    _, text = _load()
    assert text.count("materialize_verified_dot_archives.py") >= 3
    assert "--entry \"$R11_ARCHIVE=$R11_MD5\"" in text
    assert "--entry \"$R21_ARCHIVE=$R21_MD5\"" in text
    for forbidden in ("unzip ", "extractall", "zipfile"):
        assert forbidden not in text.lower()
