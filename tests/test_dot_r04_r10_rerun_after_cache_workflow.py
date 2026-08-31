from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r04-r10-rerun-after-cache-v1.yml"


def _load() -> tuple[dict, str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    assert isinstance(value, dict)
    return value, text


def test_recovery_is_default_branch_workflow_run_only() -> None:
    value, text = _load()
    triggers = value.get("on", value.get(True))
    assert triggers["workflow_run"] == {
        "workflows": ["DOT R01-R10 gpuserver6000 cache prewarm v1"],
        "types": ["completed"],
    }
    recover = value["jobs"]["recover"]
    condition = recover["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in condition
    assert "github.event.workflow_run.event == 'push'" in condition
    assert "github.event.workflow_run.head_branch == 'main'" in condition
    assert "ref: main" in text


def test_recovery_is_bound_to_exact_frozen_target() -> None:
    value, text = _load()
    env = value["env"]
    assert env["TARGET_RUN_ID"] == "33434695566"
    assert env["TARGET_HEAD_SHA"] == "9e1b77b2e70685881db7f188a95a3a91443275e8"
    assert (
        env["TARGET_WORKFLOW_PATH"]
        == ".github/workflows/dot-rope-cut3r-heldout-confirmation-v1.yml"
    )
    assert (
        env["TARGET_PROVIDER_JOB"]
        == "Seal marker-free R04-R10 CUT3R predictions on gpuserver6000"
    )
    assert env["ARCHIVE_MD5"] == "ca546ff5f22c0279123ccb18509858ee"
    assert "frozen target revision changed" in text
    assert "frozen target workflow changed" in text


def test_cache_receipt_information_boundary_is_verified() -> None:
    _, text = _load()
    assert "prob4d.dot-r01-r10-gpuserver6000-cache-prewarm-result" in text
    assert "R01-10.zip" in text
    for boundary in (
        "archive_members_enumerated",
        "archive_extracted",
        "normal_view_images_opened",
        "two_dimensional_markers_opened",
        "three_dimensional_markers_opened",
        "scientific_prediction_constructed",
        "scientific_evaluation_performed",
    ):
        assert boundary in text
    assert "cache prewarm crossed its information boundary" in text


def test_valid_or_active_scientific_run_is_never_rerun() -> None:
    _, text = _load()
    assert 'decision = "no-op-target-still-active"' in text
    assert 'decision = "no-op-terminal-success"' in text
    assert "target already has a terminal result artifact; rerun forbidden" in text
    assert "frozen target exceeded bounded recovery attempts" in text
    assert "rerun-failed-jobs" in text
    assert "/rerun" not in text.replace("/rerun-failed-jobs", "")


def test_only_hosted_recovery_job_has_actions_write() -> None:
    value, _ = _load()
    assert value["jobs"]["contract"]["permissions"] == {"contents": "read"}
    assert value["jobs"]["recover"]["permissions"] == {
        "contents": "read",
        "actions": "write",
    }
    assert value["jobs"]["recover"]["runs-on"] == "ubuntu-latest"


def test_recovery_does_not_modify_repository_or_science() -> None:
    _, text = _load()
    lowered = text.lower()
    for forbidden in (
        "contents: write",
        "create-or-update-file",
        "git push",
        "protocols/dot-rope-cut3r-heldout-confirmation-v1.json",
        "--alpha",
        "--threshold",
        "--tune",
    ):
        assert forbidden not in lowered
    assert (
        "No protocol, request, provider, cohort, threshold, marker order, "
        "prediction, or outcome was changed."
    ) in text
