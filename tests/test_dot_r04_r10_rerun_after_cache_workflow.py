from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r04-r10-rerun-after-cache-v1.yml"


def _load() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(text: str, name: str, *, next_name: str | None = None) -> str:
    start = text.index(f"\n  {name}:")
    end = text.index(f"\n  {next_name}:", start) if next_name is not None else len(text)
    return text[start:end]


def test_recovery_is_default_branch_workflow_run_only() -> None:
    text = _load()
    trigger = (
        '  workflow_run:\n'
        '    workflows: ["DOT R01-R10 gpuserver6000 cache prewarm v1"]\n'
        '    types: [completed]\n'
    )
    assert trigger in text
    recover = _job(text, "recover")
    assert "github.event.workflow_run.conclusion == 'success'" in recover
    assert "github.event.workflow_run.event == 'push'" in recover
    assert "github.event.workflow_run.head_branch == 'main'" in recover
    assert "ref: main" in recover


def test_recovery_is_bound_to_exact_frozen_target() -> None:
    text = _load()
    assert 'TARGET_RUN_ID: "33434695566"' in text
    assert "TARGET_HEAD_SHA: 9e1b77b2e70685881db7f188a95a3a91443275e8" in text
    assert (
        "TARGET_WORKFLOW_PATH: "
        ".github/workflows/dot-rope-cut3r-heldout-confirmation-v1.yml"
    ) in text
    assert (
        "TARGET_PROVIDER_JOB: "
        "Seal marker-free R04-R10 CUT3R predictions on gpuserver6000"
    ) in text
    assert "ARCHIVE_MD5: ca546ff5f22c0279123ccb18509858ee" in text
    assert "frozen target revision changed" in text
    assert "frozen target workflow changed" in text


def test_cache_receipt_information_boundary_is_verified() -> None:
    text = _load()
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
    text = _load()
    assert 'decision = "no-op-target-still-active"' in text
    assert 'decision = "no-op-terminal-success"' in text
    assert "target already has a terminal result artifact; rerun forbidden" in text
    assert "frozen target exceeded bounded recovery attempts" in text
    assert "rerun-failed-jobs" in text
    assert "/rerun" not in text.replace("/rerun-failed-jobs", "")


def test_only_hosted_recovery_job_has_actions_write() -> None:
    text = _load()
    contract = _job(text, "contract", next_name="recover")
    recover = _job(text, "recover")

    assert "runs-on: ubuntu-latest" in contract
    assert "permissions:\n      contents: read" in contract
    assert "actions: write" not in contract

    assert "runs-on: ubuntu-latest" in recover
    assert "permissions:\n      contents: read\n      actions: write" in recover


def test_recovery_does_not_modify_repository_or_science() -> None:
    text = _load()
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
