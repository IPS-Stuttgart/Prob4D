from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/tracking-cloth-finite-orbit-hosted-recovery-v1.yml"
REQUEST = (
    ROOT
    / "protocols/execution_requests/tracking_cloth_finite_orbit_hosted_recovery_v1.json"
)


def _job(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"\n  {name}:")
    end = text.index(f"\n  {next_name}:", start) if next_name else len(text)
    return text[start:end]


def test_pull_request_review_is_separate_from_main_only_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prefix = text.split("permissions:", maxsplit=1)[0]
    assert "\n  pull_request:" in prefix
    assert "\n  push:" in prefix
    assert "branches: [main]" in prefix
    assert REQUEST.relative_to(ROOT).as_posix() in prefix
    contract = _job(text, "contract", "authorize")
    authorize = _job(text, "authorize", "evaluate")
    evaluate = _job(text, "evaluate")
    assert "github.event_name == 'pull_request'" in contract
    assert "github.event_name == 'push'" in authorize
    assert "github.event_name == 'push'" in evaluate
    assert "runs-on: ubuntu-latest" in contract
    assert "runs-on: ubuntu-latest" in authorize
    assert "runs-on: ubuntu-latest" in evaluate
    assert "runs-on: [self-hosted" not in text
    assert "runs-on: self-hosted" not in text


def test_recovery_requires_an_unassigned_artifact_free_predecessor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    authorize = _job(text, "authorize", "evaluate")
    assert 'ORIGINAL_RUN_ID: "33361712662"' in text
    assert "5c4e3b4ebfe38a10b1d1635db62b23c4f1736770" in text
    assert "Evaluate held-out collision geometry on gpuserver4090" in text
    assert 'job.get("steps")' in authorize
    assert 'job.get("runner_id") not in (None, 0)' in authorize
    assert 'job.get("runner_name") not in (None, "")' in authorize
    assert "if artifacts:" in authorize
    assert "/actions/runs/{run_id}/cancel" in authorize
    assert 'run.get("conclusion") != "cancelled"' in authorize
    assert "actions: write" in authorize
    assert "contents: write" not in authorize


def test_official_release_and_information_order_are_fixed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    evaluate = _job(text, "evaluate")
    assert (
        "https://zenodo.org/records/14644526/files/tracking_dataset.zip?download=1"
        in text
    )
    assert "b4868b702f8a42b2ea1069d0f1a3b8f6" in text
    assert 'test "$count" -eq 120' in evaluate
    assert "run_tracking_cloth_finite_orbit_real_v1.py" in evaluate
    assert "source-sealed held-out evaluation" in evaluate
    assert "Remove raw release before publication" in evaluate
    assert "Raw dataset payload appeared in evidence" in evaluate
    assert 'result["aggregate"]["target_groups"] == 56' in evaluate
    assert 'result["aggregate"]["total_cases"] >= 1000' in evaluate
    assert (
        'result["claim_boundary"]["learned_visual_provider"] is False' in evaluate
    )


def test_external_actions_are_pinned_to_full_commit_shas() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        target = stripped.split(maxsplit=1)[1]
        if target.startswith("./"):
            continue
        revision = target.rsplit("@", maxsplit=1)[1].split(maxsplit=1)[0]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_request_is_not_part_of_the_reviewed_control_plane() -> None:
    assert not REQUEST.exists()
