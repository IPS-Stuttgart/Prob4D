"""Static contract for the one-time DOT held-out queued-run recovery helper."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-rope-cut3r-heldout-queued-recovery-v1.yml"


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "protocols/execution_requests/dot_rope_cut3r_heldout_queued_recovery_v1.json",
        "protocols/execution_requests/dot_rope_cut3r_heldout_confirmation_v1.json",
        'SUPERSEDED_RUN_ID: "33363832286"',
        "SUPERSEDED_HEAD_SHA: bb2179158b27178c6ebed9be866bee829108b72a",
        "RETAINED_REQUEST_ID: 62d64df1b1b72f2b2aff0b17cf4c7aad245150f9fa1ff67712eedc0f4e109ce6",
        "RETAINED_PROTOCOL_ID: a83258295d5ecabd95017a775f334173bb48141918832fb1a065a1dff66d16ba",
        "EXPECTED_PROVIDER_JOB: Seal marker-free R04-R10 CUT3R predictions",
        "actions: write",
        '"schema_version": 2',
        'run["status"] != "queued"',
        'job["status"] != "queued"',
        'job.get("steps") not in (None, [])',
        'job.get("completed_at") is not None',
        'job.get("conclusion") is not None',
        'job.get("runner_id") not in (None, 0)',
        'job.get("runner_name") not in (None, "")',
        'expected_labels = {"self-hosted", "Linux", "X64", "gpuserver4090"}',
        'artifacts.get("total_count", 0)',
        'f"/repos/{repo}/actions/runs/{run_id}/cancel"',
        'f"/repos/{repo}/actions/runs/{run_id}/rerun"',
        'latest.get("run_attempt", 0)',
        '"scientific_inputs_changed": False',
        '"provider_payload_opened_before_recovery": False',
        '"marker_payload_opened_before_recovery": False',
    )
    for needle in required:
        assert needle in text, needle

    # GitHub may populate started_at at queue creation even while runner_id is 0.
    # Assignment/execution is therefore guarded by runner identity, steps,
    # completion, and queue state rather than by started_at alone.
    assert 'job.get("started_at") is not None' not in text

    # The recovery helper itself is hosted and must not gain DOT/GPU access.
    assert "DATASET_ROOT" not in text
    assert "/mnt/" not in text
    assert "nvidia-smi" not in text

    # The push authorization is intentionally a one-file request trigger.
    assert '[[ ${#changed[@]} -ne 1 || "${changed[0]}" != "$RECOVERY_REQUEST_PATH" ]]' in text
    assert "cancel-in-progress: false" in text


if __name__ == "__main__":
    main()
