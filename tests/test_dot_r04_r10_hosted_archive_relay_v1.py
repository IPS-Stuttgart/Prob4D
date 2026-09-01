from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "relay-dot-r04-r10-archive-v1.yml"
REQUEST = ROOT / "protocols" / "execution_requests" / "dot_r04_r10_hosted_archive_relay_v1.json"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _section(text: str, start: str, end: str | None = None) -> str:
    value = text[text.index(start) :]
    if end is not None:
        value = value[: value.index(end)]
    return value


def test_relay_control_plane_does_not_include_an_execution_request() -> None:
    assert WORKFLOW.is_file()
    assert not REQUEST.exists()


def test_relay_is_main_request_bound_and_target_identity_frozen() -> None:
    text = _text()
    assert "pull_request_target:" not in text
    assert "branches: [main]" in text
    assert 'protocols/execution_requests/dot_r04_r10_hosted_archive_relay_v1.json' in text
    assert 'TARGET_RUN_ID: "33434695566"' in text
    assert 'TARGET_JOB_ID: "99660292244"' in text
    assert 'TARGET_ATTEMPT: "2"' in text
    assert 'RECOVERY_JOB_ID: "99653502112"' in text
    assert 'MIN_RECOVERY_SECONDS: "3600"' in text
    assert "TARGET_HEAD_SHA: 9e1b77b2e70685881db7f188a95a3a91443275e8" in text
    assert 'RECOVERY_RUN_ID: "33442397966"' in text
    assert 'ARCHIVE_BYTES: "1408905061"' in text
    assert "ARCHIVE_MD5: ca546ff5f22c0279123ccb18509858ee" in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert 'len(changed)' not in text
    assert "hashlib.sha256(canonical(payload)).hexdigest() == request_id" in text


def test_relay_retires_only_overlong_transport_and_unopened_attempt2() -> None:
    text = _text()
    authorize = _section(text, "\n  authorize:", "\n  acquire:")
    assert "actions: write" in authorize
    assert "direct recovery succeeded; hosted relay is forbidden" in authorize
    assert "direct recovery has not exceeded the frozen relay threshold" in authorize
    assert "direct recovery advanced beyond archive acquisition" in authorize
    assert "target attempt 2 is no longer unopened and queued" in authorize
    assert "attempt-2 provider job already started" in authorize
    assert "cancel_and_wait(target_run_id" in authorize
    assert "cancel_and_wait(\n                  recovery_run_id" in authorize
    assert "provider_prediction_inside_relay" in authorize
    assert "marker_payload_access_inside_relay" in authorize

def test_raw_archive_download_and_install_are_checksum_bound() -> None:
    text = _text()
    acquire = _section(text, "\n  acquire:", "\n  install:")
    install = _section(text, "\n  install:", "\n  finalize:")
    assert "runs-on: ubuntu-latest" in acquire
    assert "official archive checksum changed" in acquire
    assert "official archive byte count changed" in acquire
    assert "--continue-at -" in acquire
    assert 'printf \'%s  %s\\n\' "$ARCHIVE_MD5" "$archive" | md5sum --check --strict' in acquire
    assert "split --bytes=\"$CHUNK_BYTES\"" in acquire
    assert "retention-days: 1" in acquire
    assert "compression-level: 0" in acquire
    assert "runs-on: [self-hosted, Linux, X64, gpuserver6000]" in install
    assert 'test "$RUNNER_NAME" = "workstation2"' in install
    assert "environment: trusted-self-hosted-validation" in install
    assert "permissions:\n      actions: read\n      contents: read" in install
    assert "actions: write" not in install
    assert "flock -x 9" in install
    assert "mv -f -- \"$ROOT/reconstructed.zip\" \"$CACHE_PATH\"" in install
    assert "raw_payload_uploaded_in_receipt" in install
    assert install.count("/usr/bin/python3 - <<'PY'") >= 2
    assert "rm -rf -- \"${{ steps.workspace.outputs.root }}\"" in install


def test_transient_raw_artifact_is_deleted_on_hosted_runner_before_exact_rerun() -> None:
    text = _text()
    finalize = _section(text, "\n  finalize:")
    assert "runs-on: ubuntu-latest" in finalize
    assert "permissions:\n      actions: write\n      contents: read" in finalize
    assert '/actions/artifacts/{artifact_id}' in finalize
    assert "method=\"DELETE\"" in finalize
    assert '/actions/jobs/{os.environ[\'TARGET_JOB_ID\']}/rerun' in finalize
    assert "target attempt 2 is not in the exact cancelled state" in finalize
    assert "rerun as attempt 3" in finalize
    assert "target run published evidence; automatic rerun refused" in finalize
    assert "secrets." not in text
    assert "git push" not in text
