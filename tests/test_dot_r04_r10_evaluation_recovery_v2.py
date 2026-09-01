from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r04-r10-evaluation-recovery-v2.yml"
REQUEST = ROOT / "protocols/execution_requests/dot_r04_r10_evaluation_recovery_v2.json"


def _request() -> dict[str, object]:
    value = json.loads(REQUEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_recovery_request_is_content_addressed_and_exact() -> None:
    request = _request()
    supplied = request["request_id"]
    unsigned = dict(request)
    unsigned.pop("request_id")
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == supplied
    assert supplied == "5f60d7494dbc67b694b0d011283a0b5bc3093b190c3f1e7fd1978549f48f1051"
    assert request["target_run_id"] == 33500239597
    assert request["target_run_attempt"] == 1
    assert request["target_head_sha"] == "b85dd8e8adb68ffb289868731222151968940317"
    assert request["provider_job_id"] == 99831732392
    assert request["evaluation_job_id"] == 99832885343
    assert request["provider_artifact"] == {
        "digest": "sha256:45cccd5b6f7b7d671cfc86fbcb27232d29faf5dafdeee8d0ea5a9c45ce42ae7c",
        "id": 9797527843,
        "name": (
            "dot-rope-cut3r-heldout-provider-gpuserver6000-"
            "62d64df1b1b72f2b2aff0b17cf4c7aad245150f9fa1ff67712eedc0f4e109ce6"
        ),
        "provider_bundle_id": ("57dc11d9e39258a2f620d67e39f1176cafe74252173ead8cb4ba2f76083499ec"),
    }


def test_recovery_is_hosted_only_and_never_reruns_provider() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in text
    assert "self-hosted" not in text
    assert "gpuserver4090" not in text
    assert "gpuserver6000]" not in text
    assert 'provider_rerun_authorized": false' in REQUEST.read_text(encoding="utf-8")
    assert 'provider_rerun": False' in text
    assert "actions: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "secrets." not in text


def test_recovery_binds_exact_failure_and_sealed_artifact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for value in (
        "33500239597",
        "b85dd8e8adb68ffb289868731222151968940317",
        "99831732392",
        "99832885343",
        "9797527843",
        "45cccd5b6f7b7d671cfc86fbcb27232d29faf5dafdeee8d0ea5a9c45ce42ae7c",
        "57dc11d9e39258a2f620d67e39f1176cafe74252173ead8cb4ba2f76083499ec",
        "held-out evaluator emitted no result payload",
    ):
        assert value in text
    request_text = REQUEST.read_text(encoding="utf-8")
    assert "FileExistsError" in request_text
    assert 'request_payload["failure_contract"]["exact_error_fragment"]' in text
    assert "a terminal evaluation artifact already exists" in text
    assert "provider artifact digest changed" in text


def test_provider_seal_precedes_marker_access_and_output_is_exclusive() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    recover = text[text.index("\n  recover:") :]
    workspace = recover[
        recover.index("Initialize output-exclusive hosted workspace") : recover.index(
            "Download exact immutable provider artifact"
        )
    ]

    assert 'mkdir -p "$root/dataset"' in workspace
    assert 'test ! -e "$root/provider"' in workspace
    assert 'test ! -e "$root/evaluation"' in workspace
    assert 'mkdir -p "$root/evaluation"' not in workspace
    assert recover.index("Reverify provider seal before marker download") < recover.index(
        "Download and verify exact official marker archive"
    )
    assert recover.index("Download and verify exact official marker archive") < recover.index(
        "Execute exact frozen evaluator once"
    )
    assert 'assert seal["markers_opened"] is False' in recover
    assert 'assert boundary["two_dimensional_markers_opened"] is False' in recover
    assert 'assert boundary["three_dimensional_markers_opened"] is False' in recover


def test_recovery_preserves_frozen_scientific_decision_mapping() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"heldout-strong-positive": 0' in text
    assert '"heldout-directional-positive": 0' in text
    assert '"heldout-mixed-or-negative": 0' in text
    assert '"heldout-support-negative": 4' in text
    assert '"technical-failure": 3' in text
    assert '"confirmation_retuning_performed": False' in text
    assert '"scientific_inputs_changed": False' in text
    assert '"r11_r70_opened": False' in text
    assert "verify_dot_rope_cut3r_heldout_result.py" in text


def test_recovery_authorization_checkout_fetches_base_commit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inspect = text[text.index("\n  inspect:") : text.index("\n  recover:")]
    checkout = inspect[
        inspect.index("Check out exact merged recovery revision") : inspect.index(
            "Require ordinary main request change"
        )
    ]

    assert "fetch-depth: 0" in checkout
    assert "BASE_SHA: ${{ github.event.before }}" in inspect
    assert 'git diff-tree --no-commit-id --name-only -r "$BASE_SHA" "$HEAD_SHA"' in inspect


def test_recovery_log_redirect_does_not_forward_github_credentials() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inspect = text[text.index("\n  inspect:") : text.index("\n  recover:")]

    assert "import urllib.parse" in inspect
    assert "class CrossOriginSafeRedirectHandler" in inspect
    assert 'if target.scheme != "https":' in inspect
    assert '"authorization"' in inspect
    assert '"x-github-api-version"' in inspect
    assert "if key.lower() in sensitive:" in inspect
    assert "redirected.remove_header(key)" in inspect
    assert "urllib.request.build_opener(CrossOriginSafeRedirectHandler())" in inspect
    assert "urllib.request.urlopen(log_request, timeout=120)" not in inspect
