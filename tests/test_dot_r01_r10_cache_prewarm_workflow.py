from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r01-r10-gpuserver6000-cache-prewarm-v1.yml"
REQUEST = (
    "protocols/execution_requests/"
    "dot_r01_r10_gpuserver6000_cache_prewarm_v1.json"
)


def _load() -> tuple[dict, str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    assert isinstance(value, dict)
    return value, text


def test_cache_prewarm_has_one_main_only_request_trigger() -> None:
    value, text = _load()
    triggers = value.get("on", value.get(True))
    assert triggers["push"] == {"branches": ["main"], "paths": [REQUEST]}
    assert value["permissions"] == {"contents": "read"}
    assert value["concurrency"]["cancel-in-progress"] is False
    assert 'if [[ ${#changed[@]} -ne 1 || "${changed[0]}" != "$REQUEST_PATH" ]]' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text


def test_cache_prewarm_targets_exact_frozen_run_cache() -> None:
    value, text = _load()
    env = value["env"]
    assert (
        env["CACHE_ROOT"]
        == "/home/github-runner/.cache/prob4d/"
        "dot-r04-r10-cut3r-gpuserver6000-v1/dot-v29"
    )
    assert env["ARCHIVE_NAME"] == "R01-10.zip"
    assert env["ARCHIVE_MD5"] == "ca546ff5f22c0279123ccb18509858ee"
    assert "doi:10.13021/ORC2020/XXLVXM" in text
    assert '"frozen_confirmation_sequences": [' in text
    assert '"reserved_sequences": "R11-R70"' in text


def test_cache_prewarm_is_bound_to_trusted_gpuserver6000() -> None:
    value, text = _load()
    job = value["jobs"]["prewarm"]
    assert job["runs-on"] == ["self-hosted", "gpuserver6000"]
    assert job["environment"] == "trusted-self-hosted-validation"
    assert job["permissions"] == {"contents": "read"}
    assert 'test "$RUNNER_NAME" = "workstation2"' in text


def test_cache_prewarm_is_resumable_and_never_opens_archive_members() -> None:
    _, text = _load()
    assert "--continue-at" in text
    assert "--retry-all-errors" in text
    assert "md5sum --check --strict" in text
    lowered = text.lower()
    for forbidden in (
        "unzip ",
        "zipfile",
        "extractall",
        "run_dot_rope_cut3r_heldout_confirmation.py predict",
        "run_dot_rope_cut3r_heldout_confirmation.py evaluate",
    ):
        assert forbidden not in lowered
    for boundary in (
        '"archive_members_enumerated": False',
        '"archive_extracted": False',
        '"normal_view_images_opened": False',
        '"two_dimensional_markers_opened": False',
        '"three_dimensional_markers_opened": False',
        '"scientific_prediction_constructed": False',
        '"scientific_evaluation_performed": False',
    ):
        assert boundary in text


def test_request_explicitly_forbids_scientific_access() -> None:
    _, text = _load()
    for field in (
        '"archive_may_be_extracted": False',
        '"images_may_be_opened": False',
        '"markers_may_be_opened": False',
        '"scientific_evaluation_authorized": False',
    ):
        assert field in text
