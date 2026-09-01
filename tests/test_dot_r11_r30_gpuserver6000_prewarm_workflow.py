from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r11-r30-gpuserver6000-prewarm-v1.yml"
REQUEST = "protocols/execution_requests/dot_r11_r30_gpuserver6000_prewarm_v1.json"


def _workflow() -> tuple[dict, str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    assert isinstance(value, dict)
    return value, text


def test_prewarm_has_one_main_only_additive_trigger() -> None:
    value, _ = _workflow()
    triggers = value.get("on", value.get(True))
    assert triggers == {
        "push": {
            "branches": ["main"],
            "paths": [REQUEST],
        }
    }
    assert value["permissions"] == {"contents": "read"}
    assert value["concurrency"]["cancel-in-progress"] is False


def test_prewarm_is_bound_to_gpuserver6000_and_trusted_environment() -> None:
    value, text = _workflow()
    job = value["jobs"]["prewarm"]
    assert job["runs-on"] == ["self-hosted", "gpuserver6000"]
    assert job["environment"] == "trusted-self-hosted-validation"
    assert job["permissions"] == {"contents": "read"}
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert "scientific_evaluation_authorized" in text
    assert '"scientific_evaluation_authorized": False' in text


def test_prewarm_pins_exact_provider_and_official_archives() -> None:
    _, text = _workflow()
    assert "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf" in text
    assert "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103" in text
    assert "R11-20.zip" in text
    assert "23ce3e7067465d3edabe20b4c7cfa388" in text
    assert "R21-30.zip" in text
    assert "8aee77f79d1aff6e1f3fd21886b251a0" in text
    assert "doi:10.13021/ORC2020/XXLVXM" in text


def test_prewarm_cannot_open_dot_or_run_science() -> None:
    _, text = _workflow()
    lowered = text.lower()
    forbidden = (
        "unzip ",
        "zipfile",
        "extractall",
        "run_dot_rope_query_selective_heldout.py predict",
        "run_dot_rope_query_selective_heldout.py seal",
        "run_dot_rope_query_selective_heldout.py evaluate",
    )
    for token in forbidden:
        assert token not in lowered
    for boundary in (
        '"archives_opened": False',
        '"normal_view_images_opened": False',
        '"two_dimensional_markers_opened": False',
        '"three_dimensional_markers_opened": False',
        '"scientific_predictions_constructed": False',
        '"scientific_evaluation_performed": False',
    ):
        assert boundary in text
    assert '"reserved_sequences": "R31-R70"' in text


def test_prewarm_requires_exactly_one_request_change() -> None:
    _, text = _workflow()
    assert 'if [[ ${#changed[@]} -ne 1 || "${changed[0]}" != "$REQUEST_PATH" ]]' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert 'test "$EVENT_BEFORE" != "0000000000000000000000000000000000000000"' in text
