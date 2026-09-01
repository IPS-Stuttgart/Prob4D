from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dot_support_negative_maps_to_expected_failed_process_outcome() -> None:
    paths = [
        ROOT / ".github/workflows/dot-rope-cut3r-heldout-confirmation-v1.yml",
        ROOT / ".github/workflows/dot-r04-r10-postprocess-v1.yml",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert 'decision in {"heldout-support-negative", "technical-failure"}' in text
        assert 'else "success"' in text
        assert 'os.environ["EXECUTION_OUTCOME"] != expected_outcome' in text
