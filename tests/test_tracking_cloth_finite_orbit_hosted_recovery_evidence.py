from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "evidence/tracking-cloth-finite-orbit-hosted-recovery-v1/README.md"


def test_recovery_pr_contains_no_claimed_result() -> None:
    text = README.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "No scientific result is committed" in text
    assert "separate request-only commit" in text
    assert "positive, negative, or technical" in normalized
