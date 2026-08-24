from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "cut3r-exact-retry-invocation.md"
REQUEST = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_deform360_source_freeze_v2.json"
)


def test_exact_retry_invocation_binds_the_retained_zero_evidence_request() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    request = REQUEST.read_text(encoding="utf-8")

    assert "8b923e8cd67ca65f09312cffe305e36852f36fbb" in text
    assert "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e" in text
    assert "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e" in request
    assert "20 minutes" in text
    assert "target-closed" in text
