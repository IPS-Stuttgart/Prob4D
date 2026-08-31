from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/tracking-cloth-finite-orbit-hosted-recovery-v1.md"


def test_hosted_recovery_document_preserves_claim_boundary() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "64 shaking and twisting source recordings" in text
    assert "56 table-collision, stick-hitting, and self-collision recordings" in text
    assert "publish compact evidence only" in text
    assert "does not test a learned visual provider" in text
    assert "deployment calibration or safety" in text
