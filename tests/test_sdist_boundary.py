from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_excludes_repository_automation_and_generated_evidence() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft docs" in manifest
    assert "graft protocols" in manifest
    for excluded in (
        "graft .github",
        "graft evidence",
        "graft environments",
        "graft requirements",
        "graft scripts",
        "graft tests",
    ):
        assert excluded not in manifest


def test_sdist_audit_installs_and_smoke_tests_the_archive() -> None:
    audit = (ROOT / "scripts" / "ci" / "check_sdist.py").read_text(encoding="utf-8")

    assert "FORBIDDEN_PREFIXES" in audit
    assert "venv.EnvBuilder" in audit
    assert '"-m", "pip", "install"' in audit
    assert "prob4d.api.v1" in audit
    assert "REPRESENTATIVE_TESTS" not in audit
