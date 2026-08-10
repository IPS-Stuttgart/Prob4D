from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_excludes_repository_automation_and_generated_evidence() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft docs" in manifest
    assert "graft protocols" in manifest
    for directory in (
        ".github",
        "evidence",
        "environments",
        "requirements",
        "scripts",
        "tests",
    ):
        assert f"graft {directory}" not in manifest
        assert f"prune {directory}" in manifest


def test_sdist_audit_installs_and_smoke_tests_the_archive() -> None:
    audit = (ROOT / "scripts" / "ci" / "check_sdist.py").read_text(encoding="utf-8")

    assert "FORBIDDEN_PREFIXES" in audit
    assert "venv.EnvBuilder(with_pip=True)" in audit
    assert '"-m", "pip", "install"' in audit
    assert '"-m", "pip", "check"' in audit
    assert '"--no-deps"' not in audit
    assert "system_site_packages=True" not in audit
    assert "prob4d.api.v1" in audit
    assert "REPRESENTATIVE_TESTS" not in audit


def test_specialized_workflows_use_the_installed_sdist_boundary() -> None:
    workflows = (
        ROOT / ".github" / "workflows" / "finite-sample-capability.yml",
        ROOT / ".github" / "workflows" / "visual-bias-calibration.yml",
    )
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert 'python scripts/ci/check_sdist.py "$archive"' in text
        assert "grep -q '/tests/test_" not in text
        assert 'name.endswith("/tests/' not in text
        assert 'assert not any("/tests/" in name for name in names)' in text
