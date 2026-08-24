from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_provider_status.py"
SNAPSHOT = ROOT / "evidence" / "provider_status_snapshot_v1.json"
README = ROOT / "README.md"


def test_provider_status_snapshot_and_readme_are_synchronized() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=True,
    )
    readme = README.read_text(encoding="utf-8")
    assert readme.count("<!-- provider-evidence-status:begin -->") == 1
    assert readme.count("<!-- provider-evidence-status:end -->") == 1


def test_provider_status_rejects_stale_source_identity(tmp_path: Path) -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["evidence_source"]["git_blob_sha1"] = "0" * 40
    tampered = tmp_path / "provider-status.json"
    tampered.write_text(json.dumps(snapshot), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(tampered),
            "--readme",
            str(README),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pinned provider status contract changed" in result.stderr


def test_provider_status_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"contract":"a","contract":"b"}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(duplicate),
            "--readme",
            str(README),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "duplicate JSON object key" in result.stderr
