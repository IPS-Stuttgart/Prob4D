from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from prob4d import _cut3r_source_preflight_environment as environment


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provider_fixture(tmp_path: Path) -> tuple[Path, Path, str, str, int]:
    checkout = tmp_path / "cut3r"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "tests@example.invalid")
    _git(checkout, "config", "user.name", "Prob4D tests")
    _git(checkout, "remote", "add", "origin", "https://github.com/CUT3R/CUT3R.git")
    (checkout / "demo.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n",
        encoding="utf-8",
    )
    _git(checkout, "add", "demo.py")
    _git(checkout, "commit", "-qm", "fixture")

    checkpoint = tmp_path / "model.pth"
    checkpoint_bytes = b"checkpoint"
    checkpoint.write_bytes(checkpoint_bytes)
    return (
        checkout,
        checkpoint,
        _git(checkout, "rev-parse", "HEAD"),
        hashlib.sha256(checkpoint_bytes).hexdigest(),
        len(checkpoint_bytes),
    )


def _inspect(
    checkout: Path,
    checkpoint: Path,
    revision: str,
    checkpoint_sha256: str,
    checkpoint_byte_count: int,
) -> dict[str, object]:
    return environment._cut3r_surface(
        checkout,
        checkpoint,
        expected_repository="CUT3R/CUT3R",
        expected_revision=revision,
        expected_checkpoint_filename="model.pth",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_checkpoint_byte_count=checkpoint_byte_count,
    )


def test_clean_exact_provider_authorizes_tracked_demo_probe(tmp_path: Path) -> None:
    checkout, checkpoint, revision, checkpoint_sha, checkpoint_bytes = _provider_fixture(tmp_path)

    surface = _inspect(
        checkout,
        checkpoint,
        revision,
        checkpoint_sha,
        checkpoint_bytes,
    )

    assert surface["executable_probe_authorized"] is True
    assert surface["worktree_clean_including_untracked"] is True
    assert surface["origin_repository"] == "CUT3R/CUT3R"
    assert surface["demo_relative_path"] == "demo.py"
    assert surface["demo_help_status"] == 0
    assert "origin_url" not in surface


def test_untracked_checkout_content_blocks_all_executable_probes(tmp_path: Path) -> None:
    checkout, checkpoint, revision, checkpoint_sha, checkpoint_bytes = _provider_fixture(tmp_path)
    (checkout / "torch.py").write_text(
        "raise RuntimeError('untracked shadow module executed')\n",
        encoding="utf-8",
    )

    surface = _inspect(
        checkout,
        checkpoint,
        revision,
        checkpoint_sha,
        checkpoint_bytes,
    )

    assert surface["worktree_clean_including_untracked"] is False
    assert surface["executable_probe_authorized"] is False
    assert surface["demo_relative_path"] is None
    assert surface["demo_help_status"] == 127
    assert surface["dependency_probe_status"] == 127


def test_wrong_checkpoint_blocks_all_executable_probes(tmp_path: Path) -> None:
    checkout, checkpoint, revision, checkpoint_sha, checkpoint_bytes = _provider_fixture(tmp_path)

    surface = environment._cut3r_surface(
        checkout,
        checkpoint,
        expected_repository="CUT3R/CUT3R",
        expected_revision=revision,
        expected_checkpoint_filename="model.pth",
        expected_checkpoint_sha256="0" * 64,
        expected_checkpoint_byte_count=checkpoint_bytes,
    )

    assert checkpoint_sha != "0" * 64
    assert surface["executable_probe_authorized"] is False
    assert surface["demo_relative_path"] is None
    assert surface["demo_help_status"] == 127
