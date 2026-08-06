from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.prediction_provider_import import (
    PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA,
    PREDICTION_PROVIDER_IMPORT_SPEC_VERSION,
)
from prob4d.prediction_provider_scaffold import (
    scaffold_prediction_provider_import,
)


def test_scaffold_is_explicitly_incomplete_and_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "external-provider"
    specification, readme = scaffold_prediction_provider_import(destination)

    assert specification == destination / "provider-import.json"
    assert readme == destination / "README.md"
    assert (destination / "windows").is_dir()
    record = json.loads(specification.read_text(encoding="utf-8"))
    assert record["schema"] == PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA
    assert record["schema_version"] == PREDICTION_PROVIDER_IMPORT_SPEC_VERSION
    assert record["provider_revision"].startswith("REPLACE_WITH_")
    assert record["payloads"][0]["path"] == "windows/window_0000.npz"
    assert not (destination / "windows/window_0000.npz").exists()
    assert "intentionally **not importable yet**" in readme.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        scaffold_prediction_provider_import(destination)
    assert record == json.loads(specification.read_text(encoding="utf-8"))


def test_scaffold_rejects_symlink_destination(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    destination = tmp_path / "link"
    try:
        destination.symlink_to(real_directory, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="destination is a symbolic link"):
        scaffold_prediction_provider_import(destination)
