from pathlib import Path
from typing import Any

import pytest

import prob4d.promotion_evidence as promotion_evidence


def test_atomic_text_preserves_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "card.md"
    original_link = promotion_evidence.os.link

    def publish_competitor(
        source_path: Any,
        destination_path: Any,
    ) -> None:
        Path(destination_path).write_text(
            "concurrent writer\n",
            encoding="utf-8",
        )
        original_link(source_path, destination_path)

    monkeypatch.setattr(promotion_evidence.os, "link", publish_competitor)

    with pytest.raises(FileExistsError):
        promotion_evidence._atomic_write_text(destination, "our evidence\n")

    assert destination.read_text(encoding="utf-8") == "concurrent writer\n"
    assert not list(tmp_path.glob(f".{destination.name}.tmp-*"))


def test_atomic_json_is_canonical_and_non_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "card.json"

    promotion_evidence._atomic_write_json(destination, {"b": 2, "a": 1})

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": 1,\n  "b": 2\n}\n'
    )
    with pytest.raises(FileExistsError):
        promotion_evidence._atomic_write_json(destination, {"a": 3})
