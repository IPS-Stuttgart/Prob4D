from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cut3r_guide_uses_the_existing_grouped_scaffold_route() -> None:
    text = (ROOT / "docs" / "cut3r-online-provider.md").read_text(encoding="utf-8")
    assert "prob4d prediction scaffold-generic" in text
    assert "--profile cut3r-online" in text
    assert "prob4d prediction scaffold-cut3r-online" not in text
