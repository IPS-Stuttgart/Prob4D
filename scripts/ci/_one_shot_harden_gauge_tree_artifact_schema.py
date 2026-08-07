#!/usr/bin/env python3
"""Reject Boolean schema-version aliases in the portable gauge-tree artifact."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/prob4d/_gauge_tree_artifact_common.py"
TEST = ROOT / "tests/test_gauge_tree_prior_artifact.py"


def _replace(path: Path, old: str, new: str, *, expected_count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        MODULE,
        """        if value.get("schema_version") != GAUGE_TREE_PRIOR_ARTIFACT_VERSION:
            raise ValueError("unsupported gauge-tree prior artifact version")
""",
        """        schema_version = require_positive_integer(
            value.get("schema_version"),
            name="schema_version",
        )
        if schema_version != GAUGE_TREE_PRIOR_ARTIFACT_VERSION:
            raise ValueError("unsupported gauge-tree prior artifact version")
""",
    )

    text = TEST.read_text(encoding="utf-8")
    if "test_loader_rejects_boolean_schema_version" in text:
        raise SystemExit("Boolean schema-version regression already exists")
    marker = """def test_loader_rejects_unknown_fields_and_duplicate_keys(tmp_path: Path) -> None:
"""
    insertion = """def test_loader_rejects_boolean_schema_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = True
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        load_gauge_tree_prior_artifact(manifest_path)


"""
    if text.count(marker) != 1:
        raise SystemExit("cannot locate gauge-tree test insertion point")
    TEST.write_text(text.replace(marker, insertion + marker), encoding="utf-8")


if __name__ == "__main__":
    main()
