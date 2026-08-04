from __future__ import annotations

from pathlib import Path


def main() -> None:
    source = Path("src/prob4d/uncertainty.py")
    text = source.read_text(encoding="utf-8")
    old = "        group_ids = tuple(map(str, self.group_ids))\n"
    new = (
        "        if not isinstance(self.group_ids, tuple) or any(\n"
        "            type(value) is not str for value in self.group_ids\n"
        "        ):\n"
        "            raise TypeError(\n"
        "                \"group_ids must be a canonical tuple of strings\"\n"
        "            )\n"
        "        group_ids = self.group_ids\n"
    )
    if text.count(old) != 1:
        raise SystemExit("group_ids normalization anchor changed")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")

    tests = Path("tests/test_scientific_scalar_validation.py")
    text = tests.read_text(encoding="utf-8")
    anchor = (
        "def test_group_report_rejects_fractional_group_count_without_truncation() -> None:\n"
        "    with pytest.raises(TypeError, match=r\"group_counts\\[0\\].*genuine integer\"):\n"
        "        _group_report(group_counts=(1.5, 1))\n\n\n"
    )
    addition = anchor + (
        "@pytest.mark.parametrize(\n"
        "    \"group_ids\",\n"
        "    (\n"
        "        (1, \"b\"),\n"
        "        [\"a\", \"b\"],\n"
        "        {\"a\", \"b\"},\n"
        "        \"ab\",\n"
        "    ),\n"
        ")\n"
        "def test_group_report_rejects_noncanonical_group_ids(\n"
        "    group_ids: object,\n"
        ") -> None:\n"
        "    with pytest.raises(TypeError, match=\"canonical tuple of strings\"):\n"
        "        _group_report(group_ids=group_ids)\n\n\n"
    )
    if text.count(anchor) != 1:
        raise SystemExit("group-id test insertion anchor changed")
    tests.write_text(text.replace(anchor, addition, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
