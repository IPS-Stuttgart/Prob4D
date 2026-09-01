#!/usr/bin/env python3
"""Apply the bounded DOT held-out provider output handoff repair."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / "scripts/science/run_dot_rope_cut3r_heldout_confirmation.py"
    text = path.read_text(encoding="utf-8")
    predict_anchor = "def predict(args: argparse.Namespace) -> int:\n"
    paired_anchor = "\n\ndef _paired_difference(\n"
    helper = '''def _prepare_exclusive_output(path: Path) -> Path:
    """Remove only a pre-created empty directory before exclusive creation."""

    if path.is_symlink():
        raise FileExistsError(f"provider output path is a symbolic link: {path}")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"provider output path is not empty: {path}")
        path.rmdir()
    return path


'''
    if text.count(predict_anchor) != 1 or "def _prepare_exclusive_output" in text:
        raise SystemExit("held-out predictor shape changed")
    start = text.index(predict_anchor)
    end = text.index(paired_anchor, start)
    prefix = text[:start]
    predict = text[start:end]
    suffix = text[end:]

    old_adapted = "    adapted = _base_protocol(protocol)\n"
    new_adapted = (
        "    output_dir = _prepare_exclusive_output(args.output_dir)\n"
        "    adapted = _base_protocol(protocol)\n"
    )
    if predict.count(old_adapted) != 1:
        raise SystemExit("held-out predict protocol assignment changed")
    predict = predict.replace(old_adapted, new_adapted, 1)

    old_output = "                output_dir=args.output_dir,\n"
    new_output = "                output_dir=output_dir,\n"
    if predict.count(old_output) != 1:
        raise SystemExit("held-out provider output argument changed")
    predict = predict.replace(old_output, new_output, 1)
    path.write_text(prefix + helper + predict + suffix, encoding="utf-8", newline="\n")

    test = ROOT / "tests/test_dot_rope_cut3r_heldout_output.py"
    test.write_text(
        '''from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_cut3r_heldout_confirmation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_heldout_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precreated_empty_output_is_removed_for_exclusive_provider_creation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output = tmp_path / "provider"
    output.mkdir()

    assert module._prepare_exclusive_output(output) == output
    assert not output.exists()


def test_nonempty_or_symbolic_output_fails_closed(tmp_path: Path) -> None:
    module = _load_script()
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        module._prepare_exclusive_output(nonempty)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "preserve"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symbolic link"):
        module._prepare_exclusive_output(link)
''',
        encoding="utf-8",
        newline="\n",
    )

    changelog = ROOT / "CHANGELOG.d/dot-r04-r10-provider-output-v1.md"
    changelog.write_text(
        "# Fixed\n\n"
        "- Permit the frozen DOT R04–R10 wrapper to remove only a pre-created "
        "empty provider directory before the hash-pinned predictor performs "
        "its exclusive creation. Nonempty, file, and symbolic-link outputs "
        "still fail closed; all scientific inputs remain unchanged.\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
