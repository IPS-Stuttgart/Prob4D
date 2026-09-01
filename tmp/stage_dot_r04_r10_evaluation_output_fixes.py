#!/usr/bin/env python3
"""Stage no-clobber fixes for the DOT held-out evaluator and its recovery lane."""

from __future__ import annotations

from pathlib import Path

MAIN_WORKFLOW = Path(".github/workflows/dot-rope-cut3r-heldout-confirmation-v1.yml")
POSTPROCESS_WORKFLOW = Path(".github/workflows/dot-r04-r10-postprocess-v1.yml")
MAIN_TEST = Path("tests/test_dot_rope_cut3r_heldout_confirmation_workflow.py")
POSTPROCESS_TEST = Path("tests/test_dot_r04_r10_postprocess_workflow.py")

STAGED_MAIN_WORKFLOW = Path("tmp/patched-dot-rope-cut3r-heldout-confirmation-v1.yml")
STAGED_POSTPROCESS_WORKFLOW = Path("tmp/patched-dot-r04-r10-postprocess-v1.yml")
STAGED_MAIN_TEST = Path("tmp/patched-test-dot-rope-cut3r-heldout-confirmation-workflow.py")
STAGED_POSTPROCESS_TEST = Path("tmp/patched-test-dot-r04-r10-postprocess-workflow.py")

MAIN_OLD = '/usr/bin/mkdir -p "$root/provider" "$root/dataset" "$root/evaluation"'
MAIN_NEW = '/usr/bin/mkdir -p "$root/provider" "$root/dataset"'
POSTPROCESS_OLD = 'mkdir -p "$root/provider" "$root/dataset" "$root/evaluation"'
POSTPROCESS_NEW = 'mkdir -p "$root/provider" "$root/dataset"'

MAIN_REGRESSION = r'''


def test_confirmation_hosted_evaluator_reserves_output_creation_for_evaluator() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    evaluation = text[text.index("\n  evaluate:") : text.index("\n  publish:")]
    initialization = evaluation[
        evaluation.index("Initialize isolated evaluation workspace") : evaluation.index(
            "Check out exact authorized revision"
        )
    ]

    assert '"$root/evaluation"' not in initialization
    assert '/usr/bin/mkdir -p "$root/provider" "$root/dataset"' in initialization
    assert '--output-dir "$root/evaluation"' in evaluation
'''

POSTPROCESS_REGRESSION = r'''


def test_recovery_reserves_output_creation_for_frozen_evaluator() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    recover = _job(text, "recover_evaluation", "materialize")
    initialization = recover[
        recover.index("Initialize isolated hosted recovery workspace") : recover.index(
            "Download exact immutable provider artifact"
        )
    ]

    assert '"$root/evaluation"' not in initialization
    assert 'mkdir -p "$root/provider" "$root/dataset"' in initialization
    assert "--output-dir" in recover
    assert "/evaluation" in recover
'''


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


def append_once(text: str, addition: str, *, marker: str, label: str) -> str:
    if marker in text:
        raise SystemExit(f"{label}: regression already present")
    return text.rstrip() + addition + "\n"


def main() -> int:
    main_workflow = replace_once(
        MAIN_WORKFLOW.read_text(encoding="utf-8"),
        MAIN_OLD,
        MAIN_NEW,
        label="held-out evaluator workspace",
    )
    postprocess_workflow = replace_once(
        POSTPROCESS_WORKFLOW.read_text(encoding="utf-8"),
        POSTPROCESS_OLD,
        POSTPROCESS_NEW,
        label="postprocess evaluator workspace",
    )
    main_test = append_once(
        MAIN_TEST.read_text(encoding="utf-8"),
        MAIN_REGRESSION,
        marker="test_confirmation_hosted_evaluator_reserves_output_creation_for_evaluator",
        label="held-out evaluator regression",
    )
    postprocess_test = append_once(
        POSTPROCESS_TEST.read_text(encoding="utf-8"),
        POSTPROCESS_REGRESSION,
        marker="test_recovery_reserves_output_creation_for_frozen_evaluator",
        label="postprocess evaluator regression",
    )

    if '--output-dir "$root/evaluation"' not in main_workflow:
        raise SystemExit("held-out evaluator no longer targets the reserved output path")
    if "--output-dir" not in postprocess_workflow or "/evaluation" not in postprocess_workflow:
        raise SystemExit("postprocess evaluator no longer targets the reserved output path")

    STAGED_MAIN_WORKFLOW.write_text(main_workflow, encoding="utf-8")
    STAGED_POSTPROCESS_WORKFLOW.write_text(postprocess_workflow, encoding="utf-8")
    STAGED_MAIN_TEST.write_text(main_test, encoding="utf-8")
    STAGED_POSTPROCESS_TEST.write_text(postprocess_test, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
