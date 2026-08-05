"""Apply the reviewed PR 95 provenance repairs exactly once."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, *, name: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name} anchor changed in {path}: occurrences={count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "src/prob4d/material_identity_stream.py",
    "zip(result, result[1:])):",
    "zip(result, result[1:], strict=False)):",
    name="strict integer-tuple ordering",
)
replace_once(
    "src/prob4d/material_identity_stream.py",
    '''def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
''',
    '''def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value
''',
    name="canonical string validation",
)
replace_once(
    "src/prob4d/material_identity_stream.py",
    '''def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
''',
    '''def _git_sha(value: Any, *, name: str) -> str:
    revision = _strict_string(value, name=name)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return revision


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
''',
    name="Git revision validator",
)
replace_once(
    "src/prob4d/material_identity_stream.py",
    '''            "source_revision": _strict_string(
                self.source_revision,
                name="source_revision",
            ),
''',
    '''            "source_revision": _git_sha(
                self.source_revision,
                name="source_revision",
            ),
''',
    name="source revision admission",
)
replace_once(
    "src/prob4d/material_identity_stream.py",
    '''        }
        if type(self.updates) is not tuple or not all(
''',
    '''        }
        repository = identifiers["source_repository"]
        if (
            repository.count("/") != 1
            or repository.startswith("/")
            or repository.endswith("/")
        ):
            raise ValueError("source_repository must have owner/name form")
        if type(self.updates) is not tuple or not all(
''',
    name="repository identity admission",
)
replace_once(
    "tests/test_material_identity_stream.py",
    'source_revision="2d6df37",',
    'source_revision="2d6df37" + "0" * 33,',
    name="exact revision fixture",
)
replace_once(
    "tests/test_material_identity_stream.py",
    '''def test_root_contract_rejects_coercion_aliases() -> None:
''',
    '''def test_root_contract_requires_canonical_provenance() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        create_material_identity_stream(
            sequence_id="sequence",
            case_id="case",
            stream_id="camera0",
            source_repository="Prob4D",
            source_revision="a" * 40,
            root_window_id="w0",
        )
    with pytest.raises(ValueError, match="40-character Git SHA"):
        create_material_identity_stream(
            sequence_id="sequence",
            case_id="case",
            stream_id="camera0",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision="revision",
            root_window_id="w0",
        )


def test_root_contract_rejects_coercion_aliases() -> None:
''',
    name="canonical provenance regressions",
)
replace_once(
    "docs/material-identity-stream.md",
    '    source_revision="<exact-revision>",',
    '    source_revision="<40-character lowercase Git SHA>",',
    name="revision example",
)
replace_once(
    "docs/material-identity-stream.md",
    '''```

Pairwise association results used for append must be directed as:
''',
    '''```

Root provenance is canonical and content-bearing: `source_repository` must use
exact `owner/name` form and `source_revision` must be a lowercase 40-character
Git SHA. Branch names, tags, abbreviated SHAs, and whitespace-padded aliases fail
closed so the stream cannot claim an ambiguous source revision.

Pairwise association results used for append must be directed as:
''',
    name="provenance documentation",
)
