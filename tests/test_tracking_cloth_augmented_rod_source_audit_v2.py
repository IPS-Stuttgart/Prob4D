from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_tracking_cloth_augmented_rod_source_v2.py"


def module():
    spec = importlib.util.spec_from_file_location("augmented_rod_source_audit_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_wrapper_binds_exact_reviewed_source_audit() -> None:
    audit = module()
    base = audit._load_base()
    assert base.RESULT_SCHEMA == "prob4d.tracking-cloth-augmented-rod-source-audit-result.v1"
    assert audit.BASE_GIT_BLOB_SHA1 == "e9c5eb809f82fed679144e5b0dff70ba693175d7"
