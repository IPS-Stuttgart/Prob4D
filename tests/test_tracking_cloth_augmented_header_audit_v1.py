from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "science"
    / "audit_tracking_cloth_augmented_headers_v1.py"
)


def module():
    spec = importlib.util.spec_from_file_location("augmented_header_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_public_path_metadata_classification() -> None:
    audit = module()
    assert audit._material_size_category(
        "tracking_dataset/Self-collisions/cotton_A2_self_collision.csv"
    ) == ("cotton", "A2", "self-collision")
    assert audit._material_size_category(
        "tracking_dataset/Tablecloth/wool_A2_full_lay_high_friction.csv"
    ) == ("wool", "A2", "tablecloth")
    assert audit._material_size_category(
        "tracking_dataset/Hitting/polyester_A2_hitting.csv"
    ) == ("polyester", "A2", "hitting")


def test_audit_file_uses_header_only(monkeypatch, tmp_path: Path) -> None:
    audit = module()
    root = tmp_path / "dataset"
    path = root / "Self-collisions" / "denim_A2_self_collisions.csv"
    path.parent.mkdir(parents=True)
    path.write_text("header only", encoding="utf-8")
    markers = tuple(
        SimpleNamespace(label=str(index), unique_id=f"id-{index}")
        for index in range(1, 23)
    )
    monkeypatch.setattr(
        audit,
        "read_motive_layout",
        lambda _: SimpleNamespace(
            marker_labels=tuple(marker.label for marker in markers),
            markers=markers,
            header_row_count=7,
            data_start_row=7,
            length_units="Meters",
        ),
    )
    protocol = {
        "dataset": {
            "cloth_marker_labels": [str(index) for index in range(1, 21)],
            "cloth_only_marker_count": 20,
            "expected_size": "A2",
        }
    }
    row = audit._audit_file(path, root, protocol)
    assert row["augmented_layout_candidate"]
    assert row["cloth_labels_complete"]
    assert row["extra_marker_labels"] == ["21", "22"]
    assert row["marker_trajectory_values_parsed"] is False


def test_non_augmented_layout_is_not_a_candidate(monkeypatch, tmp_path: Path) -> None:
    audit = module()
    root = tmp_path / "dataset"
    path = root / "Shake" / "cotton_A2_shake.csv"
    path.parent.mkdir(parents=True)
    path.write_text("header only", encoding="utf-8")
    markers = tuple(
        SimpleNamespace(label=str(index), unique_id=f"id-{index}")
        for index in range(1, 21)
    )
    monkeypatch.setattr(
        audit,
        "read_motive_layout",
        lambda _: SimpleNamespace(
            marker_labels=tuple(marker.label for marker in markers),
            markers=markers,
            header_row_count=7,
            data_start_row=7,
            length_units="Meters",
        ),
    )
    protocol = {
        "dataset": {
            "cloth_marker_labels": [str(index) for index in range(1, 21)],
            "cloth_only_marker_count": 20,
            "expected_size": "A2",
        }
    }
    row = audit._audit_file(path, root, protocol)
    assert not row["augmented_layout_candidate"]
