from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_dot_rope_reserve_support.py"
PROTOCOL = ROOT / "protocols" / "dot-rope-reserve-support-audit-v1.json"


def _load_module():
    name = "dot_rope_reserve_support_audit_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _archive_name(sequence_number: int) -> str:
    start = ((sequence_number - 1) // 10) * 10 + 1
    return f"R{start:02d}-{start + 9:02d}.zip"


def _coordinate_rows(visible: int, total: int = 12) -> str:
    rows = []
    for index in range(total):
        if index < visible:
            rows.append(f"{10.0 + index:.3f} {20.0 + index:.3f}\n")
        else:
            rows.append("-1 -1\n")
    return "".join(rows)


def _make_dataset(root: Path) -> Path:
    archive_paths: dict[str, Path] = {}
    for start in range(11, 62, 10):
        name = f"R{start:02d}-{start + 9:02d}.zip"
        path = root / name
        archive_paths[name] = path
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for sequence_number in range(start, start + 10):
                sequence = f"R{sequence_number:02d}"
                for camera_index in range(1, 11):
                    camera = f"cam{camera_index:03d}"
                    visible = 8
                    if sequence == "R11" and camera == "cam003":
                        visible = 10
                    if sequence == "R12":
                        visible = 7
                    for frame in range(1, 8):
                        member = (
                            f"{sequence}/coordinates/2d/"
                            f"frame{frame:06d}_{camera}.txt"
                        )
                        archive.writestr(member, _coordinate_rows(visible))
                archive.writestr(
                    f"{sequence}/coordinates/3d/frame000001_cam001.txt",
                    "this must never be parsed\n",
                )
    metadata = {
        "dataset_persistent_id": "doi:10.13021/ORC2020/XXLVXM",
        "files": [
            {
                "filename": name,
                "datafile_id": 1000 + index,
                "byte_count": path.stat().st_size,
                "md5": _md5(path),
            }
            for index, (name, path) in enumerate(sorted(archive_paths.items()))
        ],
    }
    metadata_path = root / "official-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def test_protocol_and_archive_mapping_are_frozen() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["dataset"]["sequence_start"] == 11
    assert protocol["dataset"]["sequence_stop"] == 70
    assert protocol["camera_selection_rule"][
        "minimum_common_visible_markers"
    ] == 8
    assert module._archive_for_sequence(11) == "R11-20.zip"
    assert module._archive_for_sequence(20) == "R11-20.zip"
    assert module._archive_for_sequence(21) == "R21-30.zip"
    assert module._archive_for_sequence(70) == "R61-70.zip"


def test_coordinate_parser_preserves_row_identity_and_missing_markers() -> None:
    module = _load_module()
    rows = module._parse_coordinates(
        b"1,2\n-1 -1\n3 4 99\n",
        member="fixture.txt",
    )
    assert rows == [(1.0, 2.0), (-1.0, -1.0), (3.0, 4.0)]
    assert module._visible_indices(rows) == {0, 2}


def test_complete_synthetic_audit_is_deterministic_and_reads_only_2d(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    metadata = _make_dataset(dataset)
    output = tmp_path / "output"
    args = argparse.Namespace(
        protocol=PROTOCOL,
        dataset_root=dataset,
        official_metadata=metadata,
        output_dir=output,
        repository_revision="a" * 40,
    )
    assert module.evaluate(args) == 0
    result = json.loads((output / "result.json").read_text())
    custody = json.loads((output / "accessed-members.json").read_text())
    assert result["qualified_sequences"] == [
        f"R{index:02d}" for index in range(11, 71) if index != 12
    ]
    assert result["unsupported_sequences"] == ["R12"]
    selected = {
        row["sequence"]: row for row in result["selected_cameras"]
    }
    assert selected["R11"]["selected_camera"] == "cam003"
    assert selected["R11"]["common_visible_marker_count"] == 10
    assert selected["R12"]["selected_camera"] == "cam001"
    assert selected["R12"]["qualified"] is False
    assert selected["R13"]["selected_camera"] == "cam001"
    assert result["camera_row_count"] == 600
    assert result["accessed_member_count"] == 4200
    assert len(custody["members"]) == 4200
    assert custody["three_dimensional_coordinate_values_opened"] is False
    assert custody["normal_or_uv_images_opened"] is False
    assert all("/coordinates/2d/" in row["member"] for row in custody["members"])
    assert not any("/coordinates/3d/" in row["member"] for row in custody["members"])
    assert set(path.name for path in output.iterdir()) == {
        "accessed-members.json",
        "camera-support.csv",
        "result.json",
        "selected-cameras.csv",
        "summary.md",
    }


def test_protocol_rejects_an_expanded_information_boundary(tmp_path: Path) -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL.read_text())
    protocol["information_boundary"][
        "three_dimensional_coordinate_values_opened"
    ] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    try:
        module._load_protocol(path)
    except ValueError as error:
        assert "information boundary" in str(error)
    else:
        raise AssertionError("expanded information boundary was accepted")
