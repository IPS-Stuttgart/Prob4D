from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected one match, found {count}")
    return text.replace(old, new)


script_path = Path("scripts/science/build_cut3r_deform360_source_freeze.py")
script = script_path.read_text(encoding="utf-8")
script = replace_once(
    script,
    """    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_record in enumerate(records):
""",
    """    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    seen_group_ids: set[str] = set()
    for index, raw_record in enumerate(records):
""",
    name="group-id roster state",
)
script = replace_once(
    script,
    """        if include_role:
            normalized["group_id"] = _strict_string(
                record.get("group_id"),
                name=f"protocol.{field}[{index}].group_id",
            )
            role = _strict_string(
""",
    """        if include_role:
            group_id = _strict_string(
                record.get("group_id"),
                name=f"protocol.{field}[{index}].group_id",
            )
            if group_id in seen_group_ids:
                raise ValueError(f"protocol.{field} repeats group_id {group_id!r}")
            seen_group_ids.add(group_id)
            normalized["group_id"] = group_id
            role = _strict_string(
""",
    name="duplicate group-id guard",
)
script = replace_once(
    script,
    """    episode_by_group: dict[str, Path] = {}
    supported_by_group: dict[str, set[str]] = {}
    support_rows: list[dict[str, Any]] = []
    centers_by_camera: dict[str, list[FloatArray]] = {}
    for record in source_groups:
""",
    """    episode_by_group: dict[str, Path] = {}
    supported_by_group: dict[str, set[str]] = {}
    support_rows: list[dict[str, Any]] = []
    calibration_inputs: list[dict[str, Any]] = []
    centers_by_camera: dict[str, list[FloatArray]] = {}
    for record in source_groups:
""",
    name="calibration input collection",
)
script = replace_once(
    script,
    """        group_id = cast(str, record["group_id"])
        episode_by_group[group_id] = episode
        intrinsics = _load_numpy_mapping(
            episode / "undistorted_intrinsics.npy",
            name=f"{group_id} intrinsics",
        )
        extrinsics = _load_numpy_mapping(
            episode / "extrinsics.npy",
            name=f"{group_id} extrinsics",
        )
""",
    """        group_id = cast(str, record["group_id"])
        episode_id = cast(int, record["episode_id"])
        episode_by_group[group_id] = episode
        intrinsics_path = episode / "undistorted_intrinsics.npy"
        extrinsics_path = episode / "extrinsics.npy"
        intrinsics_sha, intrinsics_size = _sha256_file(
            intrinsics_path,
            name=f"{group_id} intrinsics",
        )
        extrinsics_sha, extrinsics_size = _sha256_file(
            extrinsics_path,
            name=f"{group_id} extrinsics",
        )
        intrinsics = _load_numpy_mapping(
            intrinsics_path,
            name=f"{group_id} intrinsics",
        )
        extrinsics = _load_numpy_mapping(
            extrinsics_path,
            name=f"{group_id} extrinsics",
        )
        calibration_inputs.append(
            {
                "group_id": group_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "intrinsics": {
                    "relative_path": (
                        f"{object_id}/episode_{episode_id:04d}/undistorted_intrinsics.npy"
                    ),
                    "sha256": intrinsics_sha,
                    "byte_count": intrinsics_size,
                },
                "extrinsics": {
                    "relative_path": f"{object_id}/episode_{episode_id:04d}/extrinsics.npy",
                    "sha256": extrinsics_sha,
                    "byte_count": extrinsics_size,
                },
            }
        )
""",
    name="calibration byte binding",
)
script = replace_once(
    script,
    """    common_supported = set.intersection(*supported_by_group.values())
    support_rows.sort(key=lambda item: (item["group_id"], item["camera"]))
    decision = SUPPORT_PASS if len(common_supported) >= minimum_common else SUPPORT_NEGATIVE
""",
    """    common_supported = set.intersection(*supported_by_group.values())
    support_rows.sort(key=lambda item: (item["group_id"], item["camera"]))
    calibration_inputs.sort(key=lambda item: item["group_id"])
    decision = SUPPORT_PASS if len(common_supported) >= minimum_common else SUPPORT_NEGATIVE
""",
    name="calibration input ordering",
)
script = replace_once(
    script,
    """        "support": {
            "required_frame_interval": [frame_start, frame_stop],
            "common_supported_camera_count": len(common_supported),
            "common_supported_cameras": sorted(common_supported),
            "minimum_common_supported_cameras": minimum_common,
            "stream_rows": support_rows,
        },
        "camera_panel": None,
""",
    """        "support": {
            "required_frame_interval": [frame_start, frame_stop],
            "common_supported_camera_count": len(common_supported),
            "common_supported_cameras": sorted(common_supported),
            "minimum_common_supported_cameras": minimum_common,
            "stream_rows": support_rows,
        },
        "camera_calibration_inputs": calibration_inputs,
        "camera_panel": None,
""",
    name="calibration input publication",
)
script_path.write_text(script, encoding="utf-8")


test_path = Path("tests/test_cut3r_deform360_source_freeze.py")
test = test_path.read_text(encoding="utf-8")
test = replace_once(
    test,
    """    assert len(result["camera_panel"]["selected_cameras"]) == 4
    assert len(result["source_cases"]) == 40
    forbidden = {item["object_id"] for item in result["forbidden_target_groups"]}
""",
    """    assert len(result["camera_panel"]["selected_cameras"]) == 4
    assert len(result["source_cases"]) == 40
    assert len(result["camera_calibration_inputs"]) == 10
    for calibration in result["camera_calibration_inputs"]:
        for name in ("intrinsics", "extrinsics"):
            identity = calibration[name]
            path = fixture["processed_root"] / identity["relative_path"]
            assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert identity["byte_count"] == path.stat().st_size
    forbidden = {item["object_id"] for item in result["forbidden_target_groups"]}
""",
    name="calibration hash assertions",
)
test += """


def test_rejects_duplicate_source_group_ids(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
    protocol["source_groups"][1]["group_id"] = protocol["source_groups"][0]["group_id"]
    _write_json(fixture["protocol"], protocol)

    with pytest.raises(ValueError, match="repeats group_id"):
        module.build_source_freeze(
            repository=fixture["repository"],
            protocol_path=fixture["protocol"],
            selection_path=fixture["selection"],
            processed_root=fixture["processed_root"],
            cut3r_checkout=fixture["cut3r"],
            checkpoint_path=fixture["checkpoint"],
            prob4d_wheel=fixture["wheel"],
            output_directory=fixture["output"],
        )


def test_source_freeze_id_binds_exact_calibration_bytes(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    arguments = {
        "repository": fixture["repository"],
        "protocol_path": fixture["protocol"],
        "selection_path": fixture["selection"],
        "processed_root": fixture["processed_root"],
        "cut3r_checkout": fixture["cut3r"],
        "checkpoint_path": fixture["checkpoint"],
        "prob4d_wheel": fixture["wheel"],
        "output_directory": fixture["output"],
    }
    first = module.build_source_freeze(**arguments)

    protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
    group = protocol["source_groups"][0]
    intrinsics_path = (
        fixture["processed_root"]
        / group["object_id"]
        / f"episode_{group['episode_id']:04d}"
        / "undistorted_intrinsics.npy"
    )
    intrinsics = np.load(intrinsics_path, allow_pickle=True).item()
    intrinsics["cam-a"] = intrinsics["cam-a"].copy()
    intrinsics["cam-a"][0, 0] += 1.0
    np.save(intrinsics_path, intrinsics, allow_pickle=True)

    arguments["output_directory"] = tmp_path / "output-after-calibration-change"
    second = module.build_source_freeze(**arguments)
    assert first["source_freeze_id"] != second["source_freeze_id"]
"""
test_path.write_text(test, encoding="utf-8")

Path(__file__).unlink(missing_ok=True)
