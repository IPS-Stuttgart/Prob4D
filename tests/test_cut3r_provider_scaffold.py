from __future__ import annotations

import json

import pytest

from prob4d.prediction_provider_scaffold import (
    CUT3R_ONLINE_PROVIDER_PROFILE,
    main,
    scaffold_cut3r_online_provider_import,
)


def test_cut3r_online_scaffold_freezes_single_pass_causal_profile(tmp_path) -> None:
    specification, readme = scaffold_cut3r_online_provider_import(
        tmp_path / "cut3r-online"
    )
    payload = json.loads(specification.read_text(encoding="utf-8"))

    assert payload["provider_family"] == "CUT3R-online"
    assert payload["provider_repository"] == "CUT3R/CUT3R"
    assert payload["coordinate_semantics"] == "sequence-local-sim3"
    assert payload["flow_semantics"] == "absent"
    assert payload["ray_semantics"] == "absent"
    assert payload["metadata"]["profile"] == CUT3R_ONLINE_PROVIDER_PROFILE
    assert payload["metadata"]["inference_mode"] == "online-recurrent-single-pass"
    assert payload["metadata"]["revisit_count"] == 1
    assert payload["metadata"]["state_update"] is True
    assert payload["metadata"]["future_frames_used"] is False
    assert payload["metadata"]["uses_truth"] is False
    assert payload["metadata"]["uses_downstream_physical_innovation"] is False

    descriptor = payload["payloads"][0]
    assert descriptor["product_role"] == "external-sequence"
    assert descriptor["window_id"] == "cut3r-online-sequence"
    assert descriptor["frame_lineage"] == [
        {
            "output_frame_id": 0,
            "source_frame_start": 0,
            "source_frame_stop_exclusive": 1,
            "contributor_ids": ["cut3r-online-recurrent-state"],
        }
    ]
    readme_text = readme.read_text(encoding="utf-8")
    assert "revisit_count=1" in readme_text
    assert "[0, t + 1)" in readme_text


def test_cut3r_online_scaffold_refuses_replacement(tmp_path) -> None:
    destination = tmp_path / "cut3r-online"
    scaffold_cut3r_online_provider_import(destination)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        scaffold_cut3r_online_provider_import(destination)


def test_grouped_scaffold_command_selects_cut3r_profile(tmp_path, capsys) -> None:
    destination = tmp_path / "cut3r-online"
    assert main([str(destination), "--profile", "cut3r-online"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ready_for_import"] is False
    payload = json.loads((destination / "provider-import.json").read_text())
    assert payload["metadata"]["profile"] == CUT3R_ONLINE_PROVIDER_PROFILE
