import pytest

from prob4d.camera_panel_support import (
    CameraPanelFrameSupportV1,
    CameraPanelSupportPolicyV1,
    CameraPanelSupportReportV1,
)


def test_frame_rejects_zero_cell_contributors() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CameraPanelFrameSupportV1(
            frame_index=0,
            contributing_view_ids=("camera-a",),
            spatially_supported_view_ids=(),
            seed_cell_counts_by_view={"camera-a": 0},
            supported=False,
            reason_codes=("insufficient-spatially-supported-views",),
        )


def test_report_derives_supported_views_from_stored_cell_counts() -> None:
    policy = CameraPanelSupportPolicyV1(
        minimum_view_count=1,
        minimum_seed_cell_count_per_view=2,
        minimum_supported_frame_fraction=1.0,
    )
    forged = CameraPanelFrameSupportV1(
        frame_index=0,
        contributing_view_ids=("camera-a",),
        spatially_supported_view_ids=("camera-a",),
        seed_cell_counts_by_view={"camera-a": 1},
        supported=True,
        reason_codes=(),
    )

    with pytest.raises(
        ValueError,
        match="supported-view IDs contradict stored seed-cell counts",
    ):
        CameraPanelSupportReportV1(
            panel_id="forged-count-threshold",
            causal_frame_stop=1,
            required_frame_indices=(0,),
            declared_view_ids=("camera-a",),
            seed_cell_grid_shape=(1, 2),
            policy=policy,
            frame_results=(forged,),
            support_feasible=True,
            decision_reason="camera-panel-spatial-support-feasible",
        )


def test_report_recomputes_frame_reasons_from_the_frozen_policy() -> None:
    policy = CameraPanelSupportPolicyV1(
        minimum_view_count=2,
        minimum_seed_cell_count_per_view=1,
        minimum_supported_frame_fraction=1.0,
        require_all_declared_views=True,
    )
    forged = CameraPanelFrameSupportV1(
        frame_index=0,
        contributing_view_ids=("camera-a",),
        spatially_supported_view_ids=("camera-a",),
        seed_cell_counts_by_view={"camera-a": 1},
        supported=True,
        reason_codes=(),
    )

    with pytest.raises(ValueError, match="contradicts the frozen panel policy"):
        CameraPanelSupportReportV1(
            panel_id="forged-policy-result",
            causal_frame_stop=1,
            required_frame_indices=(0,),
            declared_view_ids=("camera-a", "camera-b"),
            seed_cell_grid_shape=(1, 1),
            policy=policy,
            frame_results=(forged,),
            support_feasible=True,
            decision_reason="camera-panel-spatial-support-feasible",
        )
