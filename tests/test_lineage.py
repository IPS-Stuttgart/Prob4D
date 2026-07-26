import numpy as np
import pytest

from prob4d.lineage import (
    audit_motioncrafter_product_frame,
    motioncrafter_temporal_lineage_manifest,
)


def manifest(window_size: int = 25, overlap: int = 8) -> dict[str, object]:
    return {
        "format_version": 1,
        "config": {"window_size": window_size, "overlap": overlap},
        "temporal_lineage": motioncrafter_temporal_lineage_manifest(
            window_size=window_size,
            overlap=overlap,
        ),
    }


def test_disjoint_endpoint_fails_when_first_window_reads_cutoff_frame() -> None:
    frames = np.arange(110, 160)

    audit = audit_motioncrafter_product_frame(
        manifest(),
        frames,
        product="disjoint",
        output_frame=133,
        cutoff_frame=134,
    )

    assert audit.source_frame_min == 110
    assert audit.source_frame_max == 134
    assert audit.source_window_indices == (0,)
    assert audit.admissible is False


def test_prefix_aligned_disjoint_endpoint_is_admissible() -> None:
    frames = np.arange(109, 159)

    audit = audit_motioncrafter_product_frame(
        manifest(),
        frames,
        product="disjoint",
        output_frame=133,
        cutoff_frame=134,
    )

    assert audit.source_frame_min == 109
    assert audit.source_frame_max == 133
    assert audit.admissible is True


def test_latent_overlap_reports_both_contributing_windows() -> None:
    frames = np.arange(109, 159)

    audit = audit_motioncrafter_product_frame(
        manifest(),
        frames,
        product="latent_linear",
        output_frame=133,
        cutoff_frame=134,
    )

    assert audit.source_frame_min == 109
    assert audit.source_frame_max == 150
    assert audit.source_window_indices == (0, 1)
    assert audit.admissible is False


def test_legacy_manifest_lineage_is_reconstructed_without_gpu_rerun() -> None:
    frames = np.arange(109, 159)
    legacy = {"format_version": 1, "config": {"window_size": 25, "overlap": 8}}

    audit = audit_motioncrafter_product_frame(
        legacy,
        frames,
        product="disjoint",
        output_frame=133,
        cutoff_frame=134,
    )

    assert audit.lineage_source == "reconstructed_from_legacy_manifest_config"
    assert audit.admissible is True


def test_unknown_lineage_schema_fails_closed() -> None:
    broken = manifest()
    broken["temporal_lineage"]["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported temporal-lineage"):
        audit_motioncrafter_product_frame(
            broken,
            np.arange(109, 159),
            product="disjoint",
            output_frame=133,
            cutoff_frame=134,
        )
