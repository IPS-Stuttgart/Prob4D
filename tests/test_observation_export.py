import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.observation_export import (
    _group_metadata,
    build_prob4d_observation_belief,
    deterministic_covariance_root,
    joint_gauge_factor,
    load_metric_gauge_anchor,
)
from prob4d.observation_factors import sim3_point_jacobian
from prob4d.sim3 import Sim3


def _lineage(window_size: int = 2, overlap: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": "motioncrafter_sliding_window_v1",
        "frame_index_source": "prediction archive frame_indices",
        "source_bounds": "inclusive source-video frame identifiers",
        "products": {
            "disjoint_baseline": {"window_size": window_size, "overlap": 0},
            "latent_linear_baseline": {
                "window_size": window_size,
                "overlap": overlap,
            },
            "overlap_windows": {
                "window_size_source": "prediction archive frame count",
                "overlap": 0,
            },
        },
    }


def _points(offset: float = 0.0) -> np.ndarray:
    values = np.zeros((2, 2, 2, 3), dtype=np.float64)
    values[..., 0] = np.asarray([[0.0, 1.0], [0.2, 1.2]])
    values[..., 1] = np.asarray([[0.0, 0.1], [1.0, 1.1]])
    values[..., 2] = 1.0
    values[1, ..., 0] += 0.1
    values += offset
    return values


def _write_prefix_fixture(root: Path, *, future_payload: bytes = b"not an npz") -> Path:
    (root / "windows").mkdir(parents=True)
    PredictionWindow(
        window_id="window_0000",
        frame_indices=np.asarray([0, 1]),
        point_map=_points(),
        valid_mask=np.ones((2, 2, 2), dtype=bool),
    ).to_npz(root / "windows" / "window_0000.npz")
    (root / "windows" / "window_0001.npz").write_bytes(future_payload)
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": "c" * 40,
        "config": {
            "model_type": "determ",
            "window_size": 2,
            "overlap": 1,
            "frame_start": 0,
            "frame_stop": 3,
            "frame_stride": 1,
            "seed": 7,
        },
        "temporal_lineage": _lineage(),
        "overlap_windows": [
            {
                "window_id": "window_0000",
                "path": "windows/window_0000.npz",
                "start_frame": 0,
                "stop_frame": 2,
            },
            {
                "window_id": "window_0001",
                "path": "windows/window_0001.npz",
                "start_frame": 1,
                "stop_frame": 3,
            },
        ],
        "disjoint_baseline": "unread-disjoint.npz",
        "latent_linear_baseline": "unread-latent.npz",
    }
    path = root / "predictions.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _build(manifest: Path, **kwargs):
    return build_prob4d_observation_belief(
        manifest,
        case_id="case-a",
        causal_frame_stop=2,
        pixel_stride=1,
        source_revision="a" * 40,
        **kwargs,
    )


def test_export_never_opens_excluded_future_payload(tmp_path: Path) -> None:
    manifest = _write_prefix_fixture(tmp_path)
    first = _build(manifest)
    assert first.metadata["causal_information_boundary"][
        "excluded_future_payloads_opened"
    ] == 0
    (tmp_path / "windows" / "window_0001.npz").write_bytes(b"changed future")
    second = _build(manifest)
    assert second.artifact_id == first.artifact_id


def test_export_digest_changes_when_selected_payload_changes(tmp_path: Path) -> None:
    manifest = _write_prefix_fixture(tmp_path)
    first = _build(manifest)
    PredictionWindow(
        window_id="window_0000",
        frame_indices=np.asarray([0, 1]),
        point_map=_points(offset=0.5),
        valid_mask=np.ones((2, 2, 2), dtype=bool),
    ).to_npz(tmp_path / "windows" / "window_0000.npz")
    second = _build(manifest)
    assert second.artifact_id != first.artifact_id
    assert second.source_artifact_sha256 != first.source_artifact_sha256


def test_manifest_bounds_must_match_selected_archive(tmp_path: Path) -> None:
    manifest = _write_prefix_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["overlap_windows"][0]["start_frame"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frame IDs disagree"):
        _build(manifest)


def test_metric_export_requires_and_records_anchor(tmp_path: Path) -> None:
    manifest = _write_prefix_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires a metric gauge anchor"):
        _build(manifest, coordinate_mode="metric_anchored")
    anchor_path = tmp_path / "anchor.npz"
    mean = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    covariance = np.eye(7) * 1e-4
    np.savez_compressed(anchor_path, mean=mean, covariance=covariance)
    anchor = load_metric_gauge_anchor(anchor_path)
    artifact = _build(
        manifest,
        coordinate_mode="metric_anchored",
        metric_anchor=anchor,
    )
    assert artifact.metadata["coordinate_mode"] == "metric_anchored"
    assert artifact.metadata["metric_claim_authorized"] is True
    assert artifact.metadata["metric_gauge_anchor"]["source_sha256"]
    assert np.min(artifact.mean_xyz_m[:, 2]) > 3.0


def test_joint_gauge_factor_preserves_cross_window_covariance() -> None:
    generator = np.random.default_rng(7)
    basis = generator.normal(size=(14, 5))
    covariance = basis @ basis.T
    root, retained = deterministic_covariance_root(covariance, max_rank=None)
    first_point = np.asarray([[1.0, 0.2, 0.3]])
    second_point = np.asarray([[0.1, 1.2, 0.4]])
    first_factor = joint_gauge_factor(
        first_point, Sim3.identity(), root[:7]
    )[0]
    second_factor = joint_gauge_factor(
        second_point, Sim3.identity(), root[7:]
    )[0]
    first_jacobian = sim3_point_jacobian(Sim3.identity(), first_point)[0]
    second_jacobian = sim3_point_jacobian(Sim3.identity(), second_point)[0]
    expected = first_jacobian @ covariance[:7, 7:] @ second_jacobian.T
    np.testing.assert_allclose(first_factor @ second_factor.T, expected, atol=1e-10)
    assert retained == pytest.approx(1.0)


def test_group_composite_weight_caps_duplicate_information() -> None:
    groups = np.zeros(8, dtype=np.int64)
    entities = np.tile(np.arange(4, dtype=np.int64), 2)
    reliability = np.ones(8)
    _, _, weight = _group_metadata(
        groups,
        entities,
        reliability,
        effective_samples_per_group=2.0,
        group_prior_quantile=0.25,
    )
    duplicated_groups = np.tile(groups, 2)
    duplicated_entities = np.tile(entities, 2)
    duplicated_reliability = np.tile(reliability, 2)
    _, _, duplicated_weight = _group_metadata(
        duplicated_groups,
        duplicated_entities,
        duplicated_reliability,
        effective_samples_per_group=2.0,
        group_prior_quantile=0.25,
    )
    assert weight[0] * len(groups) == pytest.approx(2.0)
    assert duplicated_weight[0] * len(duplicated_groups) == pytest.approx(2.0)
