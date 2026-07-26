from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.observation_export import (
    MetricGaugeAnchor,
    build_prob4d_observation_belief,
    gauge_covariance_factor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
    select_causal_overlap_windows,
)
from prob4d.sim3 import Sim3


def _anchor() -> MetricGaugeAnchor:
    return MetricGaugeAnchor(
        window_id="window_0000",
        global_from_local=Sim3.identity(),
        covariance=np.eye(7) * 1e-6,
        coordinate_frame="phystwin-world",
        source_kind="prefix_registration",
        source_artifact_sha256="a" * 64,
    )


def _window(path: Path, frames: list[int], offset: float = 0.0) -> None:
    grid = np.asarray(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    points = np.stack([grid + np.asarray([offset, 0.0, 0.0]) for _ in frames])
    np.savez_compressed(
        path,
        frame_indices=np.asarray(frames),
        point_map=points,
        valid_mask=np.ones(points.shape[:-1], dtype=bool),
    )


def _manifest(
    directory: Path,
    *,
    name: str,
    include_future: bool,
) -> Path:
    entries = [
        {
            "window_id": "window_0000",
            "path": "window_0000.npz",
            "start_frame": 0,
            "stop_frame": 4,
        },
        {
            "window_id": "window_0001",
            "path": "window_0001.npz",
            "start_frame": 2,
            "stop_frame": 6,
        },
    ]
    if include_future:
        entries.append(
            {
                "window_id": "window_0002",
                "path": "future-does-not-exist.npz",
                "start_frame": 4,
                "stop_frame": 8,
            }
        )
    payload = {
        "format_version": 1,
        "motioncrafter_commit": "b" * 40,
        "config": {
            "model_type": "determ",
            "window_size": 4,
            "overlap": 2,
            "height": 2,
            "width": 2,
            "seed": 42,
            "frame_start": 0,
            "frame_stop": 8 if include_future else 6,
            "frame_stride": 1,
        },
        "temporal_lineage": {
            "schema_version": 1,
            "model": "motioncrafter_sliding_window_v1",
            "products": {
                "overlap_windows": {
                    "window_size_source": "prediction archive frame count",
                    "overlap": 0,
                }
            },
        },
        "overlap_windows": entries,
        "disjoint_baseline": "unused.npz",
        "latent_linear_baseline": "unused2.npz",
    }
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_metric_anchor_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    save_metric_gauge_anchor(path, _anchor())
    restored = load_metric_gauge_anchor(path)
    assert restored.artifact_id == _anchor().artifact_id
    assert restored.coordinate_frame == "phystwin-world"


def test_causal_selection_does_not_open_future_payload(tmp_path: Path) -> None:
    _window(tmp_path / "window_0000.npz", [0, 1, 2, 3])
    _window(tmp_path / "window_0001.npz", [2, 3, 4, 5], offset=0.1)
    manifest = _manifest(tmp_path, name="predictions.json", include_future=True)
    selection = select_causal_overlap_windows(
        manifest,
        causal_frame_stop=6,
        metric_anchor=_anchor(),
    )
    assert [item.window_id for item in selection.windows] == [
        "window_0000",
        "window_0001",
    ]
    assert selection.skipped_window_count == 1
    assert selection.artifact_lineage_metadata(causal_frame_stop=6)[
        "future_prediction_payloads_opened"
    ] == 0


def test_future_append_does_not_change_exported_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _window(tmp_path / "window_0000.npz", [0, 1, 2, 3])
    _window(tmp_path / "window_0001.npz", [2, 3, 4, 5], offset=0.1)
    prefix = _manifest(tmp_path, name="prefix.json", include_future=False)
    appended = _manifest(tmp_path, name="appended.json", include_future=True)

    def fake_gauge_estimates(
        windows, *, gauge_mode, fixed_lag, metric_anchor
    ):
        del gauge_mode, fixed_lag
        return [], {
            window.window_id: GaugeEstimate(
                window.window_id,
                metric_anchor.global_from_local,
                metric_anchor.covariance,
            )
            for window in windows
        }

    monkeypatch.setattr(
        "prob4d.observation_export._gauge_estimates",
        fake_gauge_estimates,
    )

    common = dict(
        case_id="case-a",
        causal_frame_stop=6,
        metric_anchor=_anchor(),
        pixel_stride=1,
        gauge_mode="sequential",
        source_revision="c" * 40,
    )
    first = build_prob4d_observation_belief(prefix, **common)
    second = build_prob4d_observation_belief(appended, **common)

    assert first.source_artifact_sha256 == second.source_artifact_sha256
    assert first.artifact_id == second.artifact_id
    np.testing.assert_array_equal(first.mean_xyz_m, second.mean_xyz_m)


def test_selection_rejects_payload_frame_ids_that_cross_cutoff(tmp_path: Path) -> None:
    _window(tmp_path / "window_0000.npz", [0, 1, 2, 5])
    manifest = _manifest(tmp_path, name="bad.json", include_future=False)
    payload = json.loads(manifest.read_text())
    payload["overlap_windows"] = payload["overlap_windows"][:1]
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="frame IDs disagree"):
        select_causal_overlap_windows(
            manifest,
            causal_frame_stop=6,
            metric_anchor=_anchor(),
        )


def test_gauge_factor_recovers_linearized_marginal() -> None:
    covariance = np.diag([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    points = np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    factor = gauge_covariance_factor(
        points,
        Sim3.identity(),
        covariance,
        include_translation=True,
    )
    marginal = np.einsum("nir,njr->nij", factor, factor)

    expected = []
    for point in points:
        skew = np.asarray(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ]
        )
        jacobian = np.zeros((3, 7))
        jacobian[:, 0] = point
        jacobian[:, 1:4] = -skew
        jacobian[:, 4:7] = np.eye(3)
        expected.append(jacobian @ covariance @ jacobian.T)
    np.testing.assert_allclose(marginal, np.asarray(expected), atol=1e-12)
