from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.dot_rope_cut3r_study import (
    Sim3,
    bilinear_sample,
    content_id,
    covariance_closures,
    fit_sim3,
    parse_coordinate_text,
    rotvec_to_matrix,
    sim3_to_vector,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "dot-rope-cut3r-native-provider-v1.json"
SCRIPT = ROOT / "scripts" / "science" / "run_dot_rope_cut3r_native_provider.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_rope_cut3r_provider_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_has_canonical_identity_and_frozen_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_id = protocol.pop("protocol_id")
    assert content_id(protocol) == protocol_id
    assert protocol["source_sequences"] == ["R01", "R02", "R03"]
    assert protocol["reserved_sequences"] == "R04-R70"
    assert protocol["frames"] == list(range(1, 8))
    assert protocol["uncertainty"]["means_held_fixed"] is True
    assert protocol["evaluation"]["marker_support_policy"] == {
        "policy_version": 2,
        "minimum_valid_provider_truth_markers_per_frame": 3,
        "minimum_common_provider_markers_per_frame": 3,
        "rationale": (
            "Three finite correspondences are the proper-Sim3 minimum; registered "
            "overlap and metric-fit stages aggregate multiple fixed frame groups."
        ),
    }


def test_request_validator_recomputes_identity(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.chdir(ROOT)
    protocol_blob = "1" * 40
    request = {
        "schema": "prob4d.dot-rope-cut3r-native-provider-request",
        "schema_version": 1,
        "protocol_path": "protocols/dot-rope-cut3r-native-provider-v1.json",
        "protocol_git_blob_sha": protocol_blob,
        "runtime_smoke_authorized": True,
        "normal_view_prediction_authorized": True,
        "marker_2d_evaluation_authorized": True,
        "marker_3d_evaluation_authorized": True,
        "source_sequences": ["R01", "R02", "R03"],
        "reserved_sequences": "R04-R70",
        "target_payloads_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": "source only",
    }
    request["request_id"] = content_id(request)
    request_path = ROOT / "build-test-dot-cut3r-request.json"
    try:
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = module.validate_request(
            request_path,
            Path("protocols/dot-rope-cut3r-native-provider-v1.json"),
            protocol_blob,
        )
    finally:
        request_path.unlink(missing_ok=True)
    assert result["request_id"] == request["request_id"]


def test_fit_sim3_recovers_proper_transform() -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(size=(64, 3))
    expected = Sim3(
        1.7,
        rotvec_to_matrix(np.asarray([0.2, -0.1, 0.05])),
        np.asarray([0.3, -0.7, 1.1]),
    )
    target = expected.apply(source)
    measured = fit_sim3(source, target)
    np.testing.assert_allclose(measured.scale, expected.scale, atol=1.0e-10)
    np.testing.assert_allclose(measured.rotation, expected.rotation, atol=1.0e-10)
    np.testing.assert_allclose(measured.translation, expected.translation, atol=1.0e-10)


def test_coordinate_parser_and_bilinear_sampling() -> None:
    points = parse_coordinate_text("1,2\n3,4\n5,6\n", 2)
    np.testing.assert_array_equal(points, np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    field = np.zeros((2, 2, 1), dtype=np.float64)
    field[0, 1, 0] = 2.0
    field[1, 0, 0] = 4.0
    field[1, 1, 0] = 6.0
    sampled, valid = bilinear_sample(field, np.asarray([[0.5, 0.5], [5.0, 5.0]]))
    np.testing.assert_allclose(sampled[0, 0], 3.0)
    assert valid.tolist() == [True, False]


def _marker_test_run(frames: list[int]) -> dict[str, np.ndarray]:
    height = 4
    width = 4
    rows, columns = np.mgrid[:height, :width]
    point_map = np.stack(
        (
            columns.astype(np.float64),
            rows.astype(np.float64),
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "points": np.repeat(point_map[None, ...], len(frames), axis=0),
        "confidence": np.ones((len(frames), height, width), dtype=np.float64),
        "original_sizes": np.repeat(
            np.asarray([[width, height]], dtype=np.int64),
            len(frames),
            axis=0,
        ),
    }


def _three_marker_payload() -> tuple[np.ndarray, np.ndarray]:
    coordinates_2d = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=np.float64,
    )
    coordinates_3d = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    return coordinates_2d, coordinates_3d


def test_three_valid_markers_are_sufficient_for_one_registered_frame() -> None:
    module = _load_script()
    run = _marker_test_run([1])
    coordinates_2d, coordinates_3d = _three_marker_payload()

    provider, truth, indices = module._sample_markers(
        run,
        1,
        coordinates_2d,
        coordinates_3d,
    )

    assert provider.shape == (3, 3)
    np.testing.assert_array_equal(truth, coordinates_3d)
    np.testing.assert_array_equal(indices, np.arange(3))


def test_three_common_markers_per_frame_are_sufficient_for_overlap() -> None:
    module = _load_script()
    run = _marker_test_run([1, 2])
    coordinates = _three_marker_payload()
    frame_payloads = {1: coordinates, 2: coordinates}

    source, target, groups = module._collect_pair(
        run,
        run,
        frame_payloads,
        [1, 2],
    )

    assert source.shape == (6, 3)
    np.testing.assert_array_equal(source, target)
    np.testing.assert_array_equal(groups, [1, 1, 1, 2, 2, 2])


def test_fewer_than_three_valid_markers_fail_with_a_counted_reason() -> None:
    module = _load_script()
    run = _marker_test_run([1])
    coordinates_2d = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]],
        dtype=np.float64,
    )
    coordinates_3d = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match=r"frame 1 has 2 valid marker samples; need at least 3",
    ):
        module._sample_markers(run, 1, coordinates_2d, coordinates_3d)


def test_registered_covariance_closures_are_psd_and_fixed_dimension() -> None:
    center_transform = Sim3(
        1.0,
        rotvec_to_matrix(np.asarray([0.05, -0.03, 0.02])),
        np.asarray([0.1, 0.2, -0.1]),
    )
    center = sim3_to_vector(center_transform)
    covariance = np.diag([0.01, 0.01, 0.01, 0.04, 0.03, 0.02, 0.005])
    probes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bootstrap = [
        Sim3(
            1.0 + 0.002 * index,
            rotvec_to_matrix(np.asarray([0.01 * index, -0.003 * index, 0.002 * index])),
            np.asarray([0.001 * index, 0.0, -0.001 * index]),
        )
        for index in range(-12, 13)
    ]
    closures = covariance_closures(
        center,
        covariance,
        probes,
        bootstrap,
        scalar_inflation=4.0,
        finite_difference_steps=np.full(7, 1.0e-3),
        orbit_nodes=9,
        tensor_gh_order=3,
    )
    assert set(closures) == {
        "local_first_order",
        "axis_spherical_radial",
        "scalar_inflation",
        "pointwise_quadratic",
        "shared_quadratic_curvature",
        "dominant_rotation_orbit",
        "tensor_gauss_hermite",
        "cluster_bootstrap_fallback",
    }
    for value in closures.values():
        assert value.shape == (9, 9)
        assert np.min(np.linalg.eigvalsh(value)) >= -1.0e-9
    off_diagonal = closures["shared_quadratic_curvature"] - np.diag(
        np.diag(closures["shared_quadratic_curvature"])
    )
    assert np.max(np.abs(off_diagonal)) > 0.0
    assert (
        np.count_nonzero(
            closures["pointwise_quadratic"] - np.diag(np.diag(closures["pointwise_quadratic"]))
        )
        == 0
    )
