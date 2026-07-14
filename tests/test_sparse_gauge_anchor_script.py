from pathlib import Path

import numpy as np

from scripts.export_sparse_gauge_anchors import load_external_reference


def test_load_external_reference_resizes_points_and_mask(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    points = np.zeros((2, 2, 3, 3), dtype=np.float32)
    points[..., 2] = 1.0
    points[0, 0, 0] = np.nan
    np.savez(path, point_map=points)

    reference = load_external_reference(path, (4, 6))

    assert reference.point_map.shape == (2, 4, 6, 3)
    assert reference.valid_mask.shape == (2, 4, 6)
    assert not reference.valid_mask[0, 0, 0]
    assert np.isfinite(reference.point_map).all()
