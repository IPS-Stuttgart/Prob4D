from argparse import Namespace

import pytest

from prob4d.fusion_memory_benchmark import run_benchmark


def _arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "frames": 2,
        "height": 8,
        "width": 12,
        "contributors": 3,
        "seed": 17,
        "method": "covariance_intersection",
        "fusion_tile_size": 16,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dense_fusion_memory_benchmark_emits_reproducible_contract() -> None:
    first = run_benchmark(_arguments())
    second = run_benchmark(_arguments())

    assert first["schema"] == "prob4d.dense-fusion-memory-benchmark"
    assert first["version"] == 1
    assert first["configuration"] == second["configuration"]
    assert first["configuration"]["frames"] == 2
    assert first["output"] == second["output"]
    assert len(first["output"]["artifact_digest"]) == 64
    assert first["memory_bytes"]["retained_prediction_vectors"] > 0
    assert first["memory_bytes"]["retained_structured_covariance"] > 0
    assert first["memory_bytes"]["fused_output"] > 0
    peak_rss = first["memory_bytes"]["peak_process_rss"]
    assert peak_rss is None or peak_rss > 0
    assert first["timing_seconds"]["fusion"] >= 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frames", 0, "frames must be positive"),
        ("height", 0, "height and width must be positive"),
        ("contributors", 0, "contributors must be positive"),
        ("fusion_tile_size", 0, "fusion tile size must be positive"),
    ],
)
def test_dense_fusion_memory_benchmark_rejects_invalid_sizes(
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(_arguments(**{field: value}))
