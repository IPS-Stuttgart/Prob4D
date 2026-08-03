from argparse import Namespace

import pytest

from prob4d.evaluation_memory_benchmark import run_benchmark


def _arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "frames": 3,
        "height": 6,
        "width": 8,
        "seed": 23,
        "include_flow": True,
        "evaluation_chunk_size": 7,
    }
    values.update(overrides)
    return Namespace(**values)


def test_evaluation_memory_benchmark_emits_reproducible_contract() -> None:
    first = run_benchmark(_arguments())
    second = run_benchmark(_arguments())

    assert first["schema"] == "prob4d.evaluation-memory-benchmark"
    assert first["version"] == 1
    assert first["configuration"] == second["configuration"]
    assert first["output"] == second["output"]
    assert len(first["output"]["metrics_digest"]) == 64
    assert first["memory_bytes"]["retained_prediction"] > 0
    assert first["memory_bytes"]["retained_truth"] > 0
    assert (
        first["memory_bytes"]["legacy_evaluate_sequence_point_covariance_copies"]
        > first["memory_bytes"]["retained_scalar_diagnostics_upper_bound"]
    )
    peak_rss = first["memory_bytes"]["peak_process_rss"]
    assert peak_rss is None or peak_rss > 0
    assert first["timing_seconds"]["evaluation"] >= 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frames", 1, "frames must be at least two"),
        ("height", 0, "height and width must be positive"),
        ("evaluation_chunk_size", 0, "evaluation chunk size must be positive"),
    ],
)
def test_evaluation_memory_benchmark_rejects_invalid_sizes(
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(_arguments(**{field: value}))
