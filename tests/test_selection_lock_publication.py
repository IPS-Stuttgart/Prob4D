from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from prob4d.selection_lock import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    MetricOrderV1,
    SelectionRuleV1,
    build_selection_lock,
    load_selection_lock,
    write_selection_lock,
)

SOURCE_REVISION = "a" * 40


def make_lock(marker: str):
    candidates = (
        CandidateSpecV1(
            candidate_id="fallback",
            method_id="physical-fallback",
            complexity_rank=0,
            parameters={"visual_update": False},
        ),
        CandidateSpecV1(
            candidate_id="candidate",
            method_id="persistent-explicit-joint-gauge",
            complexity_rank=1,
            parameters={"visual_update": True},
        ),
    )
    rows = (
        CalibrationMetricRowV1(
            group_id="object-a",
            candidate_id="fallback",
            metrics={"rmse_mm": 5.0},
        ),
        CalibrationMetricRowV1(
            group_id="object-a",
            candidate_id="candidate",
            metrics={"rmse_mm": 2.0},
        ),
        CalibrationMetricRowV1(
            group_id="object-b",
            candidate_id="fallback",
            metrics={"rmse_mm": 5.5},
        ),
        CalibrationMetricRowV1(
            group_id="object-b",
            candidate_id="candidate",
            metrics={"rmse_mm": 2.5},
        ),
    )
    return build_selection_lock(
        experiment_id="selection-lock-publication-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=SOURCE_REVISION,
        candidates=candidates,
        calibration_rows=rows,
        selection_rule=SelectionRuleV1(
            primary=MetricOrderV1("rmse_mm", "minimize"),
        ),
        metadata={"marker": marker},
    )


def temporary_files(path: Path) -> tuple[Path, ...]:
    return tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_selection_lock_publication_never_replaces_retained_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection-lock.json"
    first = make_lock("first")
    second = make_lock("second")

    write_selection_lock(first, path)
    retained = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_selection_lock(second, path)

    assert path.read_bytes() == retained
    assert load_selection_lock(path).selection_lock_id == first.selection_lock_id
    assert temporary_files(path) == ()


def test_concurrent_selection_lock_publication_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection-lock.json"
    locks = (make_lock("left"), make_lock("right"))
    barrier = Barrier(len(locks))

    def publish(lock) -> str | None:
        barrier.wait()
        try:
            write_selection_lock(lock, path)
        except FileExistsError:
            return None
        return lock.selection_lock_id

    with ThreadPoolExecutor(max_workers=len(locks)) as executor:
        results = tuple(executor.map(publish, locks))

    winners = tuple(result for result in results if result is not None)
    assert len(winners) == 1
    assert load_selection_lock(path).selection_lock_id == winners[0]
    assert temporary_files(path) == ()
