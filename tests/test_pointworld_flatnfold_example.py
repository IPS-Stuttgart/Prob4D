from __future__ import annotations

from pathlib import Path

from prob4d.pointworld_flatnfold_support import (
    build_pointworld_flatnfold_support_request,
)
from prob4d.provider_support_feasibility import (
    evaluate_provider_support_feasibility,
)


def test_committed_pointworld_flatnfold_support_example_is_replayable() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = root / "examples/pointworld-flatnfold-support-inventory.example.json"
    request = build_pointworld_flatnfold_support_request(inventory)
    result = evaluate_provider_support_feasibility(request)

    assert result.support_feasible is True
    assert result.stream_count == 3
    assert request.metadata["statistical_unit"] == "complete-physical-garment"
