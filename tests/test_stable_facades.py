from __future__ import annotations

import sys


def test_contract_and_source_facades_do_not_load_gpu_stacks() -> None:
    import prob4d.contracts as contracts
    import prob4d.source as source

    assert contracts.OBSERVATION_BELIEF_VERSION == 1
    assert contracts.OBSERVATION_FACTOR_SCHEMA_VERSION == 4
    assert source.WINDOWED_4D_SOURCE_MANIFEST_VERSION == 1
    assert {"torch", "diffusers", "decord"}.isdisjoint(sys.modules)


def test_focused_facades_exclude_experiment_helpers() -> None:
    import prob4d.contracts as contracts
    import prob4d.source as source

    assert not hasattr(contracts, "run_experiment")
    assert not hasattr(contracts, "estimate_causal_multi_edge_gauge_graph")
    assert not hasattr(source, "MotionCrafterAdapter")
