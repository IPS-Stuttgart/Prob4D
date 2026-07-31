from __future__ import annotations

import numpy as np
import pytest

from prob4d.observation_export import JointGaugePosterior
from prob4d.sim3 import Sim3


def _posterior(
    covariance: np.ndarray,
    estimates: dict[str, Sim3],
    *,
    parents: tuple[str | None, ...] = (None, "w0"),
    alignment_indices: tuple[int | None, ...] = (None, 0),
) -> JointGaugePosterior:
    return JointGaugePosterior(
        window_ids=("w0", "w1"),
        estimates=estimates,
        joint_covariance=covariance,
        mode="sequential_joint_spanning_tree_v1",
        cross_window_covariance_preserved=True,
        parent_window_ids=parents,
        selected_alignment_indices=alignment_indices,
    )


def test_joint_gauge_posterior_defensively_freezes_inputs() -> None:
    covariance = np.eye(14)
    estimates = {"w0": Sim3.identity(), "w1": Sim3.identity()}
    posterior = _posterior(covariance, estimates)

    covariance[0, 0] = 99.0
    estimates["w0"] = Sim3(
        scale=2.0,
        rotation=np.eye(3),
        translation=np.zeros(3),
    )

    assert posterior.joint_covariance[0, 0] == pytest.approx(1.0)
    assert posterior.estimates["w0"].scale == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        posterior.joint_covariance[0, 0] = 2.0
    with pytest.raises(TypeError):
        posterior.estimates["w0"] = Sim3.identity()  # type: ignore[index]


def test_joint_gauge_posterior_requires_causal_parent_order() -> None:
    with pytest.raises(ValueError, match="parent must precede"):
        _posterior(
            np.eye(14),
            {"w0": Sim3.identity(), "w1": Sim3.identity()},
            parents=("w1", None),
        )


def test_joint_gauge_posterior_rejects_invalid_alignment_lineage() -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        _posterior(
            np.eye(14),
            {"w0": Sim3.identity(), "w1": Sim3.identity()},
            alignment_indices=(None, -1),
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        _posterior(
            np.eye(14),
            {"w0": Sim3.identity(), "w1": Sim3.identity()},
            alignment_indices=(None, True),
        )


def test_joint_gauge_posterior_rejects_non_sim3_estimates() -> None:
    with pytest.raises(TypeError, match="Sim3"):
        JointGaugePosterior(
            window_ids=("w0",),
            estimates={"w0": object()},  # type: ignore[dict-item]
            joint_covariance=np.eye(7),
            mode="test",
            cross_window_covariance_preserved=False,
        )
