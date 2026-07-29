import itertools

import numpy as np
import pytest

from prob4d.alignment import AlignmentResult, WindowAlignment
from prob4d.alignment_cycles import audit_alignment_cycles
from prob4d.sim3 import Sim3


def edge(
    reference_id: str,
    moving_id: str,
    transform: Sim3,
    *,
    residual_rms: float = 0.01,
    correspondences: int = 100,
) -> WindowAlignment:
    return WindowAlignment(
        reference_id=reference_id,
        moving_id=moving_id,
        common_frames=np.array([3, 4], dtype=np.int64),
        result=AlignmentResult(
            transform=transform,
            covariance=np.eye(7) * 1e-4,
            residual_rms=residual_rms,
            inlier_fraction=0.95,
            num_correspondences=correspondences,
        ),
    )


def exact_triangle() -> list[WindowAlignment]:
    a_from_b = Sim3.from_vector(
        np.array([0.03, 0.02, -0.01, 0.04, 0.2, -0.1, 0.05])
    )
    b_from_c = Sim3.from_vector(
        np.array([-0.02, -0.03, 0.01, 0.02, -0.1, 0.08, 0.03])
    )
    a_from_c = a_from_b.compose(b_from_c)
    return [
        edge("a", "b", a_from_b),
        edge("b", "c", b_from_c),
        edge("a", "c", a_from_c),
    ]


def test_exact_triangle_has_zero_cycle_residual() -> None:
    audit = audit_alignment_cycles(exact_triangle(), representative_radius=0.5)

    assert audit.cycle_count == 1
    cycle = audit.cycles[0]
    assert cycle.cycle_id == "a<-b<-c"
    assert cycle.representative_displacement < 1e-12
    assert cycle.rotation_error_rad < 1e-12
    assert cycle.translation_error < 1e-12
    assert audit.passed is None


def test_translation_inconsistency_fails_declared_displacement_gate() -> None:
    alignments = exact_triangle()
    direct = alignments[-1]
    transform = direct.result.transform
    perturbed = Sim3(
        scale=transform.scale,
        rotation=transform.rotation,
        translation=transform.translation + np.array([0.1, 0.0, 0.0]),
    )
    alignments[-1] = edge("a", "c", perturbed)

    audit = audit_alignment_cycles(
        alignments,
        representative_radius=0.5,
        maximum_representative_displacement=0.05,
    )

    assert audit.failed_cycle_count == 1
    assert audit.passed is False
    assert audit.cycles[0].passed is False
    np.testing.assert_allclose(
        audit.cycles[0].representative_displacement,
        0.1,
        atol=1e-12,
    )


def test_cycle_audit_is_invariant_to_alignment_order() -> None:
    alignments = exact_triangle()
    expected = audit_alignment_cycles(alignments).to_dict()

    for permutation in itertools.permutations(alignments):
        assert audit_alignment_cycles(permutation).to_dict() == expected


def test_duplicate_directed_edges_are_rejected() -> None:
    alignments = exact_triangle()
    alignments.append(alignments[0])

    with pytest.raises(ValueError, match="duplicate directed alignment edge"):
        audit_alignment_cycles(alignments)


def test_graph_without_triangle_returns_empty_audit() -> None:
    alignments = exact_triangle()[:2]

    audit = audit_alignment_cycles(
        alignments,
        maximum_representative_displacement=0.01,
    )

    assert audit.cycle_count == 0
    assert audit.failed_cycle_count == 0
    assert audit.passed is True
    assert audit.maximum_observed_representative_displacement == 0.0
    assert audit.to_dict()["statistical_interpretation"].startswith(
        "unnormalized diagnostic"
    )
