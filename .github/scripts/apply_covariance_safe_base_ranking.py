"""Apply the covariance-safe schema-v2 base association ranking change."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, *, name: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name} anchor changed in {path}: occurrences={count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


source = "src/prob4d/cross_window_tracklets.py"
replace_once(
    source,
    "_ASSOCIATION_SCHEMA_VERSION = 2\n",
    "_ASSOCIATION_SCHEMA_VERSION = 2\n"
    "RANKING_SEMANTICS = (\n"
    '    "isotropic-geometric-mutual-best-covariance-diagnostic-v1"\n'
    ")\n",
    name="ranking semantics constant",
)
replace_once(
    source,
    '            "schema_version": _ASSOCIATION_SCHEMA_VERSION,\n',
    '            "schema_version": _ASSOCIATION_SCHEMA_VERSION,\n'
    '            "ranking_semantics": RANKING_SEMANTICS,\n',
    name="ranking semantics descriptor",
)
replace_once(
    source,
    '''    Optional covariance arrays must already be in the global frame and align with
    the flattened observations in each tracklet set. They may contain local point
    uncertainty, gauge uncertainty, or both. Supplying one side only is rejected.
''',
    '''    Optional covariance arrays must already be in the global frame and align with
    the flattened observations in each tracklet set. They may contain local point
    uncertainty, gauge uncertainty, or both. They determine the reported
    ``normalized_rms`` diagnostic but cannot improve mutual-best rank merely by
    becoming wider. Supplying one side only is rejected.
''',
    name="covariance ranking docstring",
)
replace_once(
    source,
    '''        weighted_rms = float(np.sqrt(np.sum(weights * distances**2) / support))
        normalized_rms = float(np.sqrt(np.sum(weights * normalized_squares) / support))
        support_fraction = min(1.0, support / config.minimum_effective_support)
        score = float(support_fraction * np.exp(-0.5 * normalized_rms**2))
''',
    '''        weighted_rms = float(np.sqrt(np.sum(weights * distances**2) / support))
        normalized_rms = float(np.sqrt(np.sum(weights * normalized_squares) / support))
        support_fraction = min(1.0, support / config.minimum_effective_support)
        geometric_normalized_rms = weighted_rms / config.isotropic_distance_scale_m
        # Covariance affects the reported normalized residual, not mutual-best
        # ranking. Otherwise a candidate can improve merely by becoming wider and
        # less informative. A covariance-aware admission rule requires a separate
        # source-calibrated method version.
        score = float(
            support_fraction * np.exp(-0.5 * geometric_normalized_rms**2)
        )
''',
    name="covariance-safe ranking",
)
replace_once(
    source,
    '''    "CrossWindowAssociationResult",
    "associate_cross_window_tracklets",
''',
    '''    "CrossWindowAssociationResult",
    "RANKING_SEMANTICS",
    "associate_cross_window_tracklets",
''',
    name="ranking semantics export",
)

tests = "tests/test_cross_window_tracklets.py"
replace_once(
    tests,
    '''    assert loose_candidate.compatibility_score > tight_candidate.compatibility_score
    assert loose_candidate.normalized_rms < tight_candidate.normalized_rms


def test_covariance_score_uses_reduced_mahalanobis_rms() -> None:
''',
    '''    assert loose_candidate.compatibility_score == tight_candidate.compatibility_score
    assert loose_candidate.normalized_rms < tight_candidate.normalized_rms
    assert loose_result.to_dict()["ranking_semantics"] == (
        "isotropic-geometric-mutual-best-covariance-diagnostic-v1"
    )


def test_covariance_inflation_cannot_change_base_mutual_best_rank() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
        ],
    )
    left_covariance = np.zeros((2, 3, 3), dtype=np.float64)
    right_covariance = np.zeros((4, 3, 3), dtype=np.float64)
    right_covariance[2:] = np.eye(3)
    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(
            covariance_floor_m2=1e-12,
            maximum_weighted_rms_m=0.05,
            maximum_shared_frame_distance_m=0.05,
            minimum_compatibility_score=0.0,
            minimum_score_margin=0.0,
        ),
        left_global_covariance_m2=left_covariance,
        right_global_covariance_m2=right_covariance,
    )

    assert result.accepted_pairs == ((0, 0),)
    precise, inflated = result.candidates
    assert precise.compatibility_score == inflated.compatibility_score
    assert precise.weighted_rms_m == inflated.weighted_rms_m
    assert precise.normalized_rms > inflated.normalized_rms


def test_covariance_score_uses_reduced_mahalanobis_rms() -> None:
''',
    name="covariance-inflation regression",
)

documentation = "docs/cross-window-tracklet-association.md"
replace_once(
    documentation,
    '''The compatibility score is a source-side ranking statistic, not a calibrated
posterior probability. It combines the Gaussian-shaped normalized residual score
with an effective-support factor. Promotion would require independent calibration
and a downstream guarded physical-prediction gate.
''',
    '''The compatibility score is a source-side ranking statistic, not a calibrated
posterior probability. It combines the covariance-independent isotropic geometric
residual score with an effective-support factor. Supplied covariance affects the
reported `normalized_rms` diagnostic only. This prevents a candidate from becoming
mutual-best merely because its uncertainty volume is larger. A covariance-aware
admission rule requires a separately versioned, independently calibrated method
and a downstream guarded physical-prediction gate.
''',
    name="ranking documentation",
)
replace_once(
    documentation,
    '''`CrossWindowAssociationResult.descriptor()` emits the complete semantic result:
configuration, candidates, accepted links, unmatched tracks, and rejection
accounting. Schema version 2 includes corrected evaluated-pair semantics and the
strict construction contract. `result_id` is the SHA-256 digest of the canonical
finite-JSON encoding of that descriptor. `to_dict()` adds the ID to the
descriptor for compact result retention.
''',
    '''`CrossWindowAssociationResult.descriptor()` emits the complete semantic result:
configuration, ranking semantics, candidates, accepted links, unmatched tracks,
and rejection accounting. Schema version 2 includes corrected evaluated-pair
semantics and the strict construction contract. `result_id` is the SHA-256 digest
of the canonical finite-JSON encoding of that descriptor. `to_dict()` adds the ID
to the descriptor for compact result retention.
''',
    name="ranking identity documentation",
)
