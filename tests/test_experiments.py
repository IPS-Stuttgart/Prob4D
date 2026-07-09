from prob4d.experiments import run_synthetic_ablation


def test_synthetic_ablation_covers_all_requested_variants() -> None:
    rows, calibration = run_synthetic_ablation(
        seed=3,
        num_frames=45,
        height=4,
        width=6,
    )

    assert [row.key for row in rows] == [
        "disjoint",
        "latent_linear",
        "decoded_uniform",
        "precision",
        "ci",
        "ci_smoothed",
        "ci_smoothed_anchored",
    ]
    assert calibration.count > 0
    assert rows[3].sequence_metrics.coverage_95 <= rows[4].sequence_metrics.coverage_95
    assert rows[-1].gauge_metrics is not None

