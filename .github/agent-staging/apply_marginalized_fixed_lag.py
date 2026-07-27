from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence in {path}, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "src/prob4d/observation_export.py",
    """from .gauge import (
    FixedLagGaugeSmoother,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
""",
    """from .gauge import RelativeGaugeConstraint, SequentialGaugeEstimator
from .marginalized_gauge import MarginalizedFixedLagGaugeSmoother
""",
)
replace_once(
    "src/prob4d/observation_export.py",
    '"""Return the legacy fixed-lag marginals as an explicitly approximate posterior."""',
    '"""Return fixed-lag marginals with a Schur-marginalized boundary prior."""',
)
replace_once(
    "src/prob4d/observation_export.py",
    "estimates = FixedLagGaugeSmoother(lag=fixed_lag).smooth(",
    "estimates = MarginalizedFixedLagGaugeSmoother(lag=fixed_lag).smooth(",
)
replace_once(
    "src/prob4d/observation_export.py",
    'mode="fixed_lag_block_diagonal_approximation_v1",',
    'mode="fixed_lag_schur_boundary_block_diagonal_v2",',
)
replace_once(
    "src/prob4d/observation_export.py",
    """            "fixed_lag covariance treats marginalized boundary gauges as exact; "
            "pass allow_approximate_fixed_lag_covariance=True only for an explicitly "
            "labelled reconstruction ablation"
""",
    """            "fixed_lag preserves its moving boundary prior but not historical "
            "cross-window covariance; pass allow_approximate_fixed_lag_covariance=True "
            "only for an explicitly labelled reconstruction ablation"
""",
)
replace_once(
    "src/prob4d/observation_export.py",
    """            "fixed_lag_boundary_covariance_is_approximate": (
                gauge_mode == "fixed_lag"
            ),
""",
    """            "fixed_lag_boundary_covariance_is_approximate": (
                gauge_mode == "fixed_lag"
            ),
            "fixed_lag_boundary_prior": (
                "schur_complement_v1" if gauge_mode == "fixed_lag" else None
            ),
""",
)
replace_once(
    "tests/test_observation_export.py",
    'match="marginalized boundary gauges as exact"',
    'match="not historical cross-window covariance"',
)
replace_once(
    "src/prob4d/provider_manifest.py",
    '    "metric_anchor_covariance_propagation",\n',
    '    "metric_anchor_covariance_propagation",\n'
    '    "marginalized_fixed_lag_boundary_prior",\n',
)
replace_once(
    "src/prob4d/provider_manifest.py",
    """                "causal sequential spanning tree by default; fixed-lag block-diagonal "
                "covariance is an explicit approximate reconstruction control"
""",
    """                "causal sequential spanning tree by default; fixed-lag carries a "
                "Schur-marginalized boundary prior but exports block-diagonal historical "
                "marginals as an approximate reconstruction control"
""",
)
replace_once(
    "src/prob4d/provider_manifest.py",
    """            "calibration_semantics": (
""",
    """            "fixed_lag_covariance_semantics": (
                "expired gauges are Schur-marginalized into the active boundary; the "
                "portable all-window covariance remains block-diagonal and non-strict"
            ),
            "calibration_semantics": (
""",
)
replace_once(
    "README.md",
    """The production default is a causal sequential spanning tree with the **full
joint cross-window gauge covariance** propagated from the metric anchor. A rank
cap is accepted only when the retained covariance-trace fraction satisfies the
explicit threshold. The legacy fixed-lag covariance is available solely as an
opt-in reconstruction ablation because its current boundary treatment fixes
marginalized gauges at their posterior means. See [the causal observation export
contract](docs/observation-belief-export.md).
""",
    """The production default is a causal sequential spanning tree with the **full
joint cross-window gauge covariance** propagated from the metric anchor. A rank
cap is accepted only when the retained covariance-trace fraction satisfies the
explicit threshold. Fixed-lag mode now carries a Schur-complement information
prior when a gauge leaves the active window, rather than fixing that boundary
with zero uncertainty. Its portable all-window covariance still contains only
historical marginal blocks, so it remains an opt-in reconstruction ablation. See
[the causal observation export contract](docs/observation-belief-export.md).
""",
)
replace_once(
    "docs/architecture.md",
    """The legacy fixed-lag covariance is an opt-in reconstruction control because its
current boundary treatment fixes marginalized gauges at posterior means. The
provider makes no prospective calibration or physical-twin-improvement claim.
""",
    """Fixed-lag smoothing carries a Schur-complement information prior when gauges
leave the active window, so the moving boundary does not become exact. The
portable all-window covariance still exports only historical marginal blocks and
therefore remains an opt-in reconstruction control. The provider makes no
prospective calibration or physical-twin-improvement claim.
""",
)
replace_once(
    "docs/provider-contract.md",
    """The production default is a causal sequential spanning tree. It preserves the
uncertainty of the selected causal constraints without pretending that redundant
dense alignment edges are independent. The fixed-lag covariance path remains an
explicit reconstruction control and is not labelled as strict stream contract
v2 because its current boundary treatment fixes marginalized gauges at posterior
means.
""",
    """The production default is a causal sequential spanning tree. It preserves the
uncertainty of the selected causal constraints without pretending that redundant
dense alignment edges are independent. Fixed-lag smoothing now carries a
Schur-complement information prior across the moving boundary, but its portable
all-window covariance contains only block-diagonal historical marginals. It
therefore remains an explicit reconstruction control and is not labelled as
strict stream contract v2.
""",
)
replace_once(
    "docs/observation-belief-export.md",
    """The legacy `--gauge-mode fixed_lag` path remains available only with
`--allow-approximate-fixed-lag-covariance`. Its current covariance treats gauges
outside the active lag as exact posterior means and exports only block-diagonal
marginals. It is suitable for a labelled reconstruction ablation, not for the
strict stream-v2 Bayesian uncertainty claim. A future fixed-lag implementation
must carry a marginalized boundary information prior before this acknowledgement
can be removed.
""",
    """The `--gauge-mode fixed_lag` path remains available only with
`--allow-approximate-fixed-lag-covariance`. When the oldest active gauge expires,
its factors and the previous boundary prior are linearized and eliminated through
a Schur complement. The active boundary therefore retains that uncertainty. The
portable all-window artifact still exports only block-diagonal historical
marginals and cannot reconstruct historical cross-window covariance. Fixed-lag
mode is consequently suitable for a labelled reconstruction ablation, not for
the strict stream-v2 Bayesian uncertainty claim.
""",
)
replace_once(
    "CHANGELOG.md",
    "# Changelog\n\nAll notable changes to Prob4D are documented here.\n",
    """# Changelog

All notable changes to Prob4D are documented here.

## Unreleased

### Changed

- Fixed-lag gauge smoothing now Schur-marginalizes expired gauges into an
  uncertainty-bearing boundary prior. Portable historical covariance remains an
  explicit block-diagonal reconstruction approximation.
""",
)
