#!/usr/bin/env python3
"""Apply the exact source-tree changes for the Prob4D 0.4.0 boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(
            f"expected one occurrence in {path}: {old!r}; found {text.count(old)}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def _replace_all(path: str, old: str, new: str, *, minimum: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count < minimum:
        raise SystemExit(
            f"expected at least {minimum} occurrences in {path}: {old!r}; found {count}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def _update_changelog() -> None:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    opening = "## Unreleased\n\n### Added\n"
    if text.count(opening) != 1:
        raise SystemExit("CHANGELOG Unreleased opening changed")
    added = """## Unreleased

No changes yet.

## 0.4.0 — 2026-08-10

### Added

- Portable `O(K)` causal gauge-tree prior artifacts, tree-backed sparse
  observation-factor stacks, and direct sparse observation production without
  materializing or serializing the complete joint gauge covariance.
- Portable and claim-bearing tree-sparse observation artifacts that bind exact
  provider, calibration, runtime-revision, causal-lineage, observation, and
  sparse-prior identities for strict BayesianPhysTwin admission.
- Outcome-blind provider support feasibility over a frozen causal-prefix stream
  roster, geometry, camera calibration, metric anchors, admission rule, and
  predeclared technical exclusions before residual generation.
- Spatially stratified causal tracklets, persistent seed-cell lineage,
  frame-budget-preserving correlation groups, and a frozen-roster camera-panel
  support audit with policy-derived decisions.
- Replay-complete held-out provider evidence that binds calibration selection,
  exact provider-report bytes, sealed target decisions, exact fallback,
  bootstrap settings, and the deterministic Prob4D-to-BayesianPhysTwin
  promotion report in one content-addressed artifact.
- Paired joint-covariance dependence ablations comparing full shared covariance,
  marginal-preserving independence, and conditional-only controls with
  equal-group bootstrap summaries.
"""
    text = text.replace(opening, added, 1)

    changed_marker = "\n### Changed\n"
    if text.count(changed_marker) < 1:
        raise SystemExit("CHANGELOG Changed section missing")
    changed = """
### Changed

- The canonical grouped `prob4d` registry now owns every command target and
  historical `prob4d-*` executable. Legacy names remain operational through lazy
  compatibility wrappers that print the exact grouped replacement and a
  documented pre-1.0 removal policy.
- Claim-bearing tree-sparse artifacts are consumable downstream without dense
  gauge-prior serialization; provider API v1 remains frozen at bundle schema 3,
  while provider API v2 and the extended tree-sparse manifest advertise schema 4
  and the new sparse envelope contracts.
- Spatial seed-cell factors share the historical frame-level generalized-Bayes
  budget instead of multiplying effective likelihood power across cells.
"""
    text = text.replace(changed_marker, changed, 1)

    fixed_marker = "\n### Fixed\n"
    if text.count(fixed_marker) < 1:
        raise SystemExit("CHANGELOG Fixed section missing")
    fixed = """
### Fixed

- Selection-lock publication is durable and atomically no-clobber, including
  concurrent-writer and different-content rejection.
- Legacy PhysTwin input loading, numerical controls, camera geometry, and
  nearest-neighbor validation fail closed on ambiguous or malformed inputs.
- Sparse observation-factor contracts reject coercive identities and lossy
  indices, verify covariance geometry in bounded chunks, and preserve exact
  conditional-versus-marginal accounting.
- Camera-panel support cannot omit a declared camera, amplify a frame's
  likelihood budget by splitting it into cells, or forge supported views and
  frame decisions inconsistent with retained cell counts and the frozen policy.
"""
    text = text.replace(fixed_marker, fixed, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    _replace_once(
        "src/prob4d/__init__.py",
        '__version__ = "0.3.1"',
        '__version__ = "0.4.0"',
    )
    _replace_once(
        "src/prob4d/provider_manifest.py",
        'PROB4D_PROVIDER_PACKAGE_VERSION = "0.3.1"',
        'PROB4D_PROVIDER_PACKAGE_VERSION = "0.4.0"',
    )
    _replace_once(
        "tests/test_release_metadata.py",
        'EXPECTED_VERSION = "0.3.1"',
        'EXPECTED_VERSION = "0.4.0"',
    )
    _replace_once(
        "tests/test_provider_manifest.py",
        'assert PROB4D_PROVIDER_PACKAGE_VERSION == "0.3.1"',
        'assert PROB4D_PROVIDER_PACKAGE_VERSION == "0.4.0"',
    )
    _replace_all(
        ".github/workflows/tests.yml",
        '"0.3.1"',
        '"0.4.0"',
        minimum=2,
    )
    _replace_all(
        "tests/test_ecosystem_release_capsule.py",
        "prob4d-0.3.1",
        "prob4d-0.4.0",
        minimum=2,
    )
    _replace_once(
        "tests/test_ecosystem_release_capsule.py",
        "prob4d-0.3.2",
        "prob4d-0.4.1",
    )
    _replace_once(
        "tests/test_release_metadata.py",
        '        "docs/ecosystem-release-capsule.md",\n',
        '        "docs/ecosystem-release-capsule.md",\n'
        '        "docs/releases/0.4.0.md",\n',
    )
    _update_changelog()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
