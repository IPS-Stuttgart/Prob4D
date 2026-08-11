# Distribution boundaries

Prob4D publishes two conceptually different artifacts.

## Python distribution

The wheel and source distribution contain the installable library, typed contract
data, project metadata, documentation, and frozen protocol descriptions. They do
not contain GitHub workflows, generated evidence, CI environments, repository
tests, or one-off maintenance scripts.

The package includes both `py.typed` and the root `__init__.pyi` compatibility
stub. Runtime root exports are lazy, while static type checkers retain the exact
historical root inventory. The installed distribution can emit a
content-addressed public API manifest covering the root and both versioned
façades.

The source-distribution audit installs the built archive into an isolated virtual
environment and exercises the installed package, lazy root behavior, versioned
public APIs, public API manifest, contract data, and canonical CLI. It
intentionally does not run tests copied into the archive.

## Evidence capsule

Scientific evidence remains a separate, content-addressed artifact. Evidence
capsules bind the exact Prob4D, BayesianPhysTwin, and Causal4D revisions,
dependency locks, protocol identifiers, execution identity, and checksums needed
to reproduce a declared result. They are produced by explicit evidence workflows
and are not part of the package consumed by normal Python users.

A public API manifest or green installed-wheel capsule is compatibility and
provenance evidence. Neither is provider-accuracy, calibration, physical-query,
Causal4D intervention, deployment-safety, or state-of-the-art evidence.

This separation prevents repository-maintenance state from changing package
installation semantics and prevents stale copied tests or workflows from
blocking a valid source distribution.
