# Distribution boundaries

Prob4D publishes two conceptually different artifacts.

## Python distribution

The wheel and source distribution contain the installable library, typed contract
data, project metadata, documentation, and frozen protocol descriptions. They do
not contain GitHub workflows, generated evidence, CI environments, repository
tests, or one-off maintenance scripts.

The source-distribution audit installs the built archive into an isolated virtual
environment and exercises the installed package, public API, contract data, and
canonical CLI. It intentionally does not run tests copied into the archive.

## Evidence capsule

Scientific evidence remains a separate, content-addressed artifact. Evidence
capsules bind the exact Prob4D, BayesianPhysTwin, and Causal4D revisions,
dependency locks, protocol identifiers, execution identity, and checksums needed
to reproduce a declared result. They are produced by explicit evidence workflows
and are not part of the package consumed by normal Python users.

This separation prevents repository-maintenance state from changing package
installation semantics and prevents stale copied tests or workflows from blocking
a valid source distribution.
