# Cross-stack moving-head drift diagnostic

The claim-bearing cross-stack workflow remains pinned to exact reviewed
BayesianPhysTwin and Causal4D revisions. Its installed-wheel result is stable and
replayable, but a pin cannot reveal incompatibilities introduced later on a
companion repository's default branch.

`Cross-stack moving-head drift diagnostic` is a separate scheduled and manually
dispatchable workflow. It checks out the current `main` revision of Prob4D,
BayesianPhysTwin, and Causal4D, records all three resolved 40-character Git
revisions, builds exactly three wheels, installs only those wheels in an isolated
environment, and runs the existing three-repository metamorphic invariants.

The retained artifact includes the resolved revisions, wheel SHA-256 values,
JUnit output, and the complete test log. A failure is an interoperability drift
signal. It is not a scientific result, a release qualification, a provider
promotion decision, or permission to update any frozen evidence reference.

When the diagnostic finds drift, repair the owning repository and then update a
claim-bearing pin only through a normal reviewed pull request whose exact-head
pinned workflow passes.
