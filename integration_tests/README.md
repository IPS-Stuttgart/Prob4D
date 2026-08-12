# Prob4D-Owned Ecosystem Integration Tests

Files named `test_three_repository_*.py` in this directory are executed by the
BayesianPhysTwin installed-wheel golden path after exact Prob4D,
BayesianPhysTwin, and Causal4D source snapshots have been built into wheels and
installed into a clean environment.

These tests own producer-side assumptions. They must import only installed
public package surfaces, must not reach into any source checkout, and must not
open scientific data or generated evidence. Repository-local support modules
and fixtures may live below this directory and are staged with the tests.

The first owned test verifies that all three independently implemented packages
validate and materialize the same content-locked `phys4d.observation_belief.v1`
corpus. Passing establishes contract interoperability only; it does not
establish provider accuracy, calibration, physical benefit, intervention
benefit, deployment safety, or state of the art.
