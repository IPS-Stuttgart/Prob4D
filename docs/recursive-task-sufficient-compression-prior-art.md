# Prior-art boundary for recursive task sufficiency

This research branch must not claim that recursive estimation of state functions is new.

Relevant established lines include:

- Luenberger and subsequent reduced-order / functional observers;
- minimum-order observers for linear functions of the state;
- functional observability and detectability;
- state aggregation / exact quotient models for linear systems; and
- reduced-order and distributed Kalman filtering.

The candidate Prob4D contribution is narrower: given a recursively sufficient registered task
state, identify and audit the minimum directions of an already supplied correlated
measurement-noise factor that preserve every task-state Gaussian posterior update, with exact
full-factor fallback when the recursive task state or factor rank cannot be reduced.

A future manuscript must separately establish novelty against the functional-observer literature
and against lossless measurement transformations for distributed estimation with correlated
noise. The controlled study on this branch is mechanism evidence only; it is not a prior-art
novelty determination and not a robotics-performance result.
