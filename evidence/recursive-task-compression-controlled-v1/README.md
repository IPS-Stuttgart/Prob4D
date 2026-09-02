# Recursive task-sufficient compression — controlled mechanism evidence

This record binds the first immutable controlled execution of the recursive task-state composition
in PR #475. It is mechanism evidence only; it opens no dataset and establishes no robotics or
learned-provider result.

## Registered separation

The controlled linear-Gaussian design deliberately separates three dimensions:

- complete state: **20**;
- current reported task: **3**; and
- exact strict recursive task closure: **4**.

The supplied correlated measurement-noise factor has rank **8**. The current-task-only projection
retains rank **3** and is exact at the first update, but it does not preserve the fourth closure
coordinate that drives a later task component. The closure-aware projection retains rank **4**.

## Result

Across all eight registered updates, closure-aware compression matches the complete task-state
filter to numerical precision:

- maximum posterior-mean absolute error: `2.42861286636753e-16`;
- maximum posterior-covariance absolute error: `6.938893903907228e-18`.

The rank-3 current-task control is initially exact (`2.3592239273284576e-16` mean error) but reaches
`0.05289867007400557` at the next update and a maximum later error of
`0.05336337058687904`.

A registered nuisance-to-task transition violation expands the audited closure from **4 to 5**
instead of silently reusing the smaller representation.

## Provenance

- PR: `IPS-Stuttgart/Prob4D#475`;
- branch head executed: `c35142afec22bc70eafb5b0f28a10761fbda039e`;
- GitHub pull-request merge revision executed by Actions:
  `3e8bfbb0cbf2be06ec9e631e2ccbd3ac77ead51f`;
- workflow run: `33590089678`;
- workflow job: `100122141300`;
- artifact: `9831421895`;
- artifact SHA-256: `c4b25bec15bbe2401d8f1f14b3f14825807c352f6dff67d9be39f4cec693f180`;
- protocol SHA-256: `4a0d7cdab7e4b6a45aac780fd52a75ae9d649eebe169f2cc3715506d135d8191`;
- result SHA-256: `a455c1b235be5ff0dd5f61b706f996c2d5d7da3d1726c521f8a095f3e29412d4`;
- Python `3.12.14`, NumPy `2.2.6`.

The authoritative machine-readable values are in `summary.json` and the immutable Actions artifact.

## Claim boundary

This execution supports the algebraic statement that one-step preservation of only today's task is
not generally recursively sufficient, while preserving a registered strict task-state closure can
compose exactly through the controlled LTI Gaussian recursion.

It does **not** establish novelty of functional filtering, minimum-order functional observers, or
state aggregation. It does not preserve observation evidence and does not establish nonlinear,
public-real-data, learned-provider, closed-loop-control, deployment, or state-of-the-art claims.
