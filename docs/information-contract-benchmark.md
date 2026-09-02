# Probabilistic 4-D information-contract benchmark

## Purpose

Most 4-D benchmarks rank a method by point or trajectory error. That is
insufficient when a reconstruction is consumed by a physical estimator or
controller. Two submissions with similar point error can differ radically in
whether they:

- represent calibrated uncertainty;
- preserve cross-point and cross-time dependence;
- expose unobservable gauge directions;
- answer a registered physical query without inventing unsupported state;
- return the exact caller-owned fallback when a query is not identifiable; or
- communicate the information needed for a decision efficiently.

This benchmark defines a provider-neutral, NumPy-only scorecard for those
properties. It intentionally has **no single scalar leaderboard**. Accuracy
cannot compensate for a false gauge admission, a fabricated full-rank
covariance, or a broken fallback contract.

The current implementation is an experimental benchmark MVP. Its deterministic
smoke suite is development evidence only; no learned provider or public dataset
result is claimed by this change.

## Evaluation axes

A case declares the tasks it supports. Dependencies are fail-closed: for example,
`dependence` requires `calibration`, `gauge` requires `query`, and `fallback`
requires `gauge`.

| Task | Required evidence | Primary outputs |
| --- | --- | --- |
| `forecast` | truth and provider mean | coordinate RMSE, point RMSE, mean point error |
| `calibration` | conditional 3-D covariance and shared low-rank factor | Gaussian NLL, normalized NEES, marginal coverage and width |
| `dependence` | same mean and coordinate marginals | full-vs-marginal-matched-diagonal NLL and NEES gains |
| `query` | registered linear physical queries | query RMSE, NLL, NEES, coverage and width |
| `gauge` | declared nullspace basis and provider admission decisions | exact nullspace sensitivity, false accepts and false rejects |
| `fallback` | caller-owned fallback and reported query moments | branch consistency and exact rejected-query fallback |
| `decision` | finite hypotheses, quotient masses and action losses | exact worst-case regret, admission correctness and realized regret |
| `communication` | covariance representation | shared rank and dense-to-structured covariance byte ratios |

The evaluator distinguishes **performance metrics** from **contract checks**.
A case can achieve low error while failing its information contract. Such a
submission remains visible in the scorecard but does not pass the contract.

## Statistical units

Every case has a `group_id`. The benchmark reports:

1. individual case results;
2. equal-case means;
3. per-group means; and
4. equal-group means.

A physical object, specimen, or independently acquired session should normally
be one group. Frames, points, overlapping windows, and multiple queries from one
recording must not be relabelled as independent groups. Dataset-specific suites
must freeze this rule before target outcomes are opened.

## Suite manifest

A suite is a JSON object with schema
`prob4d.information-contract-suite`, version 1:

```json
{
  "schema_name": "prob4d.information-contract-suite",
  "schema_version": 1,
  "suite_id": "example-suite-v1",
  "aggregation_unit": "group_id",
  "thresholds": {
    "coverage_probability": 0.9,
    "gauge_sensitivity_tolerance": 1e-10,
    "moment_atol": 1e-12,
    "relative_rank_tolerance": 1e-10
  },
  "claim_boundary": "Bounded statement specific to this frozen suite.",
  "cases": [
    {
      "case_id": "object-01/session-01",
      "group_id": "object-01",
      "payload": "cases/object-01-session-01.npz",
      "payload_sha256": "<lowercase sha256>",
      "tasks": [
        "forecast",
        "calibration",
        "dependence",
        "query",
        "gauge",
        "fallback",
        "decision",
        "communication"
      ],
      "metadata": {
        "dataset": "public-dataset-name",
        "split": "held-out"
      }
    }
  ]
}
```

Manifest and payload fields not registered by version 1 are rejected. Payload
paths must be relative, remain inside the suite directory, and match their
declared SHA-256 exactly. NPZ files are opened with `allow_pickle=False`.

The companion machine-readable manifest schema is
[`benchmarks/information_contract_v1/suite.schema.json`](../benchmarks/information_contract_v1/suite.schema.json).
Array shapes and cross-array dependencies are enforced by the evaluator because
JSON Schema cannot inspect NPZ members.

## NPZ payload

### Forecast and covariance

The state is a nonempty sequence of 3-D rows.

| Array | Shape | Meaning |
| --- | --- | --- |
| `truth_xyz_m` | `(N, 3)` | held metric truth |
| `prediction_mean_xyz_m` | `(N, 3)` | submitted provider mean |
| `conditional_covariance_m2` | `(N, 3, 3)` | positive-definite row-local covariance |
| `shared_factor_m` | `(N, 3, R)` | shared factor \(U\), yielding \(C=D+UU^\top\) |

All values must be finite. The local covariance is checked for symmetry and
positive definiteness. The evaluator uses Woodbury and determinant identities;
it never needs to materialize the dense \(3N\times3N\) covariance.

The dependence control replaces the submitted covariance with a diagonal
covariance having the **same coordinate marginals**. Consequently, a measured
NLL gain isolates the submitted cross-coordinate dependence rather than a
change in mean or marginal variance.

### Queries, gauge and fallback

| Array | Shape | Meaning |
| --- | --- | --- |
| `query_matrix` | `(Q, 3N)` | registered linear queries |
| `nullspace_basis` | `(3N, G)` | declared unobservable directions |
| `query_admitted` | `(Q,)` Boolean | provider admission decisions |
| `reported_query_mean` | `(Q,)` | moments returned to the caller |
| `reported_query_variance` | `(Q,)` | returned positive variances |
| `fallback_mean_xyz_m` | `(N, 3)` | caller-owned physical fallback |
| `fallback_conditional_covariance_m2` | `(N, 3, 3)` | fallback local covariance |
| `fallback_shared_factor_m` | `(N, 3, R_f)` | fallback shared factor |

The evaluator orthonormalizes the declared nullspace span. For query row \(q\)
and orthonormal nullspace basis \(N\), the normalized sensitivity is

\[
s(q)=\frac{\lVert qN\rVert_2}{\lVert q\rVert_2}.
\]

The structurally expected admission is \(s(q)\leq\tau_g\), where \(\tau_g\) is
frozen in the suite. False acceptance and false rejection are reported
separately.

For each query, the submitted output must equal the candidate moments when
admitted and the fallback moments when rejected. Rejected-query moments are
required to be exactly equal to the fallback arrays after deterministic
projection; a merely close alternative is not called exact fallback.

This gauge check validates only the supplied basis. It does not prove that the
basis is the true observation-equivalence class of a learned provider.

### Finite-action decision record

| Array | Shape | Meaning |
| --- | --- | --- |
| `decision_loss_by_hypothesis` | `(H, A)` | registered loss for each finite hypothesis/action |
| `hypothesis_prior` | `(H,)` | nonnegative prior; positive entries define support |
| `quotient_class` | `(H,)` integer | contiguous class labels `0..C-1` |
| `quotient_mass` | `(C,)` | posterior class masses |
| `reported_worst_case_regret` | `(A,)` | submitted certificate |
| `selected_action` | scalar integer | returned action |
| `fallback_action` | scalar integer | caller-owned fallback |
| `decision_admitted` | scalar Boolean | submitted admission |
| `regret_tolerance` | scalar | frozen nonnegative tolerance |
| `realized_action_loss` | `(A,)` | held losses used only for evaluation |

For actions \(a,b\), the exact compatible-belief gap is

\[
\overline\Delta(a,b)=
\sum_c \lambda_c
\max_{i:\pi(i)=c,\;p_i>0}
\left(\ell_{ia}-\ell_{ib}\right).
\]

Worst-case regret is
\(\overline R(a)=\max_b\overline\Delta(a,b)\).
The evaluator recomputes this quantity, verifies the reported vector, applies a
deterministic smallest-index minimax tie rule, and checks admission and exact
fallback semantics. Realized target regret is reported separately; it is not
confused with the registered finite-support guarantee.

## Scorecard interpretation

The output schema is
`prob4d.information-contract-benchmark-result`, version 1. Each case contains:

- task metrics;
- explicit Boolean contract checks;
- a case-level `contract_pass`;
- immutable payload identity; and
- metadata copied from the suite.

The aggregate contains equal-case and equal-group means plus contract-failure
counts. It does **not** contain an overall score.

Recommended comparison order:

1. verify identical suite and task coverage;
2. reject or separately label contract-failing submissions;
3. compare accuracy and probabilistic scores;
4. compare decision utility and fallback frequency;
5. compare covariance payload and runtime.

A method that omits uncertainty can still enter a `forecast` case. It cannot
claim calibration, dependence, gauge, or decision-contract performance without
supplying the corresponding registered evidence.

## Deterministic smoke control

Run:

```bash
python -m prob4d.information_contract_benchmark smoke /tmp/prob4d-contract-smoke
python -m prob4d.information_contract_benchmark evaluate \
  /tmp/prob4d-contract-smoke/suite.json \
  /tmp/prob4d-contract-smoke/replayed.json
cmp /tmp/prob4d-contract-smoke/result.json \
  /tmp/prob4d-contract-smoke/replayed.json
```

The smoke fixture contains two independent groups:

- one exact admissible decision with a shared-dependence covariance; and
- one deliberately ambiguous decision that must return the registered fallback.

It also includes one gauge-invariant and one gauge-sensitive query. The fixture
is generated with deterministic ZIP metadata, so its NPZ bytes and hashes are
stable across repeated generation with the same NumPy array format.

The smoke result is not a scientific benchmark result. It checks evaluator
semantics and serialization only.

## Public real-data benchmark plan

The benchmark layer is designed to consume existing public evidence without
claiming that every panel is already complete.

### Panel A: DEFORM DLO

The official DEFORM release provides real DLO1--DLO5 train/evaluation
trajectories. A frozen suite can score:

- forecast and query accuracy;
- finite-action decision certificates;
- exact physical fallback;
- source-support versus realized-regret mismatch; and
- equal-DLO rather than equal-window aggregation.

The current DLO4/DLO5 decision study is a natural first adapter, but the adapter
must be generated from the exact sealed prediction and outcome artifacts. A
paper-side summary JSON is not a substitute for the original per-case payload.

### Panel B: Tracking Cloth Deformation

The public motion-capture release can score:

- full versus marginal-matched dependence;
- query-sufficient covariance compression;
- finite-orbit gauge admission;
- exact fallback; and
- bytes per query or decision.

Independent recordings or physical specimens, not windows, must be the
aggregation groups.

### Panel C: learned-provider stress panels

A learned-provider panel should add:

- common-mode visual bias;
- cross-provider corroboration with explicit shared dependence;
- nullspace/gauge reporting;
- provider-support failure; and
- target-free admission before held outcomes are opened.

Agreement between two providers is not correctness when both share a bias.
Independent metric anchors or held physical outcomes remain necessary.

## Governance for claim-bearing suites

A claim-bearing public suite should publish:

1. dataset version, license and checksums;
2. exact case roster and statistical-unit grouping;
3. causal observation cutoff;
4. sealed submission identities;
5. query, gauge, action, loss and fallback definitions;
6. thresholds frozen before target outcomes;
7. evaluator revision and result hash;
8. all negative and unsupported cases without replacement; and
9. a precise claim boundary.

A benchmark version must never silently change the case roster or thresholds.
Scientific changes require a new suite ID and versioned result. Technical
re-execution is allowed only when no scored outcome was opened or when exact
byte-for-byte replay is demonstrated.

## Current boundary

Version 1 is deliberately narrow:

- metric rows are 3-D;
- covariance is block-local plus a shared low-rank factor;
- queries are linear;
- the gauge basis is supplied rather than inferred;
- decision losses use a finite hypothesis and action set;
- the conformal validity of a target-support envelope is outside this evaluator;
- no sequential control or counterfactual robot-action outcome is inferred.

These restrictions make the contract auditable. Future versions can add
nonlinear query witnesses, continuous actions, sequential decision costs, and
dataset adapters without weakening version-1 semantics.
