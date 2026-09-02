# Truth-separated information-contract submissions

## Why the split is necessary

A convenient local replay bundle may co-locate predictions and held outcomes.
That format is useful for deterministic testing, but it is not sufficient for a
scientific benchmark: a provider could alter truth, the declared nullspace,
query definitions, action losses, or fallback while simultaneously supplying
its predictions.

The sealed interface therefore separates every case into two hash-bound
payloads:

- the **challenge payload** owns target truth, registered queries, ambiguity
  classes, action losses, tolerances, and caller-owned fallback;
- the **submission payload** owns provider means, covariance, query admissions,
  reported certificates, and selected actions.

The evaluator rejects any array placed on the wrong side and requires the exact
array set implied by the declared tasks. Challenge and submission case rosters
must match exactly.

This is a stronger information barrier than the self-contained
`prob4d.information-contract-suite` replay format. The replay format remains
useful internally; public scorecards should use the truth-separated interface.

## Information-order classes

Every challenge declares one of two modes:

| Mode | Meaning | Permitted result class |
| --- | --- | --- |
| `retrospective-open-target` | target outcomes were already available when the adapter was constructed | diagnostic replay |
| `prospective-sealed-target` | a separately auditable process sealed the submission before target opening | prospective held-out result |

A retrospective result cannot be relabelled as prospective by setting a Boolean
in the submission. The submission mode and seal declaration must agree with the
challenge mode. The resulting scorecard carries
`prospective_claim_eligible=false` for every retrospective challenge.

The evaluator validates declarations and content identities. It cannot by
itself prove an external chronology. A claim-bearing prospective release must
also retain the producer artifact identity, target-opening receipt, workflow
identity, and dataset manifest referenced by the challenge.

## Challenge-owned arrays

The challenge side owns:

```text
truth_xyz_m
fallback_mean_xyz_m
fallback_conditional_covariance_m2
fallback_shared_factor_m
query_matrix
nullspace_basis
decision_loss_by_hypothesis
hypothesis_prior
quotient_class
quotient_mass
fallback_action
regret_tolerance
realized_action_loss
finite_query_value_by_hypothesis
finite_query_tolerance
```

## Submission-owned arrays

The provider side owns:

```text
prediction_mean_xyz_m
conditional_covariance_m2
shared_factor_m
query_admitted
reported_query_mean
reported_query_variance
reported_worst_case_regret
selected_action
decision_admitted
finite_query_admitted
```

An NPZ with an array owned by the other side fails before scoring. Extra arrays
also fail: a provider cannot hide target selection, a second unregistered
prediction, or a post-hoc confidence score in the payload.

## Finite-query identifiability

The original linear-gauge axis checks local sensitivity of a registered query
matrix to a supplied nullspace span. That is useful but not globally sufficient.
A query can have zero first derivative at one representative and still vary
over a finite observation-equivalence class.

The truth-separated interface adds the task `finite_query`. For supported
hypotheses $h_i$, quotient class $\pi(i)$, query values $q_k(h_i)$, and
registered tolerance $\tau_k$, it computes

$$
w_{ck}
=
\max_{i:\pi(i)=c,\,p_i>0}q_k(h_i)
-
\min_{i:\pi(i)=c,\,p_i>0}q_k(h_i),
$$

and admits query $k$ exactly when

$$
\max_c w_{ck}\leq \tau_k.
$$

The scorecard reports false accepts, false rejects, class widths,
mass-weighted widths, and the number of queries that a local nullspace test
admits while the finite quotient rejects. This directly exposes the
stationary-derivative blind spot rather than allowing a local gate to stand in
for global identifiability.

The finite values and quotient are challenge-owned. A provider may submit only
its admission decision.

## Producer manifest

Each submission binds:

- provider and contract names;
- implementation, model, and calibration revisions;
- output coordinate frame;
- causal cutoff;
- dependence-group identities;
- producer output-manifest SHA-256;
- target-outcome and target-tuning declarations; and
- retrospective or prospective submission mode.

Two providers that share an input or calibration dependence group must declare
that common group. Distinct implementation names do not imply independent
errors.

## Communication accounting

The self-contained evaluator previously reported the complete joined NPZ file
size. In a public benchmark that would charge the provider for challenge-owned
truth and losses. The sealed scorecard therefore reports:

- challenge payload bytes;
- submission payload bytes;
- temporary joined evaluator bytes;
- covariance array bytes; and
- dense-to-structured semantic ratios.

`payload_file_bytes` is redefined at the sealed layer as the submitted provider
payload size. The lower-level joined size remains visible for reproducibility.

## Commands

Generate and replay the deterministic truth-separated control:

```bash
python -m prob4d.information_contract_sealed smoke \
  /tmp/prob4d-information-contract-sealed

python -m prob4d.information_contract_sealed evaluate \
  /tmp/prob4d-information-contract-sealed/challenge/challenge.json \
  /tmp/prob4d-information-contract-sealed/submission/submission.json \
  /tmp/prob4d-information-contract-sealed/replayed.json

cmp /tmp/prob4d-information-contract-sealed/result.json \
  /tmp/prob4d-information-contract-sealed/replayed.json
```

The smoke challenge is deliberately labelled
`retrospective-open-target`. It demonstrates:

- byte-stable challenge/submission joining;
- target-truth ownership;
- exact case-roster matching;
- local-versus-finite query disagreement;
- exact query and decision fallback;
- dependence-aware Gaussian scoring; and
- provider-only communication accounting.

It is development evidence, not provider performance.

## Public-data adapter requirements

The first public adapters must be generated from the original per-case
prediction and outcome records, not from a paper summary table. A release must
include:

1. a content-addressed dataset roster and license identifier;
2. one challenge case per registered statistical unit or nested evaluation case;
3. complete group labels that prevent windows or points from becoming
   independent replicates;
4. a producer manifest with exact source, model, and calibration identities;
5. separate challenge and submission payload hashes;
6. a declared information-order class;
7. every failed, unsupported, and fallback case without replacement; and
8. a scorecard generated by a fixed evaluator revision.

For already opened DEFORM and Tracking Cloth results, the first adapters are
retrospective diagnostics. New paper claims require a separately frozen
prospective challenge.
