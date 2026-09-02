# Truth-separated information-contract submissions

## Why the base suite is not enough for scientific claims

The base `prob4d.information-contract-suite` format is useful for deterministic
local replay because one NPZ contains every array needed by the evaluator. That
same convenience is inappropriate as the only public challenge interface:
provider outputs, held truth, registered losses, ambiguity definitions, and the
caller-owned fallback would be co-located in one mutable file.

A low-error submission could therefore be impossible to distinguish from an
adapter assembled after target inspection. A provider could also redefine the
nullspace, quotient, loss, or fallback that is supposed to test it. Hashing the
combined NPZ does not solve this ownership problem; it only hashes the leakage.

The truth-separated interface adds two independently hashed manifests:

- the **challenge** owns target truth, queries, gauge/nullspace or finite
  ambiguity, action losses, realized outcomes, and fallback;
- the **submission** owns provider means, uncertainty, query admissions,
  reported certificates, and selected actions.

The evaluator rejects a challenge payload containing provider-owned arrays and a
submission payload containing challenge-owned arrays. It joins the two only in a
temporary directory and applies the existing benchmark evaluator to the joined
core arrays.

## Information-order labels

Every challenge declares exactly one mode:

- `retrospective-open-target`: outcomes were already public or open when the
  adapter was created. The result is always labelled
  `retrospective-diagnostic`.
- `prospective-sealed-target`: the challenge was frozen before target opening.
  The submission must declare `prospective-sealed`, no target use, no target
  tuning, and that predictions were sealed before truth.

The Boolean declaration is necessary but not sufficient evidence. A prospective
paper claim must additionally retain an independently verifiable prediction
artifact, workflow/run identity, digest, and target-opening receipt. The
benchmark result therefore says `prospective_claim_eligible`, not “prospective
claim proven.”

A retrospective adapter may reproduce old results exactly and is valuable for
format validation. It must not be relabelled as a held-out experiment.

## Finite-query identifiability

The original `gauge` task evaluates a **local linear** question. For query row
`q` and declared nullspace basis `N`, it tests whether `qN` is negligible. A
finite symmetry can leave the derivative zero at one representative while the
query changes elsewhere on the observation-equivalence orbit.

The truth-separated interface therefore adds `finite_query`. The challenge owns:

| Array | Shape | Meaning |
| --- | --- | --- |
| `finite_query_value_by_hypothesis` | `(H,Q)` | query values on finite supported hypotheses |
| `finite_query_tolerance` | `(Q,)` | maximum allowed within-class range |
| `hypothesis_prior` | `(H,)` | positive entries define support |
| `quotient_class` | `(H,)` | declared observation/query-equivalence class |
| `quotient_mass` | `(C,)` | class masses, used for descriptive weighted width |

The submission owns `finite_query_admitted`, a Boolean vector of length `Q`.
For each class and query, the evaluator computes

\[
w_{cq}=\max_{i\in c,\;p_i>0}q_i-
        \min_{i\in c,\;p_i>0}q_i.
\]

The query is admitted exactly when

\[
\max_c w_{cq}\leq \tau_q.
\]

When both local `gauge` and `finite_query` are declared, the scorecard reports
how often the local test admits a query that the finite ambiguity rejects. This
is descriptive disagreement between two guarantees, not automatically a bug in
the local differential calculation.

## Strict ownership table

Challenge-owned arrays:

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

Submission-owned arrays:

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

The exact declared task set determines the exact allowed array set. Extra arrays
are rejected even when their names are otherwise registered. This prevents a
case from carrying undeclared target channels or silently changing what is being
scored.

## Dataset and producer provenance

The challenge binds a dataset manifest by path and SHA-256, plus dataset ID,
version, license, public/private classification, and information order. The
submission binds:

- provider contract;
- implementation, model, and calibration revisions;
- output coordinate frame;
- causal cutoff semantics;
- explicit dependence groups;
- producer-output manifest digest;
- target-use and target-tuning declarations; and
- submission mode.

These fields make provenance inspectable. They do not make a false declaration
true; claim-bearing releases still need externally reviewable artifacts.

## Communication accounting

The base local suite reports the complete joined NPZ size. A public communication
benchmark should not charge the provider for challenge-owned truth and losses.
The truth-separated result therefore records three values:

- challenge payload bytes;
- submission payload bytes; and
- temporary joined-evaluation payload bytes.

`payload_file_bytes` is redefined at this outer interface as the provider-owned
submission payload size. The dense and structured covariance array calculations
remain unchanged.

## Reproduce the sealed conformance control

```bash
python -m prob4d.information_contract_sealed smoke \
  /tmp/prob4d-contract-sealed

python -m prob4d.information_contract_sealed evaluate \
  /tmp/prob4d-contract-sealed/challenge/challenge.json \
  /tmp/prob4d-contract-sealed/submission/submission.json \
  /tmp/prob4d-contract-sealed/replayed.json

cmp /tmp/prob4d-contract-sealed/result.json \
  /tmp/prob4d-contract-sealed/replayed.json
```

The controlled pair is intentionally retrospective and synthetic. It checks:

- truth/submission ownership;
- exact case-roster matching;
- local versus finite query semantics;
- exact query and decision fallback;
- dependence-aware covariance scoring;
- provider-only communication accounting; and
- deterministic replay.

The controlled query set contains a case that is locally insensitive to the
supplied tangent nullspace but varies within every finite quotient class. The
local and finite checks both remain visible so the benchmark cannot promote one
as a substitute for the other.

## Public-data adapter rule

A public-data adapter must be generated from original per-case predictions and
outcomes, not from an aggregate paper table. For each panel:

1. publish a challenge dataset manifest with the complete case roster and
   independent grouping units;
2. publish challenge payloads without provider outputs;
3. publish one submission manifest per provider or ablation;
4. preserve original model, calibration, causal cutoff, and dependence
   identities;
5. label already-open data as retrospective;
6. seal any new prospective submission before target access; and
7. retain negative, support-negative, and technical-failure cases without
   replacement.

The first retrospective adapters should reproduce the retained DEFORM DLO
finite-decision result and Tracking Cloth dependence/compression result. Their
purpose is byte-level interface validation. New benchmark claims require a
separately frozen prospective panel.

## Remaining limitations

Version 1 still assumes metric 3-D rows, linear local queries, finite global
query hypotheses, finite action portfolios, and block-local plus low-rank
covariance. The challenge owns the declared ambiguity but the evaluator does not
prove that the declaration equals a learned provider's true observation
symmetry. It does not infer causal mechanisms, certify continuous robot control,
or turn retrospective public data into counterfactual action outcomes.
