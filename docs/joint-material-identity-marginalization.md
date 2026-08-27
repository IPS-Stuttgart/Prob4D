# Joint one-to-one material-identity marginalization

Prob4D already represents cross-window material identity as a calibrated local
mixture with an explicit null/no-match hypothesis. That contract is exact for
one target-local track. Applying several local mixtures independently is not,
however, a valid multi-track identity distribution: two target tracks can reuse
the same source endpoint in the same joint hypothesis.

This module adds an exact distribution over **one-to-one partial matchings**. It
keeps soft ambiguity, unlike a hard Hungarian assignment, while assigning zero
probability to source reuse.

## Model

Let target track `t` have local candidate set `C_t`. A candidate is either the
mandatory null hypothesis `null` or a link to one window-local source endpoint.
Prob4D supplies calibrated local log weights `log q_t(a_t)`. Define the valid
joint assignment set

```text
A = {a = (a_1, ..., a_T): every linked source endpoint occurs at most once}.
```

The source-side joint prior is

```text
p(a) = exp(sum_t log q_t(a_t)) / Z_0,   a in A.
```

A downstream consumer may supply candidate log likelihoods
`ell_t(a_t | x, g)` conditional on its continuous physical state `x` and the
complete explicit joint `Sim(3)` gauge nuisance `g`. Under conditional
factorization across target tracks,

```text
log p(y | x, g) = log Z_1 - log Z_0,

Z_1 = sum_{a in A} exp(sum_t [log q_t(a_t) + ell_t(a_t | x, g)]).
```

Thus the continuous state and shared gauge are not duplicated or assumed
independent. The discrete identity variable is integrated exactly conditional
on them.

## Why neither existing extreme is sufficient

Independent soft mixtures preserve ambiguity but assign nonzero probability to
impossible collisions. A hard one-to-one assignment avoids collisions but
throws away uncertainty and can make an arbitrary choice when alternatives are
close. The joint operator preserves both requirements:

- every linked source is used at most once;
- every target retains its exact null fallback;
- all valid assignments remain weighted and marginalized;
- posterior per-target identity probabilities and a deterministic global MAP
  assignment are reported separately.

For two targets that each assign equal local weight to `null` and the same one
source, independent mixtures allocate `1/4` probability to using that source
twice. The joint model has exactly three valid assignments—`null/null`,
`link/null`, and `null/link`—with probability `1/3` each. It therefore preserves
symmetry rather than selecting one target arbitrarily.

## Exact sparse algorithm

Targets and linked source endpoints form a bipartite candidate graph. Targets
that share no source candidates belong to independent connected components. For
each component, Prob4D performs an exact forward-backward dynamic program whose
state is the bit mask of already-used sources. It computes:

- the prior and posterior matching log-partition functions;
- exact per-target candidate marginals;
- a deterministic global MAP matching and its probability;
- joint assignment entropy and effective assignment count; and
- the probability mass that the corresponding independent local model assigns
  to source collisions.

For a component with `T_k` targets, `S_k` linked sources, and at most `M_k`
candidates per target, the worst-case cost is
`O(T_k * M_k * 2**S_k)`. The public function defaults to at most 16 linked
sources per component and fails closed when that bound is exceeded. Source-side
gating should keep ambiguous components small; callers may raise the explicit
limit after reviewing the computational consequence.

```python
from prob4d.joint_material_identity import (
    marginalize_joint_identity_log_likelihoods,
)

result = marginalize_joint_identity_log_likelihoods(
    mixtures,                       # canonical tuple, one mixture per target
    tuple(m.candidate_ids for m in mixtures),
    candidate_log_likelihoods,      # tuple of target-local vectors
)

print(result.log_marginal_likelihood)
print(result.posterior_candidate_probabilities)
print(result.posterior_independent_collision_probability)
```

The function requires all mixtures to share their causal window order,
calibration, association rule, source revisions, and weight/null semantics. It
also requires canonical target ordering and exact candidate-ID alignment.

## Controlled mechanism study

The checked result in
`evidence/joint-material-identity-controlled-v1/summary.json` uses three target
tracks, two source tracks, and 13 valid partial matchings. Observations are drawn
from the one-to-one model with Gaussian candidate likelihoods. Each regime uses
32,768 trials, 128 groups of 256 trials, and 5,000 paired group-bootstrap
resamples. The production operator is independently audited against direct
assignment enumeration on eight trials per regime; the maximum absolute error
is `8.88e-16`.

| Regime | Independent prior collision mass | Joint NLL | Independent-soft NLL | Independent − joint, 95% CI | Hard-MAP NLL | Joint Brier | Independent Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Separated | 0.1858 | 3.4933 | 3.5461 | 0.0528 [0.0503, 0.0553] | 4.0822 | 0.02477 | 0.02924 |
| Moderate | 0.6134 | 4.1409 | 4.4526 | 0.3117 [0.3056, 0.3179] | 6.5214 | 0.07445 | 0.10163 |
| Ambiguous | 0.7279 | 4.3576 | 4.7844 | 0.4269 [0.4194, 0.4344] | 8.4320 | 0.09461 | 0.13305 |

The controlled result supports the proposed mechanism: the cost of independent
soft matching grows with its impossible collision mass, while hard matching
loses sharply when the posterior is genuinely multimodal. Because the data are
generated from the one-to-one model, this is a mechanism and implementation
check, not evidence of real-provider or physical-prediction benefit.

Rebuild and verify the retained result with:

```bash
python -m prob4d.joint_identity_controlled_study build \
  --output evidence/joint-material-identity-controlled-v1/summary.json \
  --overwrite
python -m prob4d.joint_identity_controlled_study verify \
  evidence/joint-material-identity-controlled-v1/summary.json
```

## Paper contribution and required real gate

The defensible paper contribution is not “we invented assignment
marginalization.” Probabilistic data association is established. The contribution
is the combination needed for recursive long-sequence 4D reconstruction:

1. calibrated local cross-window identity hypotheses with an explicit null;
2. exact one-to-one joint identity marginalization;
3. conditional compatibility with Prob4D's shared joint `Sim(3)` gauge
   uncertainty instead of independent-window covariance; and
4. propagation of the accepted joint belief into BayesianPhysTwin, with
   Causal4D remaining downstream of the physical-state acceptance boundary.

This directly contrasts with long-sequence 4D pipelines that align overlapping
chunks and then make a hard one-to-one tracklet assignment, such as LongDPM
(arXiv:2605.17303). The method should be presented as **uncertainty-preserving
cross-window identity fusion**, not as a new Hungarian algorithm or a universal
multi-object tracker.

A physical-benefit claim still requires one frozen prospective experiment on a
support-feasible real provider and an unseen physical object/session cohort. The
same BayesianPhysTwin guard and exact fallback must compare:

1. newest-window/null-only state;
2. independent local soft mixtures;
3. hard one-to-one MAP matching; and
4. joint one-to-one marginalization.

Primary inference should be at the object/session level and report physical
forecast error, proper predictive score, calibration, acceptance rate, and
harmful accepted updates. Only the accepted BayesianPhysTwin belief and its
lineage may be passed to Causal4D.

## Claim boundary

This operator establishes exact source-calibrated marginalization over valid
one-to-one partial matchings conditional on caller-supplied factorized candidate
likelihoods. It does not establish provider competence, a physical-state update,
Causal4D benefit, deployment safety, or state of the art.
