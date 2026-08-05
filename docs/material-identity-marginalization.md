# Material-identity marginalization

`prob4d.material_identity_mixture` is an experimental boundary for carrying
uncertainty about cross-window material identity without rewriting any Prob4D
observation ID.

## Motivation

Within-window causal tracklets provide persistent local identities. Cross-window
association can propose links between overlapping windows, but one hard link
hides ambiguity. A wider covariance must also not make an association rank better
merely because it is less informative.

The mixture contract therefore keeps a small set of local hypotheses for one
target track:

- exactly one **null** hypothesis, representing the unchanged newest-window
  reference;
- zero or more linked source endpoints, each stored as its original
  `(window_id, track_id)` pair;
- an ordered admitted-window sequence with the target window last;
- the source association result identity and source score;
- a source-calibrated log weight;
- the association-rule, calibration, producer, and implementation identities.

Every linked source endpoint must occur before the target in the bound window
order. No global point ID, connected component, or provider-v2 observation
rewrite is created.

## Exact downstream marginalization

A downstream consumer can evaluate its own log likelihood `ell_a` for every
identity hypothesis `a` and compute

```text
log p(y | x, g) = logsumexp_a(log q_a + ell_a),
```

where `q_a` is the normalized source-calibrated identity weight and `g` can be
the existing complete joint `Sim(3)` gauge nuisance. Candidate IDs must match the
mixture order exactly, preventing likelihood rows from being silently assigned
to the wrong endpoint.

`marginalize_identity_log_likelihoods` returns the marginal log likelihood and
the posterior identity probabilities. A likelihood power of zero returns the
source prior without evaluating impossible `-inf` likelihood rows.

For consumers that need one Gaussian approximation,
`moment_match_gaussian_identity_hypotheses` applies the law of total covariance:

```text
mean = sum_a q_a mean_a
cov  = sum_a q_a cov_a
     + sum_a q_a (mean_a - mean)(mean_a - mean)^T.
```

The second term is retained explicitly as between-hypothesis uncertainty.

## Fallback behavior

A mixture containing only the null hypothesis returns the null likelihood, mean,
and covariance exactly. The contract always requires exactly one null
hypothesis, so rejected or absent cross-window links preserve the newest-window
reference rather than dropping the case.

## Information and claim boundary

The log weights must be calibrated only on declared source/calibration objects or
sessions. The artifact does not decide whether BayesianPhysTwin accepts an
update, does not establish provider competence, and does not establish a
Causal4D intervention benefit. Promotion requires a separately frozen
object/session-held-out comparison of newest-window, hard-link,
identity-marginalized, oracle, and exact-fallback arms.
