# CUT3R recurrent-state recovery v2

`prob4d.cut3r_recurrent_state_recovery_v2` is an additive, source-only analysis
of the same three frozen CUT3R arms used by the v1 report:

| Role | Arm |
| --- | --- |
| Uninterrupted recurrent reference | `native-continuous` |
| Restarted-window baseline | `restarted-newest` |
| Prob4D treatment | `restarted-prob4d-fused` |

It does not alter the provider, windowing, source roster, source-competence
decisions, readiness order, target protocol, BayesianPhysTwin guard, or Causal4D
handoff. Version 1 remains readable and reproducible. Version 2 exists because a
ratio can become unstable even when every input artifact is valid.

## Scientific question

For each lower-is-better source metric, define the absolute Prob4D gain

\[
G = E_{\mathrm{restarted\mbox{-}newest}}
    - E_{\mathrm{restarted\mbox{-}Prob4D}}
\]

and the recurrent-state gap

\[
D = E_{\mathrm{restarted\mbox{-}newest}}
    - E_{\mathrm{native\mbox{-}continuous}}.
\]

The recovery fraction is

\[
R = \frac{G}{D}.
\]

Version 2 makes `G` the primary endpoint. `R` is secondary and is defined only
when `D` exceeds the prospectively frozen practical-separation floor for that
metric. This prevents an arbitrarily small positive denominator from producing
an arbitrarily large but scientifically uninformative recovery fraction.

The fraction is never clipped:

- `R < 0` means Prob4D fusion is harmful relative to the restarted baseline;
- `0 < R < 1` means partial recovery;
- `R = 1` means the complete restart gap is closed;
- `R > 1` means fused restarted windows outperform uninterrupted recurrence; and
- `undefined-recurrence-gap-not-practically-separated` means the recurrent-state
  gap is not large enough to support the ratio interpretation.

Absolute gain, recurrent-state gap, and their intervals remain available even
when the ratio is undefined.

## Prospective specification

The checked-in specification is
[`protocols/cut3r_recurrent_state_recovery_v2.json`][v2-spec].
It is content-addressed and freezes:

- absolute Prob4D gain as the primary endpoint;
- the recovery fraction as a denominator-gated secondary endpoint;
- a 95% exact empirical-bootstrap interval;
- a minimum valid-denominator probability of `0.8`;
- exact enumeration for at most ten evaluable complete source groups;
- strictly positive metric-specific recurrence-gap floors;
- leave-one-complete-group-out sensitivity reporting;
- `source_outcomes_opened_before_specification=false`; and
- `target_access=forbidden`.

The specification is additive analysis hardening frozen while the retained
source-input job was still queued. It does not replace or mutate that exact
execution request.

The practical-separation floors are:

| Metric | Frozen floor |
| --- | ---: |
| Point RMSE | `0.0001 m` (`0.1 mm`) |
| Endpoint RMSE | `0.0001 m` (`0.1 mm`) |
| Seam RMSE | `0.0001 m` (`0.1 mm`) |
| Absolute drift slope | `0.00001 m/frame` (`0.01 mm/frame`) |
| Fixed-scale proper score | `0.01` |

These values are prospective interpretation floors, not accuracy, calibration,
or sensor-resolution claims. Changing one after source outcomes are opened
requires a new analysis version; the v2 artifact must not be rewritten.

## Exact small-sample inference

The v1 analysis uses a deterministic Monte Carlo group bootstrap. Version 2
instead enumerates the exact equal-group empirical-bootstrap distribution.

For `n` complete source groups, an ordered empirical-bootstrap sample contains
`n` draws with replacement. Version 2 enumerates every multinomial count vector
and weights it by its exact ordered-sample multiplicity

\[
\frac{n!}{\prod_i c_i!}.
\]

The multiplicities sum to `n**n`, which is checked at runtime. The number of
count vectors is also checked against

\[
\binom{2n-1}{n-1}.
\]

For ten evaluable groups, this evaluates 92,378 count vectors representing all
10,000,000,000 ordered resamples without pretending that 10,000 pseudorandom
replicates provide 10,000 independent pieces of evidence.

The report includes exact weighted nearest-rank intervals for:

1. absolute Prob4D gain;
2. the recurrent-state gap; and
3. the recovery fraction, conditional on the recurrence gap exceeding its
   frozen practical floor.

It also records the exact ordered-resample count and probability for valid and
invalid denominators. The recovery-fraction interval is withheld when that valid
probability is below the frozen minimum, while the gain and denominator
intervals remain reported.

Frames, points, cameras, seeds, and tracks remain nested observations. Only
complete source objects or acquisition sessions receive independent bootstrap
mass.

## Leave-one-group-out sensitivity

For every metric, the report recomputes the equal-group aggregate after omitting
each complete evaluable source group. It retains:

- the complete omission table;
- minimum and maximum absolute Prob4D gain;
- whether the gain changes sign;
- whether denominator eligibility changes; and
- the range of defined recovery fractions.

This is a sensitivity diagnostic, not an alternative selection rule. A sign
reversal or denominator-status change is retained rather than used to choose a
more favorable subset.

## Evidence validation

Version 2 consumes the same two complete
`prob4d.cut3r-source-competence-report-v2` chains as version 1:

1. `restarted-prob4d-fused` versus `restarted-newest`; and
2. `native-continuous` versus `restarted-newest`.

Before any v2 statistic is built, the existing strict v1 builder revalidates the
comparison lock, both source-competence locks, both common-support locks, all
canonical records, both reports, byte-identical restarted-window baseline rows,
technical-failure evidence, cohort binding, and provider identities. The v2
report binds the resulting validation-bridge report identity and then performs
its own exact analysis.

## Build, verify, and summarize

After both common-support source reports exist:

```bash
prob4d prediction cut3r-recovery-v2 build \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/fusion/source-competence-lock.json \
  outputs/cut3r/fusion/common-support-lock-v2.json \
  outputs/cut3r/fusion/source-competence-records-v2.json \
  outputs/cut3r/fusion/source-competence-v2.json \
  outputs/cut3r/recurrence/source-competence-lock.json \
  outputs/cut3r/recurrence/common-support-lock-v2.json \
  outputs/cut3r/recurrence/source-competence-records-v2.json \
  outputs/cut3r/recurrence/source-competence-v2.json \
  protocols/cut3r_recurrent_state_recovery_v2.json \
  --output outputs/cut3r/recurrent-state-recovery-v2.json
```

Rebuild every statistic and verify the retained bytes:

```bash
prob4d prediction cut3r-recovery-v2 verify \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/fusion/source-competence-lock.json \
  outputs/cut3r/fusion/common-support-lock-v2.json \
  outputs/cut3r/fusion/source-competence-records-v2.json \
  outputs/cut3r/fusion/source-competence-v2.json \
  outputs/cut3r/recurrence/source-competence-lock.json \
  outputs/cut3r/recurrence/common-support-lock-v2.json \
  outputs/cut3r/recurrence/source-competence-records-v2.json \
  outputs/cut3r/recurrence/source-competence-v2.json \
  protocols/cut3r_recurrent_state_recovery_v2.json \
  outputs/cut3r/recurrent-state-recovery-v2.json
```

Use `summarize` with the same bound inputs and `--json` for a compact view that
foregrounds absolute gain, its exact interval, denominator probability, the
secondary recovery fraction, and leave-one-group-out sensitivity.

## Claim boundary

This is descriptive source mechanism evidence. It does not select a provider,
repair a negative source gate, authorize target access, establish physical-query
benefit in BayesianPhysTwin, establish counterfactual benefit in Causal4D,
establish deployment safety, or establish state of the art. A negative,
undefined, denominator-unstable, or leave-one-group-sensitive result remains a
complete result and must not be retuned on the same opened source groups.

[v2-spec]: ../protocols/cut3r_recurrent_state_recovery_v2.json
