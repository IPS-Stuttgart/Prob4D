# CUT3R recurrent-state recovery

`prob4d.cut3r_recurrent_state_recovery` adds a source-only, report-only analysis to
the frozen CUT3R comparison. It answers a specific mechanism question:

> How much of the error caused by restarting CUT3R for every causal window is
> recovered by Prob4D fusion of those same restarted windows?

The analysis does not introduce a new estimator, select a candidate, change a
readiness gate, or authorize target access.

## Three frozen arms

The report combines the two claim-eligible contrasts already registered by the
CUT3R comparison lock:

| Role | Arm |
| --- | --- |
| Uninterrupted recurrent reference | `native-continuous` |
| Restarted-window baseline | `restarted-newest` |
| Prob4D treatment | `restarted-prob4d-fused` |

For every lower-is-better metric \(E\), the descriptive recovery fraction is

\[
\operatorname{Recovery}(E)=
\frac{E_{\mathrm{restarted\mbox{-}newest}}-
      E_{\mathrm{restarted\mbox{-}Prob4D}}}
     {E_{\mathrm{restarted\mbox{-}newest}}-
      E_{\mathrm{native\mbox{-}continuous}}}.
\]

Interpretation:

- `0`: Prob4D recovers none of the restart penalty;
- `1`: Prob4D closes the entire gap to uninterrupted recurrence;
- between `0` and `1`: partial recovery;
- above `1`: fused restarted windows outperform uninterrupted recurrence;
- below `0`: fusion is harmful; and
- `undefined-native-not-better`: uninterrupted recurrence does not provide a
  positive denominator beyond the frozen metric-specific tolerance.

The report retains the numerator and denominator. It never clips the fraction.

## Evidence prerequisites

The command consumes two complete
`prob4d.cut3r-source-competence-report-v2` evidence chains:

1. `restarted-prob4d-fused` versus `restarted-newest`; and
2. `native-continuous` versus `restarted-newest`.

Before computing any statistic, it verifies all bound locks, records, and reports
through their existing strict loaders. It then requires:

- the canonical registered contrast in each source-competence lock;
- identical cohort binding, group definition, and restarted-newest provider
  manifest identity;
- identical technical-failure evidence;
- byte-identical normalized `restarted-newest` rows across both contrasts; and
- equal restarted-newest aggregate metrics in every evaluable group.

Each v2 contrast already proves exact candidate/baseline metric-support identity.
The byte-identical common baseline therefore establishes transitive common support
for all three arms on every evaluable group. A changed support hash, row, score,
reference count, or baseline value fails closed.

## Reported metrics

The source report covers:

- point RMSE;
- endpoint RMSE;
- arm-neutral fixed-scale proper score;
- seam RMSE; and
- absolute drift slope.

It reports the three arm values, Prob4D gain, recurrent-state gap, recovery
fraction, and status for every complete object/session group and for the
equal-group aggregate.

## Uncertainty interval

Aggregate intervals use deterministic paired bootstrap resampling of complete
source groups. Frames, seeds, cameras, points, and tracks remain nested
observations and are never resampled as independent evidence units.

The implementation uses `sha256-counter-equal-group-bootstrap-v1`, so the same
bound evidence and analysis specification reproduce identical resamples without
relying on process-global random state. Replicates whose recurrence denominator
does not exceed the metric-specific frozen minimum gap are retained as invalid
denominator counts rather than silently divided. The interval is withheld when
the valid-replicate fraction is below the frozen minimum.

## Frozen analysis specification

A prospective source-only specification is provided at
`protocols/cut3r_recurrent_state_recovery_v1.json`. It freezes:

- bootstrap seed;
- bootstrap replicate count;
- confidence level;
- a dimensionally appropriate minimum recurrence gap for every metric; and
- the minimum fraction of valid bootstrap denominators.

The specification is embedded in and authenticated by the report identity.

## Build, verify, and summarize

After both common-support v2 reports exist:

```bash
prob4d prediction cut3r-recovery build \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/fusion/source-competence-lock.json \
  outputs/cut3r/fusion/common-support-lock-v2.json \
  outputs/cut3r/fusion/source-competence-records-v2.json \
  outputs/cut3r/fusion/source-competence-v2.json \
  outputs/cut3r/recurrence/source-competence-lock.json \
  outputs/cut3r/recurrence/common-support-lock-v2.json \
  outputs/cut3r/recurrence/source-competence-records-v2.json \
  outputs/cut3r/recurrence/source-competence-v2.json \
  protocols/cut3r_recurrent_state_recovery_v1.json \
  --output outputs/cut3r/recurrent-state-recovery-v1.json
```

Independently rebuild the report from all bound evidence:

```bash
prob4d prediction cut3r-recovery verify \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/fusion/source-competence-lock.json \
  outputs/cut3r/fusion/common-support-lock-v2.json \
  outputs/cut3r/fusion/source-competence-records-v2.json \
  outputs/cut3r/fusion/source-competence-v2.json \
  outputs/cut3r/recurrence/source-competence-lock.json \
  outputs/cut3r/recurrence/common-support-lock-v2.json \
  outputs/cut3r/recurrence/source-competence-records-v2.json \
  outputs/cut3r/recurrence/source-competence-v2.json \
  protocols/cut3r_recurrent_state_recovery_v1.json \
  outputs/cut3r/recurrent-state-recovery-v1.json
```

Use `summarize` with the same arguments and `--json` for a compact machine-readable
view.

## Claim boundary

This report is descriptive source mechanism evidence. A positive recovery
fraction does not establish held-out provider competence, BayesianPhysTwin
physical-query benefit, Causal4D intervention benefit, deployment safety, or
state of the art. A negative or undefined result remains informative because it
separates irrecoverable recurrent information from useful restarted-window
fusion without changing the frozen target protocol.
