# Held-out study preflight

`prob4d study preflight` is a high-level, target-free design command for one
sealed held-out provider study. It composes two existing scientific boundaries
without adding a target-side method or changing the frozen decision:

1. the finite-sample capability report derived from calibration and target
   object/session counts; and
2. a source-bound sensitivity report derived from declared paired-difference
   standard deviations.

The command reads the promotion lock and optional cohort binding. It does not
read provider payloads, target predictions, physical-query outcomes, or target
metrics.

## Example

```bash
prob4d study preflight promotion-lock.json \
  --cohort-binding deform360-cohort-binding.json \
  --source-summary-id <source-summary-sha256> \
  --source-metric deployed_minus_physical_rmse_mm \
  --paired-sd source-estimate=0.85 \
  --paired-sd conservative=1.20 \
  --coverage 0.90 \
  --coverage 0.95 \
  --power 0.80 \
  --power 0.90 \
  --confidence 0.95 \
  --accepted-groups 6 \
  --accepted-groups 12 \
  --output-dir outputs/study-preflight
```

The output directory contains:

```text
finite_sample_capability.json
finite_sample_capability.md
study_sensitivity.json
study_sensitivity.md
```

Every destination is checked before the first file is written. Existing
artifacts are never replaced.

## Paired-effect sensitivity

For target-group count `n`, source-declared paired standard deviation `s`, power
`1-beta`, and confidence `1-alpha`, the reported normal-approximation minimum
detectable effect is

```text
MDE = (z_critical + z_(1-beta)) * s / sqrt(n).
```

For a two-sided interval, `z_critical = z_(1-alpha/2)`; for a one-sided interval,
`z_critical = z_(1-alpha)`. The report also records the corresponding confidence
interval half-width and standardized effect size.

The source standard deviations are scenarios, not estimates silently inferred
from the target. Each scenario has a stable name and is bound to an exact
`source_summary_id`. A scenario may be a source estimate, an upper confidence
bound, or another preregistered conservative value. The report does not choose
between scenarios after target access.

When the promotion lock declares a positive query-superiority margin, each row
states whether an effect exactly equal to that margin is detectable at the
requested power under the scenario. A zero margin is reported as not applicable;
statistical sensitivity cannot certify an effect of exactly zero.

## Harmful-update resolution

The accepted-group count is unknown before the target run, so the command accepts
one or more explicit denominator scenarios through `--accepted-groups`. The
default scenario is all frozen target groups.

For each denominator, the report contains:

- the rate contribution of one harmful accepted update;
- the frozen maximum harmful-update count;
- an exact one-sided Clopper--Pearson upper rate bound when zero harmful updates
  are observed; and
- the corresponding upper rate bound at the frozen allowed count.

These values describe what the group count can resolve. They do not replace the
frozen count-based decision, predict the accepted-group count, or authorize a
rate claim on unopened data.

## Python façade

The same target-free composition is available without manually selecting the
low-level builders:

```python
from prob4d.study import HeldoutProviderStudy
from prob4d.study_sensitivity import PairedDifferenceScenarioV1

study = HeldoutProviderStudy.from_lock("promotion-lock.json")
preflight = study.preflight(
    source_summary_id="<source-summary-sha256>",
    source_metric="deployed_minus_physical_rmse_mm",
    paired_difference_scenarios=(
        PairedDifferenceScenarioV1("source-estimate", 0.85),
        PairedDifferenceScenarioV1("conservative", 1.20),
    ),
)
```

The façade is intentionally read-only and target-free. Claim-bearing source
qualification, one-shot target evaluation, and independent verification remain
owned by the existing `fresh-provider-readiness`, `evaluate provider`, and
`experiment heldout-provider` contracts.

## Interpretation boundary

The sensitivity calculation is a transparent normal approximation. It does not
establish that paired target differences are normal, independent, or
exchangeable. It must not change the promotion lock, the target roster, the
bootstrap decision, the fallback rule, or the target-side analysis.

A result that is not decisive with a small frozen target cohort must be reported
as inconclusive at the declared resolution, not rewritten as evidence of no
effect. Conversely, an apparently favorable target result does not rescue a
failed support, mean, identity, gauge/dependence, covariance, or query-relevance
gate.
