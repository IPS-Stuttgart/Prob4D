# Finite-sample capability preflight

The held-out provider gate uses complete physical objects or independent
acquisition sessions as statistical units. Before opening provider payloads or
target outcomes, the finite-sample preflight converts the frozen group counts
into explicit split-conformal order-statistic ranks and small-target diagnostic
resolution.

The command is target-free:

```bash
prob4d-finite-sample-preflight \
  promotion-lock.json \
  --cohort-binding deform360-cohort-binding.json \
  --coverage 0.90 \
  --coverage 0.95 \
  --output finite-sample-capability.json \
  --markdown finite-sample-capability.md
```

The optional cohort binding must identify exactly the same calibration and target
groups as the promotion lock. For the official-Hub Deform360 binding, the command
also reports the sheet and volumetric calibration strata separately.

## Split-conformal rank calculation

For `n` independent calibration groups and requested coverage `1 - alpha`, the
one-sided split-conformal order-statistic rank is

```text
k = ceil((n + 1) * (1 - alpha)).
```

A finite threshold exists only when `k <= n`. The largest finite nominal level is
therefore `n / (n + 1)`. The report retains, for every requested level and
population:

- the exact order-statistic rank;
- whether the threshold is finite;
- the associated finite-sample coverage lower bound when finite;
- the maximum finite coverage supported by the available groups; and
- the minimum number of calibration groups required for that level.

For the frozen Deform360 design this implies:

| Calibration population | Groups | Maximum finite level | 90% | 95% |
| --- | ---: | ---: | :---: | :---: |
| All calibration objects | 10 | 10/11 = 90.9% | finite | unavailable |
| Sheet stratum | 5 | 5/6 = 83.3% | unavailable | unavailable |
| Volumetric stratum | 5 | 5/6 = 83.3% | unavailable | unavailable |

Consequently, an overall finite 90% threshold is possible, while separate
nominal-90% guarantees within either five-object stratum are not. Strata remain
important for worst-group and calibration-shift diagnostics, but they must not be
reported as independent nominal-90% calibration claims.

## Target-group resolution

The report also records quantities implied by the frozen target design:

- the empirical probability mass represented by one bootstrap resample;
- the number of leave-one-group-out sensitivity replications; and
- the exact sign probability of observing all paired effects in the favorable
  direction under a symmetric null.

These diagnostics complement, rather than replace, the registered paired
object/session bootstrap. The complete object-level effect vector, worst-object
regression, and leave-one-object-out sensitivity should remain visible in the
final report.

## Fail-closed use

Use `--require-primary-finite` to return exit code 3 when any requested coverage
level is unavailable for the complete calibration cohort. Use
`--require-strata-finite` to require every declared stratum as well. The JSON and
Markdown evidence are still written for a valid negative preflight so that an
unsupported claim cannot disappear from the record.

The report is content-addressed, rejects duplicate keys, non-finite JSON,
coercive numeric aliases, changed derived ranks, inconsistent strata, cohort-lock
mismatch, and output replacement.

## Claim boundary

A finite rank is a capability statement under the declared independent-group and
exchangeability assumptions. It does not prove that those assumptions hold, that
the provider is competent, that target coverage is calibrated, or that a guarded
BayesianPhysTwin update improves the physical query. Provider competence and
downstream benefit remain separate conjunctive held-out gates.
