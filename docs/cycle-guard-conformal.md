# Finite-sample calibration of the normalized cycle guard

The uncertainty-normalized source-cycle guard introduced in the preceding study
retained perfect injected-edge detection and cut the worst clean false-fallback
rate by more than half, but it missed the preregistered absolute `0.10` ceiling.
That target result is retained unchanged. This follow-up evaluates a new
calibration rule on disjoint calibration and target seeds.

## Candidate

Let `s_1, ..., s_n` be maximum source-only uncertainty-normalized cycle scores
from clean calibration trials. For requested miscoverage `alpha`, the candidate
uses the split-conformal upper order statistic

```text
k = ceil((n + 1) * (1 - alpha))
q = k-th smallest calibration score.
```

A future clean source score is admitted when it is at most `q`. Under
exchangeability between calibration and future clean source scores,

```text
P(s_future > q) <= (n + 1 - k) / (n + 1) <= alpha.
```

This is a marginal finite-sample statement. It is not a conditional guarantee for
an unknown source-noise subgroup, and the normalized score is not interpreted as
a chi-square statistic. The implementation fails closed when `alpha` is below the
finite resolution `1 / (n + 1)`.

The calibration object records the requested miscoverage, calibration count,
one-based order-statistic rank, realized bound, threshold, and a row-order-
invariant SHA-256 digest of the canonical score multiset.

## Frozen follow-up protocol

`protocols/cycle-guard-conformal-v1.json` uses:

- 96 balanced clean calibration trials;
- 128 target trials in each of six regimes, 768 target trials total;
- calibration seed `408260804` and target seed `731260804`;
- requested conformal miscoverage `0.05`;
- the unchanged source-only normalized score and exact tree fallback;
- the production tree, unguarded graph, raw guard, empirical normalized guard,
  and conformal normalized guard;
- 2,000 paired target-trial bootstrap resamples.

The seeds are disjoint from the preceding normalized-guard study. No observation
from that earlier target split is used to fit or score the new candidate.

Run the frozen study with:

```bash
prob4d diagnostic cycle-guard-conformal \
  --output-dir outputs/cycle-guard-conformal \
  --calibration-trials 96 \
  --target-trials-per-scenario 128 \
  --calibration-seed 408260804 \
  --target-seed 731260804 \
  --conformal-miscoverage 0.05 \
  --bootstrap-resamples 2000
```

The command returns `0` only when every registered criterion passes and `3` for a
valid negative result. It writes JSON, aggregate CSV, raw-trial CSV, Markdown,
and complete SHA-256 checksums before returning either decision code.

## Promotion boundary

Even a passing controlled study does not promote the graph or cycle guard to the
claim-bearing provider. Production remains the single-parent joint spanning tree.
Promotion still requires held-out physical-object/session evidence and the
separately sealed BayesianPhysTwin harmful-update and fallback gates.
