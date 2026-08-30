# Complete-information audit for query-conditioned updates

Status: **experimental kernel and completed controlled mechanism study**. This
advances the analytic/controlled parts of issue #339. It does not close the real
provider qualification or downstream physical-value stages of #333/#49.

## Scientific question

A partially observable reconstruction can constrain a particular query without
identifying every latent coordinate. However, a useful query covariance does not
authorize an arbitrary complete-belief update. A proposal may invent precision,
count the same observation repeatedly, or move an unsupported mean while keeping
its covariance entirely correct.

The existing `query_observability` module characterizes information supplied by a
factor. The new `query_information_audit` module instead checks a proposed
posterior against a **separately retained expected likelihood**. It is an
integration-boundary audit, not a new Gaussian estimator or calibration method.

## Both Gaussian natural parameters are necessary

Let the complete prior be Gaussian with mean `m0` and positive-definite covariance
`P0`. Let the independently retained likelihood have information matrix `Lambda`
and natural parameter `eta`. A candidate `(mc, Pc)` must satisfy both identities:

```text
Pc^{-1} - P0^{-1} = Lambda
Pc^{-1} mc - P0^{-1} m0 = eta.
```

Testing only the first identity misses an unjustified mean shift. Testing only
whether information lies in the observable subspace misses double counting
within that subspace.

For `P0 = L L.T`, the implementation compares

```text
L.T Pc^{-1} L - I       with L.T Lambda L
L.T Pc^{-1} (mc - m0)   with L.T (eta - Lambda m0).
```

The centering avoids dependence on the coordinate origin; prior whitening makes
these residual norms invariant to invertible linear reparameterizations, up to
numerical error. The direct/nullspace sensitivity fraction, in contrast, is
reported in the declared local chart and is not claimed to be invariant under
arbitrary changes of its metric.

These are classical Gaussian identities. Neither Gaussian conditioning nor
preserving unobservable directions is claimed as a new theorem. Relevant
observability/registration precedents include Huang, Mourikis and Roumeliotis,
*Observability-based Rules for Designing Consistent EKF SLAM Estimators* (2010,
DOI 10.1177/0278364909353640); Tuna et al., *X-ICP* (arXiv:2211.16335); and
Hatleskog and Alexis, *Probabilistic Degeneracy Detection for Point-to-Plane Error
Minimization* (arXiv:2410.10784). The prospective application contribution is the
chain from dependent learned 4D predictions to justified downstream queries.

## Query decision and its limits

After the audit, `assess_query_update` checks the largest eigenvalue of the
standardized query covariance, not just its diagonal, against one. The caller
supplies the query Jacobian and positive one-standard-deviation tolerance scales.
A minimum variance-reduction condition prevents admitting a no-information update.

The four routes distinguish directly supported queries, queries bounded partly
by the complete prior, unresolved queries, and invalid complete updates. These
are metadata only: BayesianPhysTwin remains responsible for complete-belief
selection and caller-owned fallback. No existing BPT or provider-v2 route is
silently replaced.

The expected likelihood must not be reverse-engineered from the proposal being
checked. A mutually consistent but wrong factor passes, as it should for an
algebraic audit. A wrong prior, nonlinear approximation error, unmodelled visual
bias, a wrong query Jacobian, or a shared provenance error is not certified away.
One-standard-deviation tolerances are not frequentist coverage guarantees. An
admitted update can still increase realized error.

## Registered study

The numerical settings and source hashes were committed at
`eb846570085cb33ceb7cfcfc9ae7cbcac01fadbe` before the first simulation.

The experiment uses 48 configurations, 128 independent paired seeds, three point
queries, four proposal conditions, and matched/shifted-prior worlds. DLO-like
point geometry defines a local Sim(3) design. Gaussian sufficient statistics are
projected onto the retained geometry subspace. This is **not** a nonlinear
alignment benchmark, a learned-provider run, a physical rollout, or a real DLO
experiment. Known exchangeable window dependence is part of the generator.

Eight comparators are retained: physical fallback; rejection of every deficient
factor; point-only transform; pseudoinverse covariance; fabricated full-rank
completion; exact observable-subspace posterior; query precision without the
information audit; and the audited query policy. Correct ungated Gaussian
inference is the reference, not a straw-man that selective fallback should beat.

All intervals use 10,000 paired bootstrap resamples of complete seeds. The 48
configurations and three queries are nested, not independent statistical units.
Intervals are descriptive rather than multiplicity-corrected significance claims.
Point-only uncertainty scores remain undefined, without an invented noise floor.

## Results and costs

For clean factors under the matched prior, the audited policy admits 83.33% of
rank-deficient queries: all 4,096 on-axis cases, all 4,096 near-axis cases, and
2,048 of 4,096 off-axis cases. Blanket rank rejection admits none of them.
These counts are repeated decisions within 128 independent seeds, not thousands
of independent real objects.

Across all clean configurations, point-query RMSE is 20.348 mm for physical
fallback, 16.698 mm for blanket rank rejection, 8.119 mm for the audited policy,
and **3.688 mm for the correct ungated posterior**. The audited policy's nominal
90% joint coverage is 89.752% [89.166%, 90.343%]. Its mean marginal 90% interval
width is 7.338 mm, versus 3.659 mm for the exact posterior. Thus abstention has a
substantial, visible precision and accuracy cost.

The matched-prior corruption panels compare the same candidate with and without
the audit:

| Proposal condition | RMSE without / with audit (mm) | 90% coverage without / with | Acceptance without / with |
|---|---:|---:|---:|
| Exact | 8.119 / 8.119 | 89.752% / 89.752% | 88.194% / 88.194% |
| Fabricated nullspace precision | 5.296 / 16.787 | 66.558% / 90.137% | 99.306% / 32.639% |
| Unsupported mean, correct covariance | 8.463 / 16.787 | 80.914% / 90.137% | 88.194% / 32.639% |
| Repeated information | 7.940 / 15.332 | 69.721% / 89.209% | 88.889% / 44.444% |

Among actually corrupted cases, the query-only policy admits 12,288/12,288
fabricated-precision cases, 10,240/12,288 unsupported-mean cases, and 8,192/9,216
double-counted cases. The audit admits zero in each group. Full-rank cases have
no injected nullspace corruption; independent-window cases have no repetition
corruption. They remain in the complete panels rather than being hidden.

**Coverage improvement is not free performance improvement.** Full fallback
also discards useful observed information. NLL becomes worse in the mean-shift
and repeated-information panels: -16.528 to -13.208 and -16.482 to -14.237,
respectively (lower is better). This rules out a claim that the audit uniformly
improves proper scores or point accuracy.

The prespecified wrong-prior stress is equally important. Even with an exact,
audit-valid proposal, coverage falls to **75.982%** [75.336%, 76.611%]. An
unsupported mean change can even accidentally compensate for that prior error.
The audit establishes fidelity to the declared inputs, not truth or a no-harm
property. This negative result is retained without retuning.

## Reproduction and independent verification

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/science/query_information_audit_study.py \
  --protocol protocols/query-information-audit-study-v1.json \
  --output /tmp/prob4d-query-information-study
python scripts/science/verify_query_information_audit_study.py \
  /tmp/prob4d-query-information-study
```

The study refuses to overwrite an output directory. It retains raw sufficient
statistics, complete candidate means/covariances, decisions, seed-level metrics,
and protocol/code/artifact hashes. The reference execution used Python 3.13.5
and NumPy 2.3.5. Bitwise equality across numerical platforms is not promised.

The independent verifier does not import the estimator or study scorer. It uses
measurement-space Kalman conditioning and independently reconstructs every arm's
scores and each admission decision. It verified 147,456 decisions and 49,152
seed-metric cells, including explicitly missing point-only uncertainty metrics.
Maximum mean disagreement was 4.24e-14 in local coordinates; maximum covariance
disagreement was 9.16e-18; maximum score disagreement was 1.14e-12.

Local tests passed 17 cases, with the full-repository integration case skipped
in the partial offline workspace. The full repository's Python 3.10--3.14 suites,
runtime-floor tests, provider contract, builds, and installed-artifact smoke tests
passed at core commit `92dfa41f12ea1ddbd6f2919cada327634027e876` (run
`33279488851`). Later documentation/verifier changes require their own head checks.

## What would make the paper materially stronger

The next material evidence is one source-qualified real provider followed by a
fresh, grouped downstream-query evaluation, not another collection of synthetic
wins. Preserve the partial factor and construct the valid candidate correctly;
use the audit as a boundary check rather than advertise it as a superior
estimator. Measure the accuracy and decision costs of whole-belief fallback.

The existing PointWorld/Flat'n'Fold path in #333 is the concrete candidate. Its
source asset/runtime/action/identity checks must pass before target access.
Freeze source calibration, query metrics/tolerances, grouping, comparators and
the BPT guard. Compare with the physical fallback, strongest simple deterministic
predictor, correct ungated partial-factor inference, and the guarded policy.
Report proper scores together with widths, physical-query error, admission,
accepted-update harm and worst-group regret. A policy that rejects everything is
not evidence of provider value. Causal4D remains a separate downstream test after
this bridge passes; it cannot repair an upstream competence failure.

This work does not open any historical target, retry a terminal provider result,
or require new hardware. It does not change existing paper claims or promote
any provider.
