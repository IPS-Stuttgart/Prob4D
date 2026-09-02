# Evidence registration: posterior rank--distortion study

This file records the claim-bearing protocol before interpreting the study
artifact.

- Branch: `research/query-rank-distortion-frontier-v1`
- Implementation: `src/prob4d/posterior_rank_distortion.py`
- Dense theorem controls: `tests/test_posterior_rank_distortion.py`
- Multi-seed study: `scripts/research/posterior_rank_distortion_study.py`
- Workflow: `.github/workflows/posterior-rank-distortion-study.yml`
- Frozen strata: `7:1`, `7:3`, `14:3`, and `28:5`
- Seeds per stratum: 128, including the preregistered seed-93 control
- Primary objective: posterior-normalized covariance trace contraction
- Primary comparison: Euclidean posterior-response SVD at identical rank
- Secondary comparison: latent covariance-energy PCA at identical rank
- Required theorem audit: closed-form and exact downdate distortion agree
- Required control: rank-1 SVD/optimum ratio exceeds 1.24 for `7:3`, seed 93

No real-data, nonlinear, closed-loop, observation-likelihood, or full-posterior
KL claim is authorized by this protocol.  Promotion beyond synthetic
prevalence and effect size requires a separately frozen physical-data factor
export and evaluation.
