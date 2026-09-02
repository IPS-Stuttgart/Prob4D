# Public DEFORM source-selected falsification witness

**Classification:** retrospective public-data falsification diagnostic.

This result tests whether a physical query selected using only source trajectories can expose an information-contract failure on held real trajectories. It uses the official public DEFORM DLO4/DLO5 release at commit `b73b8b8ecc033caefa693fab7898741d4e6dbeff`.

## Information order

- 80 training trajectories, 40 per DLO, fit source bias and covariance.
- A disjoint set of 32 training trajectories, 16 per DLO, selects one common unit 3-D point-displacement direction.
- The witness is sealed before the evaluation checkout.
- 28 complete evaluation trajectories are then opened exactly once.
- The evaluation contains 106,400 nested internal-node forecast rows; the complete trajectory remains the independent group.
- No held query reselection is performed.

The DLO4/DLO5 evaluation cohort was already open in earlier experiments. The result is therefore permanently retrospective and cannot be promoted to a prospective confirmation.

## Frozen witness

The generalized-eigen witness is

```text
q = [-0.252078, 0.964578, -0.077759]
```

and is dominated by the metric y direction. Under the registered overconfident negative control, its equal-source-group normalized error ratio is `20.4630`.

On the held trajectories, that same frozen query gives:

| Submission | Coordinate RMSE [m] | Query nNEES | Query 90% coverage | Query NLL |
| --- | ---: | ---: | ---: | ---: |
| Bias-corrected, overconfident | 0.0344835 | **18.2447** | **55.47%** | **5.3829** |
| Bias-corrected, full covariance | 0.0344835 | 0.9122 | 91.26% | -1.7855 |
| Transported velocity, full covariance | 0.0344824 | 0.9121 | 91.26% | -1.7855 |
| Transported velocity, diagonal covariance | 0.0344824 | 0.9315 | 91.08% | -1.7886 |

The failure witness therefore transfers from the disjoint source trajectories to all 28 held trajectory groups: the explicit overconfidence control remains badly underdispersed on the frozen physical direction.

## Interpretation

This is a positive validation of the **witness-selection protocol**, not a new learned-provider result. The negative control deliberately scales a source-fitted covariance by `0.05`. It verifies that the benchmark can:

1. find the most diagnostic physical direction without reading held outcomes;
2. bind that direction content-addressably;
3. reproduce the failure on held real trajectories; and
4. retain complete-trajectory aggregation rather than treating 106,400 nested rows as independent experiments.

The registered control pair did **not** exhibit a point-accuracy/query-calibration ranking reversal: source bias correction slightly worsened held coordinate RMSE, so transported velocity won both comparisons. This negative fact is retained rather than selecting another pair after seeing the target. A non-artificial ranking reversal remains a prospective benchmark requirement.

## Provenance

- Workflow run: `33635740556`
- Evaluated revision: `294af7fffeced18e13c06f76a0aaae016e2beb48`
- Complete result artifact: `9848722738`
- Artifact digest: `sha256:fc873742ab6c8bcc022a42ce5b9c4c6b808f27ec30e8cc0f46d902b5809e034c`
- Source artifact: `9848703491`
- Source artifact digest: `sha256:758ba7c4c30ca7570c3c34583ef942b555b87db3c78a03b2601a5d6b9e391e5e`
- Witness ID: `04db55d4037d594d03c58cdb661c1f0c8329e28f16b4d526df88f3db0aca9fab`

The full hash-bound source payload and per-group result remain in the Actions artifact. This directory retains only the compact claim-bound summary and provenance.
