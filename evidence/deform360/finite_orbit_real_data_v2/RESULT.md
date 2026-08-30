# Broader Deform360 finite-orbit real-data pilot v2

## Execution

- Decision: **completed-retrospective-real-data-pilot**
- GitHub run: `33335597467` on `workstation1`
- Evaluated revision: `7ef009324cda282e30e994df47393d0d37ffb49a`
- Result ID: `5700a8d643b97898263800195a75fbf5a587165fa28ce2f70c4397f7c89f4117`
- Artifact: `deform360-finite-orbit-real-data-v2-33335597467-1` (ID `9738929881`)
- Artifact SHA-256: `be60bce5ffb6b6fdfc720a5a88f6d49477d2d5f5c77d3c3a2e51c960c8238e21`

## Support

- Requested objects: **18**
- Safely materialized source/target units: **36/36**
- Complete pairs passing the frozen principal-axis geometry gate: **16**
- Unsupported pair components: `075-leather` target episode 1 and `078-fishing-line` source episode 0.

## Registered outcomes

Errors and regret are normalized by the current principal-axis endpoint span. Negative regret favors the candidate.

| Query | H | Fallback error | Ungated error | Ungated regret (95% object bootstrap CI) | Harmful ungated windows | Shared acceptance |
|---|---:|---:|---:|---:|---:|---:|
| span_change | 1 | 0.008301 | 0.008226 | -0.000076 [-0.000936, +0.001093] | 0.026 | 0.000 |
| span_change | 2 | 0.015763 | 0.016264 | +0.000501 [-0.000739, +0.002055] | 0.047 | 0.000 |
| span_change | 4 | 0.029750 | 0.032133 | +0.002383 [+0.000278, +0.004646] | 0.074 | 0.000 |
| centroid_axis_progress | 1 | 0.004963 | 0.004744 | -0.000219 [-0.000596, +0.000232] | 0.017 | 0.000 |
| centroid_axis_progress | 2 | 0.009725 | 0.009352 | -0.000373 [-0.001008, +0.000404] | 0.027 | 0.000 |
| centroid_axis_progress | 4 | 0.019233 | 0.019172 | -0.000061 [-0.001006, +0.001198] | 0.040 | 0.000 |
| named_endpoint_axis_progress | 1 | 0.009726 | 0.009247 | -0.000479 [-0.001297, +0.000378] | 0.024 | 0.000 |
| named_endpoint_axis_progress | 2 | 0.019219 | 0.018443 | -0.000776 [-0.002062, +0.000587] | 0.033 | 0.000 |
| named_endpoint_axis_progress | 4 | 0.038460 | 0.038251 | -0.000210 [-0.001959, +0.001798] | 0.053 | 0.000 |

## Interpretation

- The local-canonical, independent-orbit, and shared-orbit rules all rejected every registered query/horizon cell and therefore reproduced the exact persistence fallback.
- All object-disjoint shared lower confidence bounds were negative. The broader cross-object source data did not certify a positive candidate advantage.
- The ungated candidate had small mean improvements in six cells, but no improvement confidence interval excluded zero.
- The only statistically directional result was negative: span change at horizon 4 had regret `+0.002383`, 95% CI `[+0.000278,+0.004646]`.
- Across all 8,736 ungated target windows, mean regret was `+0.000068` and the harmful-window rate was `3.77%`.

## Scientific conclusion

This broader object-disjoint run does **not** validate nontrivial shared-orbit admission. It shows that the frozen conservative rule fails closed under cross-object heterogeneity, but its acceptance is zero. The earlier rope-only positive mechanism example therefore does not generalize to this 16-object cohort under the present constant-velocity provider and calibration rule.

## Claim boundary

Retrospective real measured Deform360 geometry evidence for a principal-axis C2 finite-orbit query-admission mechanism only; not fresh confirmation, learned-provider competence, BayesianPhysTwin or Causal4D benefit, deployment safety, or state of the art.
