# Real DOT/CUT3R Bayesian prefix development result

**Decision: strong improvement over metric-aligned CUT3R, but no overall Bayesian
point-accuracy superiority over last-residual correction. No automatic promotion.**

The fixed CPU experiment completed on all three already-open DOT source sequences:
3 ordinary predictions, 0 support fallbacks, 0 unsealable cases, 3 scored
sequences, and 28 scored 3D marker rows across six later frames. All predictions
were sealed before this run accessed later 3D scoring measurements. No new CUT3R
inference, RGB decoding, recording, or protected-target access occurred.

## Equal-sequence results

RMSE and full coordinate interval width are percentages of the prefix-only
marker bounding-box diagonal. NLL is per coordinate in these normalized units;
lower is better. Coverage is marginal 3D 90% ellipsoid coverage. Frames are
weighted equally within each sequence and sequences equally in the aggregate.

| Arm | 3D RMSE (% span) | NLL/coordinate | NEES/3 | Coverage | Full width (% span) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUT3R, initial frames 1-2 alignment | 20.0325 | -0.6494 | 1.4292 | 85.00% | 33.5486 |
| CUT3R, full frames 1-5 alignment | 13.5885 | -1.0436 | 0.6409 | 91.67% | 33.5486 |
| Last residual | **2.6661** | -1.3502 | 0.0277 | 100.00% | 33.5486 |
| Bayesian, independent observation noise | 3.2058 | **-2.4873** | 0.6329 | 100.00% | 7.9538 |
| Bayesian, shared observation error | 2.9383 | -2.4552 | 0.4217 | 100.00% | 9.1256 |

The shared-error Bayesian arm reduces aggregate RMSE by **78.3765%** relative
to full-prefix-aligned CUT3R and improves all three sequences. It improves RMSE
by **8.3427%** relative to independent-noise Bayes, but has slightly worse NLL
(+0.03215 nats/coordinate). It is **10.2120% worse** in aggregate RMSE than
last-residual correction, despite beating that control on two of three sequences.

| Sequence | Full-prefix CUT3R | Last residual | Bayes independent | Bayes shared |
| --- | ---: | ---: | ---: | ---: |
| R01 | 19.0405 | **1.8284** | 3.4900 | 3.7871 |
| R02 | 11.2198 | 1.9856 | 2.2562 | **1.6717** |
| R03 | 10.5053 | 4.1841 | 3.8711 | **3.3562** |

## What this does and does not establish

Early sparse 3D residual correction clearly helps this CUT3R reconstruction
interface. Bayesian conditioning works numerically and can substantially reduce
error, but the simplest last-residual mean remains best overall on this test.
This is evidence for prefix correction, not evidence that Bayesian inference is
necessary to obtain the point-accuracy improvement.

The Bayesian intervals are narrower and score better than the fixed Gaussian
wrappers on the deterministic controls. Those wrappers were **not independently
calibrated**, however. Therefore this does not show superiority over a calibrated
last-residual/conformal baseline. Likewise 100% coverage on 28 dependent rows
does not establish 90% population calibration. Shared dependence did not jointly
win on both mean accuracy and proper score against the independent Bayesian arm.

This is sparse-prefix-supervised, observed-frame reconstruction with common
current-frame RGB and released 2D marker query locations. The later 3D markers
are held out by this run, but these source sequences were already opened in
earlier work. It is not a fresh-object confirmation, unseen-future physical
forecast, fully automatic tracker, BayesianPhysTwin transfer, or SOTA result.
The next comparison should include source-calibrated last-residual uncertainty;
this result does not authorize opening a larger protected cohort or retuning
the frozen arms on these scoring measurements.

## Provenance

- Frozen implementation: `d70e0a426d1ec3e5da512e941c4e72f7915fb8b7`.
- Protocol ID: `30ce2d2825f3ef68798b402ae2945d0c4cd74216c36b35c327a37994f5837c14`.
- Result ID: `f8d138921e3bf9ce02721a9d4f7154ae5a4182fd56360dd2c40134a34e2a4941`.
- Result file SHA-256: `4cfaa725dbb740daf3e84d3d1f9a2eb8809443ec3055ba1d543718869848395b`.
- Barrier ID: `9a7d448192121533228c4ebbc5cb3ff9a16f0752d04ecb979ed992ccfc5492ab`.
- Provider bundle: `952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7`.
- Downloaded provider ZIP SHA-256: `565667959ca697c744a57528cd21308d0f040127bcb9296a4b8e9138b5c521c6`.
- Native execution: Python 3.10.12, NumPy 2.2.6; no GPU used.
- Raw predictions: `gpuserver4090:/home/florianpfaff/source-only/cut3r-bayesian-prefix-dev-v1-result`.

A second, scalar-form metric computation from the unchanged sealed predictions
and source scoring markers matched all per-sequence metrics within
`4.45e-16`. It also reverified prediction hashes and canonical result/barrier
identities. This is numerical reproduction by the same task, not independent
human review or independent data. Its receipt is `numerical-verification.json`.
The raw result's aggregate `scored_rows` is the mean count per sequence; the
total scored-row count is 28, as recorded by the verification receipt.
