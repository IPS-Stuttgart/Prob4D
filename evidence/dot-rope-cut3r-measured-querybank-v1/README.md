# Measured DOT/CUT3R query-bank evidence

This compact receipt records the successful source-only experiment at Prob4D
revision `6d7e39130ffe85b3b5440774f53049076e0b1697`.

## Execution identity

- GitHub Actions run: `33422414154`
- Data/evaluation job: `99587926143`
- Artifact: `9769470690` (`sha256:2096301161ba827fd4b4694ba9e87fb321ed8f021d3c0d9abfba829182f2499f`)
- Result ID: `c54653900f1583c201a3dd2f0fdb689b6737e36075588fa28515ceeb38e4fb65`
- Result SHA-256: `8763e90fdd84388b60264a1ef9042cb8fa10eb344dbe9c888fbdd9d4d04fafa1`
- Protocol ID: `9b171a0b54e96a22f50e8d271721581c351fec73c0543dbf4428fc176d73a74f`
- Provider bundle ID: `952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7`
- Frozen dependence calibration: `943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7`,
  `alpha=0.85`

The workflow downloaded and MD5-verified official DOT V29 `R01-10.zip`, reused
the immutable marker-free CUT3R provider artifact, and opened only the previously
authorized R01--R03 marker payloads. Provider inference was not rerun and
R04--R70 remained unopened.

## Main result

The measured clustered-bootstrap covariance has rank seven for every source
sequence. The exact posterior-sufficient rank for joint banks of 1, 2, 4, and 8
registered probes is respectively 3, 6, 7, and 7.

Full and compressed gains, posterior covariances, and realized posterior means
agree numerically:

- maximum relative gain error:
  `5.272e-14`;
- maximum relative posterior-covariance error:
  `3.010e-13`;
- maximum realized mean difference:
  `1.788e-13`
  provider units.

The resident-model factor-plus-projection payload crosses below the direct joint
query cache at **four simultaneous probe queries** for all three sequences. Its
mean raw payload is 50.2% of the cache at four queries and 18.8% at eight
queries; compressed NPZ payload is 52.6% and 23.2%, respectively.

The self-contained factor remains larger than the cache in every registered
bank. Direct cached updates are 6.6--11.9 times faster. When the resident factor
is already present but the query gain has not yet been materialized, the
one-time cache construction is paid back after roughly 1,098--1,431 repeated
query evaluations.

At 10 Mbit/s, the resident factor route has lower modeled one-time
construction/serialization/transmission/materialization latency for the four-
and eight-query banks. At 100 Mbit/s it wins only for eight queries. At
1,000 Mbit/s the cache remains faster for all banks. The self-contained factor
does not win in any registered bandwidth scenario.

The PSD-consistent complete-joint spectral approximation under the same
incremental byte budget is valid but substantially less faithful: its mean NLL
is worse by about 0.92--1.06 nats per query dimension, and its joint coverage
falls as the query bank grows. These calibration-style metrics use measured
bootstrap parameter draws plus the frozen independent remainder; they are not a
new held-out-provider calibration claim.

## Claim boundary

This result supports a conditional systems claim only:

> Exact query-bound factor transmission is advantageous when several physical
> queries reuse an already-resident, measured low-dimensional nuisance model
> and communication is material. A direct cached Gaussian query message remains
> preferable for a single immutable query, a nonresident common model, or many
> repeated online evaluations.

It does not establish R04--R70 transfer, learned-provider state of the art,
BayesianPhysTwin benefit, Causal4D benefit, deployment calibration, or safety.
