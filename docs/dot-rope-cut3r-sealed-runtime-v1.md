# DOT rope validation through the sealed CUT3R runtime

This workflow evaluates the current Prob4D uncertainty treatment on the complete
official DOT V29 release mounted on `gpuserver4090` at:

```text
/mnt/seagate10tb/florianpfaff/datasets/dot
```

## Scientific question

The bounded source-development experiment asks whether preserving the dependence
and relative-gauge uncertainty between overlapping CUT3R windows gives a better
probabilistic account of a rope query than diagonal, independent, or ridge-like
closures.

The frozen provider and scoring protocol is
`protocols/dot-rope-cut3r-native-provider-v1.json`. It uses only rope sequences
`R01`, `R02`, and `R03` from `R01-10.zip`. Sequences `R04-R70` remain unopened.

## Information order

1. A synthetic forward pass verifies the sealed CUDA 12.6 CUT3R runtime without
   opening DOT.
2. The provider job opens only normal-view RGB frames for `R01-R03` and seals
   all prediction products.
3. A separate job downloads that immutable provider artifact.
4. Only then does the evaluation job open the registered two- and
   three-dimensional DOT marker coordinates for `R01-R03`.

No marker coordinate is available to model inference or provider selection.

## Runtime identity

The workflow binds the runtime produced by run `33326569532`:

- CUT3R revision `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`;
- checkpoint SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`;
- PyTorch 2.11 with CUDA 12.6;
- compiled native RoPE for the RTX 4090;
- content-addressed runtime and compatibility-patch receipts.

The provider copies and re-attests the compiled runtime. It does not rebuild the
extension and does not discover an interpreter from mutable repository
variables.

## Endpoints

The retained result reports, for each registered covariance closure:

- normalized Gaussian negative log score per dimension;
- Mahalanobis error;
- nominal 95% coverage;
- predictive standard deviation relative to rope span.

It also reports reconstruction and stitching error for the continuous run,
identity stitching, estimated Sim(3) stitching, and the oracle-window
diagnostic.

## Claim boundary

This is real-data **source-development** evidence on `R01-R03`. It does not
establish held-out transfer, independent real uncertainty calibration,
BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the
art. Promotion to `R04-R70` requires a separately frozen decision based only on
the registered source result.
