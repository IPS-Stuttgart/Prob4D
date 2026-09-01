# Terminal DOT/CUT3R confirmation result: support negative

The source-frozen DOT/CUT3R confirmation on sequences R04--R10 is now
terminal. The exact sealed provider artifact and exact frozen evaluator were
successfully recovered and executed in GitHub Actions run `33550572437`.

The result is **held-out support-negative**. Evaluation terminated before the
registered NLL comparison because pooled provider/truth support was below its
predeclared minimum. Thus the source-selected shared-dependence value
`alpha=0.85` is not supported as a held-out result.

This is an important evidential boundary:

- the provider itself completed and was sealed before marker access;
- the negative is not caused by the prior packaging failure;
- no provider rerun, confirmation retuning, or scientific-input change occurred;
- R11--R70 remained unopened by this experiment;
- no BayesianPhysTwin or Causal4D outcome was computed.

The paper-facing conclusion should be that the registered CUT3R route did not
supply enough qualified correspondence support for the intended held-out
probabilistic comparison. It should not be reported as a failed NLL result, and
it should not be rescued by changing the support rule after target access.

Evidence is retained under
`evidence/dot-cut3r-r04-r10-heldout-negative-v1/`, result ID
`efe88e93985e88d42efd8375e24af40b04cfe8bf18bc1352cd83cf2b893861d0`.
