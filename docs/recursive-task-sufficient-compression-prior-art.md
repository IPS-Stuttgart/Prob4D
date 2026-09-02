# Prior-art boundary for recursive task sufficiency

This research branch must not claim that recursive estimation of state functions is new.

## Established recursive functional-estimation lineage

The prior-art audit already rules out several broad novelty formulations:

- Kondo (1978), *Design of Linear Functional Filters for Linear Stochastic Systems*,
  DOI `10.9746/sicetr1965.14.613`, constructs lower-order linear unbiased filters for linear
  functions of stochastic-system state and studies their use in LQG feedback.
- Kondo, Sunaga, and Sakamoto (1979), *Design of linear functional filters for linear stochastic
  systems*, DOI `10.1080/00207177908922685`, likewise treats lower-order functional filtering
  for stochastic linear systems.
- Fernando, Trinh, and Jennings (2010), *Functional Observability and the Design of Minimum Order
  Linear Functional Observers*, DOI `10.1109/TAC.2010.2042761`, explicitly solves the
  minimum-order linear functional-observer problem for LTI systems.
- Darouach and subsequent work gives functional-observability/detectability conditions and
  functional-observer constructions; the modern literature continues to refine minimum-order
  and observer-existence questions.
- Reduced-order / functional stochastic filtering is broader than deterministic observers; e.g.
  Nagpal, Helmick, and Sims (1987), *Reduced-order estimation. Part 1. Filtering*, and later
  optimal unbiased functional-filtering work treat recursive reduced-order estimators directly.

Therefore neither "recursive task estimation", "minimum task-state order", nor "exact recursive
functional filtering" is a defensible standalone novelty claim for this branch.

## Established measurement-compression lineage

Lossless sensor transformations are also established. Duan and Li (2011), DOI
`10.1109/TSP.2010.2084574`, design linear transformations of sensor data that retain centralized
fusion performance. Liu et al. (2013), DOI `10.1016/j.jprocont.2013.09.009`, extend lossless
linear-transformation reasoning to cross-correlated measurement noises. This prevents a broad
claim of being the first lossless correlated-noise compression method for recursive estimation.

Goal-oriented Bayesian reduction is established as well. Spantini et al. (2017), DOI
`10.1137/16M1082123`, derive goal-oriented low-rank approximations for linear-Gaussian inverse
problems, and related work covers posterior maps and quantity-of-interest reduction.

## Narrow candidate distinction that remains

The current search did not identify an equivalent result with all of the following restrictions and
objects simultaneously:

1. all measurement rows are retained unchanged;
2. the physical/task-state representation is fixed rather than replaced by a measurement
   transformation;
3. all non-shared covariance terms are retained unchanged;
4. only a supplied latent factor `U` in `S = A + U U^T` is projected;
5. the retained latent subspace is necessary and sufficient for the registered Gaussian posterior
   and has minimum rank **within that factor-projection family**;
6. the registered query is enlarged only as required by an independently audited recursive
   task-state closure; and
7. the method fails closed to the original full factor when the registered closure or factor cannot
   be reduced.

This is the candidate Prob4D contribution: compose an established recursively sufficient task
state with the existing necessary/sufficient shared-noise-factor projection and make the resulting
exactness/fallback boundary explicit and auditable. The recursive induction itself is simple once
the correct task state is supplied; novelty must be located in the constrained correlated-factor
interface, not in functional filtering.

This distinction still requires a specialist literature review. In particular, search for stochastic
functional filters that explicitly factor or transform correlated measurement covariance, and for
lossless distributed transformations whose algebra can be rewritten as an equivalent latent-noise
factor projection. If such a result is equivalent after a change of variables, the recursive theorem
should be presented as an application/composition rather than a new theorem.

The controlled study on this branch is mechanism evidence only. It is not a prior-art novelty
determination and not a robotics-performance result.
