# Proof sketch: recursive exactness by task-state closure

Let `T` have orthonormal rows spanning the registered closure and define `z_t = T x_t`.
The closure audit gives matrices `A`, `D`, and `B` such that

\[
T F=A T,\qquad H=D T,\qquad L=B T.
\]

For exogenous Gaussian process and measurement noises,

\[
z_{t+1}=A z_t+T w_t,\qquad y_t=D z_t+v_t,\qquad q_t=B z_t
\]

is therefore an exact marginal state-space model for the registered task state and observations.
No discarded state coordinate appears in either the transition or observation equations.

At update `t`, let the full innovation covariance be

\[
S_t=R_t+D P_t^z D^\top=A_t^{(y)}+U_tU_t^\top,
\]

and let `C_t = P_t^z D^T`. The existing posterior-preserving compression theorem chooses a
projector `V_t` whose range contains

\[
\operatorname{range}(U_t^\top S_t^{-1}C_t^\top).
\]

Replacing `U_t` by `U_t V_t` leaves the posterior mean of `z_t` unchanged for every innovation
and leaves its posterior covariance unchanged. Thus the compressed and full filters leave update
`t` with the same Gaussian belief over `z_t`.

Prediction through the exact reduced dynamics then gives the same next prior,

\[
\mathcal N(A m_{t|t}^z, A P_{t|t}^z A^\top+TQ_tT^\top),
\]

for both filters. Induction over updates therefore gives the same posterior over `z_t`, and hence
the same posterior over every registered task `q_t=Bz_t`, for every measurement sequence.

The proof does not require the compressed model to preserve the observation evidence. It only uses
posterior parity of the recursively sufficient state. Consequently this result cannot be used for
likelihood-based model selection without retaining the full observation model.

The minimum retained shared-factor rank at each update is still the existing one-step quantity

\[
\operatorname{rank}(U_t^\top S_t^{-1}C_t^\top)\leq\dim z_t.
\]

What changes under recursion is the correct query: `z_t` must contain the complete registered
recursive task closure rather than only the task reported at the current instant.
