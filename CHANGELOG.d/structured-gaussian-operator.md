Add a stable structured Gaussian operator for dense, sparse, and tree-sparse
observation-factor stacks. It provides exact inverse covariance actions,
precision quadratics, log determinants, and Gaussian negative log likelihoods
without materializing the full observation covariance, reports factor storage,
and fails closed on singular conditional row covariances. Also canonicalize the
axis sign of exact-pi `SO(3)` logarithms so serialized `Sim3` vectors and content
identities do not depend on eigensolver sign choices. These changes add numerical
and proper-score infrastructure only; they do not establish calibration or
provider promotion.
