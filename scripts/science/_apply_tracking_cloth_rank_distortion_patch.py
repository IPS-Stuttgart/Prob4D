"""One-shot transformation of the verified cloth pipeline into the frontier study."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

SCRIPT = Path("scripts/science/run_tracking_cloth_rank_distortion.py")
TESTS = Path("tests/test_tracking_cloth_rank_distortion.py")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement target, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def replace_region(path: Path, start: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    path.write_text(
        text[:start_index] + dedent(replacement).lstrip() + "\n\n" + text[end_index:],
        encoding="utf-8",
    )


if "prob4d.tracking-cloth-rank-distortion-real.v1" in SCRIPT.read_text(
    encoding="utf-8"
):
    print("rank-distortion transformation already applied")
    raise SystemExit(0)

replace_once(
    SCRIPT,
    dedent(
        '''\
        """Recording-disjoint posterior-preserving compression on real cloth trajectories.

        This experiment uses the public Tracking Cloth Deformation dataset's motion-capture
        CSV files.  It fits one local Gaussian query/observation model per cloth size and
        recording-disjoint fold, constructs a full structured observation covariance, and
        compares the full posterior with exact posterior-preserving shared-factor reduction.

        The script never exports raw trajectories and makes no learned-provider claim.
        """
        '''
    ),
    dedent(
        '''\
        """Recording-disjoint posterior rank--distortion on real cloth trajectories.

        This experiment uses the public Tracking Cloth Deformation motion-capture CSVs.
        It fits one local Gaussian query/observation model per cloth size and
        recording-disjoint fold, decomposes the observation covariance as S=A+UU.T, and
        compares equal-rank generalized-eigen, response-SVD, and covariance-PCA factors.

        The script never exports raw trajectories and makes no learned-provider claim.
        """
        '''
    ),
)
replace_once(
    SCRIPT,
    dedent(
        '''\
        from prob4d.posterior_preserving_compression import (
            compress_shared_factor_for_posterior,
        )
        '''
    ),
    "from prob4d.posterior_rank_distortion import posterior_rank_distortion_frontier\n",
)
replace_once(
    SCRIPT,
    'SCHEMA = "prob4d.tracking-cloth-posterior-compression-real.v1"',
    'SCHEMA = "prob4d.tracking-cloth-rank-distortion-real.v1"',
)

replace_region(
    SCRIPT,
    "def evaluate_fold(",
    "def _aggregate(",
    r'''
    def _response_svd_basis(
        shared: np.ndarray,
        prior: np.ndarray,
        cross: np.ndarray,
        observation_covariance: np.ndarray,
    ) -> np.ndarray:
        rank = shared.shape[1]
        solved = np.linalg.solve(
            observation_covariance,
            np.concatenate((shared, cross.T), axis=1),
        )
        solved_cross = solved[:, rank:]
        posterior = prior - cross @ solved_cross
        posterior = 0.5 * (posterior + posterior.T)
        root = np.linalg.cholesky(posterior)
        response = shared.T @ solved_cross
        normalized_response = np.linalg.solve(root, response.T).T
        left, _, _ = np.linalg.svd(normalized_response, full_matrices=True)
        return left


    def _candidate_evaluation(
        *,
        retained_rank: int,
        factor: np.ndarray,
        conditional: np.ndarray,
        prior: np.ndarray,
        cross: np.ndarray,
        test_q: np.ndarray,
        test_y: np.ndarray,
        query_mean: np.ndarray,
        observation_mean: np.ndarray,
        full_gain: np.ndarray,
        full_posterior: np.ndarray,
        full_means: np.ndarray,
    ) -> dict[str, Any]:
        observation_covariance = conditional + factor @ factor.T
        gain, posterior = _method_from_observation_covariance(
            prior,
            cross,
            observation_covariance,
        )
        record = _metrics(
            test_q,
            test_y,
            query_mean,
            observation_mean,
            gain,
            posterior,
        )
        contraction = 0.5 * (
            (full_posterior - posterior) + (full_posterior - posterior).T
        )
        root = np.linalg.cholesky(full_posterior)
        normalized = _whiten_symmetric(root, contraction)
        normalized = 0.5 * (normalized + normalized.T)
        eigenvalues = np.linalg.eigvalsh(normalized)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(eigenvalues[0]) < -1e-9 * scale:
            raise ValueError("factor projection produced non-PSD posterior contraction")
        innovation = test_y.reshape(len(test_y), -1) - observation_mean
        means = query_mean + innovation @ gain.T
        mean_difference = means - full_means
        whitened_mean_difference = np.linalg.solve(root, mean_difference.T)
        gain_denominator = max(float(np.linalg.norm(full_gain, ord="fro")), 1e-30)
        covariance_denominator = max(
            float(np.linalg.norm(full_posterior, ord="fro")),
            1e-30,
        )
        record.update(
            {
                "retained_rank": retained_rank,
                "payload_bytes": int(factor.nbytes),
                "normalized_covariance_trace_loss": max(
                    float(np.trace(normalized)),
                    0.0,
                ),
                "maximum_normalized_covariance_contraction": max(
                    float(eigenvalues[-1]),
                    0.0,
                ),
                "relative_gain_error": float(
                    np.linalg.norm(gain - full_gain, ord="fro") / gain_denominator
                ),
                "relative_posterior_covariance_difference": float(
                    np.linalg.norm(posterior - full_posterior, ord="fro")
                    / covariance_denominator
                ),
                "maximum_realized_mean_difference_m": float(
                    np.max(np.linalg.norm(mean_difference, axis=1))
                ),
                "heldout_normalized_mean_shift_risk": float(
                    np.mean(np.sum(whitened_mean_difference**2, axis=0))
                    / prior.shape[0]
                ),
            }
        )
        return record


    def evaluate_fold(
        train: list[RecordingSamples],
        test: list[RecordingSamples],
        protocol: dict[str, Any],
        fold_index: int,
        size: str,
    ) -> dict[str, Any]:
        train_y = np.concatenate([record.observations_m for record in train])
        train_q = np.concatenate([record.queries_m for record in train])
        test_y = np.concatenate([record.observations_m for record in test])
        test_q = np.concatenate([record.queries_m for record in test])
        mean, joint = _joint_covariance(
            train_q,
            train_y,
            float(protocol["joint_covariance_shrinkage"]),
            float(protocol["joint_covariance_ridge_fraction"]),
        )
        qdim = 3
        prior = joint[:qdim, :qdim]
        cross = joint[:qdim, qdim:]
        observation_covariance = joint[qdim:, qdim:]
        query_mean = mean[:qdim]
        observation_mean = mean[qdim:]
        conditional, shared, beta = decompose_shared_covariance(
            observation_covariance,
            float(protocol["maximum_conditional_block_fraction"]),
            float(protocol["factor_eigenvalue_relative_tolerance"]),
        )
        count = observation_covariance.shape[0] // 3
        original_rank = shared.shape[1]
        frontier = posterior_rank_distortion_frontier(
            shared.reshape(count, 3, original_rank),
            prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=DenseInnovation(observation_covariance),
            numerical_relative_tolerance=float(protocol["rank_relative_tolerance"]),
        )
        retained_ranks = [int(value) for value in protocol["retained_ranks"]]
        if retained_ranks != sorted(set(retained_ranks)):
            raise ValueError("retained_ranks must be strictly increasing and unique")
        if not retained_ranks or retained_ranks[0] < 0:
            raise ValueError("retained_ranks must be nonempty and nonnegative")
        if retained_ranks[-1] > original_rank:
            raise ValueError("retained_ranks exceed the shared-factor rank")

        full_gain, full_posterior = _posterior(
            prior,
            cross,
            observation_covariance,
        )
        full_metrics = _metrics(
            test_q,
            test_y,
            query_mean,
            observation_mean,
            full_gain,
            full_posterior,
        )
        full_metrics.update(
            {
                "retained_rank": original_rank,
                "payload_bytes": int(shared.nbytes),
                "normalized_covariance_trace_loss": 0.0,
                "maximum_normalized_covariance_contraction": 0.0,
                "relative_gain_error": 0.0,
                "relative_posterior_covariance_difference": 0.0,
                "maximum_realized_mean_difference_m": 0.0,
                "heldout_normalized_mean_shift_risk": 0.0,
            }
        )
        test_innovation = test_y.reshape(len(test_y), -1) - observation_mean
        full_means = query_mean + test_innovation @ full_gain.T
        response_svd_basis = _response_svd_basis(
            shared,
            prior,
            cross,
            observation_covariance,
        )
        covariance_pca_basis = np.linalg.svd(
            shared,
            full_matrices=False,
        )[2].T

        rank_results: list[dict[str, Any]] = []
        optimality_tolerance = float(protocol["optimality_relative_tolerance"])
        for retained_rank in retained_ranks:
            point = frontier.point(retained_rank)
            optimal_factor = point.compressed_factor_m.reshape(
                observation_covariance.shape[0],
                retained_rank,
            )
            response_svd_factor = shared @ response_svd_basis[:, :retained_rank]
            covariance_pca_factor = shared @ covariance_pca_basis[:, :retained_rank]
            methods = {
                "optimal_generalized_eigen": _candidate_evaluation(
                    retained_rank=retained_rank,
                    factor=optimal_factor,
                    conditional=conditional,
                    prior=prior,
                    cross=cross,
                    test_q=test_q,
                    test_y=test_y,
                    query_mean=query_mean,
                    observation_mean=observation_mean,
                    full_gain=full_gain,
                    full_posterior=full_posterior,
                    full_means=full_means,
                ),
                "response_svd": _candidate_evaluation(
                    retained_rank=retained_rank,
                    factor=response_svd_factor,
                    conditional=conditional,
                    prior=prior,
                    cross=cross,
                    test_q=test_q,
                    test_y=test_y,
                    query_mean=query_mean,
                    observation_mean=observation_mean,
                    full_gain=full_gain,
                    full_posterior=full_posterior,
                    full_means=full_means,
                ),
                "covariance_pca": _candidate_evaluation(
                    retained_rank=retained_rank,
                    factor=covariance_pca_factor,
                    conditional=conditional,
                    prior=prior,
                    cross=cross,
                    test_q=test_q,
                    test_y=test_y,
                    query_mean=query_mean,
                    observation_mean=observation_mean,
                    full_gain=full_gain,
                    full_posterior=full_posterior,
                    full_means=full_means,
                ),
            }
            optimal_trace = float(
                methods["optimal_generalized_eigen"][
                    "normalized_covariance_trace_loss"
                ]
            )
            audit_scale = max(
                float(point.audited_normalized_covariance_trace_loss),
                optimal_trace,
                1.0,
            )
            if (
                abs(
                    optimal_trace
                    - float(point.audited_normalized_covariance_trace_loss)
                )
                > optimality_tolerance * audit_scale
            ):
                raise ValueError(
                    "frontier point failed direct posterior-distortion audit"
                )
            for baseline in ("response_svd", "covariance_pca"):
                baseline_trace = float(
                    methods[baseline]["normalized_covariance_trace_loss"]
                )
                comparison_scale = max(optimal_trace, baseline_trace, 1.0)
                if optimal_trace > (
                    baseline_trace + optimality_tolerance * comparison_scale
                ):
                    raise ValueError(
                        f"generalized-eigen optimum lost to {baseline} at rank "
                        f"{retained_rank}"
                    )
            rank_results.append(
                {
                    "retained_rank": retained_rank,
                    "frontier": point.summary(),
                    "methods": methods,
                }
            )

        exact_rank = int(frontier.numerical_exact_rank)
        exact_matches = [
            record for record in rank_results if record["retained_rank"] == exact_rank
        ]
        if len(exact_matches) != 1:
            raise ValueError(
                "retained_ranks must contain the numerical exact rank exactly once"
            )
        exact_method = exact_matches[0]["methods"]["optimal_generalized_eigen"]
        return {
            "size": size,
            "fold": fold_index,
            "train_recording_count": len(train),
            "test_recording_count": len(test),
            "train_window_count": int(len(train_q)),
            "test_window_count": int(len(test_q)),
            "test_recordings": sorted(record.relative_path for record in test),
            "median_horizon_seconds": float(
                np.median(
                    np.concatenate([record.horizon_seconds for record in test])
                )
            ),
            "observation_count": count,
            "original_shared_rank": original_rank,
            "conditional_block_fraction": beta,
            "frontier": {
                "query_dimension": frontier.query_dimension,
                "numerical_exact_rank": exact_rank,
                "shared_precision_max_eigenvalue": (
                    frontier.shared_precision_max_eigenvalue
                ),
                "generalized_eigenvalues": frontier.generalized_eigenvalues.tolist(),
            },
            "full": full_metrics,
            "rank_results": rank_results,
            "exact_rank_full_parity": {
                "retained_rank": exact_rank,
                "relative_gain_error": exact_method["relative_gain_error"],
                "relative_posterior_covariance_error": exact_method[
                    "relative_posterior_covariance_difference"
                ],
                "maximum_realized_mean_difference_m": exact_method[
                    "maximum_realized_mean_difference_m"
                ],
            },
            "payload_bytes": {
                "full_shared_factor": int(shared.nbytes),
                "exact_frontier_factor": int(exact_method["payload_bytes"]),
                "cached_full_query_message": int(
                    full_gain.nbytes + full_posterior.nbytes
                ),
            },
        }
    ''',
)

replace_region(
    SCRIPT,
    "def _aggregate(",
    "def _load_protocol(",
    r'''
    def _aggregate_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
        weights = np.asarray(
            [record["sample_count"] for record in records],
            dtype=np.float64,
        )
        values: dict[str, Any] = {
            "sample_count": int(weights.sum()),
            "posterior_valid_fold_count": int(
                sum(bool(record["posterior_valid"]) for record in records)
            ),
            "fold_count": len(records),
            "minimum_posterior_eigenvalue_m2": float(
                min(record["minimum_posterior_eigenvalue_m2"] for record in records)
            ),
            "maximum_query_error_m": float(
                max(record["maximum_query_error_m"] for record in records)
            ),
            "payload_bytes_sum": int(
                sum(record["payload_bytes"] for record in records)
            ),
            "normalized_covariance_trace_loss_mean": float(
                np.mean(
                    [
                        record["normalized_covariance_trace_loss"]
                        for record in records
                    ]
                )
            ),
            "normalized_covariance_trace_loss_median": float(
                np.median(
                    [
                        record["normalized_covariance_trace_loss"]
                        for record in records
                    ]
                )
            ),
            "normalized_covariance_trace_loss_maximum": float(
                max(
                    record["normalized_covariance_trace_loss"]
                    for record in records
                )
            ),
            "maximum_normalized_covariance_contraction": float(
                max(
                    record["maximum_normalized_covariance_contraction"]
                    for record in records
                )
            ),
            "maximum_relative_gain_error": float(
                max(record["relative_gain_error"] for record in records)
            ),
            "maximum_relative_posterior_covariance_difference": float(
                max(
                    record["relative_posterior_covariance_difference"]
                    for record in records
                )
            ),
            "maximum_realized_mean_difference_m": float(
                max(
                    record["maximum_realized_mean_difference_m"]
                    for record in records
                )
            ),
            "heldout_normalized_mean_shift_risk": float(
                np.sum(
                    weights
                    * np.asarray(
                        [
                            record["heldout_normalized_mean_shift_risk"]
                            for record in records
                        ]
                    )
                )
                / np.sum(weights)
            ),
        }
        rmse = np.asarray([record["query_rmse_m"] for record in records])
        values["query_rmse_m"] = float(
            math.sqrt(np.sum(weights * rmse**2) / np.sum(weights))
        )
        for metric in (
            "mean_query_nll_nats",
            "mean_normalized_nees",
            "coverage_90",
        ):
            valid_indices = [
                index
                for index, record in enumerate(records)
                if record[metric] is not None
            ]
            if not valid_indices:
                values[metric] = None
                continue
            valid_weights = weights[valid_indices]
            items = np.asarray(
                [records[index][metric] for index in valid_indices],
                dtype=np.float64,
            )
            values[metric] = float(
                np.sum(valid_weights * items) / np.sum(valid_weights)
            )
        return values


    def _aggregate(
        folds: list[dict[str, Any]],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        total_test = sum(fold["test_window_count"] for fold in folds)
        retained_ranks = [int(value) for value in protocol["retained_ranks"]]
        aggregate_ranks: list[dict[str, Any]] = []
        strict_tolerance = float(protocol["strict_improvement_relative_tolerance"])
        maximum_optimality_violation = 0.0

        for retained_rank in retained_ranks:
            fold_rank_records = []
            for fold in folds:
                matches = [
                    record
                    for record in fold["rank_results"]
                    if record["retained_rank"] == retained_rank
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "fold does not contain exactly one requested rank"
                    )
                fold_rank_records.append(matches[0])
            methods = {
                name: _aggregate_metric_records(
                    [record["methods"][name] for record in fold_rank_records]
                )
                for name in (
                    "optimal_generalized_eigen",
                    "response_svd",
                    "covariance_pca",
                )
            }
            response_improvements: list[float] = []
            response_relative_improvements: list[float] = []
            covariance_improvements: list[float] = []
            strict_response_count = 0
            strict_covariance_count = 0
            unique_count = 0
            exact_count = 0
            boundary_gaps: list[float] = []
            for record in fold_rank_records:
                point = record["frontier"]
                optimum = float(
                    record["methods"]["optimal_generalized_eigen"][
                        "normalized_covariance_trace_loss"
                    ]
                )
                response = float(
                    record["methods"]["response_svd"][
                        "normalized_covariance_trace_loss"
                    ]
                )
                covariance = float(
                    record["methods"]["covariance_pca"][
                        "normalized_covariance_trace_loss"
                    ]
                )
                response_gap = response - optimum
                covariance_gap = covariance - optimum
                response_improvements.append(response_gap)
                covariance_improvements.append(covariance_gap)
                response_relative_improvements.append(
                    response_gap / max(abs(response), 1e-30)
                )
                response_scale = max(abs(response), abs(optimum), 1.0)
                covariance_scale = max(abs(covariance), abs(optimum), 1.0)
                if response_gap > strict_tolerance * response_scale:
                    strict_response_count += 1
                if covariance_gap > strict_tolerance * covariance_scale:
                    strict_covariance_count += 1
                maximum_optimality_violation = max(
                    maximum_optimality_violation,
                    optimum - response,
                    optimum - covariance,
                )
                unique_count += int(bool(point["optimal_subspace_unique"]))
                exact_count += int(bool(point["exact_posterior"]))
                if point["boundary_generalized_eigengap"] is not None:
                    boundary_gaps.append(
                        float(point["boundary_generalized_eigengap"])
                    )

            aggregate_ranks.append(
                {
                    "retained_rank": retained_rank,
                    "frontier": {
                        "optimal_subspace_unique_fold_count": unique_count,
                        "exact_posterior_fold_count": exact_count,
                        "minimum_boundary_generalized_eigengap": (
                            min(boundary_gaps) if boundary_gaps else None
                        ),
                    },
                    "methods": methods,
                    "comparisons": {
                        "response_svd_strict_improvement_fold_count": (
                            strict_response_count
                        ),
                        "response_svd_trace_improvement_mean": float(
                            np.mean(response_improvements)
                        ),
                        "response_svd_trace_improvement_median": float(
                            np.median(response_improvements)
                        ),
                        "response_svd_relative_trace_improvement_mean": float(
                            np.mean(response_relative_improvements)
                        ),
                        "covariance_pca_strict_improvement_fold_count": (
                            strict_covariance_count
                        ),
                        "covariance_pca_trace_improvement_mean": float(
                            np.mean(covariance_improvements)
                        ),
                        "covariance_pca_trace_improvement_median": float(
                            np.median(covariance_improvements)
                        ),
                    },
                }
            )

        full_records = [fold["full"] for fold in folds]
        full = _aggregate_metric_records(full_records)
        parity = [fold["exact_rank_full_parity"] for fold in folds]
        full_bytes = sum(
            fold["payload_bytes"]["full_shared_factor"] for fold in folds
        )
        exact_bytes = sum(
            fold["payload_bytes"]["exact_frontier_factor"] for fold in folds
        )
        return {
            "fold_count": len(folds),
            "test_window_count": total_test,
            "retained_ranks": retained_ranks,
            "numerical_exact_ranks": sorted(
                {
                    int(fold["frontier"]["numerical_exact_rank"])
                    for fold in folds
                }
            ),
            "original_shared_rank_min": min(
                int(fold["original_shared_rank"]) for fold in folds
            ),
            "original_shared_rank_max": max(
                int(fold["original_shared_rank"]) for fold in folds
            ),
            "maximum_optimality_violation": max(
                maximum_optimality_violation,
                0.0,
            ),
            "exact_rank_full_parity": {
                "maximum_relative_gain_error": max(
                    float(value["relative_gain_error"]) for value in parity
                ),
                "maximum_relative_posterior_covariance_error": max(
                    float(value["relative_posterior_covariance_error"])
                    for value in parity
                ),
                "maximum_realized_mean_difference_m": max(
                    float(value["maximum_realized_mean_difference_m"])
                    for value in parity
                ),
            },
            "summed_full_shared_factor_bytes": full_bytes,
            "summed_exact_frontier_factor_bytes": exact_bytes,
            "exact_rank_shared_factor_payload_reduction_ratio": (
                float(full_bytes / exact_bytes) if exact_bytes else None
            ),
            "full": full,
            "ranks": aggregate_ranks,
        }
    ''',
)

replace_region(
    SCRIPT,
    "def _load_protocol(",
    "def _summary_markdown(",
    r'''
    def _load_protocol(path: Path) -> dict[str, Any]:
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol.get("schema") != SCHEMA:
            raise ValueError("unsupported protocol schema")
        required = {
            "fold_count",
            "lag_frames",
            "horizon_frames",
            "stride_frames",
            "maximum_windows_per_recording",
            "minimum_recordings_per_size",
            "joint_covariance_shrinkage",
            "joint_covariance_ridge_fraction",
            "maximum_conditional_block_fraction",
            "factor_eigenvalue_relative_tolerance",
            "rank_relative_tolerance",
            "retained_ranks",
            "optimality_relative_tolerance",
            "strict_improvement_relative_tolerance",
            "required_maximum_relative_parity_error",
        }
        missing = sorted(required - set(protocol))
        if missing:
            raise ValueError(f"protocol is missing fields: {missing}")
        ranks = protocol["retained_ranks"]
        if (
            not isinstance(ranks, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in ranks
            )
            or ranks != sorted(set(ranks))
            or not ranks
            or ranks[0] < 0
        ):
            raise ValueError(
                "retained_ranks must be a nonempty increasing integer list"
            )
        return protocol
    ''',
)

replace_region(
    SCRIPT,
    "def _summary_markdown(",
    "def run(",
    r'''
    def _summary_markdown(result: dict[str, Any]) -> str:
        if result["status"] != "evaluated-real-rank-distortion":
            return (
                "# Tracking Cloth rank--distortion evaluation\n\n"
                f"Status: **{result['status']}**\n\n"
                f"Reason: {result.get('reason', 'unspecified')}\n"
            )
        aggregate = result["aggregate"]
        accepted = result["inventory"]["accepted_recording_count"]
        csv_count = result["inventory"]["csv_file_count"]
        parity = aggregate["exact_rank_full_parity"]
        reduction = aggregate["exact_rank_shared_factor_payload_reduction_ratio"]
        lines = [
            "# Tracking Cloth posterior rank--distortion evaluation",
            "",
            "Status: **evaluated real trajectories**",
            "",
            f"- Parsed cloth-only recordings: {accepted} / {csv_count}",
            f"- Recording-disjoint folds: {aggregate['fold_count']}",
            f"- Held-out windows: {aggregate['test_window_count']}",
            (
                "- Original shared rank: "
                f"{aggregate['original_shared_rank_min']}–"
                f"{aggregate['original_shared_rank_max']}"
            ),
            f"- Evaluated retained ranks: {aggregate['retained_ranks']}",
            f"- Numerical exact ranks: {aggregate['numerical_exact_ranks']}",
            (
                "- Maximum theorem-optimality violation: "
                f"{aggregate['maximum_optimality_violation']:.3e}"
            ),
            (
                "- Exact-rank maximum relative gain / covariance error: "
                f"{parity['maximum_relative_gain_error']:.3e} / "
                f"{parity['maximum_relative_posterior_covariance_error']:.3e}"
            ),
            (
                "- Exact-rank maximum realized posterior-mean difference: "
                f"{1000.0 * parity['maximum_realized_mean_difference_m']:.3e} mm"
            ),
            f"- Exact-rank shared-factor payload reduction: {reduction:.2f}x",
            "",
            "## Rank frontier",
            "",
            (
                "| rank | optimal D | response-SVD D | mean trace improvement | "
                "strict folds | valid folds | optimal RMSE [mm] | SVD RMSE [mm] |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for rank_record in aggregate["ranks"]:
            optimal = rank_record["methods"]["optimal_generalized_eigen"]
            response = rank_record["methods"]["response_svd"]
            comparisons = rank_record["comparisons"]
            lines.append(
                "| "
                f"{rank_record['retained_rank']} | "
                f"{optimal['normalized_covariance_trace_loss_mean']:.6g} | "
                f"{response['normalized_covariance_trace_loss_mean']:.6g} | "
                f"{comparisons['response_svd_trace_improvement_mean']:.6g} | "
                f"{comparisons['response_svd_strict_improvement_fold_count']}/"
                f"{aggregate['fold_count']} | "
                f"{optimal['posterior_valid_fold_count']}/"
                f"{aggregate['fold_count']} | "
                f"{1000.0 * optimal['query_rmse_m']:.3f} | "
                f"{1000.0 * response['query_rmse_m']:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    "The generalized-eigen method is globally optimal only for "
                    "the registered normalized posterior-covariance trace "
                    "contraction within each fitted `U -> U V` family. Held-out "
                    "RMSE, NLL, NEES, coverage, posterior validity, eigengaps, "
                    "and non-unique boundaries are reported rather than inferred "
                    "from that theorem."
                ),
                "",
                (
                    "The experiment is a recording-disjoint local-Gaussian "
                    "mechanism study on real motion-capture trajectories. It does "
                    "not establish a learned 4-D provider, deployment calibration, "
                    "or BayesianPhysTwin/Causal4D physical benefit."
                ),
                "",
            ]
        )
        return "\n".join(lines)
    ''',
)

replace_once(
    SCRIPT,
    '        "status": "not-evaluated",\n        "inventory": {',
    '        "status": "not-evaluated",\n        "inventory": {',
)
replace_once(
    SCRIPT,
    dedent(
        '''\
            "claim_boundary": {
                "real_motion_capture_trajectories": True,
                "recording_disjoint_evaluation": True,
                "learned_4d_provider_evaluated": False,
                "real_covariance_calibration_claimed": False,
                "bayesian_phystwin_benefit_claimed": False,
                "causal4d_benefit_claimed": False,
            },
        '''
    ),
    dedent(
        '''\
            "claim_boundary": {
                "real_motion_capture_trajectories": True,
                "training_only_joint_covariance_fit": True,
                "recording_disjoint_heldout_evaluation": True,
                "rank_distortion_theorem_evaluated": True,
                "learned_4d_provider_evaluated": False,
                "deployment_covariance_calibration_claimed": False,
                "arbitrary_task_loss_optimality_claimed": False,
                "recursive_exactness_claimed": False,
                "bayesian_phystwin_benefit_claimed": False,
                "causal4d_benefit_claimed": False,
            },
        '''
    ),
)
replace_once(SCRIPT, "aggregate = _aggregate(evaluations)", "aggregate = _aggregate(evaluations, protocol)")
replace_once(
    SCRIPT,
    dedent(
        '''\
                if aggregate["maximum_relative_gain_error"] > float(
                    protocol["required_maximum_relative_parity_error"]
                ):
                    raise ValueError("posterior gain parity exceeded the protocol limit")
                if aggregate["maximum_relative_posterior_covariance_error"] > float(
                    protocol["required_maximum_relative_parity_error"]
                ):
                    raise ValueError("posterior covariance parity exceeded the protocol limit")
        '''
    ),
    dedent(
        '''\
                parity = aggregate["exact_rank_full_parity"]
                if parity["maximum_relative_gain_error"] > float(
                    protocol["required_maximum_relative_parity_error"]
                ):
                    raise ValueError("exact-rank posterior gain parity exceeded the protocol limit")
                if parity["maximum_relative_posterior_covariance_error"] > float(
                    protocol["required_maximum_relative_parity_error"]
                ):
                    raise ValueError(
                        "exact-rank posterior covariance parity exceeded the protocol limit"
                    )
                if aggregate["maximum_optimality_violation"] > float(
                    protocol["optimality_relative_tolerance"]
                ):
                    raise ValueError("the registered global-optimality inequality was violated")
        '''
    ),
)
replace_once(
    SCRIPT,
    '                "status": "evaluated-real-trajectories",',
    '                "status": "evaluated-real-rank-distortion",',
)
replace_once(
    SCRIPT,
    '        "schema": "prob4d.tracking-cloth-posterior-compression-manifest.v1",',
    '        "schema": "prob4d.tracking-cloth-rank-distortion-manifest.v1",',
)
replace_once(
    SCRIPT,
    '    return 0 if result["status"] == "evaluated-real-trajectories" else 3',
    '    return 0 if result["status"] == "evaluated-real-rank-distortion" else 3',
)

replace_once(
    TESTS,
    "run_tracking_cloth_posterior_compression.py",
    "run_tracking_cloth_rank_distortion.py",
)
replace_once(
    TESTS,
    '"run_tracking_cloth_posterior_compression", SCRIPT',
    '"run_tracking_cloth_rank_distortion", SCRIPT',
)
replace_region(
    TESTS,
    "def _protocol(",
    "def _grid(",
    r'''
    def _protocol(**updates: object) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": MODULE.SCHEMA,
            "fold_count": 3,
            "lag_frames": 3,
            "horizon_frames": 6,
            "stride_frames": 6,
            "maximum_windows_per_recording": 16,
            "minimum_recordings_per_size": 3,
            "joint_covariance_shrinkage": 0.15,
            "joint_covariance_ridge_fraction": 1e-5,
            "maximum_conditional_block_fraction": 0.20,
            "factor_eigenvalue_relative_tolerance": 1e-12,
            "rank_relative_tolerance": 1e-12,
            "retained_ranks": [0, 1, 2, 3],
            "optimality_relative_tolerance": 1e-8,
            "strict_improvement_relative_tolerance": 1e-8,
            "required_maximum_relative_parity_error": 1e-7,
        }
        value.update(updates)
        return value
    ''',
)
replace_region(
    TESTS,
    "def test_recording_disjoint_real_trajectory_study_preserves_posterior(",
    "def test_balanced_hash_assignment_populates_all_folds(",
    r'''
    def test_recording_disjoint_real_trajectory_study_builds_rank_frontier(
        tmp_path: Path,
    ) -> None:
        dataset = tmp_path / "dataset"
        for size, offset in (("A2", 0), ("A3", 100)):
            for index in range(6):
                _write_recording(
                    dataset
                    / "Free-hanging"
                    / f"material_{size}_shake_fast_hands_{index}.csv",
                    size=size,
                    seed=offset + index,
                )
        protocol_path = tmp_path / "protocol.json"
        protocol_path.write_text(
            json.dumps(_protocol(), sort_keys=True),
            encoding="utf-8",
        )
        output = tmp_path / "output"
        result = MODULE.run(dataset, protocol_path, output, "a" * 40)
        assert result["status"] == "evaluated-real-rank-distortion"
        aggregate = result["aggregate"]
        assert aggregate["fold_count"] == 6
        assert aggregate["retained_ranks"] == [0, 1, 2, 3]
        assert aggregate["numerical_exact_ranks"] == [3]
        assert aggregate["original_shared_rank_min"] >= 36
        assert aggregate["maximum_optimality_violation"] < 1e-8

        for rank_record in aggregate["ranks"]:
            optimum = rank_record["methods"]["optimal_generalized_eigen"]
            response = rank_record["methods"]["response_svd"]
            covariance = rank_record["methods"]["covariance_pca"]
            assert (
                optimum["normalized_covariance_trace_loss_mean"]
                <= response["normalized_covariance_trace_loss_mean"] + 1e-8
            )
            assert (
                optimum["normalized_covariance_trace_loss_mean"]
                <= covariance["normalized_covariance_trace_loss_mean"] + 1e-8
            )

        exact = next(
            record for record in aggregate["ranks"] if record["retained_rank"] == 3
        )
        parity = aggregate["exact_rank_full_parity"]
        assert parity["maximum_relative_gain_error"] < 1e-8
        assert parity["maximum_relative_posterior_covariance_error"] < 1e-8
        assert parity["maximum_realized_mean_difference_m"] < 1e-10
        np.testing.assert_allclose(
            exact["methods"]["optimal_generalized_eigen"]["query_rmse_m"],
            aggregate["full"]["query_rmse_m"],
            atol=1e-12,
            rtol=1e-10,
        )
        assert aggregate["exact_rank_shared_factor_payload_reduction_ratio"] > 10.0
        assert not list(output.rglob("*.csv"))
        assert (output / "result.json").is_file()
        assert (output / "inventory.json").is_file()
        assert (output / "manifest.json").is_file()
        assert (output / "summary.md").is_file()
    ''',
)

print("tracking-cloth rank-distortion transformation applied")
