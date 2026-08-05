"""Grouped CLI for portable material-identity streams and mixtures.

The commands in this module do not fit calibration parameters or decide whether a
BayesianPhysTwin update is accepted. They seal externally source-calibrated
mixtures, validate stream/mixture artifacts, and exercise exact downstream
likelihood marginalization or Gaussian moment matching with candidate-ID checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
    load_material_identity_mixture,
    marginalize_identity_log_likelihoods,
    moment_match_gaussian_identity_hypotheses,
    write_material_identity_mixture,
)
from .material_identity_stream import load_material_identity_stream


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or invalid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _list(value: Any, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _endpoint(value: Any, *, name: str) -> LocalTrackEndpoint:
    return LocalTrackEndpoint.from_mapping(value, name=name)


def _candidate(value: Any, *, index: int) -> MaterialIdentityCandidateV1:
    name = f"candidates[{index}]"
    mapping = _mapping(value, name=name)
    _exact_fields(
        mapping,
        {
            "source_endpoint",
            "association_result_id",
            "source_score",
            "calibrated_log_weight",
            "metadata",
        },
        name=name,
    )
    source_raw = mapping["source_endpoint"]
    source = (
        None
        if source_raw is None
        else _endpoint(source_raw, name=f"{name}.source_endpoint")
    )
    return MaterialIdentityCandidateV1(
        source_endpoint=source,
        association_result_id=mapping["association_result_id"],
        source_score=mapping["source_score"],
        calibrated_log_weight=mapping["calibrated_log_weight"],
        metadata=_mapping(mapping["metadata"], name=f"{name}.metadata"),
    )


def mixture_from_config(value: Any) -> MaterialIdentityMixtureV1:
    """Build one canonical mixture from externally calibrated source-side weights."""

    mapping = _mapping(value, name="material-identity mixture configuration")
    _exact_fields(
        mapping,
        {
            "target_endpoint",
            "window_order",
            "causal_frame_stop",
            "association_rule_id",
            "calibration_id",
            "tracklet_producer_revision",
            "association_revision",
            "candidates",
            "metadata",
        },
        name="material-identity mixture configuration",
    )
    window_order = _list(mapping["window_order"], name="window_order")
    candidates = _list(mapping["candidates"], name="candidates")
    return MaterialIdentityMixtureV1(
        target_endpoint=_endpoint(mapping["target_endpoint"], name="target_endpoint"),
        window_order=tuple(window_order),
        causal_frame_stop=mapping["causal_frame_stop"],
        association_rule_id=mapping["association_rule_id"],
        calibration_id=mapping["calibration_id"],
        tracklet_producer_revision=mapping["tracklet_producer_revision"],
        association_revision=mapping["association_revision"],
        candidates=tuple(
            _candidate(item, index=index) for index, item in enumerate(candidates)
        ),
        metadata=_mapping(mapping["metadata"], name="metadata"),
    )


def _mixture_summary(mixture: MaterialIdentityMixtureV1) -> dict[str, object]:
    return {
        "mixture_id": mixture.mixture_id,
        "target_endpoint": mixture.target_endpoint.to_dict(),
        "window_order": list(mixture.window_order),
        "causal_frame_stop": mixture.causal_frame_stop,
        "candidate_ids": list(mixture.candidate_ids),
        "probabilities": mixture.probabilities.tolist(),
        "null_probability": mixture.null_probability,
        "identity_entropy_nats": mixture.identity_entropy_nats,
        "effective_hypothesis_count": mixture.effective_hypothesis_count,
    }


def _stream_summary(path: Path) -> dict[str, object]:
    stream = load_material_identity_stream(path)
    return {
        "artifact_id": stream.artifact_id,
        "sequence_id": stream.sequence_id,
        "case_id": stream.case_id,
        "stream_id": stream.stream_id,
        "source_repository": stream.source_repository,
        "source_revision": stream.source_revision,
        "admitted_window_ids": list(stream.admitted_window_ids),
        "causal_frame_stop": stream.causal_frame_stop,
        "update_count": len(stream.updates),
        "hypothesis_count": stream.hypothesis_count,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def _build(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d identity build-mixture",
        description=(
            "seal one material-identity mixture from externally source-calibrated "
            "candidate log weights"
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(arguments)
    mixture = mixture_from_config(
        _load_json(parsed.config, name="material-identity mixture configuration")
    )
    write_material_identity_mixture(
        parsed.output,
        mixture,
        overwrite=parsed.overwrite,
    )
    _print_json(_mixture_summary(mixture))
    return 0


def _validate_mixture(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d identity validate-mixture",
        description="strictly validate one portable material-identity mixture",
    )
    parser.add_argument("mixture", type=Path)
    parsed = parser.parse_args(arguments)
    _print_json(_mixture_summary(load_material_identity_mixture(parsed.mixture)))
    return 0


def _validate_stream(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d identity validate-stream",
        description="strictly validate one append-only material-identity stream",
    )
    parser.add_argument("stream", type=Path)
    parsed = parser.parse_args(arguments)
    _print_json(_stream_summary(parsed.stream))
    return 0


def _marginalize(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d identity marginalize",
        description="marginalize downstream log likelihoods over one identity mixture",
    )
    parser.add_argument("mixture", type=Path)
    parser.add_argument("likelihoods", type=Path)
    parsed = parser.parse_args(arguments)
    mixture = load_material_identity_mixture(parsed.mixture)
    value = _load_json(parsed.likelihoods, name="identity likelihood input")
    _exact_fields(
        value,
        {"candidate_ids", "log_likelihoods", "likelihood_power"},
        name="identity likelihood input",
    )
    candidate_ids = tuple(_list(value["candidate_ids"], name="candidate_ids"))
    log_likelihoods = np.asarray(
        _list(value["log_likelihoods"], name="log_likelihoods"),
        dtype=np.float64,
    )
    result = marginalize_identity_log_likelihoods(
        mixture,
        candidate_ids,
        log_likelihoods,
        likelihood_power=value["likelihood_power"],
    )
    _print_json(
        {
            "candidate_ids": list(result.candidate_ids),
            "log_marginal_likelihood": result.log_marginal_likelihood,
            "posterior_probabilities": result.posterior_probabilities.tolist(),
            "identity_entropy_nats": result.identity_entropy_nats,
            "effective_hypothesis_count": result.effective_hypothesis_count,
            "likelihood_power": result.likelihood_power,
            "semantics": result.semantics,
        }
    )
    return 0


def _moment_match(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d identity moment-match",
        description="moment-match Gaussian hypotheses with between-identity uncertainty",
    )
    parser.add_argument("mixture", type=Path)
    parser.add_argument("hypotheses", type=Path)
    parsed = parser.parse_args(arguments)
    mixture = load_material_identity_mixture(parsed.mixture)
    value = _load_json(parsed.hypotheses, name="identity Gaussian input")
    _exact_fields(
        value,
        {"candidate_ids", "means", "covariances", "probabilities"},
        name="identity Gaussian input",
    )
    candidate_ids = tuple(_list(value["candidate_ids"], name="candidate_ids"))
    probabilities_raw = value["probabilities"]
    probabilities = (
        None
        if probabilities_raw is None
        else np.asarray(_list(probabilities_raw, name="probabilities"), dtype=np.float64)
    )
    result = moment_match_gaussian_identity_hypotheses(
        mixture,
        candidate_ids,
        np.asarray(_list(value["means"], name="means"), dtype=np.float64),
        np.asarray(_list(value["covariances"], name="covariances"), dtype=np.float64),
        probabilities=probabilities,
    )
    _print_json(
        {
            "candidate_ids": list(result.candidate_ids),
            "probabilities": result.probabilities.tolist(),
            "mean": result.mean.tolist(),
            "covariance": result.covariance.tolist(),
            "within_hypothesis_covariance": (
                result.within_hypothesis_covariance.tolist()
            ),
            "between_hypothesis_covariance": (
                result.between_hypothesis_covariance.tolist()
            ),
            "identity_entropy_nats": result.identity_entropy_nats,
            "effective_hypothesis_count": result.effective_hypothesis_count,
            "semantics": result.semantics,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a grouped material-identity artifact or inference command."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="prob4d identity",
        description=__doc__,
    )
    parser.add_argument(
        "command",
        choices=(
            "build-mixture",
            "validate-mixture",
            "validate-stream",
            "marginalize",
            "moment-match",
        ),
        help=(
            "build or validate portable identity artifacts, or evaluate one "
            "frozen downstream marginalization input"
        ),
    )
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.command == "build-mixture":
        return _build(remaining)
    if parsed.command == "validate-mixture":
        return _validate_mixture(remaining)
    if parsed.command == "validate-stream":
        return _validate_stream(remaining)
    if parsed.command == "marginalize":
        return _marginalize(remaining)
    if parsed.command == "moment-match":
        return _moment_match(remaining)
    raise AssertionError("unreachable material-identity command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "mixture_from_config"]
