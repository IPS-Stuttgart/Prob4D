"""Runtime source-revision attestation for versioned provider exports."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal

from ._build_identity import BUILD_IDENTITY_FILENAME, load_build_identity

RevisionSource = Literal[
    "installed_vcs_metadata",
    "source_checkout",
    "deployment_environment",
    "unavailable",
]


@dataclass(frozen=True)
class RuntimeRevisionAttestation:
    """Comparison between a declared provider revision and the executing package."""

    expected_revision: str
    observed_revision: str | None
    source: RevisionSource
    clean_checkout: bool | None
    matched: bool
    independently_verified: bool

    def as_metadata(self) -> dict[str, object]:
        """Return a finite JSON-compatible provenance record."""

        return {
            "expected_revision": self.expected_revision,
            "observed_revision": self.observed_revision,
            "source": self.source,
            "clean_checkout": self.clean_checkout,
            "matched": self.matched,
            "independently_verified": self.independently_verified,
        }


def _validated_revision(value: str, *, name: str) -> str:
    revision = str(value)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase 40- or 64-character Git commit")
    return revision


def _build_identity_path() -> Path:
    return Path(__file__).resolve().with_name(BUILD_IDENTITY_FILENAME)


def _installed_build_identity_revision() -> str | None:
    identity_path = _build_identity_path()
    if not identity_path.is_file():
        return None
    try:
        identity = load_build_identity(
            identity_path,
            package_root=identity_path.parent,
        )
    except ValueError as error:
        raise RuntimeError("installed Prob4D build identity is invalid") from error
    revision = identity["source_revision"]
    clean = identity["source_tree_clean"]
    if revision is None or clean is not True:
        raise RuntimeError(
            "installed Prob4D build identity does not attest a clean source revision"
        )
    return _validated_revision(str(revision), name="installed build revision")


def _installed_direct_url_revision() -> str | None:
    try:
        direct_url = distribution("prob4d").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    if not commit_id:
        return None
    return _validated_revision(str(commit_id), name="installed VCS revision")


def _installed_vcs_revision() -> str | None:
    """Resolve content-verified build metadata before legacy PEP 610 metadata."""

    embedded = _installed_build_identity_revision()
    if embedded is not None:
        return embedded
    return _installed_direct_url_revision()


def _source_checkout_revision(
    checkout_root: Path,
) -> tuple[str, bool] | None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(checkout_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return (
        _validated_revision(revision, name="source-checkout revision"),
        not bool(status.strip()),
    )


def _resolve_runtime_revision(
    *,
    checkout_root: Path | None = None,
) -> tuple[str | None, RevisionSource, bool | None]:
    installed = _installed_vcs_revision()
    if installed is not None:
        return installed, "installed_vcs_metadata", None

    root = Path(checkout_root) if checkout_root is not None else Path(__file__).resolve().parents[2]
    checkout = _source_checkout_revision(root)
    if checkout is not None:
        revision, clean = checkout
        return revision, "source_checkout", clean

    environment = os.environ.get("PROB4D_RUNTIME_REVISION")
    if environment:
        return (
            _validated_revision(environment, name="PROB4D_RUNTIME_REVISION"),
            "deployment_environment",
            None,
        )
    return None, "unavailable", None


def inspect_runtime_revision(
    expected_revision: str,
    *,
    checkout_root: Path | None = None,
) -> RuntimeRevisionAttestation:
    """Inspect available runtime provenance without requiring a successful match.

    Exploratory callers may record an environment-supplied deployment assertion,
    but it is deliberately not labelled independent verification.
    """

    expected = _validated_revision(expected_revision, name="expected revision")
    observed, source, clean = _resolve_runtime_revision(checkout_root=checkout_root)
    matched = observed == expected
    independently_verified = matched and source in {
        "installed_vcs_metadata",
        "source_checkout",
    }
    if clean is False:
        independently_verified = False
    return RuntimeRevisionAttestation(
        expected_revision=expected,
        observed_revision=observed,
        source=source,
        clean_checkout=clean,
        matched=matched,
        independently_verified=independently_verified,
    )


def assert_runtime_revision(
    expected_revision: str,
    *,
    checkout_root: Path | None = None,
) -> RuntimeRevisionAttestation:
    """Fail closed unless the executing package independently matches the commit.

    Claim-bearing export accepts a content-verified installed build identity,
    legacy PEP 610 VCS installation metadata, or a completely clean source
    checkout. ``PROB4D_RUNTIME_REVISION`` can describe an exploratory packaged
    deployment, but an unauthenticated environment variable cannot establish which
    code bytes are executing and therefore never satisfies this function.
    """

    attestation = inspect_runtime_revision(
        expected_revision,
        checkout_root=checkout_root,
    )
    if attestation.observed_revision is None:
        raise RuntimeError(
            "claim-bearing export cannot determine the executing Prob4D revision; "
            "install a content-identified release/VCS artifact or run from a clean "
            "source checkout"
        )
    if not attestation.matched:
        raise RuntimeError(
            "claim-bearing export revision mismatch: expected "
            f"{attestation.expected_revision}, observed "
            f"{attestation.observed_revision} from {attestation.source}"
        )
    if attestation.clean_checkout is False:
        raise RuntimeError(
            "claim-bearing export refuses a source checkout with tracked or untracked modifications"
        )
    if not attestation.independently_verified:
        raise RuntimeError(
            "claim-bearing export requires independent VCS revision evidence; "
            f"{attestation.source} is recordable only for exploratory export"
        )
    return attestation


__all__ = [
    "RevisionSource",
    "RuntimeRevisionAttestation",
    "assert_runtime_revision",
    "inspect_runtime_revision",
]
