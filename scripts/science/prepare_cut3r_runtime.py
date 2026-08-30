#!/usr/bin/env python3
"""Build or verify CUT3R's native RoPE kernel and emit an immutable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from prob4d.cut3r_runtime_contract import require_compiled_cut3r_rope

_TRUSTED_REQUEST_PATH = (
    "protocols/execution_requests/dot_rope_cut3r_sealed_runtime_v1.json"
)
_TRUSTED_CUT3R_REVISION = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
_TRUSTED_CHECKPOINT_SHA256 = (
    "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
)
_TRUSTED_MODEL_RELATIVE_PATH = Path("src/dust3r/model.py")
_TRUSTED_MODEL_GIT_BLOB_SHA1 = "7ed9f6106fb063686990c874ede99876ebc939ab"
_ORIGINAL_LOAD_CALL = '    ckpt = torch.load(model_path, map_location="cpu")\n'
_PATCHED_LOAD_CALL = (
    '    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)\n'
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run CUT3R's pinned curope setup.py before verification.",
    )
    parser.add_argument(
        "--expected-artifact-id",
        help="Fail unless the verified receipt matches this frozen artifact ID.",
    )
    return parser


def _write_no_clobber(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _content_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_trusted_checkpoint_compatibility(
    checkout: Path,
) -> dict[str, object] | None:
    """Patch one hash-pinned CUT3R loader in the isolated DOT runtime copy."""

    if os.environ.get("REQUEST_PATH") != _TRUSTED_REQUEST_PATH:
        return None
    if os.environ.get("CUT3R_REVISION") != _TRUSTED_CUT3R_REVISION:
        raise RuntimeError("trusted CUT3R checkpoint compatibility revision changed")
    if os.environ.get("CUT3R_CHECKPOINT_SHA256") != _TRUSTED_CHECKPOINT_SHA256:
        raise RuntimeError("trusted CUT3R checkpoint compatibility digest changed")

    checkpoint_text = os.environ.get("CUT3R_RUNTIME_CHECKPOINT")
    if not checkpoint_text:
        raise RuntimeError("trusted CUT3R checkpoint path is missing")
    checkpoint = Path(checkpoint_text).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise RuntimeError("trusted CUT3R checkpoint is not a regular file")
    if _sha256(checkpoint) != _TRUSTED_CHECKPOINT_SHA256:
        raise RuntimeError("trusted CUT3R checkpoint bytes changed")

    model_path = checkout / _TRUSTED_MODEL_RELATIVE_PATH
    if model_path.is_symlink():
        raise RuntimeError("trusted CUT3R model source must not be a symbolic link")
    resolved_model = model_path.resolve(strict=True)
    try:
        resolved_model.relative_to(checkout)
    except ValueError as error:
        raise RuntimeError("trusted CUT3R model source escapes the checkout") from error
    if not resolved_model.is_file():
        raise RuntimeError("trusted CUT3R model source is not a regular file")

    source = resolved_model.read_bytes()
    source_blob = _git_blob_sha1(source)
    if source_blob != _TRUSTED_MODEL_GIT_BLOB_SHA1:
        raise RuntimeError("trusted CUT3R model source bytes changed")
    text = source.decode("utf-8")
    if text.count(_ORIGINAL_LOAD_CALL) != 1 or _PATCHED_LOAD_CALL in text:
        raise RuntimeError("trusted CUT3R checkpoint load call changed")

    patched = text.replace(_ORIGINAL_LOAD_CALL, _PATCHED_LOAD_CALL, 1).encode("utf-8")
    resolved_model.write_bytes(patched)
    record: dict[str, object] = {
        "schema": "prob4d.cut3r-trusted-checkpoint-load-compatibility",
        "schema_version": 1,
        "status": "trusted-legacy-checkpoint-loader-enabled",
        "request_path": _TRUSTED_REQUEST_PATH,
        "cut3r_revision": _TRUSTED_CUT3R_REVISION,
        "source_member": _TRUSTED_MODEL_RELATIVE_PATH.as_posix(),
        "source_git_blob_sha1": source_blob,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": _TRUSTED_CHECKPOINT_SHA256,
        "torch_load_policy": "weights_only=False at the exact pinned CUT3R load_model call",
        "security_boundary": (
            "Legacy pickle loading is enabled only after the exact CUT3R revision, "
            "source blob, execution request, and checkpoint SHA-256 are verified."
        ),
    }
    record["artifact_id"] = _content_id(record)
    return record


def main() -> int:
    args = _parser().parse_args()
    checkout = args.cut3r_checkout.expanduser().resolve(strict=True)
    curope_root = checkout / "src/croco/models/curope"

    if args.build:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=curope_root,
            check=True,
        )

    receipt = require_compiled_cut3r_rope(checkout)
    if (
        args.expected_artifact_id is not None
        and receipt["artifact_id"] != args.expected_artifact_id
    ):
        raise SystemExit(
            "verified CUT3R runtime receipt differs from the frozen artifact ID"
        )
    compatibility = _prepare_trusted_checkpoint_compatibility(checkout)
    _write_no_clobber(args.output, receipt)
    if compatibility is not None:
        _write_no_clobber(
            args.output.with_name("checkpoint-load-compatibility.json"),
            compatibility,
        )
    print(receipt["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
