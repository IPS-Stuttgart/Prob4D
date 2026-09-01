#!/usr/bin/env python3
"""Build or verify CUT3R's native RoPE kernel and emit immutable receipts."""

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
_TRUSTED_HELDOUT_RECOVERY_REQUEST_PATH = (
    "protocols/execution_requests/"
    "dot_rope_cut3r_heldout_confirmation_gpuserver6000_v1.json"
)
_TRUSTED_CUT3R_REVISION = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
_TRUSTED_CHECKPOINT_SHA256 = (
    "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
)
_TRUSTED_CUROPE_PATCH_RELATIVE_PATH = Path(
    ".github/patches/cut3r-curope-torch211-cu126.patch"
)
_TRUSTED_CUROPE_PATCH_GIT_BLOB_SHA1 = "9127464c77b571b9586144cabe24a4eed8667db0"
_TRUSTED_CUROPE_KERNELS_RELATIVE_PATH = Path("src/croco/models/curope/kernels.cu")
_TRUSTED_CUROPE_KERNELS_GIT_BLOB_SHA1 = "7156cd1bb935cb1f0be45e58add53f9c21505c20"
_TRUSTED_CUROPE_SETUP_RELATIVE_PATH = Path("src/croco/models/curope/setup.py")
_TRUSTED_CUROPE_SETUP_GIT_BLOB_SHA1 = "02ddb0912370a67a49fd2bb91164cf2f1da8648e"
_PATCHED_DISPATCH_CALL = (
    'AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda"'
)
_PATCHED_SM89_TARGET = 'all_cuda_archs = ["-gencode", "arch=compute_89,code=sm_89"]'
_TRUSTED_MODEL_RELATIVE_PATH = Path("src/dust3r/model.py")
_TRUSTED_MODEL_GIT_BLOB_SHA1 = "7ed9f6106fb063686990c874ede99876ebc939ab"
_ORIGINAL_LOAD_CALL = '    ckpt = torch.load(model_path, map_location="cpu")\n'
_PATCHED_LOAD_CALL = (
    '    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)\n'
)
_TRUSTED_PROVIDER_SCRIPT_RELATIVE_PATH = Path(
    "scripts/science/run_dot_rope_cut3r_native_provider.py"
)
_TRUSTED_PROVIDER_SCRIPT_GIT_BLOB_SHA1 = "612c8ae61b0a64d464256a11992b46c486c88012"
_ORIGINAL_SMOKE_FRAME_CALL = (
    "            frame_paths = _make_synthetic_frames(Path(temporary), count=3)\n"
)
_PATCHED_SMOKE_FRAME_CALL = (
    '            frame_paths = _make_synthetic_frames(Path(temporary) / "frames", count=3)\n'
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


def _trusted_request_selected() -> bool:
    return os.environ.get("REQUEST_PATH") == _TRUSTED_REQUEST_PATH


def _trusted_heldout_recovery_selected() -> bool:
    return os.environ.get("REQUEST_PATH") == _TRUSTED_HELDOUT_RECOVERY_REQUEST_PATH


def _require_trusted_runtime_identity() -> None:
    if os.environ.get("CUT3R_REVISION") != _TRUSTED_CUT3R_REVISION:
        raise RuntimeError("trusted CUT3R compatibility revision changed")
    if os.environ.get("CUT3R_CHECKPOINT_SHA256") != _TRUSTED_CHECKPOINT_SHA256:
        raise RuntimeError("trusted CUT3R compatibility checkpoint digest changed")


def _resolved_regular_member(root: Path, relative: Path, *, name: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError(f"trusted {name} must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"trusted {name} escapes its root") from error
    if not resolved.is_file():
        raise RuntimeError(f"trusted {name} is not a regular file")
    return resolved


def _prepare_trusted_curope_compatibility(
    checkout: Path,
    repository_root: Path,
) -> dict[str, object] | None:
    """Apply the exact reviewed PyTorch 2.11/SM89 curope compatibility patch."""

    if not _trusted_heldout_recovery_selected():
        return None
    _require_trusted_runtime_identity()

    checkout = checkout.expanduser().resolve(strict=True)
    repository_root = repository_root.expanduser().resolve(strict=True)
    patch_path = _resolved_regular_member(
        repository_root,
        _TRUSTED_CUROPE_PATCH_RELATIVE_PATH,
        name="curope compatibility patch",
    )
    patch_bytes = patch_path.read_bytes()
    patch_blob = _git_blob_sha1(patch_bytes)
    if patch_blob != _TRUSTED_CUROPE_PATCH_GIT_BLOB_SHA1:
        raise RuntimeError("trusted curope compatibility patch bytes changed")

    source_members = (
        (
            "kernels",
            _TRUSTED_CUROPE_KERNELS_RELATIVE_PATH,
            _TRUSTED_CUROPE_KERNELS_GIT_BLOB_SHA1,
        ),
        (
            "setup",
            _TRUSTED_CUROPE_SETUP_RELATIVE_PATH,
            _TRUSTED_CUROPE_SETUP_GIT_BLOB_SHA1,
        ),
    )
    before: dict[str, dict[str, str]] = {}
    resolved_members: dict[str, Path] = {}
    for name, relative, expected_blob in source_members:
        member = _resolved_regular_member(checkout, relative, name=f"curope {name} source")
        source = member.read_bytes()
        source_blob = _git_blob_sha1(source)
        if source_blob != expected_blob:
            raise RuntimeError(f"trusted curope {name} source bytes changed")
        before[name] = {
            "path": relative.as_posix(),
            "git_blob_sha1": source_blob,
            "sha256": hashlib.sha256(source).hexdigest(),
        }
        resolved_members[name] = member

    subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=checkout,
        check=True,
    )

    kernels = resolved_members["kernels"].read_text(encoding="utf-8")
    setup = resolved_members["setup"].read_text(encoding="utf-8")
    if _PATCHED_DISPATCH_CALL not in kernels or "tokens.type()" in kernels:
        raise RuntimeError("trusted curope dispatch compatibility was not applied")
    if _PATCHED_SM89_TARGET not in setup or "cuda.get_gencode_flags()" in setup:
        raise RuntimeError("trusted curope SM89 build target was not applied")

    after = {
        name: {
            "path": relative.as_posix(),
            "git_blob_sha1": _git_blob_sha1(resolved_members[name].read_bytes()),
            "sha256": _sha256(resolved_members[name]),
        }
        for name, relative, _ in source_members
    }
    record: dict[str, object] = {
        "schema": "prob4d.cut3r-curope-pytorch211-sm89-compatibility",
        "schema_version": 1,
        "status": "trusted-curope-pytorch211-sm89-patch-applied",
        "request_path": _TRUSTED_HELDOUT_RECOVERY_REQUEST_PATH,
        "cut3r_revision": _TRUSTED_CUT3R_REVISION,
        "checkpoint_sha256": _TRUSTED_CHECKPOINT_SHA256,
        "patch": {
            "path": _TRUSTED_CUROPE_PATCH_RELATIVE_PATH.as_posix(),
            "git_blob_sha1": patch_blob,
            "sha256": hashlib.sha256(patch_bytes).hexdigest(),
        },
        "members_before": before,
        "members_after": after,
        "compatibility_scope": (
            "Replace the removed Tensor.type() dispatch API and compile CUT3R curope "
            "only for the registered NVIDIA Ada SM89 provider device."
        ),
        "scientific_boundary": (
            "The patch changes only native extension compilation compatibility; it "
            "does not change the checkpoint, provider weights, images, source "
            "calibration, alpha, query definition, cohort, comparator, statistic, or "
            "decision rule."
        ),
    }
    record["artifact_id"] = _content_id(record)
    return record


def _prepare_trusted_checkpoint_compatibility(
    checkout: Path,
) -> dict[str, object] | None:
    """Patch one hash-pinned CUT3R loader in the isolated DOT runtime copy."""

    if not _trusted_request_selected():
        return None
    _require_trusted_runtime_identity()

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


def _prepare_trusted_smoke_workspace_compatibility(
    repository_root: Path,
) -> dict[str, object] | None:
    """Patch the registered pre-data smoke workspace call in the checked-out source."""

    if not _trusted_request_selected():
        return None
    _require_trusted_runtime_identity()

    root = repository_root.expanduser().resolve(strict=True)
    source_path = root / _TRUSTED_PROVIDER_SCRIPT_RELATIVE_PATH
    if source_path.is_symlink():
        raise RuntimeError("trusted DOT provider source must not be a symbolic link")
    resolved_source = source_path.resolve(strict=True)
    try:
        resolved_source.relative_to(root)
    except ValueError as error:
        raise RuntimeError("trusted DOT provider source escapes the checkout") from error
    if not resolved_source.is_file():
        raise RuntimeError("trusted DOT provider source is not a regular file")

    source = resolved_source.read_bytes()
    source_blob = _git_blob_sha1(source)
    if source_blob != _TRUSTED_PROVIDER_SCRIPT_GIT_BLOB_SHA1:
        raise RuntimeError("trusted DOT provider source bytes changed")
    text = source.decode("utf-8")
    if text.count(_ORIGINAL_SMOKE_FRAME_CALL) != 1 or _PATCHED_SMOKE_FRAME_CALL in text:
        raise RuntimeError("trusted DOT runtime-smoke workspace call changed")

    patched = text.replace(
        _ORIGINAL_SMOKE_FRAME_CALL,
        _PATCHED_SMOKE_FRAME_CALL,
        1,
    ).encode("utf-8")
    resolved_source.write_bytes(patched)
    record: dict[str, object] = {
        "schema": "prob4d.dot-cut3r-runtime-smoke-workspace-compatibility",
        "schema_version": 1,
        "status": "trusted-smoke-child-workspace-enabled",
        "request_path": _TRUSTED_REQUEST_PATH,
        "cut3r_revision": _TRUSTED_CUT3R_REVISION,
        "checkpoint_sha256": _TRUSTED_CHECKPOINT_SHA256,
        "source_member": _TRUSTED_PROVIDER_SCRIPT_RELATIVE_PATH.as_posix(),
        "source_git_blob_sha1": source_blob,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "workspace_policy": (
            "TemporaryDirectory owns the parent; synthetic frames are written to "
            "one newly created child directory."
        ),
        "information_boundary": (
            "The repair changes only the dataset-free runtime smoke workspace and "
            "does not open DOT normal-view images or marker payloads."
        ),
    }
    record["artifact_id"] = _content_id(record)
    return record


def main() -> int:
    args = _parser().parse_args()
    checkout = args.cut3r_checkout.expanduser().resolve(strict=True)
    repository_root = Path(__file__).resolve().parents[2]
    curope_root = checkout / "src/croco/models/curope"

    curope_compatibility = _prepare_trusted_curope_compatibility(
        checkout,
        repository_root,
    )
    if curope_compatibility is not None:
        _write_no_clobber(
            args.output.with_name("curope-compatibility.json"),
            curope_compatibility,
        )

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
    checkpoint_compatibility = _prepare_trusted_checkpoint_compatibility(checkout)
    smoke_compatibility = _prepare_trusted_smoke_workspace_compatibility(repository_root)
    _write_no_clobber(args.output, receipt)
    if checkpoint_compatibility is not None:
        _write_no_clobber(
            args.output.with_name("checkpoint-load-compatibility.json"),
            checkpoint_compatibility,
        )
    if smoke_compatibility is not None:
        _write_no_clobber(
            args.output.with_name("runtime-smoke-compatibility.json"),
            smoke_compatibility,
        )
    print(receipt["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
