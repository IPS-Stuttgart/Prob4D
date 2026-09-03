"""Optional, causal RGB execution against a pinned official CUT3R checkout.

Torch is deliberately imported only when constructing the runtime. The NumPy
importers and the rest of Prob4D do not acquire a GPU dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ._strict_json import require_sha256
from .cut3r_runtime_contract import require_compiled_cut3r_rope

SUPPORTED_CUT3R_REVISION = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
OFFICIAL_512_CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"

# Reviewed build-only variant: modern ATen scalar_type(), and SM89 compilation.
# No model, preprocessing, state-update, or prediction source is changed.
NATIVE_BUILD_COMPATIBILITY_SHA256 = {
    "src/croco/models/curope/kernels.cu": (
        "0ae1a9517eba744fd619f98b3761766d384ab090d35df4314d8946db72c5ca51"
    ),
    "src/croco/models/curope/setup.py": (
        "aaec74fff578fe13ba17a90506e9c70fd08abe03e5a468dfb2f6162cda0e0c0a"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordinary_path(path: Path) -> Path:
    """Resolve an existing path without accepting symlinked ancestors."""
    path = path.expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("CUT3R input/runtime paths must not traverse symbolic links")
    return path.resolve(strict=True)


def verify_checkout(
    checkout: Path,
    revision: str,
    *,
    allow_native_build_compatibility: bool = False,
) -> dict[str, str]:
    if revision != SUPPORTED_CUT3R_REVISION:
        raise ValueError("unsupported CUT3R revision; qualify the changed RGB interface first")
    root = ordinary_path(checkout)

    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        ).stdout

    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("CUT3R checkout HEAD differs from the requested revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        changed = git("diff", "HEAD", "--name-only", "-z").decode().strip("\0").split("\0")
        if not allow_native_build_compatibility or any(
            name not in NATIVE_BUILD_COMPATIBILITY_SHA256
            or file_sha256(ordinary_path(root / name)) != NATIVE_BUILD_COMPATIBILITY_SHA256[name]
            for name in changed
        ):
            raise ValueError(
                "CUT3R tracked source is dirty outside the reviewed native build variant"
            )
    # Also bind untracked Python modules: an earlier runtime repair may shadow
    # the compiled kernel or an upstream import without changing tracked files.
    sources = {}
    for directory in (root / "src",):
        for path in sorted(directory.rglob("*.py")):
            if path.is_symlink():
                raise ValueError("CUT3R Python source must not be symlinked")
            sources[path.relative_to(root).as_posix()] = file_sha256(ordinary_path(path))
    if "src/dust3r/model.py" not in sources:
        raise ValueError("CUT3R model source missing")
    return sources


class Cut3RNativeRuntime:
    """One recurrent state per sequence, one RGB frame resident at a time.

    The official ``inference_step`` is a ray-map query API, not an RGB state update.
    We instead reuse the official RGB encoder and recurrent decoder step. No
    revisiting, global alignment, or sequence-wide image encoding is performed.
    """

    def __init__(
        self,
        checkout: Path,
        checkpoint: Path,
        *,
        checkpoint_sha256: str,
        revision: str = SUPPORTED_CUT3R_REVISION,
        device: str = "cuda:0",
        image_size: int = 512,
        seed: int = 42,
        allow_native_build_compatibility: bool = False,
    ) -> None:
        if image_size not in (224, 512):
            raise ValueError("CUT3R image_size must be 224 or 512")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        expected = require_sha256(checkpoint_sha256, name="trusted checkpoint SHA-256")
        checkpoint = ordinary_path(checkpoint)
        if not checkpoint.is_file() or file_sha256(checkpoint) != expected:
            raise ValueError("CUT3R checkpoint SHA-256 mismatch")
        checkout = ordinary_path(checkout)
        sources = verify_checkout(
            checkout,
            revision,
            allow_native_build_compatibility=allow_native_build_compatibility,
        )
        for name in ("dust3r", "src.dust3r", "models"):
            if name in sys.modules:
                raise RuntimeError("start a fresh process before initializing native CUT3R")
        for path in (checkout, checkout / "src"):
            sys.path.insert(0, str(path))
        try:
            rope = require_compiled_cut3r_rope(checkout)
            torch = importlib.import_module("torch")
            model_module = importlib.import_module("dust3r.model")
            image_module = importlib.import_module("dust3r.utils.image")
            camera_module = importlib.import_module("dust3r.utils.camera")
            post_module = importlib.import_module("dust3r.post_process")
        except ImportError as error:
            raise RuntimeError(
                "CUT3R runtime dependencies missing; see docs/cut3r-native-inference.md"
            ) from error
        selected_device = torch.device(device)
        if selected_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("native CUT3R requires CUDA with compiled RoPE")
        self.torch: Any = torch
        self.device = selected_device
        self.image_size = image_size
        self.seed = seed
        self.load_images = image_module.load_images
        self.decode_pose = camera_module.pose_encoding_to_camera
        self.estimate_focal = post_module.estimate_focal_knowing_depth
        self.reset()
        # Official checkpoints contain argparse/config objects and load_model
        # evaluates their architecture. Only use trusted, digest-verified weights.
        # Scope modern Torch compatibility to loading; do not patch upstream files.
        previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            self.model = model_module.load_model(str(checkpoint), "cpu", verbose=False)
        finally:
            if previous is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
        head = self.model.head_type
        if (head == "dpt" and image_size != 512) or (head == "linear" and image_size != 224):
            raise ValueError("CUT3R checkpoint head and image_size disagree")
        self.model = self.model.to(selected_device).eval()
        self.identity: dict[str, Any] = {
            "cut3r_revision": revision,
            "checkpoint_sha256": expected,
            "checkpoint_byte_count": checkpoint.stat().st_size,
            "source_sha256s": sources,
            "native_rope": rope,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": str(selected_device),
            "image_size": image_size,
            "seed": seed,
            "execution_mode": "rgb-recurrent-stream-v1",
            "revisit_count": 1,
            "global_alignment": False,
            "sequence_wide_encoding": False,
            "native_build_compatibility_permitted": allow_native_build_compatibility,
        }

    def reset(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.torch.manual_seed(self.seed)
        self.torch.cuda.manual_seed_all(self.seed)
        self.torch.backends.cudnn.benchmark = False
        self.state: tuple[Any, ...] | None = None
        self.frame_count = 0

    def prepare_view(self, path: Path) -> dict[str, Any]:
        torch = self.torch
        image = self.load_images([str(path)], size=self.image_size, verbose=False)[0]
        tensor = image["img"].to(self.device)
        return {
            "img": tensor,
            "true_shape": torch.as_tensor(image["true_shape"], device=self.device),
            "img_mask": torch.ones(1, dtype=torch.bool, device=self.device),
            "ray_mask": torch.zeros(1, dtype=torch.bool, device=self.device),
            "update": torch.ones(1, dtype=torch.bool, device=self.device),
            "reset": torch.zeros(1, dtype=torch.bool, device=self.device),
        }

    def step(self, path: Path) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            view = self.prepare_view(path)
            features, positions, _ = self.model._encode_image(view["img"], view["true_shape"])
            feature = features[-1]
            first = self.state is None
            if first:
                state_feat, state_pos = self.model._init_state(feature, positions)
                memory = self.model.pose_retriever.mem.expand(feature.shape[0], -1, -1)
                self.state = (state_feat, state_pos, state_feat.clone(), memory, memory.clone())
            assert self.state is not None
            state_feat, state_pos, initial_feat, memory, initial_mem = self.state
            # The upstream index selects the initial pose token versus a memory
            # query; only views[i] is accessed. Keep this container constant-size.
            prediction, (state_feat, memory) = self.model._forward_decoder_step(
                [view, view],
                int(not first),
                feature,
                positions,
                view["true_shape"],
                initial_feat,
                initial_mem,
                state_feat,
                state_pos,
                memory,
            )
            self.state = (state_feat, state_pos, initial_feat, memory, initial_mem)
            self.frame_count += 1
            return self.decode_prediction(prediction)

    def decode_prediction(self, prediction: dict[str, Any]) -> dict[str, np.ndarray]:
        torch = self.torch
        points = prediction["pts3d_in_self_view"].detach().float().cpu()
        confidence = prediction["conf_self"].detach().float().cpu()
        poses = self.decode_pose(prediction["camera_pose"].clone()).detach().cpu()
        _, height, width, _ = points.shape
        principal = torch.tensor([[width // 2, height // 2]], dtype=points.dtype)
        focal = self.estimate_focal(points, principal, focal_mode="weiszfeld")
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = intrinsics[1, 1] = focal[0]
        intrinsics[:2, 2] = principal[0]
        return {
            "points": points[0].numpy(),
            "confidence": confidence[0].numpy(),
            "pose": poses[0].numpy().astype(np.float64),
            "intrinsics": intrinsics.numpy().astype(np.float64),
        }
