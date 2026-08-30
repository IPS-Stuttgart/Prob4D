#!/usr/bin/env python3
"""Bind the DOT provider workflow to the sealed persistent CUT3R runtime."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/dot-rope-cut3r-native-provider-v1.yml")
POLICY_TEST = Path("tests/test_trusted_self_hosted_validation_policy.py")


def replace_step(text: str, start: str, following: str, replacement: str) -> str:
    pattern = re.compile(
        rf"      - name: {re.escape(start)}\n.*?(?=      - name: {re.escape(following)}\n)",
        re.DOTALL,
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"workflow step changed: {start!r}")
    return updated


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    anchor = "  DATASET_ROOT: /mnt/seagate10tb/florianpfaff/datasets/dot\n"
    addition = anchor + """  CUT3R_RUNTIME_ROOT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1
  CUT3R_RUNTIME_CHECKOUT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/CUT3R
  CUT3R_RUNTIME_PYTHON: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/venv/bin/python
  CUT3R_RUNTIME_CHECKPOINT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/cut3r_512_dpt_4_64.pth
  CUT3R_RUNTIME_MANIFEST: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/bootstrap-manifest.json
  CUT3R_REVISION: 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
  CUT3R_CHECKPOINT_SHA256: 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103
"""
    if text.count(anchor) != 1:
        raise SystemExit("dataset-root anchor changed")
    text = text.replace(anchor, addition, 1)

    text = replace_step(
        text,
        "Select the retained CUT3R interpreter",
        "Copy and bind the native-RoPE runtime",
        """      - name: Bind the sealed persistent CUT3R runtime
        id: python
        shell: bash
        run: |
          set -euo pipefail
          test -x "$CUT3R_RUNTIME_PYTHON"
          test -d "$CUT3R_RUNTIME_CHECKOUT/.git"
          test -f "$CUT3R_RUNTIME_CHECKPOINT"
          test -f "$CUT3R_RUNTIME_MANIFEST"
          test "$(/usr/bin/git -C "$CUT3R_RUNTIME_CHECKOUT" rev-parse HEAD)" = \\
            "$CUT3R_REVISION"
          test "$(sha256sum "$CUT3R_RUNTIME_CHECKPOINT" | awk '{print $1}')" = \\
            "$CUT3R_CHECKPOINT_SHA256"
          MANIFEST="$CUT3R_RUNTIME_MANIFEST" \\
            EXPECTED_REVISION="$CUT3R_REVISION" \\
            EXPECTED_CHECKPOINT_SHA256="$CUT3R_CHECKPOINT_SHA256" \\
            EXPECTED_PYTHON="$CUT3R_RUNTIME_PYTHON" \\
            "$CUT3R_RUNTIME_PYTHON" - <<'PY'
          import json
          import os
          from pathlib import Path

          manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
          if manifest.get("cut3r_revision") != os.environ["EXPECTED_REVISION"]:
              raise SystemExit("runtime manifest binds another CUT3R revision")
          checkpoint = manifest.get("checkpoint") or {}
          if checkpoint.get("sha256") != os.environ["EXPECTED_CHECKPOINT_SHA256"]:
              raise SystemExit("runtime manifest binds another checkpoint")
          if manifest.get("runtime_python") != os.environ["EXPECTED_PYTHON"]:
              raise SystemExit("runtime manifest binds another interpreter")
          if not str(manifest.get("torch_version", "")).startswith("2.11.0"):
              raise SystemExit("runtime manifest binds another Torch version")
          if manifest.get("torch_cuda_version") != "12.6":
              raise SystemExit("runtime manifest binds another CUDA version")
          if manifest.get("cuda_available") is not True:
              raise SystemExit("runtime manifest does not attest CUDA")
          if manifest.get("dot_dataset_accessed") is not False:
              raise SystemExit("runtime bootstrap crossed the DOT data boundary")
          native_id = manifest.get("native_rope_artifact_id")
          if not isinstance(native_id, str) or len(native_id) != 64:
              raise SystemExit("runtime manifest has no native RoPE identity")
          PY
          PYTHONPATH="$CUT3R_RUNTIME_CHECKOUT:$CUT3R_RUNTIME_CHECKOUT/src" \\
            "$CUT3R_RUNTIME_PYTHON" - <<'PY'
          import torch
          from dust3r.model import ARCroco3DStereo

          print(f"dust3r_model_import={ARCroco3DStereo.__name__}")
          print(f"torch={torch.__version__}")
          print(f"torch_cuda={torch.version.cuda}")
          print(f"cuda_device={torch.cuda.get_device_name(0)}")
          if torch.version.cuda != "12.6" or not torch.cuda.is_available():
              raise SystemExit("persistent CUT3R runtime is not CUDA 12.6")
          PY
          echo "python=$CUT3R_RUNTIME_PYTHON" >> "$GITHUB_OUTPUT"
""",
    )

    text = replace_step(
        text,
        "Copy and bind the native-RoPE runtime",
        "Run synthetic native-RoPE forward pass",
        """      - name: Copy and bind the sealed native-RoPE runtime
        shell: bash
        run: |
          set -euo pipefail
          root="${{ steps.workspace.outputs.root }}"
          /usr/bin/mkdir -p "$root/CUT3R"
          /usr/bin/cp -a "$CUT3R_RUNTIME_CHECKOUT/." "$root/CUT3R/"
          PYTHONPATH="$GITHUB_WORKSPACE/src" \\
            "$CUT3R_RUNTIME_PYTHON" \\
            scripts/science/prepare_cut3r_runtime.py \\
              --cut3r-checkout "$root/CUT3R" \\
              --output "$root/provider/runtime-receipt.json"
          MANIFEST="$CUT3R_RUNTIME_MANIFEST" \\
            RECEIPT="$root/provider/runtime-receipt.json" \\
            "$CUT3R_RUNTIME_PYTHON" - <<'PY'
          import json
          import os
          from pathlib import Path

          manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
          receipt = json.loads(Path(os.environ["RECEIPT"]).read_text(encoding="utf-8"))
          if receipt.get("artifact_id") != manifest.get("native_rope_artifact_id"):
              raise SystemExit("copied native RoPE differs from sealed runtime")
          PY
          RECEIPT="$root/provider/runtime-receipt.json" \\
            "$CUT3R_RUNTIME_PYTHON" - <<'PY'
          import json
          import os
          import tarfile
          from pathlib import Path

          receipt = json.loads(Path(os.environ["RECEIPT"]).read_text(encoding="utf-8"))
          root = Path(os.environ["RECEIPT"]).parents[1]
          checkout = root / "CUT3R"
          relative = Path(receipt["extension"]["relative_path"])
          with tarfile.open(root / "provider" / "native-rope-extension.tar.gz", "w:gz") as archive:
              archive.add(checkout / relative, arcname=relative.as_posix(), recursive=False)
          PY
""",
    )

    variable = "          CUT3R_CHECKPOINT: ${{ vars.CUT3R_CHECKPOINT }}"
    fixed = (
        "          CUT3R_CHECKPOINT: "
        "/home/github-runner/.cache/prob4d/cut3r-runtime-v1/"
        "cut3r_512_dpt_4_64.pth"
    )
    if text.count(variable) != 2:
        raise SystemExit("checkpoint variable occurrence count changed")
    text = text.replace(variable, fixed)
    WORKFLOW.write_text(text, encoding="utf-8")

    tests = POLICY_TEST.read_text(encoding="utf-8")
    test_anchor = (
        '    assert "DATASET_ROOT: /mnt/seagate10tb/florianpfaff/datasets/dot" in text\n'
    )
    test_addition = test_anchor + """    assert "CUT3R_RUNTIME_ROOT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1" in text
    assert "/home/github-runner/.cache/prob4d/cut3r-runtime-v1/venv/bin/python" in text
    assert "/home/github-runner/.cache/prob4d/cut3r-runtime-v1/cut3r_512_dpt_4_64.pth" in text
    assert "vars.CUT3R_CHECKOUT" not in text
    assert "vars.CUT3R_CHECKPOINT" not in text
    assert 'scripts/science/prepare_cut3r_runtime.py \\\\' in text
    assert "--build" not in text[text.index("\\n  provider:"):text.index("\\n  evaluate:")]
"""
    if tests.count(test_anchor) != 1:
        raise SystemExit("self-hosted policy test anchor changed")
    tests = tests.replace(test_anchor, test_addition, 1)
    POLICY_TEST.write_text(tests, encoding="utf-8")


if __name__ == "__main__":
    main()
