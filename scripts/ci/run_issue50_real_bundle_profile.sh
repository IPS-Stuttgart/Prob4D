#!/usr/bin/env bash
set -euo pipefail

readonly ESTIMATOR_REVISION="bfe1b1b87323751fe9013f8d667fa0bb5f605a69"
readonly MOTIONCRAFTER_REVISION="9cb4e9679f5f34e249945544052464ef46324bc2"
readonly SOURCE_VIDEO="/mnt/lexar4tb/datasets/deform360/contact-trust-calibration-v1/002-rope-silk-ep0008/episode_0000/brics-odroid-007_cam0/undistorted.mp4"
readonly LOCK_PATH="protocols/issue50-real-bundle-profile-v1.json"

lock_id="$(python scripts/ci/profile_real_prediction_bundle.py verify-lock --lock "${LOCK_PATH}")"
root="/mnt/lexar4tb/prob4d/issue50-real-bundle-profile-v1"
run_root="${root}/${lock_id}"
upstream="${root}/MotionCrafter-${MOTIONCRAFTER_REVISION}"
environment="${root}/env-bfe1b1b8-py312-torch211-cu128"
bundle="${run_root}/prediction-bundle"
store="${run_root}/prediction-store-float32"
work="${RUNNER_TEMP}/issue50-real-profile"
evidence="${work}/evidence"

rm -rf "${work}"
mkdir -p "${evidence}"

seal_evidence() {
  local exit_status=$?
  set +e
  {
    echo "repository=${GITHUB_REPOSITORY:-unknown}"
    echo "execution_revision=${GITHUB_SHA:-unknown}"
    echo "run_id=${GITHUB_RUN_ID:-unknown}"
    echo "runner_name=${RUNNER_NAME:-unknown}"
    echo "runner_os=${RUNNER_OS:-unknown}"
    echo "runner_arch=${RUNNER_ARCH:-unknown}"
    echo "script_exit_status=${exit_status}"
    echo
    uname -a
    echo
    lscpu
    echo
    free -b
    echo
    nvidia-smi
    echo
    df -T "${SOURCE_VIDEO}" "${run_root}"
    echo
    findmnt -T "${run_root}"
  } > "${evidence}/host-and-filesystem.txt" 2>&1
  python - <<'PY' > "${evidence}/numpy-config.txt" 2>&1
import numpy

print("numpy", numpy.__version__)
numpy.show_config()
PY
  (
    cd "${evidence}" || exit 0
    find . -type f ! -name SHA256SUMS -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum \
      > SHA256SUMS
    sha256sum --check --strict SHA256SUMS
  )
  return "${exit_status}"
}
trap seal_evidence EXIT

cp "${LOCK_PATH}" "${evidence}/profile-lock.json"
printf '%s\n' "${lock_id}" > "${evidence}/profile-lock-id.txt"
git rev-parse HEAD > "${evidence}/execution-revision.txt"
git rev-parse "${ESTIMATOR_REVISION}" > "${evidence}/estimator-base-revision.txt"
git diff --exit-code "${ESTIMATOR_REVISION}" HEAD -- \
  src/prob4d \
  pyproject.toml \
  environments/motioncrafter-inference.txt \
  scripts/bootstrap_motioncrafter_env.sh
git diff --name-only "${ESTIMATOR_REVISION}" HEAD \
  | LC_ALL=C sort \
  > "${evidence}/reviewed-paths.txt"
python -m py_compile scripts/ci/profile_real_prediction_bundle.py

python - "${LOCK_PATH}" "${evidence}/input-verification.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


source = Path(lock["source_video"]["path"])
stat = source.stat()
if source.is_symlink() or not source.is_file():
    raise SystemExit("frozen source is not a regular file")
if stat.st_size != lock["source_video"]["bytes"]:
    raise SystemExit("frozen source byte count changed")
if sha256(source) != lock["source_video"]["sha256"]:
    raise SystemExit("frozen source SHA-256 changed")
forbidden = (
    "adaptive-confirmation",
    "heldout",
    "holdout",
    "prospective",
    "reserved",
    "/target/",
)
if any(token in str(source).lower() for token in forbidden):
    raise SystemExit("frozen source entered a protected path")
probe = subprocess.run(
    [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(source),
    ],
    check=True,
    capture_output=True,
    text=True,
)
raw = json.loads(probe.stdout)
stream = raw["streams"][0]
observed = {
    "codec_name": stream["codec_name"],
    "width": int(stream["width"]),
    "height": int(stream["height"]),
    "avg_frame_rate": stream["avg_frame_rate"],
    "frame_count": int(stream["nb_frames"]),
    "duration_seconds": float(raw["format"]["duration"]),
}
if observed != lock["source_video"]["container"]:
    raise SystemExit(f"source container changed: {observed!r}")

models = {}
for role, expected in lock["model_cache"]["models"].items():
    snapshot = Path(expected["snapshot_path"])
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise SystemExit(f"{role} snapshot unavailable")
    digest = hashlib.sha256()
    logical_bytes = 0
    file_count = 0
    for member in sorted(
        snapshot.rglob("*"),
        key=lambda value: value.relative_to(snapshot).as_posix(),
    ):
        if member.is_dir():
            continue
        relative = member.relative_to(snapshot).as_posix()
        member_stat = member.stat()
        target = os.readlink(member) if member.is_symlink() else ""
        digest.update(relative.encode() + b"\0")
        digest.update(str(member_stat.st_size).encode() + b"\0")
        digest.update(target.encode() + b"\0")
        logical_bytes += member_stat.st_size
        file_count += 1
    if digest.hexdigest() != expected["layout_sha256"]:
        raise SystemExit(f"{role} snapshot layout changed")
    if logical_bytes != expected["logical_bytes"]:
        raise SystemExit(f"{role} snapshot logical bytes changed")
    models[role] = {
        "revision": expected["revision"],
        "path": str(snapshot),
        "layout_sha256": digest.hexdigest(),
        "logical_bytes": logical_bytes,
        "file_count": file_count,
        "model_tensor_bytes_opened": False,
    }
report = {
    "schema": "prob4d.issue50-frozen-input-verification",
    "version": 1,
    "source_video": {
        "path": str(source),
        "sha256": lock["source_video"]["sha256"],
        "bytes": stat.st_size,
        "container": observed,
    },
    "models": models,
    "information_boundary": {
        "source_video_bytes_hashed": True,
        "source_video_frames_decoded": False,
        "model_tensor_bytes_opened": False,
        "truth_or_target_outcomes_opened": False,
    },
}
output.write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader,nounits \
  | tee "${evidence}/gpu.csv"

if [[ ! -d "${upstream}/.git" ]]; then
  rm -rf "${upstream}"
  git clone https://github.com/TencentARC/MotionCrafter.git "${upstream}"
fi
if ! git -C "${upstream}" cat-file -e "${MOTIONCRAFTER_REVISION}^{commit}" \
    2>/dev/null; then
  git -C "${upstream}" fetch origin "${MOTIONCRAFTER_REVISION}"
fi
git -C "${upstream}" checkout --detach "${MOTIONCRAFTER_REVISION}"
git -C "${upstream}" reset --hard "${MOTIONCRAFTER_REVISION}"
git -C "${upstream}" clean -ffdx
test "$(git -C "${upstream}" rev-parse HEAD)" = "${MOTIONCRAFTER_REVISION}"
test -z "$(git -C "${upstream}" status --porcelain=v1 --untracked-files=all)"
origin="$(git -C "${upstream}" remote get-url origin)"
case "${origin}" in
  https://github.com/TencentARC/MotionCrafter.git|https://github.com/TencentARC/MotionCrafter)
    ;;
  *)
    echo "unexpected MotionCrafter origin: ${origin}" >&2
    exit 1
    ;;
esac
printf 'revision=%s\norigin=%s\nclean=true\n' \
  "${MOTIONCRAFTER_REVISION}" "${origin}" \
  > "${evidence}/motioncrafter-checkout.txt"

bash scripts/bootstrap_motioncrafter_env.sh "${upstream}" "${environment}" \
  2>&1 | tee "${evidence}/environment-bootstrap.log"
uv_bin="$(command -v uv || true)"
if [[ -z "${uv_bin}" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
fi
test -x "${uv_bin}"
set +e
"${uv_bin}" pip check --python "${environment}/bin/python" \
  > "${evidence}/environment-check.txt" 2>&1
uv_check_status=$?
set -e
cat "${evidence}/environment-check.txt"
if (( uv_check_status != 0 )); then
  expected_count="$(grep -Fc 'The package `decord` was built for a different platform' \
    "${evidence}/environment-check.txt")"
  package_count="$(grep -c '^The package `' "${evidence}/environment-check.txt" || true)"
  if [[ "${expected_count}" != "1" || "${package_count}" != "1" ]]; then
    echo "unexpected uv dependency incompatibility" >&2
    exit 1
  fi
fi
PYTHONPATH="${upstream}" "${environment}/bin/python" - <<'PY'
import decord
import motioncrafter
import prob4d
import torch

assert torch.cuda.is_available()
assert torch.__version__ == "2.11.0+cu128"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
print("decord", decord.__version__)
PY
"${uv_bin}" pip freeze --python "${environment}/bin/python" \
  | LC_ALL=C sort \
  > "${evidence}/environment-freeze.txt"
"${environment}/bin/python" -VV > "${evidence}/python-version.txt" 2>&1
export PATH="${environment}/bin:${PATH}"
export PYTHONPATH="${upstream}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${run_root}" "${bundle}"
mapfile -t entries < <(
  find "${bundle}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | LC_ALL=C sort
)
resume=()
if [[ -f "${bundle}/predictions.json" ]]; then
  prob4d-motioncrafter --output-dir "${bundle}" --verify-only \
    | tee "${evidence}/generation-verify-before.json"
elif [[ -f "${bundle}/progress.json" ]]; then
  resume+=(--resume)
elif (( ${#entries[@]} )); then
  printf 'unexpected incomplete bundle members:\n%s\n' "${entries[*]}" >&2
  exit 1
fi
started="$(date +%s)"
if [[ ! -f "${bundle}/predictions.json" ]]; then
  prob4d-motioncrafter \
    "${SOURCE_VIDEO}" \
    --upstream-root "${upstream}" \
    --output-dir "${bundle}" \
    --model-type determ \
    --unet-path TencentARC/MotionCrafter \
    --unet-revision fc7b18d5657184607bf4501b02d64ada7540b4e3 \
    --vae-path TencentARC/MotionCrafter \
    --vae-revision fc7b18d5657184607bf4501b02d64ada7540b4e3 \
    --image-vae-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --image-vae-revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
    --base-pipeline-path stabilityai/stable-video-diffusion-img2vid-xt \
    --base-pipeline-revision 9e43909513c6714f1bc78bcb44d96e733cd242aa \
    --cache-dir /home/github-runner/.cache/huggingface/hub \
    --height 320 \
    --width 640 \
    --window-size 25 \
    --overlap 8 \
    --num-inference-steps 5 \
    --guidance-scale 1.0 \
    --decode-chunk-size 25 \
    --seed 20260806 \
    --seed-policy derived-per-call \
    --frame-start 0 \
    --frame-stop 59 \
    --frame-stride 1 \
    "${resume[@]}" \
    2>&1 | tee "${evidence}/generation.log"
fi
stopped="$(date +%s)"
printf 'wall_seconds=%s\n' "$((stopped - started))" \
  > "${evidence}/generation-time.txt"
prob4d-motioncrafter --output-dir "${bundle}" --verify-only \
  | tee "${evidence}/generation-verification.json"
cp "${bundle}/predictions.json" "${evidence}/prediction-manifest.json"
if [[ -f "${bundle}/progress.json" ]]; then
  cp "${bundle}/progress.json" "${evidence}/prediction-progress.json"
fi

if [[ ! -e "${store}" ]]; then
  /usr/bin/time -v \
    -o "${evidence}/store-materialization-time.txt" \
    prob4d storage materialize \
      "${bundle}/predictions.json" "${store}" \
      --dense-storage-dtype float32 \
    > "${evidence}/store-materialization.json"
fi
prob4d storage validate "${store}" \
  | tee "${evidence}/store-summary.json"
mkdir -p "${evidence}/store-manifests"
while IFS= read -r -d '' manifest; do
  relative="${manifest#${store}/}"
  destination="${evidence}/store-manifests/${relative}"
  mkdir -p "$(dirname "${destination}")"
  cp "${manifest}" "${destination}"
done < <(find "${store}" -name manifest.json -type f -print0)

mkdir -p "${work}/eager" "${work}/mmap"
/usr/bin/time -v \
  -o "${evidence}/eager-process-time.txt" \
  python scripts/ci/profile_real_prediction_bundle.py arm \
    --backend eager_npz \
    --input "${bundle}/predictions.json" \
    --lock "${LOCK_PATH}" \
    --output "${work}/eager" \
  2>&1 | tee "${evidence}/eager-arm.log"
cp "${work}/eager/arm-report.json" "${evidence}/eager-arm-report.json"

/usr/bin/time -v \
  -o "${evidence}/mmap-process-time.txt" \
  python scripts/ci/profile_real_prediction_bundle.py arm \
    --backend mmap_npy \
    --input "${store}" \
    --lock "${LOCK_PATH}" \
    --output "${work}/mmap" \
  2>&1 | tee "${evidence}/mmap-arm.log"
cp "${work}/mmap/arm-report.json" "${evidence}/mmap-arm-report.json"

python scripts/ci/profile_real_prediction_bundle.py compare \
  --lock "${LOCK_PATH}" \
  --eager-report "${work}/eager/arm-report.json" \
  --mmap-report "${work}/mmap/arm-report.json" \
  --output "${evidence}/comparison.json" \
  --markdown "${evidence}/SUMMARY.md" \
  2>&1 | tee "${evidence}/comparison.log"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "${evidence}/SUMMARY.md" >> "${GITHUB_STEP_SUMMARY}"
fi
