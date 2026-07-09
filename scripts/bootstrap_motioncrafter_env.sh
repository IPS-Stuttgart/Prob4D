#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_root="${1:-"${repo_root}/../MotionCrafter"}"
venv_root="${2:-"${repo_root}/../prob4d-motioncrafter-venv"}"

if [[ ! -d "${upstream_root}/motioncrafter" ]]; then
  git clone --depth 1 https://github.com/TencentARC/MotionCrafter.git "${upstream_root}"
fi

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv_bin="${HOME}/.local/bin/uv"
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  "${uv_bin}" venv --python python3 "${venv_root}"
fi

"${uv_bin}" pip install \
  --python "${venv_root}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' \
  'torchvision==0.26.0+cu128'

"${uv_bin}" pip install \
  --python "${venv_root}/bin/python" \
  --requirement "${repo_root}/environments/motioncrafter-inference.txt"

"${uv_bin}" pip install --python "${venv_root}/bin/python" --editable "${repo_root}"

PYTHONPATH="${upstream_root}" "${venv_root}/bin/python" -c \
  'import motioncrafter, prob4d, torch; assert torch.cuda.is_available()'

echo "MotionCrafter environment ready: ${venv_root}"

