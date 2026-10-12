#!/usr/bin/env bash
# GPU-node setup: gpu extra, then build causal-conv1d (no PyPI wheels) with a pip-installed nvcc.
# Runs on a CPU machine too (compiling needs no GPU), so do it before switching to H200s.
#   bash scripts/setup_gpu.sh          # TORCH_CUDA_ARCH_LIST defaults to 9.0 (H100/H200)
# Lessons baked in: Lightning studios redirect uv envs (UV_LIGHTNING_VIRTUALENV_ROOT) and `uv pip`
# targets the conda env unless --python is given; nvcc must match torch's CUDA exactly (13.0, not
# 13.x latest); .venv can be a symlink (find misses it); the wheels ship only versioned sonames.
set -euo pipefail
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
echo "PROGRESS_PHASE sync 1/4"
echo "PROGRESS 0/4"
uv sync --group dev --extra gpu --inexact   # --inexact keeps a previously built causal-conv1d
.venv/bin/python -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "PROGRESS 1/4"
echo "PROGRESS_PHASE nvcc 2/4"
CU=$(.venv/bin/python -c "import torch; print(torch.version.cuda.split('.')[0])")
CUV=$(.venv/bin/python -c "import torch; print(torch.version.cuda)")   # e.g. 13.0: nvcc must match torch's headers exactly
uv pip install --python .venv/bin/python setuptools wheel ninja packaging \
  "nvidia-cuda-nvcc==${CUV}.*" "nvidia-cuda-crt==${CUV}.*" "nvidia-nvvm==${CUV}.*" \
  "nvidia-cuda-runtime==${CUV}.*" "nvidia-cuda-cccl==${CUV}.*"
# CUDA wheels live under site-packages/nvidia/cu<major>; .venv can be a symlink, so ask Python
export CUDA_HOME=$(.venv/bin/python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu${CU}'))")
echo "CUDA_HOME=$CUDA_HOME"
test -x "$CUDA_HOME/bin/nvcc"
[ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
# the wheels ship only versioned sonames; the linker needs the plain names
for so in "$CUDA_HOME"/lib/lib*.so.*; do base="${so%%.so.*}.so"; [ -e "$base" ] || ln -s "$(basename "$so")" "$base"; done
export PATH="$CUDA_HOME/bin:$PATH" CPATH="$CUDA_HOME/include:${CPATH:-}" LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:${LIBRARY_PATH:-}"
nvcc --version | tail -2
echo "PROGRESS 2/4"
echo "PROGRESS_PHASE causal-conv1d 3/4"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}"
uv pip install --python .venv/bin/python --no-build-isolation --no-deps 'causal-conv1d==1.7.0'
echo "PROGRESS 3/4"
echo "PROGRESS_PHASE verify 4/4"
.venv/bin/python -c "import causal_conv1d, fla; print('causal_conv1d', causal_conv1d.__version__, '| fla', fla.__version__)"
.venv/bin/pytest -q 2>&1 | tail -1
echo "PROGRESS 4/4"
