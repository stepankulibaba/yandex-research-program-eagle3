#!/usr/bin/env bash
# One-time: a real CUDA 12.8 compiler for SGLang, without root.
#
# SGLang 0.5.9 compiles some kernels at start-up (e.g. RoPE, via tvm_ffi + nvcc). The server has no CUDA toolkit,
# and NVIDIA's pip wheel for nvcc ships no nvcc binary, so the toolkit comes from conda-forge via micromamba
# (a single static binary). torch in venv_sglang is built for CUDA 12.8, hence 12.8.
#
# Result: ../cuda-home with bin/ (nvcc), include/, lib64/ — what tools expect in CUDA_HOME.
# orchestrate.py gives it to the SGLang processes only; DeepSpeed and NeMo are not affected.
set -Eeuo pipefail
DATA="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$DATA/cuda-12.8"
CUDA_DIR="$DATA/cuda-home"
MAMBA="$DATA/micromamba/bin/micromamba"

if [ ! -x "$PREFIX/bin/nvcc" ]; then
  if [ ! -x "$MAMBA" ]; then
    mkdir -p "$DATA/micromamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$DATA/micromamba" bin/micromamba
  fi
  MAMBA_ROOT_PREFIX="$DATA/micromamba" "$MAMBA" create -y -p "$PREFIX" -c conda-forge \
      "cuda-nvcc=12.8" "cuda-cudart-dev=12.8" "cuda-cccl=12.8" "cuda-nvrtc-dev=12.8" "cuda-driver-dev=12.8"
fi
# CUDA 12.8 accepts host compilers up to gcc 14; the system one is newer. gcc/g++ 13 go into the same prefix,
# and its bin/ comes first on PATH for SGLang processes, so nvcc (and the JIT link step) use them.
if [ ! -x "$PREFIX/bin/g++" ]; then
  MAMBA_ROOT_PREFIX="$DATA/micromamba" "$MAMBA" install -y -p "$PREFIX" -c conda-forge "gcc=13" "gxx=13"
fi
export PATH="$PREFIX/bin:$PATH"

# conda keeps headers and libraries under targets/x86_64-linux; build the usual CUDA_HOME layout from links.
TARGET="$PREFIX/targets/x86_64-linux"
mkdir -p "$CUDA_DIR"
ln -sfn "$PREFIX/bin" "$CUDA_DIR/bin"
ln -sfn "$TARGET/include" "$CUDA_DIR/include"
ln -sfn "$TARGET/lib" "$CUDA_DIR/lib64"
if [ -d "$PREFIX/nvvm" ]; then ln -sfn "$PREFIX/nvvm" "$CUDA_DIR/nvvm"; fi

# Check: compile and link a tiny kernel the way JIT builds do.
CHECK="$(mktemp -d)"
printf '__global__ void k(float *x) { x[threadIdx.x] += 1.0f; }\nint main() { return 0; }\n' > "$CHECK/check.cu"
"$CUDA_DIR/bin/nvcc" -ccbin "$PREFIX/bin/g++" -arch=sm_90 -I"$CUDA_DIR/include" -L"$CUDA_DIR/lib64" -lcudart \
    "$CHECK/check.cu" -o "$CHECK/check"
rm -rf "$CHECK"
"$CUDA_DIR/bin/nvcc" --version | tail -n 2
"$PREFIX/bin/g++" --version | head -n 1
echo "CUDA toolkit ready: $CUDA_DIR"
