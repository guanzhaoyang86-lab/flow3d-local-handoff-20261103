#!/bin/bash
set -euo pipefail
export LBM500K_ARM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export LBM500K_PACKAGE_ROOT="$(cd -- "$LBM500K_ARM_ROOT/.." && pwd -P)"
export PATH="$LBM500K_ARM_ROOT/venv/bin:$LBM500K_ARM_ROOT/llvm20/bin:$PATH"
export LD_LIBRARY_PATH="$LBM500K_ARM_ROOT/llvm20/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LLVM_DIR="$LBM500K_ARM_ROOT/llvm20/lib/cmake/llvm"
export CMAKE_PREFIX_PATH="$LBM500K_ARM_ROOT/llvm20${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CC="$LBM500K_ARM_ROOT/llvm20/bin/clang"
export CXX="$LBM500K_ARM_ROOT/llvm20/bin/clang++"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
unset PYTHONPATH

