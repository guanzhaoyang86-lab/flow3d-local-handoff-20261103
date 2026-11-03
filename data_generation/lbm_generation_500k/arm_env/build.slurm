#!/bin/bash -l
#SBATCH --job-name=taichi-arm-build
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${LBM500K_ARM_DIR:?Submit using the prepared ARM pipeline}"
source "$LBM500K_ARM_DIR/arm_runtime.sh"
[[ "$(uname -m)" == aarch64 ]] || { printf 'ARM host required.\n' >&2; exit 1; }
if python -B "$LBM500K_ARM_ROOT/check_build.py" --check-build; then
    exec srun --ntasks=1 --gpu-bind=none python -B "$LBM500K_ARM_ROOT/validate_gpu.py"
fi
cd -- "$LBM500K_ARM_ROOT/taichi_source"
[[ "$(git rev-parse HEAD)" == 93c190f603b676ec1f8871e4f46cf5d5be89548e ]]
[[ "$("$LBM500K_ARM_ROOT/llvm20/bin/llvm-config" --version)" == 20.1.8 ]]
"$LBM500K_ARM_ROOT/llvm20/bin/llvm-config" --targets-built | grep -q AArch64
"$LBM500K_ARM_ROOT/llvm20/bin/llvm-config" --targets-built | grep -q NVPTX
# Compile the translation unit that failed in the first attempt before doing
# the rest of the wheel build. This runs only within the Slurm allocation.
"$CC" -c taichi/runtime/llvm/runtime_module/runtime.cpp \
    -o "$LBM500K_ARM_ROOT/runtime_cuda.preflight.bc" -fno-exceptions \
    -emit-llvm -std=c++17 -D ARCH_cuda -I "$LBM500K_ARM_ROOT/taichi_source"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-8}"
export BUILD_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TAICHI_CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCLANG_EXECUTABLE=$CC -DLLVM_DIR=$LLVM_DIR -DCMAKE_PREFIX_PATH=$LBM500K_ARM_ROOT/llvm20 -DCMAKE_CXX_FLAGS=-I$LBM500K_ARM_ROOT/llvm20/include -DCMAKE_SHARED_LINKER_FLAGS=-L$LBM500K_ARM_ROOT/llvm20/lib -DCMAKE_EXE_LINKER_FLAGS=-L$LBM500K_ARM_ROOT/llvm20/lib -DCMAKE_MODULE_LINKER_FLAGS=-L$LBM500K_ARM_ROOT/llvm20/lib -DUSE_STDCPP=OFF -DTI_WITH_LLVM=ON -DTI_WITH_CUDA=ON -DTI_WITH_CUDA_TOOLKIT=OFF -DTI_WITH_VULKAN=OFF -DTI_WITH_GGUI=OFF -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_BUILD_TESTS=OFF -DTI_BUILD_EXAMPLES=OFF -DTI_WITH_C_API=OFF"
mkdir -p "$LBM500K_ARM_ROOT/wheels"
python -m pip wheel --no-build-isolation --no-deps . --wheel-dir "$LBM500K_ARM_ROOT/wheels"
lbm500k_wheel="$(python - "$LBM500K_ARM_ROOT/wheels" <<'PY'
from pathlib import Path
import sys
wheels=list(Path(sys.argv[1]).glob('taichi-*.whl'))
if not wheels:raise SystemExit('No Taichi wheel was built')
print(max(wheels,key=lambda p:p.stat().st_mtime))
PY
)"
python -m pip install --no-deps --force-reinstall "$lbm500k_wheel"
cd -- "$LBM500K_ARM_ROOT"
python "$LBM500K_ARM_ROOT/write_build_report.py" "$lbm500k_wheel"
exec srun --ntasks=1 --gpu-bind=none python -B "$LBM500K_ARM_ROOT/validate_gpu.py"

