#!/usr/bin/env bash
# One-time fix for Triton libcuda compile failures on Nebius CUDA 13 VMs.
# Run from repo root: bash setup/03_fix_vllm_triton.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_STUBS="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs"
DRIVER_LIB="/usr/lib/x86_64-linux-gnu"

if [ ! -f "${CUDA_STUBS}/libcuda.so" ]; then
  echo "ERROR: ${CUDA_STUBS}/libcuda.so not found"
  exit 1
fi

source "${REPO_ROOT}/vllm_venv/bin/activate"
TRITON_LIB="$(python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/lib'))")"

echo "==> Copy libcuda into Triton backend lib: ${TRITON_LIB}"
sudo cp -f "${CUDA_STUBS}/libcuda.so" "${TRITON_LIB}/libcuda.so"
sudo cp -f "${DRIVER_LIB}/libcuda.so.1" "${TRITON_LIB}/libcuda.so.1"

echo "==> Ensure system libcuda.so symlink exists"
if [ ! -e "${DRIVER_LIB}/libcuda.so" ]; then
  sudo ln -sf "${DRIVER_LIB}/libcuda.so.1" "${DRIVER_LIB}/libcuda.so"
fi
sudo ldconfig

echo "==> Remove broken TRITON_LIBCUDA_PATH from ~/.bashrc (causes -L.../libcuda.so gcc bug)"
sed -i '/TRITON_LIBCUDA_PATH/d' ~/.bashrc
sed -i 's|/lib/stubs/libcuda.so|/lib/stubs|g' ~/.bashrc

echo "==> Append correct vLLM env to ~/.bashrc"
grep -q 'NEBIUS_VLLM_ENV' ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# NEBIUS_VLLM_ENV — do not set TRITON_LIBCUDA_PATH (breaks gcc -L paths)
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
EOF

echo "==> Link test"
unset TRITON_LIBCUDA_PATH
export LD_LIBRARY_PATH="${CUDA_STUBS}:${DRIVER_LIB}"
echo 'int main(void){return 0;}' | gcc -x c - -lcuda -o /tmp/cuda_test
echo "libcuda link OK"

echo ""
echo "Done. Run: source ~/.bashrc && bash speculators/run_eagle3_pipeline.sh launch_verifier"
