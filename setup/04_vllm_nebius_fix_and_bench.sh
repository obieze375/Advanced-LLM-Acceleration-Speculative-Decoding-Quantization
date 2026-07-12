#!/usr/bin/env bash
# Aggressive one-shot vLLM fix for Nebius CUDA 13 + Task 4 shortcut.
# Run from repo root: bash setup/04_vllm_nebius_fix_and_bench.sh [test|task4]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-test}"
CUDA_STUBS="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs"
DRIVER_LIB="/usr/lib/x86_64-linux-gnu"
DRAFT_LOCAL="speculators-upstream/output/checkpoints/checkpoint_best"
DRAFT_HF="RedHatAI/Qwen3-8B-speculator.eagle3"
FP8="quantization/Qwen3-8B-FP8-Dynamic"

echo "==> [1/6] System build deps + CUDA headers"
sudo apt-get update -qq
sudo apt-get install -y build-essential python3.12-dev 2>/dev/null || sudo apt-get install -y build-essential python3-dev
sudo apt-get install -y cuda-toolkit-13-0 2>/dev/null || sudo apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0 2>/dev/null || true

echo "==> [2/6] Clean broken env (TRITON_LIBCUDA_PATH breaks gcc -L)"
unset TRITON_LIBCUDA_PATH
sed -i '/TRITON_LIBCUDA_PATH/d' ~/.bashrc
sed -i 's|/lib/stubs/libcuda.so|/lib/stubs|g' ~/.bashrc

export LD_LIBRARY_PATH="${CUDA_STUBS}:${DRIVER_LIB}:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/cuda-13.0/include:${CPATH:-}"
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
unset VLLM_TORCH_COMPILE_LEVEL

echo "==> [3/6] Triton libcuda copy"
source "${REPO_ROOT}/vllm_venv/bin/activate"
TRITON_LIB="$(python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/lib'))")"
sudo cp -f "${CUDA_STUBS}/libcuda.so" "${TRITON_LIB}/libcuda.so"
sudo cp -f "${DRIVER_LIB}/libcuda.so.1" "${TRITON_LIB}/libcuda.so.1"
sudo ldconfig

echo "==> [4/6] Pre-compile Triton (shows real error if still broken)"
python - <<'PY' || { echo "TRITON STILL BROKEN — paste this output"; exit 1; }
import os
os.environ.pop("TRITON_LIBCUDA_PATH", None)
from triton.backends.nvidia.driver import CudaUtils
CudaUtils()
print("OK: Triton CudaUtils compiled")
PY

echo "==> [5/6] Pick draft model"
if [ -f "${DRAFT_LOCAL}/config.json" ]; then
  DRAFT="${DRAFT_LOCAL}"
  echo "Using local draft: ${DRAFT}"
else
  DRAFT="${DRAFT_HF}"
  echo "No local checkpoint — using HuggingFace draft: ${DRAFT}"
  echo "(Fine for Task 4 benchmarks; training not required for bench numbers.)"
fi

if [ ! -f "${FP8}/config.json" ]; then
  echo "==> FP8 model missing — running quantize_fp8.py"
  source "${REPO_ROOT}/comp_venv/bin/activate"
  cd "${REPO_ROOT}/quantization"
  python quantize_fp8.py
  cd "${REPO_ROOT}"
  source "${REPO_ROOT}/vllm_venv/bin/activate"
fi

COMMON=(--dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --max-concurrency 8 --enforce-eager)

vllm_smoke_test () {
  echo "==> Smoke test: vllm serve (15s)..."
  timeout 120 vllm serve Qwen/Qwen3-8B --port 8765 --gpu-memory-utilization 0.85 --enforce-eager &
  PID=$!
  sleep 45
  if curl -sf http://127.0.0.1:8765/health >/dev/null 2>&1 || curl -sf http://127.0.0.1:8765/v1/models >/dev/null 2>&1; then
    echo "OK: vLLM server responded"
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    return 0
  fi
  kill $PID 2>/dev/null || true
  wait $PID 2>/dev/null || true
  echo "FAIL: vLLM did not start in 45s"
  return 1
}

if [ "${MODE}" = "test" ]; then
  vllm_smoke_test
  echo ""
  echo "SUCCESS. Run Task 4 benchmarks:"
  echo "  bash setup/04_vllm_nebius_fix_and_bench.sh task4"
  exit 0
fi

if [ "${MODE}" != "task4" ]; then
  echo "Usage: $0 [test|task4]"
  exit 1
fi

mkdir -p benchmark
vllm_smoke_test || exit 1

echo "==> [6/6] Task 4 — four vllm bench serve runs (this takes ~30-60 min total)"
echo "1/4 baseline..."
vllm bench serve --model Qwen/Qwen3-8B "${COMMON[@]}" 2>&1 | tee benchmark/vllm_bench_baseline.txt

echo "2/4 spec decode (2 draft tokens)..."
vllm bench serve --model Qwen/Qwen3-8B \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":2,\"draft_tensor_parallel_size\":1}" \
  "${COMMON[@]}" 2>&1 | tee benchmark/vllm_bench_spec.txt

echo "3/4 fp8..."
vllm bench serve --model "${FP8}" "${COMMON[@]}" 2>&1 | tee benchmark/vllm_bench_fp8.txt

echo "4/4 fp8 + spec (1 draft token)..."
vllm bench serve --model "${FP8}" \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":1,\"draft_tensor_parallel_size\":1}" \
  "${COMMON[@]}" 2>&1 | tee benchmark/vllm_bench_fp8_spec.txt

echo ""
echo "DONE. Results in benchmark/vllm_bench_*.txt"
echo "Look for 'Output token throughput' in each file."
