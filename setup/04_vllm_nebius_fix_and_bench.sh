#!/usr/bin/env bash
# Nebius CUDA 13 vLLM fix + Task 4 benchmarks (vllm serve + vllm bench serve).
# Run from repo root: bash setup/04_vllm_nebius_fix_and_bench.sh [test|task4]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-test}"
CUDA_STUBS="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs"
DRIVER_LIB="/usr/lib/x86_64-linux-gnu"
DRAFT_LOCAL="${REPO_ROOT}/speculators-upstream/output/checkpoints/checkpoint_best"
DRAFT_HF="RedHatAI/Qwen3-8B-speculator.eagle3"
FP8="${REPO_ROOT}/quantization/Qwen3-8B-FP8-Dynamic"
PORT=8000

setup_env () {
  unset TRITON_LIBCUDA_PATH
  export LD_LIBRARY_PATH="${CUDA_STUBS}:${DRIVER_LIB}:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
  export CPATH="/usr/local/cuda-13.0/include:${CPATH:-}"
  export VLLM_USE_V1=0
  export TORCH_COMPILE_DISABLE=1
  unset VLLM_TORCH_COMPILE_LEVEL
  source "${REPO_ROOT}/vllm_venv/bin/activate"
}

fix_triton () {
  echo "==> Triton libcuda fix"
  TRITON_LIB="$(python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/lib'))")"
  sudo cp -f "${CUDA_STUBS}/libcuda.so" "${TRITON_LIB}/libcuda.so"
  sudo cp -f "${DRIVER_LIB}/libcuda.so.1" "${TRITON_LIB}/libcuda.so.1"
  python - <<'PY'
import os
os.environ.pop("TRITON_LIBCUDA_PATH", None)
from triton.backends.nvidia.driver import CudaUtils
CudaUtils()
print("OK: Triton CudaUtils compiled")
PY
}

stop_server () {
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  sleep 3
}

wait_for_server () {
  echo "==> Waiting for vLLM on :${PORT}..."
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
      || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "==> Server ready"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: server did not start on port ${PORT}"
  return 1
}

# Terminal A equivalent: vllm serve (speculative-config goes HERE, not on bench)
start_server () {
  local model="$1"
  local spec_json="${2:-}"
  stop_server
  echo "==> Starting server: ${model}"
  if [ -n "${spec_json}" ]; then
    vllm serve "${model}" \
      --port "${PORT}" \
      --gpu-memory-utilization 0.85 \
      --enforce-eager \
      --generation-config vllm \
      --speculative-config "${spec_json}" \
      > /tmp/vllm_serve.log 2>&1 &
  else
    vllm serve "${model}" \
      --port "${PORT}" \
      --gpu-memory-utilization 0.85 \
      --enforce-eager \
      --generation-config vllm \
      > /tmp/vllm_serve.log 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_server || { tail -50 /tmp/vllm_serve.log; kill "${SERVER_PID}" 2>/dev/null || true; return 1; }
}

# Terminal B equivalent: vllm bench serve (client only — no speculative-config here)
run_bench () {
  local model="$1"
  local outfile="$2"
  vllm bench serve \
    --model "${model}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80 \
    --max-concurrency 8 \
    2>&1 | tee "${outfile}"
}

run_config () {
  local label="$1"
  local model="$2"
  local spec_json="${3:-}"
  local outfile="$4"
  echo ""
  echo "========== ${label} =========="
  start_server "${model}" "${spec_json}"
  run_bench "${model}" "${outfile}"
  stop_server
}

echo "==> [1/4] System deps (optional)"
sudo apt-get update -qq
sudo apt-get install -y build-essential python3.12-dev curl 2>/dev/null || sudo apt-get install -y build-essential curl

echo "==> [2/4] Env + Triton"
setup_env
sed -i '/TRITON_LIBCUDA_PATH/d' ~/.bashrc
fix_triton

echo "==> [3/4] Draft + FP8 paths"
if [ -f "${DRAFT_LOCAL}/config.json" ]; then
  DRAFT="${DRAFT_LOCAL}"
else
  DRAFT="${DRAFT_HF}"
fi
echo "Draft: ${DRAFT}"

if [ ! -f "${FP8}/config.json" ]; then
  echo "==> Running FP8 quantization"
  source "${REPO_ROOT}/comp_venv/bin/activate"
  cd "${REPO_ROOT}/quantization" && python quantize_fp8.py
  cd "${REPO_ROOT}"
  setup_env
fi

if [ "${MODE}" = "test" ]; then
  start_server "Qwen/Qwen3-8B" ""
  stop_server
  echo ""
  echo "SUCCESS. Run: bash setup/04_vllm_nebius_fix_and_bench.sh task4"
  exit 0
fi

if [ "${MODE}" != "task4" ]; then
  echo "Usage: $0 [test|task4]"
  exit 1
fi

mkdir -p benchmark

echo "==> [4/4] Task 4 benchmarks (serve + bench, ~30-60 min)"
SPEC_BF16='{"method":"eagle3","model":"'"${DRAFT}"'","num_speculative_tokens":2,"draft_tensor_parallel_size":1}'
SPEC_FP8='{"method":"eagle3","model":"'"${DRAFT}"'","num_speculative_tokens":1,"draft_tensor_parallel_size":1}'

run_config "1/4 baseline" "Qwen/Qwen3-8B" "" "benchmark/vllm_bench_baseline.txt"
run_config "2/4 spec decode" "Qwen/Qwen3-8B" "${SPEC_BF16}" "benchmark/vllm_bench_spec.txt"
run_config "3/4 fp8" "${FP8}" "" "benchmark/vllm_bench_fp8.txt"
run_config "4/4 fp8 + spec" "${FP8}" "${SPEC_FP8}" "benchmark/vllm_bench_fp8_spec.txt"

echo ""
echo "DONE. Results: benchmark/vllm_bench_*.txt"
echo "Look for 'Output token throughput (tok/s)' in each file."
