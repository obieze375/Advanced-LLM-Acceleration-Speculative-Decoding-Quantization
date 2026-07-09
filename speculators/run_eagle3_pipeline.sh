#!/usr/bin/env bash
# ===========================================================================
# EAGLE-3 offline training pipeline for Qwen/Qwen3-8B (verifier), trained
# BEFORE FP8 quantization is applied (see /REPORT.md for justification).
#
# Mirrors: docs.vllm.ai/projects/speculators/.../train_eagle3_offline
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UPSTREAM="${REPO_ROOT}/speculators-upstream"

MODEL="Qwen/Qwen3-8B"
OUT="${UPSTREAM}/output"
HS_OUT="${OUT}/hidden_states"
CKPT_OUT="${OUT}/checkpoints"
PORT=8000
MAX_SAMPLES=3000
SEQ_LENGTH=2048

cd "${UPSTREAM}"

# ---------------------------------------------------------------------------
# Step 1: Prepare data (speculators_venv)
#   ShareGPT-style conversations, as required by the assignment.
# ---------------------------------------------------------------------------
prepare_data () {
  source "${REPO_ROOT}/speculators_venv/bin/activate"
  python scripts/prepare_data.py \
    --model "${MODEL}" \
    --data sharegpt \
    --output "${OUT}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}"
}

# ---------------------------------------------------------------------------
# Step 2: Launch vLLM server to serve the FULL-PRECISION (bf16) verifier
#   and expose internal hidden states (vllm_venv).
#   Full precision is used deliberately: EAGLE-3's training targets should
#   not carry FP8 quantization noise (see REPORT.md, Section 3).
# ---------------------------------------------------------------------------
launch_verifier () {
  source "${REPO_ROOT}/vllm_venv/bin/activate"
  CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py \
    "${MODEL}" \
    -- --port "${PORT}" --gpu-memory-utilization 0.85
  # Single H100-80GB: no data/tensor parallelism needed for an 8B model.
  # Wait for "Application startup complete" before proceeding to Step 3.
}

# ---------------------------------------------------------------------------
# Step 3: Generate hidden states offline (speculators_venv)
# ---------------------------------------------------------------------------
generate_hidden_states () {
  source "${REPO_ROOT}/speculators_venv/bin/activate"
  python scripts/data_generation_offline.py \
    --preprocessed-data "${OUT}" \
    --endpoint "http://localhost:${PORT}/v1" \
    --output "${HS_OUT}" \
    --max-samples "${MAX_SAMPLES}" \
    --concurrency 8 \
    --request-timeout 60 \
    --validate-outputs
}

# ---------------------------------------------------------------------------
# Step 4: Stop the vLLM verifier server manually (Ctrl+C in its terminal)
#   before Step 5 -- offline training does not need vLLM running.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 5: Train the EAGLE-3 draft head against cached hidden states
#   (speculators_venv). Single H100 -> single-GPU training path.
# ---------------------------------------------------------------------------
train_draft_head () {
  source "${REPO_ROOT}/speculators_venv/bin/activate"
  python scripts/train.py \
    --verifier-name-or-path "${MODEL}" \
    --data-path "${OUT}" \
    --hidden-states-path "${HS_OUT}" \
    --save-path "${CKPT_OUT}" \
    --draft-vocab-size 32000 \
    --epochs 5 \
    --lr 1e-4 \
    --total-seq-len "${SEQ_LENGTH}" \
    --on-missing raise
}

# ---------------------------------------------------------------------------
# Step 6/7: Inspect + smoke-test the trained draft head against the
#   FULL-PRECISION verifier before moving to quantization.
# ---------------------------------------------------------------------------
smoke_test () {
  source "${REPO_ROOT}/vllm_venv/bin/activate"
  echo "Checkpoints:"
  ls -1 "${CKPT_OUT}"
  echo "Serve with: vllm serve ${CKPT_OUT}/checkpoint_best --port ${PORT}"
  echo "Then run:   vllm chat --url http://localhost:${PORT}/v1"
}

case "${1:-}" in
  prepare_data)            prepare_data ;;
  launch_verifier)         launch_verifier ;;
  generate_hidden_states)  generate_hidden_states ;;
  train_draft_head)        train_draft_head ;;
  smoke_test)              smoke_test ;;
  *)
    echo "Usage: $0 {prepare_data|launch_verifier|generate_hidden_states|train_draft_head|smoke_test}"
    echo "Run launch_verifier in one terminal, then generate_hidden_states in another."
    exit 1
    ;;
esac
