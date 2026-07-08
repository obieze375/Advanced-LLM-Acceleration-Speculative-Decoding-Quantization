#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# comp_venv : llm-compressor FP8 dynamic quantization of the verifier model
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY=python3.12

$PY -m venv comp_venv
source comp_venv/bin/activate
pip install --upgrade pip uv

# Pinned per assignment spec
uv pip install "llmcompressor==0.12.0"
uv pip install "transformers" "accelerate"

# Needed only to *evaluate* the FP8 checkpoint after quantization (separate
# process from vllm_venv is fine, or just use vllm_venv for eval -- see
# quantization/quantize_fp8.py header comment).
echo "comp_venv ready. Activate with: source comp_venv/bin/activate"
