#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vllm_venv : serving the verifier (for hidden-state extraction) and running
#             baseline / speculative / quantized / combined benchmarks
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY=python3.12

$PY -m venv vllm_venv
source vllm_venv/bin/activate
pip install --upgrade pip uv

# Pinned per assignment spec; [bench] adds datasets etc. for `vllm bench serve`
uv pip install "vllm[bench]==0.20.0"
uv pip install "fastapi<0.137"

echo "vllm_venv ready. Activate with: source vllm_venv/bin/activate"
echo "Sanity check:"
python -c "import vllm, fastapi, datasets; print('vllm', vllm.__version__, '| fastapi', fastapi.__version__, '| datasets', datasets.__version__)"
