#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# speculators_venv : data preparation, hidden-state generation, EAGLE-3 training
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY=python3.12

$PY -m venv speculators_venv
source speculators_venv/bin/activate
pip install --upgrade pip uv

# speculators is installed editable from the git tag v0.5.0 per assignment spec
git clone https://github.com/vllm-project/speculators.git speculators-upstream
cd speculators-upstream
git checkout v0.5.0
uv pip install -e .

# optional experiment trackers (install manually per speculators docs, since
# they are NOT auto-installed with the base package)
uv pip install wandb tensorboard

cd ..
echo "speculators_venv ready. Activate with: source speculators_venv/bin/activate"
