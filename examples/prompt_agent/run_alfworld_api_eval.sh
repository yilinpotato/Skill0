#!/usr/bin/env bash
# Direct API evaluation for ALFWorld

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set +u
source "${CONDA_SH_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate skillRL
set -u

python3 scripts/eval_alfworld_api.py "$@"
