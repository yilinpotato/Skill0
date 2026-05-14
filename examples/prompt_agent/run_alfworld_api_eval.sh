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

export CACHE_ROOT="${CACHE_ROOT:-/home/myl/.cache}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/skillrl_outputs}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export MODEL_NAME="${MODEL_NAME:-deepseek-v4-pro}"
export API_EVAL_EPISODES="${API_EVAL_EPISODES:-100}"
export API_EVAL_NUM_ENVS="${API_EVAL_NUM_ENVS:-4}"
export API_EVAL_MAX_STEPS="${API_EVAL_MAX_STEPS:-40}"
export API_EVAL_MAX_TOKENS="${API_EVAL_MAX_TOKENS:-512}"
export API_EVAL_TEMPERATURE="${API_EVAL_TEMPERATURE:-0.0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skillrl_mvp}"
export WANDB_NAME="${WANDB_NAME:-alfworld_api_baseline_deepseek_v4pro}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-alfworld_api_baseline}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT/wandb}"

python3 scripts/eval_alfworld_api.py "$@"
