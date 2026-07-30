#!/usr/bin/env bash
# Qwen3-0.6B Skill0 scale arm.  Delegate to the canonical fixed single-stage
# launcher so its GRPO and progressive-internalization controls stay unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
        export MODEL_PATH="/GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    else
        export MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    fi
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_qwen3_0.6b_thinking_v6}"

exec bash "$SCRIPT_DIR/run_alfworld_fixed_single_stage.sh" "$@"
