#!/usr/bin/env bash
# Qwen3-0.6B Skill0 scale arm.  It keeps all canonical WebShop GRPO settings
# and progressive skill internalization settings unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
        export MODEL_PATH="/GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    else
        export MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    fi
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_webshop_skill0_v3}"

exec bash "$SCRIPT_DIR/run_webshop_fixed_single_stage.sh" "$@"
