#!/usr/bin/env bash
# Qwen3-0.6B Skill0 A800 two-GPU efficiency profile.
# Data parallelism is handled by verl/FSDP; vLLM tensor parallelism remains 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export SKILL0_A800_EFFICIENCY_PROFILE="${SKILL0_A800_EFFICIENCY_PROFILE:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_qwen3_0.6b_a800_2gpu_v1}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-alfworld_skill0_0.6b_a800_2gpu}"

exec bash "$SCRIPT_DIR/run_alfworld_fixed_single_stage_qwen3_0.6b.sh" "$@"
