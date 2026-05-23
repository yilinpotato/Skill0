#!/usr/bin/env bash
# ALFWorld GRPO single-stage training for A800 80GB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export TRACE_SH="${TRACE_SH:-0}"
if [[ "$TRACE_SH" == "1" ]]; then
  set -x
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_qwen3_4b_thinking_v6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# vLLM V1 uses its own CuMem memory pool. PyTorch expandable segments are
# incompatible with that pool and can also make GPU memory diagnosis misleading.
unset PYTORCH_CUDA_ALLOC_CONF

# ============================================================================
# A800 80GB single GPU configuration
# ============================================================================

n_gpus_per_node=1

echo "============================================"
echo "ALFWorld GRPO training（A800 80GB single GPU）"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "检测到 GPU 数量: $n_gpus_per_node"
echo "============================================"
echo ""

# Conservative single-node defaults. ALFWorld creates one Ray actor per
# train_batch_size * group_size plus validation envs, so aggressive rollout
# parallelism can occupy all CPU slots before the GPU worker is scheduled.
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"     # 每步 16 个任务
export GROUP_SIZE="${GROUP_SIZE:-8}"                # 每任务 8 条 rollout -> 每步 128 条
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-64}"
export ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.05}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"

# 修复 2：提高学习率 + warmup
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

# 修复 3：Entropy bonus
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.05}"

# 修复 4：奖励归一化（当前 SkillRL 代码尚未实际读取这两个字段，仅保留为实验记录）
export NORMALIZE_REWARD="${NORMALIZE_REWARD:-True}"
export REWARD_CLIP="${REWARD_CLIP:-10.0}"

# 修复 5：渐进式技能内化
export GLOBAL_TOP_K_SCHEDULE="${GLOBAL_TOP_K_SCHEDULE:-[12,12,12,8,8,8,6,6,6,4,4,4,2,2,2,0]}"
export GLOBAL_INTERNALIZATION_MODE="${GLOBAL_INTERNALIZATION_MODE:-skillzero}"
export SKILLZERO_INCLUDE_TASK_SPECIFIC="${SKILLZERO_INCLUDE_TASK_SPECIFIC:-True}"
export ONPOLICY_HELPFULNESS_EVAL_ENABLED="${ONPOLICY_HELPFULNESS_EVAL_ENABLED:-False}"
export ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE="${ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE:-False}"
export ONPOLICY_HELPFULNESS_DELTA_THRESHOLD="${ONPOLICY_HELPFULNESS_DELTA_THRESHOLD:-0.01}"
export ONPOLICY_HELPFULNESS_PATIENCE="${ONPOLICY_HELPFULNESS_PATIENCE:-1}"
export ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND="${ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND:-1}"
export ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS="${ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS:-0}"

# 其他训练配置
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-160}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-25}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export MAX_STEPS="${MAX_STEPS:-40}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-20}"
export DENSE_REWARD="${DENSE_REWARD:-True}"
export TRAIN_TASK_TYPE="${TRAIN_TASK_TYPE:-[pick_two_obj_and_place,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep]}"
export EVAL_TASK_TYPE="${EVAL_TASK_TYPE:-[pick_two_obj_and_place,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep]}"
export INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.05}"
export PRINT_TRAJECTORIES="${PRINT_TRAJECTORIES:-False}"
export WANDB_PROJECT="${WANDB_PROJECT:-skillrl_mvp}"
export WANDB_NAME="${WANDB_NAME:-$EXPERIMENT_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-alfworld_fixed_single_stage}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT/wandb}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-4}"
export WANDB_LOG_TRAJECTORIES="${WANDB_LOG_TRAJECTORIES:-True}"
export WANDB_LOG_CONTEXTS="${WANDB_LOG_CONTEXTS:-True}"
export WANDB_MAX_TRAJECTORIES="${WANDB_MAX_TRAJECTORIES:-4}"
export WANDB_CONTEXT_SAMPLES_PER_ROLLOUT="${WANDB_CONTEXT_SAMPLES_PER_ROLLOUT:-5}"
export WANDB_MAX_TRAJECTORY_STEPS="${WANDB_MAX_TRAJECTORY_STEPS:-8}"
export WANDB_MAX_TEXT_CHARS="${WANDB_MAX_TEXT_CHARS:-400}"
export WANDB_MAX_SUMMARY_ROWS="${WANDB_MAX_SUMMARY_ROWS:-200}"
export WANDB_MAX_STEP_ROWS="${WANDB_MAX_STEP_ROWS:-1000}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
export WANDB_GRAPHQL_TIMEOUT="${WANDB_GRAPHQL_TIMEOUT:-60}"
export COMPACT_CONSOLE_OUTPUT="${COMPACT_CONSOLE_OUTPUT:-True}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10000}"
# Qwen3 thinking models are prompted inside an opened <think> block and often
# need substantially more than 64 tokens to reach </think><action>...</action>.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# A800 80GB vLLM / FSDP colocated settings.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-1}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
export RESUME_DATALOADER_STATE="${RESUME_DATALOADER_STATE:-False}"
export ENABLE_RESOURCE_MONITOR="${ENABLE_RESOURCE_MONITOR:-1}"


# 数据准备
source scripts/setup_alfworld_cache.sh

run_dir="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
mkdir -p "$run_dir"
mkdir -p "$WANDB_DIR"
diagnostics_dir="$run_dir/diagnostics/$(date +%Y%m%d_%H%M%S)"
monitor_pid=""
cleanup_monitor() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

if [[ "$ENABLE_RESOURCE_MONITOR" == "1" ]]; then
  export RESOURCE_DIAGNOSTICS_DIR="$diagnostics_dir"
  bash scripts/resource_monitor.sh "$diagnostics_dir" "$$" &
  monitor_pid="$!"
  echo "[resource_monitor] pid=$monitor_pid log_dir=$diagnostics_dir"
fi

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$DATA_ROOT" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

# 构建训练参数
ppo_mini_batch_size=$((TRAIN_DATA_SIZE * GROUP_SIZE))

ppo_args=(
    algorithm.adv_estimator=grpo
    "data.train_files=$DATA_ROOT/text/train.parquet"
    "data.val_files=$DATA_ROOT/text/test.parquet"
    "data.train_batch_size=$TRAIN_DATA_SIZE"
    "data.val_batch_size=$VAL_DATA_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True

    # 模型配置
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "++actor_rollout_ref.model.override_config.pad_token_id=128009"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # 优化器（关键修复）⭐
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
    "actor_rollout_ref.actor.optim.warmup_style=$LR_SCHEDULER"
    "actor_rollout_ref.actor.optim.total_training_steps=$TOTAL_TRAINING_STEPS"

    # PPO 配置
    "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl

    # Entropy bonus ⭐
    "++actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"

    # 奖励归一化配置记录：当前训练代码未读取 normalize_reward/reward_clip
    "++actor_rollout_ref.actor.normalize_reward=$NORMALIZE_REWARD"
    "++actor_rollout_ref.actor.reward_clip=$REWARD_CLIP"

    # FSDP 配置
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"

    # Rollout 配置（A800 80GB single GPU）
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP_SIZE"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=0.8
    "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.max_model_len=$VLLM_MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True

    # Reference policy
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    # 无效动作惩罚
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    "actor_rollout_ref.actor.invalid_action_penalty_coef=$INVALID_ACTION_PENALTY_COEF"

    # 环境配置
    algorithm.use_kl_in_reward=False
    env.env_name=alfworld/AlfredTWEnv
    env.seed=0
    "env.history_length=$HISTORY_LENGTH"
    "env.max_steps=$MAX_STEPS"
    "env.rollout.n=$GROUP_SIZE"
    "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
    "++env.alfworld.use_dense_reward=$DENSE_REWARD"
    "++env.alfworld.train_task_type=$TRAIN_TASK_TYPE"
    "++env.alfworld.eval_task_type=$EVAL_TASK_TYPE"

    # 渐进式技能内化 ⭐
    ++env.use_skills_only_memory=True
    ++env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json
    ++env.skills_only_memory.top_k=12
    "++env.skills_only_memory.global_top_k_schedule=$GLOBAL_TOP_K_SCHEDULE"
    "++env.skills_only_memory.global_internalization_mode=$GLOBAL_INTERNALIZATION_MODE"
    "++env.skills_only_memory.skillzero_include_task_specific=$SKILLZERO_INCLUDE_TASK_SPECIFIC"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_enabled=$ONPOLICY_HELPFULNESS_EVAL_ENABLED"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_auto_internalize=$ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_delta_threshold=$ONPOLICY_HELPFULNESS_DELTA_THRESHOLD"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_patience=$ONPOLICY_HELPFULNESS_PATIENCE"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_remove_per_round=$ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND"
    "++env.skills_only_memory.onpolicy_helpfulness_eval_min_active_skills=$ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS"
    ++env.skills_only_memory.task_specific_top_k=null
    ++env.skills_only_memory.enable_dynamic_update=False
    ++env.skills_only_memory.eval_internalization_modes=True

    # Trainer 配置
    trainer.critic_warmup=0
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.project_name=$WANDB_PROJECT"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$n_gpus_per_node"
    trainer.nnodes=1
    "trainer.default_local_dir=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
    "++trainer.log_val_generations=$LOG_VAL_GENERATIONS"
    "++trainer.wandb_log_trajectories=$WANDB_LOG_TRAJECTORIES"
    "++trainer.wandb_log_contexts=$WANDB_LOG_CONTEXTS"
    "++trainer.wandb_max_trajectories=$WANDB_MAX_TRAJECTORIES"
    "++trainer.wandb_context_samples_per_rollout=$WANDB_CONTEXT_SAMPLES_PER_ROLLOUT"
    "++trainer.wandb_max_trajectory_steps=$WANDB_MAX_TRAJECTORY_STEPS"
    "++trainer.wandb_max_text_chars=$WANDB_MAX_TEXT_CHARS"
    "++trainer.wandb_max_summary_rows=$WANDB_MAX_SUMMARY_ROWS"
    "++trainer.wandb_max_step_rows=$WANDB_MAX_STEP_ROWS"
    "++trainer.metrics_jsonl_path=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/metrics.jsonl"
    "++trainer.trajectory_log_path=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/trajectories.json"
    "++trainer.context_log_path=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/contexts.json"
    ++trainer.trajectory_include_prompt=False
    "++trainer.print_trajectories=$PRINT_TRAJECTORIES"
    "++trainer.compact_console_output=$COMPACT_CONSOLE_OUTPUT"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.total_epochs=$TOTAL_TRAINING_STEPS"
    "++trainer.resume_dataloader_state=$RESUME_DATALOADER_STATE"
    trainer.val_before_train=True
    "ray_init.num_cpus=$RAY_NUM_CPUS"
)

# 保存配置
{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "CACHE_ROOT=$CACHE_ROOT"
  echo "ALFWORLD_DATA=$ALFWORLD_DATA"
  echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
  echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "WANDB_NAME=$WANDB_NAME"
  echo "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
  echo "WANDB_DIR=$WANDB_DIR"
  echo "LOG_VAL_GENERATIONS=$LOG_VAL_GENERATIONS"
  echo "WANDB_LOG_TRAJECTORIES=$WANDB_LOG_TRAJECTORIES"
  echo "WANDB_LOG_CONTEXTS=$WANDB_LOG_CONTEXTS"
  echo "WANDB_MAX_TRAJECTORIES=$WANDB_MAX_TRAJECTORIES"
  echo "WANDB_CONTEXT_SAMPLES_PER_ROLLOUT=$WANDB_CONTEXT_SAMPLES_PER_ROLLOUT"
  echo "WANDB_MAX_TRAJECTORY_STEPS=$WANDB_MAX_TRAJECTORY_STEPS"
  echo "WANDB_INIT_TIMEOUT=$WANDB_INIT_TIMEOUT"
  echo "WANDB_GRAPHQL_TIMEOUT=$WANDB_GRAPHQL_TIMEOUT"
  echo "N_GPUS=$n_gpus_per_node"
  echo "TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE"
  echo "GROUP_SIZE=$GROUP_SIZE"
  echo "VAL_DATA_SIZE=$VAL_DATA_SIZE"
  echo "ENV_WORKER_CPUS=$ENV_WORKER_CPUS"
  echo "RAY_NUM_CPUS=$RAY_NUM_CPUS"
  echo "ROLLOUTS_PER_STEP=$((TRAIN_DATA_SIZE * GROUP_SIZE))"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "LR_WARMUP_STEPS=$LR_WARMUP_STEPS"
  echo "ENTROPY_COEFF=$ENTROPY_COEFF"
  echo "NORMALIZE_REWARD=$NORMALIZE_REWARD"
  echo "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS"
  echo "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
  echo "VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS"
  echo "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS"
  echo "VLLM_TP_SIZE=$VLLM_TP_SIZE"
  echo "OPTIMIZER_OFFLOAD=$OPTIMIZER_OFFLOAD"
  echo "RESUME_DATALOADER_STATE=$RESUME_DATALOADER_STATE"
  echo "DIAGNOSTICS_DIR=$diagnostics_dir"
  echo "GLOBAL_TOP_K_SCHEDULE=$GLOBAL_TOP_K_SCHEDULE"
  echo "GLOBAL_INTERNALIZATION_MODE=$GLOBAL_INTERNALIZATION_MODE"
  echo "SKILLZERO_INCLUDE_TASK_SPECIFIC=$SKILLZERO_INCLUDE_TASK_SPECIFIC"
  echo "ONPOLICY_HELPFULNESS_EVAL_ENABLED=$ONPOLICY_HELPFULNESS_EVAL_ENABLED"
  echo "ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE=$ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE"
  echo "ONPOLICY_HELPFULNESS_DELTA_THRESHOLD=$ONPOLICY_HELPFULNESS_DELTA_THRESHOLD"
  echo "ONPOLICY_HELPFULNESS_PATIENCE=$ONPOLICY_HELPFULNESS_PATIENCE"
  echo "ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND=$ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND"
  echo "ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS=$ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS"
} | tee "$run_dir/run_config.env"

printf '%s\n' "${ppo_args[@]}" > "$run_dir/ppo_args.txt"

if [[ ! -f "$run_dir/latest_checkpointed_iteration.txt" && "${KEEP_OLD_METRICS:-0}" != "1" ]]; then
  : > "$run_dir/metrics.jsonl"
fi

# 启动训练
python3 -m verl.trainer.main_ppo "${ppo_args[@]}" "$@" \
  2>&1 | tee "$run_dir/output.log"

# 训练完成后生成可视化
echo ""
echo "训练完成！生成训练曲线..."
python3 scripts/plot_alfworld_metrics.py "$run_dir/metrics.jsonl"

echo ""
echo "============================================"
echo "训练完成！"
echo "============================================"
echo "输出目录: $run_dir"
echo "训练曲线: $run_dir/training_metrics.png"
echo ""
echo "断点续训："
echo "  export EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "  bash $0"
echo "============================================"
