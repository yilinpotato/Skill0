#!/usr/bin/env bash
# SkillRL 训练脚本 - 稳定版（解决 OOM + 优化内化效果）
# 基于原始脚本优化，添加了更保守的显存配置和更好的技能内化策略

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# 配置版本选择
# ============================================================================
# 使用环境变量 CONFIG_PROFILE 选择配置：
# - stable: 稳定版（低显存占用，适合单卡）
# - balanced: 平衡版（中等显存，适合多 GPU 或单卡大显存）
# - performance: 性能版（高显存，适合多卡）
CONFIG_PROFILE="${CONFIG_PROFILE:-stable}"

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
  shift
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_text_llama32_3b_global_internalize_lora_${CONFIG_PROFILE}}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if [[ "$ENGINE" == "vllm" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

# 检测 GPU 数量
if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
  n_gpus_per_node="$N_GPUS_PER_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *","* ]]; then
  n_gpus_per_node=1
else
  n_gpus_per_node=2
fi

# ============================================================================
# 配置文件：根据 CONFIG_PROFILE 设置参数
# ============================================================================

if [[ "$CONFIG_PROFILE" == "stable" ]]; then
  echo "=== 使用稳定配置（Stable Profile）==="
  echo "特点：低显存占用，适合单卡 24GB，不易 OOM"

  # 训练规模
  default_train_data_size=1
  default_val_data_size=8
  default_group_size=1                    # 降低 rollout 批次

  # 显存控制（保守）
  default_vllm_gpu_memory_utilization=0.25  # 从 0.32 降到 0.25
  default_vllm_max_num_batched_tokens=1536  # 从 2048 降到 1536
  default_vllm_max_num_seqs=1               # 从 2 降到 1

  # 训练批次
  default_ppo_mini_batch_size=1
  default_ppo_micro_batch_size_per_gpu=1
  default_log_prob_micro_batch_size_per_gpu=1

  # 序列长度
  default_max_prompt_length=2560            # 从 3072 降到 2560
  default_max_response_length=96            # 从 128 降到 96

  # LoRA 配置（增加容量以提升内化效果）
  default_lora_rank=64                      # 从 32 提升到 64
  default_lora_alpha=128                    # 从 64 提升到 128

  # 技能内化策略（减少技能数量）
  default_global_top_k_schedule="[6,6,3]"   # 从 [12,12,6] 降到 [6,6,3]

  # 训练步数（延长训练）
  default_total_training_steps=120          # 从 60 提升到 120

  # 奖励信号（强化）
  default_invalid_action_penalty_coef=0.05  # 从 0.02 提升到 0.05

  default_actor_optimizer_offload=True

elif [[ "$CONFIG_PROFILE" == "balanced" ]]; then
  echo "=== 使用平衡配置（Balanced Profile）==="
  echo "特点：中等显存占用，适合多 GPU 或单卡 40GB 以上"

  default_train_data_size=2
  default_val_data_size=8
  default_group_size=2

  default_vllm_gpu_memory_utilization=0.35
  default_vllm_max_num_batched_tokens=2048
  default_vllm_max_num_seqs=2

  default_ppo_mini_batch_size=2
  default_ppo_micro_batch_size_per_gpu=1
  default_log_prob_micro_batch_size_per_gpu=1

  default_max_prompt_length=3072
  default_max_response_length=128

  default_lora_rank=64
  default_lora_alpha=128

  default_global_top_k_schedule="[8,6,4]"
  default_total_training_steps=100
  default_invalid_action_penalty_coef=0.03

  default_actor_optimizer_offload=True

elif [[ "$CONFIG_PROFILE" == "performance" ]]; then
  echo "=== 使用性能配置（Performance Profile）==="
  echo "特点：高显存占用，适合多卡或单卡 80GB"

  default_train_data_size=4
  default_val_data_size=8
  default_group_size=4

  default_vllm_gpu_memory_utilization=0.45
  default_vllm_max_num_batched_tokens=4096
  default_vllm_max_num_seqs=4

  default_ppo_mini_batch_size=4
  default_ppo_micro_batch_size_per_gpu=2
  default_log_prob_micro_batch_size_per_gpu=2

  default_max_prompt_length=3072
  default_max_response_length=128

  default_lora_rank=128
  default_lora_alpha=256

  default_global_top_k_schedule="[12,8,4]"
  default_total_training_steps=150
  default_invalid_action_penalty_coef=0.02

  default_actor_optimizer_offload=False

else
  echo "错误：未知的 CONFIG_PROFILE: $CONFIG_PROFILE"
  echo "可选值：stable, balanced, performance"
  exit 1
fi

# ============================================================================
# 参数覆盖（环境变量优先级最高）
# ============================================================================

train_data_size="${TRAIN_DATA_SIZE:-$default_train_data_size}"
val_data_size="${VAL_DATA_SIZE:-$default_val_data_size}"
group_size="${GROUP_SIZE:-$default_group_size}"
total_training_steps="${TOTAL_TRAINING_STEPS:-$default_total_training_steps}"
total_epochs="${TOTAL_EPOCHS:-$total_training_steps}"
test_freq="${TEST_FREQ:-10}"
save_freq="${SAVE_FREQ:-10}"
max_steps="${MAX_STEPS:-50}"
max_prompt_length="${MAX_PROMPT_LENGTH:-$default_max_prompt_length}"
max_response_length="${MAX_RESPONSE_LENGTH:-$default_max_response_length}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-0.8}"
global_top_k_schedule="${GLOBAL_TOP_K_SCHEDULE:-$default_global_top_k_schedule}"
dense_reward="${DENSE_REWARD:-True}"
actor_lr="${ACTOR_LR:-5e-6}"
invalid_action_penalty_coef="${INVALID_ACTION_PENALTY_COEF:-$default_invalid_action_penalty_coef}"
lora_rank="${LORA_RANK:-$default_lora_rank}"
lora_alpha="${LORA_ALPHA:-$default_lora_alpha}"
lora_target_modules="${LORA_TARGET_MODULES:-all-linear}"
model_pad_token_id="${MODEL_PAD_TOKEN_ID:-128009}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-$default_ppo_mini_batch_size}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-$default_ppo_micro_batch_size_per_gpu}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$default_log_prob_micro_batch_size_per_gpu}"
vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-$default_vllm_gpu_memory_utilization}"
vllm_max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-$default_vllm_max_num_batched_tokens}"
vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-$default_vllm_max_num_seqs}"

if [[ -n "${VLLM_LOAD_FORMAT:-}" ]]; then
  vllm_load_format="$VLLM_LOAD_FORMAT"
elif [[ "$MODEL_PATH" == *"Llama"* || "$MODEL_PATH" == *"llama"* ]]; then
  vllm_load_format="safetensors"
else
  vllm_load_format="dummy_dtensor"
fi

actor_optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-$default_actor_optimizer_offload}"
num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

# 诊断目录
diagnostics_root="${DIAGNOSTICS_ROOT:-$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/diagnostics}"
diagnostics_dir="${DIAGNOSTICS_DIR:-$diagnostics_root/$(date +%Y%m%d_%H%M%S)}"
enable_resource_monitor="${ENABLE_RESOURCE_MONITOR:-1}"
export RESOURCE_DIAGNOSTICS_DIR="$diagnostics_dir"
trainer_logger="${TRAINER_LOGGER:-[console]}"
resume_dataloader_state="${RESUME_DATALOADER_STATE:-True}"

# ============================================================================
# 打印配置摘要
# ============================================================================

echo "============================================"
echo "SkillRL 训练配置摘要"
echo "============================================"
echo "配置文件: $CONFIG_PROFILE"
echo "实验名称: $EXPERIMENT_NAME"
echo "GPU 数量: $n_gpus_per_node"
echo ""
echo "--- 训练规模 ---"
echo "训练数据: $train_data_size"
echo "验证数据: $val_data_size"
echo "Rollout 组大小: $group_size"
echo "总训练步数: $total_training_steps"
echo ""
echo "--- 显存控制 ---"
echo "vLLM 显存占用: $vllm_gpu_memory_utilization"
echo "最大批次 tokens: $vllm_max_num_batched_tokens"
echo "最大并发序列: $vllm_max_num_seqs"
echo "PPO mini-batch: $ppo_mini_batch_size"
echo "PPO micro-batch/GPU: $ppo_micro_batch_size_per_gpu"
echo ""
echo "--- 序列长度 ---"
echo "最大 prompt: $max_prompt_length"
echo "最大 response: $max_response_length"
echo "Rollout 最大长度: $rollout_max_model_len"
echo ""
echo "--- LoRA 配置 ---"
echo "LoRA rank: $lora_rank"
echo "LoRA alpha: $lora_alpha"
echo "LoRA target: $lora_target_modules"
echo ""
echo "--- 技能内化 ---"
echo "Global top-k 调度: $global_top_k_schedule"
echo "密集奖励: $dense_reward"
echo "无效动作惩罚: $invalid_action_penalty_coef"
echo ""
echo "--- 输出路径 ---"
echo "输出目录: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
echo "诊断目录: $diagnostics_dir"
echo "============================================"
echo ""

# 保存配置快照
mkdir -p "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
config_snapshot="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/run_config.env"
cat > "$config_snapshot" <<EOF
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
CONFIG_PROFILE=$CONFIG_PROFILE
MODEL_PATH=$MODEL_PATH
EXPERIMENT_NAME=$EXPERIMENT_NAME
ENGINE=$ENGINE
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
n_gpus_per_node=$n_gpus_per_node
TRAIN_DATA_SIZE=$train_data_size
VAL_DATA_SIZE=$val_data_size
GROUP_SIZE=$group_size
TOTAL_TRAINING_STEPS=$total_training_steps
TOTAL_EPOCHS=$total_epochs
TEST_FREQ=$test_freq
SAVE_FREQ=$save_freq
MAX_STEPS=$max_steps
MAX_PROMPT_LENGTH=$max_prompt_length
MAX_RESPONSE_LENGTH=$max_response_length
ROLLOUT_MAX_MODEL_LEN=$rollout_max_model_len
ROLLOUT_TEMPERATURE=$rollout_temperature
GLOBAL_TOP_K_SCHEDULE=$global_top_k_schedule
DENSE_REWARD=$dense_reward
ACTOR_LR=$actor_lr
INVALID_ACTION_PENALTY_COEF=$invalid_action_penalty_coef
LORA_RANK=$lora_rank
LORA_ALPHA=$lora_alpha
LORA_TARGET_MODULES=$lora_target_modules
MODEL_PAD_TOKEN_ID=$model_pad_token_id
PPO_MINI_BATCH_SIZE=$ppo_mini_batch_size
PPO_MICRO_BATCH_SIZE_PER_GPU=$ppo_micro_batch_size_per_gpu
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$log_prob_micro_batch_size_per_gpu
VLLM_GPU_MEMORY_UTILIZATION=$vllm_gpu_memory_utilization
VLLM_MAX_NUM_BATCHED_TOKENS=$vllm_max_num_batched_tokens
VLLM_MAX_NUM_SEQS=$vllm_max_num_seqs
VLLM_LOAD_FORMAT=$vllm_load_format
ACTOR_OPTIMIZER_OFFLOAD=$actor_optimizer_offload
DIAGNOSTICS_DIR=$diagnostics_dir
TRAINER_LOGGER=$trainer_logger
RESUME_DATALOADER_STATE=$resume_dataloader_state
EOF

echo "配置已保存到: $config_snapshot"
echo ""

# ============================================================================
# 启动资源监控
# ============================================================================

if [[ "$enable_resource_monitor" == "1" ]]; then
  echo "启动资源监控..."
  mkdir -p "$diagnostics_dir"
  bash scripts/resource_monitor.sh "$diagnostics_dir" &
  monitor_pid=$!
  echo "资源监控 PID: $monitor_pid"
  trap "kill $monitor_pid 2>/dev/null || true" EXIT
fi

# ============================================================================
# 数据预处理
# ============================================================================

echo "准备数据..."
python3 -m examples.data_preprocess.prepare \
  --train_data_size "$train_data_size" \
  --val_data_size "$val_data_size"

# ============================================================================
# 构建训练参数
# ============================================================================

ppo_args=(
  data.train_files="$DATA_ROOT/train"
  data.val_files="$DATA_ROOT/val"
  data.train_batch_size="$train_data_size"
  data.val_batch_size="$val_data_size"
  data.max_prompt_length="$max_prompt_length"
  data.max_response_length="$max_response_length"

  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank="$lora_rank"
  actor_rollout_ref.model.lora_alpha="$lora_alpha"
  actor_rollout_ref.model.target_modules="$lora_target_modules"
  actor_rollout_ref.model.pad_token_id="$model_pad_token_id"

  actor_rollout_ref.actor.optim.lr="$actor_lr"
  actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size_per_gpu"
  actor_rollout_ref.actor.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch_size_per_gpu"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$actor_optimizer_offload"

  actor_rollout_ref.rollout.name="$ENGINE"
  actor_rollout_ref.rollout.temperature="$rollout_temperature"
  actor_rollout_ref.rollout.gpu_memory_utilization="$vllm_gpu_memory_utilization"
  actor_rollout_ref.rollout.max_model_len="$rollout_max_model_len"
  actor_rollout_ref.rollout.max_num_batched_tokens="$vllm_max_num_batched_tokens"
  actor_rollout_ref.rollout.max_num_seqs="$vllm_max_num_seqs"
  actor_rollout_ref.rollout.load_format="$vllm_load_format"

  env.env_name=alfworld_text
  env.max_steps="$max_steps"
  env.rollout.n="$group_size"
  env.alfworld.use_dense_reward="$dense_reward"
  env.alfworld.invalid_action_penalty_coef="$invalid_action_penalty_coef"
  env.skills_only_memory.global_top_k_schedule="$global_top_k_schedule"

  trainer.logger="$trainer_logger"
  trainer.project_name=skillrl_mvp
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node="$n_gpus_per_node"
  trainer.default_local_dir="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
  trainer.metrics_jsonl_path="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/metrics.jsonl"
  trainer.save_freq="$save_freq"
  trainer.test_freq="$test_freq"
  trainer.total_training_steps="$total_training_steps"
  trainer.total_epochs="$total_epochs"
  trainer.resume_dataloader_state="$resume_dataloader_state"
  trainer.val_before_train=True

  "$@"
)

# 保存参数列表
ppo_args_file="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/ppo_args.txt"
printf '%s\n' "${ppo_args[@]}" > "$ppo_args_file"
echo "训练参数已保存到: $ppo_args_file"
echo ""

# ============================================================================
# 启动训练
# ============================================================================

echo "开始训练..."
echo "日志输出: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/output.log"
echo ""

python3 -m verl.trainer.main_ppo "${ppo_args[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/output.log"

# ============================================================================
# 训练完成后自动生成可视化
# ============================================================================

echo ""
echo "训练完成！生成训练曲线..."
python3 plot.py --exp-dir "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME" --use-jsonl

echo ""
echo "============================================"
echo "训练完成摘要"
echo "============================================"
echo "实验名称: $EXPERIMENT_NAME"
echo "输出目录: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
echo ""
echo "查看结果："
echo "  训练日志: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/output.log"
echo "  训练曲线: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/training_metrics.png"
echo "  训练摘要: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/training_metrics_summary.txt"
echo "  Checkpoints: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/global_step_*/"
echo ""
echo "断点续训："
echo "  export EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "  bash $0"
echo "============================================"
