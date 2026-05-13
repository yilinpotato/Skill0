#!/usr/bin/env bash
# 阶段2：基于 SFT 预热模型进行 RL 微调（双 3090 优化版）

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# 配置
# ============================================================================

export MODEL_PATH="${SFT_MODEL_PATH:-/data2/myl/skillrl_outputs/sft_warmup/global_step_1000}"
export DATA_ROOT="${DATA_ROOT:-/data2/myl/skillrl_data/verl-agent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/myl/skillrl_outputs}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_rl_after_sft}"

# 检测 GPU 数量
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *","* ]]; then
  n_gpus_per_node=1
else
  n_gpus_per_node=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 2)
fi

# batch size（SFT 预热后可以用稍大的 batch）
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-4}"

# 学习率（SFT 预热后用稍高的 LR）
export ACTOR_LR="${ACTOR_LR:-2e-5}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"

# Entropy / reward
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.01}"
export NORMALIZE_REWARD="${NORMALIZE_REWARD:-True}"

# 技能内化调度（SFT 后可以更快减少技能依赖）
export GLOBAL_TOP_K_SCHEDULE="${GLOBAL_TOP_K_SCHEDULE:-[12,8,8,4,4,2,2,0]}"

# 其他
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export DENSE_REWARD="${DENSE_REWARD:-True}"
export INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.05}"

# 双 3090 显存参数
if [[ "$n_gpus_per_node" -ge 2 ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
  export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
  export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
  export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
  export VLLM_TP_SIZE="${VLLM_TP_SIZE:-$n_gpus_per_node}"
else
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
  export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
  export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-2}"
  export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
  export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
fi

echo "============================================"
echo "阶段 2：RL 微调（SFT 预热后，双 3090）"
echo "============================================"
echo "SFT 模型路径: $MODEL_PATH"
echo "实验名称: $EXPERIMENT_NAME"
echo "GPU 数量: $n_gpus_per_node"
echo "每步 rollout: $((TRAIN_DATA_SIZE * GROUP_SIZE))"
echo "============================================"
echo ""

# 数据准备
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$DATA_ROOT" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size 8

ppo_mini_batch_size=$((TRAIN_DATA_SIZE * GROUP_SIZE))

ppo_args=(
    algorithm.adv_estimator=grpo
    "data.train_files=$DATA_ROOT/text/train.parquet"
    "data.val_files=$DATA_ROOT/text/test.parquet"
    "data.train_batch_size=$TRAIN_DATA_SIZE"
    data.val_batch_size=8
    data.max_prompt_length=3072
    data.max_response_length=128
    data.filter_overlong_prompts=True
    data.return_raw_chat=True

    # 模型（从 SFT checkpoint 开始）
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "+actor_rollout_ref.model.override_config.pad_token_id=128009"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # 优化器
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "+actor_rollout_ref.actor.optim.warmup_steps=$LR_WARMUP_STEPS"
    "+actor_rollout_ref.actor.optim.lr_scheduler_type=cosine"
    "+actor_rollout_ref.actor.optim.total_steps=$TOTAL_TRAINING_STEPS"

    # PPO 配置
    "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl

    "+actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"
    "+actor_rollout_ref.actor.normalize_reward=$NORMALIZE_REWARD"
    "+actor_rollout_ref.actor.reward_norm_type=running"

    # FSDP
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"

    # Rollout（双卡 tensor parallelism）
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP_SIZE"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=0.8
    "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.max_model_len=3200
    "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True

    # Reference policy
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    # 无效动作惩罚
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    "actor_rollout_ref.actor.invalid_action_penalty_coef=$INVALID_ACTION_PENALTY_COEF"

    # 环境
    algorithm.use_kl_in_reward=False
    env.env_name=alfworld/AlfredTWEnv
    env.seed=0
    env.max_steps=50
    "env.rollout.n=$GROUP_SIZE"
    env.resources_per_worker.num_cpus=0.1
    "+env.alfworld.use_dense_reward=$DENSE_REWARD"

    # 技能内化
    +env.use_skills_only_memory=True
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json
    +env.skills_only_memory.top_k=12
    "+env.skills_only_memory.global_top_k_schedule=$GLOBAL_TOP_K_SCHEDULE"
    +env.skills_only_memory.task_specific_top_k=null
    +env.skills_only_memory.enable_dynamic_update=False
    +env.skills_only_memory.eval_internalization_modes=True

    # Trainer
    trainer.critic_warmup=0
    trainer.logger=[console]
    trainer.project_name=skillrl_mvp
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$n_gpus_per_node"
    trainer.nnodes=1
    "trainer.default_local_dir=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
    "+trainer.metrics_jsonl_path=$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/metrics.jsonl"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.total_epochs=$TOTAL_TRAINING_STEPS"
    +trainer.resume_dataloader_state=True
    trainer.val_before_train=True
)

mkdir -p "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "SFT_MODEL_PATH=$MODEL_PATH"
  echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
  echo "N_GPUS=$n_gpus_per_node"
  echo "TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE"
  echo "GROUP_SIZE=$GROUP_SIZE"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "ENTROPY_COEFF=$ENTROPY_COEFF"
  echo "VLLM_TP_SIZE=$VLLM_TP_SIZE"
  echo "GLOBAL_TOP_K_SCHEDULE=$GLOBAL_TOP_K_SCHEDULE"
} | tee "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/run_config.env"

printf '%s\n' "${ppo_args[@]}" > "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/ppo_args.txt"

python3 -m verl.trainer.main_ppo "${ppo_args[@]}" "$@" \
  2>&1 | tee "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/output.log"

echo ""
echo "训练完成！生成训练曲线..."
python3 plot.py --exp-dir "$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME" --use-jsonl

echo ""
echo "============================================"
echo "RL 训练完成！"
echo "============================================"
echo "输出目录: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
echo "训练曲线: $OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME/training_metrics.png"
echo "============================================"
