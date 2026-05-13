#!/usr/bin/env bash
# 阶段1：收集专家轨迹 + SFT 预热
# 先用 scripts/collect_sft_trajectories.py 收集成功轨迹，再用 fsdp_sft_trainer 训练

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# 配置
# ============================================================================

export MODEL_PATH="${MODEL_PATH:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
export SKILLS_JSON="${SKILLS_JSON:-memory_data/alfworld/claude_style_skills.json}"
export TRAJ_DIR="${TRAJ_DIR:-$OUTPUT_ROOT/expert_trajectories}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$OUTPUT_ROOT/sft_warmup}"

# 强制使用单卡 GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_GPUS=1

# 轨迹收集参数（单卡 GPU 0）
export COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-16}"
export COLLECT_MIN_TRAJS="${COLLECT_MIN_TRAJS:-120}"
export COLLECT_MAX_ROUNDS="${COLLECT_MAX_ROUNDS:-20}"
export COLLECT_GPU_MEM="${COLLECT_GPU_MEM:-0.85}"   # 单卡独占，可以用更高
export COLLECT_TP_SIZE="${COLLECT_TP_SIZE:-1}"       # 单卡 TP=1
export COLLECT_TEMPERATURE="${COLLECT_TEMPERATURE:-0.5}"
export COLLECT_MAX_MODEL_LEN="${COLLECT_MAX_MODEL_LEN:-16384}"

# SFT 训练参数
export SFT_LR="${SFT_LR:-1e-4}"
export SFT_TOTAL_STEPS="${SFT_TOTAL_STEPS:-1000}"
export SFT_TRAIN_BATCH="${SFT_TRAIN_BATCH:-8}"
export SFT_MICRO_BATCH="${SFT_MICRO_BATCH:-2}"
export SFT_LORA_RANK="${SFT_LORA_RANK:-64}"
export SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-128}"

echo "============================================"
echo "阶段 1：收集专家轨迹 + SFT 预热"
echo "============================================"
echo "模型路径: $MODEL_PATH"
echo "技能文件: $SKILLS_JSON"
echo "轨迹输出: $TRAJ_DIR"
echo "SFT 输出: $SFT_OUTPUT_DIR"
echo "SFT GPU 数量: $N_GPUS"
echo "============================================"
echo ""

# ============================================================================
# Step 1：收集轨迹
# ============================================================================

echo "Step 1: 用 vLLM 在 ALFWorld 中收集成功轨迹..."
echo ""

python3 scripts/collect_sft_trajectories.py \
  --model-path "$MODEL_PATH" \
  --skills-json "$SKILLS_JSON" \
  --output-dir "$TRAJ_DIR" \
  --num-envs "$COLLECT_NUM_ENVS" \
  --min-trajectories "$COLLECT_MIN_TRAJS" \
  --max-rounds "$COLLECT_MAX_ROUNDS" \
  --gpu-memory-utilization "$COLLECT_GPU_MEM" \
  --tensor-parallel-size "$COLLECT_TP_SIZE" \
  --max-model-len "$COLLECT_MAX_MODEL_LEN" \
  --temperature "$COLLECT_TEMPERATURE"

echo ""
echo "轨迹已保存到: $TRAJ_DIR"
ls -lh "$TRAJ_DIR"/*.parquet 2>/dev/null || true
echo ""

# ============================================================================
# Step 2：SFT 训练
# ============================================================================

echo "Step 2: 使用收集的轨迹进行 SFT 训练..."
echo ""

mkdir -p "$SFT_OUTPUT_DIR"

# 使用 torchrun + fsdp_sft_trainer
# 注意：multi-turn 格式，messages 列
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$N_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAJ_DIR/train.parquet" \
    data.val_files="$TRAJ_DIR/val.parquet" \
    data.train_batch_size="$SFT_TRAIN_BATCH" \
    data.micro_batch_size_per_gpu="$SFT_MICRO_BATCH" \
    data.max_length=130000 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    model.partial_pretrain="$MODEL_PATH" \
    model.lora_rank="$SFT_LORA_RANK" \
    model.lora_alpha="$SFT_LORA_ALPHA" \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    optim.lr="$SFT_LR" \
    optim.warmup_steps_ratio=0.05 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir="$SFT_OUTPUT_DIR" \
    trainer.project_name=skillrl_mvp \
    trainer.experiment_name=sft_warmup \
    trainer.total_training_steps="$SFT_TOTAL_STEPS" \
    trainer.logger=[console] \
  2>&1 | tee "$SFT_OUTPUT_DIR/sft_training.log"

echo ""
echo "============================================"
echo "SFT 预热完成！"
echo "============================================"
echo "模型路径: $SFT_OUTPUT_DIR"
echo ""
echo "下一步：运行阶段2 RL 训练"
echo "  export SFT_MODEL_PATH=\$(ls -d $SFT_OUTPUT_DIR/global_step_* | sort -V | tail -1)"
echo "  bash examples/grpo_trainer/run_alfworld_stage2_rl.sh"
echo "============================================"
