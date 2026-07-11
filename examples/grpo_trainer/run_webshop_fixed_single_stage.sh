#!/usr/bin/env bash
# WebShop GRPO single-stage training for two A800 80GB GPUs.
#
# Reference: examples/grpo_trainer/run_webshop_skills.sh (paper setup: 8 GPUs,
# Qwen2.5-7B already SFT'd on skills, full fine-tuning, train_data_size=16,
# group_size=8, val_data_size=64). Values here follow that script wherever the
# 2xA800 hardware and the different base model (Qwen3-4B-Thinking-2507,
# intentionally kept) allow it. Deliberate deviations from the paper values,
# and why:
#   - LoRA (rank/alpha) instead of full fine-tuning, and both fsdp param /
#     optimizer offload disabled: full FT of a thinking model does not fit
#     on 2 GPUs the way it did on 8; LoRA's much smaller trainable/optimizer
#     footprint makes the offloads the paper needed largely unnecessary.
#   - train_data_size=12 / group_size=6 / val_data_size=32 (paper: 16/8/64):
#     scaled down proportionally for 2 GPUs; ppo_mini_batch_size=36 keeps the
#     paper's mini_batch/total_batch ratio of 0.5 (36/72 vs the paper's 64/128).
#   - data.max_prompt_length=8192 / max_response_length=4096 (paper: 6000/768):
#     Qwen3-Thinking emits a much longer <think> block than the paper's
#     non-thinking Qwen2.5 checkpoint, so both budgets were raised accordingly.
#   - rollout/ref micro-batch sizes, tensor_model_parallel_size=1,
#     gpu_memory_utilization=0.8, max_num_batched_tokens=16384: re-tuned for
#     2 GPUs + LoRA + the larger response budget above; not paper values.
# Everything else (max_steps=15, invalid_action_penalty_coef=0.1, kl_loss_coef,
# actor lr=1e-6, save_freq=10, test_freq=5, val_before_train=False,
# data.truncation=left, skills_only_memory top_k) matches the paper script
# as-is. Skill memory is configured for LOCAL, progressive internalization
# only (global_top_k_schedule + skillzero + on-policy helpfulness pruning,
# mirroring run_alfworld_fixed_single_stage.sh) - enable_dynamic_update (the
# separate Azure/o3 cloud skill-generation path) is explicitly disabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export TRACE_SH="${TRACE_SH:-0}"
if [[ "$TRACE_SH" == "1" ]]; then
  set -x
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-debug}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

# ── 自动判断运行环境：超算 vs 本地3090（与 ALFWorld 启动脚本一致）──────────────
if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
  RUN_ENV="超算 (supercomputer)"
  export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
  export DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
  DEFAULT_RAY_NUM_CPUS=56
else
  RUN_ENV="本地3090 (local)"
  export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
  export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/skillrl_data/verl-agent}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
  DEFAULT_RAY_NUM_CPUS="$(nproc)"
fi

# Respect user GPU selection; otherwise use at most two GPUs on the server and
# the single available GPU on a local 3090 machine.
if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "0" ]]; then
    echo "Local shared-server launcher only permits CUDA_VISIBLE_DEVICES=0." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES=0
  GPU0_ACTIVE_PIDS=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)
  if [[ -n "$GPU0_ACTIVE_PIDS" ]]; then
    echo "GPU 0 is in use by PID(s): $GPU0_ACTIVE_PIDS. Refusing to start." >&2
    exit 1
  fi
fi
NUM_VISIBLE_GPUS=${NUM_VISIBLE_GPUS:-1}

export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_qwen3_4b_thinking_2xa800_v7}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# vLLM V1 uses its own CuMem memory pool. PyTorch expandable segments are
# incompatible with that pool and can also make GPU memory diagnosis misleading.
unset PYTORCH_CUDA_ALLOC_CONF

# Each Python/Ray worker consumes many mmaps. The conservative defaults below
# keep the total number of workers well below a typical unprivileged limit.
sudo sysctl -w vm.max_map_count=1048576 2>/dev/null || true

# ============================================================================
# GPU configuration: automatically 2 GPUs on the supercomputer, 1 on local 3090.
# ============================================================================
n_gpus_per_node="${N_GPUS_PER_NODE:-$NUM_VISIBLE_GPUS}"

echo "============================================"
echo "WebShop GRPO training: $RUN_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Detected GPU count: $n_gpus_per_node"
echo "CACHE_ROOT: $CACHE_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "============================================"
echo ""

# Reference WebShop rollout budget for two A800 GPUs:
# 12 tasks x 6 rollouts = 72 trajectories per optimizer step.
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-12}"
export GROUP_SIZE="${GROUP_SIZE:-6}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
# examples/grpo_trainer/run_webshop_skills.sh (paper reference) uses 0.1 per
# env worker; WebShop's env actors are lightweight (in-process HTML parsing,
# no browser), so there is no hardware reason to deviate from that value.
export ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"

export ACTOR_LR="${ACTOR_LR:-1e-6}"

# WebShop-specific environment settings. use_small=True matches the bundled
# default 1,000-product dataset and keeps two-GPU rollouts practical.
export WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-True}"
export WEBSHOP_HUMAN_GOALS="${WEBSHOP_HUMAN_GOALS:-False}"
export MAX_STEPS="${MAX_STEPS:-15}"

# Progressive skill internalization (mirrors run_alfworld_fixed_single_stage.sh):
# the number of general/global skills injected into the prompt fades from
# top_k down to 0 over training, forcing the policy to internalize them
# instead of leaning on retrieved text forever. No cloud/Azure o3 calls are
# involved in this path (that's the separate, disabled enable_dynamic_update
# mechanism below) - it's purely a local schedule over already-authored skills.
export GLOBAL_TOP_K_SCHEDULE="${GLOBAL_TOP_K_SCHEDULE:-[6,6,6,4,4,4,3,3,3,2,2,2,1,1,1,0]}"
export GLOBAL_INTERNALIZATION_MODE="${GLOBAL_INTERNALIZATION_MODE:-skillzero}"
# WebShop has no discrete task-type taxonomy the way ALFWorld does (no
# env.alfworld.eval_task_type equivalent), so the task-specific half of
# skillzero has nothing to act on here; left False to say so honestly rather
# than implying a no-op knob does something.
export SKILLZERO_INCLUDE_TASK_SPECIFIC="${SKILLZERO_INCLUDE_TASK_SPECIFIC:-False}"
export ONPOLICY_HELPFULNESS_EVAL_ENABLED="${ONPOLICY_HELPFULNESS_EVAL_ENABLED:-True}"
export ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE="${ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE:-True}"
export ONPOLICY_HELPFULNESS_DELTA_THRESHOLD="${ONPOLICY_HELPFULNESS_DELTA_THRESHOLD:-0.01}"
export ONPOLICY_HELPFULNESS_PATIENCE="${ONPOLICY_HELPFULNESS_PATIENCE:-1}"
export ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND="${ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND:-1}"
export ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS="${ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS:-0}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-5}"
export RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.1}"
export PRINT_TRAJECTORIES="${PRINT_TRAJECTORIES:-False}"
export WANDB_PROJECT="${WANDB_PROJECT:-skillrl_mvp}"
export WANDB_NAME="${WANDB_NAME:-$EXPERIMENT_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-webshop_fixed_single_stage_2xa800}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT/wandb}"
# Paper reference (run_webshop_skills.sh) logs to both console and wandb;
# restored here (this script had silently dropped wandb from the logger list).
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
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
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# Dual-A800 FSDP with one independent vLLM rollout worker per GPU.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
# Paper reference uses 256; this had been left at 32 (the ALFWorld sibling
# script's default, apparently carried over when this script was authored).
# max_num_seqs is a vLLM scheduling cap, not a memory reservation, so raising
# it back to the paper value just avoids needlessly serializing concurrent
# rollouts and does not by itself increase KV-cache/memory usage.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
# 4,096-token WebShop rollouts need a smaller per-GPU backward micro-batch on
# two A800s. Keep the 72-trajectory rollout, but trade throughput for memory.
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-36}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
export RESUME_DATALOADER_STATE="${RESUME_DATALOADER_STATE:-False}"
export ENABLE_RESOURCE_MONITOR="${ENABLE_RESOURCE_MONITOR:-1}"

run_dir="$OUTPUT_ROOT/skillrl_mvp/$EXPERIMENT_NAME"
mkdir -p "$run_dir" "$WANDB_DIR"
# Isolate parquet dataset metadata/cache from other supercomputer jobs. Shared
# global Hugging Face caches can leave or race on *.incomplete directories.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$run_dir/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"

resume_args=()
if [[ -n "$RESUME_FROM_STEP" ]]; then
  resume_ckpt="$run_dir/global_step_$RESUME_FROM_STEP"
  if [[ ! "$RESUME_FROM_STEP" =~ ^[0-9]+$ ]]; then
    echo "RESUME_FROM_STEP must be an integer, got: $RESUME_FROM_STEP" >&2
    exit 1
  fi
  if [[ ! -d "$resume_ckpt/actor" ]]; then
    echo "Requested checkpoint does not exist: $resume_ckpt/actor" >&2
    echo "Available checkpoints under $run_dir:" >&2
    find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' -printf '  %f\n' 2>/dev/null | sort -V >&2 || true
    exit 1
  fi
  resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$resume_ckpt")
  echo "[resume] forcing restart from $resume_ckpt"
else
  echo "[resume] auto mode: will resume from latest checkpoint if present"
fi

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
  --mode text \
  --local_dir "$DATA_ROOT" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_DATA_SIZE"

ppo_args=(
  algorithm.adv_estimator=grpo
  "data.train_files=$DATA_ROOT/text/train.parquet"
  "data.val_files=$DATA_ROOT/text/test.parquet"
  "data.train_batch_size=$TRAIN_DATA_SIZE"
  "data.val_batch_size=$VAL_DATA_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=True
  data.truncation=left
  data.return_raw_chat=True

  "actor_rollout_ref.model.path=$MODEL_PATH"
  "++actor_rollout_ref.model.override_config.pad_token_id=128009"
  "actor_rollout_ref.model.lora_rank=$LORA_RANK"
  "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"

  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP_SIZE"
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
  "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFFLOAD"

  actor_rollout_ref.actor.use_invalid_action_penalty=True
  "actor_rollout_ref.actor.invalid_action_penalty_coef=$INVALID_ACTION_PENALTY_COEF"
  algorithm.use_kl_in_reward=False

  env.env_name=Webshop
  env.seed=0
  "env.max_steps=$MAX_STEPS"
  env.history_length=8
  "env.rollout.n=$GROUP_SIZE"
  "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
  "++env.webshop.use_small=$WEBSHOP_USE_SMALL"
  "++env.webshop.human_goals=$WEBSHOP_HUMAN_GOALS"

  ++env.use_skills_only_memory=True
  ++env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json
  ++env.skills_only_memory.top_k=6
  # Progressive skill internalization only - no cloud/Azure o3 calls.
  # enable_dynamic_update (set False below) is the *separate* o3-based
  # skill-generation path (see SkillUpdater); intentionally left off here.
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

  trainer.critic_warmup=0
  "trainer.logger=$TRAINER_LOGGER"
  "trainer.project_name=$WANDB_PROJECT"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$n_gpus_per_node"
  trainer.nnodes=1
  trainer.ray_wait_register_center_timeout=1200
  "trainer.default_local_dir=$run_dir"
  "++trainer.log_val_generations=$LOG_VAL_GENERATIONS"
  "++trainer.wandb_log_trajectories=$WANDB_LOG_TRAJECTORIES"
  "++trainer.wandb_log_contexts=$WANDB_LOG_CONTEXTS"
  "++trainer.wandb_max_trajectories=$WANDB_MAX_TRAJECTORIES"
  "++trainer.wandb_context_samples_per_rollout=$WANDB_CONTEXT_SAMPLES_PER_ROLLOUT"
  "++trainer.wandb_max_trajectory_steps=$WANDB_MAX_TRAJECTORY_STEPS"
  "++trainer.wandb_max_text_chars=$WANDB_MAX_TEXT_CHARS"
  "++trainer.wandb_max_summary_rows=$WANDB_MAX_SUMMARY_ROWS"
  "++trainer.wandb_max_step_rows=$WANDB_MAX_STEP_ROWS"
  "++trainer.metrics_jsonl_path=$run_dir/metrics.jsonl"
  "++trainer.trajectory_log_path=$run_dir/trajectories.json"
  "++trainer.context_log_path=$run_dir/contexts.json"
  # Human-readable, step-by-step trajectory dump (observation / prompt / model
  # reasoning / raw output / action / env result) for debugging, alongside the
  # machine-readable trajectories.json. Needs trajectory_include_prompt=True so
  # the "[PROMPT SENT TO MODEL]" section is actually populated. Only dumped
  # every trajectory_log_every_n_steps training steps to keep the I/O cost of
  # rewriting trajectories.json down over a long run; validation dumps are
  # unaffected (already gated by trainer.test_freq).
  "++trainer.readable_trajectory_log_path=$run_dir/trajectories_readable.txt"
  ++trainer.trajectory_include_prompt=True
  "++trainer.trajectory_log_every_n_steps=${TRAJECTORY_LOG_EVERY_N_STEPS:-1}"
  "++trainer.print_trajectories=$PRINT_TRAJECTORIES"
  "++trainer.compact_console_output=$COMPACT_CONSOLE_OUTPUT"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.total_epochs=$TOTAL_TRAINING_STEPS"
  "++trainer.resume_dataloader_state=$RESUME_DATALOADER_STATE"
  "${resume_args[@]}"
  trainer.val_before_train=False
)

# Scheduler-provided Ray clusters reject local CPU/GPU declarations.  Match
# SkillRL: pass a CPU limit only when this process creates its own Ray runtime.
if [[ -z "${RAY_ADDRESS:-}" ]]; then
  ppo_args+=("ray_init.num_cpus=$RAY_NUM_CPUS")
else
  echo "Using existing Ray cluster at RAY_ADDRESS=$RAY_ADDRESS"
fi

{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  for key in MODEL_PATH PROJECT_ROOT CACHE_ROOT DATA_ROOT OUTPUT_ROOT EXPERIMENT_NAME WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_DIR; do
    echo "$key=${!key}"
  done
  for key in TRAIN_DATA_SIZE GROUP_SIZE VAL_DATA_SIZE ENV_WORKER_CPUS RAY_NUM_CPUS ACTOR_LR TOTAL_TRAINING_STEPS SAVE_FREQ TEST_FREQ RESUME_FROM_STEP LORA_RANK LORA_ALPHA INVALID_ACTION_PENALTY_COEF MAX_STEPS WEBSHOP_USE_SMALL WEBSHOP_HUMAN_GOALS MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_NUM_SEQS VLLM_TP_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_PER_GPU REF_LOG_PROB_MICRO_BATCH_PER_GPU OPTIMIZER_OFFLOAD RESUME_DATALOADER_STATE GLOBAL_TOP_K_SCHEDULE GLOBAL_INTERNALIZATION_MODE SKILLZERO_INCLUDE_TASK_SPECIFIC ONPOLICY_HELPFULNESS_EVAL_ENABLED ONPOLICY_HELPFULNESS_AUTO_INTERNALIZE ONPOLICY_HELPFULNESS_DELTA_THRESHOLD ONPOLICY_HELPFULNESS_PATIENCE ONPOLICY_HELPFULNESS_REMOVE_PER_ROUND ONPOLICY_HELPFULNESS_MIN_ACTIVE_SKILLS; do
    echo "$key=${!key}"
  done
  echo "N_GPUS=$n_gpus_per_node"
  echo "ROLLOUTS_PER_STEP=$((TRAIN_DATA_SIZE * GROUP_SIZE))"
  echo "DIAGNOSTICS_DIR=$diagnostics_dir"
} | tee "$run_dir/run_config.env"

printf '%s\n' "${ppo_args[@]}" > "$run_dir/ppo_args.txt"

if [[ ! -f "$run_dir/latest_checkpointed_iteration.txt" && "${KEEP_OLD_METRICS:-0}" != "1" ]]; then
  : > "$run_dir/metrics.jsonl"
fi

python3 -m verl.trainer.main_ppo "${ppo_args[@]}" "$@" 2>&1 | tee "$run_dir/output.log"

echo ""
echo "Training completed; generating metric plots..."
python3 scripts/plot_webshop_metrics.py "$run_dir/metrics.jsonl"

echo ""
echo "============================================"
echo "Training completed"
echo "============================================"
echo "Output directory: $run_dir"
echo "Training curve: $run_dir/training_metrics.png"
echo "Resume automatically: bash $0"
echo "Force resume, e.g. step 10: RESUME_FROM_STEP=10 bash $0"
echo "============================================"
