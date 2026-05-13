# SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

<div align="center">

Bridging the gap between raw experience and policy improvement through automatic skill discovery.

</div>

<p align="center">
<img src="figs/pipeline.png" width="80%" alt="SKILLRL Pipeline Overview">
</p>

## 🔥 News

- **[04/03/2026]** Released the SFT dataset on [🤗HF](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data)!
- **[03/02/2026]** Due to an accidental misconfiguration, we lost several hundred GitHub stars. If you previously starred this repo, we'd appreciate a re-star ⭐!
- **[02/23/2026]** We released all the model checkpoints on HuggingFace! Feel free to use them as warm starts for continued RL training.
- **[02/18/2026]** The code of SkillRL was released!
- **[02/10/2026]** SkillRL paper was released on [arXiv](https://arxiv.org/abs/2602.08234)!

## 📖 Overview

SkillRL is a framework that enables LLM agents to learn high-level, reusable behavioral patterns from past experiences. While traditional memory-based methods store redundant and noisy raw trajectories, SKILLRL abstracts these into a hierarchical skill library.

## 🤖 Key Features

- **Experience-based Skill Distillation**: Transforms successful trajectories into strategic patterns and failed ones into concise lessons from failure.

- **Hierarchical SKILLBANK**: Organizes knowledge into General Skills for universal strategic guidance and Task-Specific Skills for category-level heuristics.

- **Recursive Skill Evolution**: A dynamic mechanism where the skill library co-evolves with the agent's policy during RL by analyzing validation failures.

- **Context Efficiency**: Achieves 10-20% token compression compared to raw trajectory storage while enhancing reasoning utility. 

---

## ALFWorld Global Skill Internalization MVP

This workspace includes a small-model MVP for **ALFWorld text-only global skill internalization**. The goal is to use SkillRL as the execution/RL framework and fade global skills out of the prompt during GRPO so the local model learns the general behavior patterns in its parameters. This MVP does not implement cloud-side skill extraction, does not use SkillBench, and does not use old checkpoints.

### Design

- **Execution framework**: SkillRL GRPO multi-turn ALFWorld environment.
- **Default model**: local SFT checkpoint `/data2/myl/qwen3-4b`.
- **Skills**: `memory_data/alfworld/claude_style_skills.json`.
- **Internalized context**: `general_skills` plus `common_mistakes`.
- **External context retained**: `task_specific_skills`, so the first demo only tests global-skill internalization.
- **Curriculum**: `global_top_k_schedule=[12,6,0]`, meaning global skills are fully visible early, partially visible in the middle, and removed late in training.
- **Evaluation contrast**: final validation also logs `val/external_global/...` and `val/internalized_global_off/...` metrics.

### Run

```bash
conda activate skillRL
cd /data2/myl/SkillRL

# Fix ALFWorld's default ~/.cache/alfworld lookup without moving data back to /home.
bash scripts/setup_alfworld_cache.sh

# Run the 0.5B text-only internalization MVP.
bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh
```

The script writes generated parquet carrier data to `/data2/myl/skillrl_data/verl-agent/text` and checkpoints/log outputs to `/data2/myl/skillrl_outputs`, avoiding `/home` storage pressure. It records validation every `TEST_FREQ=10` steps and saves checkpoints every `SAVE_FREQ=10` steps by default.

Important output files:

```text
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/output.log
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/metrics.jsonl
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/run_config.env
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/ppo_args.txt
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/global_step_<N>/actor/
```

`metrics.jsonl` is the easiest file for comparing progress across checkpoints. Checkpoints are saved at steps 10, 20, 30, ...

Default MVP sizing is intentionally conservative for Qwen3-4B on RTX 3090-class GPUs:

- single GPU: `TRAIN_DATA_SIZE=2`, `VAL_DATA_SIZE=4`, `GROUP_SIZE=1`
- two GPUs: `TRAIN_DATA_SIZE=4`, `VAL_DATA_SIZE=8`, `GROUP_SIZE=2`
- `TOTAL_TRAINING_STEPS=60`
- `TEST_FREQ=10`
- `SAVE_FREQ=10`
- `LORA_RANK=32`
- actor/ref FSDP model dtype is `bf16`, because FlashAttention2 does not support fp32 model weights.

For single-GPU debugging, expose only GPU0. The script detects a single visible GPU and automatically switches to smaller defaults: `TRAIN_DATA_SIZE=4`, `VAL_DATA_SIZE=4`, `GROUP_SIZE=2`, `PPO_MICRO_BATCH_SIZE_PER_GPU=1`, lower vLLM memory reservation, and actor optimizer offload.

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh vllm
```

For a faster single-GPU debug pass, reduce validation and episode length. This is useful for checking the end-to-end loop, but `MAX_STEPS < 50` changes the ALFWorld task horizon and should not be used for the final MVP result.

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_DATA_SIZE=2 VAL_DATA_SIZE=2 GROUP_SIZE=1 \
TOTAL_TRAINING_STEPS=10 TEST_FREQ=10 MAX_STEPS=25 \
MAX_RESPONSE_LENGTH=256 \
  bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh vllm
```

Reducing `VAL_DATA_SIZE` and increasing `TEST_FREQ` mainly saves evaluation time. Reducing `TRAIN_DATA_SIZE`, `GROUP_SIZE`, or `MAX_STEPS` changes the training signal and is best treated as debugging only.

For a higher-signal single-GPU MVP run, keep the full ALFWorld horizon and let the script use dense progress reward plus a lower rollout temperature:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_DATA_SIZE=4 VAL_DATA_SIZE=4 GROUP_SIZE=2 \
TOTAL_TRAINING_STEPS=80 TEST_FREQ=20 MAX_STEPS=50 \
MAX_RESPONSE_LENGTH=256 ROLLOUT_TEMPERATURE=0.6 \
  bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh vllm
```

For short debug runs (`TOTAL_TRAINING_STEPS<=30`), the script keeps `GLOBAL_TOP_K_SCHEDULE=[12]` by default so you are measuring the environment loop and action quality instead of immediately turning global skills off. Longer runs default to `[12,6,0]` for internalization.

### Resource Diagnostics

The ALFWorld MVP script starts a lightweight resource monitor by default. Logs are written under:

```text
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/diagnostics/<timestamp>/
```

Key files:

- `resource_monitor.log`: periodic `nvidia-smi`, system memory, and top process snapshots.
- `resource_events.log`: threshold events and the Python/Ray worker PIDs that were signaled.
- `python_stack_pid*.log`: Python stack traces and PyTorch CUDA memory summaries captured when GPU or system memory crosses the threshold.

Useful knobs:

```bash
ENABLE_RESOURCE_MONITOR=1
RESOURCE_MONITOR_INTERVAL=5
GPU_MEMORY_WARN_PCT=90
CPU_MEMORY_WARN_PCT=92
RESOURCE_TRACE_COOLDOWN=60
ENABLE_RESOURCE_TRACE_SIGNAL=0
DIAGNOSTICS_DIR=/data2/myl/skillrl_outputs/debug_diagnostics
```

By default the monitor only records threshold events. Set `ENABLE_RESOURCE_TRACE_SIGNAL=1` to send `SIGUSR1` to Python/Ray workers and dump the current Python stack plus CUDA memory summary. Leave it off for normal training because some Ray/vLLM subprocesses can exit if they receive `SIGUSR1` before the Python handler is installed.

### Results And Evaluation

Training metrics are printed to the console because the MVP uses `trainer.logger=['console']`. Check the terminal/tmux scrollback for lines such as:

```text
step:<N> - ... episode/success_rate:... val/success_rate:...
Final validation metrics: {...}
```

Diagnostics are under:

```text
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/diagnostics/
```

Checkpoints are saved under:

```text
/data2/myl/skillrl_outputs/skillrl_mvp/<experiment>/global_step_<N>/
```

The MVP script now defaults `SAVE_FREQ` to `TOTAL_TRAINING_STEPS`, so a final checkpoint is saved. To disable checkpointing:

```bash
SAVE_FREQ=-1 bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh vllm
```

Run base-model evaluation only:

```bash
CUDA_VISIBLE_DEVICES=0 \
VAL_DATA_SIZE=8 MAX_STEPS=50 MAX_RESPONSE_LENGTH=256 \
  bash examples/grpo_trainer/eval_alfworld_global_internalize_0_5b.sh vllm
```

Run base-vs-trained evaluation after training:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAINED_CKPT=/data2/myl/skillrl_outputs/skillrl_mvp/alfworld_text_qwen25_0_5b_global_internalize/global_step_80 \
VAL_DATA_SIZE=8 MAX_STEPS=50 MAX_RESPONSE_LENGTH=256 \
  bash examples/grpo_trainer/eval_alfworld_global_internalize_0_5b.sh vllm
```

Summarize logs:

```bash
python scripts/summarize_alfworld_eval.py \
  /data2/myl/skillrl_outputs/evals/alfworld_text_qwen25_0_5b_global_internalize
```

The original larger probe size can still be requested explicitly when the node has enough free RAM:

```bash
TRAIN_DATA_SIZE=16 VAL_DATA_SIZE=64 GROUP_SIZE=8 \
  bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh
```

For a fast smoke test:

```bash
TOTAL_TRAINING_STEPS=2 TEST_FREQ=1 \
  bash examples/grpo_trainer/run_alfworld_global_internalize_0_5b.sh
```

### Hardware Note

Two RTX 3090 GPUs are reasonable for this 0.5B full-parameter GRPO MVP. Full-parameter 8B GRPO on two 3090s is not recommended because actor, reference policy, vLLM rollout, optimizer state, and long ALFWorld prompts create high memory pressure and poor throughput. For 8B, prefer LoRA/QLoRA or more GPUs.

---

## 📥 Model Download

You can directly download the model weights by following the links below.

<table>
  <thead>
    <tr>
      <th align="center">Task</th>
      <th align="center">Model</th>
      <th align="center">Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2">🧭 ALFWorld</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🛍️ WebShop</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🔍 Search</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-RL">🤗 HuggingFace</a></td>
    </tr>
  </tbody>
</table>


---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/aiming-lab/SkillRL.git
cd SkillRL

pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .

pip install openai
```

### Environment Setup

**ALFWorld**
```bash
pip install alfworld
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0

# Download PDDL & Game files and pre-trained MaskRCNN detector
alfworld-download -f
```

**WebShop**
```bash
cd agent_system/environments/env_package/webshop
./setup.sh -d all
```

**Search**
```bash
cd agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

**API Setup**
```
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT=""
```

---

## 🏃 Training

### Memory Data Generation
The first step of our training pipeline uses the base model to generate memory data. This data serves as the foundation for the agent's initial experiences. The specific prompt used to guide this generation can be found at: `memory_data/prompt/prompt.txt`.

### Supervised Fine-Tuning (SFT)
Prior to RL, we perform SFT to endow the model with basic task capabilities and instruction-following alignment. We use [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) as our framework for the SFT stage. The SFT data was released on [🤗 HF](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data) now!

### RL With SkillBank

#### Template Mode

Template mode uses keyword matching to detect the task category and injects all skills for that category into the prompt.  No embedding model is required.

```bash
# ALFWorld
export MODEL_PATH=YOUR_SFT_CKPT
bash examples/grpo_trainer/run_alfworld_skills.sh

# WebShop
bash examples/grpo_trainer/run_webshop_skills.sh

# Search
bash examples/grpo_trainer/run_search_skills.sh
```

Key config flags added by these scripts:

```
+env.use_skills_only_memory=True
+env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json
+env.skills_only_memory.top_k=6              
+env.skills_only_memory.enable_dynamic_update=True
+env.skills_only_memory.update_threshold=0.4
+env.skills_only_memory.max_new_skills=3
```

#### Embedding Mode

Embedding mode uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) to rank skills by semantic similarity to the task description.  Both general skills and task-specific skills are searched cross-category and only the top-k most relevant are injected.  Skill embeddings are pre-computed once at startup.

```bash
export MODEL_PATH=YOUR_SFT_CKPT

python3 -m verl.trainer.main_ppo \
    ... \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=Qwen/Qwen3-Embedding-0.6B \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5
```

---

## ⚙️ Skill Memory Configuration

All parameters live under `env.skills_only_memory.*` (Hydra / OmegaConf).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skills_json_path` | str | — | **Required.** Path to the skills JSON. |
| `retrieval_mode` | str | `"template"` | `"template"` or `"embedding"`. |
| `embedding_model_path` | str | `"Qwen/Qwen3-Embedding-0.6B"` | Local path or HF model ID.  Only used when `retrieval_mode=embedding`. |
| `top_k` | int | `6` | Number of general skills injected per episode. |
| `task_specific_top_k` | int | `None` | Max task-specific skills per episode.  `None` = all (template) / same as `top_k` (embedding). |
| `enable_dynamic_update` | bool | `False` | Evolve the skill bank during training using validation failures. |
| `update_threshold` | float | `0.4` | Min success rate below which skills are updated. |
| `max_new_skills` | int | `3` | Maximum new skills added per update cycle. |
---

## 📋 Skill Bank Format

Skills are stored in a JSON file with three top-level keys:

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Systematic Exploration",
      "principle": "Search every plausible surface exactly once …",
      "when_to_apply": "Anytime the goal object count is not yet met …"
    }
  ],
  "task_specific_skills": {
    "pick_and_place": [
      {
        "skill_id": "pnp_001",
        "title": "Direct Path Planning",
        "principle": "Navigate directly to the target receptacle …",
        "when_to_apply": "After picking up the object …"
      }
    ],
    "clean": [ … ],
    "heat":  [ … ]
  },
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "Repeating the same action after it fails.",
      "why_it_happens": "Agent does not track action history.",
      "how_to_avoid": "Check the admissible actions list and try an alternative."
    }
  ]
}
```

### Generating a New Skill Bank

Use the provided generation scripts (requires Azure API access):

```bash
# ALFWorld
python skill_generation/alfworld.py \
    --memory_path memory_data/alfworld/generated_memories_alfworld_total.json \
    --output_path memory_data/alfworld/claude_style_skills.json

# WebShop
python skill_generation/webshop.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json

# Search
python skill_generation/search.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json
```

---

## 📚 Citation
If you find our work helpful, please consider citing:

```bibtex
@article{xia2026skillrl,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Chen, Haifeng and others},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```

## 🙏 Acknowledgement
We would like to express our gratitude to the open-source community and the following projects for making this work possible: 
[verl-agent](https://github.com/langfengQ/verl-agent), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [Qwen](https://github.com/QwenLM/Qwen), etc.
