# ALFWorld 超算部署说明

## 1. 目标范围

本文档对应当前仓库中的 ALFWorld 训练与评测闭环，覆盖以下流程：

- `examples/grpo_trainer/run_alfworld_fixed_single_stage.sh`
- `examples/grpo_trainer/run_alfworld_smoke.sh`
- `scripts/eval_alfworld_api.py`

本文档不覆盖：

- WebShop
- AppWorld
- Search 检索服务
- 多模态环境

## 2. 依赖约束

当前本地可运行环境的核心版本如下：

```text
Python 3.10
torch 2.8.0
transformers 4.57.3
vllm 0.11.0
ray 2.50.0
wandb 0.26.1
peft 0.17.0
hydra-core 1.3.2
omegaconf 2.3.0
alfworld 0.4.2
textworld 1.7.0
sentence-transformers 5.4.1
accelerate 1.13.0
flash-attn 2.7.4.post1
```

仓库中的部署文件：

- `requirements-skillRL.txt`
- `environment-skillRL.yml`

这两份文件对应的是：

- CUDA 12.8
- 单环境部署
- A800 80GB 单卡默认配置
- ALFWorld text-only 训练与评测

如果超算节点不是 CUDA 12.8，不要直接照搬这两份文件，需要同步调整：

- `torch`
- `flash-attn`
- `vllm`

## 3. 系统前提

超算节点需要满足以下条件：

- Linux x86_64
- NVIDIA GPU
- 驱动可兼容 CUDA 12.8
- 可用的 `gcc/g++` 11 或以上
- 可用的 Java 11
- 可访问 Hugging Face / PyPI / W&B，或已经准备好离线 wheel 与模型文件

建议在作业脚本中显式加载：

```bash
module load cuda/12.8
module load gcc/11
module load java/11
```

如果超算没有 `module` 系统，则确保对应二进制已经在 `PATH` 中。

## 4. 环境创建

### 4.1 使用 conda environment 文件

在仓库根目录执行：

```bash
conda env create -f environment-skillRL.yml
conda activate skillRL
pip install -e .
```

### 4.2 使用 requirements 文件

如果超算上只允许已有 conda 环境中安装 pip 包：

```bash
conda create -n skillRL python=3.10 -y
conda activate skillRL
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 binutils_linux-64=2.40 openjdk=11 faiss-cpu -y
pip install -r requirements-skillRL.txt
pip install -e .
```

## 5. ALFWorld 数据准备

### 5.1 下载官方 ALFWorld 资源

首次部署建议把 ALFWorld 数据放在项目目录下，避免依赖登录节点或容器的 `$HOME` cache：

```bash
cd /GLOBALFS/hit_wxia_1/myl/SkillRL
mkdir -p .cache/alfworld
ALFWORLD_DATA=$PWD/.cache/alfworld alfworld-download -f
```

下载完成后需要确认以下目录存在：

```text
/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld/json_2.1.1/
/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld/logic/
/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld/detectors/
```

### 5.2 配置 ALFWorld 数据路径

本仓库的配置文件 `agent_system/environments/env_package/alfworld/configs/config_tw.yaml` 使用环境变量：

```text
$ALFWORLD_DATA
```

超算默认配置为：

```bash
export ALFWORLD_DATA=/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld
```

也可以不手动设置。当前训练脚本会默认使用：

```text
$PROJECT_ROOT/.cache/alfworld
```

### 5.3 本仓库自带的 cache 准备脚本

当前主训练脚本会执行：

```bash
source scripts/setup_alfworld_cache.sh
```

该脚本现在只做项目内 cache 准备：

```text
PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT:-$REPO_ROOT/.cache}
ALFWORLD_DATA=${ALFWORLD_DATA:-$PROJECT_CACHE_ROOT/alfworld}
```

如果 `json_2.1.1` 或 `logic` 缺失，脚本会提示执行 `ALFWORLD_DATA=$PROJECT_ROOT/.cache/alfworld alfworld-download -f`。

## 6. 模型与输出目录

当前主训练脚本需要以下路径变量：

```bash
export MODEL_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507
export ALFWORLD_DATA=/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld
export DATA_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_data/verl-agent
export OUTPUT_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_outputs
```

当前项目根目录建议使用：

```text
/GLOBALFS/hit_wxia_1/myl/SkillRL/
```

## 7. 训练前自检

建议在超算上先做以下检查。

### 7.1 Python 包检查

```bash
python scripts/diagnose.py
```

### 7.2 ALFWorld 导入检查

```bash
python - <<'PY'
import alfworld
import textworld
print("alfworld ok")
print("textworld ok")
PY
```

### 7.3 vLLM / PyTorch / CUDA 检查

```bash
python - <<'PY'
import torch, vllm
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.is_available())
print("vllm", vllm.__version__)
PY
```

## 8. 推荐部署顺序

### 8.1 先跑 API 评测

不训练，先确认 ALFWorld 环境能否正常交互：

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=token-abc123 \
MODEL_NAME=your_model_name \
python3 scripts/eval_alfworld_api.py --episodes 1 --num-envs 1
```

### 8.2 再跑冒烟训练

```bash
EXPERIMENT_NAME=alfworld_smoke_hpc \
TOTAL_TRAINING_STEPS=1 \
TRAIN_DATA_SIZE=1 \
GROUP_SIZE=1 \
bash examples/grpo_trainer/run_alfworld_smoke.sh
```

### 8.3 最后跑正式训练

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507 \
ALFWORLD_DATA=/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld \
DATA_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_data/verl-agent \
OUTPUT_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_outputs \
EXPERIMENT_NAME=alfworld_qwen3_4b_thinking_hpc \
bash examples/grpo_trainer/run_alfworld_fixed_single_stage.sh
```

## 9. 超算常见问题

### 9.1 `flash-attn` 编译失败

原因通常是：

- CUDA 版本不匹配
- `gcc` 版本不匹配
- 节点上没有可用 NVCC

处理方式：

- 先确认 `nvcc --version`
- 先确认 `gcc --version`
- 必要时换成与节点 CUDA 对应的 `torch` / `flash-attn` / `vllm`

### 9.2 `ValueError: No available memory for the cache blocks`

这是 vLLM KV cache 显存不足。A800 80GB 当前默认值为：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.70
VLLM_MAX_NUM_BATCHED_TOKENS=8192
VLLM_MAX_NUM_SEQS=8
TRAIN_DATA_SIZE=16
GROUP_SIZE=4
```

如果仍然 OOM，再按下面的保守参数回退：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.30
VLLM_MAX_NUM_BATCHED_TOKENS=1024
VLLM_MAX_NUM_SEQS=1
GROUP_SIZE=1
TRAIN_DATA_SIZE=1
```

### 9.3 W&B 超时

如果超算到外网不稳定，可加：

```bash
export WANDB_INIT_TIMEOUT=120
export WANDB_GRAPHQL_TIMEOUT=120
```

如果完全不能联网，可临时关闭：

```bash
export TRAINER_LOGGER="['console']"
```

## 10. 建议保留的最小作业脚本骨架

```bash
#!/bin/bash
set -euo pipefail

module load cuda/12.8
module load gcc/11
module load java/11

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skillRL

cd /GLOBALFS/hit_wxia_1/myl/SkillRL

export ALFWORLD_DATA=/GLOBALFS/hit_wxia_1/myl/SkillRL/.cache/alfworld
export MODEL_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507
export DATA_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_data/verl-agent
export OUTPUT_ROOT=/GLOBALFS/hit_wxia_1/myl/SkillRL/skillrl_outputs
export CUDA_VISIBLE_DEVICES=0
export WANDB_INIT_TIMEOUT=120
export WANDB_GRAPHQL_TIMEOUT=120

bash examples/grpo_trainer/run_alfworld_smoke.sh
```

## 11. 当前建议

对超算首轮部署，建议按以下顺序推进：

1. 创建环境并 `pip install -e .`
2. 手动下载并配置 `ALFWORLD_DATA`
3. 先跑 `scripts/eval_alfworld_api.py`
4. 再跑 `run_alfworld_smoke.sh`
5. 最后跑 `run_alfworld_fixed_single_stage.sh`

这样能最快定位问题是在：

- Python 环境
- CUDA / vLLM / flash-attn
- ALFWorld 数据路径
- 训练脚本配置
