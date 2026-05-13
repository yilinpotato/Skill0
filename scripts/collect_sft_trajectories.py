#!/usr/bin/env python3
"""Collect successful ALFWorld trajectories for SFT warmup.

Runs the local model on ALFWorld with skill memory, collects successful
episodes, and saves them as multi-turn parquet for fsdp_sft_trainer.

Usage:
    python3 scripts/collect_sft_trajectories.py \
        --model-path /GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507 \
        --skills-json memory_data/alfworld/claude_style_skills.json \
        --output-dir skillrl_outputs/expert_trajectories \
        --num-envs 16 \
        --min-trajectories 100
"""

import os, sys, re, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ray

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            str(Path(os.environ.get("CACHE_ROOT", "/GLOBALFS/hit_wxia_1/.cache")) / "modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507"),
        ),
    )
    p.add_argument("--skills-json", default="memory_data/alfworld/claude_style_skills.json")
    default_output_dir = Path(os.environ.get("OUTPUT_ROOT", Path(__file__).resolve().parent.parent / "skillrl_outputs")) / "expert_trajectories"
    p.add_argument("--output-dir", default=str(default_output_dir))
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--top-k-skills", type=int, default=12)
    p.add_argument("--min-trajectories", type=int, default=100)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--tensor-parallel-size", type=int, default=0)  # 0 = auto-detect
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--history-window", type=int, default=6)  # last N assistant turns sent to vLLM
    p.add_argument("--val-split", type=float, default=0.1)
    return p.parse_args()


def extract_action(text: str) -> str:
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    last_line = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return last_line[-1] if last_line else "look"


def main():
    args = parse_args()
    os.chdir(Path(__file__).parent.parent)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from agent_system.environments.env_package.alfworld import (
        build_alfworld_envs, alfworld_projection,
    )
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())

    alf_config_path = str(
        Path(__file__).parent.parent
        / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )

    env_config = OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "max_steps": args.max_steps,
            "history_length": 5,
            "alfworld": {
                "eval_dataset": "eval_in_distribution",
                "use_dense_reward": False,
            },
            "rollout": {"n": 1},
            "resources_per_worker": {"num_cpus": 0.1},
            "use_skills_only_memory": True,
            "skills_only_memory": {
                "skills_json_path": args.skills_json,
                "top_k": args.top_k_skills,
                "retrieval_mode": "template",
                "task_specific_top_k": None,
                "global_skill_top_k": None,
            },
        },
        "data": {
            "train_batch_size": args.num_envs,
            "val_batch_size": args.num_envs,
        },
    })

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    import torch
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else torch.cuda.device_count()
    tp_size = max(1, tp_size)
    print(f"Loading model with vLLM (tensor_parallel_size={tp_size}, max_model_len={args.max_model_len}) ...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    all_trajectories = []

    for round_idx in range(args.max_rounds):
        if len(all_trajectories) >= args.min_trajectories:
            break

        print(f"\n=== Round {round_idx + 1}/{args.max_rounds} | "
              f"collected {len(all_trajectories)}/{args.min_trajectories} ===")

        envs = build_alfworld_envs(
            alf_config_path,
            seed=args.seed + round_idx * 1000,
            env_num=args.num_envs,
            group_n=1,
            is_train=False,
            env_kwargs={"eval_dataset": "eval_in_distribution", "use_dense_reward": False},
            resources_per_worker={"num_cpus": 0.1, "num_gpus": 0.0},
        )
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, env_config)

        obs_dict, _ = env_manager.reset({})
        env_dones = [False] * args.num_envs
        # messages per env: list of {"role": ..., "content": ...}
        histories = [[] for _ in range(args.num_envs)]

        for step_idx in range(args.max_steps):
            if all(env_dones):
                break

            active = [i for i in range(args.num_envs) if not env_dones[i]]
            prompts = []
            for i in active:
                obs_text = obs_dict["text"][i]
                # Keep only the last `history_window` user/assistant pairs to avoid exceeding max_model_len.
                # histories[i] alternates user/assistant, so 2*window tail entries = window rounds.
                window = args.history_window * 2
                windowed_history = histories[i][-window:] if len(histories[i]) > window else histories[i]
                turn_messages = windowed_history + [{"role": "user", "content": obs_text}]
                prompt = tokenizer.apply_chat_template(
                    turn_messages, add_generation_prompt=True, tokenize=False
                )
                prompts.append(prompt)

            outputs = llm.generate(prompts, sampling_params)
            raw_actions = [extract_action(o.outputs[0].text) for o in outputs]

            full_actions = ["look"] * args.num_envs
            for idx, i in enumerate(active):
                full_actions[i] = raw_actions[idx]

            # Record turns before stepping
            for idx, i in enumerate(active):
                obs_text = obs_dict["text"][i]
                action = raw_actions[idx]
                histories[i].append({"role": "user", "content": obs_text})
                histories[i].append({"role": "assistant", "content": f"<think>Executing action.</think>\n<action>{action}</action>"})

            next_obs, rewards, dones, _ = env_manager.step(full_actions)

            for i in range(args.num_envs):
                if not env_dones[i] and dones[i]:
                    env_dones[i] = True
                    won = float(rewards[i]) >= 5.0  # sparse: 10.0 on win
                    if won and len(histories[i]) >= 2:
                        # Save only the windowed portion to keep SFT sequences bounded
                        window = args.history_window * 2
                        saved = histories[i][-window:] if len(histories[i]) > window else histories[i]
                        all_trajectories.append({"messages": saved})
                        print(f"  env {i}: SUCCESS  (step {step_idx + 1}, "
                              f"total={len(all_trajectories)})")

            obs_dict = next_obs

        success_rate = sum(1 for t in all_trajectories if t) / (args.num_envs * (round_idx + 1))
        print(f"  Round done. Cumulative success rate: {success_rate:.2%}")

    print(f"\nCollected {len(all_trajectories)} successful trajectories.")
    if not all_trajectories:
        print("WARNING: No trajectories collected. The base model may need more attempts or a stronger temperature.")
        sys.exit(1)

    # Train/val split
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_trajectories))
    n_val = max(1, int(len(all_trajectories) * args.val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = pd.DataFrame([all_trajectories[i] for i in train_idx])
    val_df = pd.DataFrame([all_trajectories[i] for i in val_idx])

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved {len(train_df)} train + {len(val_df)} val trajectories.")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
