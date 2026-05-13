#!/usr/bin/env python3
"""Direct API evaluation for ALFWorld.

Uses an OpenAI-compatible endpoint to test a model on ALFWorld without training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import ray
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "token-abc123"))
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507")))
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-dataset", default="eval_in_distribution")
    p.add_argument("--use-dense-reward", action="store_true")
    p.add_argument("--output-dir", default="/data2/myl/skillrl_outputs/alfworld_api_eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import (
        alfworld_projection,
        build_alfworld_envs,
    )

    ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
    alf_config_path = str(
        Path(__file__).resolve().parent.parent
        / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )

    env_config = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "max_steps": args.max_steps,
                "history_length": 2,
                "alfworld": {
                    "eval_dataset": args.eval_dataset,
                    "use_dense_reward": args.use_dense_reward,
                },
                "rollout": {"n": 1},
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0.0},
            },
            "data": {
                "train_batch_size": args.num_envs,
                "val_batch_size": args.num_envs,
            },
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"alfworld_api_eval_{run_tag}.json")

    results = []
    all_success = []

    for episode in range(args.episodes):
        envs = build_alfworld_envs(
            alf_config_path,
            seed=args.seed + episode * 1000,
            env_num=args.num_envs,
            group_n=1,
            is_train=False,
            env_kwargs={
                "eval_dataset": args.eval_dataset,
                "use_dense_reward": args.use_dense_reward,
            },
            resources_per_worker={"num_cpus": 0.1, "num_gpus": 0.0},
        )
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, env_config)

        obs, _ = env_manager.reset({})
        dones = [False] * args.num_envs
        episode_records = [
            {"episode": episode, "env_index": i, "steps": []}
            for i in range(args.num_envs)
        ]

        for step in range(args.max_steps):
            if all(dones):
                break

            actions = []
            for i in range(args.num_envs):
                if dones[i]:
                    actions.append("look")
                    continue
                prompt = obs["text"][i]
                resp = requests.post(
                    f"{args.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {args.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "n": 1,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                payload = resp.json()
                text = (payload["choices"][0]["message"]["content"] or "").strip()
                actions.append(text)
                episode_records[i]["steps"].append(
                    {
                        "step": step + 1,
                        "prompt": prompt,
                        "raw_response": text,
                    }
                )

            obs, rewards, step_dones, infos = env_manager.step(actions)

            for i in range(args.num_envs):
                if dones[i]:
                    continue
                episode_records[i]["steps"][-1].update(
                    {
                        "env_action": infos[i].get("env_action"),
                        "is_action_valid": infos[i].get("is_action_valid"),
                        "reward": float(rewards[i]),
                        "done": bool(step_dones[i]),
                        "won": bool(infos[i].get("won", False)),
                        "next_observation": obs["text"][i],
                    }
                )
                if step_dones[i]:
                    dones[i] = True
                    all_success.append(bool(infos[i].get("won", False)))

        for i in range(args.num_envs):
            episode_records[i]["success"] = bool(any(
                s.get("won", False) for s in episode_records[i]["steps"]
            ))
        results.extend(episode_records)

        print(
            f"episode={episode} success_rate="
            f"{np.mean([r['success'] for r in episode_records]):.3f}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "success_rate": float(np.mean(all_success)) if all_success else 0.0,
                "episodes": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(f"saved: {out_path}")
    print(f"overall_success_rate={float(np.mean(all_success)) if all_success else 0.0:.3f}")


if __name__ == "__main__":
    main()
