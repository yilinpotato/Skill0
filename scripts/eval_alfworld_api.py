#!/usr/bin/env python3
"""Direct API evaluation for ALFWorld.

Uses an OpenAI-compatible endpoint to test a model on ALFWorld without training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import ray
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _env_default(name: str, value: str) -> str:
    return os.environ.get(name, value)


def _check_alfworld_data(root: Path) -> None:
    required = [root / "json_2.1.1", root / "logic", root / "detectors"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "ALFWorld cache is incomplete. Missing: "
            + ", ".join(missing)
            + f". Set ALFWORLD_DATA to a complete cache, e.g. {root}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=_env_default("OPENAI_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--model", default=_env_default("MODEL_NAME", "deepseek-v4-pro"))
    p.add_argument("--num-envs", type=int, default=int(os.environ.get("API_EVAL_NUM_ENVS", "4")))
    p.add_argument("--episodes", type=int, default=int(os.environ.get("API_EVAL_EPISODES", "100")))
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=float(os.environ.get("API_EVAL_TEMPERATURE", "0.0")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("API_EVAL_MAX_TOKENS", "512")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-dataset", default="eval_in_distribution")
    p.add_argument("--use-dense-reward", action="store_true")
    output_root = _env_default("OUTPUT_ROOT", str(Path(__file__).resolve().parent.parent / "skillrl_outputs"))
    p.add_argument("--output-dir", default=os.path.join(output_root, "alfworld_api_eval"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    p.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    p.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR"))
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def _post_chat_completion(args, prompt):
    started = time.perf_counter()
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
    latency = time.perf_counter() - started
    resp.raise_for_status()
    payload = resp.json()
    choice = payload["choices"][0]["message"]
    usage = payload.get("usage", {}) or {}
    return {
        "content": (choice.get("content") or "").strip(),
        "reasoning_content": (choice.get("reasoning_content") or "").strip(),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "latency_s": latency,
    }


def _episode_summary(records):
    step_rows = [step for episode in records for step in episode["steps"]]
    total_prompt_tokens = sum(step.get("prompt_tokens", 0) for step in step_rows)
    total_completion_tokens = sum(step.get("completion_tokens", 0) for step in step_rows)
    total_tokens = sum(step.get("total_tokens", 0) for step in step_rows)
    latencies = [step.get("latency_s", 0.0) for step in step_rows if step.get("latency_s") is not None]
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([r["success"] for r in records])) if records else 0.0,
        "avg_episode_steps": float(np.mean([r["num_steps"] for r in records])) if records else 0.0,
        "avg_episode_prompt_tokens": float(np.mean([r["prompt_tokens_total"] for r in records])) if records else 0.0,
        "avg_episode_completion_tokens": float(np.mean([r["completion_tokens_total"] for r in records])) if records else 0.0,
        "avg_episode_total_tokens": float(np.mean([r["total_tokens"] for r in records])) if records else 0.0,
        "avg_step_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_completion_tokens": int(total_completion_tokens),
        "total_tokens": int(total_tokens),
    }


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    cache_root = Path(os.environ.get("CACHE_ROOT", "/GLOBALFS/hit_wxia_1/.cache"))
    alfworld_data = Path(os.environ.get("ALFWORLD_DATA", cache_root / "alfworld"))
    _check_alfworld_data(alfworld_data)

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
    detail_path = os.path.join(args.output_dir, f"alfworld_api_eval_{run_tag}.json")
    jsonl_path = os.path.join(args.output_dir, f"alfworld_api_eval_{run_tag}.jsonl")
    summary_path = os.path.join(args.output_dir, f"alfworld_api_eval_{run_tag}.summary.json")
    state_path = os.path.join(args.output_dir, "alfworld_api_eval.state.json")

    completed_episodes = 0
    all_records = []
    step_counter = 0
    if args.resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        completed_episodes = int(state.get("completed_episodes", 0))
        all_records = list(state.get("episodes", []))
        step_counter = int(state.get("step_counter", 0))

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                group=args.wandb_group,
                dir=args.wandb_dir,
                config=vars(args),
            )
        except Exception as exc:
            print(f"[wandb] disabled: {exc}")
            wandb_run = None

    def log_step(step_payload):
        if wandb_run is not None:
            wandb_run.log(step_payload)

    wall_started = time.perf_counter()

    remaining = max(0, args.episodes - completed_episodes)
    episode_offset = completed_episodes
    while remaining > 0:
        batch_size = min(args.num_envs, remaining)
        envs = build_alfworld_envs(
            alf_config_path,
            seed=args.seed + episode_offset * 1000,
            env_num=batch_size,
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
        dones = [False] * batch_size
        episode_records = [
            {"episode": episode_offset + i, "env_index": i, "steps": []}
            for i in range(batch_size)
        ]

        for step in range(args.max_steps):
            if all(dones):
                break

            active_indices = [i for i, done in enumerate(dones) if not done]
            prompts = [obs["text"][i] for i in active_indices]

            api_results = {}
            with ThreadPoolExecutor(max_workers=max(1, len(active_indices))) as pool:
                futures = {
                    pool.submit(_post_chat_completion, args, prompt): idx
                    for idx, prompt in zip(active_indices, prompts)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    api_results[idx] = future.result()

            actions = ["look"] * batch_size
            for i in active_indices:
                api = api_results[i]
                actions[i] = api["content"] or "look"
                episode_records[i]["steps"].append(
                    {
                        "step": step + 1,
                        "prompt": obs["text"][i],
                        "reasoning_content": api["reasoning_content"],
                        "raw_response": api["content"],
                        "model_output": api["content"],
                        "prompt_tokens": api["prompt_tokens"],
                        "completion_tokens": api["completion_tokens"],
                        "total_tokens": api["total_tokens"],
                        "latency_s": api["latency_s"],
                    }
                )
                step_payload = {
                    "eval/step_prompt_tokens": api["prompt_tokens"],
                    "eval/step_completion_tokens": api["completion_tokens"],
                    "eval/step_total_tokens": api["total_tokens"],
                    "eval/step_latency_s": api["latency_s"],
                }
                step_counter += 1
                log_step(step_payload | {"eval/global_step": step_counter})

            obs, rewards, step_dones, infos = env_manager.step(actions)

            for i in active_indices:
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

        for i in range(batch_size):
            steps = episode_records[i]["steps"]
            episode_records[i]["success"] = bool(any(s.get("won", False) for s in steps))
            episode_records[i]["num_steps"] = len(steps)
            episode_records[i]["prompt_tokens_total"] = int(sum(s.get("prompt_tokens", 0) for s in steps))
            episode_records[i]["completion_tokens_total"] = int(sum(s.get("completion_tokens", 0) for s in steps))
            episode_records[i]["total_tokens"] = int(sum(s.get("total_tokens", 0) for s in steps))
            episode_records[i]["avg_step_latency_s"] = float(np.mean([s.get("latency_s", 0.0) for s in steps])) if steps else 0.0
            all_records.append(episode_records[i])

        batch_summary = _episode_summary(episode_records)
        print(
            f"episodes={len(all_records)}/{args.episodes} "
            f"batch_success_rate={batch_summary['success_rate']:.3f} "
            f"avg_total_tokens={batch_summary['avg_episode_total_tokens']:.1f} "
            f"avg_latency_s={batch_summary['avg_step_latency_s']:.2f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval/batch_success_rate": batch_summary["success_rate"],
                    "eval/batch_avg_episode_total_tokens": batch_summary["avg_episode_total_tokens"],
                    "eval/batch_avg_episode_prompt_tokens": batch_summary["avg_episode_prompt_tokens"],
                    "eval/batch_avg_episode_completion_tokens": batch_summary["avg_episode_completion_tokens"],
                    "eval/batch_avg_step_latency_s": batch_summary["avg_step_latency_s"],
                    "eval/episodes_seen": len(all_records),
                }
            )

        remaining -= batch_size
        episode_offset += batch_size

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "completed_episodes": len(all_records),
                    "step_counter": step_counter,
                    "episodes": all_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    summary = _episode_summary(all_records)
    summary["wall_time_s"] = float(time.perf_counter() - wall_started)
    summary["episodes_requested"] = int(args.episodes)
    summary["model"] = args.model
    summary["base_url"] = args.base_url

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "summary": summary,
                "episodes": all_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if wandb_run is not None:
        import wandb

        table = wandb.Table(
            columns=[
                "episode",
                "success",
                "num_steps",
                "prompt_tokens_total",
                "completion_tokens_total",
                "total_tokens",
                "avg_step_latency_s",
            ]
        )
        for record in all_records:
            table.add_data(
                record["episode"],
                record["success"],
                record["num_steps"],
                record["prompt_tokens_total"],
                record["completion_tokens_total"],
                record["total_tokens"],
                record["avg_step_latency_s"],
            )
        wandb_run.log(
            {
                "eval/final_success_rate": summary["success_rate"],
                "eval/final_total_tokens": summary["total_tokens"],
                "eval/final_prompt_tokens": summary["total_prompt_tokens"],
                "eval/final_completion_tokens": summary["total_completion_tokens"],
                "eval/final_avg_step_latency_s": summary["avg_step_latency_s"],
                "eval/episodes_table": table,
            }
        )
        wandb_run.finish()

    print(f"saved: {detail_path}")
    print(f"saved: {jsonl_path}")
    print(f"saved: {summary_path}")
    print(f"overall_success_rate={summary['success_rate']:.3f}")
    print(f"total_tokens={summary['total_tokens']}")
    print(f"wall_time_s={summary['wall_time_s']:.1f}")


if __name__ == "__main__":
    main()
