#!/usr/bin/env python3
"""Create CSV and PNG summaries from a WebShop training metrics JSONL file."""

import argparse
import json
from pathlib import Path


SERIES = [
    ("episode/success_rate", "success_rate"),
    ("episode/valid_action_ratio", "valid_action_ratio"),
    ("episode/reward/mean", "episode_reward_mean"),
    ("critic/score/mean", "critic_score_mean"),
    ("actor/entropy_loss", "actor_entropy_loss"),
    ("actor/kl_loss", "actor_kl_loss"),
    ("response_length/mean", "response_length_mean"),
    ("perf/max_memory_allocated_gb", "max_memory_allocated_gb"),
]

VAL_SERIES = [
    ("val/success_rate", "val_success_rate"),
    ("val/external_global/success_rate", "val_external_global_success_rate"),
    ("val/internalized_global_off/success_rate", "val_internalized_global_off_success_rate"),
    ("val/text/test_score", "val_test_score"),
]


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get("metrics", {})
            step = int(record.get("step", metrics.get("training/global_step", len(rows) + 1)))
            rows.append((step, metrics))
    return rows


def write_csv(rows, output_csv: Path):
    keys = [key for key, _ in SERIES + VAL_SERIES]
    with output_csv.open("w", encoding="utf-8") as file:
        file.write("step," + ",".join(keys) + "\n")
        for step, metrics in rows:
            file.write(str(step) + "," + ",".join(str(metrics.get(key, "")) for key in keys) + "\n")


def plot(rows, output_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for ax, (key, label) in zip(axes.flatten(), SERIES):
        points = [(step, metrics[key]) for step, metrics in rows if key in metrics]
        if points:
            ax.plot(*zip(*points), marker="o", linewidth=1.6, markersize=3)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.suptitle("WebShop Skill Internalization Training Metrics", fontsize=16)
    fig.savefig(output_png, dpi=180)

    validation_png = output_png.with_name(output_png.stem + "_validation.png")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for key, label in VAL_SERIES:
        points = [(step, metrics[key]) for step, metrics in rows if key in metrics]
        if points:
            ax.plot(*zip(*points), marker="o", linewidth=1.8, label=label)
    ax.set_title("WebShop Validation Metrics")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.savefig(validation_png, dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.metrics_jsonl)
    out_dir = args.out_dir or args.metrics_jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "training_metrics_summary.csv"
    output_png = out_dir / "training_metrics.png"
    write_csv(rows, output_csv)
    plot(rows, output_png)
    print(f"rows={len(rows)}")
    print(f"csv={output_csv}")
    print(f"plot={output_png}")
    print(f"validation_plot={output_png.with_name(output_png.stem + '_validation.png')}")


if __name__ == "__main__":
    main()
