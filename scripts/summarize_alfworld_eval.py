#!/usr/bin/env python3
import ast
import re
import sys
from pathlib import Path


KEYS = [
    "val/success_rate",
    "val/external_global/success_rate",
    "val/internalized_global_off/success_rate",
    "val/text/test_score",
    "val/external_global/text/test_score",
    "val/internalized_global_off/text/test_score",
    "val/global_skill_top_k",
    "val/external_global/global_skill_top_k",
    "val/internalized_global_off/global_skill_top_k",
]


def extract_final_metrics(text: str):
    marker = "Final validation metrics:"
    idx = text.rfind(marker)
    if idx == -1:
        marker = "Initial validation metrics:"
        idx = text.rfind(marker)
    if idx == -1:
        return None

    tail = text[idx + len(marker):]
    match = re.search(r"\{.*?\}", tail, flags=re.DOTALL)
    if not match:
        return None

    raw = match.group(0)
    raw = raw.replace('"\n', '').replace("\n", " ")
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_alfworld_eval.py LOG_OR_DIR [...]", file=sys.stderr)
        sys.exit(1)

    logs = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            logs.extend(sorted(path.glob("*.log")))
        else:
            logs.append(path)

    for log in logs:
        if not log.exists():
            print(f"{log}: missing")
            continue
        metrics = extract_final_metrics(log.read_text(errors="ignore"))
        print(f"\n== {log} ==")
        if not metrics:
            print("No final/initial validation metrics found.")
            continue
        for key in KEYS:
            if key in metrics:
                print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
