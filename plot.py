#!/usr/bin/env python3
"""
训练指标可视化脚本
自动从 output.log 或 metrics.jsonl 提取训练指标并生成折线图
"""
import re
import json
import os
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体
def setup_chinese_font():
    """配置中文字体支持"""
    # 尝试常见的中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'WenQuanYi Zen Hei',    # 文泉驿正黑
        'Noto Sans CJK SC',     # 思源黑体
        'SimHei',               # 黑体
        'Microsoft YaHei',      # 微软雅黑
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            return

    # 如果没有中文字体，使用英文标签
    print("警告: 未找到中文字体，将使用英文标签")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return False

def parse_from_log(log_path):
    """从 output.log 解析训练指标"""
    step_pattern = re.compile(r"step:(\d+)")
    reward_pattern = re.compile(r"episode/reward/mean:([-\d.]+)")
    success_pattern = re.compile(r"episode/success_rate:([-\d.]+)")
    loss_pattern = re.compile(r"actor/pg_loss:([-\d.eE]+)")
    kl_pattern = re.compile(r"actor/ppo_kl:([-\d.eE]+)")

    steps, rewards, success_rates, losses, kls = [], [], [], [], []

    with open(log_path, 'r') as f:
        for line in f:
            step_m = step_pattern.search(line)
            if not step_m:
                continue

            reward_m = reward_pattern.search(line)
            success_m = success_pattern.search(line)
            loss_m = loss_pattern.search(line)
            kl_m = kl_pattern.search(line)

            if step_m and reward_m and success_m and loss_m:
                steps.append(int(step_m.group(1)))
                rewards.append(float(reward_m.group(1)))
                success_rates.append(float(success_m.group(1)))
                losses.append(float(loss_m.group(1)))
                kls.append(float(kl_m.group(1)) if kl_m else 0.0)

    return steps, rewards, success_rates, losses, kls

def parse_from_jsonl(jsonl_path):
    """从 metrics.jsonl 解析训练指标"""
    steps, rewards, success_rates, losses, kls = [], [], [], [], []

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                metrics = data.get('metrics', {})

                step = data.get('step')
                if step is None:
                    continue

                reward = metrics.get('episode/reward/mean')
                success = metrics.get('episode/success_rate')
                loss = metrics.get('actor/pg_loss')
                kl = metrics.get('actor/ppo_kl', 0.0)

                if reward is not None and success is not None and loss is not None:
                    steps.append(step)
                    rewards.append(reward)
                    success_rates.append(success)
                    losses.append(loss)
                    kls.append(kl)
            except json.JSONDecodeError:
                continue

    return steps, rewards, success_rates, losses, kls

def plot_metrics(steps, rewards, success_rates, losses, kls, output_dir, use_chinese=True):
    """绘制训练指标折线图"""
    if not steps:
        print("错误: 没有找到有效的训练数据")
        return

    # 创建 2x2 子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 标签（中英文）
    if use_chinese:
        labels = {
            'step': '训练步数',
            'reward': '平均奖励',
            'success': '成功率',
            'loss': '策略梯度损失',
            'kl': 'KL散度',
            'title': '训练过程指标监控'
        }
    else:
        labels = {
            'step': 'Training Step',
            'reward': 'Mean Reward',
            'success': 'Success Rate',
            'loss': 'Policy Gradient Loss',
            'kl': 'KL Divergence',
            'title': 'Training Metrics'
        }

    # 子图1: 奖励曲线
    ax1.plot(steps, rewards, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel(labels['step'], fontsize=12)
    ax1.set_ylabel(labels['reward'], fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{labels['reward']} (Max: {max(rewards):.2f})", fontsize=12)

    # 子图2: 成功率曲线
    ax2.plot(steps, success_rates, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel(labels['step'], fontsize=12)
    ax2.set_ylabel(labels['success'], fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{labels['success']} (Max: {max(success_rates):.2%})", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # 子图3: 损失曲线
    ax3.plot(steps, losses, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel(labels['step'], fontsize=12)
    ax3.set_ylabel(labels['loss'], fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"{labels['loss']}", fontsize=12)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 子图4: KL散度曲线
    ax4.plot(steps, kls, 'm-', linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel(labels['step'], fontsize=12)
    ax4.set_ylabel(labels['kl'], fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f"{labels['kl']}", fontsize=12)

    plt.suptitle(labels['title'], fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练指标图已保存到: {output_path}")

    # 保存统计摘要
    summary_path = os.path.join(output_dir, 'training_metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"训练统计摘要\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"总训练步数: {max(steps)}\n")
        f.write(f"平均奖励: 最小={min(rewards):.3f}, 最大={max(rewards):.3f}, 最终={rewards[-1]:.3f}\n")
        f.write(f"成功率: 最小={min(success_rates):.2%}, 最大={max(success_rates):.2%}, 最终={success_rates[-1]:.2%}\n")
        f.write(f"策略损失: 最小={min(losses):.6f}, 最大={max(losses):.6f}, 最终={losses[-1]:.6f}\n")
        f.write(f"KL散度: 最小={min(kls):.6f}, 最大={max(kls):.6f}, 最终={kls[-1]:.6f}\n")
    print(f"训练摘要已保存到: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='训练指标可视化工具')
    parser.add_argument('--exp-dir', type=str,
                       default='/data2/myl/skillrl_outputs/skillrl_mvp/alfworld_text_llama32_3b_global_internalize_lora_v2',
                       help='实验输出目录')
    parser.add_argument('--use-jsonl', action='store_true',
                       help='优先使用 metrics.jsonl 而不是 output.log')

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"错误: 实验目录不存在: {exp_dir}")
        sys.exit(1)

    # 配置字体
    use_chinese = setup_chinese_font()

    # 解析数据
    jsonl_path = exp_dir / 'metrics.jsonl'
    log_path = exp_dir / 'output.log'

    if args.use_jsonl and jsonl_path.exists():
        print(f"从 metrics.jsonl 读取数据: {jsonl_path}")
        steps, rewards, success_rates, losses, kls = parse_from_jsonl(jsonl_path)
    elif log_path.exists():
        print(f"从 output.log 读取数据: {log_path}")
        steps, rewards, success_rates, losses, kls = parse_from_log(log_path)
    else:
        print(f"错误: 未找到 output.log 或 metrics.jsonl")
        sys.exit(1)

    # 绘制图表
    plot_metrics(steps, rewards, success_rates, losses, kls, str(exp_dir), use_chinese)

if __name__ == '__main__':
    main()