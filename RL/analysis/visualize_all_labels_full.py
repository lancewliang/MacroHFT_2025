#!/usr/bin/env python3
"""
完整版：可视化多个Label的训练集和验证集价格走势
包含训练集和验证集的单文件详细图
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 配置
DATASET = "ETHUSDT"
LABELS = [1, 2, 3]
DATA_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/data"
OUTPUT_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/analysis"
MAX_TRAIN_FILES_DETAIL = 15  # 训练集最多显示前15个文件的详细图
MAX_VAL_FILES_DETAIL = 15    # 验证集最多显示前15个文件的详细图

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 100

print("="*80)
print("完整版多Label价格走势可视化 (包含训练集和验证集单文件详细图)")
print("="*80)

# 加载标签索引
with open(f"{DATA_ROOT}/{DATASET}/train/slope_labels.pkl", 'rb') as f:
    train_index = pickle.load(f)
with open(f"{DATA_ROOT}/{DATASET}/val/slope_labels.pkl", 'rb') as f:
    val_index = pickle.load(f)

def generate_single_file_plot(df, file_id, file_type, label_id, output_path):
    """生成单个文件的详细价格图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(df['close'].values, color='blue', linewidth=1.5, label='Close Price')

    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    total_return = (end_price / start_price - 1) * 100
    max_price = df['close'].max()
    min_price = df['close'].min()
    volatility = df['close'].pct_change().std() * 100

    ax.scatter([0], [start_price], color='green', s=100, zorder=5, label='Start')
    ax.scatter([len(df)-1], [end_price], color='red', s=100, zorder=5, label='End')
    ax.axhline(y=start_price, color='gray', linestyle='--', alpha=0.5)

    color = 'green' if total_return > 0 else 'red'
    ax.set_title(f'Label {label_id} - {file_type} File {file_id} | Return: {total_return:.2f}%',
                fontsize=14, fontweight='bold', color=color)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    stats_text = f'Start: ${start_price:.2f}\nEnd: ${end_price:.2f}\nReturn: {total_return:.2f}%\nMax: ${max_price:.2f}\nMin: ${min_price:.2f}\nVolatility: {volatility:.4f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def analyze_and_visualize_label(label_id):
    """完整分析单个label（包含训练集和验证集的单文件图）"""

    print(f"\n{'='*80}")
    print(f"处理 Label {label_id}")
    print(f"{'='*80}")

    label_dir = f"{OUTPUT_ROOT}/label{label_id}_analysis"
    os.makedirs(label_dir, exist_ok=True)

    train_files = train_index[label_id]
    val_files = val_index[label_id]

    print(f"\nLabel {label_id} 文件数量:")
    print(f"  训练集: {len(train_files)} 个文件")
    print(f"  验证集: {len(val_files)} 个文件")

    # ========================================================================
    # 1. 汇总对比图
    # ========================================================================
    print(f"  [1/5] 生成汇总对比图...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # 训练集
    train_stats = []
    sample_size = min(30, len(train_files))

    for i, file_id in enumerate(train_files[:sample_size]):
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/train/df_{file_id}.feather")
            normalized_price = (df['close'] / df.iloc[0]['close']) * 100
            total_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100

            color = 'green' if total_return > 0 else 'red'
            alpha = 0.3

            ax1.plot(normalized_price.values, color=color, alpha=alpha, linewidth=0.8,
                    label=f'File{file_id}' if i < 5 else None)

            train_stats.append({
                'file_id': file_id,
                'return': total_return,
                'volatility': df['close'].pct_change().std() * 100,
            })
        except Exception as e:
            pass

    ax1.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_title(f'Label {label_id} - Training Set (Normalized to 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if len([l for l in ax1.get_legend_handles_labels()[1] if l]) > 0:
        ax1.legend(loc='upper left', fontsize=8)

    train_df = pd.DataFrame(train_stats)
    up_count = (train_df['return'] > 0).sum()
    down_count = (train_df['return'] < 0).sum()

    stats_text = f'Files: {len(train_stats)}\nUp: {up_count} ({up_count/len(train_stats)*100:.0f}%)\nDown: {down_count}\nAvg: {train_df["return"].mean():.2f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 验证集
    val_stats = []

    for i, file_id in enumerate(val_files):
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/val/df_{file_id}.feather")
            normalized_price = (df['close'] / df.iloc[0]['close']) * 100
            total_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100

            color = 'green' if total_return > 0 else 'red'
            alpha = 0.5

            ax2.plot(normalized_price.values, color=color, alpha=alpha, linewidth=1.2,
                    label=f'File{file_id}')

            val_stats.append({
                'file_id': file_id,
                'return': total_return,
                'volatility': df['close'].pct_change().std() * 100,
            })
        except Exception as e:
            pass

    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title(f'Label {label_id} - Validation Set (Normalized to 100)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Normalized Price', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if len(val_files) <= 20:
        ax2.legend(loc='upper left', fontsize=8, ncol=2)

    val_df = pd.DataFrame(val_stats)
    up_count_val = (val_df['return'] > 0).sum()
    down_count_val = (val_df['return'] < 0).sum()

    stats_text_val = f'Files: {len(val_stats)}\nUp: {up_count_val} ({up_count_val/len(val_stats)*100:.0f}%)\nDown: {down_count_val}\nAvg: {val_df["return"].mean():.2f}%'
    ax2.text(0.02, 0.98, stats_text_val, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{label_dir}/summary_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. 收益率分布图
    # ========================================================================
    print(f"  [2/5] 生成收益率分布图...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    train_returns = train_df['return'].values
    ax1.hist(train_returns, bins=min(20, len(train_returns)//2),
            color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=train_returns.mean(), color='orange', linestyle='--', linewidth=2)
    ax1.set_title(f'Label {label_id} - Training Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    val_returns = val_df['return'].values
    ax2.hist(val_returns, bins=min(15, len(val_returns)//2),
            color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=val_returns.mean(), color='orange', linestyle='--', linewidth=2)
    ax2.set_title(f'Label {label_id} - Validation Returns', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{label_dir}/return_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. 训练集单文件详细图
    # ========================================================================
    print(f"  [3/5] 生成训练集单文件详细图 (前{MAX_TRAIN_FILES_DETAIL}个)...")

    train_files_dir = f"{label_dir}/training_files"
    os.makedirs(train_files_dir, exist_ok=True)

    for file_id in train_files[:MAX_TRAIN_FILES_DETAIL]:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/train/df_{file_id}.feather")
            output_path = f"{train_files_dir}/train_file_{file_id}.png"
            generate_single_file_plot(df, file_id, 'Train', label_id, output_path)
        except Exception as e:
            print(f"    警告: 训练集文件{file_id}处理失败")

    # ========================================================================
    # 4. 验证集单文件详细图
    # ========================================================================
    print(f"  [4/5] 生成验证集单文件详细图 (前{MAX_VAL_FILES_DETAIL}个)...")

    val_files_dir = f"{label_dir}/validation_files"
    os.makedirs(val_files_dir, exist_ok=True)

    for file_id in val_files[:MAX_VAL_FILES_DETAIL]:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/val/df_{file_id}.feather")
            output_path = f"{val_files_dir}/val_file_{file_id}.png"
            generate_single_file_plot(df, file_id, 'Val', label_id, output_path)
        except Exception as e:
            print(f"    警告: 验证集文件{file_id}处理失败")

    # ========================================================================
    # 5. 统计报告
    # ========================================================================
    print(f"  [5/5] 生成统计报告...")

    with open(f"{label_dir}/statistics_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Label {label_id} 完整统计报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"训练集 ({len(train_df)} 个文件):\n")
        f.write(f"  平均收益率: {train_df['return'].mean():.2f}%\n")
        f.write(f"  中位数: {train_df['return'].median():.2f}%\n")
        f.write(f"  标准差: {train_df['return'].std():.2f}%\n")
        f.write(f"  范围: [{train_df['return'].min():.2f}%, {train_df['return'].max():.2f}%]\n")
        f.write(f"  上涨: {(train_df['return'] > 0).sum()} ({(train_df['return'] > 0).sum()/len(train_df)*100:.1f}%)\n")
        f.write(f"  下跌: {(train_df['return'] < 0).sum()} ({(train_df['return'] < 0).sum()/len(train_df)*100:.1f}%)\n")
        f.write(f"  波动率: {train_df['volatility'].mean():.4f}%\n\n")

        f.write(f"验证集 ({len(val_df)} 个文件):\n")
        f.write(f"  平均收益率: {val_df['return'].mean():.2f}%\n")
        f.write(f"  中位数: {val_df['return'].median():.2f}%\n")
        f.write(f"  标准差: {val_df['return'].std():.2f}%\n")
        f.write(f"  范围: [{val_df['return'].min():.2f}%, {val_df['return'].max():.2f}%]\n")
        f.write(f"  上涨: {(val_df['return'] > 0).sum()} ({(val_df['return'] > 0).sum()/len(val_df)*100:.1f}%)\n")
        f.write(f"  下跌: {(val_df['return'] < 0).sum()} ({(val_df['return'] < 0).sum()/len(val_df)*100:.1f}%)\n")
        f.write(f"  波动率: {val_df['volatility'].mean():.4f}%\n\n")

        mean_diff = abs(train_df['return'].mean() - val_df['return'].mean())
        vol_diff = abs(train_df['volatility'].mean() - val_df['volatility'].mean())
        f.write(f"差异:\n")
        f.write(f"  收益率差: {mean_diff:.2f}%\n")
        f.write(f"  波动率差: {vol_diff:.4f}%\n\n")

        f.write("诊断:\n")
        if val_df['return'].mean() > 3:
            f.write(f"  ✓ 验证集上涨{val_df['return'].mean():.2f}%，适合做多\n")
        elif val_df['return'].mean() < -3:
            f.write(f"  ⚠️ 验证集下跌{abs(val_df['return'].mean()):.2f}%，做多难盈利\n")
        else:
            f.write(f"  ≈ 验证集横盘({val_df['return'].mean():.2f}%)，交易成本影响大\n")

    print(f"  ✓ Label {label_id} 完成")
    return train_df, val_df

# 主程序
all_results = {}

for label_id in LABELS:
    try:
        train_df, val_df = analyze_and_visualize_label(label_id)
        all_results[label_id] = {'train': train_df, 'val': val_df}
    except Exception as e:
        print(f"\n  ✗ Label {label_id} 失败: {e}")

# 跨Label对比图
if len(all_results) > 0:
    print(f"\n{'='*80}")
    print("生成跨Label对比图")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, len(LABELS), figsize=(6*len(LABELS), 10))
    if len(LABELS) == 1:
        axes = axes.reshape(-1, 1)

    for idx, label_id in enumerate(LABELS):
        if label_id not in all_results:
            continue

        train_df = all_results[label_id]['train']
        val_df = all_results[label_id]['val']

        axes[0, idx].boxplot([train_df['return'].values], tick_labels=[f'Label {label_id}'])
        axes[0, idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, idx].set_title(f'Label {label_id} - Train', fontweight='bold')
        axes[0, idx].set_ylabel('Return (%)')
        axes[0, idx].grid(True, alpha=0.3)

        axes[1, idx].boxplot([val_df['return'].values], tick_labels=[f'Label {label_id}'])
        axes[1, idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, idx].set_title(f'Label {label_id} - Val', fontweight='bold')
        axes[1, idx].set_ylabel('Return (%)')
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/labels_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n{'='*80}")
print("完成!")
print(f"{'='*80}")
print(f"\n结果目录: {OUTPUT_ROOT}/")
for label_id in LABELS:
    print(f"\nlabel{label_id}_analysis/")
    print(f"  ├── summary_comparison.png      (汇总对比)")
    print(f"  ├── return_distribution.png     (收益率分布)")
    print(f"  ├── training_files/             (训练集单文件详细图)")
    print(f"  ├── validation_files/           (验证集单文件详细图)")
    print(f"  └── statistics_report.txt       (统计报告)")
