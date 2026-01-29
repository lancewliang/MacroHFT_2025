#!/usr/bin/env python3
"""
分钟级交易机会分析：更公平的数据分布统计
适用于分钟级交易策略，统计文件内部的真实交易机会
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
COMMISSION_FEE = 0.0005  # 单边手续费 0.05%

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 100

print("="*80)
print("分钟级交易机会分析 (Minute-Level Trading Opportunities)")
print("="*80)

# 加载标签索引
with open(f"{DATA_ROOT}/{DATASET}/train/slope_labels.pkl", 'rb') as f:
    train_index = pickle.load(f)
with open(f"{DATA_ROOT}/{DATASET}/val/slope_labels.pkl", 'rb') as f:
    val_index = pickle.load(f)


def analyze_minute_level_stats(df, commission_fee=0.0005):
    """
    计算分钟级别的详细统计指标

    返回:
        dict: 包含多个统计指标的字典
    """
    # 基础价格信息
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']

    # 分钟级收益率
    returns = df['close'].pct_change().dropna()

    # 首尾收益（传统方法）
    hold_return = (end_price / start_price - 1) * 100

    # 分钟级统计
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    # 累计交易机会（理想情况：每次都能抓住）
    cumulative_long_opportunity = positive_returns.sum() * 100  # 做多机会
    cumulative_short_opportunity = abs(negative_returns.sum()) * 100  # 做空机会

    # 考虑交易成本的净收益（假设每分钟都交易）
    # 每次交易双边成本 = 2 * commission_fee
    total_trades_long = len(positive_returns)
    total_trades_short = len(negative_returns)

    net_long_opportunity = cumulative_long_opportunity - (total_trades_long * 2 * commission_fee * 100)
    net_short_opportunity = cumulative_short_opportunity - (total_trades_short * 2 * commission_fee * 100)

    # 最大回撤（从最高点到最低点的最大跌幅）
    cumulative_returns = (1 + df['close'].pct_change().fillna(0)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # 趋势段分析（连续上涨/下跌段）
    price_changes = df['close'].diff()
    trends = []
    current_trend = {'type': None, 'start': 0, 'length': 0, 'cumulative_change': 0}

    for i, change in enumerate(price_changes[1:], start=1):
        if change > 0:
            trend_type = 'up'
        elif change < 0:
            trend_type = 'down'
        else:
            trend_type = 'flat'

        if trend_type == current_trend['type']:
            current_trend['length'] += 1
            current_trend['cumulative_change'] += change
        else:
            if current_trend['type'] is not None:
                trends.append(current_trend.copy())
            current_trend = {'type': trend_type, 'start': i, 'length': 1, 'cumulative_change': change}

    # 添加最后一个趋势
    if current_trend['type'] is not None:
        trends.append(current_trend)

    # 统计趋势段
    up_trends = [t for t in trends if t['type'] == 'up']
    down_trends = [t for t in trends if t['type'] == 'down']

    up_trend_count = len(up_trends)
    down_trend_count = len(down_trends)
    avg_up_trend_length = np.mean([t['length'] for t in up_trends]) if up_trends else 0
    avg_down_trend_length = np.mean([t['length'] for t in down_trends]) if down_trends else 0
    max_up_trend_return = max([t['cumulative_change']/start_price*100 for t in up_trends]) if up_trends else 0
    max_down_trend_return = min([t['cumulative_change']/start_price*100 for t in down_trends]) if down_trends else 0

    # 波动率
    volatility = returns.std() * 100

    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(len(returns))) if returns.std() > 0 else 0

    return {
        # 传统指标
        'hold_return': hold_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,

        # 分钟级统计
        'total_minutes': len(df),
        'positive_minutes': len(positive_returns),
        'negative_minutes': len(negative_returns),
        'positive_pct': len(positive_returns) / len(returns) * 100,
        'negative_pct': len(negative_returns) / len(returns) * 100,
        'avg_minute_return': returns.mean() * 100,

        # 累计交易机会（毛收益）
        'cumulative_long_opportunity': cumulative_long_opportunity,
        'cumulative_short_opportunity': cumulative_short_opportunity,
        'total_opportunity': cumulative_long_opportunity + cumulative_short_opportunity,

        # 考虑交易成本后的净收益
        'net_long_opportunity': net_long_opportunity,
        'net_short_opportunity': net_short_opportunity,
        'net_total_opportunity': net_long_opportunity + net_short_opportunity,

        # 极值
        'max_minute_gain': returns.max() * 100,
        'max_minute_loss': returns.min() * 100,

        # 趋势段统计
        'up_trend_count': up_trend_count,
        'down_trend_count': down_trend_count,
        'avg_up_trend_length': avg_up_trend_length,
        'avg_down_trend_length': avg_down_trend_length,
        'max_up_trend_return': max_up_trend_return,
        'max_down_trend_return': max_down_trend_return,
    }


def analyze_label_minute_level(label_id):
    """完整分析单个label的分钟级交易机会"""

    print(f"\n{'='*80}")
    print(f"处理 Label {label_id}")
    print(f"{'='*80}")

    label_dir = f"{OUTPUT_ROOT}/label{label_id}_minute_analysis"
    os.makedirs(label_dir, exist_ok=True)

    train_files = train_index[label_id]
    val_files = val_index[label_id]

    # 分析训练集
    print(f"  分析训练集 ({len(train_files)} 个文件)...")
    train_stats = []
    for file_id in train_files:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/train/df_{file_id}.feather")
            stats = analyze_minute_level_stats(df, COMMISSION_FEE)
            stats['file_id'] = file_id
            train_stats.append(stats)
        except Exception as e:
            print(f"    警告: 训练集文件{file_id}处理失败: {e}")

    # 分析验证集
    print(f"  分析验证集 ({len(val_files)} 个文件)...")
    val_stats = []
    for file_id in val_files:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/val/df_{file_id}.feather")
            stats = analyze_minute_level_stats(df, COMMISSION_FEE)
            stats['file_id'] = file_id
            val_stats.append(stats)
        except Exception as e:
            print(f"    警告: 验证集文件{file_id}处理失败: {e}")

    train_df = pd.DataFrame(train_stats)
    val_df = pd.DataFrame(val_stats)

    # 生成可视化
    print(f"  生成可视化图表...")

    # 1. 传统hold return vs 实际交易机会对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 训练集：传统return vs 交易机会
    ax = axes[0, 0]
    x_pos = np.arange(len(train_df))
    width = 0.35
    ax.bar(x_pos - width/2, train_df['hold_return'], width, label='Hold Return (传统)', alpha=0.7, color='gray')
    ax.bar(x_pos + width/2, train_df['net_total_opportunity'], width, label='Net Trading Opportunity (分钟级)', alpha=0.7, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 训练集: 传统Hold vs 分钟级交易机会', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 验证集：传统return vs 交易机会
    ax = axes[0, 1]
    x_pos = np.arange(len(val_df))
    ax.bar(x_pos - width/2, val_df['hold_return'], width, label='Hold Return (传统)', alpha=0.7, color='gray')
    ax.bar(x_pos + width/2, val_df['net_total_opportunity'], width, label='Net Trading Opportunity (分钟级)', alpha=0.7, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 验证集: 传统Hold vs 分钟级交易机会', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 训练集：做多vs做空机会
    ax = axes[1, 0]
    x_pos = np.arange(len(train_df))
    ax.bar(x_pos - width/2, train_df['net_long_opportunity'], width, label='Long Opportunity (做多)', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, train_df['net_short_opportunity'], width, label='Short Opportunity (做空)', alpha=0.7, color='red')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Net Return (%)')
    ax.set_title(f'Label {label_id} - 训练集: 做多 vs 做空机会（扣除手续费）', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 验证集：做多vs做空机会
    ax = axes[1, 1]
    x_pos = np.arange(len(val_df))
    ax.bar(x_pos - width/2, val_df['net_long_opportunity'], width, label='Long Opportunity (做多)', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, val_df['net_short_opportunity'], width, label='Short Opportunity (做空)', alpha=0.7, color='red')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Net Return (%)')
    ax.set_title(f'Label {label_id} - 验证集: 做多 vs 做空机会（扣除手续费）', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{label_dir}/opportunity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 箱线图对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    data_to_plot = [train_df['hold_return'], train_df['net_long_opportunity'],
                    train_df['net_short_opportunity'], train_df['net_total_opportunity']]
    ax.boxplot(data_to_plot, labels=['Hold\nReturn', 'Long\nOpp', 'Short\nOpp', 'Total\nOpp'])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 训练集收益分布对比', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    data_to_plot = [val_df['hold_return'], val_df['net_long_opportunity'],
                    val_df['net_short_opportunity'], val_df['net_total_opportunity']]
    ax.boxplot(data_to_plot, labels=['Hold\nReturn', 'Long\nOpp', 'Short\nOpp', 'Total\nOpp'])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 验证集收益分布对比', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{label_dir}/return_distribution_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 生成详细统计报告
    print(f"  生成统计报告...")
    with open(f"{label_dir}/minute_level_statistics.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Label {label_id} 分钟级交易机会统计报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"📊 数据集概况:\n")
        f.write(f"  训练集文件数: {len(train_df)}\n")
        f.write(f"  验证集文件数: {len(val_df)}\n")
        f.write(f"  单边手续费: {COMMISSION_FEE*100:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("训练集统计\n")
        f.write("="*80 + "\n\n")

        f.write("1️⃣ 传统方法 (首尾价格Hold Return):\n")
        f.write(f"  平均收益率: {train_df['hold_return'].mean():.2f}%\n")
        f.write(f"  中位数: {train_df['hold_return'].median():.2f}%\n")
        f.write(f"  标准差: {train_df['hold_return'].std():.2f}%\n")
        f.write(f"  范围: [{train_df['hold_return'].min():.2f}%, {train_df['hold_return'].max():.2f}%]\n\n")

        f.write("2️⃣ 分钟级交易机会 (毛收益，未扣除手续费):\n")
        f.write(f"  平均做多机会: {train_df['cumulative_long_opportunity'].mean():.2f}%\n")
        f.write(f"  平均做空机会: {train_df['cumulative_short_opportunity'].mean():.2f}%\n")
        f.write(f"  平均总机会: {train_df['total_opportunity'].mean():.2f}%\n\n")

        f.write("3️⃣ 分钟级交易机会 (净收益，已扣除手续费):\n")
        f.write(f"  平均做多净收益: {train_df['net_long_opportunity'].mean():.2f}%\n")
        f.write(f"  平均做空净收益: {train_df['net_short_opportunity'].mean():.2f}%\n")
        f.write(f"  平均总净收益: {train_df['net_total_opportunity'].mean():.2f}%\n\n")

        f.write("4️⃣ 分钟级分布:\n")
        f.write(f"  平均正收益分钟占比: {train_df['positive_pct'].mean():.1f}%\n")
        f.write(f"  平均负收益分钟占比: {train_df['negative_pct'].mean():.1f}%\n")
        f.write(f"  平均单分钟收益: {train_df['avg_minute_return'].mean():.4f}%\n\n")

        f.write("5️⃣ 趋势段统计:\n")
        f.write(f"  平均上涨段数: {train_df['up_trend_count'].mean():.1f}\n")
        f.write(f"  平均下跌段数: {train_df['down_trend_count'].mean():.1f}\n")
        f.write(f"  平均上涨段长度: {train_df['avg_up_trend_length'].mean():.1f} 分钟\n")
        f.write(f"  平均下跌段长度: {train_df['avg_down_trend_length'].mean():.1f} 分钟\n")
        f.write(f"  最大单段上涨: {train_df['max_up_trend_return'].mean():.2f}%\n")
        f.write(f"  最大单段下跌: {train_df['max_down_trend_return'].mean():.2f}%\n\n")

        f.write("6️⃣ 风险指标:\n")
        f.write(f"  平均波动率: {train_df['volatility'].mean():.4f}%\n")
        f.write(f"  平均最大回撤: {train_df['max_drawdown'].mean():.2f}%\n")
        f.write(f"  平均夏普比率: {train_df['sharpe_ratio'].mean():.2f}\n\n")

        f.write("="*80 + "\n")
        f.write("验证集统计\n")
        f.write("="*80 + "\n\n")

        f.write("1️⃣ 传统方法 (首尾价格Hold Return):\n")
        f.write(f"  平均收益率: {val_df['hold_return'].mean():.2f}%\n")
        f.write(f"  中位数: {val_df['hold_return'].median():.2f}%\n")
        f.write(f"  标准差: {val_df['hold_return'].std():.2f}%\n")
        f.write(f"  范围: [{val_df['hold_return'].min():.2f}%, {val_df['hold_return'].max():.2f}%]\n\n")

        f.write("2️⃣ 分钟级交易机会 (毛收益，未扣除手续费):\n")
        f.write(f"  平均做多机会: {val_df['cumulative_long_opportunity'].mean():.2f}%\n")
        f.write(f"  平均做空机会: {val_df['cumulative_short_opportunity'].mean():.2f}%\n")
        f.write(f"  平均总机会: {val_df['total_opportunity'].mean():.2f}%\n\n")

        f.write("3️⃣ 分钟级交易机会 (净收益，已扣除手续费):\n")
        f.write(f"  平均做多净收益: {val_df['net_long_opportunity'].mean():.2f}%\n")
        f.write(f"  平均做空净收益: {val_df['net_short_opportunity'].mean():.2f}%\n")
        f.write(f"  平均总净收益: {val_df['net_total_opportunity'].mean():.2f}%\n\n")

        f.write("4️⃣ 分钟级分布:\n")
        f.write(f"  平均正收益分钟占比: {val_df['positive_pct'].mean():.1f}%\n")
        f.write(f"  平均负收益分钟占比: {val_df['negative_pct'].mean():.1f}%\n")
        f.write(f"  平均单分钟收益: {val_df['avg_minute_return'].mean():.4f}%\n\n")

        f.write("5️⃣ 趋势段统计:\n")
        f.write(f"  平均上涨段数: {val_df['up_trend_count'].mean():.1f}\n")
        f.write(f"  平均下跌段数: {val_df['down_trend_count'].mean():.1f}\n")
        f.write(f"  平均上涨段长度: {val_df['avg_up_trend_length'].mean():.1f} 分钟\n")
        f.write(f"  平均下跌段长度: {val_df['avg_down_trend_length'].mean():.1f} 分钟\n")
        f.write(f"  最大单段上涨: {val_df['max_up_trend_return'].mean():.2f}%\n")
        f.write(f"  最大单段下跌: {val_df['max_down_trend_return'].mean():.2f}%\n\n")

        f.write("6️⃣ 风险指标:\n")
        f.write(f"  平均波动率: {val_df['volatility'].mean():.4f}%\n")
        f.write(f"  平均最大回撤: {val_df['max_drawdown'].mean():.2f}%\n")
        f.write(f"  平均夏普比率: {val_df['sharpe_ratio'].mean():.2f}\n\n")

        f.write("="*80 + "\n")
        f.write("关键对比与诊断\n")
        f.write("="*80 + "\n\n")

        hold_return_val = val_df['hold_return'].mean()
        net_opportunity_val = val_df['net_total_opportunity'].mean()
        long_opportunity_val = val_df['net_long_opportunity'].mean()
        short_opportunity_val = val_df['net_short_opportunity'].mean()

        f.write(f"📈 验证集传统Hold收益: {hold_return_val:.2f}%\n")
        f.write(f"💰 验证集分钟级净机会: {net_opportunity_val:.2f}%\n")
        f.write(f"📊 差异倍数: {net_opportunity_val/hold_return_val:.1f}x\n\n" if abs(hold_return_val) > 0.1 else "")

        f.write("🎯 策略建议:\n")
        if long_opportunity_val > 5 and long_opportunity_val > abs(short_opportunity_val):
            f.write(f"  ✓ 做多策略优势明显 (净收益: {long_opportunity_val:.2f}%)\n")
        elif short_opportunity_val > 5 and short_opportunity_val > long_opportunity_val:
            f.write(f"  ✓ 做空策略优势明显 (净收益: {short_opportunity_val:.2f}%)\n")
        elif long_opportunity_val > 2 and short_opportunity_val > 2:
            f.write(f"  ✓ 双向交易机会充足 (做多: {long_opportunity_val:.2f}%, 做空: {short_opportunity_val:.2f}%)\n")
        elif net_opportunity_val < 2:
            f.write(f"  ⚠️ 交易机会有限，手续费侵蚀明显，建议减少交易频率或不交易\n")

        if abs(hold_return_val) < 1 and net_opportunity_val > 10:
            f.write(f"  🔍 注意：首尾价格几乎持平({hold_return_val:.2f}%)，但内部波动大，分钟级交易优势显著\n")

    print(f"  ✓ Label {label_id} 完成")
    return train_df, val_df


# 主程序
all_results = {}

for label_id in LABELS:
    try:
        train_df, val_df = analyze_label_minute_level(label_id)
        all_results[label_id] = {'train': train_df, 'val': val_df}
    except Exception as e:
        print(f"\n  ✗ Label {label_id} 失败: {e}")

# 跨Label对比
if len(all_results) > 1:
    print(f"\n{'='*80}")
    print("生成跨Label对比")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    labels_list = sorted(all_results.keys())

    # 训练集对比
    hold_returns_train = [all_results[l]['train']['hold_return'].mean() for l in labels_list]
    net_opps_train = [all_results[l]['train']['net_total_opportunity'].mean() for l in labels_list]

    ax = axes[0, 0]
    x = np.arange(len(labels_list))
    width = 0.35
    ax.bar(x - width/2, hold_returns_train, width, label='Hold Return', alpha=0.7, color='gray')
    ax.bar(x + width/2, net_opps_train, width, label='Net Trading Opp', alpha=0.7, color='green')
    ax.set_xlabel('Label')
    ax.set_ylabel('Return (%)')
    ax.set_title('训练集: 传统Hold vs 分钟级交易', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # 验证集对比
    hold_returns_val = [all_results[l]['val']['hold_return'].mean() for l in labels_list]
    net_opps_val = [all_results[l]['val']['net_total_opportunity'].mean() for l in labels_list]

    ax = axes[0, 1]
    ax.bar(x - width/2, hold_returns_val, width, label='Hold Return', alpha=0.7, color='gray')
    ax.bar(x + width/2, net_opps_val, width, label='Net Trading Opp', alpha=0.7, color='green')
    ax.set_xlabel('Label')
    ax.set_ylabel('Return (%)')
    ax.set_title('验证集: 传统Hold vs 分钟级交易', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # 验证集：做多vs做空机会
    long_opps_val = [all_results[l]['val']['net_long_opportunity'].mean() for l in labels_list]
    short_opps_val = [all_results[l]['val']['net_short_opportunity'].mean() for l in labels_list]

    ax = axes[1, 0]
    ax.bar(x - width/2, long_opps_val, width, label='Long Opp', alpha=0.7, color='blue')
    ax.bar(x + width/2, short_opps_val, width, label='Short Opp', alpha=0.7, color='red')
    ax.set_xlabel('Label')
    ax.set_ylabel('Net Return (%)')
    ax.set_title('验证集: 做多 vs 做空机会', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # 收益倍数对比
    multipliers = []
    for l in labels_list:
        hold_r = all_results[l]['val']['hold_return'].mean()
        net_opp = all_results[l]['val']['net_total_opportunity'].mean()
        if abs(hold_r) > 0.1:
            multipliers.append(net_opp / hold_r)
        else:
            multipliers.append(0)

    ax = axes[1, 1]
    colors = ['green' if m > 1 else 'red' for m in multipliers]
    ax.bar(x, multipliers, color=colors, alpha=0.7)
    ax.set_xlabel('Label')
    ax.set_ylabel('Multiplier (倍数)')
    ax.set_title('验证集: 分钟级交易收益 / Hold收益', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/cross_label_minute_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n{'='*80}")
print("完成!")
print(f"{'='*80}")
print(f"\n结果目录: {OUTPUT_ROOT}/")
for label_id in LABELS:
    print(f"\nlabel{label_id}_minute_analysis/")
    print(f"  ├── opportunity_comparison.png          (机会对比)")
    print(f"  ├── return_distribution_boxplot.png     (收益分布箱线图)")
    print(f"  └── minute_level_statistics.txt         (详细统计报告)")
print(f"\ncross_label_minute_comparison.png          (跨Label对比)")
