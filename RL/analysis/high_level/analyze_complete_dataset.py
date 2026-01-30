#!/usr/bin/env python3
"""
High-Level完整数据集分析
直接分析验证集和测试集的完整价格走势和统计特征
不进行label分类，因为high-level训练使用的是whole目录的完整文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 配置
DATASET = "ETHUSDT"
DATA_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/data"
OUTPUT_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/analysis/high_level"
COMMISSION_FEE = 0.0005  # 单边手续费 0.05%

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 100

print("="*80)
print("High-Level 完整数据集分析")
print("="*80)


def analyze_full_dataset(df, dataset_name):
    """分析完整数据集"""

    print(f"\n{'='*80}")
    print(f"分析 {dataset_name.upper()} 数据集")
    print(f"{'='*80}")

    output_dir = f"{OUTPUT_ROOT}/{dataset_name}_complete_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # 基础信息
    print(f"  数据长度: {len(df)} 行")
    print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    print(f"  时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")

    # 计算收益率
    df['return'] = df['close'].pct_change()

    # ========================================================================
    # 1. 完整价格曲线
    # ========================================================================
    print(f"  [1/6] 生成完整价格曲线...")

    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # 原始价格
    ax = axes[0]
    ax.plot(df['close'].values, linewidth=0.5, color='blue', alpha=0.7)
    ax.set_title(f'{dataset_name.upper()} - Close Price', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps (Minutes)')
    ax.set_ylabel('Price (USDT)')
    ax.grid(True, alpha=0.3)

    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    total_return = (end_price / start_price - 1) * 100

    stats_text = f'Start: ${start_price:.2f}\nEnd: ${end_price:.2f}\nTotal Return: {total_return:.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 归一化价格
    ax = axes[1]
    normalized_price = (df['close'] / df.iloc[0]['close']) * 100
    ax.plot(normalized_price.values, linewidth=0.5, color='green', alpha=0.7)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{dataset_name.upper()} - Normalized Price (Start = 100)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps (Minutes)')
    ax.set_ylabel('Normalized Price')
    ax.grid(True, alpha=0.3)

    # 累计收益曲线
    ax = axes[2]
    cumulative_return = (1 + df['return'].fillna(0)).cumprod()
    ax.plot(cumulative_return.values, linewidth=0.8, color='purple', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{dataset_name.upper()} - Cumulative Return', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps (Minutes)')
    ax.set_ylabel('Cumulative Return (Initial = 1)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. 收益率分布
    # ========================================================================
    print(f"  [2/6] 生成收益率分布...")

    returns = df['return'].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 收益率直方图
    ax = axes[0, 0]
    ax.hist(returns, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=returns.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.4f}%')
    ax.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 收益率时间序列
    ax = axes[0, 1]
    ax.plot(returns.values, linewidth=0.3, color='blue', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Returns Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)

    # 正负收益分布
    ax = axes[1, 0]
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    data = [positive_returns * 100, negative_returns * 100]
    labels = ['Positive Returns', 'Negative Returns']
    colors = ['green', 'red']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Return (%)')
    ax.set_title('Positive vs Negative Returns', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q-Q图（正态性检验）
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(returns.values, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/return_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. 波动率分析
    # ========================================================================
    print(f"  [3/6] 生成波动率分析...")

    # 计算滚动波动率
    rolling_vol_60 = returns.rolling(window=60).std() * 100  # 1小时
    rolling_vol_360 = returns.rolling(window=360).std() * 100  # 6小时
    rolling_vol_1440 = returns.rolling(window=1440).std() * 100  # 1天

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # 滚动波动率
    ax = axes[0]
    ax.plot(rolling_vol_60.values, linewidth=0.5, alpha=0.7, label='60-min Rolling Vol', color='blue')
    ax.plot(rolling_vol_360.values, linewidth=0.8, alpha=0.8, label='360-min Rolling Vol', color='green')
    ax.plot(rolling_vol_1440.values, linewidth=1.0, alpha=0.9, label='1440-min Rolling Vol', color='red')
    ax.set_title(f'{dataset_name.upper()} - Rolling Volatility', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps (Minutes)')
    ax.set_ylabel('Volatility (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 波动率分布
    ax = axes[1]
    ax.hist(rolling_vol_60.dropna(), bins=50, alpha=0.5, label='60-min', color='blue')
    ax.hist(rolling_vol_360.dropna(), bins=50, alpha=0.5, label='360-min', color='green')
    ax.hist(rolling_vol_1440.dropna(), bins=50, alpha=0.5, label='1440-min', color='red')
    ax.set_title('Volatility Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/volatility_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. 交易机会分析
    # ========================================================================
    print(f"  [4/6] 生成交易机会分析...")

    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    # 累计交易机会
    cumulative_long_opportunity = positive_returns.sum() * 100
    cumulative_short_opportunity = abs(negative_returns.sum()) * 100

    # 考虑手续费
    total_trades_long = len(positive_returns)
    total_trades_short = len(negative_returns)

    net_long_opportunity = cumulative_long_opportunity - (total_trades_long * 2 * COMMISSION_FEE * 100)
    net_short_opportunity = cumulative_short_opportunity - (total_trades_short * 2 * COMMISSION_FEE * 100)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 做多做空机会对比
    ax = axes[0, 0]
    opportunities = ['Long\nGross', 'Long\nNet', 'Short\nGross', 'Short\nNet', 'Total\nNet']
    values = [
        cumulative_long_opportunity,
        net_long_opportunity,
        cumulative_short_opportunity,
        net_short_opportunity,
        net_long_opportunity + net_short_opportunity
    ]
    colors = ['lightblue', 'blue', 'lightcoral', 'red', 'purple']

    ax.bar(opportunities, values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Return (%)')
    ax.set_title('Trading Opportunities', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    # Hold vs 交易机会
    ax = axes[0, 1]
    comparison = ['Hold\nReturn', 'Net Trading\nOpportunity']
    comp_values = [total_return, net_long_opportunity + net_short_opportunity]
    comp_colors = ['gray', 'green']

    ax.bar(comparison, comp_values, color=comp_colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Return (%)')
    ax.set_title('Hold vs Trading Opportunity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    for i, v in enumerate(comp_values):
        ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    # 累计收益曲线对比（理想情况）
    ax = axes[1, 0]
    cumulative_long = positive_returns.cumsum() * 100
    cumulative_short = abs(negative_returns).cumsum() * 100

    # 填充索引以匹配原始数据长度
    cumulative_long_full = pd.Series(0, index=df.index)
    cumulative_short_full = pd.Series(0, index=df.index)

    for idx in positive_returns.index:
        cumulative_long_full[idx] = positive_returns.loc[:idx].sum() * 100
    for idx in negative_returns.index:
        cumulative_short_full[idx] = abs(negative_returns.loc[:idx]).sum() * 100

    ax.plot(cumulative_long_full.values, label='Cumulative Long Opp', color='blue', alpha=0.7, linewidth=1)
    ax.plot(cumulative_short_full.values, label='Cumulative Short Opp', color='red', alpha=0.7, linewidth=1)
    ax.plot((cumulative_long_full + cumulative_short_full).values, label='Total Opp', color='purple', alpha=0.8, linewidth=1.5)
    ax.set_title('Cumulative Trading Opportunities', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 正负收益分钟占比
    ax = axes[1, 1]
    positive_pct = len(positive_returns) / len(returns) * 100
    negative_pct = len(negative_returns) / len(returns) * 100
    zero_pct = 100 - positive_pct - negative_pct

    sizes = [positive_pct, negative_pct, zero_pct]
    labels_pie = [f'Positive\n{positive_pct:.1f}%', f'Negative\n{negative_pct:.1f}%', f'Zero\n{zero_pct:.1f}%']
    colors_pie = ['green', 'red', 'gray']

    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('Minute Return Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trading_opportunities.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. 风险指标
    # ========================================================================
    print(f"  [5/6] 生成风险指标分析...")

    # 最大回撤
    cumulative_returns = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # 回撤曲线
    ax = axes[0]
    ax.fill_between(range(len(drawdown)), drawdown.values, 0, color='red', alpha=0.3)
    ax.plot(drawdown.values, color='red', linewidth=0.8, alpha=0.8)
    ax.axhline(y=max_drawdown, color='darkred', linestyle='--', linewidth=2, label=f'Max Drawdown: {max_drawdown:.2f}%')
    ax.set_title(f'{dataset_name.upper()} - Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 滚动夏普比率
    ax = axes[1]
    rolling_mean = returns.rolling(window=1440).mean()
    rolling_std = returns.rolling(window=1440).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(1440)  # 年化

    ax.plot(rolling_sharpe.values, linewidth=0.8, color='purple', alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
    ax.set_title(f'{dataset_name.upper()} - Rolling Sharpe Ratio (1440-min window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/risk_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 6. 统计报告
    # ========================================================================
    print(f"  [6/6] 生成统计报告...")

    with open(f"{output_dir}/complete_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} 完整数据集统计报告\n")
        f.write("="*80 + "\n\n")

        f.write("📊 基础信息:\n")
        f.write(f"  数据长度: {len(df)} 行\n")
        f.write(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}\n")
        f.write(f"  时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天\n\n")

        f.write("="*80 + "\n")
        f.write("价格统计\n")
        f.write("="*80 + "\n\n")

        f.write("💰 价格信息:\n")
        f.write(f"  起始价格: ${start_price:.2f}\n")
        f.write(f"  结束价格: ${end_price:.2f}\n")
        f.write(f"  最高价格: ${df['close'].max():.2f}\n")
        f.write(f"  最低价格: ${df['close'].min():.2f}\n")
        f.write(f"  平均价格: ${df['close'].mean():.2f}\n")
        f.write(f"  总收益率: {total_return:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("收益率统计\n")
        f.write("="*80 + "\n\n")

        f.write("📈 收益率分布:\n")
        f.write(f"  平均收益率: {returns.mean()*100:.4f}%\n")
        f.write(f"  中位数: {returns.median()*100:.4f}%\n")
        f.write(f"  标准差: {returns.std()*100:.4f}%\n")
        f.write(f"  偏度: {returns.skew():.4f}\n")
        f.write(f"  峰度: {returns.kurtosis():.4f}\n")
        f.write(f"  最大单步收益: {returns.max()*100:.4f}%\n")
        f.write(f"  最大单步亏损: {returns.min()*100:.4f}%\n\n")

        f.write("📊 收益率分类:\n")
        f.write(f"  正收益分钟数: {len(positive_returns)} ({positive_pct:.2f}%)\n")
        f.write(f"  负收益分钟数: {len(negative_returns)} ({negative_pct:.2f}%)\n")
        f.write(f"  零收益分钟数: {len(returns) - len(positive_returns) - len(negative_returns)} ({zero_pct:.2f}%)\n\n")

        f.write("="*80 + "\n")
        f.write("交易机会分析\n")
        f.write("="*80 + "\n\n")

        f.write("💵 理论交易机会 (假设完美捕捉每个分钟的收益):\n")
        f.write(f"  做多总机会 (毛收益): {cumulative_long_opportunity:.2f}%\n")
        f.write(f"  做空总机会 (毛收益): {cumulative_short_opportunity:.2f}%\n")
        f.write(f"  总机会 (毛收益): {cumulative_long_opportunity + cumulative_short_opportunity:.2f}%\n\n")

        f.write(f"💰 考虑手续费后 (单边手续费{COMMISSION_FEE*100:.2f}%):\n")
        f.write(f"  做多净机会: {net_long_opportunity:.2f}%\n")
        f.write(f"  做空净机会: {net_short_opportunity:.2f}%\n")
        f.write(f"  总净机会: {net_long_opportunity + net_short_opportunity:.2f}%\n\n")

        f.write("📊 对比:\n")
        f.write(f"  传统Hold收益: {total_return:.2f}%\n")
        f.write(f"  理论最大净收益: {net_long_opportunity + net_short_opportunity:.2f}%\n")
        if abs(total_return) > 0.1:
            f.write(f"  理论收益倍数: {(net_long_opportunity + net_short_opportunity) / total_return:.2f}x\n\n")

        f.write("="*80 + "\n")
        f.write("波动率分析\n")
        f.write("="*80 + "\n\n")

        f.write("📉 波动率统计:\n")
        f.write(f"  整体波动率: {returns.std()*100:.4f}%\n")
        f.write(f"  60分钟平均波动率: {rolling_vol_60.mean():.4f}%\n")
        f.write(f"  360分钟平均波动率: {rolling_vol_360.mean():.4f}%\n")
        f.write(f"  1440分钟平均波动率: {rolling_vol_1440.mean():.4f}%\n\n")

        f.write("="*80 + "\n")
        f.write("风险指标\n")
        f.write("="*80 + "\n\n")

        f.write("⚠️ 风险统计:\n")
        f.write(f"  最大回撤: {max_drawdown:.2f}%\n")

        # 计算夏普比率
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = (mean_return / std_return * np.sqrt(525600)) if std_return > 0 else 0  # 年化 (分钟数)
        f.write(f"  夏普比率 (年化): {sharpe_ratio:.4f}\n")

        # VaR (Value at Risk)
        var_95 = returns.quantile(0.05) * 100
        var_99 = returns.quantile(0.01) * 100
        f.write(f"  VaR (95%): {var_95:.4f}%\n")
        f.write(f"  VaR (99%): {var_99:.4f}%\n\n")

        f.write("="*80 + "\n")
        f.write("策略建议\n")
        f.write("="*80 + "\n\n")

        f.write("🎯 基于数据特征的策略建议:\n\n")

        if net_long_opportunity > net_short_opportunity * 1.5:
            f.write(f"  ✓ 做多机会更丰富 (做多: {net_long_opportunity:.2f}% vs 做空: {net_short_opportunity:.2f}%)\n")
            f.write(f"    建议偏向做多策略\n\n")
        elif net_short_opportunity > net_long_opportunity * 1.5:
            f.write(f"  ✓ 做空机会更丰富 (做空: {net_short_opportunity:.2f}% vs 做多: {net_long_opportunity:.2f}%)\n")
            f.write(f"    建议偏向做空策略\n\n")
        else:
            f.write(f"  ≈ 做多做空机会均衡 (做多: {net_long_opportunity:.2f}%, 做空: {net_short_opportunity:.2f}%)\n")
            f.write(f"    建议双向交易策略\n\n")

        if net_long_opportunity + net_short_opportunity > total_return * 3:
            f.write(f"  🔍 内部波动远大于整体趋势\n")
            f.write(f"    理论机会是Hold收益的 {(net_long_opportunity + net_short_opportunity) / total_return:.1f} 倍\n")
            f.write(f"    高频交易策略可能更有优势\n\n")

        if max_drawdown < -10:
            f.write(f"  ⚠️ 最大回撤较大 ({max_drawdown:.2f}%)\n")
            f.write(f"    需要严格的风险控制和止损策略\n\n")

        if sharpe_ratio < 0.5:
            f.write(f"  ⚠️ 夏普比率较低 ({sharpe_ratio:.2f})\n")
            f.write(f"    风险调整后收益不佳，需谨慎交易\n\n")

    print(f"  ✓ {dataset_name.upper()} 分析完成")
    print(f"  📁 结果保存在: {output_dir}/")


# 主程序
print(f"\n加载数据...")

# 检查数据文件
val_file = f"{DATA_ROOT}/{DATASET}/whole/val.feather"
test_file = f"{DATA_ROOT}/{DATASET}/whole/test.feather"

datasets = []

if os.path.exists(val_file):
    print(f"  ✓ 找到验证集: {val_file}")
    val_df = pd.read_feather(val_file)
    datasets.append(('val', val_df))
else:
    print(f"  ⚠️ 验证集文件不存在: {val_file}")

if os.path.exists(test_file):
    print(f"  ✓ 找到测试集: {test_file}")
    test_df = pd.read_feather(test_file)
    datasets.append(('test', test_df))
else:
    print(f"  ⚠️ 测试集文件不存在: {test_file}")

if not datasets:
    print("\n❌ 错误: 没有找到验证集或测试集文件")
    exit(1)

# 分析每个数据集
for dataset_name, df in datasets:
    analyze_full_dataset(df, dataset_name)

print(f"\n{'='*80}")
print("✓ 全部完成!")
print(f"{'='*80}")
print(f"\n📁 结果目录: {OUTPUT_ROOT}/")
print(f"  ├── val_complete_analysis/")
print(f"  │   ├── price_curves.png              (价格曲线)")
print(f"  │   ├── return_distribution.png       (收益率分布)")
print(f"  │   ├── volatility_analysis.png       (波动率分析)")
print(f"  │   ├── trading_opportunities.png     (交易机会)")
print(f"  │   ├── risk_metrics.png              (风险指标)")
print(f"  │   └── complete_analysis_report.txt  (完整报告)")
print(f"  └── test_complete_analysis/")
print(f"      └── ... (同上)")
