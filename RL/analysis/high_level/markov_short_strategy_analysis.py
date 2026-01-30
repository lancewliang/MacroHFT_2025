#!/usr/bin/env python3
"""
High-Level马尔科夫链做空策略分析
分析验证集和测试集上做空策略的利润空间
费率：万分之5 (0.05%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict

# 配置
DATASET = "ETHUSDT"
DATA_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/data"
OUTPUT_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/analysis/high_level"
COMMISSION_FEE = 0.0005  # 单边手续费 0.05% (万分之5)

# 马尔科夫状态配置
PRICE_CHANGE_BINS = [-np.inf, -0.002, -0.001, -0.0005, 0, 0.0005, 0.001, 0.002, np.inf]
PRICE_STATE_LABELS = ['大跌', '中跌', '较小跌', '小跌', '小涨', '较小涨', '中涨', '大涨']

# 动作空间
ACTIONS = ['hold', 'short', 'close']  # 持有空仓、开空、平空
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("High-Level 马尔科夫链做空策略分析")
print(f"费率: {COMMISSION_FEE*100:.2f}% (万分之{COMMISSION_FEE*10000:.0f})")
print("="*80)


class MarkovShortTradingMDP:
    """马尔科夫决策过程用于做空策略优化"""

    def __init__(self, price_change_bins, commission_fee=0.0005):
        self.price_change_bins = price_change_bins
        self.n_price_states = len(price_change_bins) - 1
        self.commission_fee = commission_fee

        # 状态空间：(price_state, position_state)
        # position_state: 0=空仓, 1=持有空头
        self.position_states = ['empty', 'short']
        self.n_states = self.n_price_states * len(self.position_states)

        # 动作空间
        self.actions = ACTIONS
        self.n_actions = len(self.actions)

        # 转移概率矩阵
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states))

        # 奖励矩阵
        self.rewards = np.zeros((self.n_states, self.n_actions))

        # 统计数据
        self.price_state_counts = np.zeros(self.n_price_states)
        self.price_transition_counts = np.zeros((self.n_price_states, self.n_price_states))

    def discretize_price_change(self, price_change):
        """将连续的价格变化离散化为状态"""
        state = np.digitize(price_change, self.price_change_bins) - 1
        return max(0, min(state, self.n_price_states - 1))

    def state_to_index(self, price_state, position_state):
        """将(price_state, position_state)转换为状态索引"""
        pos_idx = self.position_states.index(position_state)
        return price_state * len(self.position_states) + pos_idx

    def index_to_state(self, state_idx):
        """将状态索引转换为(price_state, position_state)"""
        price_state = state_idx // len(self.position_states)
        pos_idx = state_idx % len(self.position_states)
        position_state = self.position_states[pos_idx]
        return price_state, position_state

    def learn_from_data(self, df):
        """从数据中学习转移概率"""
        returns = df['close'].pct_change().dropna()

        # 离散化价格状态
        price_states = [self.discretize_price_change(r) for r in returns]

        # 统计价格状态转移
        for i in range(len(price_states) - 1):
            current_state = price_states[i]
            next_state = price_states[i + 1]
            self.price_state_counts[current_state] += 1
            self.price_transition_counts[current_state, next_state] += 1

        print(f"    学习了 {len(price_states)} 个状态转移")

    def finalize_transition_probs(self):
        """归一化转移概率"""
        # 归一化转移概率
        for i in range(self.n_price_states):
            if self.price_state_counts[i] > 0:
                self.price_transition_counts[i] /= self.price_state_counts[i]

        # 填充完整的状态转移概率矩阵
        for price_state in range(self.n_price_states):
            for position in self.position_states:
                current_state_idx = self.state_to_index(price_state, position)

                for action in self.actions:
                    action_idx = ACTION_TO_IDX[action]

                    # 根据当前仓位和动作，确定下一个仓位
                    if position == 'empty':
                        if action == 'short':
                            next_position = 'short'
                        else:  # hold or close
                            next_position = 'empty'
                    elif position == 'short':
                        if action == 'close':
                            next_position = 'empty'
                        else:
                            next_position = 'short'

                    # 填充转移概率
                    for next_price_state in range(self.n_price_states):
                        next_state_idx = self.state_to_index(next_price_state, next_position)
                        self.transition_probs[current_state_idx, action_idx, next_state_idx] = \
                            self.price_transition_counts[price_state, next_price_state]

    def calculate_rewards(self):
        """计算每个状态-动作对的即时奖励"""
        # 价格变化的期望值
        price_change_midpoints = []
        for i in range(len(self.price_change_bins) - 1):
            left = self.price_change_bins[i]
            right = self.price_change_bins[i + 1]
            if np.isinf(left):
                left = -0.003
            if np.isinf(right):
                right = 0.003
            midpoint = (left + right) / 2
            price_change_midpoints.append(midpoint)

        for state_idx in range(self.n_states):
            price_state, position = self.index_to_state(state_idx)
            price_change = price_change_midpoints[price_state]

            for action in self.actions:
                action_idx = ACTION_TO_IDX[action]
                reward = 0

                if position == 'empty':
                    if action == 'short':
                        # 开空仓：支付手续费
                        reward = -self.commission_fee
                    else:
                        # 空仓不交易
                        reward = 0

                elif position == 'short':
                    if action == 'close':
                        # 平空仓：获得价格下跌收益，支付手续费
                        reward = -price_change - self.commission_fee
                    else:
                        # 持有空仓：获得价格下跌收益
                        reward = -price_change

                self.rewards[state_idx, action_idx] = reward

    def value_iteration(self, gamma=0.99, theta=1e-6, max_iterations=1000):
        """值迭代算法求解最优策略"""
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)

        for iteration in range(max_iterations):
            delta = 0
            V_old = V.copy()

            for state in range(self.n_states):
                Q_values = np.zeros(self.n_actions)

                for action in range(self.n_actions):
                    immediate_reward = self.rewards[state, action]
                    expected_future_value = np.sum(
                        self.transition_probs[state, action, :] * V_old
                    )
                    Q_values[action] = immediate_reward + gamma * expected_future_value

                V[state] = np.max(Q_values)
                policy[state] = np.argmax(Q_values)

                delta = max(delta, abs(V[state] - V_old[state]))

            if delta < theta:
                print(f"    值迭代在第 {iteration + 1} 次迭代后收敛")
                break

        return V, policy

    def simulate_trading(self, df, policy):
        """使用策略模拟交易"""
        returns = df['close'].pct_change().dropna()
        price_states = [self.discretize_price_change(r) for r in returns]

        position = 'empty'
        cumulative_profit = 0
        profits = []
        actions_taken = []
        positions_history = []

        entry_price = 0
        num_trades = 0

        for i, (price_state, price_return) in enumerate(zip(price_states, returns)):
            state_idx = self.state_to_index(price_state, position)
            action_idx = policy[state_idx]
            action = self.actions[action_idx]
            actions_taken.append(action)

            # 执行动作并计算收益
            profit = 0

            if position == 'empty':
                if action == 'short':
                    position = 'short'
                    entry_price = df.iloc[i + 1]['close']  # +1 因为 returns 比 df 短 1
                    profit = -self.commission_fee
                    num_trades += 1

            elif position == 'short':
                if action == 'close':
                    profit = -price_return - self.commission_fee
                    position = 'empty'
                    num_trades += 1
                else:
                    profit = -price_return

            cumulative_profit += profit
            profits.append(cumulative_profit)
            positions_history.append(1 if position == 'short' else 0)

        return cumulative_profit, profits, actions_taken, positions_history, num_trades


def analyze_dataset_short_strategy(df, dataset_name):
    """分析单个数据集的做空策略"""

    print(f"\n{'='*80}")
    print(f"分析 {dataset_name.upper()} 数据集")
    print(f"{'='*80}")

    output_dir = f"{OUTPUT_ROOT}/{dataset_name}_short_strategy_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # 基础信息
    print(f"  数据长度: {len(df)} 行")
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    hold_return = (end_price / start_price - 1) * 100

    # 创建MDP模型
    print(f"  [1/6] 创建马尔科夫模型...")
    mdp = MarkovShortTradingMDP(PRICE_CHANGE_BINS, COMMISSION_FEE)

    print(f"  [2/6] 学习转移概率...")
    mdp.learn_from_data(df)
    mdp.finalize_transition_probs()

    print(f"  [3/6] 计算奖励函数...")
    mdp.calculate_rewards()

    print(f"  [4/6] 求解最优策略...")
    V, policy = mdp.value_iteration()

    print(f"  [5/6] 模拟交易...")
    total_profit, profits, actions, positions, num_trades = mdp.simulate_trading(df, policy)

    # 分析不同策略
    print(f"  [6/6] 对比分析...")

    # 策略1: Hold (不交易)
    hold_profit = 0

    # 策略2: 纯做空 (一直持有空头)
    returns = df['close'].pct_change().dropna()
    pure_short_profit = (-returns.sum() - 2 * COMMISSION_FEE) * 100  # 开仓关仓各一次手续费

    # 策略3: 最优马尔科夫策略
    optimal_profit = total_profit * 100

    # 策略4: 理论最大做空机会（捕捉所有下跌）
    negative_returns = returns[returns < 0]
    max_short_opportunity = (abs(negative_returns.sum()) - len(negative_returns) * 2 * COMMISSION_FEE) * 100

    # ========================================================================
    # 可视化
    # ========================================================================

    # 1. 策略收益对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 收益对比柱状图
    ax = axes[0, 0]
    strategies = ['Hold\n(不交易)', 'Pure Short\n(一直做空)', 'Optimal\nMarkov', 'Max Short\nOpportunity']
    strategy_profits = [hold_profit, pure_short_profit, optimal_profit, max_short_opportunity]
    colors = ['gray', 'orange', 'green' if optimal_profit > 0 else 'red', 'lightblue']

    bars = ax.bar(strategies, strategy_profits, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Profit (%)', fontsize=12)
    ax.set_title(f'{dataset_name.upper()} - Strategy Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(strategy_profits):
        ax.text(i, v, f'{v:.2f}%', ha='center',
               va='bottom' if v > 0 else 'top',
               fontweight='bold', fontsize=10)

    # 累计收益曲线
    ax = axes[0, 1]
    ax.plot(profits, label='Optimal Markov', color='green' if optimal_profit > 0 else 'red', linewidth=2)

    # 纯做空策略的累计收益
    pure_short_cumulative = (-returns.cumsum() * 100).values
    # 减去开仓手续费
    pure_short_cumulative = pure_short_cumulative - COMMISSION_FEE * 100
    ax.plot(pure_short_cumulative, label='Pure Short', color='orange', alpha=0.7, linewidth=1.5)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'{dataset_name.upper()} - Cumulative Profit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Cumulative Profit (%)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 持仓情况
    ax = axes[1, 0]
    ax.fill_between(range(len(positions)), positions, alpha=0.3, color='red', label='Short Position')
    ax.set_title(f'{dataset_name.upper()} - Position History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Position (1=Short, 0=Empty)', fontsize=12)
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加统计文本
    stats_text = f'Total Trades: {num_trades}\n'
    stats_text += f'Short Time: {sum(positions)/len(positions)*100:.1f}%\n'
    stats_text += f'Profit: {optimal_profit:.2f}%\n'
    stats_text += f'Profit per Trade: {optimal_profit/num_trades:.2f}%' if num_trades > 0 else 'No Trades'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 动作分布
    ax = axes[1, 1]
    action_counts = {a: actions.count(a) for a in ACTIONS}
    action_labels = list(action_counts.keys())
    action_values = list(action_counts.values())
    action_colors = ['gray', 'red', 'blue']

    wedges, texts, autotexts = ax.pie(action_values, labels=action_labels, colors=action_colors,
                                       autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title(f'{dataset_name.upper()} - Action Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 价格状态转移概率热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mdp.price_transition_counts, cmap='YlOrRd', aspect='auto')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', fontsize=11)

    # 设置刻度
    ax.set_xticks(np.arange(len(PRICE_STATE_LABELS)))
    ax.set_yticks(np.arange(len(PRICE_STATE_LABELS)))
    ax.set_xticklabels(PRICE_STATE_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(PRICE_STATE_LABELS)

    # 添加数值标注
    for i in range(len(PRICE_STATE_LABELS)):
        for j in range(len(PRICE_STATE_LABELS)):
            text = ax.text(j, i, f'{mdp.price_transition_counts[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f'{dataset_name.upper()} - Price State Transition Probability Matrix',
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Next State', fontsize=12)
    ax.set_ylabel('Current State', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/transition_probability_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 最优策略可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for pos_idx, pos_name in enumerate(mdp.position_states):
        ax = axes[pos_idx]
        policy_matrix = np.zeros((mdp.n_price_states, 1))

        for price_state in range(mdp.n_price_states):
            state_idx = mdp.state_to_index(price_state, pos_name)
            action_idx = policy[state_idx]
            policy_matrix[price_state, 0] = action_idx

        # 使用 imshow 代替 seaborn heatmap
        im = ax.imshow(policy_matrix, cmap='tab10', aspect='auto', vmin=0, vmax=len(ACTIONS)-1)

        # 设置刻度
        ax.set_yticks(np.arange(len(PRICE_STATE_LABELS)))
        ax.set_yticklabels(PRICE_STATE_LABELS)
        ax.set_xticks([0])
        ax.set_xticklabels([pos_name])

        # 添加数值标注
        for i in range(mdp.n_price_states):
            text = ax.text(0, i, f'{int(policy_matrix[i, 0])}',
                          ha="center", va="center", color="white", fontsize=12, fontweight='bold')

        ax.set_title(f'Position: {pos_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Price State', fontsize=11)

    # 添加动作图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i/len(ACTIONS)), label=action)
                      for i, action in enumerate(ACTIONS)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(ACTIONS), fontsize=11)

    plt.suptitle(f'{dataset_name.upper()} - Optimal Short Strategy Policy',
                fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimal_policy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 价格状态分布
    returns = df['close'].pct_change().dropna()
    price_states_dist = [mdp.discretize_price_change(r) for r in returns]

    fig, ax = plt.subplots(figsize=(12, 6))
    state_counts = [price_states_dist.count(i) for i in range(mdp.n_price_states)]
    ax.bar(range(mdp.n_price_states), state_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(mdp.n_price_states))
    ax.set_xticklabels(PRICE_STATE_LABELS, rotation=45, ha='right')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{dataset_name.upper()} - Price State Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加百分比
    total = sum(state_counts)
    for i, count in enumerate(state_counts):
        ax.text(i, count, f'{count/total*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_state_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 生成报告
    # ========================================================================

    with open(f"{output_dir}/short_strategy_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} 做空策略马尔科夫分析报告\n")
        f.write("="*80 + "\n\n")

        f.write("📊 配置信息:\n")
        f.write(f"  数据长度: {len(df)} 行\n")
        f.write(f"  价格状态数: {mdp.n_price_states}\n")
        f.write(f"  单边手续费: {COMMISSION_FEE*100:.2f}% (万分之{COMMISSION_FEE*10000:.0f})\n")
        f.write(f"  双边手续费: {2*COMMISSION_FEE*100:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("市场基本信息\n")
        f.write("="*80 + "\n\n")

        f.write("💰 价格走势:\n")
        f.write(f"  起始价格: ${start_price:.2f}\n")
        f.write(f"  结束价格: ${end_price:.2f}\n")
        f.write(f"  Hold收益: {hold_return:.2f}%\n")
        f.write(f"  Hold做空收益: {-hold_return:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("策略收益对比\n")
        f.write("="*80 + "\n\n")

        f.write("📈 各策略表现:\n\n")

        f.write("1️⃣ Hold策略 (不交易):\n")
        f.write(f"  收益: {hold_profit:.2f}%\n")
        f.write(f"  说明: 不做任何交易，保持现金\n\n")

        f.write("2️⃣ 纯做空策略 (一直持有空头):\n")
        f.write(f"  收益: {pure_short_profit:.2f}%\n")
        f.write(f"  说明: 开盘做空，收盘平仓\n")
        f.write(f"  手续费成本: {2*COMMISSION_FEE*100:.2f}%\n")
        if pure_short_profit > 0:
            f.write(f"  ✓ 纯做空策略有利润\n\n")
        else:
            f.write(f"  ✗ 纯做空策略亏损\n\n")

        f.write("3️⃣ 最优马尔科夫策略:\n")
        f.write(f"  收益: {optimal_profit:.2f}%\n")
        f.write(f"  交易次数: {num_trades}\n")
        f.write(f"  持有空头时间占比: {sum(positions)/len(positions)*100:.1f}%\n")
        if num_trades > 0:
            f.write(f"  平均每笔交易收益: {optimal_profit/num_trades:.2f}%\n")
            f.write(f"  总手续费成本: {num_trades*COMMISSION_FEE*100:.2f}%\n")
        if optimal_profit > 0:
            f.write(f"  ✓ 最优策略有利润\n\n")
        else:
            f.write(f"  ✗ 最优策略亏损\n\n")

        f.write("4️⃣ 理论最大做空机会 (捕捉所有下跌):\n")
        f.write(f"  理论最大收益: {max_short_opportunity:.2f}%\n")
        f.write(f"  说明: 完美预测每次下跌，仅在下跌时做空\n")
        f.write(f"  需要交易次数: {len(negative_returns)}\n")
        f.write(f"  手续费成本: {len(negative_returns)*2*COMMISSION_FEE*100:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("策略效率分析\n")
        f.write("="*80 + "\n\n")

        if pure_short_profit != 0:
            f.write(f"📊 最优策略 vs 纯做空:\n")
            f.write(f"  收益提升: {optimal_profit - pure_short_profit:.2f}%\n")
            if abs(pure_short_profit) > 0.01:
                f.write(f"  提升倍数: {optimal_profit / pure_short_profit:.2f}x\n\n")

        if max_short_opportunity > 0:
            f.write(f"📊 最优策略 vs 理论最大:\n")
            f.write(f"  实现率: {optimal_profit / max_short_opportunity * 100:.2f}%\n")
            f.write(f"  说明: 最优策略实现了理论最大收益的 {optimal_profit / max_short_opportunity * 100:.1f}%\n\n")

        f.write("="*80 + "\n")
        f.write("价格状态分析\n")
        f.write("="*80 + "\n\n")

        f.write("📉 价格状态分布:\n")
        state_counts = [price_states_dist.count(i) for i in range(mdp.n_price_states)]
        total_states = sum(state_counts)
        for i, (label, count) in enumerate(zip(PRICE_STATE_LABELS, state_counts)):
            pct = count / total_states * 100
            f.write(f"  {label}: {count} ({pct:.2f}%)\n")
        f.write("\n")

        # 统计最常见的状态转移
        f.write("📊 主要状态转移 (Top 5):\n")
        transitions = []
        for i in range(mdp.n_price_states):
            for j in range(mdp.n_price_states):
                if mdp.price_transition_counts[i, j] > 0:
                    transitions.append((i, j, mdp.price_transition_counts[i, j]))
        transitions.sort(key=lambda x: x[2], reverse=True)
        for i, j, prob in transitions[:5]:
            f.write(f"  {PRICE_STATE_LABELS[i]} -> {PRICE_STATE_LABELS[j]}: {prob:.3f}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("关键结论\n")
        f.write("="*80 + "\n\n")

        f.write("🎯 做空策略可行性分析:\n\n")

        if optimal_profit > 1:
            f.write(f"  ✓✓✓ 做空策略高度可行\n")
            f.write(f"      最优策略收益 {optimal_profit:.2f}% > 1%\n")
            f.write(f"      有明确的利润空间\n\n")
        elif optimal_profit > 0:
            f.write(f"  ✓ 做空策略勉强可行\n")
            f.write(f"      最优策略收益 {optimal_profit:.2f}%，利润空间较小\n")
            f.write(f"      需要精确执行以覆盖手续费\n\n")
        elif optimal_profit > -0.5:
            f.write(f"  ≈ 做空策略几乎不可行\n")
            f.write(f"      最优策略亏损 {abs(optimal_profit):.2f}%，接近盈亏平衡\n")
            f.write(f"      手续费基本吃掉所有利润\n\n")
        else:
            f.write(f"  ✗✗✗ 做空策略不可行\n")
            f.write(f"      最优策略亏损 {abs(optimal_profit):.2f}%\n")
            f.write(f"      即使完美执行也会亏损\n\n")

        if pure_short_profit > optimal_profit:
            f.write(f"  💡 特别提示:\n")
            f.write(f"      纯做空策略 ({pure_short_profit:.2f}%) 优于最优策略 ({optimal_profit:.2f}%)\n")
            f.write(f"      说明：简单持有空头比频繁交易更好\n")
            f.write(f"      原因：减少交易次数降低手续费成本\n\n")

        if hold_return < 0 and optimal_profit < -hold_return:
            f.write(f"  🔍 市场特征分析:\n")
            f.write(f"      市场下跌 {abs(hold_return):.2f}%，理论上做空有利\n")
            f.write(f"      但最优策略仅获利 {optimal_profit:.2f}%\n")
            f.write(f"      说明：价格波动大，回调频繁，做空难度高\n\n")

        f.write("📝 策略建议:\n\n")

        if optimal_profit > 1:
            f.write(f"  1. 使用马尔科夫最优策略进行做空\n")
            f.write(f"  2. 关注价格状态转移，在合适时机开仓/平仓\n")
            f.write(f"  3. 控制交易频率，避免过度交易\n")
        elif optimal_profit > 0:
            f.write(f"  1. 做空策略可行，但需谨慎执行\n")
            f.write(f"  2. 考虑减少交易频率以降低手续费成本\n")
            f.write(f"  3. 设置严格的入场条件，提高交易质量\n")
        elif pure_short_profit > 0:
            f.write(f"  1. 考虑简单的纯做空策略\n")
            f.write(f"  2. 减少交易次数，降低手续费影响\n")
            f.write(f"  3. 关注整体趋势，避免频繁进出\n")
        else:
            f.write(f"  1. 该市场环境不适合做空策略\n")
            f.write(f"  2. 考虑观望或使用其他策略\n")
            f.write(f"  3. 如果必须交易，严格控制仓位和止损\n")

    print(f"  ✓ {dataset_name.upper()} 分析完成")
    print(f"  📁 结果保存在: {output_dir}/")

    return {
        'hold_return': hold_return,
        'pure_short_profit': pure_short_profit,
        'optimal_profit': optimal_profit,
        'max_short_opportunity': max_short_opportunity,
        'num_trades': num_trades,
        'short_time_pct': sum(positions)/len(positions)*100
    }


# 主程序
print(f"\n加载数据...")

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
results = {}
for dataset_name, df in datasets:
    results[dataset_name] = analyze_dataset_short_strategy(df, dataset_name)

# 生成对比总结
print(f"\n{'='*80}")
print("生成对比总结")
print(f"{'='*80}")

if len(results) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    dataset_names = list(results.keys())

    # 1. 收益对比
    ax = axes[0, 0]
    x = np.arange(len(dataset_names))
    width = 0.2

    hold_returns = [results[d]['hold_return'] for d in dataset_names]
    pure_shorts = [results[d]['pure_short_profit'] for d in dataset_names]
    optimal_profits = [results[d]['optimal_profit'] for d in dataset_names]

    ax.bar(x - width, hold_returns, width, label='Hold Return', color='gray', alpha=0.7)
    ax.bar(x, pure_shorts, width, label='Pure Short', color='orange', alpha=0.7)
    ax.bar(x + width, optimal_profits, width, label='Optimal Markov', color='green', alpha=0.7)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Return (%)')
    ax.set_title('Strategy Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in dataset_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 交易频率
    ax = axes[0, 1]
    num_trades = [results[d]['num_trades'] for d in dataset_names]
    colors_freq = ['steelblue', 'coral']

    bars = ax.bar(dataset_names, num_trades, color=colors_freq, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trading Frequency', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, trades in zip(bars, num_trades):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{int(trades)}', ha='center', va='bottom', fontweight='bold')

    # 3. 持仓时间占比
    ax = axes[1, 0]
    short_time_pcts = [results[d]['short_time_pct'] for d in dataset_names]

    bars = ax.bar(dataset_names, short_time_pcts, color=['red', 'darkred'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Short Position Time (%)')
    ax.set_title('Short Position Holding Time', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, pct in zip(bars, short_time_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. 策略效率（最优/理论最大）
    ax = axes[1, 1]
    efficiencies = []
    for d in dataset_names:
        max_opp = results[d]['max_short_opportunity']
        opt_profit = results[d]['optimal_profit']
        if max_opp > 0:
            eff = (opt_profit / max_opp) * 100
        else:
            eff = 0
        efficiencies.append(eff)

    colors_eff = ['green' if e > 50 else 'orange' if e > 20 else 'red' for e in efficiencies]
    bars = ax.bar(dataset_names, efficiencies, color=colors_eff, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Strategy Efficiency (Optimal/Theoretical Max)', fontweight='bold', fontsize=14)
    ax.axhline(y=50, color='green', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/short_strategy_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

# 生成总结报告
with open(f"{OUTPUT_ROOT}/short_strategy_summary.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("做空策略马尔科夫分析总结报告\n")
    f.write("="*80 + "\n\n")

    f.write(f"📊 配置:\n")
    f.write(f"  费率: {COMMISSION_FEE*100:.2f}% (万分之{COMMISSION_FEE*10000:.0f})\n")
    f.write(f"  双边手续费: {2*COMMISSION_FEE*100:.2f}%\n\n")

    for dataset_name, result in results.items():
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} 数据集\n")
        f.write("="*80 + "\n\n")

        f.write("📈 策略表现:\n")
        f.write(f"  Hold收益: {result['hold_return']:.2f}%\n")
        f.write(f"  纯做空收益: {result['pure_short_profit']:.2f}%\n")
        f.write(f"  最优马尔科夫收益: {result['optimal_profit']:.2f}%\n")
        f.write(f"  理论最大做空机会: {result['max_short_opportunity']:.2f}%\n\n")

        f.write("📊 交易统计:\n")
        f.write(f"  交易次数: {result['num_trades']}\n")
        f.write(f"  持仓时间占比: {result['short_time_pct']:.1f}%\n\n")

        f.write("🎯 结论:\n")
        if result['optimal_profit'] > 1:
            f.write(f"  ✓✓✓ 做空策略高度可行，有明显利润空间\n\n")
        elif result['optimal_profit'] > 0:
            f.write(f"  ✓ 做空策略可行，但利润空间有限\n\n")
        else:
            f.write(f"  ✗ 做空策略不可行，会造成亏损\n\n")

    f.write("="*80 + "\n")
    f.write("总体建议\n")
    f.write("="*80 + "\n\n")

    val_profit = results.get('val', {}).get('optimal_profit', 0)
    test_profit = results.get('test', {}).get('optimal_profit', 0)

    f.write("📝 基于分析的策略建议:\n\n")

    if val_profit > 0 and test_profit > 0:
        f.write("  ✓ 做空策略在验证集和测试集上都有利润\n")
        f.write("  ✓ 策略具有较好的泛化性\n")
        f.write("  ✓ 建议采用马尔科夫最优策略进行做空\n\n")
    elif val_profit > 0 and test_profit <= 0:
        f.write("  ⚠️ 做空策略在验证集有利润，但测试集亏损\n")
        f.write("  ⚠️ 策略泛化性较差\n")
        f.write("  ⚠️ 需要考虑市场环境变化，谨慎使用\n\n")
    elif val_profit <= 0 and test_profit > 0:
        f.write("  ⚠️ 做空策略在测试集有利润，但验证集亏损\n")
        f.write("  ⚠️ 可能存在数据分布差异\n")
        f.write("  ⚠️ 需要进一步验证策略稳定性\n\n")
    else:
        f.write("  ✗ 做空策略在验证集和测试集上都亏损\n")
        f.write("  ✗ 不建议使用做空策略\n")
        f.write("  ✗ 考虑其他策略或观望\n\n")

    f.write("💡 关键启示:\n")
    f.write("  1. 手续费对高频做空策略影响巨大\n")
    f.write("  2. 减少交易频率是控制成本的关键\n")
    f.write("  3. 需要精确识别高质量的做空机会\n")
    f.write("  4. 市场环境变化对策略表现影响显著\n")

print(f"\n{'='*80}")
print("✓ 全部分析完成!")
print(f"{'='*80}")
print(f"\n📁 结果目录: {OUTPUT_ROOT}/")
print(f"  ├── val_short_strategy_analysis/")
print(f"  │   ├── strategy_comparison.png")
print(f"  │   ├── transition_probability_heatmap.png")
print(f"  │   ├── optimal_policy.png")
print(f"  │   ├── price_state_distribution.png")
print(f"  │   └── short_strategy_analysis_report.txt")
print(f"  ├── test_short_strategy_analysis/")
print(f"  │   └── ... (同上)")
print(f"  ├── short_strategy_comparison_summary.png")
print(f"  └── short_strategy_summary.txt")
