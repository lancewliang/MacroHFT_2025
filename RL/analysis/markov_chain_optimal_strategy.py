#!/usr/bin/env python3
"""
马尔科夫链最优交易策略分析
使用马尔科夫决策过程(MDP)寻找训练集中的最大交易利润和最优策略
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
import seaborn as sns

# 配置
DATASET = "ETHUSDT"
LABELS = [1, 2, 3]
DATA_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/data"
OUTPUT_ROOT = "/home/lanceliang/opt/aiwork/MacroHFT/analysis"
COMMISSION_FEE = 0.0005  # 单边手续费 0.05%

# 马尔科夫状态配置
PRICE_CHANGE_BINS = [-np.inf, -0.001, -0.0005, 0, 0.0005, 0.001, np.inf]  # 价格变化区间
PRICE_STATE_LABELS = ['大跌', '中跌', '小跌', '小涨', '中涨', '大涨']

# 动作空间
ACTIONS = ['hold', 'long', 'short', 'close']  # 持有、做多、做空、平仓
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("马尔科夫链最优交易策略分析")
print("="*80)

# 加载标签索引
with open(f"{DATA_ROOT}/{DATASET}/train/slope_labels.pkl", 'rb') as f:
    train_index = pickle.load(f)
with open(f"{DATA_ROOT}/{DATASET}/val/slope_labels.pkl", 'rb') as f:
    val_index = pickle.load(f)


class MarkovTradingMDP:
    """马尔科夫决策过程用于交易策略优化"""

    def __init__(self, price_change_bins, commission_fee=0.0005):
        self.price_change_bins = price_change_bins
        self.n_price_states = len(price_change_bins) - 1
        self.commission_fee = commission_fee

        # 状态空间：(price_state, position_state)
        # position_state: 0=空仓, 1=多头, 2=空头
        self.position_states = ['empty', 'long', 'short']
        self.n_states = self.n_price_states * len(self.position_states)

        # 动作空间
        self.actions = ACTIONS
        self.n_actions = len(self.actions)

        # 转移概率矩阵: [state, action, next_state]
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states))

        # 奖励矩阵: [state, action]
        self.rewards = np.zeros((self.n_states, self.n_actions))

        # 统计数据
        self.price_state_counts = np.zeros(self.n_price_states)
        self.price_transition_counts = np.zeros((self.n_price_states, self.n_price_states))

    def discretize_price_change(self, price_change):
        """将连续的价格变化离散化为状态"""
        return np.digitize(price_change, self.price_change_bins) - 1

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
        """从数据中学习转移概率和奖励"""
        # 计算价格变化
        returns = df['close'].pct_change().dropna()

        # 离散化价格状态
        price_states = [self.discretize_price_change(r) for r in returns]

        # 统计价格状态转移
        for i in range(len(price_states) - 1):
            current_state = price_states[i]
            next_state = price_states[i + 1]
            self.price_state_counts[current_state] += 1
            self.price_transition_counts[current_state, next_state] += 1

        # 计算转移概率（归一化）
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
                        if action == 'long':
                            next_position = 'long'
                        elif action == 'short':
                            next_position = 'short'
                        else:  # hold or close
                            next_position = 'empty'
                    elif position == 'long':
                        if action == 'close':
                            next_position = 'empty'
                        else:
                            next_position = 'long'
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
        # 价格变化的期望值（每个价格状态的中点）
        price_change_midpoints = []
        for i in range(len(self.price_change_bins) - 1):
            left = self.price_change_bins[i]
            right = self.price_change_bins[i + 1]
            if np.isinf(left):
                left = -0.002
            if np.isinf(right):
                right = 0.002
            midpoint = (left + right) / 2
            price_change_midpoints.append(midpoint)

        for state_idx in range(self.n_states):
            price_state, position = self.index_to_state(state_idx)
            price_change = price_change_midpoints[price_state]

            for action in self.actions:
                action_idx = ACTION_TO_IDX[action]
                reward = 0

                if position == 'empty':
                    if action == 'long':
                        # 开多仓：支付手续费
                        reward = -self.commission_fee
                    elif action == 'short':
                        # 开空仓：支付手续费
                        reward = -self.commission_fee
                    else:
                        # 空仓不交易
                        reward = 0

                elif position == 'long':
                    if action == 'close':
                        # 平多仓：获得价格上涨收益，支付手续费
                        reward = price_change - self.commission_fee
                    else:
                        # 持有多仓：获得价格上涨收益
                        reward = price_change

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
                # 计算每个动作的Q值
                Q_values = np.zeros(self.n_actions)

                for action in range(self.n_actions):
                    immediate_reward = self.rewards[state, action]
                    expected_future_value = np.sum(
                        self.transition_probs[state, action, :] * V_old
                    )
                    Q_values[action] = immediate_reward + gamma * expected_future_value

                # 选择最优动作
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

        for i, (price_state, price_return) in enumerate(zip(price_states, returns)):
            state_idx = self.state_to_index(price_state, position)
            action_idx = policy[state_idx]
            action = self.actions[action_idx]
            actions_taken.append(action)

            # 执行动作并计算收益
            profit = 0

            if position == 'empty':
                if action == 'long':
                    position = 'long'
                    profit = -self.commission_fee
                elif action == 'short':
                    position = 'short'
                    profit = -self.commission_fee

            elif position == 'long':
                if action == 'close':
                    profit = price_return - self.commission_fee
                    position = 'empty'
                else:
                    profit = price_return

            elif position == 'short':
                if action == 'close':
                    profit = -price_return - self.commission_fee
                    position = 'empty'
                else:
                    profit = -price_return

            cumulative_profit += profit
            profits.append(cumulative_profit)

        return cumulative_profit, profits, actions_taken


def analyze_label_with_markov(label_id):
    """使用马尔科夫链分析单个label"""

    print(f"\n{'='*80}")
    print(f"分析 Label {label_id}")
    print(f"{'='*80}")

    label_dir = f"{OUTPUT_ROOT}/label{label_id}_markov_analysis"
    os.makedirs(label_dir, exist_ok=True)

    train_files = train_index[label_id]
    val_files = val_index[label_id]

    # 创建MDP模型
    mdp = MarkovTradingMDP(PRICE_CHANGE_BINS, COMMISSION_FEE)

    print(f"  [1/5] 从训练集学习转移概率...")
    # 从所有训练集文件学习转移概率
    for file_id in train_files:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/train/df_{file_id}.feather")
            mdp.learn_from_data(df)
        except Exception as e:
            print(f"    警告: 训练集文件{file_id}处理失败: {e}")

    # 归一化转移概率
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            prob_sum = np.sum(mdp.transition_probs[state, action, :])
            if prob_sum > 0:
                mdp.transition_probs[state, action, :] /= prob_sum

    print(f"  [2/5] 计算奖励函数...")
    mdp.calculate_rewards()

    print(f"  [3/5] 使用值迭代求解最优策略...")
    V, policy = mdp.value_iteration()

    print(f"  [4/5] 在训练集上模拟最优策略...")
    train_results = []
    for file_id in train_files[:30]:  # 限制文件数量以节省时间
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/train/df_{file_id}.feather")
            total_profit, profits, actions = mdp.simulate_trading(df, policy)

            # 计算统计信息
            action_counts = {a: actions.count(a) for a in ACTIONS}

            train_results.append({
                'file_id': file_id,
                'total_profit': total_profit * 100,  # 转换为百分比
                'hold_return': (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100,
                'num_trades': sum(1 for a in actions if a in ['long', 'short', 'close']),
                'action_counts': action_counts
            })
        except Exception as e:
            print(f"    警告: 文件{file_id}模拟失败: {e}")

    print(f"  [5/5] 在验证集上模拟最优策略...")
    val_results = []
    for file_id in val_files:
        try:
            df = pd.read_feather(f"{DATA_ROOT}/{DATASET}/val/df_{file_id}.feather")
            total_profit, profits, actions = mdp.simulate_trading(df, policy)

            action_counts = {a: actions.count(a) for a in ACTIONS}

            val_results.append({
                'file_id': file_id,
                'total_profit': total_profit * 100,
                'hold_return': (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100,
                'num_trades': sum(1 for a in actions if a in ['long', 'short', 'close']),
                'action_counts': action_counts
            })
        except Exception as e:
            print(f"    警告: 文件{file_id}模拟失败: {e}")

    # 生成可视化
    print(f"  [6/6] 生成可视化和报告...")

    train_df = pd.DataFrame(train_results)
    val_df = pd.DataFrame(val_results)

    # 1. 收益对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    x = np.arange(len(train_df))
    width = 0.35
    ax.bar(x - width/2, train_df['hold_return'], width, label='Hold Return', alpha=0.7, color='gray')
    ax.bar(x + width/2, train_df['total_profit'], width, label='Markov Optimal', alpha=0.7, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 训练集: Hold vs Markov最优策略', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(val_df))
    ax.bar(x - width/2, val_df['hold_return'], width, label='Hold Return', alpha=0.7, color='gray')
    ax.bar(x + width/2, val_df['total_profit'], width, label='Markov Optimal', alpha=0.7, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Label {label_id} - 验证集: Hold vs Markov最优策略', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{label_dir}/markov_vs_hold.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 价格状态转移概率热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mdp.price_transition_counts, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=PRICE_STATE_LABELS, yticklabels=PRICE_STATE_LABELS, ax=ax)
    ax.set_title(f'Label {label_id} - 价格状态转移概率矩阵', fontweight='bold', fontsize=14)
    ax.set_xlabel('下一状态')
    ax.set_ylabel('当前状态')
    plt.tight_layout()
    plt.savefig(f"{label_dir}/transition_probability_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 最优策略可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for pos_idx, pos_name in enumerate(mdp.position_states):
        ax = axes[pos_idx]
        policy_matrix = np.zeros((mdp.n_price_states, 1))

        for price_state in range(mdp.n_price_states):
            state_idx = mdp.state_to_index(price_state, pos_name)
            action_idx = policy[state_idx]
            policy_matrix[price_state, 0] = action_idx

        sns.heatmap(policy_matrix, annot=True, fmt='.0f', cmap='tab10',
                   yticklabels=PRICE_STATE_LABELS, xticklabels=[pos_name],
                   cbar_kws={'ticks': range(len(ACTIONS))}, ax=ax, vmin=0, vmax=len(ACTIONS)-1)
        ax.set_title(f'持仓: {pos_name}', fontweight='bold')
        ax.set_ylabel('价格状态')

    # 添加动作图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i/len(ACTIONS)), label=action)
                      for i, action in enumerate(ACTIONS)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(ACTIONS), fontsize=12)

    plt.suptitle(f'Label {label_id} - 最优策略 (动作选择)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{label_dir}/optimal_policy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 生成统计报告
    with open(f"{label_dir}/markov_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Label {label_id} 马尔科夫链最优策略分析报告\n")
        f.write("="*80 + "\n\n")

        f.write("📊 模型配置:\n")
        f.write(f"  价格状态数: {mdp.n_price_states}\n")
        f.write(f"  持仓状态数: {len(mdp.position_states)}\n")
        f.write(f"  总状态数: {mdp.n_states}\n")
        f.write(f"  动作数: {mdp.n_actions}\n")
        f.write(f"  单边手续费: {COMMISSION_FEE*100:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("训练集结果\n")
        f.write("="*80 + "\n\n")

        f.write("📈 收益对比:\n")
        f.write(f"  平均Hold收益: {train_df['hold_return'].mean():.2f}%\n")
        f.write(f"  平均Markov最优收益: {train_df['total_profit'].mean():.2f}%\n")
        f.write(f"  收益提升: {train_df['total_profit'].mean() - train_df['hold_return'].mean():.2f}%\n")
        f.write(f"  提升倍数: {train_df['total_profit'].mean() / train_df['hold_return'].mean():.2f}x\n\n"
                if abs(train_df['hold_return'].mean()) > 0.1 else "\n")

        f.write("📊 交易统计:\n")
        f.write(f"  平均交易次数: {train_df['num_trades'].mean():.1f}\n")
        f.write(f"  最大交易次数: {train_df['num_trades'].max()}\n")
        f.write(f"  最小交易次数: {train_df['num_trades'].min()}\n\n")

        f.write("="*80 + "\n")
        f.write("验证集结果\n")
        f.write("="*80 + "\n\n")

        f.write("📈 收益对比:\n")
        f.write(f"  平均Hold收益: {val_df['hold_return'].mean():.2f}%\n")
        f.write(f"  平均Markov最优收益: {val_df['total_profit'].mean():.2f}%\n")
        f.write(f"  收益差异: {val_df['total_profit'].mean() - val_df['hold_return'].mean():.2f}%\n\n")

        f.write("📊 交易统计:\n")
        f.write(f"  平均交易次数: {val_df['num_trades'].mean():.1f}\n")
        f.write(f"  最大交易次数: {val_df['num_trades'].max()}\n")
        f.write(f"  最小交易次数: {val_df['num_trades'].min()}\n\n")

        f.write("="*80 + "\n")
        f.write("关键发现\n")
        f.write("="*80 + "\n\n")

        train_improvement = train_df['total_profit'].mean() - train_df['hold_return'].mean()
        val_improvement = val_df['total_profit'].mean() - val_df['hold_return'].mean()

        f.write("🎯 策略效果:\n")
        if train_improvement > 1:
            f.write(f"  ✓ 训练集上，最优策略显著优于Hold (提升{train_improvement:.2f}%)\n")
        else:
            f.write(f"  ⚠️ 训练集上，最优策略提升有限 (仅{train_improvement:.2f}%)\n")

        if val_improvement > 1:
            f.write(f"  ✓ 验证集上，最优策略优于Hold (提升{val_improvement:.2f}%)\n")
        elif val_improvement > 0:
            f.write(f"  ≈ 验证集上，最优策略略优于Hold (提升{val_improvement:.2f}%)\n")
        else:
            f.write(f"  ✗ 验证集上，最优策略未能超越Hold (差{abs(val_improvement):.2f}%)\n")

        f.write(f"\n💡 理论最大可获得利润:\n")
        f.write(f"  训练集: {train_df['total_profit'].mean():.2f}%\n")
        f.write(f"  验证集: {val_df['total_profit'].mean():.2f}%\n")

        if val_df['total_profit'].mean() < 0.5:
            f.write(f"\n⚠️ 警告: 即使使用最优策略，验证集上的收益也很有限\n")
            f.write(f"   这说明该label的数据分布不适合高频交易策略\n")

    print(f"  ✓ Label {label_id} 完成")
    return train_df, val_df, mdp


# 主程序
all_results = {}

for label_id in LABELS:
    try:
        train_df, val_df, mdp = analyze_label_with_markov(label_id)
        all_results[label_id] = {'train': train_df, 'val': val_df, 'mdp': mdp}
    except Exception as e:
        print(f"\n  ✗ Label {label_id} 失败: {e}")
        import traceback
        traceback.print_exc()

# 跨Label对比
if len(all_results) > 0:
    print(f"\n{'='*80}")
    print("生成跨Label对比")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    labels_list = sorted(all_results.keys())

    # 训练集对比
    hold_train = [all_results[l]['train']['hold_return'].mean() for l in labels_list]
    markov_train = [all_results[l]['train']['total_profit'].mean() for l in labels_list]

    ax = axes[0, 0]
    x = np.arange(len(labels_list))
    width = 0.35
    ax.bar(x - width/2, hold_train, width, label='Hold', alpha=0.7, color='gray')
    ax.bar(x + width/2, markov_train, width, label='Markov Optimal', alpha=0.7, color='green')
    ax.set_xlabel('Label')
    ax.set_ylabel('Return (%)')
    ax.set_title('训练集: Hold vs Markov最优策略', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # 验证集对比
    hold_val = [all_results[l]['val']['hold_return'].mean() for l in labels_list]
    markov_val = [all_results[l]['val']['total_profit'].mean() for l in labels_list]

    ax = axes[0, 1]
    ax.bar(x - width/2, hold_val, width, label='Hold', alpha=0.7, color='gray')
    ax.bar(x + width/2, markov_val, width, label='Markov Optimal', alpha=0.7, color='green')
    ax.set_xlabel('Label')
    ax.set_ylabel('Return (%)')
    ax.set_title('验证集: Hold vs Markov最优策略', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # 收益提升
    improvement_train = [m - h for m, h in zip(markov_train, hold_train)]
    improvement_val = [m - h for m, h in zip(markov_val, hold_val)]

    ax = axes[1, 0]
    ax.bar(x, improvement_train, color='green', alpha=0.7)
    ax.set_xlabel('Label')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('训练集: Markov策略收益提升', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    ax = axes[1, 1]
    colors = ['green' if i > 0 else 'red' for i in improvement_val]
    ax.bar(x, improvement_val, color=colors, alpha=0.7)
    ax.set_xlabel('Label')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('验证集: Markov策略收益提升', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Label {l}' for l in labels_list])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/cross_label_markov_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n{'='*80}")
print("完成!")
print(f"{'='*80}")
print(f"\n结果目录: {OUTPUT_ROOT}/")
for label_id in LABELS:
    print(f"\nlabel{label_id}_markov_analysis/")
    print(f"  ├── markov_vs_hold.png                      (收益对比)")
    print(f"  ├── transition_probability_heatmap.png      (状态转移概率)")
    print(f"  ├── optimal_policy.png                      (最优策略)")
    print(f"  └── markov_analysis_report.txt              (分析报告)")
print(f"\ncross_label_markov_comparison.png               (跨Label对比)")
