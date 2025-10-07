from collections import deque
import pandas as pd
import numpy as np
def _generate_trade_profit_loss_records(trade_records):
    """
    生成交易盈亏记录
    
    根据买卖记录，使用最近的卖出记录抵消买入记录来生成交易盈亏记录。
    允许1笔买入用多笔卖出抵消。
    
    返回:
        DataFrame: 包含交易盈亏记录的数据框，列包括：
            - datetime: 交易时间
            - buy_id: 买入交易ID
            - sell_id: 卖出交易ID
            - buy_price: 买入价格
            - sell_price: 卖出价格
            - amount: 交易金额
            - profit_loss: 盈亏金额
            - quantity: 交易数量
    """
    # 分离买入和卖出记录
    buy_records = [record for record in trade_records if record['type'] == 'buy']
    sell_records = [record for record in trade_records if record['type'] == 'sell']
    
    # 按时间排序
    buy_records.sort(key=lambda x: x['datetime'])
    sell_records.sort(key=lambda x: x['datetime'])
    
    profit_loss_records = []
    
    # 使用队列处理买入记录
    
    buy_queue = deque(buy_records)
    
    # 处理每笔卖出记录
    for sell_record in sell_records:
        remaining_sell_quantity = sell_record['quantity']
        
        while remaining_sell_quantity > 0 and buy_queue:
            buy_record = buy_queue[0]
            
            # 计算可抵消的数量
            if buy_record['quantity'] <= remaining_sell_quantity:
                # 整笔买入记录被完全抵消
                matched_quantity = buy_record['quantity']
                buy_queue.popleft()  # 移除已完全抵消的买入记录
            else:
                # 部分抵消买入记录
                matched_quantity = remaining_sell_quantity
                # 更新买入记录的剩余数量
                buy_record['quantity'] -= matched_quantity
                buy_record['amount'] = buy_record['quantity'] * buy_record['price'] * (1 + buy_record['commission_fee'] / (buy_record['quantity'] * buy_record['price']))
            
            # 计算盈亏
            buy_price = buy_record['price']
            sell_price = sell_record['price']
            profit_loss = (sell_price - buy_price) * matched_quantity - buy_record['commission_fee'] * (matched_quantity / buy_record['quantity']) - sell_record['commission_fee'] * (matched_quantity / sell_record['quantity'])
            
            # 计算交易金额
            amount = sell_price * matched_quantity
            
            # 添加到盈亏记录
            profit_loss_records.append({
                'datetime': sell_record['datetime'],
                'buy_id': buy_record['id'],
                'sell_id': sell_record['id'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'amount': amount,
                'profit_loss': profit_loss,
                'quantity': matched_quantity
            })
            
            # 更新剩余卖出数量
            remaining_sell_quantity -= matched_quantity
    
    # 转换为DataFrame并返回
    return pd.DataFrame(profit_loss_records)

def profit_loss_statistics(trade_records):
    """
    生成日/月维度交易统计
    
    根据交易盈亏记录，生成日维度和月维度的交易统计信息。
    
    返回:
        tuple: (daily_stats, monthly_stats)
            - daily_stats: 日维度交易统计DataFrame
            - monthly_stats: 月维度交易统计DataFrame
            每个DataFrame包含以下列：
            - date/month: 日期/月份
            - trade_count: 交易次数
            - total_amount: 交易金额
            - total_quantity: 交易数量
            - profit_count: 盈利次数
            - profit_amount: 盈利金额
            - loss_amount: 亏损金额
    """
    # 获取交易盈亏记录
    profit_loss_records = generate_trade_profit_loss_records(trade_records)
    
    if profit_loss_records.empty:
        # 如果没有交易记录，返回空的DataFrame
        daily_columns = ['date', 'trade_count', 'total_amount', 'total_quantity', 'profit_count', 'profit_amount', 'loss_amount']
        monthly_columns = ['month', 'trade_count', 'total_amount', 'total_quantity', 'profit_count', 'profit_amount', 'loss_amount']
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=monthly_columns)
    
    # 转换datetime列为datetime类型
    profit_loss_records['datetime'] = pd.to_datetime(profit_loss_records['datetime'])
    
    # 提取日期和月份
    profit_loss_records['date'] = profit_loss_records['datetime'].dt.date
    profit_loss_records['month'] = profit_loss_records['datetime'].dt.to_period('M')
    
    # 计算每笔交易的盈亏状态
    profit_loss_records['is_profit'] = profit_loss_records['profit_loss'] > 0
    
    # 日维度统计
    daily_stats = profit_loss_records.groupby('date').agg(
        trade_count=('profit_loss', 'count'),
        total_amount=('amount', 'sum'),
        total_quantity=('quantity', 'sum'),
        profit_count=('is_profit', 'sum'),
        profit_amount=('profit_loss', lambda x: x[x > 0].sum()),
        loss_amount=('profit_loss', lambda x: abs(x[x < 0].sum()))
    ).reset_index()
    
    # 确保盈利金额和亏损金额在没有盈利或亏损时为0
    daily_stats['profit_amount'] = daily_stats['profit_amount'].fillna(0)
    daily_stats['loss_amount'] = daily_stats['loss_amount'].fillna(0)
    
    # 月维度统计
    monthly_stats = profit_loss_records.groupby('month').agg(
        trade_count=('profit_loss', 'count'),
        total_amount=('amount', 'sum'),
        total_quantity=('quantity', 'sum'),
        profit_count=('is_profit', 'sum'),
        profit_amount=('profit_loss', lambda x: x[x > 0].sum()),
        loss_amount=('profit_loss', lambda x: abs(x[x < 0].sum()))
    ).reset_index()
    
    # 确保盈利金额和亏损金额在没有盈利或亏损时为0
    monthly_stats['profit_amount'] = monthly_stats['profit_amount'].fillna(0)
    monthly_stats['loss_amount'] = monthly_stats['loss_amount'].fillna(0)
    
    # 将月份转换为字符串格式
    monthly_stats['month'] = monthly_stats['month'].astype(str)
    
    return daily_stats, monthly_stats

def calculate_trading_metrics(trade_records):
    """
    计算交易指标
    
    基于交易记录计算以下指标：
    - 总交易金额
    - 年化波动率 = 每日收益率标准差
    - 交易胜率 = 盈利次数/总交易次数
    - 盈亏比 = 平均每次盈利金额/平均每次亏损金额
    - 总交易次数
    - 交易频率 = 总交易次数/交易天数
    
    返回:
        dict: 包含计算指标的字典
            - total_amount: 总交易金额
            - annualized_volatility: 年化波动率
            - win_rate: 交易胜率
            - profit_loss_ratio: 盈亏比
            - total_trades: 总交易次数
            - trade_frequency: 交易频率
    """
    # 获取交易盈亏记录
    profit_loss_records = _generate_trade_profit_loss_records(trade_records)
    
    if profit_loss_records.empty:
        # 如果没有交易记录，返回默认值
        return {
            'total_amount': 0,
            'annualized_volatility': 0,
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'total_trades': 0,
            'trade_frequency': 0
        }
    
    # 转换datetime列为datetime类型
    profit_loss_records['datetime'] = pd.to_datetime(profit_loss_records['datetime'])
    
    # 提取日期
    profit_loss_records['date'] = profit_loss_records['datetime'].dt.date
    
    # 计算总交易金额
    total_amount = profit_loss_records['amount'].sum()
    
    # 计算每日收益率
    daily_returns = profit_loss_records.groupby('date').agg(
        daily_profit_loss=('profit_loss', 'sum'),
        daily_amount=('amount', 'sum')
    ).reset_index()
    
    # 计算每日收益率 = 每日盈亏 / 每日交易金额
    daily_returns['daily_return'] = daily_returns['daily_profit_loss'] / daily_returns['daily_amount']
    
    # 计算年化波动率 = 每日收益率标准差
    daily_std = daily_returns['daily_return'].std()
    annualized_volatility = daily_std if not np.isnan(daily_std) else 0
    
    # 计算交易胜率 = 盈利次数 / 总交易次数
    profit_loss_records['is_profit'] = profit_loss_records['profit_loss'] > 0
    total_trades = len(profit_loss_records)
    profit_trades = profit_loss_records['is_profit'].sum()
    win_rate = profit_trades / total_trades if total_trades > 0 else 0
    
    # 计算盈亏比 = 平均每次盈利金额 / 平均每次亏损金额
    profit_amounts = profit_loss_records[profit_loss_records['profit_loss'] > 0]['profit_loss']
    loss_amounts = profit_loss_records[profit_loss_records['profit_loss'] < 0]['profit_loss'].abs()
    
    avg_profit = profit_amounts.mean() if len(profit_amounts) > 0 else 0
    avg_loss = loss_amounts.mean() if len(loss_amounts) > 0 else 0
    
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
    
    # 计算交易天数
    trading_days = profit_loss_records['date'].nunique()
    
    # 计算交易频率 = 总交易次数 / 交易天数
    trade_frequency = total_trades / trading_days if trading_days > 0 else 0
    
    return {
        'total_amount': total_amount,
        'annualized_volatility': annualized_volatility,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': total_trades,
        'trade_frequency': trade_frequency
    }