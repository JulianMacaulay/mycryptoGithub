"""
完整的回测系统
支持从币安API或CSV文件加载数据，运行策略，生成回测报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys
import requests
import ccxt
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加策略模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.turtle_strategy import TurtleStrategy
from strategies.base_strategy import BaseStrategy


class BacktestEngine:
    """
    回测引擎
    负责执行策略回测，记录交易，计算收益
    """
    
    def __init__(self, initial_capital: float = 10000, commission_rate: float = 0.001, 
                 leverage: float = 5.0, position_ratio: float = 0.5):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（默认0.1%）
            leverage: 杠杆倍数（默认5倍）
            position_ratio: 仓位比例（默认1.0，即100%，0.5表示50%）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.leverage = leverage
        self.position_ratio = position_ratio
        
        # 计算可用资金（考虑仓位比例）
        self.available_capital = initial_capital * position_ratio
        
        # 账户状态（使用可用资金，因为只有这部分可以交易）
        self.balance = self.available_capital
        self.equity = self.available_capital
        self.position_size = 0.0  # 持仓数量（正数=多头，负数=空头）
        self.position_value = 0.0  # 持仓价值
        self.entry_price = 0.0  # 入场价格
        self.entry_idx = -1  # 入场索引
        self.entry_count = 0  # 加仓次数
        
        # 交易记录
        self.trades = []  # 所有交易记录
        self.equity_curve = []  # 权益曲线
        self.signals = []  # 所有信号记录
        
    def reset(self):
        """重置回测引擎"""
        # 重置时使用可用资金（考虑仓位比例）
        self.balance = self.available_capital
        self.equity = self.available_capital
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_count = 0
        self.trades = []
        self.equity_curve = []
        self.signals = []
    
    def open_position(self, signal: str, price: float, size: float, 
                     current_idx: int, reason: str = ""):
        """
        开仓
        
        Args:
            signal: 'long' 或 'short'
            price: 开仓价格
            size: 开仓数量（绝对值）
            current_idx: 当前索引
            reason: 开仓原因
        """
        # 计算最大可开仓位（基于可用资金和杠杆）
        max_position_value = self.balance * self.leverage  # 最大仓位价值
        max_size = max_position_value / price if price > 0 else 0  # 最大数量
        
        # 取较小值（策略计算的size和最大可开仓位）
        actual_size = min(size, max_size) if max_size > 0 else 0
        
        if actual_size <= 0:
            # 无法开仓，直接返回
            return
        
        if signal == 'long':
            self.position_size = actual_size
            self.entry_price = price
            self.entry_idx = current_idx
            self.entry_count = 1
            
            # 计算成本（考虑杠杆）
            # 开仓价值 = actual_size * price
            # 保证金 = 开仓价值 / 杠杆倍数
            # 成本 = 保证金 * (1 + 手续费率)
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'open_long',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })
            
        elif signal == 'short':
            self.position_size = -actual_size  # 负数表示空头
            self.entry_price = price  # 价格始终是正数
            self.entry_idx = current_idx
            self.entry_count = 1
            
            # 做空：也需要保证金（类似做多），扣除资金
            # 在期货交易中，做空也需要保证金，只是持仓方向相反
            # 计算成本（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'open_short',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })
    
    def add_position(self, signal: str, price: float, size: float,
                    current_idx: int, reason: str = ""):
        """
        加仓
        
        Args:
            signal: 'add_long' 或 'add_short'
            price: 加仓价格
            size: 加仓数量（绝对值）
            current_idx: 当前索引
            reason: 加仓原因
        """
        # 计算最大可加仓位（基于可用资金和杠杆）
        max_position_value = self.balance * self.leverage  # 最大仓位价值
        max_size = max_position_value / price if price > 0 else 0  # 最大数量
        
        # 取较小值（策略计算的size和最大可开仓位）
        actual_size = min(size, max_size) if max_size > 0 else 0
        
        if actual_size <= 0:
            # 无法加仓（资金不足）
            return
        
        if signal == 'add_long' and self.position_size > 0:
            # 计算新的平均入场价格
            total_size = self.position_size + actual_size
            total_cost = self.position_size * self.entry_price + actual_size * price
            self.entry_price = total_cost / total_size
            self.position_size = total_size
            self.entry_count += 1
            
            # 扣除成本（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'add_long',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })
            
        elif signal == 'add_short' and self.position_size < 0:
            # 计算新的平均入场价格（价格始终是正数）
            current_size = abs(self.position_size)
            total_size = current_size + actual_size
            total_cost = current_size * self.entry_price + actual_size * price
            self.entry_price = total_cost / total_size  # 价格始终是正数
            self.position_size = -total_size  # 负数表示空头
            self.entry_count += 1
            
            # 做空加仓：也需要扣除保证金（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'add_short',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })
    
    def close_position(self, price: float, current_idx: int, reason: str = ""):
        """
        平仓
        
        Args:
            price: 平仓价格
            current_idx: 当前索引
            reason: 平仓原因
        """
        if self.position_size == 0:
            return None
        
        # 计算盈亏
        if self.position_size > 0:  # 平多
            # 多头：买入价 entry_price，卖出价 price
            gross_pnl = (price - self.entry_price) * self.position_size
            # 平仓时：收回保证金 + 盈亏，扣除平仓手续费
            # 开仓时扣除了：margin * (1 + commission_rate) = margin + margin * commission_rate
            # 平仓时收回：margin + gross_pnl - close_cost
            size = self.position_size
            entry_margin = (self.entry_price * size) / self.leverage
            close_cost = (size * price) * self.commission_rate
            # 计算开仓手续费（开仓时已经扣除，但需要在pnl中体现，以便CSV中记录的是净盈亏）
            # 开仓手续费 = (entry_price * size / leverage) * commission_rate
            open_cost = entry_margin * self.commission_rate
            # 净盈亏 = 毛盈亏 - 平仓手续费 - 开仓手续费（用于CSV记录）
            pnl = gross_pnl - close_cost - open_cost
            # 余额计算：收回保证金 + 毛盈亏 - 平仓手续费（开仓手续费已在开仓时扣除）
            self.balance += entry_margin + gross_pnl - close_cost
        else:  # 平空
            # 空头：开仓价 entry_price（正数），平仓价 price
            # 做空时已经扣除了保证金
            # 平仓时：如果价格下跌，盈利；如果价格上涨，亏损
            # 盈亏 = (entry_price - price) * size
            size = abs(self.position_size)  # 获取数量（正数）
            entry_price = self.entry_price  # entry_price已经是正数，不需要abs
            gross_pnl = (entry_price - price) * size
            
            # 平仓时：收回保证金 + 盈亏，扣除平仓手续费
            # 开仓时扣除了：margin * (1 + commission_rate) = margin + margin * commission_rate
            # 平仓时收回：margin + gross_pnl - close_cost
            entry_margin = (entry_price * size) / self.leverage
            close_cost = (size * price) * self.commission_rate
            # 计算开仓手续费（开仓时已经扣除，但需要在pnl中体现，以便CSV中记录的是净盈亏）
            # 开仓手续费 = (entry_price * size / leverage) * commission_rate
            open_cost = entry_margin * self.commission_rate
            # 净盈亏 = 毛盈亏 - 平仓手续费 - 开仓手续费（用于CSV记录）
            pnl = gross_pnl - close_cost - open_cost
            # 余额计算：收回保证金 + 毛盈亏 - 平仓手续费（开仓手续费已在开仓时扣除）
            self.balance += entry_margin + gross_pnl - close_cost
        
        # 记录交易（保存平仓前的entry_price，确保是正数）
        trade_type = 'close_long' if self.position_size > 0 else 'close_short'
        # 保存entry_price（确保是正数）
        saved_entry_price = abs(self.entry_price) if self.entry_price < 0 else self.entry_price
        self.trades.append({
            'type': trade_type,
            'price': price,
            'size': abs(self.position_size),
            'idx': current_idx,
            'entry_price': saved_entry_price,  # 确保是正数
            'pnl': pnl,  # 记录净盈亏（已扣除平仓手续费和开仓手续费）
            'balance': self.balance,
            'equity': self.balance,  # 平仓后无持仓，权益=余额
            'reason': reason,
            'entry_count': self.entry_count
        })
        
        # 重置持仓
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_count = 0
        
        return pnl
    
    def update_equity(self, current_price: float):
        """
        更新权益（未实现盈亏）
        
        Args:
            current_price: 当前价格
        """
        if self.position_size == 0:
            self.equity = self.balance
        elif self.position_size > 0:  # 多头
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            self.equity = self.balance + unrealized_pnl
            self.position_value = self.position_size * current_price
        else:  # 空头
            # 做空：entry_price是正数（开仓价），position_size是负数
            # 未实现盈亏 = (开仓价 - 当前价) * 数量
            size = abs(self.position_size)  # 获取数量（正数）
            entry_price = self.entry_price  # entry_price已经是正数，不需要abs
            unrealized_pnl = (entry_price - current_price) * size
            self.equity = self.balance + unrealized_pnl
            self.position_value = size * current_price
        
        # 记录权益曲线：交易账户权益 + 未投入资金（与最终权益计算保持一致）
        total_equity = self.equity + (self.initial_capital - self.available_capital)
        self.equity_curve.append(total_equity)


class BacktestSystem:
    """
    完整的回测系统
    """
    
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10000, 
                 leverage: float = 5.0, position_ratio: float = 0.5):
        """
        初始化回测系统
        
        Args:
            strategy: 策略实例
            initial_capital: 初始资金
            leverage: 杠杆倍数（默认5倍）
            position_ratio: 仓位比例（默认1.0，即100%，0.5表示用50%的资金）
        """
        self.strategy = strategy
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            leverage=leverage,
            position_ratio=position_ratio
        )
        self.data = None
        
    def load_data_from_csv(self, filepath: str, symbol: str = None):
        """
        从CSV文件加载数据
        
        Args:
            filepath: CSV文件路径
            symbol: 如果CSV包含多个币种，指定要使用的币种
        """
        df = pd.read_csv(filepath)
        
        # 如果包含symbol列，筛选特定币种
        if 'symbol' in df.columns and symbol:
            # 支持多种格式：ETHUSDT, ETH/USDT, ETH-USDT
            symbol_variants = [symbol, symbol.replace('USDT', '/USDT'), symbol.replace('/', 'USDT')]
            df_filtered = df[df['symbol'].isin(symbol_variants)].copy()
            if len(df_filtered) > 0:
                df = df_filtered
                print(f"筛选币种: {symbol}，找到 {len(df)} 条记录")
            else:
                print(f"警告: 未找到币种 {symbol}，使用所有数据")
        
        # 确保时间列为datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            # 尝试将第一列作为时间索引
            df.index = pd.to_datetime(df.index)
        
        # 确保有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        # 按时间排序
        df = df.sort_index()
        
        self.data = df
        print(f"成功加载数据: {len(df)} 条记录")
        print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
        
        return df
    
    def load_data_from_binance(self, symbol: str, interval: str = '1h', 
                               limit: int = 1000, start_time: int = None, end_time: int = None):
        """
        从币安API加载数据
        
        Args:
            symbol: 交易对（如 'BTCUSDT'）
            interval: K线周期（如 '1h', '1d'）
            limit: 获取数量
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
        """
        print(f"从币安获取 {symbol} 数据...")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        all_klines = []
        end_time_param = end_time
        
        # 分页获取数据
        while len(all_klines) < limit:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    interval, 
                    since=start_time,
                    limit=min(1000, limit - len(all_klines)),
                    params={'endTime': end_time_param} if end_time_param else {}
                )
                
                if not ohlcv:
                    break
                
                all_klines.extend(ohlcv)
                
                if len(ohlcv) < 1000 or len(all_klines) >= limit:
                    break
                
                # 更新end_time用于下次请求
                end_time_param = ohlcv[0][0] - 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"获取数据失败: {e}")
                break
        
        # 转换为DataFrame
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_index()
        
        self.data = df
        print(f"成功获取数据: {len(df)} 条记录")
        print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
        
        return df
    
    def run_backtest(self, max_entries: int = 3):
        """
        运行回测
        
        Args:
            max_entries: 最大加仓次数
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 重置引擎
        self.engine.reset()
        
        # 初始化策略
        self.strategy.initialize(self.data)
        
        print(f"\n开始回测...")
        print(f"初始资金: {self.engine.initial_capital}")
        print(f"数据量: {len(self.data)} 条")
        print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        
        # 遍历每个K线
        for idx in range(len(self.data)):
            current_bar = self.data.iloc[idx]
            current_price = current_bar['close']
            
            # 更新权益
            self.engine.update_equity(current_price)
            
            # 检查止损（移动止盈）
            if self.engine.position_size != 0:
                stop_signal = self.strategy.check_stop_loss(
                    self.data, idx, 
                    self.engine.position_size, 
                    self.engine.entry_price
                )
                if stop_signal:
                    pnl = self.engine.close_position(
                        stop_signal['price'], 
                        idx, 
                        stop_signal['reason']
                    )
                    if pnl is not None:
                        self.strategy.update_trade_result(pnl)
                    continue
            
            # 检查均线交叉退出
            if self.engine.position_size != 0:
                signal = self.strategy.generate_signals(self.data, idx, self.engine.position_size)
                if signal['signal'] in ['close_long', 'close_short']:
                    pnl = self.engine.close_position(current_price, idx, signal['reason'])
                    if pnl is not None:
                        self.strategy.update_trade_result(pnl)
                    continue
            
            # 检查加仓
            if self.engine.position_size != 0 and self.engine.entry_count < max_entries:
                add_signal = self.strategy.check_add_position(
                    self.data, idx,
                    self.engine.position_size,
                    self.engine.entry_price
                )
                if add_signal:
                    # 计算加仓数量
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        add_size = self.strategy.get_position_size(
                            self.engine.balance,
                            atr_value,
                            add_signal['price'],
                            self.engine.leverage
                        )
                        # 限制加仓数量不超过最大可开仓位
                        max_add_value = self.engine.balance * self.engine.leverage
                        max_add_size = max_add_value / add_signal['price'] if add_signal['price'] > 0 else 0
                        actual_add_size = min(add_size, max_add_size) if max_add_size > 0 else 0
                        if actual_add_size > 0:
                            self.engine.add_position(
                                add_signal['signal'],
                                add_signal['price'],
                                actual_add_size,
                                idx,
                                add_signal['reason']
                            )
                    continue
            
            # 检查入场信号
            if self.engine.position_size == 0:
                signal = self.strategy.generate_signals(self.data, idx, 0)
                
                if signal['signal'] == 'long':
                    # 计算仓位大小
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        position_size = self.strategy.get_position_size(
                            self.engine.balance,
                            atr_value,
                            signal['price'],
                            self.engine.leverage
                        )
                        self.engine.open_position(
                            'long',
                            signal['price'],
                            position_size,
                            idx,
                            signal['reason']
                        )
                
                elif signal['signal'] == 'short':
                    # 计算仓位大小
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        position_size = self.strategy.get_position_size(
                            self.engine.balance,
                            atr_value,
                            signal['price'],
                            self.engine.leverage
                        )
                        self.engine.open_position(
                            'short',
                            signal['price'],
                            position_size,
                            idx,
                            signal['reason']
                        )
        
        # 最后平仓（如果有持仓）
        if self.engine.position_size != 0:
            last_price = self.data.iloc[-1]['close']
            pnl = self.engine.close_position(last_price, len(self.data) - 1, "回测结束平仓")
            if pnl is not None:
                self.strategy.update_trade_result(pnl)
        
        # 计算最终权益
        # 交易账户权益（只包含投入交易的部分）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        # 最终总资产 = 交易账户权益 + 未投入资金
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)
        # 收益率计算：用总资产变化除以初始资金
        # 例如：初始10000，用5000赚了1000，最终总资产=14156，收益率 = (14156-10000)/10000 = 41.56%
        total_return = (final_equity - self.engine.initial_capital) / self.engine.initial_capital * 100
        
        print(f"\n回测完成！")
        print(f"初始资金: {self.engine.initial_capital:,.2f}")
        print(f"可用资金: {self.engine.available_capital:,.2f} (仓位比例: {self.engine.position_ratio * 100:.1f}%)")
        print(f"杠杆倍数: {self.engine.leverage}x")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"总交易次数: {len([t for t in self.engine.trades if 'close' in t['type']])}")
    
    def generate_report(self) -> Dict:
        """
        生成回测报告
        
        Returns:
            包含各种统计信息的字典
        """
        # 提取所有平仓交易
        closed_trades = [t for t in self.engine.trades if 'close' in t['type']]
        
        # 计算最终权益（如果有持仓，使用当前权益；否则使用余额）
        # 交易账户权益（只包含投入交易的部分）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        # 最终总资产 = 交易账户权益 + 未投入资金
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)
        
        # 计算总收益率：用总资产变化除以初始资金
        # 例如：初始10000，用5000赚了1000，最终总资产=14156，收益率 = (14156-10000)/10000 = 41.56%
        total_return = (final_equity - self.engine.initial_capital) / self.engine.initial_capital * 100
        
        if not closed_trades:
            return {
                'initial_capital': self.engine.initial_capital,
                'available_capital': self.engine.available_capital,
                'leverage': self.engine.leverage,
                'position_ratio': self.engine.position_ratio,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'total_pnl': final_equity - self.engine.initial_capital
            }
        
        # 计算统计指标
        pnls = [t['pnl'] for t in closed_trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else (float('inf') if winning_trades else 0)
        
        # 计算最大回撤：最大回撤百分比相对于初始资金（不是峰值）
        equity_array = np.array(self.engine.equity_curve)
        if len(equity_array) > 0:
            peak = equity_array[0]
            max_drawdown = 0
            max_drawdown_pct = 0
            
            for equity in equity_array:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                # 最大回撤百分比 = 最大回撤 / 初始资金 * 100
                # 例如：初始10000，最大回撤2000，回撤百分比 = 2000/10000 = 20%
                drawdown_pct = drawdown / self.engine.initial_capital if self.engine.initial_capital > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            
            max_drawdown_pct = max_drawdown_pct * 100  # 转换为百分比
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        report = {
            'initial_capital': self.engine.initial_capital,
            'available_capital': self.engine.available_capital,
            'leverage': self.engine.leverage,
            'position_ratio': self.engine.position_ratio,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
        
        return report
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率（参考标准计算方式）"""
        if len(self.engine.equity_curve) < 2:
            return 0.0
        
        # 计算每期收益率
        returns = []
        for i in range(1, len(self.engine.equity_curve)):
            prev_equity = self.engine.equity_curve[i-1]
            curr_equity = self.engine.equity_curve[i]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # 年化夏普比率（假设数据是小时级别，252*24=6048小时/年）
        # 如果是日线，使用252；如果是小时线，使用6048
        periods_per_year = 252  # 默认按日线计算
        if self.data is not None and len(self.data) > 0:
            time_diff = (self.data.index[-1] - self.data.index[0]).total_seconds() / 3600
            if time_diff > 0:
                data_points = len(self.data)
                hours_per_point = time_diff / data_points
                if hours_per_point < 2:
                    periods_per_year = 252 * 24  # 小时线
                elif hours_per_point < 12:
                    periods_per_year = 252 * 2  # 4小时线
                else:
                    periods_per_year = 252  # 日线
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        return sharpe
    
    def plot_results(self, save_path: str = None):
        """
        绘制回测结果（基于交易记录中的盈亏数据）
        
        Args:
            save_path: 保存路径（可选）
        """
        if self.data is None:
            print("没有数据可绘制")
            return
        
        # 从交易记录中提取盈亏数据，生成权益曲线
        print("正在从交易记录生成权益曲线...")
        
        # 获取所有平仓交易（只有平仓交易才有pnl）
        closed_trades = [t for t in self.engine.trades if 'close' in t['type']]
        
        if not closed_trades:
            print("没有交易记录可绘制")
            return
        
        # 按时间索引排序
        closed_trades.sort(key=lambda x: x['idx'])
        
        # 计算累计盈亏（基于交易记录中的pnl）
        cumulative_pnl = 0
        equity_points = []  # [(时间索引, 总权益)]
        
        # 初始权益 = 可用资金
        initial_trading_equity = self.engine.available_capital
        
        # 构建每个K线的权益值
        trade_idx = 0  # 当前处理的交易索引
        
        for idx in range(len(self.data)):
            # 累加这个K线之前的所有盈亏
            while trade_idx < len(closed_trades) and closed_trades[trade_idx]['idx'] < idx:
                cumulative_pnl += closed_trades[trade_idx].get('pnl', 0)
                trade_idx += 1
            
            # 如果这个K线有平仓交易，加上这笔交易的盈亏
            current_pnl = 0
            while trade_idx < len(closed_trades) and closed_trades[trade_idx]['idx'] == idx:
                current_pnl += closed_trades[trade_idx].get('pnl', 0)
                trade_idx += 1
            
            # 计算当前权益 = 初始可用资金 + 累计盈亏
            current_trading_equity = initial_trading_equity + cumulative_pnl + current_pnl
            
            # 总权益 = 交易账户权益 + 未投入资金
            total_equity = current_trading_equity + (self.engine.initial_capital - self.engine.available_capital)
            
            equity_points.append((idx, total_equity))
            
            # 更新累计盈亏（用于下一个K线）
            cumulative_pnl += current_pnl
        
        # 提取时间和权益值
        timestamps = [self.data.index[idx] for idx, _ in equity_points]
        equities = [eq for _, eq in equity_points]
        
        # 计算收益率：相对于初始资金
        initial_equity = self.engine.initial_capital
        returns = [(eq - initial_equity) / self.engine.initial_capital * 100 for eq in equities]
        
        # 计算最终权益（与报告保持一致）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)
        
        print("正在创建图表...")
        
        # 创建图表（参考标准格式：资金曲线 + 收益率曲线）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{self.strategy.name} 回测结果', fontsize=16, fontweight='bold')
        
        # 1. 权益曲线（资金曲线）
        print("正在绘制权益曲线...")
        ax1.plot(timestamps, equities, linewidth=1.5, color='blue', alpha=0.8, label='权益曲线')
        ax1.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.7, label='初始资金')
        
        # 添加最终权益标注
        ax1.axhline(y=final_equity, color='red', linestyle='--', alpha=0.5, 
                   label=f'最终权益: {final_equity:,.2f}')
        # 在图表上添加文本标注
        ax1.text(timestamps[-1], final_equity, f' {final_equity:,.2f}', 
                verticalalignment='bottom', fontsize=10, color='red')
        
        ax1.set_title('权益曲线（基于交易记录）', fontsize=14, fontweight='bold')
        ax1.set_ylabel('权益 (USDT)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率曲线
        print("正在绘制收益率曲线...")
        ax2.plot(timestamps, returns, linewidth=1.5, color='green', alpha=0.8, label='收益率')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='零线')
        ax2.set_title('收益率曲线', fontsize=14, fontweight='bold')
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 格式化时间轴（参考标准方式）
        print("正在格式化时间轴...")
        if len(timestamps) > 100:
            # 数据点多时，减少时间轴标签
            step = max(1, len(timestamps) // 10)
            for ax in [ax1, ax2]:
                ax.set_xticks(timestamps[::step])
                ax.tick_params(axis='x', rotation=45)
        else:
            # 数据点少时，正常格式化
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)
        
        print("正在调整布局...")
        plt.tight_layout()
        
        if save_path:
            print(f"正在保存图片到: {save_path}")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"回测结果图已保存到: {save_path}")
        
        print("正在显示图表...")
        try:
            plt.show(block=True)
            print("图表显示完成")
        except Exception as e:
            print(f"图表显示失败: {str(e)}")
            print("尝试保存图片...")
            backup_path = save_path or 'backtest_result_backup.png'
            plt.savefig(backup_path, dpi=200, bbox_inches='tight')
            print(f"图片已保存为 {backup_path}")
            plt.close()
    
    def export_trades_to_csv(self, filepath: str = None):
        """
        导出交易记录到CSV文件
        
        Args:
            filepath: 保存路径，如果为None则自动生成
        """
        if not self.engine.trades:
            print("没有交易记录可导出")
            return
        
        # 准备交易记录数据
        trades_data = []
        for trade in self.engine.trades:
            trade_record = {
                '交易类型': trade['type'],
                '时间索引': trade['idx'],
                '时间': self.data.index[trade['idx']] if self.data is not None and trade['idx'] < len(self.data) else 'N/A',
                '价格': trade['price'],
                '数量': trade['size'],
                '原因': trade.get('reason', ''),
            }
            
            # 如果是平仓交易，添加盈亏信息
            if 'close' in trade['type']:
                trade_record['入场价格'] = trade.get('entry_price', 'N/A')
                trade_record['盈亏'] = trade.get('pnl', 0)
                trade_record['加仓次数'] = trade.get('entry_count', 0)
            
            # 余额：该笔交易后的现金余额（不包括未实现盈亏）
            # 说明：余额是账户中的现金，开仓时会扣除保证金，平仓时会收回
            trade_record['余额'] = trade.get('balance', 0)
            
            # 权益：该笔交易后的账户总价值（余额 + 未实现盈亏）
            # 说明：如果有持仓，权益 = 余额 + 未实现盈亏；如果无持仓，权益 = 余额
            trade_record['权益'] = trade.get('equity', trade.get('balance', 0))
            
            trades_data.append(trade_record)
        
        # 转换为DataFrame
        df = pd.DataFrame(trades_data)
        
        # 生成文件名
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"trades_{self.strategy.name}_{timestamp}.csv"
        
        # 保存到CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n交易记录已保存到: {filepath}")
        print(f"共 {len(trades_data)} 笔交易记录")
        
        return filepath
    
    def print_report(self):
        """打印回测报告（参考标准格式）"""
        report = self.generate_report()
        
        if not report or report['total_trades'] == 0:
            print("\n" + "=" * 60)
            print("回测报告")
            print("=" * 60)
            print(f"初始资金: {report.get('initial_capital', 0):,.2f}")
            print(f"最终权益: {report.get('final_equity', 0):,.2f}")
            print(f"总收益率: {report.get('total_return', 0):.2f}%")
            print(f"总交易次数: {report.get('total_trades', 0)}")
            print("=" * 60)
            return
        
        print("\n" + "=" * 60)
        print("回测报告")
        print("=" * 60)
        print(f"初始资金: {report['initial_capital']:,.2f}")
        print(f"可用资金: {report['available_capital']:,.2f} (仓位比例: {report['position_ratio']*100:.1f}%)")
        print(f"杠杆倍数: {report['leverage']}x")
        print(f"最终权益: {report['final_equity']:,.2f}")
        print(f"总收益率: {report['total_return']:.2f}%")
        print(f"总交易次数: {report['total_trades']}")
        print(f"盈利交易: {report['winning_trades']}")
        print(f"胜率: {report['win_rate']:.1f}%")
        
        print(f"\n风险指标:")
        print(f"  最大回撤: {report['max_drawdown']:,.2f}")
        print(f"  最大回撤百分比: {report['max_drawdown_pct']:.2f}%")
        print(f"  盈亏比: {report['profit_factor']:.2f}" if report['profit_factor'] != float('inf') else f"  盈亏比: ∞")
        print(f"  夏普比率: {report['sharpe_ratio']:.2f}")
        print(f"  平均盈利: {report['avg_win']:.2f}")
        print(f"  平均亏损: {report['avg_loss']:.2f}")
        print("=" * 60)
        
        # 自动导出交易记录
        self.export_trades_to_csv()


def main():
    """主函数：示例用法"""
    # 创建策略
    strategy_params = {
        'n_entries': 3,
        'atr_length': 20,
        'bo_length': 20,
        'fs_length': 55,
        'te_length': 10,
        'use_filter': False,
        'mas': 10,
        'mal': 20
    }
    
    strategy = TurtleStrategy(strategy_params)
    
    # 创建回测系统
    backtest = BacktestSystem(strategy, initial_capital=10000)
    
    # 方式1: 从CSV文件加载数据
    csv_file = "segment_1_data_ccxt_20251106_195714.csv"
    if os.path.exists(csv_file):
        print(f"从CSV文件加载数据: {csv_file}")
        backtest.load_data_from_csv(csv_file, symbol='ETHUSDT')  # 修改为ETHUSDT
    else:
        # 方式2: 从币安API加载数据
        print("从币安API加载数据...")
        backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=2000)
    
    # 运行回测
    backtest.run_backtest(max_entries=3)
    
    # 生成报告
    backtest.print_report()
    
    # 绘制结果
    backtest.plot_results(save_path='backtest_result.png')


if __name__ == "__main__":
    main()

