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

# 添加策略模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.turtle_strategy import TurtleStrategy
from strategies.base_strategy import BaseStrategy


class BacktestEngine:
    """
    回测引擎
    负责执行策略回测，记录交易，计算收益
    """
    
    def __init__(self, initial_capital: float = 10000, commission_rate: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（默认0.1%）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # 账户状态
        self.balance = initial_capital
        self.equity = initial_capital
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
        self.balance = self.initial_capital
        self.equity = self.initial_capital
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
        if signal == 'long':
            self.position_size = size
            self.entry_price = price
            self.entry_idx = current_idx
            self.entry_count = 1
            
            # 计算成本（包含手续费）
            cost = size * price * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'open_long',
                'price': price,
                'size': size,
                'idx': current_idx,
                'balance': self.balance,
                'reason': reason
            })
            
        elif signal == 'short':
            self.position_size = -size
            self.entry_price = price
            self.entry_idx = current_idx
            self.entry_count = 1
            
            # 做空：先卖出，获得资金
            cost = size * price * (1 + self.commission_rate)
            self.balance += size * price - cost
            
            self.trades.append({
                'type': 'open_short',
                'price': price,
                'size': size,
                'idx': current_idx,
                'balance': self.balance,
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
        if signal == 'add_long' and self.position_size > 0:
            # 计算新的平均入场价格
            total_size = self.position_size + size
            total_cost = self.position_size * self.entry_price + size * price
            self.entry_price = total_cost / total_size
            self.position_size = total_size
            self.entry_count += 1
            
            # 扣除成本
            cost = size * price * (1 + self.commission_rate)
            self.balance -= cost
            
            self.trades.append({
                'type': 'add_long',
                'price': price,
                'size': size,
                'idx': current_idx,
                'balance': self.balance,
                'reason': reason
            })
            
        elif signal == 'add_short' and self.position_size < 0:
            # 计算新的平均入场价格
            total_size = abs(self.position_size) + size
            total_cost = abs(self.position_size) * abs(self.entry_price) + size * price
            self.entry_price = -(total_cost / total_size)
            self.position_size = -total_size
            self.entry_count += 1
            
            # 做空加仓：获得资金
            cost = size * price * (1 + self.commission_rate)
            self.balance += size * price - cost
            
            self.trades.append({
                'type': 'add_short',
                'price': price,
                'size': size,
                'idx': current_idx,
                'balance': self.balance,
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
            return
        
        # 计算盈亏
        if self.position_size > 0:  # 平多
            pnl = (price - self.entry_price) * self.position_size
            cost = self.position_size * price * self.commission_rate
            self.balance += self.position_size * price - cost
        else:  # 平空
            pnl = (abs(self.entry_price) - price) * abs(self.position_size)
            cost = abs(self.position_size) * price * self.commission_rate
            self.balance += abs(self.position_size) * abs(self.entry_price) - cost
        
        # 记录交易
        trade_type = 'close_long' if self.position_size > 0 else 'close_short'
        self.trades.append({
            'type': trade_type,
            'price': price,
            'size': abs(self.position_size),
            'idx': current_idx,
            'entry_price': self.entry_price,
            'pnl': pnl,
            'balance': self.balance,
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
            unrealized_pnl = (abs(self.entry_price) - current_price) * abs(self.position_size)
            self.equity = self.balance + unrealized_pnl
            self.position_value = abs(self.position_size) * current_price
        
        self.equity_curve.append(self.equity)


class BacktestSystem:
    """
    完整的回测系统
    """
    
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10000):
        """
        初始化回测系统
        
        Args:
            strategy: 策略实例
            initial_capital: 初始资金
        """
        self.strategy = strategy
        self.engine = BacktestEngine(initial_capital=initial_capital)
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
            df = df[df['symbol'] == symbol].copy()
        
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
    
    def run_backtest(self, max_entries: int = 3, risk_ratio: float = 1.0):
        """
        运行回测
        
        Args:
            max_entries: 最大加仓次数
            risk_ratio: 风险比例（百分比）
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
                    # 计算加仓数量（使用相同的风险比例）
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        add_size = self.strategy.get_position_size(
                            self.engine.equity,
                            risk_ratio,
                            atr_value,
                            add_signal['price']
                        )
                        self.engine.add_position(
                            add_signal['signal'],
                            add_signal['price'],
                            add_size,
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
                            self.engine.equity,
                            risk_ratio,
                            atr_value,
                            signal['price']
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
                            self.engine.equity,
                            risk_ratio,
                            atr_value,
                            signal['price']
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
        
        print(f"\n回测完成！")
        print(f"最终权益: {self.engine.equity:.2f}")
        print(f"总收益率: {(self.engine.equity / self.engine.initial_capital - 1) * 100:.2f}%")
        print(f"总交易次数: {len([t for t in self.engine.trades if 'close' in t['type']])}")
    
    def generate_report(self) -> Dict:
        """
        生成回测报告
        
        Returns:
            包含各种统计信息的字典
        """
        if not self.engine.trades:
            return {}
        
        # 提取所有平仓交易
        closed_trades = [t for t in self.engine.trades if 'close' in t['type']]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'final_equity': self.engine.equity,
                'total_return': (self.engine.equity / self.engine.initial_capital - 1) * 100
            }
        
        # 计算统计指标
        pnls = [t['pnl'] for t in closed_trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_return = (self.engine.equity / self.engine.initial_capital - 1) * 100
        total_pnl = sum(pnls)
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        
        # 计算最大回撤
        equity_array = np.array(self.engine.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        report = {
            'initial_capital': self.engine.initial_capital,
            'final_equity': self.engine.equity,
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
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
        
        return report
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        if len(self.engine.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(self.engine.equity_curve) / self.engine.equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe = (np.mean(returns) - risk_free_rate / 252) / np.std(returns) * np.sqrt(252)
        return sharpe
    
    def plot_results(self, save_path: str = None):
        """
        绘制回测结果
        
        Args:
            save_path: 保存路径（可选）
        """
        if self.data is None or not self.engine.equity_curve:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{self.strategy.name} 回测结果', fontsize=16, fontweight='bold')
        
        # 1. 价格和信号
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['close'], label='收盘价', linewidth=1, alpha=0.7)
        
        # 绘制指标
        if hasattr(self.strategy, 'donchian_hi'):
            ax1.plot(self.data.index, self.strategy.donchian_hi, 
                    label='Donchian High (20)', color='green', alpha=0.5, linewidth=0.8)
            ax1.plot(self.data.index, self.strategy.donchian_lo, 
                    label='Donchian Low (20)', color='red', alpha=0.5, linewidth=0.8)
        
        # 标记交易点
        for trade in self.engine.trades:
            if 'open' in trade['type']:
                color = 'green' if 'long' in trade['type'] else 'red'
                marker = '^' if 'long' in trade['type'] else 'v'
                ax1.scatter(self.data.index[trade['idx']], trade['price'], 
                           color=color, marker=marker, s=50, alpha=0.7, zorder=5)
            elif 'close' in trade['type']:
                ax1.scatter(self.data.index[trade['idx']], trade['price'], 
                           color='black', marker='x', s=50, alpha=0.7, zorder=5)
        
        ax1.set_ylabel('价格', fontsize=10)
        ax1.set_title('价格走势和交易信号', fontsize=12)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 权益曲线
        ax2 = axes[1]
        ax2.plot(self.data.index[:len(self.engine.equity_curve)], 
                self.engine.equity_curve, label='权益曲线', linewidth=2, color='blue')
        ax2.axhline(y=self.engine.initial_capital, color='gray', 
                   linestyle='--', label='初始资金', alpha=0.5)
        ax2.set_ylabel('权益 (USDT)', fontsize=10)
        ax2.set_title('权益曲线', fontsize=12)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤
        ax3 = axes[2]
        equity_array = np.array(self.engine.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        ax3.fill_between(self.data.index[:len(drawdown)], drawdown, 0, 
                        color='red', alpha=0.3, label='回撤')
        ax3.set_ylabel('回撤 (%)', fontsize=10)
        ax3.set_xlabel('时间', fontsize=10)
        ax3.set_title('回撤曲线', fontsize=12)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def print_report(self):
        """打印回测报告"""
        report = self.generate_report()
        
        if not report:
            print("没有交易记录")
            return
        
        print("\n" + "=" * 60)
        print("回测报告")
        print("=" * 60)
        print(f"初始资金: {report['initial_capital']:.2f} USDT")
        print(f"最终权益: {report['final_equity']:.2f} USDT")
        print(f"总收益率: {report['total_return']:.2f}%")
        print(f"总盈亏: {report['total_pnl']:.2f} USDT")
        print(f"最大回撤: {report['max_drawdown']:.2f}%")
        print(f"夏普比率: {report['sharpe_ratio']:.2f}")
        print("-" * 60)
        print(f"总交易次数: {report['total_trades']}")
        print(f"盈利次数: {report['winning_trades']}")
        print(f"亏损次数: {report['losing_trades']}")
        print(f"胜率: {report['win_rate']:.2f}%")
        print(f"平均盈利: {report['avg_win']:.2f} USDT")
        print(f"平均亏损: {report['avg_loss']:.2f} USDT")
        print(f"盈亏比: {report['profit_factor']:.2f}")
        print("=" * 60)


def main():
    """主函数：示例用法"""
    # 创建策略
    strategy_params = {
        'n_entries': 3,
        'risk_ratio': 1.0,
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
    csv_file = "all_symbols_data_ccxt_20251106_195714.csv"
    if os.path.exists(csv_file):
        print(f"从CSV文件加载数据: {csv_file}")
        backtest.load_data_from_csv(csv_file, symbol='BTCUSDT')
    else:
        # 方式2: 从币安API加载数据
        print("从币安API加载数据...")
        backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=1000)
    
    # 运行回测
    backtest.run_backtest(max_entries=3, risk_ratio=1.0)
    
    # 生成报告
    backtest.print_report()
    
    # 绘制结果
    backtest.plot_results(save_path='backtest_result.png')


if __name__ == "__main__":
    main()

