"""
马丁策略（Martingale Strategy）
每次亏损后，下次开仓时仓位翻倍，直到盈利为止
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy


class MartingaleStrategy(BaseStrategy):
    """
    马丁策略
    
    策略逻辑：
    1. 基于简单的趋势判断（如均线）或随机入场
    2. 每次亏损后，下次开仓时仓位翻倍
    3. 盈利后重置仓位倍数
    4. 可以设置最大仓位倍数限制，防止过度加仓
    """
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典，包含：
                - base_position_ratio: 基础仓位比例（默认0.1，即10%资金）
                - multiplier: 亏损后仓位倍数（默认2.0，即翻倍）
                - max_multiplier: 最大仓位倍数（默认8.0，防止过度加仓）
                - ma_short: 短周期均线（默认10，用于趋势判断）
                - ma_long: 长周期均线（默认20，用于趋势判断）
                - use_trend_filter: 是否使用趋势过滤（默认True）
                - stop_loss_pct: 止损百分比（默认0.05，即5%）
                - take_profit_pct: 止盈百分比（默认0.10，即10%）
        """
        super().__init__(params)
        
        # 策略参数
        self.base_position_ratio = params.get('base_position_ratio', 0.1)
        self.multiplier = params.get('multiplier', 2.0)
        self.max_multiplier = params.get('max_multiplier', 8.0)
        self.ma_short = params.get('ma_short', 10)
        self.ma_long = params.get('ma_long', 20)
        self.use_trend_filter = params.get('use_trend_filter', True)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = params.get('take_profit_pct', 0.10)
        
        # 策略状态
        self.current_multiplier = 1.0  # 当前仓位倍数
        self.last_trade_profit = 0.0  # 上次交易盈亏
        self.last_trade_type = None  # 上次交易类型（'long'或'short'）
        
        # 指标缓存
        self.ma_short_line = None
        self.ma_long_line = None
        
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算指标
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        # 计算均线
        self.ma_short_line = data['close'].rolling(window=self.ma_short).mean()
        self.ma_long_line = data['close'].rolling(window=self.ma_long).mean()
        
    def generate_signals(self, data: pd.DataFrame, current_idx: int, position_size: float = 0) -> Dict:
        """
        生成交易信号
        
        Args:
            data: 历史数据DataFrame
            current_idx: 当前数据索引
            position_size: 当前持仓大小（正数=多头，负数=空头，0=无持仓）
            
        Returns:
            信号字典
        """
        if current_idx < self.ma_long:
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '数据不足'}
        
        current_bar = data.iloc[current_idx]
        current_price = current_bar['close']
        
        # 获取均线值
        ma_short = self.ma_short_line.iloc[current_idx]
        ma_long = self.ma_long_line.iloc[current_idx]
        
        if pd.isna(ma_short) or pd.isna(ma_long):
            return {'signal': 'hold', 'price': current_price, 'reason': '指标未就绪'}
        
        # 如果有持仓，检查止损和止盈
        if position_size != 0:
            entry_price = current_bar.get('entry_price', current_price)  # 从外部传入，这里用当前价格作为默认值
            
            # 多头止损/止盈
            if position_size > 0:
                # 止损
                if current_price <= entry_price * (1 - self.stop_loss_pct):
                    return {
                        'signal': 'close_long',
                        'price': current_price,
                        'reason': f'止损（亏损{self.stop_loss_pct*100:.1f}%）'
                    }
                # 止盈
                if current_price >= entry_price * (1 + self.take_profit_pct):
                    return {
                        'signal': 'close_long',
                        'price': current_price,
                        'reason': f'止盈（盈利{self.take_profit_pct*100:.1f}%）'
                    }
            
            # 空头止损/止盈
            elif position_size < 0:
                # 止损
                if current_price >= entry_price * (1 + self.stop_loss_pct):
                    return {
                        'signal': 'close_short',
                        'price': current_price,
                        'reason': f'止损（亏损{self.stop_loss_pct*100:.1f}%）'
                    }
                # 止盈
                if current_price <= entry_price * (1 - self.take_profit_pct):
                    return {
                        'signal': 'close_short',
                        'price': current_price,
                        'reason': f'止盈（盈利{self.take_profit_pct*100:.1f}%）'
                    }
        
        # 无持仓时，检查入场信号
        if position_size == 0:
            # 趋势判断
            if self.use_trend_filter:
                # 均线金叉：做多
                if ma_short > ma_long and self.ma_short_line.iloc[current_idx-1] <= self.ma_long_line.iloc[current_idx-1]:
                    return {
                        'signal': 'long',
                        'price': current_price,
                        'reason': '均线金叉做多'
                    }
                # 均线死叉：做空
                elif ma_short < ma_long and self.ma_short_line.iloc[current_idx-1] >= self.ma_long_line.iloc[current_idx-1]:
                    return {
                        'signal': 'short',
                        'price': current_price,
                        'reason': '均线死叉做空'
                    }
            else:
                # 不使用趋势过滤：简单的价格突破
                if current_price > ma_long:
                    return {
                        'signal': 'long',
                        'price': current_price,
                        'reason': '价格突破均线做多'
                    }
                elif current_price < ma_long:
                    return {
                        'signal': 'short',
                        'price': current_price,
                        'reason': '价格跌破均线做空'
                    }
        
        return {'signal': 'hold', 'price': current_price, 'reason': '无信号'}
    
    def get_position_size(self, account_balance: float, entry_price: float, leverage: float = 1.0) -> float:
        """
        计算仓位大小（考虑马丁倍数）
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            leverage: 杠杆倍数
            
        Returns:
            仓位大小（数量）
        """
        if entry_price <= 0:
            return 0
        
        # 基础仓位价值 = 账户余额 * 基础仓位比例 * 当前倍数
        position_value = account_balance * self.base_position_ratio * self.current_multiplier * leverage
        
        # 仓位数量 = 仓位价值 / 入场价格
        position_size = position_value / entry_price
        
        return position_size
    
    def update_trade_result(self, profit: float):
        """
        更新交易结果，调整仓位倍数
        
        Args:
            profit: 上次交易的盈亏（正数=盈利，负数=亏损）
        """
        self.last_trade_profit = profit
        
        if profit < 0:
            # 亏损：增加仓位倍数（翻倍）
            self.current_multiplier = min(self.current_multiplier * self.multiplier, self.max_multiplier)
        else:
            # 盈利：重置仓位倍数
            self.current_multiplier = 1.0
    
    def check_stop_loss(self, data: pd.DataFrame, current_idx: int,
                       position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查止损（已在generate_signals中实现）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            止损信号字典或None
        """
        # 止损逻辑已在generate_signals中实现
        return None
    
    def check_add_position(self, data: pd.DataFrame, current_idx: int,
                          position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查是否需要加仓（马丁策略通常不加仓，而是通过倍数调整下次开仓）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            加仓信号字典或None（马丁策略通常返回None）
        """
        # 马丁策略通常不在持仓中加仓，而是通过倍数调整下次开仓
        return None

