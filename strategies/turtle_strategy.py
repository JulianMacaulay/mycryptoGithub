"""
海龟交易策略
基于唐奇安通道突破的交易策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy
from .indicators import (
    calculate_atr, 
    calculate_donchian_high, 
    calculate_donchian_low,
    calculate_sma,
    detect_crossover,
    detect_crossunder
)


class TurtleStrategy(BaseStrategy):
    """
    原版海龟交易策略
    
    策略逻辑：
    1. 使用20日和55日唐奇安通道进行突破交易
    2. 支持加仓（最多3次，每0.5N加一次）
    3. 使用10日高低点进行移动止盈
    4. 使用均线交叉作为退出信号
    5. 可选的上次盈利过滤
    """
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典，包含：
                - n_entries: 最大加仓次数（默认3）
                - risk_ratio: 风险比例%（默认1.0）
                - atr_length: ATR周期（默认20）
                - bo_length: 短周期突破（默认20）
                - fs_length: 长周期突破（默认55）
                - te_length: 移动止盈周期（默认10）
                - use_filter: 是否使用上次盈利过滤（默认False）
                - mas: 短周期均线（默认10）
                - mal: 长周期均线（默认20）
        """
        super().__init__(params)
        
        # 策略参数
        self.n_entries = params.get('n_entries', 3)
        self.risk_ratio = params.get('risk_ratio', 1.0)
        self.atr_length = params.get('atr_length', 20)
        self.bo_length = params.get('bo_length', 20)
        self.fs_length = params.get('fs_length', 55)
        self.te_length = params.get('te_length', 10)
        self.use_filter = params.get('use_filter', False)
        self.mas = params.get('mas', 10)
        self.mal = params.get('mal', 20)
        
        # 策略状态
        self.last_trade_loss = True  # 上次交易是否亏损
        self.last_entry_price = None  # 上次入场价格
        self.entry_count = 0  # 当前加仓次数
        
        # 指标缓存
        self.atr = None
        self.donchian_hi = None
        self.donchian_lo = None
        self.fs_donchian_hi = None
        self.fs_donchian_lo = None
        self.exit_lowest = None
        self.exit_highest = None
        self.ma_short = None
        self.ma_long = None
        
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算所有指标
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        # 计算ATR
        self.atr = calculate_atr(data['high'], data['low'], data['close'], self.atr_length)
        
        # 计算唐奇安通道（延后1个周期）
        self.donchian_hi = calculate_donchian_high(data['high'], self.bo_length, shift=1)
        self.donchian_lo = calculate_donchian_low(data['low'], self.bo_length, shift=1)
        
        # 计算长周期唐奇安通道（Failsafe）
        self.fs_donchian_hi = calculate_donchian_high(data['high'], self.fs_length, shift=1)
        self.fs_donchian_lo = calculate_donchian_low(data['low'], self.fs_length, shift=1)
        
        # 计算移动止盈
        self.exit_lowest = calculate_donchian_low(data['low'], self.te_length, shift=1)
        self.exit_highest = calculate_donchian_high(data['high'], self.te_length, shift=1)
        
        # 计算均线
        self.ma_short = calculate_sma(data['close'], self.mas)
        self.ma_long = calculate_sma(data['close'], self.mal)
        
    def generate_signals(self, data: pd.DataFrame, current_idx: int, position_size: float = 0) -> Dict:
        """
        生成交易信号
        
        Args:
            data: 历史数据DataFrame
            current_idx: 当前数据索引
            position_size: 当前持仓大小（正数=多头，负数=空头，0=无持仓）
            
        Returns:
            信号字典，包含：
            - signal: 'long', 'short', 'close_long', 'close_short', 'add_long', 'add_short', 'hold'
            - price: 信号价格
            - reason: 信号原因
            - position_size: 建议仓位大小（可选）
        """
        if current_idx < max(self.atr_length, self.fs_length, self.mal):
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '数据不足'}
        
        current_bar = data.iloc[current_idx]
        prev_idx = current_idx - 1
        
        # 获取当前指标值
        atr_value = self.atr.iloc[current_idx]
        donchian_hi = self.donchian_hi.iloc[prev_idx] if prev_idx >= 0 else None
        donchian_lo = self.donchian_lo.iloc[prev_idx] if prev_idx >= 0 else None
        fs_donchian_hi = self.fs_donchian_hi.iloc[prev_idx] if prev_idx >= 0 else None
        fs_donchian_lo = self.fs_donchian_lo.iloc[prev_idx] if prev_idx >= 0 else None
        
        ma_short = self.ma_short.iloc[current_idx]
        ma_long = self.ma_long.iloc[current_idx]
        
        # 检查是否有有效值
        if pd.isna(atr_value) or pd.isna(donchian_hi) or pd.isna(donchian_lo):
            return {'signal': 'hold', 'price': current_bar['close'], 'reason': '指标未就绪'}
        
        # 检测均线交叉（用于退出）
        crossunder = detect_crossunder(self.ma_short, self.ma_long)
        crossover = detect_crossover(self.ma_short, self.ma_long)
        
        # 判断是否允许突破（过滤条件）
        allow_breakout = not self.use_filter or self.last_trade_loss
        
        # 入场信号
        long_entry = allow_breakout and current_bar['high'] > donchian_hi
        short_entry = allow_breakout and current_bar['low'] < donchian_lo
        
        # 长周期突破（Failsafe）- 只在无持仓时触发
        long_entry_fs = (position_size == 0) and current_bar['high'] > fs_donchian_hi
        short_entry_fs = (position_size == 0) and current_bar['low'] < fs_donchian_lo
        
        # 返回信号（优先级：退出 > 入场）
        signals = []
        
        # 检查退出信号（均线交叉）
        if crossunder.iloc[current_idx] and position_size > 0:
            signals.append({
                'signal': 'close_long',
                'price': current_bar['close'],
                'reason': '均线死叉退出'
            })
        
        if crossover.iloc[current_idx] and position_size < 0:
            signals.append({
                'signal': 'close_short',
                'price': current_bar['close'],
                'reason': '均线金叉退出'
            })
        
        # 检查入场信号（只在无持仓时）
        if position_size == 0:
            if long_entry:
                signals.append({
                    'signal': 'long',
                    'price': donchian_hi,
                    'reason': '20日通道突破做多'
                })
            
            if short_entry:
                signals.append({
                    'signal': 'short',
                    'price': donchian_lo,
                    'reason': '20日通道突破做空'
                })
            
            if long_entry_fs:
                signals.append({
                    'signal': 'long',
                    'price': fs_donchian_hi,
                    'reason': '55日通道突破做多（Failsafe）'
                })
            
            if short_entry_fs:
                signals.append({
                    'signal': 'short',
                    'price': fs_donchian_lo,
                    'reason': '55日通道突破做空（Failsafe）'
                })
        
        # 返回第一个信号（优先级：退出 > 入场）
        if signals:
            return signals[0]
        
        return {'signal': 'hold', 'price': current_bar['close'], 'reason': '无信号'}
    
    def check_add_position(self, data: pd.DataFrame, current_idx: int, 
                          position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查是否需要加仓
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小（正数=多头，负数=空头）
            last_entry_price: 上次入场价格
            
        Returns:
            加仓信号字典或None
        """
        if current_idx < self.atr_length:
            return None
        
        atr_value = self.atr.iloc[current_idx]
        current_bar = data.iloc[current_idx]
        
        if pd.isna(atr_value) or atr_value == 0:
            return None
        
        # 多头加仓：价格上涨0.5N
        if position_size > 0:
            add_price = last_entry_price + 0.5 * atr_value
            if current_bar['high'] >= add_price:
                return {
                    'signal': 'add_long',
                    'price': add_price,
                    'reason': f'多头加仓（价格达到{add_price:.4f}）'
                }
        
        # 空头加仓：价格下跌0.5N
        elif position_size < 0:
            add_price = last_entry_price - 0.5 * atr_value
            if current_bar['low'] <= add_price:
                return {
                    'signal': 'add_short',
                    'price': add_price,
                    'reason': f'空头加仓（价格达到{add_price:.4f}）'
                }
        
        return None
    
    def check_stop_loss(self, data: pd.DataFrame, current_idx: int,
                       position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查止损（移动止盈）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            止损信号字典或None
        """
        if current_idx < self.te_length:
            return None
        
        current_bar = data.iloc[current_idx]
        prev_idx = current_idx - 1
        
        exit_lowest = self.exit_lowest.iloc[prev_idx] if prev_idx >= 0 else None
        exit_highest = self.exit_highest.iloc[prev_idx] if prev_idx >= 0 else None
        
        if pd.isna(exit_lowest) or pd.isna(exit_highest):
            return None
        
        # 多头止损：价格跌破10日最低点
        if position_size > 0:
            if current_bar['low'] < exit_lowest:
                return {
                    'signal': 'close_long',
                    'price': exit_lowest,
                    'reason': '移动止盈（跌破10日最低点）'
                }
        
        # 空头止损：价格涨破10日最高点
        elif position_size < 0:
            if current_bar['high'] > exit_highest:
                return {
                    'signal': 'close_short',
                    'price': exit_highest,
                    'reason': '移动止盈（涨破10日最高点）'
                }
        
        return None
    
    def update_trade_result(self, profit: float):
        """
        更新交易结果（用于过滤条件）
        
        Args:
            profit: 上次交易的盈亏（正数=盈利，负数=亏损）
        """
        self.last_trade_loss = profit < 0

