"""
RSI震荡策略（RSI Oscillation Strategy）
基于RSI指标的超买超卖策略，适合震荡市场
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

# 尝试导入talib，如果失败则使用自定义函数
try:
    import talib
    TALIB_AVAILABLE = False
except ImportError:
    TALIB_AVAILABLE = False


class RSIOscillationStrategy(BaseStrategy):
    """
    RSI震荡策略
    
    策略逻辑：
    1. RSI < 超卖阈值（默认30）时买入
    2. RSI > 超买阈值（默认70）时卖出
    3. 支持反向开仓：如果有多头持仓但RSI > 超买阈值，先平多再开空
    4. 如果有多头持仓但RSI < 超卖阈值，先平多再开多（重新入场）
    5. 适合震荡市场，通过RSI的超买超卖信号进行交易
    """
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典，包含：
                - rsi_period: RSI周期（默认14）
                - oversold_level: 超卖阈值（默认30，RSI < 30时买入）
                - overbought_level: 超买阈值（默认70，RSI > 70时卖出）
                - enable_reverse: 是否启用反向开仓（默认True）
                - stop_loss_pct: 止损百分比（默认0.05，即5%）
                - take_profit_pct: 止盈百分比（默认0.10，即10%）
                - rsi_exit_level: RSI退出阈值（默认50，RSI回到50时平仓，可选）
                - use_rsi_exit: 是否使用RSI退出（默认False，使用止损止盈）
        """
        super().__init__(params)
        
        # 策略参数
        self.rsi_period = params.get('rsi_period', 14)
        self.oversold_level = params.get('oversold_level', 30)
        self.overbought_level = params.get('overbought_level', 70)
        self.enable_reverse = params.get('enable_reverse', True)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = params.get('take_profit_pct', 0.10)
        self.rsi_exit_level = params.get('rsi_exit_level', 50)
        self.use_rsi_exit = params.get('use_rsi_exit', False)
        
        # 指标缓存
        self.rsi = None
        
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算RSI指标
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        close = data['close']
        
        # 计算RSI
        if TALIB_AVAILABLE:
            close_array = close.values.astype(np.float64)
            rsi_array = talib.RSI(close_array, timeperiod=self.rsi_period)
            self.rsi = pd.Series(rsi_array, index=data.index)
        else:
            # 使用自定义RSI计算函数
            self.rsi = self._calculate_rsi(close, self.rsi_period)
        
        print(f"RSI震荡策略初始化完成：")
        print(f"  RSI周期: {self.rsi_period}")
        print(f"  超卖阈值: {self.oversold_level}")
        print(f"  超买阈值: {self.overbought_level}")
        print(f"  反向开仓: {'启用' if self.enable_reverse else '禁用'}")
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI（相对强弱指标）
        
        Args:
            close: 收盘价序列
            period: 周期（默认14）
            
        Returns:
            RSI序列
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
        if current_idx < self.rsi_period:
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '数据不足'}
        
        current_bar = data.iloc[current_idx]
        current_price = current_bar['close']
        
        # 获取RSI值
        rsi_value = self.rsi.iloc[current_idx]
        
        if pd.isna(rsi_value):
            return {'signal': 'hold', 'price': current_price, 'reason': 'RSI未就绪'}
        
        # 如果有持仓，先检查止损和止盈
        if position_size != 0:
            # 从回测系统获取入场价格
            if self.engine is not None:
                default_pos = self.engine.get_position_by_id('default')
                if default_pos is not None:
                    entry_price = default_pos['entry_price']
                else:
                    entry_price = current_price  # 如果无法获取，使用当前价格
            else:
                entry_price = current_bar.get('entry_price', current_price)
            
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
                
                # RSI退出（如果启用）
                if self.use_rsi_exit and rsi_value >= self.rsi_exit_level:
                    return {
                        'signal': 'close_long',
                        'price': current_price,
                        'reason': f'RSI退出（RSI={rsi_value:.1f}）'
                    }
                
                # 反向开仓：如果有多头持仓但RSI > 超买阈值，先平多再开空
                if self.enable_reverse and rsi_value > self.overbought_level:
                    return {
                        'signal': 'close_long',
                        'price': current_price,
                        'reason': f'RSI超买平多（RSI={rsi_value:.1f}）',
                        'reverse_signal': 'short',  # 标记需要反向开仓
                        'reverse_price': current_price
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
                
                # RSI退出（如果启用）
                if self.use_rsi_exit and rsi_value <= self.rsi_exit_level:
                    return {
                        'signal': 'close_short',
                        'price': current_price,
                        'reason': f'RSI退出（RSI={rsi_value:.1f}）'
                    }
                
                # 反向开仓：如果有空头持仓但RSI < 超卖阈值，先平空再开多
                if self.enable_reverse and rsi_value < self.oversold_level:
                    return {
                        'signal': 'close_short',
                        'price': current_price,
                        'reason': f'RSI超卖平空（RSI={rsi_value:.1f}）',
                        'reverse_signal': 'long',  # 标记需要反向开仓
                        'reverse_price': current_price
                    }
        
        # 无持仓时，检查入场信号
        if position_size == 0:
            # RSI < 超卖阈值：买入
            if rsi_value < self.oversold_level:
                return {
                    'signal': 'long',
                    'price': current_price,
                    'reason': f'RSI超卖买入（RSI={rsi_value:.1f}）'
                }
            
            # RSI > 超买阈值：做空
            elif rsi_value > self.overbought_level:
                return {
                    'signal': 'short',
                    'price': current_price,
                    'reason': f'RSI超买做空（RSI={rsi_value:.1f}）'
                }
        
        return {'signal': 'hold', 'price': current_price, 'reason': f'无信号（RSI={rsi_value:.1f}）'}
    
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
        检查是否需要加仓（RSI震荡策略通常不加仓）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            加仓信号字典或None
        """
        # RSI震荡策略通常不加仓
        return None
    
    def update_trade_result(self, profit: float):
        """
        更新交易结果（RSI震荡策略可以用于统计或调整参数）
        
        Args:
            profit: 上次交易的盈亏（正数=盈利，负数=亏损）
        """
        # RSI震荡策略可以在这里记录交易结果，用于后续优化
        # 目前不需要特殊处理，但方法必须存在以兼容回测系统
        pass


