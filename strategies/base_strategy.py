"""
策略基类
所有策略都应该继承这个基类
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Tuple


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典
        """
        self.params = params
        self.name = self.__class__.__name__
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算指标等
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """
        生成交易信号
        
        Args:
            data: 历史数据DataFrame
            current_idx: 当前数据索引
            
        Returns:
            信号字典，包含：
            - signal: 'long', 'short', 'close_long', 'close_short', 'add_long', 'add_short', 'hold'
            - price: 信号价格
            - reason: 信号原因
        """
        pass
    
    def get_position_size(self, account_balance: float, risk_ratio: float, 
                         atr: float, entry_price: float) -> float:
        """
        计算仓位大小（基于风险比例）
        
        Args:
            account_balance: 账户余额
            risk_ratio: 风险比例（百分比）
            atr: ATR值
            entry_price: 入场价格
            
        Returns:
            仓位大小（数量）
        """
        risk_amount = account_balance * (risk_ratio / 100)
        stop_loss_distance = atr * 2  # 默认2倍ATR止损
        position_size = risk_amount / stop_loss_distance
        return position_size
    
    def on_bar(self, data: pd.DataFrame, current_idx: int, 
               position: Dict, account: Dict) -> Dict:
        """
        每个K线周期调用一次
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position: 当前持仓信息 {'size': float, 'entry_price': float, 'entry_idx': int}
            account: 账户信息 {'balance': float, 'equity': float}
            
        Returns:
            交易信号字典
        """
        return self.generate_signals(data, current_idx)

