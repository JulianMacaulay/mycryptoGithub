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
    
    def get_position_size(self, account_balance: float, entry_price: float, leverage: float = 1.0) -> float:
        """
        计算仓位大小
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            leverage: 杠杆倍数（默认1.0，即无杠杆）
            
        Returns:
            仓位大小（数量）
        """
        if entry_price <= 0:
            return 0
        
        # 能开的仓位价值 = 账户余额 * 杠杆倍数
        # 能开的数量 = 仓位价值 / 入场价格
        position_size = (account_balance * leverage) / entry_price
        
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

