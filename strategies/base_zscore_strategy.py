"""
Z-score计算策略基类
所有Z-score计算策略都应该继承这个基类
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseZScoreStrategy(ABC):
    """Z-score计算策略基类"""
    
    def __init__(self, **kwargs):
        """
        初始化策略
        
        Args:
            **kwargs: 策略特定参数
        """
        self.name = self.__class__.__name__
        self.params = kwargs
    
    @abstractmethod
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float]) -> float:
        """
        计算当前Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        pass
    
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        return self.name
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"{self.name} - Z-score计算策略"
    
    def validate_input(self, historical_spreads: List[float], min_length: int = 2) -> bool:
        """
        验证输入数据
        
        Args:
            historical_spreads: 历史价差序列
            min_length: 最小数据长度
            
        Returns:
            bool: 数据是否有效
        """
        if not historical_spreads or len(historical_spreads) < min_length:
            return False
        return True

