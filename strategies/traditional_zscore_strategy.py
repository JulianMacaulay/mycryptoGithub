"""
传统Z-score计算策略
使用均值和标准差计算Z-score
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List
import numpy as np


class TraditionalZScoreStrategy(BaseZScoreStrategy):
    """传统Z-score计算策略（使用均值和标准差）"""
    
    def __init__(self, **kwargs):
        """
        初始化传统Z-score策略
        
        Args:
            **kwargs: 策略参数（当前无需额外参数）
        """
        super().__init__(**kwargs)
        self.name = "传统方法"
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        计算当前Z-score（传统方法：使用均值和标准差）
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=2):
            return 0.0
        
        # 计算均值和标准差
        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)
        
        # 如果标准差为0，返回0
        if spread_std == 0:
            return 0.0
        
        # 计算Z-score
        z_score = (current_spread - spread_mean) / spread_std
        
        return z_score
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return "传统方法（均值和标准差）"


