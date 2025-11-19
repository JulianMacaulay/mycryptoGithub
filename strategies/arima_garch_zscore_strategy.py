"""
ARIMA-GARCH Z-score计算策略
使用ARIMA模型预测均值，GARCH模型预测波动率
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List, Tuple, Optional
import numpy as np

# 尝试导入ARIMA和GARCH相关库
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("警告: statsmodels未安装或版本过低，ARIMA功能将不可用。可以使用: pip install statsmodels")

try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False
    print("警告: arch未安装，GARCH功能将不可用。可以使用: pip install arch")


class ArimaGarchZScoreStrategy(BaseZScoreStrategy):
    """ARIMA-GARCH Z-score计算策略"""
    
    def __init__(self, arima_order: Tuple[int, int, int] = (1, 0, 1), 
                 garch_order: Tuple[int, int] = (1, 1), **kwargs):
        """
        初始化ARIMA-GARCH策略
        
        Args:
            arima_order: ARIMA模型阶数 (p, d, q)
            garch_order: GARCH模型阶数 (p, q)
            **kwargs: 其他策略参数
        """
        super().__init__(arima_order=arima_order, garch_order=garch_order, **kwargs)
        self.name = "ARIMA-GARCH模型"
        self.arima_order = arima_order
        self.garch_order = garch_order
        
        # 检查库是否可用
        if not ARIMA_AVAILABLE or not GARCH_AVAILABLE:
            raise ImportError("ARIMA或GARCH库不可用，无法使用ARIMA-GARCH策略")
        
        # 存储模型缓存（限制缓存大小）
        self._arima_garch_models = {}
        self._max_cache_size = 10
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用ARIMA-GARCH模型计算Z-score
        
        步骤：
        1. 使用ARIMA模型对历史价差序列建模，预测当前价差的均值
        2. 计算ARIMA模型的残差
        3. 使用GARCH模型对残差建模，预测当前价差的波动率
        4. 使用预测的均值和波动率计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据（ARIMA-GARCH需要更多数据）
        min_required_length = max(20, sum(self.arima_order) + 5)
        if not self.validate_input(historical_spreads, min_length=min_required_length):
            # 数据不足时返回0，不进行回退
            return 0.0
        
        try:
            # 转换为numpy数组
            spreads_array = np.array(historical_spreads)
            
            # 创建缓存键（基于数据长度和最后几个值）
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._arima_garch_models:
                arima_fitted, garch_fitted = self._arima_garch_models[cache_key]
            else:
                # 步骤1: 拟合ARIMA模型
                arima_model = ARIMA(spreads_array, order=self.arima_order)
                arima_fitted = arima_model.fit()
                
                # 步骤2: 获取ARIMA残差
                arima_residuals = arima_fitted.resid
                
                # 确保残差有足够的数据
                min_residuals_length = max(10, sum(self.garch_order) + 3)
                if len(arima_residuals) < min_residuals_length:
                    # 残差数据不足，返回0
                    return 0.0
                
                # 步骤3: 拟合GARCH模型
                garch_model = arch_model(arima_residuals, vol='Garch', 
                                        p=self.garch_order[0], q=self.garch_order[1])
                garch_fitted = garch_model.fit(disp='off')
                
                # 缓存模型（限制缓存大小）
                if len(self._arima_garch_models) >= self._max_cache_size:
                    # 清除最旧的缓存
                    oldest_key = next(iter(self._arima_garch_models))
                    del self._arima_garch_models[oldest_key]
                
                self._arima_garch_models[cache_key] = (arima_fitted, garch_fitted)
            
            # 步骤4: 预测当前价差的均值（ARIMA）
            arima_forecast = arima_fitted.forecast(steps=1)
            # 处理不同的返回格式
            if hasattr(arima_forecast, 'iloc'):
                predicted_mean = arima_forecast.iloc[0]
            elif isinstance(arima_forecast, (list, np.ndarray)):
                predicted_mean = float(arima_forecast[0])
            else:
                predicted_mean = float(arima_forecast)
            
            # 步骤5: 预测当前价差的波动率（GARCH）
            garch_forecast = garch_fitted.forecast(horizon=1)
            predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, 0])
            
            # 验证预测结果
            if predicted_volatility <= 0 or np.isnan(predicted_volatility) or np.isnan(predicted_mean):
                # 预测结果无效，返回0
                return 0.0
            
            # 步骤6: 计算Z-score
            z_score = (current_spread - predicted_mean) / predicted_volatility
            
            return z_score
            
        except Exception as e:
            # 任何错误都返回0，不进行回退（策略独立）
            print(f"ARIMA-GARCH模型计算失败: {str(e)}")
            return 0.0
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"ARIMA-GARCH模型 (ARIMA{self.arima_order}, GARCH{self.garch_order})"
    
    def clear_cache(self):
        """清除模型缓存"""
        self._arima_garch_models.clear()


