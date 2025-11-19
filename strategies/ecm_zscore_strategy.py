"""
ECM（误差修正模型）Z-score计算策略
使用误差修正模型来预测价差的均值回归，提高Z-score计算的准确性
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List, Optional
import numpy as np

# 尝试导入statsmodels用于ECM模型
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("警告: statsmodels未安装，ECM功能将不可用。可以使用: pip install statsmodels")


class EcmZScoreStrategy(BaseZScoreStrategy):
    """ECM（误差修正模型）Z-score计算策略"""
    
    def __init__(self, ecm_lag: int = 1, min_data_length: int = 30, **kwargs):
        """
        初始化ECM策略
        
        Args:
            ecm_lag: 误差修正项的滞后阶数（默认1，即使用前一期）
            min_data_length: 最小数据长度要求
            **kwargs: 其他策略参数
        """
        super().__init__(ecm_lag=ecm_lag, min_data_length=min_data_length, **kwargs)
        self.name = "ECM误差修正模型"
        self.ecm_lag = ecm_lag
        self.min_data_length = min_data_length
        
        # 检查库是否可用
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels库不可用，无法使用ECM策略")
        
        # 存储模型参数缓存
        self._ecm_params_cache = {}
        self._max_cache_size = 10
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用ECM模型计算Z-score
        
        ECM模型思想：
        1. 如果两个序列是协整的，它们之间存在长期均衡关系
        2. 短期偏离会通过误差修正项回归到长期均衡
        3. ECM模型：Δspread_t = α + β*ECM_{t-1} + ε_t
        4. 其中 ECM_{t-1} = spread_{t-1} - mean(spread) 是误差修正项
        
        对于Z-score计算：
        1. 计算历史价差的长期均值（均衡值）
        2. 计算误差修正项 ECM = spread_{t-1} - mean
        3. 使用ECM模型预测价差的均值回归
        4. 结合历史波动率计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        min_required_length = max(self.min_data_length, self.ecm_lag + 10)
        if not self.validate_input(historical_spreads, min_length=min_required_length):
            # 数据不足时返回0
            return 0.0
        
        try:
            # 转换为numpy数组
            spreads_array = np.array(historical_spreads)
            
            # 计算长期均值（均衡值）
            long_term_mean = np.mean(spreads_array)
            
            # 计算历史波动率（用于标准化）
            historical_std = np.std(spreads_array)
            
            if historical_std <= 0:
                # 波动率为0，无法计算Z-score
                return 0.0
            
            # 创建缓存键（基于数据长度和最后几个值）
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._ecm_params_cache:
                ecm_coefficient = self._ecm_params_cache[cache_key]
            else:
                # 计算误差修正项序列
                ecm_terms = spreads_array[:-1] - long_term_mean  # ECM_{t-1}
                
                # 计算价差的一阶差分
                diff_spreads = np.diff(spreads_array)  # Δspread_t
                
                # 确保长度匹配
                min_len = min(len(ecm_terms), len(diff_spreads))
                if min_len < 10:
                    # 数据不足，使用简单方法
                    ecm_coefficient = -0.1  # 默认的均值回归系数
                else:
                    # 构建ECM模型：Δspread_t = α + β*ECM_{t-1} + ε_t
                    ecm_terms_aligned = ecm_terms[-min_len:]
                    diff_spreads_aligned = diff_spreads[-min_len:]
                    
                    # 使用OLS回归估计ECM系数
                    X = add_constant(ecm_terms_aligned.reshape(-1, 1))
                    y = diff_spreads_aligned
                    
                    try:
                        model = OLS(y, X).fit()
                        ecm_coefficient = model.params[1]  # β系数（误差修正系数）
                        
                        # 验证系数合理性（应该在-1到0之间，表示均值回归）
                        if ecm_coefficient > 0 or ecm_coefficient < -1:
                            # 系数不合理，使用默认值
                            ecm_coefficient = -0.1
                    except Exception:
                        # 回归失败，使用默认值
                        ecm_coefficient = -0.1
                
                # 缓存参数（限制缓存大小）
                if len(self._ecm_params_cache) >= self._max_cache_size:
                    # 清除最旧的缓存
                    oldest_key = next(iter(self._ecm_params_cache))
                    del self._ecm_params_cache[oldest_key]
                
                self._ecm_params_cache[cache_key] = ecm_coefficient
            
            # 计算当前误差修正项
            if len(historical_spreads) > self.ecm_lag:
                ecm_current = historical_spreads[-self.ecm_lag] - long_term_mean
            else:
                ecm_current = historical_spreads[-1] - long_term_mean
            
            # 使用ECM模型预测价差的均值回归
            # 预测的价差变化：Δspread_predicted = β*ECM_{t-1}
            predicted_change = ecm_coefficient * ecm_current
            
            # 预测的价差均值（基于均值回归）
            predicted_mean = long_term_mean - predicted_change
            
            # 验证预测结果
            if np.isnan(predicted_mean):
                # 预测结果无效，使用长期均值
                predicted_mean = long_term_mean
            
            # 计算Z-score
            z_score = (current_spread - predicted_mean) / historical_std
            
            return z_score
            
        except Exception as e:
            # 任何错误都返回0
            print(f"ECM模型计算失败: {str(e)}")
            return 0.0
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"ECM误差修正模型 (滞后阶数={self.ecm_lag})"
    
    def clear_cache(self):
        """清除模型缓存"""
        self._ecm_params_cache.clear()


