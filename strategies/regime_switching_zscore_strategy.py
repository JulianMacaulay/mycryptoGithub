"""
Regime-Switching（状态转换）Z-score计算策略
使用马尔可夫状态转换模型识别市场状态，根据不同状态调整Z-score计算
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List, Optional, Tuple
import numpy as np

# 尝试导入statsmodels用于状态转换模型
try:
    # 尝试导入MarkovRegression（可能在某些版本中不可用）
    try:
        from statsmodels.tsa.regime_switching import MarkovRegression
        MARKOV_REGRESSION_AVAILABLE = True
    except ImportError:
        MARKOV_REGRESSION_AVAILABLE = False
    STATSMODELS_RS_AVAILABLE = True
except ImportError:
    STATSMODELS_RS_AVAILABLE = False
    MARKOV_REGRESSION_AVAILABLE = False
    print("警告: statsmodels未安装或版本过低，Regime-Switching功能将使用简化版本。可以使用: pip install statsmodels")


class RegimeSwitchingZScoreStrategy(BaseZScoreStrategy):
    """Regime-Switching（状态转换）Z-score计算策略"""
    
    def __init__(self, n_regimes: int = 2, min_data_length: int = 50, 
                 smoothing: bool = True, **kwargs):
        """
        初始化Regime-Switching策略
        
        Args:
            n_regimes: 状态数量（默认2，即高波动率状态和低波动率状态）
            min_data_length: 最小数据长度要求
            smoothing: 是否使用平滑概率（默认True）
            **kwargs: 其他策略参数
        """
        super().__init__(n_regimes=n_regimes, min_data_length=min_data_length,
                        smoothing=smoothing, **kwargs)
        self.name = "Regime-Switching市场状态模型"
        self.n_regimes = n_regimes
        self.min_data_length = min_data_length
        self.smoothing = smoothing
        
        # 检查库是否可用（即使MarkovRegression不可用，也可以使用简化版本）
        if not STATSMODELS_RS_AVAILABLE:
            raise ImportError("statsmodels库不可用，无法使用Regime-Switching策略")
        
        # 存储模型缓存
        self._regime_models = {}
        self._regime_params_cache = {}
        self._max_cache_size = 10
    
    def _fit_regime_switching_model(self, spreads: np.ndarray) -> Optional[object]:
        """
        拟合马尔可夫状态转换模型
        
        使用MarkovRegression模型，假设价差在不同状态下的均值和方差不同
        
        Args:
            spreads: 价差序列
            
        Returns:
            拟合的模型对象，如果失败返回None
        """
        # 如果MarkovRegression不可用，直接使用简化版本
        if not MARKOV_REGRESSION_AVAILABLE:
            return self._fit_simplified_regime_model(spreads)
        
        try:
            # 使用MarkovRegression模型
            # 假设价差在不同状态下的均值和方差不同
            model = MarkovRegression(spreads, k_regimes=self.n_regimes, 
                                     switching_variance=True, switching_mean=True)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # 如果MarkovRegression拟合失败，使用简化版本
            return self._fit_simplified_regime_model(spreads)
    
    def _fit_simplified_regime_model(self, spreads: np.ndarray) -> Optional[dict]:
        """
        拟合简化版状态转换模型（当MarkovRegression不可用时）
        
        使用K-means聚类识别状态，然后估计每个状态的参数
        
        Args:
            spreads: 价差序列
            
        Returns:
            包含状态参数的字典，如果失败返回None
        """
        try:
            # 计算价差的一阶差分（用于识别波动率状态）
            diff_spreads = np.diff(spreads)
            abs_diff = np.abs(diff_spreads)
            
            # 使用简单的阈值方法识别状态
            # 状态0：低波动率（abs_diff < 中位数）
            # 状态1：高波动率（abs_diff >= 中位数）
            threshold = np.median(abs_diff)
            
            # 识别状态
            states = (abs_diff >= threshold).astype(int)
            
            # 为每个状态估计参数
            regime_params = {}
            for regime in range(self.n_regimes):
                regime_mask = states == regime
                if np.sum(regime_mask) < 5:  # 至少需要5个数据点
                    # 如果某个状态数据不足，使用全部数据
                    regime_spreads = spreads
                else:
                    # 获取该状态对应的价差
                    regime_indices = np.where(regime_mask)[0] + 1  # +1因为diff后索引偏移
                    regime_spreads = spreads[regime_indices]
                
                regime_params[regime] = {
                    'mean': np.mean(regime_spreads),
                    'std': np.std(regime_spreads) if len(regime_spreads) > 1 else np.std(spreads)
                }
            
            # 估计状态转换概率（简化版：使用历史频率）
            transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
            for i in range(len(states) - 1):
                transition_matrix[states[i], states[i+1]] += 1
            
            # 归一化
            row_sums = transition_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            return {
                'regime_params': regime_params,
                'transition_matrix': transition_matrix,
                'current_state': states[-1] if len(states) > 0 else 0
            }
        except Exception:
            return None
    
    def _estimate_current_regime(self, spreads: np.ndarray, 
                                 model: Optional[object] = None) -> Tuple[int, dict]:
        """
        估计当前市场状态
        
        Args:
            spreads: 价差序列
            model: 拟合的模型对象（可选）
            
        Returns:
            Tuple[int, dict]: (当前状态, 状态参数字典)
        """
        if model is None:
            # 使用简化方法
            simplified_model = self._fit_simplified_regime_model(spreads)
            if simplified_model is None:
                # 估计失败，使用默认状态
                return (0, {'mean': np.mean(spreads), 'std': np.std(spreads)})
            
            current_state = simplified_model['current_state']
            regime_params = simplified_model['regime_params']
            return (current_state, regime_params.get(current_state, 
                   {'mean': np.mean(spreads), 'std': np.std(spreads)}))
        
        try:
            # 使用MarkovRegression模型
            if hasattr(model, 'smoothed_marginal_probabilities'):
                # 获取平滑概率
                smoothed_probs = model.smoothed_marginal_probabilities
                # 当前状态是概率最大的状态
                current_state = np.argmax(smoothed_probs[-1, :])
            elif hasattr(model, 'filtered_marginal_probabilities'):
                # 使用滤波概率
                filtered_probs = model.filtered_marginal_probabilities
                current_state = np.argmax(filtered_probs[-1, :])
            else:
                # 使用简化方法
                return self._estimate_current_regime(spreads, None)
            
            # 获取当前状态的参数
            if hasattr(model, 'params'):
                # 从模型参数中提取状态相关的参数
                # MarkovRegression的参数结构：[均值参数, 方差参数, 转换概率]
                n_params_per_regime = len(model.params) // (2 * self.n_regimes + self.n_regimes * (self.n_regimes - 1))
                # 简化处理：使用历史数据估计
                regime_params = {}
                for regime in range(self.n_regimes):
                    # 使用简化方法估计每个状态的参数
                    regime_params[regime] = {
                        'mean': np.mean(spreads),
                        'std': np.std(spreads)
                    }
                
                # 使用当前状态的参数
                current_regime_params = regime_params.get(current_state, 
                    {'mean': np.mean(spreads), 'std': np.std(spreads)})
            else:
                # 使用简化方法
                return self._estimate_current_regime(spreads, None)
            
            return (current_state, current_regime_params)
            
        except Exception:
            # 模型使用失败，使用简化方法
            return self._estimate_current_regime(spreads, None)
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用Regime-Switching模型计算Z-score
        
        方法：
        1. 拟合马尔可夫状态转换模型，识别市场状态
        2. 估计当前市场状态
        3. 根据当前状态使用对应的均值和标准差计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            historical_prices1: 第一个资产的历史价格序列（可选，当前未使用）
            historical_prices2: 第二个资产的历史价格序列（可选，当前未使用）
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=self.min_data_length):
            return 0.0
        
        try:
            # 转换为numpy数组
            spreads_array = np.array(historical_spreads)
            
            # 创建缓存键
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._regime_models:
                model = self._regime_models[cache_key]
                regime_params = self._regime_params_cache.get(cache_key)
            else:
                # 拟合状态转换模型
                model = self._fit_regime_switching_model(spreads_array)
                
                if model is None:
                    # 模型拟合失败，使用传统方法
                    historical_mean = np.mean(spreads_array)
                    historical_std = np.std(spreads_array)
                    if historical_std > 0:
                        return (current_spread - historical_mean) / historical_std
                    return 0.0
                
                # 估计状态参数
                current_state, regime_params = self._estimate_current_regime(spreads_array, model)
                
                # 缓存模型和参数
                if len(self._regime_models) >= self._max_cache_size:
                    oldest_key = next(iter(self._regime_models))
                    del self._regime_models[oldest_key]
                    if oldest_key in self._regime_params_cache:
                        del self._regime_params_cache[oldest_key]
                
                self._regime_models[cache_key] = model
                self._regime_params_cache[cache_key] = regime_params
            
            # 如果regime_params是None，重新估计
            if regime_params is None:
                current_state, regime_params = self._estimate_current_regime(spreads_array, model)
                self._regime_params_cache[cache_key] = regime_params
            
            # 获取当前状态的均值和标准差
            regime_mean = regime_params.get('mean', np.mean(spreads_array))
            regime_std = regime_params.get('std', np.std(spreads_array))
            
            # 验证标准差
            if regime_std <= 0 or np.isnan(regime_std):
                regime_std = np.std(spreads_array)
                if regime_std <= 0:
                    return 0.0
            
            # 计算Z-score（使用当前状态的参数）
            z_score = (current_spread - regime_mean) / regime_std
            
            # 验证结果
            if np.isnan(z_score) or np.isinf(z_score):
                # 结果无效，使用传统方法
                historical_mean = np.mean(spreads_array)
                historical_std = np.std(spreads_array)
                if historical_std > 0:
                    z_score = (current_spread - historical_mean) / historical_std
                else:
                    return 0.0
            
            return z_score
            
        except Exception as e:
            # 任何错误都返回0，并尝试使用传统方法作为后备
            try:
                spreads_array = np.array(historical_spreads)
                historical_mean = np.mean(spreads_array)
                historical_std = np.std(spreads_array)
                if historical_std > 0:
                    return (current_spread - historical_mean) / historical_std
            except:
                pass
            print(f"Regime-Switching模型计算失败: {str(e)}")
            return 0.0
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"Regime-Switching市场状态模型 ({self.n_regimes}个状态)"
    
    def clear_cache(self):
        """清除模型缓存"""
        self._regime_models.clear()
        self._regime_params_cache.clear()

