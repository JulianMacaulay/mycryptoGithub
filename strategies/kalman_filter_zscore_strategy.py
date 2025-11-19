"""
Kalman Filter（卡尔曼滤波）Z-score计算策略
使用Kalman Filter动态估计价差的均值和方差，提高Z-score计算的准确性
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List, Optional
import numpy as np


class KalmanFilterZScoreStrategy(BaseZScoreStrategy):
    """Kalman Filter（卡尔曼滤波）Z-score计算策略"""
    
    def __init__(self, process_variance: float = 0.01, observation_variance: float = 0.1, 
                 min_data_length: int = 30, **kwargs):
        """
        初始化Kalman Filter策略
        
        Args:
            process_variance: 过程噪声方差（状态转移的不确定性，默认0.01）
            observation_variance: 观测噪声方差（观测的不确定性，默认0.1）
            min_data_length: 最小数据长度要求
            **kwargs: 其他策略参数
        """
        super().__init__(process_variance=process_variance, 
                        observation_variance=observation_variance,
                        min_data_length=min_data_length, **kwargs)
        self.name = "Kalman Filter动态价差模型"
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        self.min_data_length = min_data_length
        
        # Kalman Filter状态
        # state_mean: 价差的估计均值
        # state_covariance: 估计的不确定性（协方差）
        self._state_mean = None
        self._state_covariance = None
        
        # 存储历史估计值（用于计算Z-score）
        self._estimated_spreads = []
        self._estimated_std = None
    
    def _kalman_filter_step(self, observation: float, 
                           state_mean: float, state_covariance: float) -> tuple:
        """
        执行一步Kalman Filter更新
        
        Kalman Filter模型：
        - 状态方程：x_t = x_{t-1} + w_t  (w_t ~ N(0, Q))
        - 观测方程：y_t = x_t + v_t      (v_t ~ N(0, R))
        
        其中：
        - x_t: 价差的真实值（状态）
        - y_t: 观测到的价差
        - Q: 过程噪声方差（process_variance）
        - R: 观测噪声方差（observation_variance）
        
        Args:
            observation: 当前观测到的价差
            state_mean: 当前状态均值估计
            state_covariance: 当前状态协方差估计
            
        Returns:
            tuple: (更新后的状态均值, 更新后的状态协方差)
        """
        # 预测步骤（Predict）
        # 状态转移：x_{t|t-1} = x_{t-1|t-1}
        predicted_mean = state_mean
        # 协方差更新：P_{t|t-1} = P_{t-1|t-1} + Q
        predicted_covariance = state_covariance + self.process_variance
        
        # 更新步骤（Update）
        # 观测残差：y_t - x_{t|t-1}
        innovation = observation - predicted_mean
        # 残差协方差：S = P_{t|t-1} + R
        innovation_covariance = predicted_covariance + self.observation_variance
        
        # Kalman增益：K = P_{t|t-1} / S
        if innovation_covariance <= 0:
            kalman_gain = 0.0
        else:
            kalman_gain = predicted_covariance / innovation_covariance
        
        # 更新状态估计：x_{t|t} = x_{t|t-1} + K * (y_t - x_{t|t-1})
        updated_mean = predicted_mean + kalman_gain * innovation
        # 更新协方差：P_{t|t} = (1 - K) * P_{t|t-1}
        updated_covariance = (1 - kalman_gain) * predicted_covariance
        
        return updated_mean, updated_covariance
    
    def _initialize_kalman_filter(self, initial_observations: List[float]):
        """
        初始化Kalman Filter状态
        
        Args:
            initial_observations: 初始观测值序列
        """
        if not initial_observations:
            # 使用默认值
            self._state_mean = 0.0
            self._state_covariance = 1.0
        else:
            # 使用初始观测值的均值和方差初始化
            self._state_mean = np.mean(initial_observations)
            initial_std = np.std(initial_observations)
            self._state_covariance = max(initial_std ** 2, self.observation_variance)
        
        # 重置历史估计值
        self._estimated_spreads = []
        self._estimated_std = None
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用Kalman Filter计算Z-score
        
        Kalman Filter用于动态估计价差的均值和方差：
        1. 使用Kalman Filter逐步更新价差的估计值
        2. 基于估计值计算动态均值和标准差
        3. 使用当前价差和动态统计量计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=self.min_data_length):
            # 数据不足时返回0
            return 0.0
        
        try:
            # 转换为numpy数组
            spreads_array = np.array(historical_spreads)
            
            # 如果状态未初始化，或者历史数据发生变化，重新初始化
            if (self._state_mean is None or 
                len(self._estimated_spreads) != len(historical_spreads) - 1):
                # 使用前一半数据初始化
                init_length = max(10, len(historical_spreads) // 2)
                initial_obs = historical_spreads[:init_length]
                self._initialize_kalman_filter(initial_obs)
                
                # 对初始化后的数据进行滤波
                for obs in historical_spreads[init_length:-1]:
                    self._state_mean, self._state_covariance = self._kalman_filter_step(
                        obs, self._state_mean, self._state_covariance
                    )
                    self._estimated_spreads.append(self._state_mean)
            else:
                # 只处理新增的数据点
                new_observations = historical_spreads[len(self._estimated_spreads):-1]
                for obs in new_observations:
                    self._state_mean, self._state_covariance = self._kalman_filter_step(
                        obs, self._state_mean, self._state_covariance
                    )
                    self._estimated_spreads.append(self._state_mean)
            
            # 使用Kalman Filter更新当前观测
            updated_mean, updated_covariance = self._kalman_filter_step(
                current_spread, self._state_mean, self._state_covariance
            )
            
            # 更新状态
            self._state_mean = updated_mean
            self._state_covariance = updated_covariance
            
            # 计算动态标准差
            # 方法1：使用估计值的标准差
            if len(self._estimated_spreads) >= 10:
                estimated_array = np.array(self._estimated_spreads)
                self._estimated_std = np.std(estimated_array)
            else:
                # 方法2：使用协方差的平方根（如果估计值不足）
                self._estimated_std = np.sqrt(max(self._state_covariance, self.observation_variance))
            
            # 如果标准差为0或无效，使用历史数据的标准差
            if self._estimated_std <= 0 or np.isnan(self._estimated_std):
                historical_std = np.std(spreads_array)
                if historical_std > 0:
                    self._estimated_std = historical_std
                else:
                    return 0.0
            
            # 计算Z-score：使用Kalman Filter估计的均值
            z_score = (current_spread - self._state_mean) / self._estimated_std
            
            # 验证结果
            if np.isnan(z_score) or np.isinf(z_score):
                # 如果结果无效，使用传统方法作为后备
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
            print(f"Kalman Filter计算失败: {str(e)}")
            return 0.0
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"Kalman Filter动态价差模型 (过程方差={self.process_variance:.4f}, 观测方差={self.observation_variance:.4f})"
    
    def clear_cache(self):
        """清除模型缓存和状态"""
        self._state_mean = None
        self._state_covariance = None
        self._estimated_spreads = []
        self._estimated_std = None


