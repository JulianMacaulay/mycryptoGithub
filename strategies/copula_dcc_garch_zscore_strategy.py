"""
Copula + DCC-GARCH Z-score计算策略（完整版）
使用DCC-GARCH模型估计两个资产的动态波动率和相关性，使用Copula建模依赖结构
"""

from .base_zscore_strategy import BaseZScoreStrategy
from typing import List, Optional, Tuple
import numpy as np

# 尝试导入GARCH相关库
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False
    print("警告: arch未安装，GARCH功能将不可用。可以使用: pip install arch")

# 尝试导入Copula相关库
try:
    from scipy.stats import norm, t
    from scipy.special import ndtri
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy未安装，Copula功能将不可用。可以使用: pip install scipy")


class CopulaDccGarchZScoreStrategy(BaseZScoreStrategy):
    """Copula + DCC-GARCH Z-score计算策略（完整版）"""
    
    def __init__(self, garch_order: Tuple[int, int] = (1, 1), 
                 copula_type: str = 'gaussian', min_data_length: int = 50, **kwargs):
        """
        初始化Copula + DCC-GARCH策略
        
        Args:
            garch_order: GARCH模型阶数 (p, q)，默认(1,1)
            copula_type: Copula类型，可选'gaussian'（高斯Copula）或'student'（t-Copula），默认'gaussian'
            min_data_length: 最小数据长度要求
            **kwargs: 其他策略参数
        """
        super().__init__(garch_order=garch_order, copula_type=copula_type,
                        min_data_length=min_data_length, **kwargs)
        self.name = "Copula + DCC-GARCH模型（完整版）"
        self.garch_order = garch_order
        self.copula_type = copula_type
        self.min_data_length = min_data_length
        
        # 检查库是否可用
        if not GARCH_AVAILABLE:
            raise ImportError("arch库不可用，无法使用DCC-GARCH策略")
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy库不可用，无法使用Copula功能")
        
        # 存储模型缓存
        self._garch_models = {}  # 存储两个资产的GARCH模型
        self._dcc_params_cache = {}  # 存储DCC参数
        self._copula_params_cache = {}  # 存储Copula参数
        self._max_cache_size = 10
    
    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        计算收益率序列（对数收益率）
        
        Args:
            prices: 价格序列
            
        Returns:
            np.ndarray: 收益率序列
        """
        # 使用对数收益率：r_t = log(P_t / P_{t-1})
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        return returns
    
    def _fit_garch_model(self, returns: np.ndarray) -> Optional[object]:
        """
        拟合GARCH模型
        
        Args:
            returns: 收益率序列
            
        Returns:
            拟合的GARCH模型对象，如果失败返回None
        """
        try:
            garch_model = arch_model(returns, vol='Garch', 
                                    p=self.garch_order[0], q=self.garch_order[1])
            garch_fitted = garch_model.fit(disp='off')
            return garch_fitted
        except Exception:
            return None
    
    def _estimate_dcc_parameters(self, returns1: np.ndarray, returns2: np.ndarray,
                                 garch_fitted1: object, garch_fitted2: object) -> Tuple[float, float]:
        """
        估计DCC（动态条件相关性）模型参数
        
        DCC模型：
        Q_t = (1 - α - β) * Q_bar + α * (ε_{t-1} * ε_{t-1}') + β * Q_{t-1}
        R_t = (diag(Q_t))^{-1/2} * Q_t * (diag(Q_t))^{-1/2}
        
        其中：
        - Q_t: 条件协方差矩阵
        - R_t: 条件相关系数矩阵
        - ε_t: 标准化残差
        - α, β: DCC参数（α + β < 1）
        
        简化实现：使用MLE估计α和β
        
        Args:
            returns1: 第一个资产的收益率序列
            returns2: 第二个资产的收益率序列
            garch_fitted1: 第一个资产的GARCH模型
            garch_fitted2: 第二个资产的GARCH模型
            
        Returns:
            Tuple[float, float]: (α, β) DCC参数
        """
        if len(returns1) != len(returns2) or len(returns1) < 30:
            # 数据不足，使用默认值
            return (0.05, 0.90)
        
        try:
            # 获取标准化残差
            # 从GARCH模型获取条件波动率
            conditional_vol1 = garch_fitted1.conditional_volatility
            conditional_vol2 = garch_fitted2.conditional_volatility
            
            # 标准化残差：ε_t = r_t / σ_t
            standardized_residuals1 = returns1 / (conditional_vol1 + 1e-8)
            standardized_residuals2 = returns2 / (conditional_vol2 + 1e-8)
            
            # 计算无条件协方差矩阵 Q_bar
            # Q_bar = E[ε_t * ε_t']
            Q_bar = np.cov(standardized_residuals1, standardized_residuals2)
            q_bar_11 = Q_bar[0, 0]  # 应该是1（标准化后）
            q_bar_22 = Q_bar[1, 1]  # 应该是1（标准化后）
            q_bar_12 = Q_bar[0, 1]  # 无条件相关系数
            
            # 初始化Q矩阵
            Q_t = np.array([[1.0, q_bar_12], [q_bar_12, 1.0]])
            
            # 使用简化的MLE估计α和β
            # 目标：最大化对数似然函数
            # 简化：使用网格搜索找到最优参数
            best_alpha, best_beta = 0.05, 0.90
            best_loglik = -np.inf
            
            # 网格搜索
            alpha_candidates = np.linspace(0.01, 0.15, 10)
            beta_candidates = np.linspace(0.80, 0.95, 10)
            
            for alpha in alpha_candidates:
                for beta in beta_candidates:
                    if alpha + beta >= 1.0:
                        continue
                    
                    # 计算对数似然
                    loglik = 0.0
                    Q_t_current = Q_t.copy()
                    
                    for t in range(1, len(returns1)):
                        # 更新Q矩阵
                        epsilon_t = np.array([standardized_residuals1[t-1], 
                                            standardized_residuals2[t-1]])
                        Q_t_current = (1 - alpha - beta) * np.array([[1.0, q_bar_12], 
                                                                      [q_bar_12, 1.0]]) + \
                                      alpha * np.outer(epsilon_t, epsilon_t) + \
                                      beta * Q_t_current
                        
                        # 计算条件相关系数矩阵 R_t
                        diag_inv = 1.0 / np.sqrt(np.diag(Q_t_current))
                        R_t = np.diag(diag_inv) @ Q_t_current @ np.diag(diag_inv)
                        
                        # 计算对数似然（简化版）
                        det_R = np.linalg.det(R_t)
                        if det_R > 0:
                            loglik += -0.5 * np.log(det_R)
                    
                    if loglik > best_loglik:
                        best_loglik = loglik
                        best_alpha = alpha
                        best_beta = beta
            
            return (best_alpha, best_beta)
            
        except Exception:
            # 估计失败，使用默认值
            return (0.05, 0.90)
    
    def _estimate_dcc_correlation(self, returns1: np.ndarray, returns2: np.ndarray,
                                  garch_fitted1: object, garch_fitted2: object,
                                  dcc_alpha: float, dcc_beta: float) -> float:
        """
        使用DCC模型估计当前时刻的动态相关系数
        
        Args:
            returns1: 第一个资产的收益率序列
            returns2: 第二个资产的收益率序列
            garch_fitted1: 第一个资产的GARCH模型
            garch_fitted2: 第二个资产的GARCH模型
            dcc_alpha: DCC参数α
            dcc_beta: DCC参数β
            
        Returns:
            float: 当前时刻的动态相关系数
        """
        try:
            # 获取标准化残差
            conditional_vol1 = garch_fitted1.conditional_volatility
            conditional_vol2 = garch_fitted2.conditional_volatility
            
            standardized_residuals1 = returns1 / (conditional_vol1 + 1e-8)
            standardized_residuals2 = returns2 / (conditional_vol2 + 1e-8)
            
            # 计算无条件协方差矩阵
            Q_bar = np.cov(standardized_residuals1, standardized_residuals2)
            q_bar_12 = Q_bar[0, 1]
            
            # 初始化Q矩阵
            Q_t = np.array([[1.0, q_bar_12], [q_bar_12, 1.0]])
            
            # 递归更新Q矩阵
            for t in range(1, len(returns1)):
                epsilon_t = np.array([standardized_residuals1[t-1], 
                                    standardized_residuals2[t-1]])
                Q_t = (1 - dcc_alpha - dcc_beta) * np.array([[1.0, q_bar_12], 
                                                              [q_bar_12, 1.0]]) + \
                      dcc_alpha * np.outer(epsilon_t, epsilon_t) + \
                      dcc_beta * Q_t
            
            # 计算条件相关系数矩阵 R_t
            diag_inv = 1.0 / np.sqrt(np.diag(Q_t))
            R_t = np.diag(diag_inv) @ Q_t @ np.diag(diag_inv)
            
            # 返回相关系数
            correlation = R_t[0, 1]
            
            # 确保相关性在有效范围内
            if np.isnan(correlation) or np.isinf(correlation):
                correlation = q_bar_12
            correlation = np.clip(correlation, -0.99, 0.99)
            
            return correlation
            
        except Exception:
            # 计算失败，使用简单相关系数
            return np.corrcoef(returns1, returns2)[0, 1] if len(returns1) >= 2 else 0.0
    
    def _estimate_copula_parameter(self, returns1: np.ndarray, returns2: np.ndarray) -> Tuple[float, Optional[float]]:
        """
        估计Copula参数
        
        对于高斯Copula，参数是相关系数
        对于t-Copula，需要估计相关系数和自由度
        
        Args:
            returns1: 第一个资产的收益率序列
            returns2: 第二个资产的收益率序列
            
        Returns:
            Tuple[float, Optional[float]]: (相关系数, 自由度) 对于t-Copula，自由度不为None
        """
        if len(returns1) != len(returns2) or len(returns1) < 10:
            return (0.0, None)
        
        try:
            # 转换为标准正态分布（使用经验分布函数）
            # 使用排序位置估计边际分布
            ranks1 = np.argsort(np.argsort(returns1)) / (len(returns1) - 1)
            ranks2 = np.argsort(np.argsort(returns2)) / (len(returns2) - 1)
            
            # 转换为标准正态分布
            u1 = norm.ppf(np.clip(ranks1, 0.001, 0.999))
            u2 = norm.ppf(np.clip(ranks2, 0.001, 0.999))
            
            # 估计相关系数（高斯Copula参数）
            correlation = np.corrcoef(u1, u2)[0, 1]
            
            if np.isnan(correlation) or np.isinf(correlation):
                correlation = 0.0
            correlation = np.clip(correlation, -0.99, 0.99)
            
            # 如果是t-Copula，还需要估计自由度
            if self.copula_type == 'student':
                # 使用最大似然估计自由度（简化版）
                # 尝试不同的自由度值，选择使对数似然最大的
                best_df = 5.0
                best_loglik = -np.inf
                
                for df in [3, 4, 5, 6, 7, 8, 9, 10]:
                    try:
                        # 计算t-Copula的对数似然（简化版）
                        # 实际应该使用完整的t-Copula密度函数
                        loglik = -0.5 * (df + 2) * np.sum(np.log(1 + (u1**2 + u2**2 - 2*correlation*u1*u2) / (df * (1 - correlation**2))))
                        if loglik > best_loglik:
                            best_loglik = loglik
                            best_df = float(df)
                    except:
                        continue
                
                return (correlation, best_df)
            else:
                return (correlation, None)
                
        except Exception:
            return (0.0, None)
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用Copula + DCC-GARCH模型计算Z-score（完整版）
        
        方法：
        1. 获取两个资产的价格序列，计算收益率序列
        2. 对每个资产拟合GARCH模型，估计动态波动率
        3. 使用DCC模型估计两个资产之间的动态相关性
        4. 使用Copula建模两个资产之间的依赖结构
        5. 基于动态波动率和相关性计算价差的动态方差
        6. 计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            historical_prices1: 第一个资产的历史价格序列（必需）
            historical_prices2: 第二个资产的历史价格序列（必需）
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=self.min_data_length):
            return 0.0
        
        # 检查是否有价格数据
        if historical_prices1 is None or historical_prices2 is None:
            # 如果没有价格数据，回退到简化版本（基于价差收益率）
            return self._calculate_z_score_simplified(current_spread, historical_spreads)
        
        if len(historical_prices1) != len(historical_prices2) or \
           len(historical_prices1) != len(historical_spreads):
            # 数据长度不匹配
            return self._calculate_z_score_simplified(current_spread, historical_spreads)
        
        try:
            # 转换为numpy数组
            prices1_array = np.array(historical_prices1)
            prices2_array = np.array(historical_prices2)
            spreads_array = np.array(historical_spreads)
            
            # 计算两个资产的收益率序列
            returns1 = self._calculate_returns(prices1_array)
            returns2 = self._calculate_returns(prices2_array)
            
            if len(returns1) < max(30, sum(self.garch_order) + 10):
                # 数据不足，使用传统方法
                return self._calculate_z_score_simplified(current_spread, historical_spreads)
            
            # 创建缓存键
            cache_key = (len(prices1_array), tuple(prices1_array[-3:]), tuple(prices2_array[-3:]))
            
            # 检查缓存
            if cache_key in self._garch_models:
                garch_fitted1, garch_fitted2, dcc_alpha, dcc_beta, copula_param = self._garch_models[cache_key]
            else:
                # 步骤1: 对每个资产拟合GARCH模型
                garch_fitted1 = self._fit_garch_model(returns1)
                garch_fitted2 = self._fit_garch_model(returns2)
                
                if garch_fitted1 is None or garch_fitted2 is None:
                    # GARCH拟合失败，使用传统方法
                    return self._calculate_z_score_simplified(current_spread, historical_spreads)
                
                # 步骤2: 估计DCC参数
                dcc_alpha, dcc_beta = self._estimate_dcc_parameters(returns1, returns2, 
                                                                    garch_fitted1, garch_fitted2)
                
                # 步骤3: 估计Copula参数
                copula_param, copula_df = self._estimate_copula_parameter(returns1, returns2)
                
                # 缓存模型
                if len(self._garch_models) >= self._max_cache_size:
                    oldest_key = next(iter(self._garch_models))
                    del self._garch_models[oldest_key]
                
                self._garch_models[cache_key] = (garch_fitted1, garch_fitted2, 
                                                dcc_alpha, dcc_beta, copula_param)
            
            # 步骤4: 预测动态波动率
            try:
                garch_forecast1 = garch_fitted1.forecast(horizon=1)
                garch_forecast2 = garch_fitted2.forecast(horizon=1)
                predicted_vol1 = np.sqrt(garch_forecast1.variance.values[-1, 0])
                predicted_vol2 = np.sqrt(garch_forecast2.variance.values[-1, 0])
            except Exception:
                # 预测失败，使用历史波动率
                predicted_vol1 = np.std(returns1)
                predicted_vol2 = np.std(returns2)
            
            # 验证波动率
            if predicted_vol1 <= 0 or np.isnan(predicted_vol1):
                predicted_vol1 = np.std(returns1)
            if predicted_vol2 <= 0 or np.isnan(predicted_vol2):
                predicted_vol2 = np.std(returns2)
            
            if predicted_vol1 <= 0 or predicted_vol2 <= 0:
                return self._calculate_z_score_simplified(current_spread, historical_spreads)
            
            # 步骤5: 估计动态相关系数（使用DCC模型）
            dcc_correlation = self._estimate_dcc_correlation(returns1, returns2, 
                                                            garch_fitted1, garch_fitted2,
                                                            dcc_alpha, dcc_beta)
            
            # 步骤6: 计算价差的动态方差
            # 假设价差 = price1 - hedge_ratio * price2
            # 价差的方差 = Var(price1) + hedge_ratio^2 * Var(price2) - 2 * hedge_ratio * Cov(price1, price2)
            # 对于收益率：Var(spread_return) = Var(return1) + hedge_ratio^2 * Var(return2) - 2 * hedge_ratio * Corr * Std(return1) * Std(return2)
            
            # 估计hedge_ratio（使用历史价差和价格）
            # 简化：使用OLS回归估计hedge_ratio
            try:
                # price1 = hedge_ratio * price2 + spread
                # 使用价格序列估计hedge_ratio
                X = prices2_array.reshape(-1, 1)
                y = prices1_array
                hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
            except:
                # 估计失败，使用1.0作为默认值
                hedge_ratio = 1.0
            
            # 计算价差的动态方差
            spread_variance = predicted_vol1**2 + (hedge_ratio**2) * predicted_vol2**2 - \
                            2 * hedge_ratio * dcc_correlation * predicted_vol1 * predicted_vol2
            
            # 使用Copula调整方差（考虑尾部依赖）
            if self.copula_type == 'student' and copula_df is not None:
                # t-Copula有尾部依赖，可能需要调整方差
                # 简化处理：使用Copula参数作为调整因子
                copula_adjustment = 1.0 + 0.1 * abs(copula_param)
            else:
                # 高斯Copula
                copula_adjustment = 1.0 + 0.05 * abs(copula_param)
            
            adjusted_spread_variance = spread_variance * copula_adjustment
            spread_std = np.sqrt(max(adjusted_spread_variance, 1e-8))
            
            # 步骤7: 计算价差的动态均值
            historical_mean = np.mean(spreads_array)
            
            # 步骤8: 计算Z-score
            z_score = (current_spread - historical_mean) / spread_std
            
            # 验证结果
            if np.isnan(z_score) or np.isinf(z_score):
                return self._calculate_z_score_simplified(current_spread, historical_spreads)
            
            return z_score
            
        except Exception as e:
            # 任何错误都回退到简化版本
            print(f"Copula + DCC-GARCH模型计算失败: {str(e)}")
            return self._calculate_z_score_simplified(current_spread, historical_spreads)
    
    def _calculate_z_score_simplified(self, current_spread: float, historical_spreads: List[float]) -> float:
        """
        简化版Z-score计算（当没有价格数据时使用）
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        try:
            spreads_array = np.array(historical_spreads)
            historical_mean = np.mean(spreads_array)
            historical_std = np.std(spreads_array)
            
            if historical_std > 0:
                return (current_spread - historical_mean) / historical_std
            return 0.0
        except:
            return 0.0
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"Copula + DCC-GARCH模型（完整版） (GARCH{self.garch_order}, Copula={self.copula_type})"
    
    def clear_cache(self):
        """清除模型缓存"""
        self._garch_models.clear()
        self._dcc_params_cache.clear()
        self._copula_params_cache.clear()
