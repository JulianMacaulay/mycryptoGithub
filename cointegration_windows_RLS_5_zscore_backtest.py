#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级版协整交易回测与参数优化工具（保留RLS + Regime-Switching）
最新流程假设协整筛选已在外部完成（例如通过 cointegration_stable_pairs_screening.py），
本文件仅负责：
1. 读取用户输入的已筛选协整币对并计算/校准对冲比率
2. 使用RLS递归最小二乘在滑动窗口内动态更新对冲比率
3. 在回测过程中按指定窗口大小和步长周期性地复检协整关系
4. 提供多策略Z-score计算（传统、ARIMA-GARCH、ECM、Kalman、Copula+DCC-GARCH、Regime-Switching）
5. 支持参数优化（网格/随机/贝叶斯）以及稳定性测试
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# statsmodels 基础库可用性（用于ECM等）
STATSMODELS_AVAILABLE = True
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
import itertools
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')

# 尝试导入贝叶斯优化库（如果可用）
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("警告: scikit-optimize未安装，贝叶斯优化功能将不可用。可以使用: pip install scikit-optimize")

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

# 导入Z-score策略
try:
    from strategies import TraditionalZScoreStrategy, ArimaGarchZScoreStrategy, EcmZScoreStrategy, \
        KalmanFilterZScoreStrategy, CopulaDccGarchZScoreStrategy, RegimeSwitchingZScoreStrategy, BaseZScoreStrategy

    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    print("警告: 策略模块导入失败，将使用内置方法")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 基础函数（保持不变） ====================

def load_csv_data(csv_file_path):
    """
    从CSV文件加载数据

    Args:
        csv_file_path: CSV文件路径

    Returns:
        dict: 包含各币对数据的字典
    """
    try:
        print(f"正在加载CSV文件: {csv_file_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 检查数据格式
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 如果有symbol列，按币对分组
        if 'symbol' in df.columns:
            data = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
                symbol_data.set_index('timestamp', inplace=True)
                data[symbol] = symbol_data['close']
                print(f"币对 {symbol}: {len(symbol_data)} 条数据")
        else:
            # 假设是单个币对的数据
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            data = {'BTCUSDT': df['close']}
            print(f"单个币对数据: {len(df)} 条数据")

        return data

    except Exception as e:
        print(f"加载CSV文件失败: {str(e)}")
        return None


def calculate_hedge_ratio(price1, price2):
    """
    计算对冲比率（使用OLS回归）

    Args:
        price1: 第一个币种的价格序列
        price2: 第二个币种的价格序列

    Returns:
        float: 对冲比率
    """
    print("计算对冲比率...")

    # 确保两个序列长度一致
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]

    # 使用OLS回归计算对冲比率
    X = price2_aligned.values.reshape(-1, 1)
    y = price1_aligned.values

    # 添加常数项
    X_with_const = add_constant(X)

    # 执行回归
    model = OLS(y, X_with_const).fit()
    hedge_ratio = model.params[1]  # 斜率系数

    print(f"对冲比率: {hedge_ratio:.6f}")
    print(f"R²: {model.rsquared:.6f}")
    print(f"回归统计:")
    print(f"  截距: {model.params[0]:.6f}")
    print(f"  斜率: {model.params[1]:.6f}")
    print(f"  P值: {model.pvalues[1]:.6f}")

    return hedge_ratio


# ==================== RLS（递归最小二乘）类 ====================

class RecursiveLeastSquares:
    """
    递归最小二乘（RLS）类，用于动态更新对冲比率
    模型：price1_t = β_t * price2_t + ε_t
    """

    def __init__(self, lambda_forgetting=0.99, initial_covariance=1000.0, max_change_rate=0.2):
        """
        初始化RLS

        Args:
            lambda_forgetting: 遗忘因子（0 < λ ≤ 1），接近1表示更重视历史数据
            initial_covariance: 初始协方差矩阵的对角元素（大的正数）
            max_change_rate: 对冲比率最大变化率（防止突变）
        """
        self.lambda_forgetting = lambda_forgetting
        self.initial_covariance = initial_covariance
        self.max_change_rate = max_change_rate

        # RLS状态
        self.beta = None  # 对冲比率 [截距, 斜率]
        self.P = None  # 协方差矩阵
        self.initialized = False

        # 历史记录
        self.beta_history = []  # 历史对冲比率
        self.change_history = []  # 历史变化率

    def initialize(self, initial_price1, initial_price2):
        """
        使用初始数据初始化RLS

        Args:
            initial_price1: 初始价格序列1（用于OLS估计初始值）
            initial_price2: 初始价格序列2
        """
        if len(initial_price1) < 10 or len(initial_price2) < 10:
            raise ValueError("初始化需要至少10个数据点")

        # 使用OLS估计初始对冲比率
        min_length = min(len(initial_price1), len(initial_price2))
        price1_aligned = initial_price1.iloc[:min_length] if hasattr(initial_price1, 'iloc') else initial_price1[
            :min_length]
        price2_aligned = initial_price2.iloc[:min_length] if hasattr(initial_price2, 'iloc') else initial_price2[
            :min_length]

        X = price2_aligned.values.reshape(-1, 1)
        y = price1_aligned.values
        X_with_const = add_constant(X)

        model = OLS(y, X_with_const).fit()

        # 初始化参数：β = [截距, 斜率]
        self.beta = np.array([model.params[0], model.params[1]])

        # 初始化协方差矩阵
        self.P = np.eye(2) * self.initial_covariance

        self.initialized = True
        self.beta_history = [self.beta.copy()]
        self.change_history = [0.0]

        print(f"RLS初始化完成: 初始对冲比率 = {self.beta[1]:.6f}, 截距 = {self.beta[0]:.6f}")

    def update(self, price1_t, price2_t):
        """
        更新对冲比率（RLS更新步骤）

        Args:
            price1_t: 当前时刻价格1
            price2_t: 当前时刻价格2

        Returns:
            float: 更新后的对冲比率（斜率）
        """
        if not self.initialized:
            raise ValueError("RLS未初始化，请先调用initialize()")

        # 特征向量：x_t = [1, price2_t]
        x_t = np.array([1.0, price2_t])
        y_t = price1_t

        # 预测误差
        prediction = np.dot(x_t, self.beta)
        error = y_t - prediction

        # Kalman增益
        denominator = self.lambda_forgetting + np.dot(x_t, np.dot(self.P, x_t))
        if denominator <= 0:
            # 数值问题，跳过更新
            return self.beta[1]

        K_t = np.dot(self.P, x_t) / denominator

        # 更新参数
        beta_new = self.beta + K_t * error

        # 限制变化率（防止突变）
        if len(self.beta_history) > 0:
            beta_old = self.beta_history[-1]
            change_rate = abs((beta_new[1] - beta_old[1]) / (beta_old[1] + 1e-8))

            if change_rate > self.max_change_rate:
                # 限制变化率
                max_change = self.max_change_rate * abs(beta_old[1])
                if beta_new[1] > beta_old[1]:
                    beta_new[1] = beta_old[1] + max_change
                else:
                    beta_new[1] = beta_old[1] - max_change
                # 保持截距更新
                beta_new[0] = self.beta[0] + K_t[0] * error

        # 更新协方差矩阵
        self.P = (self.P - np.outer(K_t, np.dot(self.P, x_t))) / self.lambda_forgetting

        # 更新状态
        self.beta = beta_new
        self.beta_history.append(self.beta.copy())

        # 记录变化率
        if len(self.beta_history) > 1:
            change = abs((self.beta[1] - self.beta_history[-2][1]) / (self.beta_history[-2][1] + 1e-8))
            self.change_history.append(change)
        else:
            self.change_history.append(0.0)

        return self.beta[1]

    def get_hedge_ratio(self):
        """获取当前对冲比率（斜率）"""
        if not self.initialized:
            return None
        return self.beta[1]

    def get_stability_metric(self, window=50):
        """
        计算稳定性指标（对冲比率变化的标准差）

        Args:
            window: 计算窗口大小

        Returns:
            float: 稳定性指标（越小越稳定）
        """
        if len(self.beta_history) < window:
            return None

        recent_betas = [beta[1] for beta in self.beta_history[-window:]]
        return np.std(recent_betas)

    def get_change_rate(self, window=10):
        """
        计算最近的变化率

        Args:
            window: 计算窗口大小

        Returns:
            float: 平均变化率
        """
        if len(self.change_history) < window:
            return None

        recent_changes = self.change_history[-window:]
        return np.mean(recent_changes)

    def reset(self):
        """重置RLS状态"""
        self.beta = None
        self.P = None
        self.initialized = False
        self.beta_history = []
        self.change_history = []


def advanced_adf_test(series, max_lags=None, verbose=True):
    """
    执行增强的ADF检验
    Args:
        series: 时间序列
        max_lags: 最大滞后阶数
        verbose: 是否打印详细信息

    Returns:
        dict: ADF检验结果
    """
    if verbose:
        print("执行ADF检验...")

    try:
        # 执行ADF检验
        adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')

        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        used_lag = adf_result[2]

        if verbose:
            print(f"ADF检验结果:")
            print(f"  ADF统计量: {adf_statistic:.6f}")
            print(f"  P值: {p_value:.6f}")
            print(f"  使用的滞后阶数: {used_lag}")
            print(f"  临界值:")
            for level, value in critical_values.items():
                print(f"    {level}: {value:.6f}")

        # 判断是否平稳
        is_stationary = p_value < 0.05

        if verbose:
            print(f"  是否平稳: {'是' if is_stationary else '否'}")

        return {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'used_lag': used_lag,
            'is_stationary': is_stationary
        }

    except Exception as e:
        if verbose:
            print(f"ADF检验失败: {str(e)}")
        return None


def determine_integration_order(series, max_order=2):
    """
    确定序列的积分阶数

    Args:
        series: 时间序列
        max_order: 最大检查的积分阶数

    Returns:
        int: 积分阶数（0=I(0), 1=I(1), 2=I(2), None=无法确定）
    """
    # 检验原序列
    adf_result = advanced_adf_test(series, verbose=False)
    if adf_result and adf_result['is_stationary']:
        return 0  # I(0)

    # 检验一阶差分
    if max_order >= 1:
        diff1 = series.diff().dropna()
        if len(diff1) < 50:
            return None
        adf_result = advanced_adf_test(diff1, verbose=False)
        if adf_result and adf_result['is_stationary']:
            return 1  # I(1)

    # 检验二阶差分
    if max_order >= 2:
        diff2 = series.diff().diff().dropna()
        if len(diff2) < 50:
            return None
        adf_result = advanced_adf_test(diff2, verbose=False)
        if adf_result and adf_result['is_stationary']:
            return 2  # I(2)

    return None  # 无法确定


# ==================== 正确的协整检验代码 ====================

def enhanced_cointegration_test(price1, price2, symbol1, symbol2, verbose=True, diff_order=0):
    """
    正确的协整检验（Engle-Granger方法）

    步骤：
    1. 检验原序列price1和price2的平稳性
    2. 如果原序列不平稳，检验它们的一阶差分是否平稳
    3. 只有当两个原序列都是I(1)时，才能进行协整检验
    4. 先计算最优对冲比率（OLS）
    5. 根据diff_order计算价差：
       - diff_order=0: 计算原序列的价差（price1 - β*price2）
       - diff_order=1: 计算一阶差分价差（diff1 - β*diff2）
    6. 检验价差的平稳性
    7. 如果价差平稳，才是真正的协整

    Args:
        price1: 第一个价格序列
        price2: 第二个价格序列
        symbol1: 第一个币种名称
        symbol2: 第二个币种名称
        verbose: 是否打印详细信息
        diff_order: 价差类型，0=原始价差，1=一阶差分价差

    Returns:
        dict: 检验结果
    """
    if verbose:
        print(f"\n开始协整检验: {symbol1}/{symbol2}")
        print("=" * 60)

    results = {
        'pair_name': f"{symbol1}/{symbol2}",
        'symbol1': symbol1,
        'symbol2': symbol2,
        'price1_order': None,
        'price2_order': None,
        'hedge_ratio': None,
        'spread': None,
        'spread_adf': None,
        'cointegration_found': False,
        'best_test': None,
        'diff_order': diff_order  # 记录使用的价差类型
    }

    # 步骤1: 检验price1的积分阶数
    if verbose:
        print(f"\n--- 步骤1: 检验 {symbol1} 的积分阶数 ---")
    price1_order = determine_integration_order(price1, max_order=2)
    results['price1_order'] = price1_order

    if price1_order is None:
        if verbose:
            print(f"{symbol1} 的积分阶数无法确定，跳过协整检验")
        return results

    if price1_order == 0:
        if verbose:
            print(f"{symbol1} 是 I(0)（平稳序列），不能进行协整检验")
        return results

    if verbose:
        print(f"{symbol1} 是 I({price1_order})")

    # 步骤2: 检验price2的积分阶数
    if verbose:
        print(f"\n--- 步骤2: 检验 {symbol2} 的积分阶数 ---")
    price2_order = determine_integration_order(price2, max_order=2)
    results['price2_order'] = price2_order

    if price2_order is None:
        if verbose:
            print(f"{symbol2} 的积分阶数无法确定，跳过协整检验")
        return results

    if price2_order == 0:
        if verbose:
            print(f"{symbol2} 是 I(0)（平稳序列），不能进行协整检验")
        return results

    if verbose:
        print(f"{symbol2} 是 I({price2_order})")

    # 步骤3: 检查两个序列是否同阶单整
    if price1_order != price2_order:
        if verbose:
            print(f"{symbol1} 是 I({price1_order})，{symbol2} 是 I({price2_order})，积分阶数不同，不能协整")
        return results

    # 步骤4: 只有当两个序列都是I(1)时，才进行协整检验
    if price1_order != 1:
        if verbose:
            print(f"当前只支持I(1)序列的协整检验，{symbol1}和{symbol2}都是I({price1_order})，跳过")
        return results

    if verbose:
        print(f"\n✓ {symbol1} 和 {symbol2} 都是 I(1)，可以进行协整检验")

    # 步骤5: 根据diff_order计算对冲比率和价差
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]

    if diff_order == 0:
        # 原始价差：使用原始价格计算对冲比率和价差
        if verbose:
            print(f"\n--- 步骤3: 计算最优对冲比率（OLS回归，原始价格） ---")
        hedge_ratio = calculate_hedge_ratio(price1_aligned, price2_aligned)
        results['hedge_ratio'] = hedge_ratio

        if verbose:
            print(f"\n--- 步骤4: 计算原始价差（残差） ---")
        spread = price1_aligned - hedge_ratio * price2_aligned
        results['spread'] = spread

        if verbose:
            print(f"价差统计:")
            print(f"  均值: {spread.mean():.6f}")
            print(f"  标准差: {spread.std():.6f}")
            print(f"  最小值: {spread.min():.6f}")
            print(f"  最大值: {spread.max():.6f}")

        # 步骤6: 检验原始价差的平稳性（协整检验）
        if verbose:
            print(f"\n--- 步骤5: 检验原始价差的平稳性（协整检验） ---")
    else:
        # 一阶差分价差：使用一阶差分价格计算对冲比率和价差
        if verbose:
            print(f"\n--- 步骤3: 计算一阶差分价格 ---")
        diff_price1 = price1_aligned.diff().dropna()
        diff_price2 = price2_aligned.diff().dropna()

        # 确保两个差分序列长度一致
        min_diff_length = min(len(diff_price1), len(diff_price2))
        diff_price1_aligned = diff_price1.iloc[:min_diff_length]
        diff_price2_aligned = diff_price2.iloc[:min_diff_length]

        if verbose:
            print(f"\n--- 步骤4: 计算最优对冲比率（OLS回归，一阶差分价格） ---")
        hedge_ratio = calculate_hedge_ratio(diff_price1_aligned, diff_price2_aligned)
        results['hedge_ratio'] = hedge_ratio

        if verbose:
            print(f"\n--- 步骤5: 计算一阶差分价差 ---")
        spread = diff_price1_aligned - hedge_ratio * diff_price2_aligned
        results['spread'] = spread

        if verbose:
            print(f"价差统计:")
            print(f"  均值: {spread.mean():.6f}")
            print(f"  标准差: {spread.std():.6f}")
            print(f"  最小值: {spread.min():.6f}")
            print(f"  最大值: {spread.max():.6f}")

        # 步骤6: 检验一阶差分价差的平稳性（协整检验）
        if verbose:
            print(f"\n--- 步骤6: 检验一阶差分价差的平稳性（协整检验） ---")
    spread_adf = advanced_adf_test(spread, verbose=verbose)
    results['spread_adf'] = spread_adf

    if spread_adf and spread_adf['is_stationary']:
        # 原价差平稳，协整关系成立！
        results['cointegration_found'] = True
        results['best_test'] = {
            'type': 'cointegration',
            'adf_result': spread_adf,
            'spread': spread
        }
        if verbose:
            print(f"\n 协整检验通过！{symbol1} 和 {symbol2} 存在协整关系")
            print(f"  价差是平稳的（I(0)），ADF P值: {spread_adf['p_value']:.6f}")
    else:
        if verbose:
            print(f"\n 协整检验未通过")
            print(f"  价差不平稳，ADF P值: {spread_adf['p_value']:.6f if spread_adf else 'N/A'}")

    return results


# ==================== 滚动窗口协整检验代码 ====================

def rolling_window_cointegration_test(price1, price2, symbol1, symbol2, window_size=500, step_size=100, diff_order=0):
    """
    滚动窗口协整检验

    Args:
        price1: 第一个价格序列
        price2: 第二个价格序列
        symbol1: 第一个币种名称
        symbol2: 第二个币种名称
        window_size: 窗口大小（数据条数）
        step_size: 步长（每次移动的数据条数）
        diff_order: 价差类型，0=原始价差，1=一阶差分价差

    Returns:
        dict: 滚动窗口检验结果
    """
    print(f"\n{'=' * 80}")
    print(f"滚动窗口协整检验: {symbol1}/{symbol2}")
    print(f"窗口大小: {window_size}, 步长: {step_size}")
    print(f"{'=' * 80}")

    # 确保两个序列长度一致
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]

    # 获取时间索引
    timestamps = price1_aligned.index

    # 存储所有窗口的检验结果
    window_results = []
    all_candidates = []

    # 滚动窗口
    num_windows = (min_length - window_size) // step_size + 1

    print(f"总数据点: {min_length}")
    print(f"窗口数量: {num_windows}")

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > min_length:
            end_idx = min_length
            start_idx = end_idx - window_size

        if start_idx < 0:
            continue

        # 提取窗口数据
        window_price1 = price1_aligned.iloc[start_idx:end_idx]
        window_price2 = price2_aligned.iloc[start_idx:end_idx]

        window_start_time = timestamps[start_idx]
        window_end_time = timestamps[end_idx - 1]

        print(f"\n窗口 {window_idx + 1}/{num_windows}:")
        print(f"  时间范围: {window_start_time} 到 {window_end_time}")
        print(f"  数据点: {start_idx} 到 {end_idx} (共 {len(window_price1)} 条)")

        # 对当前窗口进行协整检验
        try:
            coint_result = enhanced_cointegration_test(
                window_price1,
                window_price2,
                symbol1,
                symbol2,
                verbose=False,  # 窗口检验时不打印详细信息
                diff_order=diff_order  # 传递价差类型
            )

            # 添加窗口信息
            coint_result['window_idx'] = window_idx
            coint_result['window_start_idx'] = start_idx
            coint_result['window_end_idx'] = end_idx
            coint_result['window_start_time'] = window_start_time
            coint_result['window_end_time'] = window_end_time
            coint_result['window_size'] = len(window_price1)

            window_results.append(coint_result)

            if coint_result['cointegration_found']:
                print(f"   协整检验通过 (P值: {coint_result['spread_adf']['p_value']:.6f})")
                all_candidates.append(coint_result)
            else:
                print(f"   协整检验未通过")

        except Exception as e:
            print(f"  窗口检验出错: {str(e)}")
            continue

    # 汇总结果
    summary = {
        'pair_name': f"{symbol1}/{symbol2}",
        'symbol1': symbol1,
        'symbol2': symbol2,
        'total_windows': num_windows,
        'cointegration_windows': len(all_candidates),
        'cointegration_ratio': len(all_candidates) / num_windows if num_windows > 0 else 0,
        'window_results': window_results,
        'all_candidates': all_candidates
    }

    print(f"\n汇总结果:")
    print(f"  总窗口数: {num_windows}")
    print(f"  协整窗口数: {len(all_candidates)}")
    print(f"  协整比例: {summary['cointegration_ratio'] * 100:.1f}%")

    return summary


def rolling_window_find_cointegrated_pairs(data, window_size=500, step_size=100, diff_order=0):
    """
    滚动窗口寻找协整对

    Args:
        data: 包含各币对数据的字典
        window_size: 窗口大小（数据条数）
        step_size: 步长（每次移动的数据条数）
        diff_order: 价差类型，0=原始价差，1=一阶差分价差

    Returns:
        list: 协整对汇总结果列表
    """
    print("=" * 80)
    print("滚动窗口协整检验")
    print("=" * 80)
    print(f"窗口大小: {window_size}")
    print(f"步长: {step_size}")
    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"价差类型: {diff_type}")

    symbols = list(data.keys())
    all_summaries = []

    print(f"\n分析 {len(symbols)} 个币对...")

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]

            try:
                # 执行滚动窗口协整检验
                summary = rolling_window_cointegration_test(
                    data[symbol1],
                    data[symbol2],
                    symbol1,
                    symbol2,
                    window_size=window_size,
                    step_size=step_size,
                    diff_order=diff_order  # 传递价差类型
                )

                all_summaries.append(summary)

            except Exception as e:
                print(f"分析 {symbol1}/{symbol2} 时出错: {str(e)}")

    return all_summaries


# ==================== 第二层：协整稳定性评估 ====================

def evaluate_cointegration_stability(summary, data, diff_order=0):
    """
    评估协整关系的稳定性（第二层筛选）
    通过计算各窗口对冲比率的变化幅度来评估稳定性

    Args:
        summary: 滚动窗口检验结果汇总
        data: 原始数据字典
        diff_order: 价差类型，0=原始价差，1=一阶差分价差

    Returns:
        dict: 稳定性评估结果
    """
    symbol1 = summary['symbol1']
    symbol2 = summary['symbol2']

    if symbol1 not in data or symbol2 not in data:
        return {
            'is_stable': False,
            'stability_score': 0.0,
            'hedge_ratio_mean': None,
            'hedge_ratio_std': None,
            'hedge_ratio_cv': None,  # 变异系数
            'max_change_rate': None,
            'reason': '数据不可用'
        }

    price1 = data[symbol1]
    price2 = data[symbol2]

    # 获取所有窗口的对冲比率
    window_results = summary.get('window_results', [])
    hedge_ratios = []

    for window_result in window_results:
        if window_result.get('cointegration_found', False):
            # 获取该窗口的数据
            window_start = window_result.get('window_start', 0)
            window_end = window_result.get('window_end', len(price1))

            if window_end > len(price1) or window_end > len(price2):
                continue

            window_price1 = price1.iloc[window_start:window_end] if hasattr(price1, 'iloc') else price1[
                window_start:window_end]
            window_price2 = price2.iloc[window_start:window_end] if hasattr(price2, 'iloc') else price2[
                window_start:window_end]

            if diff_order == 0:
                # 原始价差（静默计算）
                min_length = min(len(window_price1), len(window_price2))
                price1_aligned = window_price1.iloc[:min_length] if hasattr(window_price1, 'iloc') else window_price1[
                    :min_length]
                price2_aligned = window_price2.iloc[:min_length] if hasattr(window_price2, 'iloc') else window_price2[
                    :min_length]
                X = price2_aligned.values.reshape(-1, 1)
                y = price1_aligned.values
                X_with_const = add_constant(X)
                model = OLS(y, X_with_const).fit()
                hedge_ratio = model.params[1]
            else:
                # 一阶差分价差（静默计算）
                diff_price1 = window_price1.diff().dropna()
                diff_price2 = window_price2.diff().dropna()
                if len(diff_price1) > 10 and len(diff_price2) > 10:
                    min_length = min(len(diff_price1), len(diff_price2))
                    price1_aligned = diff_price1.iloc[:min_length] if hasattr(diff_price1, 'iloc') else diff_price1[
                        :min_length]
                    price2_aligned = diff_price2.iloc[:min_length] if hasattr(diff_price2, 'iloc') else diff_price2[
                        :min_length]
                    X = price2_aligned.values.reshape(-1, 1)
                    y = price1_aligned.values
                    X_with_const = add_constant(X)
                    model = OLS(y, X_with_const).fit()
                    hedge_ratio = model.params[1]
                else:
                    continue

            hedge_ratios.append(hedge_ratio)

    if len(hedge_ratios) < 3:
        return {
            'is_stable': False,
            'stability_score': 0.0,
            'hedge_ratio_mean': None,
            'hedge_ratio_std': None,
            'hedge_ratio_cv': None,
            'max_change_rate': None,
            'reason': '协整窗口数不足'
        }

    hedge_ratios = np.array(hedge_ratios)
    hedge_ratio_mean = np.mean(hedge_ratios)
    hedge_ratio_std = np.std(hedge_ratios)

    # 变异系数（Coefficient of Variation）
    hedge_ratio_cv = hedge_ratio_std / abs(hedge_ratio_mean) if abs(hedge_ratio_mean) > 1e-8 else float('inf')

    # 计算最大变化率
    changes = np.abs(np.diff(hedge_ratios))
    relative_changes = changes / (np.abs(hedge_ratios[:-1]) + 1e-8)
    max_change_rate = np.max(relative_changes) if len(relative_changes) > 0 else 0.0

    # 稳定性评分（0-1，越高越稳定）
    # CV < 0.1: 非常稳定，CV < 0.2: 稳定，CV < 0.3: 中等，CV >= 0.3: 不稳定
    if hedge_ratio_cv < 0.1:
        stability_score = 1.0
    elif hedge_ratio_cv < 0.2:
        stability_score = 0.8
    elif hedge_ratio_cv < 0.3:
        stability_score = 0.6
    elif hedge_ratio_cv < 0.5:
        stability_score = 0.4
    else:
        stability_score = 0.2

    # 如果最大变化率过大，降低稳定性评分
    if max_change_rate > 0.3:
        stability_score *= 0.5
    elif max_change_rate > 0.2:
        stability_score *= 0.7

    # 判断是否稳定（稳定性评分 >= 0.5 且 CV < 0.3）
    is_stable = stability_score >= 0.5 and hedge_ratio_cv < 0.3

    return {
        'is_stable': is_stable,
        'stability_score': stability_score,
        'hedge_ratio_mean': hedge_ratio_mean,
        'hedge_ratio_std': hedge_ratio_std,
        'hedge_ratio_cv': hedge_ratio_cv,
        'max_change_rate': max_change_rate,
        'num_windows': len(hedge_ratios),
        'reason': '通过稳定性评估' if is_stable else f'稳定性不足 (CV={hedge_ratio_cv:.3f}, 最大变化率={max_change_rate:.3f})'
    }


def filter_by_stability(summaries, data, diff_order=0, min_stability_score=0.5):
    """
    根据稳定性评估结果过滤币对（第二层筛选）

    Args:
        summaries: 滚动窗口检验结果汇总列表
        data: 原始数据字典
        diff_order: 价差类型
        min_stability_score: 最小稳定性评分

    Returns:
        list: 通过稳定性评估的币对列表
    """
    print("\n" + "=" * 80)
    print("第二层：协整稳定性评估")
    print("=" * 80)

    stable_pairs = []

    for summary in summaries:
        symbol1 = summary['symbol1']
        symbol2 = summary['symbol2']
        pair_name = f"{symbol1}/{symbol2}"

        print(f"\n评估币对: {pair_name}")

        stability_result = evaluate_cointegration_stability(summary, data, diff_order)

        print(f"  稳定性评分: {stability_result['stability_score']:.3f}")

        # 安全地格式化数值，处理None值
        hedge_ratio_mean = stability_result.get('hedge_ratio_mean')
        hedge_ratio_std = stability_result.get('hedge_ratio_std')
        hedge_ratio_cv = stability_result.get('hedge_ratio_cv')
        max_change_rate = stability_result.get('max_change_rate')

        print(f"  对冲比率均值: {hedge_ratio_mean:.6f}" if hedge_ratio_mean is not None else "  对冲比率均值: N/A")
        print(f"  对冲比率标准差: {hedge_ratio_std:.6f}" if hedge_ratio_std is not None else "  对冲比率标准差: N/A")
        print(f"  变异系数(CV): {hedge_ratio_cv:.3f}" if hedge_ratio_cv is not None else "  变异系数(CV): N/A")
        print(f"  最大变化率: {max_change_rate:.3f}" if max_change_rate is not None else "  最大变化率: N/A")
        print(f"  评估结果: {stability_result['reason']}")

        if stability_result['is_stable'] and stability_result['stability_score'] >= min_stability_score:
            stable_pairs.append({
                **summary,
                'stability_result': stability_result
            })
            print(f"   通过稳定性评估")
        else:
            print(f"   未通过稳定性评估")

    print(f"\n稳定性评估完成: {len(stable_pairs)}/{len(summaries)} 个币对通过评估")

    return stable_pairs


def select_spread_type():
    """
    选择价差类型

    Returns:
        int: 0=原始价差, 1=一阶差分价差
    """
    print("\n" + "=" * 60)
    print("选择价差类型")
    print("=" * 60)
    print("请选择用于协整检验和回测的价差类型:")
    print("  0. 原始价差（原始价格计算的价差）")
    print("  1. 一阶差分价差（一阶差分价格计算的价差）")
    print("\n注意：")
    print("  - 如果选择原始价差，对冲比率从原始价格计算得出")
    print("  - 如果选择一阶差分价差，对冲比率从一阶差分价格计算得出")

    while True:
        try:
            choice = input("请选择 (0/1，默认0): ").strip()
            if not choice:
                diff_order = 0
                break
            elif choice == '0':
                diff_order = 0
                break
            elif choice == '1':
                diff_order = 1
                break
            else:
                print("无效选择，请输入 0 或 1")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认值：原始价差")
            diff_order = 0
            break
        except Exception as e:
            print(f"输入错误: {str(e)}，请重新输入")

    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"\n已选择: {diff_type}")
    return diff_order


def configure_rolling_window_parameters():
    """配置滚动窗口参数"""
    print("\n" + "=" * 60)
    print("滚动窗口参数配置")
    print("=" * 60)

    # 默认参数
    default_params = {
        'window_size': 500,
        'step_size': 100
    }

    print("当前默认参数:")
    print(f"  1. 窗口大小: {default_params['window_size']} 条数据")
    print(f"  2. 步长: {default_params['step_size']} 条数据")

    print("\n是否要修改参数？")
    print("输入 'y' 修改参数，直接回车使用默认参数")

    modify_choice = input("请选择: ").strip().lower()

    if modify_choice == 'y':
        print("\n请输入新的参数值（直接回车保持默认值）:")

        # 窗口大小
        window_input = input(f"窗口大小 (默认: {default_params['window_size']}): ").strip()
        if window_input:
            try:
                default_params['window_size'] = int(window_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['window_size']}")

        # 步长
        step_input = input(f"步长 (默认: {default_params['step_size']}): ").strip()
        if step_input:
            try:
                default_params['step_size'] = int(step_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['step_size']}")

        print("\n修改后的参数:")
        print(f"  1. 窗口大小: {default_params['window_size']} 条数据")
        print(f"  2. 步长: {default_params['step_size']} 条数据")

    return default_params


def display_rolling_window_candidates(summaries, data, diff_order=0, min_cointegration_ratio=0.2):
    """
    显示滚动窗口协整对候选列表

    Args:
        summaries: 滚动窗口检验结果汇总列表
        data: 原始数据字典（用于重新计算整体数据的对冲比率）
        diff_order: 价差类型，0=原始价差，1=一阶差分价差
        min_cointegration_ratio: 最小协整比例阈值（只显示协整比例>=此值的币对）

    Returns:
        list: 用户选择的币对列表（包含基于整个数据集计算的对冲比率）
    """
    # 过滤：只显示协整比例>=阈值的币对
    filtered_summaries = [s for s in summaries if s['cointegration_ratio'] >= min_cointegration_ratio]

    if not filtered_summaries:
        print(f"\n没有找到协整比例 >= {min_cointegration_ratio * 100:.1f}% 的币对")
        return []

    print("\n" + "=" * 80)
    print("滚动窗口协整对候选列表")
    print("=" * 80)
    print(f"最小协整比例阈值: {min_cointegration_ratio * 100:.1f}%")

    for i, summary in enumerate(filtered_summaries, 1):
        pair_name = summary['pair_name']
        coint_ratio = summary['cointegration_ratio']
        total_windows = summary['total_windows']
        coint_windows = summary['cointegration_windows']

        print(f"\n{i}. 币对: {pair_name}")
        print(f"   总窗口数: {total_windows}")
        print(f"   协整窗口数: {coint_windows}")
        print(f"   协整比例: {coint_ratio * 100:.1f}%")

        # 使用整个数据集重新计算对冲比率
        symbol1 = summary['symbol1']
        symbol2 = summary['symbol2']
        if symbol1 in data and symbol2 in data:
            price1 = data[symbol1]
            price2 = data[symbol2]

            if diff_order == 0:
                # 原始价差：使用原始价格计算对冲比率
                overall_hedge_ratio = calculate_hedge_ratio(price1, price2)
            else:
                # 一阶差分价差：使用一阶差分价格计算对冲比率
                diff_price1 = price1.diff().dropna()
                diff_price2 = price2.diff().dropna()
                overall_hedge_ratio = calculate_hedge_ratio(diff_price1, diff_price2)

            print(f"   整体数据对冲比率: {overall_hedge_ratio:.6f} (基于整个数据集计算)")

            # 显示最佳窗口的信息（P值最小的窗口，仅用于参考）
            if summary['all_candidates']:
                best_window = min(summary['all_candidates'],
                                  key=lambda x: x['spread_adf']['p_value'] if x['spread_adf'] else 1.0)
                print(f"   最佳窗口参考信息:")
                print(f"     时间范围: {best_window['window_start_time']} 到 {best_window['window_end_time']}")
                print(f"     窗口对冲比率: {best_window['hedge_ratio']:.6f} (仅供参考)")
                print(f"     价差ADF P值: {best_window['spread_adf']['p_value']:.6f}")
        else:
            print(f"   警告: 无法找到 {symbol1} 或 {symbol2} 的数据")

    print("\n" + "=" * 80)
    print("请选择要使用的币对（输入序号，用逗号分隔，如: 1,3,5）")
    print("=" * 80)

    while True:
        try:
            selection_input = input("请输入选择的币对序号: ").strip()
            if not selection_input:
                print("请输入至少一个币对序号")
                continue

            selected_indices = [int(x.strip()) - 1 for x in selection_input.split(',')]

            # 验证索引
            valid_selection = []
            for idx in selected_indices:
                if 0 <= idx < len(filtered_summaries):
                    summary = filtered_summaries[idx]
                    if summary['all_candidates']:
                        symbol1 = summary['symbol1']
                        symbol2 = summary['symbol2']

                        if symbol1 in data and symbol2 in data:
                            price1 = data[symbol1]
                            price2 = data[symbol2]

                            # 使用整个数据集重新计算对冲比率
                            if diff_order == 0:
                                # 原始价差：使用原始价格计算对冲比率
                                overall_hedge_ratio = calculate_hedge_ratio(price1, price2)
                            else:
                                # 一阶差分价差：使用一阶差分价格计算对冲比率
                                diff_price1 = price1.diff().dropna()
                                diff_price2 = price2.diff().dropna()
                                overall_hedge_ratio = calculate_hedge_ratio(diff_price1, diff_price2)

                            # 使用整个数据集重新进行协整检验（用于获取价差和ADF结果）
                            coint_result = enhanced_cointegration_test(
                                price1, price2, symbol1, symbol2,
                                verbose=False, diff_order=diff_order
                            )

                            # 构建币对信息（使用整体数据的对冲比率和协整结果）
                            pair_info = {
                                'pair_name': summary['pair_name'],
                                'symbol1': symbol1,
                                'symbol2': symbol2,
                                'hedge_ratio': overall_hedge_ratio,  # 使用整体数据的对冲比率
                                'spread': coint_result.get('spread'),
                                'spread_adf': coint_result.get('spread_adf'),
                                'cointegration_found': coint_result.get('cointegration_found', False),
                                'diff_order': diff_order,
                                # 保留一些窗口信息用于显示
                                'cointegration_ratio': summary['cointegration_ratio'],
                                'total_windows': summary['total_windows'],
                                'cointegration_windows': summary['cointegration_windows']
                            }

                            valid_selection.append(pair_info)
                        else:
                            print(f"序号 {idx + 1} 无法找到数据，跳过")
                    else:
                        print(f"序号 {idx + 1} 没有协整窗口，跳过")
                else:
                    print(f"序号 {idx + 1} 超出范围")

            if valid_selection:
                print(f"\n已选择 {len(valid_selection)} 个币对:")
                for selected in valid_selection:
                    diff_type = '原始价差' if selected.get('diff_order', 0) == 0 else '一阶差分价差'
                    print(
                        f"   - {selected['pair_name']} (对冲比率: {selected['hedge_ratio']:.6f}, 价差类型: {diff_type})")
                    print(f"     协整比例: {selected.get('cointegration_ratio', 0) * 100:.1f}%")
                return valid_selection
            else:
                print("没有有效的选择，请重新输入")

        except ValueError:
            print("输入格式错误，请输入数字序号，用逗号分隔")
        except KeyboardInterrupt:
            print("\n用户取消选择")
            return []


def input_preselected_pairs(data, default_diff_order=0):
    """
    根据用户输入的币对列表构建已筛选的协整对信息（无需重新协整检验）

    Args:
        data: 价格数据字典
        default_diff_order: 全局默认价差类型

    Returns:
        list: 用户输入的币对信息列表
    """
    if not data:
        print("数据为空，无法选择币对")
        return []

    available_symbols = sorted(list(data.keys()))
    print("\n" + "=" * 80)
    print("输入已筛选好的协整币对")
    print("=" * 80)
    print("提示：")
    print("  1. 已假设协整检验和稳定性筛选已经完成")
    print("  2. 请直接输入需要回测/优化的币对，例如: BTCUSDT/ETHUSDT")
    print("  3. 多个币对可使用逗号分隔，或逐个输入")
    print("  4. 输入完成后直接回车即可开始回测/优化")
    print("\n可用币种列表（前20个，全部共 {0} 个）:".format(len(available_symbols)))
    print("  " + ", ".join(available_symbols[:20]))
    if len(available_symbols) > 20:
        print("  ...")

    selected_pairs = []

    while True:
        pair_input = input("\n请输入币对 (格式: SYMBOL1/SYMBOL2，直接回车结束): ").strip()
        if not pair_input:
            if selected_pairs:
                break
            else:
                print("尚未添加任何币对，请至少输入一个币对。")
                continue

        pair_candidates = [p.strip() for p in pair_input.replace('，', ',').split(',') if p.strip()]
        for pair_text in pair_candidates:
            separator = '/'
            if '/' in pair_text:
                separator = '/'
            elif '-' in pair_text:
                separator = '-'
            elif ' ' in pair_text:
                separator = ' '
            else:
                separator = '/'
            if separator not in pair_text:
                print(f"   无法识别的币对格式: {pair_text}，请使用 SYMBOL1/SYMBOL2")
                continue

            symbol1, symbol2 = [s.strip().upper() for s in pair_text.split(separator)[:2]]

            if symbol1 == symbol2:
                print(f"   {symbol1} 与 {symbol2} 相同，忽略")
                continue

            if symbol1 not in data:
                print(f"   数据中不存在 {symbol1}，跳过")
                continue
            if symbol2 not in data:
                print(f"   数据中不存在 {symbol2}，跳过")
                continue

            pair_diff_order_input = input(
                f"  请选择 {symbol1}/{symbol2} 的价差类型 (0=原始,1=一阶, 回车沿用默认 {default_diff_order}): ").strip()
            if pair_diff_order_input in ['0', '1']:
                pair_diff_order = int(pair_diff_order_input)
            else:
                pair_diff_order = default_diff_order

            price1 = data[symbol1]
            price2 = data[symbol2]

            try:
                if pair_diff_order == 0:
                    hedge_ratio = calculate_hedge_ratio(price1, price2)
                else:
                    diff_price1 = price1.diff().dropna()
                    diff_price2 = price2.diff().dropna()
                    min_length = min(len(diff_price1), len(diff_price2))
                    if min_length < 30:
                        print(f"  × {symbol1}/{symbol2} 数据不足以计算一阶差分对冲比率 (有效数据 < 30)")
                        continue
                    hedge_ratio = calculate_hedge_ratio(
                        diff_price1.iloc[:min_length],
                        diff_price2.iloc[:min_length]
                    )
            except Exception as e:
                print(f"  × 计算 {symbol1}/{symbol2} 对冲比率失败: {str(e)}")
                continue

            manual_ratio = input(
                f"  是否为 {symbol1}/{symbol2} 自定义对冲比率？(直接回车使用 {hedge_ratio:.6f}): ").strip()
            if manual_ratio:
                try:
                    hedge_ratio = float(manual_ratio)
                except ValueError:
                    print("  输入无效，使用自动计算的对冲比率")

            pair_info = {
                'pair_name': f"{symbol1}/{symbol2}",
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'diff_order': pair_diff_order,
                'spread': None,
                'spread_adf': None,
                'cointegration_found': True,
                'cointegration_ratio': 1.0
            }

            selected_pairs.append(pair_info)
            print(f"  ✓ 已添加 {pair_info['pair_name']}，对冲比率 {hedge_ratio:.6f}，价差类型 {'原始' if pair_diff_order == 0 else '一阶差分'}")

    print(f"\n共选择 {len(selected_pairs)} 个币对用于后续回测/优化。")
    for pair in selected_pairs:
        diff_type = '原始价差' if pair['diff_order'] == 0 else '一阶差分价差'
        print(f"  - {pair['pair_name']} (对冲比率: {pair['hedge_ratio']:.6f}, 价差类型: {diff_type})")

    return selected_pairs


def input_cointegration_window_size(default_size=500):
    """
    输入协整监控窗口大小，用于RLS与稳定性判断

    Args:
        default_size: 默认窗口大小

    Returns:
        int: 窗口大小
    """
    print("\n" + "=" * 60)
    print("配置协整监控窗口大小")
    print("=" * 60)
    print("该窗口大小仅用于RLS动态监控和协整状态重新校验，不再执行自动筛选。")
    print(f"默认值: {default_size} 条数据（约等于原流程中的滚动窗口大小）")

    while True:
        user_input = input(f"请输入窗口大小 (直接回车使用 {default_size}): ").strip()
        if not user_input:
            return default_size
        try:
            window_size = int(user_input)
            if window_size <= 0:
                print("窗口大小必须为正整数，请重新输入。")
                continue
            return window_size
        except ValueError:
            print("输入无效，请输入正整数。")


def select_z_score_strategy():
    """
    选择Z-score计算策略

    Returns:
        BaseZScoreStrategy: 选择的策略对象，如果失败返回None
    """
    if not STRATEGIES_AVAILABLE:
        print("警告: 策略模块不可用，将使用传统方法")
        return None

    print("\n" + "=" * 60)
    print("选择Z-score计算策略")
    print("=" * 60)
    print("请选择Z-score计算策略:")
    print("  1. 传统方法（均值和标准差）")

    # 检查ARIMA-GARCH是否可用
    arima_garch_available = ARIMA_AVAILABLE and GARCH_AVAILABLE
    if arima_garch_available:
        print("  2. ARIMA-GARCH模型")
    else:
        print("  2. ARIMA-GARCH模型（不可用：缺少必要的库）")

    # 检查ECM是否可用
    ecm_available = STRATEGIES_AVAILABLE and STATSMODELS_AVAILABLE
    if ecm_available:
        print("  3. ECM误差修正模型（推荐用于协整交易）")
    else:
        print("  3. ECM误差修正模型（不可用：缺少必要的库）")

    # 检查Kalman Filter是否可用
    kalman_available = STRATEGIES_AVAILABLE
    if kalman_available:
        print("  4. Kalman Filter动态价差模型（推荐用于动态市场）")
    else:
        print("  4. Kalman Filter动态价差模型（不可用：缺少必要的库）")

    # 检查Copula + DCC-GARCH是否可用
    copula_dcc_available = STRATEGIES_AVAILABLE and GARCH_AVAILABLE
    if copula_dcc_available:
        print("  5. Copula + DCC-GARCH相关性/波动率模型（推荐用于相关性建模）")
    else:
        print("  5. Copula + DCC-GARCH相关性/波动率模型（不可用：缺少必要的库）")

    # 检查Regime-Switching是否可用
    regime_switching_available = STRATEGIES_AVAILABLE and STATSMODELS_AVAILABLE
    if regime_switching_available:
        print("  6. Regime-Switching市场状态模型（推荐用于状态转换市场）")
    else:
        print("  6. Regime-Switching市场状态模型（不可用：缺少必要的库）")

    print("  0. 退出程序")

    # 确定最大选择数
    max_choice = 6

    while True:
        try:
            choice = input(f"请选择 (0-{max_choice}): ").strip()

            if choice == '0':
                return None

            if choice == '1':
                strategy = TraditionalZScoreStrategy()
                print(f"已选择: {strategy.get_strategy_description()}")
                return strategy

            if choice == '2' and arima_garch_available:
                # 询问ARIMA和GARCH参数
                print("\n配置ARIMA-GARCH模型参数:")
                print("  直接回车使用默认值: ARIMA(1,0,1), GARCH(1,1)")

                arima_input = input("ARIMA阶数 (p,d,q，格式如: 1,0,1): ").strip()
                if arima_input:
                    try:
                        arima_parts = [int(x.strip()) for x in arima_input.split(',')]
                        if len(arima_parts) == 3:
                            arima_order = tuple(arima_parts)
                        else:
                            print("输入格式错误，使用默认值")
                            arima_order = (1, 0, 1)
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        arima_order = (1, 0, 1)
                else:
                    arima_order = (1, 0, 1)

                garch_input = input("GARCH阶数 (p,q，格式如: 1,1): ").strip()
                if garch_input:
                    try:
                        garch_parts = [int(x.strip()) for x in garch_input.split(',')]
                        if len(garch_parts) == 2:
                            garch_order = tuple(garch_parts)
                        else:
                            print("输入格式错误，使用默认值")
                            garch_order = (1, 1)
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        garch_order = (1, 1)
                else:
                    garch_order = (1, 1)

                try:
                    strategy = ArimaGarchZScoreStrategy(arima_order=arima_order, garch_order=garch_order)
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"ARIMA-GARCH策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '3' and ecm_available:
                # 询问ECM参数
                print("\n配置ECM误差修正模型参数:")
                print("  直接回车使用默认值: 滞后阶数=1, 最小数据长度=30")

                ecm_lag_input = input("误差修正项滞后阶数 (默认1): ").strip()
                if ecm_lag_input:
                    try:
                        ecm_lag = int(ecm_lag_input)
                        if ecm_lag < 1:
                            print("滞后阶数必须>=1，使用默认值")
                            ecm_lag = 1
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        ecm_lag = 1
                else:
                    ecm_lag = 1

                min_data_input = input("最小数据长度 (默认30): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 10:
                            print("最小数据长度必须>=10，使用默认值")
                            min_data_length = 30
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 30
                else:
                    min_data_length = 30

                try:
                    strategy = EcmZScoreStrategy(ecm_lag=ecm_lag, min_data_length=min_data_length)
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"ECM策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '4' and kalman_available:
                # 询问Kalman Filter参数
                print("\n配置Kalman Filter动态价差模型参数:")
                print("  直接回车使用默认值: 过程方差=0.01, 观测方差=0.1, 最小数据长度=30")

                process_var_input = input("过程噪声方差 (默认0.01): ").strip()
                if process_var_input:
                    try:
                        process_variance = float(process_var_input)
                        if process_variance <= 0:
                            print("过程方差必须>0，使用默认值")
                            process_variance = 0.01
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        process_variance = 0.01
                else:
                    process_variance = 0.01

                obs_var_input = input("观测噪声方差 (默认0.1): ").strip()
                if obs_var_input:
                    try:
                        observation_variance = float(obs_var_input)
                        if observation_variance <= 0:
                            print("观测方差必须>0，使用默认值")
                            observation_variance = 0.1
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        observation_variance = 0.1
                else:
                    observation_variance = 0.1

                min_data_input = input("最小数据长度 (默认30): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 10:
                            print("最小数据长度必须>=10，使用默认值")
                            min_data_length = 30
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 30
                else:
                    min_data_length = 30

                try:
                    strategy = KalmanFilterZScoreStrategy(
                        process_variance=process_variance,
                        observation_variance=observation_variance,
                        min_data_length=min_data_length
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Kalman Filter策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '5' and copula_dcc_available:
                # 询问Copula + DCC-GARCH参数
                print("\n配置Copula + DCC-GARCH相关性/波动率模型参数:")
                print("  直接回车使用默认值: GARCH(1,1), Copula类型=高斯, 最小数据长度=50")

                garch_input = input("GARCH阶数 (p,q，格式如: 1,1): ").strip()
                if garch_input:
                    try:
                        garch_parts = [int(x.strip()) for x in garch_input.split(',')]
                        if len(garch_parts) == 2:
                            garch_order = tuple(garch_parts)
                        else:
                            print("输入格式错误，使用默认值")
                            garch_order = (1, 1)
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        garch_order = (1, 1)
                else:
                    garch_order = (1, 1)

                copula_input = input("Copula类型 (gaussian/student，默认gaussian): ").strip().lower()
                if copula_input in ['gaussian', 'student']:
                    copula_type = copula_input
                else:
                    print("输入格式错误，使用默认值")
                    copula_type = 'gaussian'

                min_data_input = input("最小数据长度 (默认50): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 20:
                            print("最小数据长度必须>=20，使用默认值")
                            min_data_length = 50
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 50
                else:
                    min_data_length = 50

                try:
                    strategy = CopulaDccGarchZScoreStrategy(
                        garch_order=garch_order,
                        copula_type=copula_type,
                        min_data_length=min_data_length
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Copula + DCC-GARCH策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '6' and regime_switching_available:
                # 询问Regime-Switching参数
                print("\n配置Regime-Switching市场状态模型参数:")
                print("  直接回车使用默认值: 状态数量=2, 最小数据长度=50, 平滑概率=True")

                n_regimes_input = input("状态数量 (默认2): ").strip()
                if n_regimes_input:
                    try:
                        n_regimes = int(n_regimes_input)
                        if n_regimes < 2:
                            print("状态数量必须>=2，使用默认值")
                            n_regimes = 2
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        n_regimes = 2
                else:
                    n_regimes = 2

                min_data_input = input("最小数据长度 (默认50): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 20:
                            print("最小数据长度必须>=20，使用默认值")
                            min_data_length = 50
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 50
                else:
                    min_data_length = 50

                smoothing_input = input("是否使用平滑概率 (True/False，默认True): ").strip().lower()
                if smoothing_input in ['true', '1', 'yes', 'y']:
                    smoothing = True
                elif smoothing_input in ['false', '0', 'no', 'n']:
                    smoothing = False
                else:
                    smoothing = True

                try:
                    strategy = RegimeSwitchingZScoreStrategy(
                        n_regimes=n_regimes,
                        min_data_length=min_data_length,
                        smoothing=smoothing
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Regime-Switching策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            print(f"无效选择，请输入 0-{max_choice} 之间的数字")

        except KeyboardInterrupt:
            print("\n用户取消选择")
            return None
        except Exception as e:
            print(f"选择失败: {str(e)}，请重新选择")


def configure_trading_parameters():
    """配置交易参数"""
    print("\n" + "=" * 60)
    print("交易参数配置")
    print("=" * 60)

    # 默认参数
    default_params = {
        'lookback_period': 60,
        'z_threshold': 1.5,
        'z_exit_threshold': 0.6,
        'take_profit_pct': 0.15,
        'stop_loss_pct': 0.08,
        'max_holding_hours': 168,
        'position_ratio': 0.5,
        'leverage': 5,
        'trading_fee_rate': 0.000275  # 交易手续费率 0.0275%
    }

    print("当前默认参数:")
    print(f"  1. 回看期: {default_params['lookback_period']}")
    print(f"  2. Z-score开仓阈值: {default_params['z_threshold']}")
    print(f"  3. Z-score平仓阈值: {default_params['z_exit_threshold']}")
    print(f"  4. 止盈百分比: {default_params['take_profit_pct'] * 100:.1f}%")
    print(f"  5. 止损百分比: {default_params['stop_loss_pct'] * 100:.1f}%")
    print(f"  6. 最大持仓时间: {default_params['max_holding_hours']}小时")
    print(
        f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1 - default_params['position_ratio']) * 100:.1f}%作为安全垫)")
    print(f"  8. 杠杆: {default_params['leverage']}倍")
    print(f"  9. 交易手续费率: {default_params['trading_fee_rate'] * 100:.4f}%")

    print("\n是否要修改参数？")
    print("输入 'y' 修改参数，直接回车使用默认参数")

    modify_choice = input("请选择: ").strip().lower()

    if modify_choice == 'y':
        print("\n请输入新的参数值（直接回车保持默认值）:")

        # 回看期
        lookback_input = input(f"回看期 (默认: {default_params['lookback_period']}): ").strip()
        if lookback_input:
            try:
                default_params['lookback_period'] = int(lookback_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['lookback_period']}")

        # Z-score开仓阈值
        z_threshold_input = input(f"Z-score开仓阈值 (默认: {default_params['z_threshold']}): ").strip()
        if z_threshold_input:
            try:
                default_params['z_threshold'] = float(z_threshold_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['z_threshold']}")

        # Z-score平仓阈值
        z_exit_input = input(f"Z-score平仓阈值 (默认: {default_params['z_exit_threshold']}): ").strip()
        if z_exit_input:
            try:
                default_params['z_exit_threshold'] = float(z_exit_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['z_exit_threshold']}")

        # 止盈百分比
        take_profit_input = input(f"止盈百分比 (默认: {default_params['take_profit_pct'] * 100:.1f}%): ").strip()
        if take_profit_input:
            try:
                default_params['take_profit_pct'] = float(take_profit_input) / 100
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['take_profit_pct'] * 100:.1f}%")

        # 止损百分比
        stop_loss_input = input(f"止损百分比 (默认: {default_params['stop_loss_pct'] * 100:.1f}%): ").strip()
        if stop_loss_input:
            try:
                default_params['stop_loss_pct'] = float(stop_loss_input) / 100
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['stop_loss_pct'] * 100:.1f}%")

        # 最大持仓时间
        max_holding_input = input(f"最大持仓时间(小时) (默认: {default_params['max_holding_hours']}): ").strip()
        if max_holding_input:
            try:
                default_params['max_holding_hours'] = int(max_holding_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['max_holding_hours']}")

        # 仓位比例
        position_ratio_input = input(f"仓位比例 (默认: {default_params['position_ratio'] * 100:.1f}%): ").strip()
        if position_ratio_input:
            try:
                default_params['position_ratio'] = float(position_ratio_input) / 100
                if default_params['position_ratio'] <= 0 or default_params['position_ratio'] > 1:
                    print(f"仓位比例应在0-100%之间，使用默认值: {default_params['position_ratio'] * 100:.1f}%")
                    default_params['position_ratio'] = 0.5
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['position_ratio'] * 100:.1f}%")

        # 杠杆
        leverage_input = input(f"杠杆 (默认: {default_params['leverage']}): ").strip()
        if leverage_input:
            try:
                default_params['leverage'] = int(leverage_input)
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['leverage']}")

        # 交易手续费率
        fee_rate_input = input(f"交易手续费率 (默认: {default_params['trading_fee_rate'] * 100:.4f}%): ").strip()
        if fee_rate_input:
            try:
                default_params['trading_fee_rate'] = float(fee_rate_input) / 100
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['trading_fee_rate'] * 100:.4f}%")

        print("\n修改后的参数:")
        print(f"  1. 回看期: {default_params['lookback_period']}")
        print(f"  2. Z-score开仓阈值: {default_params['z_threshold']}")
        print(f"  3. Z-score平仓阈值: {default_params['z_exit_threshold']}")
        print(f"  4. 止盈百分比: {default_params['take_profit_pct'] * 100:.1f}%")
        print(f"  5. 止损百分比: {default_params['stop_loss_pct'] * 100:.1f}%")
        print(f"  6. 最大持仓时间: {default_params['max_holding_hours']}小时")
        print(
            f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1 - default_params['position_ratio']) * 100:.1f}%作为安全垫)")
        print(f"  8. 杠杆: {default_params['leverage']}倍")
        print(f"  9. 交易手续费率: {default_params['trading_fee_rate'] * 100:.4f}%")

    return default_params


# ==================== 高级交易流程代码 ====================

class AdvancedCointegrationTrading:
    """高级协整交易策略类（支持策略模式）"""

    def __init__(self, lookback_period=60, z_threshold=2.0, z_exit_threshold=0.5,
                 take_profit_pct=0.15, stop_loss_pct=0.08, max_holding_hours=168,
                 position_ratio=0.5, leverage=5, trading_fee_rate=0.000275,
                 z_score_strategy=None, use_arima_garch=False, arima_order=(1, 0, 1), garch_order=(1, 1),
                 use_rls=True, rls_lambda=0.99, rls_max_change_rate=0.2,
                 cointegration_check_interval=240, data_period_minutes=60, diff_order=0,
                 cointegration_window_size=500):
        """
        初始化高级协整交易策略（支持RLS动态对冲比率）

        Args:
            lookback_period: 回看期
            z_threshold: Z-score开仓阈值
            z_exit_threshold: Z-score平仓阈值
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止损百分比
            max_holding_hours: 最大持仓时间（小时）
            position_ratio: 仓位比例（默认0.5，即使用50%资金，留50%作为安全垫）
            leverage: 杠杆倍数
            trading_fee_rate: 交易手续费率（默认0.0275%，即0.000275）
            z_score_strategy: Z-score计算策略对象（BaseZScoreStrategy实例）
            use_arima_garch: 是否使用ARIMA-GARCH模型（向后兼容，如果提供了z_score_strategy则忽略此参数）
            arima_order: ARIMA模型阶数 (p, d, q)（向后兼容）
            garch_order: GARCH模型阶数 (p, q)（向后兼容）
            use_rls: 是否使用RLS动态更新对冲比率
            rls_lambda: RLS遗忘因子（0 < λ ≤ 1）
            rls_max_change_rate: RLS最大变化率（防止突变）
            cointegration_check_interval: 协整检验间隔（数据条数，默认240，对应1h周期10天）
            data_period_minutes: 数据周期（分钟，用于计算协整检验间隔，默认60分钟=1h）
            diff_order: 价差类型，0=原始价差，1=一阶差分价差
            cointegration_window_size: 协整检验窗口大小（与初始筛选的window_size一致，默认500）
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.z_exit_threshold = z_exit_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_hours = max_holding_hours
        self.position_ratio = position_ratio
        self.leverage = leverage
        self.trading_fee_rate = trading_fee_rate  # 交易手续费率（0.0275% = 0.000275）
        self.positions = {}  # 当前持仓
        self.trades = []  # 交易记录

        # RLS相关参数
        self.use_rls = use_rls
        self.rls_lambda = rls_lambda
        self.rls_max_change_rate = rls_max_change_rate
        self.cointegration_check_interval = cointegration_check_interval
        self.data_period_minutes = data_period_minutes
        self.diff_order = diff_order
        self.cointegration_window_size = cointegration_window_size  # 协整检验窗口大小（与初始筛选一致）

        # RLS实例字典（每个币对一个RLS实例）
        self.rls_instances = {}

        # 协整状态跟踪（每个币对的协整状态）
        self.cointegration_status = {}  # {pair_key: {'is_cointegrated': bool, 'last_check_index': int, 'cointegration_ratio': float}}

        # 设置Z-score策略
        if z_score_strategy is not None:
            # 使用提供的策略对象
            self.z_score_strategy = z_score_strategy
            self.use_arima_garch = isinstance(z_score_strategy,
                                              ArimaGarchZScoreStrategy) if STRATEGIES_AVAILABLE else False
            self.use_ecm = isinstance(z_score_strategy, EcmZScoreStrategy) if STRATEGIES_AVAILABLE else False
            self.use_kalman_filter = isinstance(z_score_strategy,
                                                KalmanFilterZScoreStrategy) if STRATEGIES_AVAILABLE else False
            self.use_copula_dcc_garch = isinstance(z_score_strategy,
                                                   CopulaDccGarchZScoreStrategy) if STRATEGIES_AVAILABLE else False
            self.use_regime_switching = isinstance(z_score_strategy,
                                                   RegimeSwitchingZScoreStrategy) if STRATEGIES_AVAILABLE else False
        elif use_arima_garch and STRATEGIES_AVAILABLE and ARIMA_AVAILABLE and GARCH_AVAILABLE:
            # 向后兼容：使用ARIMA-GARCH策略
            try:
                self.z_score_strategy = ArimaGarchZScoreStrategy(arima_order=arima_order, garch_order=garch_order)
                self.use_arima_garch = True
            except Exception as e:
                print(f"警告: ARIMA-GARCH策略初始化失败: {str(e)}，使用传统策略")
                self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None
                self.use_arima_garch = False
        else:
            # 使用传统策略
            self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None
            self.use_arima_garch = False
            self.use_ecm = False
            self.use_kalman_filter = False
            self.use_copula_dcc_garch = False
            self.use_regime_switching = False

        # 向后兼容：保留旧属性
        self.arima_order = arima_order
        self.garch_order = garch_order
        self._arima_garch_models = {}  # 保留以兼容旧代码

    def calculate_current_spread(self, price1, price2, hedge_ratio):
        """计算当前价差（原序列）"""
        return price1 - hedge_ratio * price2

    def calculate_position_size_beta_neutral(self, available_capital, price1, price2, hedge_ratio, signal):
        """
        基于Beta中性计算开仓数量

        Args:
            available_capital: 可用资金
            price1: symbol1价格
            price2: symbol2价格
            hedge_ratio: 对冲比率
            signal: 交易信号

        Returns:
            (symbol1_size, symbol2_size, total_capital_used) 或 (None, None, 0) 如果计算失败
        """
        if available_capital <= 0:
            return None, None, 0

        # 计算总资金占用系数
        # 总资金占用 = |symbol1_size| × (price1 + hedge_ratio × price2)
        capital_coefficient = price1 + hedge_ratio * price2

        if capital_coefficient <= 0:
            return None, None, 0

        # 计算symbol1的数量（绝对值）
        symbol1_size_abs = available_capital / capital_coefficient

        # 计算symbol2的数量（绝对值）
        symbol2_size_abs = hedge_ratio * symbol1_size_abs

        # 根据信号方向确定正负
        if signal['action'] == 'SHORT_LONG':
            symbol1_size = -symbol1_size_abs  # 做空
            symbol2_size = +symbol2_size_abs  # 做多
        elif signal['action'] == 'LONG_SHORT':
            symbol1_size = +symbol1_size_abs  # 做多
            symbol2_size = -symbol2_size_abs  # 做空
        else:
            return None, None, 0

        # 计算实际资金占用
        total_capital_used = abs(symbol1_size) * price1 + abs(symbol2_size) * price2

        return symbol1_size, symbol2_size, total_capital_used

    def initialize_rls_for_pair(self, pair_key, initial_price1, initial_price2):
        """
        为币对初始化RLS

        Args:
            pair_key: 币对标识（如 'BTCUSDT_ETHUSDT'）
            initial_price1: 初始价格序列1
            initial_price2: 初始价格序列2
        """
        if not self.use_rls:
            return

        try:
            rls = RecursiveLeastSquares(
                lambda_forgetting=self.rls_lambda,
                initial_covariance=1000.0,
                max_change_rate=self.rls_max_change_rate
            )
            rls.initialize(initial_price1, initial_price2)
            self.rls_instances[pair_key] = rls

            # 初始化协整状态
            self.cointegration_status[pair_key] = {
                'is_cointegrated': True,
                'last_check_index': 0,
                'cointegration_ratio': 1.0,  # 初始假设协整
                'last_hedge_ratio': rls.get_hedge_ratio(),
                'consecutive_failures': 0  # 连续失败计数
            }
        except Exception as e:
            print(f"警告: 为币对 {pair_key} 初始化RLS失败: {str(e)}")

    def update_rls_for_pair(self, pair_key, price1_t, price2_t):
        """
        更新币对的RLS对冲比率

        Args:
            pair_key: 币对标识
            price1_t: 当前价格1
            price2_t: 当前价格2

        Returns:
            float: 更新后的对冲比率，如果失败返回None
        """
        if not self.use_rls or pair_key not in self.rls_instances:
            return None

        try:
            rls = self.rls_instances[pair_key]
            hedge_ratio = rls.update(price1_t, price2_t)

            # 更新协整状态中的对冲比率
            if pair_key in self.cointegration_status:
                self.cointegration_status[pair_key]['last_hedge_ratio'] = hedge_ratio

            return hedge_ratio
        except Exception as e:
            print(f"警告: 更新币对 {pair_key} 的RLS失败: {str(e)}")
            return None

    def check_cointegration_periodically(self, pair_key, price1_series, price2_series,
                                         current_index, symbol1, symbol2):
        """
        定期进行协整检验

        Args:
            pair_key: 币对标识
            price1_series: 价格序列1
            price2_series: 价格序列2
            current_index: 当前数据索引
            symbol1: 币种1名称
            symbol2: 币种2名称

        Returns:
            dict: 协整检验结果
        """
        if pair_key not in self.cointegration_status:
            return {'is_cointegrated': False, 'cointegration_ratio': 0.0}

        status = self.cointegration_status[pair_key]
        last_check = status['last_check_index']

        # 检查是否需要重新检验（每N个数据点检验一次）
        if current_index - last_check < self.cointegration_check_interval:
            # 不需要检验，返回当前状态
            return {
                'is_cointegrated': status['is_cointegrated'],
                'cointegration_ratio': status.get('cointegration_ratio', 1.0)
            }

        # 需要重新检验
        print(f"\n{'=' * 60}")
        print(f"定期协整检验: {symbol1}/{symbol2} (索引: {current_index})")
        print(f"{'=' * 60}")

        # 使用与初始筛选相同的窗口大小进行协整检验
        # 如果可用数据不足，使用可用数据的80%，但至少需要100个数据点
        max_window_size = min(len(price1_series), len(price2_series))
        target_window_size = self.cointegration_window_size

        # 如果可用数据少于目标窗口大小，使用可用数据的80%
        if max_window_size < target_window_size:
            window_size = max(100, int(max_window_size * 0.8))  # 至少100个数据点
            print(f"  可用数据不足，使用可用数据的80%: {window_size} 个数据点（目标: {target_window_size}）")
        else:
            window_size = target_window_size
            print(f"  使用窗口大小: {window_size} 个数据点（与初始筛选一致）")

        if window_size < 100:  # 至少需要100个数据点
            print(f"  数据不足，跳过协整检验（需要至少100个数据点，当前{window_size}个）")
            # 如果数据不足，保持当前状态，但标记为需要更多数据
            return {
                'is_cointegrated': status['is_cointegrated'],
                'cointegration_ratio': status.get('cointegration_ratio', 1.0)
            }

        # 获取最近的数据
        recent_price1 = price1_series.iloc[-window_size:] if hasattr(price1_series, 'iloc') else price1_series[
            -window_size:]
        recent_price2 = price2_series.iloc[-window_size:] if hasattr(price2_series, 'iloc') else price2_series[
            -window_size:]

        # 执行协整检验
        try:
            coint_result = enhanced_cointegration_test(
                recent_price1, recent_price2, symbol1, symbol2,
                verbose=False, diff_order=self.diff_order
            )

            is_cointegrated = coint_result.get('cointegration_found', False)
            spread_adf = coint_result.get('spread_adf', {})
            p_value = spread_adf.get('p_value', 1.0) if spread_adf else 1.0

            print(f"  协整检验结果: {'通过' if is_cointegrated else '失败'}")
            print(f"  ADF P值: {p_value:.6f}")

            # 更新协整状态
            status['is_cointegrated'] = is_cointegrated
            status['last_check_index'] = current_index

            if is_cointegrated:
                print(f"   协整检验通过: {symbol1}/{symbol2} 仍然协整")
                # 如果之前失败过，现在恢复了，重置协整比率
                if not status.get('is_cointegrated', True):
                    print(f"   协整关系已恢复！")
                status['cointegration_ratio'] = 1.0  # 假设通过检验时比率为1.0
                status['consecutive_failures'] = 0  # 重置连续失败计数
            else:
                # 增加连续失败计数
                consecutive_failures = status.get('consecutive_failures', 0) + 1
                status['consecutive_failures'] = consecutive_failures

                print(f"  协整检验失败: {symbol1}/{symbol2} 协整关系破裂！")
                print(f"  连续失败次数: {consecutive_failures}")
                print(f"    将在 {self.cointegration_check_interval} 个数据点后重新检验")

                # 根据连续失败次数决定协整比率
                # 第1次失败：降低到0.5
                # 第2次失败：降低到0.2
                # 第3次及以上：降低到0（完全暂停）
                if consecutive_failures == 1:
                    status['cointegration_ratio'] = 0.5
                    print(f"    协整比率降低至50%，减少交易仓位")
                elif consecutive_failures == 2:
                    status['cointegration_ratio'] = 0.2
                    print(f"    协整比率降低至20%，大幅减少交易仓位")
                else:
                    status['cointegration_ratio'] = 0.0
                    print(f"    协整比率降至0%，完全暂停交易，等待协整关系修复...")

            return {
                'is_cointegrated': is_cointegrated,
                'cointegration_ratio': status.get('cointegration_ratio', 0.0),
                'coint_result': coint_result
            }

        except Exception as e:
            print(f"  协整检验出错: {str(e)}")
            # 检验出错时，保持当前状态，不改变协整状态
            return {
                'is_cointegrated': status['is_cointegrated'],
                'cointegration_ratio': status.get('cointegration_ratio', 1.0)
            }

    def adjust_position_by_cointegration_ratio(self, base_position_ratio, cointegration_ratio):
        """
        根据协整比率调整仓位

        Args:
            base_position_ratio: 基础仓位比例
            cointegration_ratio: 协整比率（0-1）

        Returns:
            float: 调整后的仓位比例
        """
        # 协整比率越高，仓位越大
        # 协整比率 < 0.2: 不使用该币对
        # 协整比率 0.2-0.4: 使用50%的基础仓位
        # 协整比率 0.4-0.6: 使用75%的基础仓位
        # 协整比率 > 0.6: 使用100%的基础仓位

        if cointegration_ratio < 0.2:
            return 0.0
        elif cointegration_ratio < 0.4:
            return base_position_ratio * 0.5
        elif cointegration_ratio < 0.6:
            return base_position_ratio * 0.75
        else:
            return base_position_ratio

    def calculate_z_score(self, current_spread, historical_spreads,
                          historical_prices1=None, historical_prices2=None):
        """
        计算当前Z-score（使用策略对象）

        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            historical_prices1: 第一个资产的历史价格序列（可选）
            historical_prices2: 第二个资产的历史价格序列（可选）

        Returns:
            float: Z-score值
        """
        # 如果使用了策略对象，调用策略的方法
        if self.z_score_strategy is not None:
            return self.z_score_strategy.calculate_z_score(current_spread, historical_spreads,
                                                           historical_prices1, historical_prices2)

        # 向后兼容：如果没有策略对象，使用旧方法
        if self.use_arima_garch:
            return self.calculate_z_score_with_arima_garch(current_spread, historical_spreads)
        else:
            return self.calculate_z_score_traditional(current_spread, historical_spreads)

    def calculate_z_score_traditional(self, current_spread, historical_spreads):
        """计算当前Z-score（传统方法：使用均值和标准差）- 向后兼容方法"""
        if len(historical_spreads) < 2:
            return 0

        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)

        if spread_std == 0:
            return 0

        return (current_spread - spread_mean) / spread_std

    def calculate_z_score_with_arima_garch(self, current_spread, historical_spreads):
        """
        使用ARIMA-GARCH模型计算Z-score

        步骤：
        1. 使用ARIMA模型对历史价差序列建模，预测当前价差的均值
        2. 计算ARIMA模型的残差
        3. 使用GARCH模型对残差建模，预测当前价差的波动率
        4. 使用预测的均值和波动率计算Z-score

        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列（列表或数组）

        Returns:
            float: Z-score值
        """
        if len(historical_spreads) < max(20, sum(self.arima_order) + 5):
            # 数据不足，回退到传统方法
            return self.calculate_z_score_traditional(current_spread, historical_spreads)

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
                try:
                    arima_model = ARIMA(spreads_array, order=self.arima_order)
                    arima_fitted = arima_model.fit()
                except Exception as e:
                    # ARIMA拟合失败，回退到传统方法
                    return self.calculate_z_score_traditional(current_spread, historical_spreads)

                # 步骤2: 获取ARIMA残差
                arima_residuals = arima_fitted.resid

                # 确保残差有足够的数据
                if len(arima_residuals) < max(10, sum(self.garch_order) + 3):
                    return self.calculate_z_score_traditional(current_spread, historical_spreads)

                # 步骤3: 拟合GARCH模型
                try:
                    garch_model = arch_model(arima_residuals, vol='Garch', p=self.garch_order[0], q=self.garch_order[1])
                    garch_fitted = garch_model.fit(disp='off')
                except Exception as e:
                    # GARCH拟合失败，回退到传统方法
                    return self.calculate_z_score_traditional(current_spread, historical_spreads)

                # 缓存模型（限制缓存大小）
                if len(self._arima_garch_models) < 10:
                    self._arima_garch_models[cache_key] = (arima_fitted, garch_fitted)
                else:
                    # 清除最旧的缓存
                    oldest_key = next(iter(self._arima_garch_models))
                    del self._arima_garch_models[oldest_key]
                    self._arima_garch_models[cache_key] = (arima_fitted, garch_fitted)

            # 步骤4: 预测当前价差的均值（ARIMA）
            try:
                arima_forecast = arima_fitted.forecast(steps=1)
                # 处理不同的返回格式
                if hasattr(arima_forecast, 'iloc'):
                    predicted_mean = arima_forecast.iloc[0]
                elif isinstance(arima_forecast, (list, np.ndarray)):
                    predicted_mean = float(arima_forecast[0])
                else:
                    predicted_mean = float(arima_forecast)
            except Exception as e:
                # 预测失败，使用最后一个值
                predicted_mean = spreads_array[-1]

            # 步骤5: 预测当前价差的波动率（GARCH）
            try:
                garch_forecast = garch_fitted.forecast(horizon=1)
                predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, 0])
                if predicted_volatility <= 0 or np.isnan(predicted_volatility):
                    predicted_volatility = np.std(spreads_array)
            except Exception as e:
                # 预测失败，使用历史标准差
                predicted_volatility = np.std(spreads_array)

            # 步骤6: 计算Z-score
            if predicted_volatility == 0:
                return 0

            z_score = (current_spread - predicted_mean) / predicted_volatility

            return z_score

        except Exception as e:
            # 任何错误都回退到传统方法
            return self.calculate_z_score_traditional(current_spread, historical_spreads)

    def generate_trading_signal(self, z_score):
        """生成交易信号"""
        if z_score > self.z_threshold:
            return {
                'action': 'SHORT_LONG',
                'description': f'Z-score过高({z_score:.3f})，做空价差',
                'confidence': min(abs(z_score) / 3.0, 1.0)
            }
        elif z_score < -self.z_threshold:
            return {
                'action': 'LONG_SHORT',
                'description': f'Z-score过低({z_score:.3f})，做多价差',
                'confidence': min(abs(z_score) / 3.0, 1.0)
            }
        else:
            return {
                'action': 'HOLD',
                'description': f'Z-score正常({z_score:.3f})，观望',
                'confidence': 0.0
            }

    def execute_trade(self, pair_info, current_prices, signal, timestamp, current_spread, available_capital):
        """
        执行交易（基于可用资金计算仓位）

        Args:
            pair_info: 币对信息
            current_prices: 当前价格字典
            signal: 交易信号
            timestamp: 时间戳
            current_spread: 当前价差
            available_capital: 可用资金
        """
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        hedge_ratio = pair_info['hedge_ratio']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]

        # 使用Beta中性方法计算开仓数量
        symbol1_size, symbol2_size, total_capital_used = self.calculate_position_size_beta_neutral(
            available_capital, price1, price2, hedge_ratio, signal
        )

        if symbol1_size is None:
            print(f"开仓失败: 可用资金不足或计算错误 (可用资金: {available_capital:.2f})")
            return None

        # 计算开仓手续费（基于交易金额）
        # 手续费 = (|symbol1_size| × price1 + |symbol2_size| × price2) × fee_rate
        open_fee = total_capital_used * self.trading_fee_rate

        # 创建持仓记录
        position = {
            'pair': f"{symbol1}_{symbol2}",
            'symbol1': symbol1,
            'symbol2': symbol2,
            'symbol1_size': symbol1_size,
            'symbol2_size': symbol2_size,
            'entry_prices': {symbol1: price1, symbol2: price2},
            'entry_spread': current_spread,  # 记录开仓时的价差
            'hedge_ratio': hedge_ratio,
            'entry_time': timestamp,
            'signal': signal,
            'capital_used': total_capital_used,  # 记录使用的资金
            'open_fee': open_fee  # 记录开仓手续费
        }

        self.positions[pair_info['pair_name']] = position

        # 记录开仓交易
        trade = {
            'timestamp': timestamp,
            'pair': pair_info['pair_name'],
            'action': 'OPEN',
            'symbol1': symbol1,
            'symbol2': symbol2,
            'symbol1_size': position['symbol1_size'],
            'symbol2_size': position['symbol2_size'],
            'symbol1_price': price1,
            'symbol2_price': price2,
            'hedge_ratio': hedge_ratio,
            'signal': signal,
            'z_score': signal.get('z_score', 0),
            'entry_spread': current_spread,
            'capital_used': total_capital_used,
            'fee': open_fee  # 记录开仓手续费
        }
        self.trades.append(trade)

        print(f"开仓: {pair_info['pair_name']}")
        print(f"   信号: {signal['description']}")
        print(f"   价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
        print(f"   价差: {current_spread:.6f}")
        print(f"   仓位: {symbol1}={position['symbol1_size']:.6f}, {symbol2}={position['symbol2_size']:.6f}")
        print(f"   使用资金: {total_capital_used:.2f} / {available_capital:.2f}")
        print(f"   开仓手续费: {open_fee:.4f}")

        return position

    def check_exit_conditions(self, pair_info, current_prices, current_z_score, timestamp, current_spread):
        """检查平仓条件（包含止盈止损）"""
        pair_name = pair_info['pair_name']
        if pair_name not in self.positions:
            return False, ""

        position = self.positions[pair_name]
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]

        # 计算实际盈亏（基于持仓价值变化）
        entry_price1 = position['entry_prices'][symbol1]
        entry_price2 = position['entry_prices'][symbol2]

        if position['signal']['action'] == 'SHORT_LONG':
            # 做空symbol1，做多symbol2
            # 做空价差，价差减少时盈利
            # symbol1_size是负数（做空），price1下跌时(price1-entry_price1)为负，负数×负数=正数（盈利）
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            # symbol2_size是正数（做多），price2上涨时(price2-entry_price2)为正，正数×正数=正数（盈利）
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        else:  # LONG_SHORT
            # 做多symbol1，做空symbol2
            # 做多价差，价差增加时盈利
            # symbol1_size是正数（做多），price1上涨时(price1-entry_price1)为正，正数×正数=正数（盈利）
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            # symbol2_size是负数（做空），price2下跌时(price2-entry_price2)为负，负数×负数=正数（盈利）
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2

        # 计算投入资金（基于原始价格）
        entry_value = abs(position['symbol1_size'] * entry_price1) + \
                      abs(position['symbol2_size'] * entry_price2)

        # 计算手续费（用于止盈止损判断）
        # 计算平仓手续费
        close_fee = (abs(position['symbol1_size']) * price1 + abs(
            position['symbol2_size']) * price2) * self.trading_fee_rate
        open_fee = position.get('open_fee', 0)
        total_fee = open_fee + close_fee

        # 计算净盈亏（只扣除平仓手续费，因为开仓手续费已经在开仓时从capital中扣除了）
        net_pnl = total_pnl - close_fee

        # 条件1: Z-score回归到均值附近
        if abs(current_z_score) < self.z_exit_threshold:
            return True, f"Z-score回归到{current_z_score:.3f}，平仓获利"

        # 条件2: 持仓时间过长
        holding_hours = (timestamp - position['entry_time']).total_seconds() / 3600
        if holding_hours > self.max_holding_hours:
            return True, f"持仓时间过长({holding_hours:.1f}小时)，强制平仓"

        # 条件3: 止盈条件（基于净盈亏）
        if entry_value > 0:
            pnl_percentage = net_pnl / entry_value
            if net_pnl > 0 and pnl_percentage > self.take_profit_pct:
                return True, f"止盈触发({pnl_percentage * 100:.1f}%)，平仓获利"

        # 条件4: 止损条件（基于净盈亏）
        if entry_value > 0:
            pnl_percentage = net_pnl / entry_value
            if net_pnl < 0 and pnl_percentage < -self.stop_loss_pct:
                return True, f"止损触发({pnl_percentage * 100:.1f}%)，平仓止损"

        return False, ""

    def close_position(self, pair_info, current_prices, reason, timestamp, current_spread):
        """平仓"""
        pair_name = pair_info['pair_name']
        if pair_name not in self.positions:
            return None

        position = self.positions[pair_name]
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]

        # 计算最终盈亏（基于持仓价值变化）
        entry_price1 = position['entry_prices'][symbol1]
        entry_price2 = position['entry_prices'][symbol2]

        if position['signal']['action'] == 'SHORT_LONG':
            # 做空symbol1，做多symbol2
            # 做空价差，价差减少时盈利
            # symbol1_size是负数（做空），price1下跌时(price1-entry_price1)为负，负数×负数=正数（盈利）
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            # symbol2_size是正数（做多），price2上涨时(price2-entry_price2)为正，正数×正数=正数（盈利）
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        else:  # LONG_SHORT
            # 做多symbol1，做空symbol2
            # 做多价差，价差增加时盈利
            # symbol1_size是正数（做多），price1上涨时(price1-entry_price1)为正，正数×正数=正数（盈利）
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            # symbol2_size是负数（做空），price2下跌时(price2-entry_price2)为负，负数×负数=正数（盈利）
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2

        # 计算平仓手续费（基于平仓时的交易金额）
        # 手续费 = (|symbol1_size| × price1 + |symbol2_size| × price2) × fee_rate
        close_fee = (abs(position['symbol1_size']) * price1 + abs(
            position['symbol2_size']) * price2) * self.trading_fee_rate

        # 计算总手续费（开仓手续费 + 平仓手续费，用于记录和显示）
        open_fee = position.get('open_fee', 0)
        total_fee = open_fee + close_fee

        # 扣除手续费后的净盈亏（只扣除平仓手续费，因为开仓手续费已经在开仓时从capital中扣除了）
        net_pnl = total_pnl - close_fee

        # 计算价差变化（用于显示）
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread

        # 记录平仓交易
        trade = {
            'timestamp': timestamp,
            'pair': pair_name,
            'action': 'CLOSE',
            'symbol1': symbol1,
            'symbol2': symbol2,
            'symbol1_size': -position['symbol1_size'],
            'symbol2_size': -position['symbol2_size'],
            'symbol1_price': price1,
            'symbol2_price': price2,
            'hedge_ratio': position['hedge_ratio'],
            'signal': {'action': 'CLOSE', 'description': reason},
            'pnl': net_pnl,  # 净盈亏（已扣除手续费）
            'gross_pnl': total_pnl,  # 毛盈亏（未扣除手续费）
            'open_fee': open_fee,  # 开仓手续费
            'close_fee': close_fee,  # 平仓手续费
            'total_fee': total_fee,  # 总手续费
            'holding_hours': (timestamp - position['entry_time']).total_seconds() / 3600
        }
        self.trades.append(trade)

        print(f"平仓: {pair_name}")
        print(f"   平仓原因: {reason}")
        print(f"   毛盈亏: {total_pnl:.2f}")
        print(f"   开仓手续费: {open_fee:.4f}")
        print(f"   平仓手续费: {close_fee:.4f}")
        print(f"   总手续费: {total_fee:.4f}")
        print(f"   净盈亏: {net_pnl:.2f}")
        print(f"   持仓时间: {trade['holding_hours']:.1f}小时")
        print(f"   开仓价格: {symbol1}={entry_price1:.2f}, {symbol2}={entry_price2:.2f}")
        print(f"   平仓价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
        print(f"   价差变化: {entry_spread:.6f} -> {current_spread:.6f} (变化: {spread_change:.6f})")

        # 移除持仓
        del self.positions[pair_name]

        return trade

    def calculate_risk_metrics(self, capital_curve):
        """计算风险指标"""
        if len(capital_curve) < 2:
            return {}

        # 计算收益率
        returns = []
        for i in range(1, len(capital_curve)):
            prev_capital = capital_curve[i - 1]['capital']
            curr_capital = capital_curve[i]['capital']
            ret = (curr_capital - prev_capital) / prev_capital
            returns.append(ret)

        returns = np.array(returns)

        # 计算最大回撤
        capital_values = [point['capital'] for point in capital_curve]
        peak = capital_values[0]
        max_drawdown = 0
        max_drawdown_pct = 0

        for capital in capital_values:
            if capital > peak:
                peak = capital
            drawdown = peak - capital
            drawdown_pct = drawdown / peak
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        # 计算盈亏比
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]

        avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

        # 自动检测数据频率并计算年化系数
        periods_per_year = 252  # 默认日线
        if len(capital_curve) >= 2:
            try:
                # 获取时间戳
                timestamps = [point['timestamp'] for point in capital_curve]
                # 计算前10个时间间隔的平均值（如果数据点少于10个，则使用全部）
                sample_size = min(10, len(timestamps) - 1)
                time_diffs = []
                for i in range(1, sample_size + 1):
                    if isinstance(timestamps[i], pd.Timestamp) and isinstance(timestamps[i - 1], pd.Timestamp):
                        diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600  # 转换为小时
                        time_diffs.append(diff)

                if time_diffs:
                    avg_hours_per_point = np.mean(time_diffs)

                    # 根据平均时间间隔判断数据频率
                    if avg_hours_per_point < 0.1:  # 小于6分钟，可能是分钟线
                        periods_per_year = 252 * 24 * 60  # 分钟线（假设1分钟）
                    elif avg_hours_per_point < 2:  # 小于2小时，可能是小时线
                        periods_per_year = 252 * 24  # 小时线
                    elif avg_hours_per_point < 12:  # 小于12小时，可能是4小时线
                        periods_per_year = 252 * 6  # 4小时线
                    elif avg_hours_per_point < 24:  # 小于24小时，可能是日线
                        periods_per_year = 252  # 日线
                    else:  # 大于24小时，可能是周线或更长
                        periods_per_year = 52  # 周线
            except Exception as e:
                # 如果检测失败，使用默认值（日线）
                periods_per_year = 252

        # 计算夏普比率（使用自动检测的年化系数）
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct * 100,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades)
        }

    def plot_equity_curve(self, capital_curve, save_path=None):
        """绘制收益率曲线图"""
        if len(capital_curve) < 2:
            print("数据点不足，无法绘制曲线")
            return

        print(f"正在处理 {len(capital_curve)} 个数据点...")

        timestamps = [point['timestamp'] for point in capital_curve]
        capitals = [point['capital'] for point in capital_curve]

        # 计算收益率
        initial_capital = capitals[0]
        returns = [(cap - initial_capital) / initial_capital * 100 for cap in capitals]

        print("正在创建图表...")

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 资金曲线
        print("正在绘制资金曲线...")
        ax1.plot(timestamps, capitals, linewidth=1.5, color='blue', alpha=0.8)
        ax1.set_title('资金曲线', fontsize=14, fontweight='bold')
        ax1.set_ylabel('资金 (元)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 收益率曲线
        print("正在绘制收益率曲线...")
        ax2.plot(timestamps, returns, linewidth=1.5, color='green', alpha=0.8)
        ax2.set_title('收益率曲线', fontsize=14, fontweight='bold')
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 添加零线
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        print("正在格式化时间轴...")
        # 简化时间轴格式化
        if len(timestamps) > 100:
            # 数据点多时，减少时间轴标签
            step = max(1, len(timestamps) // 10)
            for ax in [ax1, ax2]:
                ax.set_xticks(timestamps[::step])
                ax.tick_params(axis='x', rotation=45)
        else:
            # 数据点少时，正常格式化
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)

        print("正在调整布局...")
        plt.tight_layout()

        if save_path:
            print(f"正在保存图片到: {save_path}")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"收益率曲线图已保存到: {save_path}")

        print("正在显示图表...")
        try:
            plt.show(block=True)
            print("图表显示完成")
        except Exception as e:
            print(f"图表显示失败: {str(e)}")
            print("尝试保存图片...")
            plt.savefig('equity_curve_backup.png', dpi=200, bbox_inches='tight')
            print("图片已保存为 equity_curve_backup.png")
            plt.close()

    def backtest_cointegration_trading(self, data, selected_pairs, initial_capital=10000):
        """回测协整交易策略（使用原序列价差，支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）"""
        optimization_mode = hasattr(self, '_optimization_mode') and self._optimization_mode

        if not optimization_mode:
            print("\n" + "=" * 60)
            print("开始高级协整交易回测")
            if self.use_arima_garch:
                print("（使用ARIMA-GARCH模型）")
            elif self.use_ecm:
                print("（使用ECM误差修正模型）")
            elif self.use_kalman_filter:
                print("（使用Kalman Filter动态价差模型）")
            elif self.use_copula_dcc_garch:
                print("（使用Copula + DCC-GARCH相关性/波动率模型）")
            print("=" * 60)

        # 初始化
        # 确定投入资金：capital = initial_capital * position_ratio
        # position_ratio只在初始化时使用一次，之后不再使用
        capital = initial_capital * self.position_ratio
        results = {
            'capital_curve': [],
            'trades': [],
            'daily_returns': [],
            'positions': {},
            'pair_details': {}
        }

        # 获取所有时间点
        all_timestamps = set()
        for symbol in data.keys():
            all_timestamps.update(data[symbol].index)
        all_timestamps = sorted(list(all_timestamps))

        if not optimization_mode:
            print(f"回测时间范围: {all_timestamps[0]} 到 {all_timestamps[-1]}")
            print(f"总数据点: {len(all_timestamps)}")
            print(f"选择的币对数量: {len(selected_pairs)}")
            print(f"策略参数:")
            print(f"  初始资金: {initial_capital:,.2f}")
            print(f"  仓位比例: {self.position_ratio * 100:.1f}%")
            print(f"  投入资金: {capital:,.2f} (初始资金 × 仓位比例)")
            print(f"  杠杆: {self.leverage:.1f}倍")
            print(f"  可用资金: {capital * self.leverage:,.2f} (投入资金 × 杠杆)")
            print(f"  交易手续费率: {self.trading_fee_rate * 100:.4f}%")
            print(f"  Z-score开仓阈值: {self.z_threshold}")
            print(f"  Z-score平仓阈值: {self.z_exit_threshold}")
            print(f"  止盈百分比: {self.take_profit_pct * 100:.1f}%")
            print(f"  止损百分比: {self.stop_loss_pct * 100:.1f}%")
            print(f"  最大持仓时间: {self.max_holding_hours}小时")
            print(f"  使用RLS: {'是' if self.use_rls else '否'}")
            if self.use_rls:
                print(f"  RLS遗忘因子: {self.rls_lambda}")
                print(
                    f"  协整检验间隔: {self.cointegration_check_interval} 个数据点 (约{self.cointegration_check_interval * self.data_period_minutes / 60 / 24:.1f}天)")
            # 显示使用的策略
            if self.z_score_strategy is not None:
                strategy_name = self.z_score_strategy.get_strategy_name() if hasattr(self.z_score_strategy,
                                                                                     'get_strategy_name') else "策略对象"
                print(f"  Z-score策略: {strategy_name}")
            else:
                print(f"  使用ARIMA-GARCH: {'是' if self.use_arima_garch else '否'}")
                print(f"  使用ECM: {'是' if self.use_ecm else '否'}")
                print(f"  使用Kalman Filter: {'是' if self.use_kalman_filter else '否'}")
                print(f"  使用Copula + DCC-GARCH: {'是' if self.use_copula_dcc_garch else '否'}")

        # 初始化RLS（如果使用）
        if self.use_rls:
            print("\n初始化RLS...")
            for pair_info in selected_pairs:
                symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
                pair_key = f"{symbol1}_{symbol2}"

                if symbol1 not in data or symbol2 not in data:
                    continue

                # 使用初始窗口的数据初始化RLS
                init_window_size = min(self.lookback_period, len(data[symbol1]), len(data[symbol2]))
                if init_window_size < 30:
                    print(f"  警告: {pair_key} 数据不足，无法初始化RLS")
                    continue

                init_price1 = data[symbol1].iloc[:init_window_size] if hasattr(data[symbol1], 'iloc') else data[
                    symbol1][:init_window_size]
                init_price2 = data[symbol2].iloc[:init_window_size] if hasattr(data[symbol2], 'iloc') else data[
                    symbol2][:init_window_size]

                self.initialize_rls_for_pair(pair_key, init_price1, init_price2)
                print(f"  ✓ {pair_key} RLS初始化完成")

        # 回测循环
        for i, timestamp in enumerate(all_timestamps):
            # 获取当前价格
            current_prices = {}
            for symbol in data.keys():
                if timestamp in data[symbol].index:
                    current_prices[symbol] = data[symbol].loc[timestamp]

            # 检查每个选择的币对
            for pair_info in selected_pairs:
                symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
                pair_key = f"{symbol1}_{symbol2}"

                if symbol1 not in current_prices or symbol2 not in current_prices:
                    continue

                # 获取或更新对冲比率（使用RLS或静态值）
                if self.use_rls and pair_key in self.rls_instances:
                    # 更新RLS对冲比率
                    current_hedge_ratio = self.update_rls_for_pair(
                        pair_key, current_prices[symbol1], current_prices[symbol2]
                    )
                    if current_hedge_ratio is None:
                        current_hedge_ratio = pair_info.get('hedge_ratio', 1.0)
                else:
                    # 使用静态对冲比率
                    current_hedge_ratio = pair_info.get('hedge_ratio', 1.0)

                # 定期协整检验
                if self.use_rls and pair_key in self.cointegration_status:
                    price1_series = data[symbol1].iloc[:i + 1] if hasattr(data[symbol1], 'iloc') else data[symbol1][
                        :i + 1]
                    price2_series = data[symbol2].iloc[:i + 1] if hasattr(data[symbol2], 'iloc') else data[symbol2][
                        :i + 1]

                    coint_check_result = self.check_cointegration_periodically(
                        pair_key, price1_series, price2_series, i, symbol1, symbol2
                    )

                    # 如果协整关系破裂，根据协整比率决定是否交易
                    # 如果协整比率为0，完全暂停交易
                    cointegration_ratio = coint_check_result.get('cointegration_ratio', 1.0)
                    if cointegration_ratio <= 0:
                        # 完全暂停交易
                        continue
                    # 如果协整比率很低（<0.2），也暂停交易
                    elif cointegration_ratio < 0.2:
                        continue

                # 根据diff_order计算价差
                diff_order = pair_info.get('diff_order', self.diff_order)

                if diff_order == 0:
                    # 原始价差（使用当前对冲比率）
                    current_spread = self.calculate_current_spread(
                        current_prices[symbol1],
                        current_prices[symbol2],
                        current_hedge_ratio
                    )

                    # 获取历史原始价差数据（使用当前对冲比率或历史RLS值）
                    historical_spreads = []
                    historical_prices1 = []
                    historical_prices2 = []
                    for j in range(max(0, i - self.lookback_period), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index):
                                # 如果使用RLS，尝试获取历史对冲比率，否则使用当前值
                                hist_hedge_ratio = current_hedge_ratio
                                if self.use_rls and pair_key in self.rls_instances:
                                    rls = self.rls_instances[pair_key]
                                    if len(rls.beta_history) > (i - j):
                                        hist_hedge_ratio = rls.beta_history[-(i - j)][1]

                                hist_spread = self.calculate_current_spread(
                                    data[symbol1].loc[hist_timestamp],
                                    data[symbol2].loc[hist_timestamp],
                                    hist_hedge_ratio
                                )
                                historical_spreads.append(hist_spread)
                                historical_prices1.append(data[symbol1].loc[hist_timestamp])
                                historical_prices2.append(data[symbol2].loc[hist_timestamp])
                else:
                    # 一阶差分价差
                    if i > 0:  # 确保有前一个时间点
                        prev_timestamp = all_timestamps[i - 1]
                        if (prev_timestamp in data[symbol1].index and
                                prev_timestamp in data[symbol2].index):
                            current_diff1 = current_prices[symbol1] - data[symbol1].loc[prev_timestamp]
                            current_diff2 = current_prices[symbol2] - data[symbol2].loc[prev_timestamp]
                            current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                        else:
                            current_spread = 0
                    else:
                        current_spread = 0

                    # 获取历史一阶差分价差数据
                    historical_spreads = []
                    historical_prices1 = []
                    historical_prices2 = []
                    for j in range(max(1, i - self.lookback_period), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            prev_hist_timestamp = all_timestamps[j - 1]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index and
                                    prev_hist_timestamp in data[symbol1].index and
                                    prev_hist_timestamp in data[symbol2].index):
                                hist_diff1 = data[symbol1].loc[hist_timestamp] - data[symbol1].loc[prev_hist_timestamp]
                                hist_diff2 = data[symbol2].loc[hist_timestamp] - data[symbol2].loc[prev_hist_timestamp]
                                hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                historical_spreads.append(hist_spread)
                                historical_prices1.append(data[symbol1].loc[hist_timestamp])
                                historical_prices2.append(data[symbol2].loc[hist_timestamp])

                current_z_score = self.calculate_z_score(current_spread, historical_spreads,
                                                         historical_prices1, historical_prices2)

                # 检查平仓条件
                if pair_info['pair_name'] in self.positions:
                    should_close, close_reason = self.check_exit_conditions(
                        pair_info, current_prices, current_z_score, timestamp, current_spread
                    )

                    if should_close:
                        trade = self.close_position(pair_info, current_prices, close_reason, timestamp, current_spread)
                        if trade:
                            capital += trade['pnl']  # 净盈亏（已扣除手续费）
                            results['trades'].append(trade)

                # 检查开仓条件（只有在没有持仓时才开仓）
                elif len(self.positions) == 0:  # 单边持仓模式
                    # 根据协整比率调整仓位
                    cointegration_ratio = 1.0
                    if pair_key in self.cointegration_status:
                        cointegration_ratio = self.cointegration_status[pair_key].get('cointegration_ratio', 1.0)

                    adjusted_position_ratio = self.adjust_position_by_cointegration_ratio(
                        self.position_ratio, cointegration_ratio
                    )

                    # 如果调整后的仓位为0，跳过交易
                    if adjusted_position_ratio <= 0:
                        continue

                    signal = self.generate_trading_signal(current_z_score)
                    signal['z_score'] = current_z_score

                    if signal['action'] != 'HOLD':
                        # 计算可用资金（根据协整比率调整）
                        # position_ratio只在初始化时使用一次，之后capital已经包含了position_ratio的影响
                        # 这里使用调整后的仓位比例
                        available_capital = capital * self.leverage * (adjusted_position_ratio / self.position_ratio)

                        # 使用当前对冲比率（RLS或静态）
                        pair_info_with_rls = pair_info.copy()
                        pair_info_with_rls['hedge_ratio'] = current_hedge_ratio

                        position = self.execute_trade(pair_info_with_rls, current_prices, signal, timestamp,
                                                      current_spread, available_capital)
                        if position:
                            # 扣除开仓手续费
                            open_fee = self.trades[-1].get('fee', 0)
                            capital -= open_fee
                            results['trades'].append(self.trades[-1])

            # 记录资金曲线
            results['capital_curve'].append({
                'timestamp': timestamp,
                'capital': capital,
                'positions_count': len(self.positions)
            })

        # 从交易记录中提取第一个交易的时间戳，用于调整capital_curve的起始时间
        if len(results['trades']) > 0:
            first_trade_timestamp = results['trades'][0].get('timestamp')
            if first_trade_timestamp:
                # 只保留从第一个交易时间戳开始的capital_curve数据
                filtered_capital_curve = [
                    point for point in results['capital_curve']
                    if point['timestamp'] >= first_trade_timestamp
                ]
                # 如果过滤后的曲线为空，或者第一个点的时间戳不是第一个交易的时间戳，添加初始点
                if not filtered_capital_curve or filtered_capital_curve[0]['timestamp'] != first_trade_timestamp:
                    # 找到第一个交易时间戳对应的资金值（从原始capital_curve中查找）
                    initial_capital_at_first_trade = capital
                    for point in reversed(results['capital_curve']):
                        if point['timestamp'] <= first_trade_timestamp:
                            initial_capital_at_first_trade = point['capital']
                            break
                    filtered_capital_curve.insert(0, {
                        'timestamp': first_trade_timestamp,
                        'capital': initial_capital_at_first_trade,
                        'positions_count': 0
                    })
                results['capital_curve'] = filtered_capital_curve

        # 计算最终结果
        total_trades = len(results['trades'])
        profitable_trades = len([t for t in results['trades'] if t.get('pnl', 0) > 0])

        # 计算总手续费
        # 平仓时的total_fee已经包含了开仓手续费和平仓手续费，所以只需要统计平仓时的total_fee
        total_fees = sum([t.get('total_fee', 0) for t in results['trades'] if t.get('action') == 'CLOSE'])

        # 计算收益率：基于投入资金（capital的初始值）
        initial_invested_capital = initial_capital * self.position_ratio
        final_return = (capital - initial_invested_capital) / initial_invested_capital * 100

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(results['capital_curve'])

        if not optimization_mode:
            print(f"\n回测结果:")
            print(f"  初始资金: {initial_capital:,.2f}")
            print(f"  投入资金: {initial_invested_capital:,.2f} (初始资金 × 仓位比例)")
            print(f"  最终资金: {capital:,.2f}")
            print(f"  总收益率: {final_return:.2f}%")
            print(f"  总交易次数: {total_trades / 2}")
            print(f"  盈利交易: {profitable_trades}")
            print(f"  胜率: {profitable_trades / (total_trades / 2) * 100:.1f}%" if total_trades > 0 else "  胜率: 0%")
            print(f"  总手续费: {total_fees:,.2f}")

            print(f"\n风险指标:")
            print(f"  最大回撤: {risk_metrics.get('max_drawdown', 0):,.2f}")
            print(f"  最大回撤百分比: {risk_metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"  盈亏比: {risk_metrics.get('profit_loss_ratio', 0):.2f}")
            print(f"  夏普比率: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  平均盈利: {risk_metrics.get('avg_profit', 0):.2f}")
            print(f"  平均亏损: {risk_metrics.get('avg_loss', 0):.2f}")

        # 绘制收益率曲线图（仅在非优化模式下）
        if not hasattr(self, '_optimization_mode') or not self._optimization_mode:
            print(f"\n正在生成收益率曲线图...")
            self.plot_equity_curve(results['capital_curve'])

        # 导出交易记录为CSV
        print(f"\n正在导出交易记录...")
        print(f"交易记录数量: {len(results['trades'])}")
        if len(results['trades']) > 0:
            csv_path = self.export_trades_to_csv(results['trades'])
            if csv_path:
                import os
                full_path = os.path.abspath(csv_path)
                print(f" 交易记录已成功导出!")
                print(f"  文件路径: {full_path}")
                print(f"  文件大小: {os.path.getsize(full_path) / 1024:.2f} KB")
            else:
                print(" 导出失败，请检查错误信息")
        else:
            print("没有交易记录可导出")

        return results

    def export_trades_to_csv(self, trades, filename=None):
        """
        将交易记录导出为CSV文件

        Args:
            trades: 交易记录列表
            filename: 输出文件名（可选，默认自动生成）

        Returns:
            str: 导出的CSV文件路径，如果失败返回None
        """
        if not trades:
            print("  警告: 交易记录列表为空")
            return None

        try:
            import os
            from datetime import datetime

            print(f"  处理 {len(trades)} 条交易记录...")

            # 准备数据列表
            trade_data = []

            for i, trade in enumerate(trades):
                # 基础信息
                row = {
                    '时间戳': trade.get('timestamp', ''),
                    '币对': trade.get('pair', ''),
                    '操作': trade.get('action', ''),
                    '币种1': trade.get('symbol1', ''),
                    '币种2': trade.get('symbol2', ''),
                    '币种1数量': trade.get('symbol1_size', 0),
                    '币种2数量': trade.get('symbol2_size', 0),
                    '币种1价格': trade.get('symbol1_price', 0),
                    '币种2价格': trade.get('symbol2_price', 0),
                    '对冲比率': trade.get('hedge_ratio', 0),
                }

                # 处理信号信息（可能是字典）
                signal = trade.get('signal', {})
                if isinstance(signal, dict):
                    row['信号动作'] = signal.get('action', '')
                    row['信号描述'] = signal.get('description', '')
                    row['信号置信度'] = signal.get('confidence', 0)
                else:
                    row['信号动作'] = ''
                    row['信号描述'] = str(signal)
                    row['信号置信度'] = 0

                # 开仓特有信息
                if trade.get('action') == 'OPEN':
                    row['Z-score'] = trade.get('z_score', 0)
                    row['开仓价差'] = trade.get('entry_spread', 0)
                    row['使用资金'] = trade.get('capital_used', 0)
                    row['手续费'] = trade.get('fee', 0)
                    row['盈亏'] = ''
                    row['持仓时间(小时)'] = ''

                # 平仓特有信息
                elif trade.get('action') == 'CLOSE':
                    row['Z-score'] = ''
                    row['开仓价差'] = ''
                    row['使用资金'] = ''
                    row['手续费'] = trade.get('total_fee', 0)
                    row['毛盈亏'] = trade.get('gross_pnl', 0)
                    row['盈亏'] = trade.get('pnl', 0)  # 净盈亏
                    row['持仓时间(小时)'] = trade.get('holding_hours', 0)

                else:
                    row['Z-score'] = trade.get('z_score', '')
                    row['开仓价差'] = trade.get('entry_spread', '')
                    row['使用资金'] = trade.get('capital_used', '')
                    row['盈亏'] = trade.get('pnl', '')
                    row['持仓时间(小时)'] = trade.get('holding_hours', '')

                trade_data.append(row)

            # 创建DataFrame
            df = pd.DataFrame(trade_data)
            print(f"  已创建DataFrame，包含 {len(df)} 行，{len(df.columns)} 列")

            # 生成文件名（使用绝对路径）
            if filename is None:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'trades_export_{timestamp_str}.csv'

            # 确保文件名以.csv结尾
            if not filename.endswith('.csv'):
                filename += '.csv'

            # 获取当前工作目录的绝对路径
            current_dir = os.getcwd()
            full_path = os.path.join(current_dir, filename)

            # 导出为CSV
            print(f"  正在保存文件到: {full_path}")
            df.to_csv(full_path, index=False, encoding='utf-8-sig')

            # 验证文件是否创建成功
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"   文件保存成功，大小: {file_size} 字节")
                return full_path
            else:
                print(f"   文件保存失败，文件不存在")
                return None

        except Exception as e:
            print(f"   导出交易记录失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def test_rolling_window_cointegration_trading(csv_file_path):
    """
    滚动窗口协整分析+交易测试（支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）

    Args:
        csv_file_path: CSV文件路径
    """
    print("=" * 80)
    print("滚动窗口协整分析+交易测试（支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")

    # 2. 选择价差类型
    print("\n2. 选择价差类型")
    diff_order = select_spread_type()

    # 3. 根据筛选结果输入需要回测的币对
    print("\n3. 输入已筛选的协整币对")
    selected_pairs = input_preselected_pairs(data, default_diff_order=diff_order)

    if not selected_pairs:
        print("未输入任何币对，无法进行交易")
        return

    # 4. 配置协整监控窗口（仅用于RLS动态校验）
    print("\n4. 配置协整监控窗口参数")
    cointegration_window_size = input_cointegration_window_size()

    # 5. 策略选择和回测循环
    print("\n5. 策略选择和回测循环")
    test_count = 0

    while True:
        test_count += 1
        print(f"\n{'=' * 80}")
        print(f"第 {test_count} 次测试")
        print(f"{'=' * 80}")

        # 选择Z-score计算策略
        z_score_strategy = select_z_score_strategy()
        if z_score_strategy is None:
            print("测试结束，退出程序")
            break

        # 7. 显示选择的币对详情
        print(f"\n第 {test_count} 次测试 - 选择的币对详情")
        for pair in selected_pairs:
            diff_order = pair.get('diff_order', 0)
            diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
            print(f"\n币对: {pair['pair_name']}")
            print(f"  价差类型: {diff_type}")
            print(f"  对冲比率: {pair['hedge_ratio']:.6f} (基于整个数据集计算)")
            if pair.get('spread_adf'):
                print(f"  价差ADF P值: {pair['spread_adf']['p_value']:.6f}")
            if 'cointegration_ratio' in pair:
                print(f"  协整比例: {pair['cointegration_ratio'] * 100:.1f}%")

        # 8. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()

        # 10. 配置RLS和协整检验参数
        print(f"\n第 {test_count} 次测试 - 配置RLS和协整检验参数")
        use_rls_input = input("是否使用RLS动态对冲比率? (y/n, 默认y): ").strip().lower()
        use_rls = use_rls_input != 'n'

        # 根据数据周期计算协整检验间隔
        # 假设数据是等间隔的，计算平均间隔
        if len(data) > 0:
            first_symbol = list(data.keys())[0]
            if len(data[first_symbol]) > 1:
                time_diffs = []
                timestamps = sorted(data[first_symbol].index)
                for i in range(1, min(100, len(timestamps))):  # 只检查前100个
                    diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60  # 转换为分钟
                    time_diffs.append(diff)
                if time_diffs:
                    avg_period_minutes = np.mean(time_diffs)
                    # 10天 = 10 * 24 * 60 = 14400分钟
                    cointegration_check_interval = int(14400 / avg_period_minutes)
                    print(f"  检测到数据周期: {avg_period_minutes:.1f} 分钟")
                    print(f"  协整检验间隔: {cointegration_check_interval} 个数据点 (约10天)")
                else:
                    cointegration_check_interval = 240  # 默认值（1h周期）
                    avg_period_minutes = 60
            else:
                cointegration_check_interval = 240
                avg_period_minutes = 60
        else:
            cointegration_check_interval = 240
            avg_period_minutes = 60

        # 11. 执行交易回测
        print(f"\n第 {test_count} 次测试 - 执行交易回测")
        print(f"使用的策略: {z_score_strategy.get_strategy_description()}")
        print(f"使用RLS: {'是' if use_rls else '否'}")
        print(f"协整检验窗口大小: {cointegration_window_size} 个数据点（用于RLS监控）")
        trading_strategy = AdvancedCointegrationTrading(
            lookback_period=trading_params['lookback_period'],
            z_threshold=trading_params['z_threshold'],
            z_exit_threshold=trading_params['z_exit_threshold'],
            take_profit_pct=trading_params['take_profit_pct'],
            stop_loss_pct=trading_params['stop_loss_pct'],
            max_holding_hours=trading_params['max_holding_hours'],
            position_ratio=trading_params['position_ratio'],
            use_rls=use_rls,
            cointegration_check_interval=cointegration_check_interval,
            data_period_minutes=avg_period_minutes,
            diff_order=diff_order,
            cointegration_window_size=cointegration_window_size,
            leverage=trading_params['leverage'],
            trading_fee_rate=trading_params.get('trading_fee_rate', 0.000275),
            z_score_strategy=z_score_strategy  # 使用策略对象
        )

        results = trading_strategy.backtest_cointegration_trading(
            data,
            selected_pairs,
            initial_capital=10000
        )

        # 9. 显示交易详情
        print(f"\n第 {test_count} 次测试 - 交易详情")
        if results['trades']:
            print(f"总交易次数: {len(results['trades'])}")

            # 按币对分组显示交易
            pair_trades = {}
            for trade in results['trades']:
                pair = trade['pair']
                if pair not in pair_trades:
                    pair_trades[pair] = []
                pair_trades[pair].append(trade)

            for pair, trades in pair_trades.items():
                print(f"\n{pair} 交易记录:")
                for trade in trades:
                    action = "开仓" if trade['action'] == 'OPEN' else "平仓"
                    pnl_info = f", 盈亏: {trade.get('pnl', 0):.2f}" if trade['action'] == 'CLOSE' else ""
                    print(
                        f"  {trade['timestamp']}: {action} {trade['symbol1']}={trade['symbol1_price']:.2f}, {trade['symbol2']}={trade['symbol2_price']:.2f}{pnl_info}")
        else:
            print("本次测试无交易记录")

        print(f"\n第 {test_count} 次测试完成")
        print("=" * 80)

        # 询问是否继续
        print("\n请选择下一步操作:")
        print("  1. 继续测试（重新选择策略）")
        print("  0. 退出程序")
        continue_choice = input("请选择 (0/1): ").strip()

        if continue_choice != '1':
            print("测试结束，退出程序")
            break

    print("\n" + "=" * 80)
    print("协整交易回测完成")
    print(f"总共进行了 {test_count} 次测试")
    print("=" * 80)


# ==================== 参数优化器代码 ====================

class ParameterOptimizer:
    """
    参数优化器类（支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）
    支持三种优化方法：
    1. 网格搜索（粗粒度 + 细粒度分层搜索）
    2. 贝叶斯优化
    3. 随机搜索
    包含过拟合检测机制
    """

    def __init__(self, data, selected_pairs, initial_capital=10000,
                 objective='sharpe_ratio', stability_test=True, z_score_strategy=None,
                 cointegration_window_size=500, diff_order=0):
        """
        初始化参数优化器

        Args:
            data: 价格数据字典
            selected_pairs: 选择的协整对列表
            initial_capital: 初始资金
            objective: 优化目标 ('sharpe_ratio', 'return', 'return_drawdown_ratio')
            stability_test: 是否进行稳定性测试（过拟合检测）
            z_score_strategy: Z-score计算策略对象（BaseZScoreStrategy实例）
            cointegration_window_size: 协整检验窗口大小（与初始筛选的window_size一致，默认500）
            diff_order: 价差类型，0=原始价差，1=一阶差分价差
        """
        self.data = data
        self.selected_pairs = selected_pairs
        self.initial_capital = initial_capital
        self.objective = objective
        self.stability_test = stability_test
        self.cointegration_window_size = cointegration_window_size
        self.diff_order = diff_order

        # 定义参数搜索空间（注意：z_score_strategy不在优化范围内，需要单独设置）
        self.param_space = {
            'lookback_period': {'type': 'int', 'coarse': [30, 60, 90, 120], 'fine_step': 10},
            'z_threshold': {'type': 'float', 'coarse': [1.0, 1.5, 2.0, 2.5, 3.0], 'fine_step': 0.1},
            'z_exit_threshold': {'type': 'float', 'coarse': [0.3, 0.5, 0.7, 0.9], 'fine_step': 0.1},
            'take_profit_pct': {'type': 'float', 'coarse': [0.05, 0.10, 0.15, 0.20, 0.25], 'fine_step': 0.02},
            'stop_loss_pct': {'type': 'float', 'coarse': [0.05, 0.08, 0.10, 0.12, 0.15], 'fine_step': 0.01},
            # 'max_holding_hours': {'type': 'int', 'coarse': [72, 120, 168, 240], 'fine_step': 24},
            # 'position_ratio': {'type': 'float', 'coarse': [0.3, 0.5, 0.7, 0.9], 'fine_step': 0.1},
            # 'leverage': {'type': 'int', 'coarse': [1, 3, 5, 10], 'fine_step': 1}
        }

        # 存储所有评估结果
        self.evaluation_history = []
        self.best_params = None
        self.best_score = float('-inf')

        # 设置Z-score策略
        if z_score_strategy is not None:
            self.z_score_strategy = z_score_strategy
        else:
            # 默认使用传统策略
            self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None

    def set_z_score_strategy(self, z_score_strategy):
        """设置Z-score计算策略"""
        self.z_score_strategy = z_score_strategy

    def set_use_arima_garch(self, use_arima_garch):
        """设置是否使用ARIMA-GARCH（向后兼容方法）"""
        if use_arima_garch and STRATEGIES_AVAILABLE and ARIMA_AVAILABLE and GARCH_AVAILABLE:
            try:
                self.z_score_strategy = ArimaGarchZScoreStrategy()
            except Exception as e:
                print(f"警告: ARIMA-GARCH策略初始化失败: {str(e)}，使用传统策略")
                self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None
        else:
            self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None

    def evaluate_params(self, params: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        评估参数组合

        Args:
            params: 参数字典
            verbose: 是否打印详细信息

        Returns:
            评估结果字典
        """
        try:
            # 创建策略实例（使用策略对象）
            strategy = AdvancedCointegrationTrading(
                lookback_period=params['lookback_period'],
                z_threshold=params['z_threshold'],
                z_exit_threshold=params['z_exit_threshold'],
                take_profit_pct=params['take_profit_pct'],
                stop_loss_pct=params['stop_loss_pct'],
                max_holding_hours=params.get('max_holding_hours', 168),
                position_ratio=params.get('position_ratio', 0.5),
                leverage=params.get('leverage', 5),
                trading_fee_rate=params.get('trading_fee_rate', 0.000275),
                z_score_strategy=self.z_score_strategy,  # 使用策略对象
                cointegration_window_size=self.cointegration_window_size,  # 传递窗口大小
                diff_order=self.diff_order  # 传递价差类型
            )

            # 执行回测（不显示图表和详细输出）
            strategy._optimization_mode = True  # 标记为优化模式
            results = strategy.backtest_cointegration_trading(
                self.data,
                self.selected_pairs,
                initial_capital=self.initial_capital
            )

            # 计算风险指标
            risk_metrics = strategy.calculate_risk_metrics(results['capital_curve'])

            # 计算最终收益（使用投入资金作为基准，与回测结果保持一致）
            final_capital = results['capital_curve'][-1]['capital'] if results[
                'capital_curve'] else self.initial_capital
            position_ratio = params.get('position_ratio', 0.5)
            initial_invested_capital = self.initial_capital * position_ratio
            total_return = (final_capital - initial_invested_capital) / initial_invested_capital

            # 根据目标函数计算得分
            if self.objective == 'sharpe_ratio':
                score = risk_metrics.get('sharpe_ratio', 0)
            elif self.objective == 'return':
                score = total_return * 100
            elif self.objective == 'return_drawdown_ratio':
                max_dd = risk_metrics.get('max_drawdown_pct', 1)
                score = (total_return * 100) / max_dd if max_dd > 0 else 0
            else:
                score = risk_metrics.get('sharpe_ratio', 0)

            evaluation_result = {
                'params': params.copy(),
                'score': score,
                'total_return': total_return * 100,
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': risk_metrics.get('max_drawdown_pct', 0),
                'profit_loss_ratio': risk_metrics.get('profit_loss_ratio', 0),
                'total_trades': risk_metrics.get('total_trades', 0),
                'profitable_trades': risk_metrics.get('profitable_trades', 0),
                'final_capital': final_capital
            }

            # 记录评估历史
            self.evaluation_history.append(evaluation_result)

            if verbose:
                print(f"参数评估: 得分={score:.4f}, 收益率={total_return * 100:.2f}%, "
                      f"夏普={risk_metrics.get('sharpe_ratio', 0):.4f}")

            return evaluation_result

        except Exception as e:
            if verbose:
                print(f"参数评估失败: {str(e)}")
            return {
                'params': params.copy(),
                'score': float('-inf'),
                'error': str(e)
            }

    def test_parameter_stability(self, params: Dict[str, Any],
                                 perturbation_ratio: float = 0.1,
                                 num_tests: int = 5) -> Dict[str, Any]:
        """
        测试参数稳定性（过拟合检测）
        在参数附近生成扰动，检查结果是否稳定

        Args:
            params: 待测试的参数
            perturbation_ratio: 扰动比例（相对于参数范围）
            num_tests: 测试次数

        Returns:
            稳定性测试结果
        """
        base_result = self.evaluate_params(params, verbose=False)
        base_score = base_result['score']

        perturbed_scores = []
        perturbed_results = []

        for _ in range(num_tests):
            # 生成扰动参数
            perturbed_params = params.copy()
            for param_name, param_value in params.items():
                if param_name in self.param_space:
                    space_def = self.param_space[param_name]
                    if space_def['type'] == 'int':
                        # 整数参数：扰动步长
                        step = max(1, int(space_def['fine_step'] * perturbation_ratio))
                        perturbed_params[param_name] = param_value + random.randint(-step, step)
                        # 确保在合理范围内
                        if param_name == 'lookback_period':
                            perturbed_params[param_name] = max(10, min(200, perturbed_params[param_name]))
                        elif param_name == 'max_holding_hours':
                            perturbed_params[param_name] = max(24, min(720, perturbed_params[param_name]))
                        elif param_name == 'leverage':
                            perturbed_params[param_name] = max(1, min(20, perturbed_params[param_name]))
                    else:
                        # 浮点参数：按比例扰动
                        step = space_def['fine_step'] * perturbation_ratio
                        perturbed_params[param_name] = param_value + random.uniform(-step, step)
                        # 确保在合理范围内
                        if param_name in ['z_threshold', 'z_exit_threshold']:
                            perturbed_params[param_name] = max(0.1, min(5.0, perturbed_params[param_name]))
                        elif param_name in ['take_profit_pct', 'stop_loss_pct', 'position_ratio']:
                            perturbed_params[param_name] = max(0.01, min(1.0, perturbed_params[param_name]))

            # 评估扰动参数
            perturbed_result = self.evaluate_params(perturbed_params, verbose=False)
            perturbed_scores.append(perturbed_result['score'])
            perturbed_results.append(perturbed_result)

        # 计算稳定性指标
        if len(perturbed_scores) > 0:
            score_std = np.std(perturbed_scores)
            score_mean = np.mean(perturbed_scores)
            score_cv = abs(score_std / score_mean) if score_mean != 0 else float('inf')

            # 计算得分下降比例
            score_drop = (base_score - score_mean) / abs(base_score) if base_score != 0 else 0

            stability_result = {
                'base_score': base_score,
                'perturbed_mean_score': score_mean,
                'perturbed_std_score': score_std,
                'score_coefficient_of_variation': score_cv,
                'score_drop_ratio': score_drop,
                'is_stable': score_cv < 0.3 and score_drop < 0.2,  # 稳定性阈值
                'perturbed_results': perturbed_results
            }
        else:
            stability_result = {
                'base_score': base_score,
                'is_stable': False,
                'error': '无法生成扰动测试'
            }

        return stability_result

    def grid_search(self, coarse_first: bool = True,
                    fine_search_around_best: bool = True,
                    max_coarse_combinations: int = 100) -> Dict[str, Any]:
        """
        网格搜索优化（分层：粗粒度 + 细粒度）

        Args:
            coarse_first: 是否先进行粗粒度搜索
            fine_search_around_best: 是否在最佳参数附近进行细粒度搜索
            max_coarse_combinations: 粗粒度搜索最大组合数

        Returns:
            优化结果
        """
        print("\n" + "=" * 80)
        print("开始网格搜索优化")
        if self.z_score_strategy:
            print(f"（使用策略: {self.z_score_strategy.get_strategy_description()}）")
        print("=" * 80)

        best_result = None
        best_score = float('-inf')

        # 第一阶段：粗粒度搜索
        if coarse_first:
            print("\n第一阶段：粗粒度网格搜索")
            print("-" * 80)

            # 生成所有粗粒度参数组合
            param_names = list(self.param_space.keys())
            coarse_values = [self.param_space[name]['coarse'] for name in param_names]
            coarse_combinations = list(itertools.product(*coarse_values))

            # 限制组合数量
            if len(coarse_combinations) > max_coarse_combinations:
                print(f"粗粒度组合数过多({len(coarse_combinations)})，随机采样{max_coarse_combinations}个")
                coarse_combinations = random.sample(coarse_combinations, max_coarse_combinations)

            print(f"测试 {len(coarse_combinations)} 个粗粒度参数组合...")

            for i, combination in enumerate(coarse_combinations):
                params = dict(zip(param_names, combination))
                result = self.evaluate_params(params, verbose=(i % 10 == 0))

                if (i + 1) % 10 == 0 or i == len(coarse_combinations) - 1:
                    print(
                        f"  进度: {i + 1}/{len(coarse_combinations)} ({100 * (i + 1) / len(coarse_combinations):.1f}%), "
                        f"当前最佳得分: {best_score:.4f}")

                if result['score'] > best_score:
                    best_score = result['score']
                    best_result = result
                    print(f"   找到更好的参数组合 #{i + 1}: 得分={best_score:.4f}")

            print(f"\n粗粒度搜索完成，最佳得分: {best_score:.4f}")

        # 第二阶段：细粒度搜索（在最佳参数附近）
        if fine_search_around_best and best_result:
            print("\n第二阶段：细粒度网格搜索（在最佳参数附近）")
            print("-" * 80)

            best_params = best_result['params']
            fine_combinations = []

            # 为每个参数生成细粒度搜索范围
            for param_name in param_names:
                space_def = self.param_space[param_name]
                base_value = best_params[param_name]
                fine_step = space_def['fine_step']

                if space_def['type'] == 'int':
                    # 整数：在基础值附近生成几个值
                    fine_values = [base_value + i * fine_step for i in range(-2, 3)]
                    fine_values = [v for v in fine_values if v > 0]
                else:
                    # 浮点：在基础值附近生成几个值
                    fine_values = [base_value + i * fine_step for i in range(-2, 3)]
                    fine_values = [max(0.01, v) for v in fine_values]

                fine_combinations.append(fine_values)

            # 生成细粒度组合（限制数量）
            fine_product = list(itertools.product(*fine_combinations))
            if len(fine_product) > 200:
                fine_product = random.sample(fine_product, 200)

            print(f"测试 {len(fine_product)} 个细粒度参数组合...")

            for i, combination in enumerate(fine_product):
                params = dict(zip(param_names, combination))
                result = self.evaluate_params(params, verbose=(i % 20 == 0))

                if (i + 1) % 20 == 0 or i == len(fine_product) - 1:
                    print(f"  进度: {i + 1}/{len(fine_product)} ({100 * (i + 1) / len(fine_product):.1f}%), "
                          f"当前最佳得分: {best_score:.4f}")

                if result['score'] > best_score:
                    best_score = result['score']
                    best_result = result
                    print(f"   找到更好的参数组合 #{i + 1}: 得分={best_score:.4f}")

            print(f"\n细粒度搜索完成，最终最佳得分: {best_score:.4f}")

        # 稳定性测试
        if self.stability_test and best_result:
            print("\n进行参数稳定性测试（过拟合检测）...")
            stability = self.test_parameter_stability(best_result['params'])

            if stability['is_stable']:
                print(f" 参数稳定性良好 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")
            else:
                print(f" 参数可能不稳定 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")

            best_result['stability'] = stability

        self.best_params = best_result['params'] if best_result else None
        self.best_score = best_score

        return {
            'method': 'grid_search',
            'best_params': best_result['params'] if best_result else None,
            'best_score': best_score,
            'best_result': best_result,
            'total_evaluations': len(self.evaluation_history)
        }

    def random_search(self, n_iter: int = 100) -> Dict[str, Any]:
        """
        随机搜索优化

        Args:
            n_iter: 迭代次数

        Returns:
            优化结果
        """
        print("\n" + "=" * 80)
        print("开始随机搜索优化")
        if self.z_score_strategy:
            print(f"（使用策略: {self.z_score_strategy.get_strategy_description()}）")
        print("=" * 80)
        print(f"迭代次数: {n_iter}")

        best_result = None
        best_score = float('-inf')

        for i in range(n_iter):
            # 随机生成参数
            params = {}
            for param_name, space_def in self.param_space.items():
                if space_def['type'] == 'int':
                    # 整数：从粗粒度范围中随机选择，然后添加随机扰动
                    base = random.choice(space_def['coarse'])
                    step = space_def['fine_step']
                    params[param_name] = base + random.randint(-int(step), int(step))
                    # 确保在合理范围
                    if param_name == 'lookback_period':
                        params[param_name] = max(10, min(200, params[param_name]))
                    elif param_name == 'max_holding_hours':
                        params[param_name] = max(24, min(720, params[param_name]))
                    elif param_name == 'leverage':
                        params[param_name] = max(1, min(20, params[param_name]))
                else:
                    # 浮点：从粗粒度范围中随机选择，然后添加随机扰动
                    base = random.choice(space_def['coarse'])
                    step = space_def['fine_step']
                    params[param_name] = base + random.uniform(-step, step)
                    # 确保在合理范围
                    if param_name in ['z_threshold', 'z_exit_threshold']:
                        params[param_name] = max(0.1, min(5.0, params[param_name]))
                    elif param_name in ['take_profit_pct', 'stop_loss_pct', 'position_ratio']:
                        params[param_name] = max(0.01, min(1.0, params[param_name]))

            result = self.evaluate_params(params, verbose=(i % 10 == 0))

            if (i + 1) % 10 == 0 or i == n_iter - 1:
                print(f"  进度: {i + 1}/{n_iter} ({100 * (i + 1) / n_iter:.1f}%), "
                      f"当前最佳得分: {best_score:.4f}")

            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                print(f"   迭代 {i + 1}/{n_iter}: 找到更好的参数，得分={best_score:.4f}")

        # 稳定性测试
        if self.stability_test and best_result:
            print("\n进行参数稳定性测试（过拟合检测）...")
            stability = self.test_parameter_stability(best_result['params'])

            if stability['is_stable']:
                print(f" 参数稳定性良好 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")
            else:
                print(f" 参数可能不稳定 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")

            best_result['stability'] = stability

        self.best_params = best_result['params'] if best_result else None
        self.best_score = best_score

        return {
            'method': 'random_search',
            'best_params': best_result['params'] if best_result else None,
            'best_score': best_score,
            'best_result': best_result,
            'total_evaluations': len(self.evaluation_history)
        }

    def bayesian_optimization(self, n_calls: int = 50) -> Dict[str, Any]:
        """
        贝叶斯优化

        Args:
            n_calls: 评估次数

        Returns:
            优化结果
        """
        if not BAYESIAN_OPT_AVAILABLE:
            print("错误: scikit-optimize未安装，无法使用贝叶斯优化")
            return {'error': 'Bayesian optimization not available'}

        print("\n" + "=" * 80)
        print("开始贝叶斯优化")
        if self.z_score_strategy:
            print(f"（使用策略: {self.z_score_strategy.get_strategy_description()}）")
        print("=" * 80)
        print(f"评估次数: {n_calls}")

        # 定义搜索空间
        dimensions = []
        param_names = list(self.param_space.keys())

        for param_name in param_names:
            space_def = self.param_space[param_name]
            if space_def['type'] == 'int':
                min_val = min(space_def['coarse'])
                max_val = max(space_def['coarse'])
                dimensions.append(Integer(min_val, max_val, name=param_name))
            else:
                min_val = min(space_def['coarse'])
                max_val = max(space_def['coarse'])
                dimensions.append(Real(min_val, max_val, name=param_name))

        # 定义目标函数
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            # 确保整数参数为整数
            for param_name, param_value in params.items():
                if param_name in self.param_space:
                    space_def = self.param_space[param_name]
                    if space_def['type'] == 'int':
                        params[param_name] = int(round(param_value))

            result = self.evaluate_params(params, verbose=False)
            return -result['score']  # 最小化负得分（即最大化得分）

        # 执行贝叶斯优化
        result_bo = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )

        # 提取最佳参数并确保整数参数为整数
        best_params = {}
        for i, param_name in enumerate(param_names):
            param_value = result_bo.x[i]
            space_def = self.param_space[param_name]
            if space_def['type'] == 'int':
                # 确保整数参数为整数
                best_params[param_name] = int(round(param_value))
                # 确保在合理范围内
                if param_name == 'lookback_period':
                    best_params[param_name] = max(10, min(200, best_params[param_name]))
                elif param_name == 'max_holding_hours':
                    best_params[param_name] = max(24, min(720, best_params[param_name]))
                elif param_name == 'leverage':
                    best_params[param_name] = max(1, min(20, best_params[param_name]))
            else:
                best_params[param_name] = float(param_value)
                # 确保在合理范围内
                if param_name in ['z_threshold', 'z_exit_threshold']:
                    best_params[param_name] = max(0.1, min(5.0, best_params[param_name]))
                elif param_name in ['take_profit_pct', 'stop_loss_pct', 'position_ratio']:
                    best_params[param_name] = max(0.01, min(1.0, best_params[param_name]))

        best_score = -result_bo.fun

        # 重新评估最佳参数以获取完整结果
        best_result = self.evaluate_params(best_params, verbose=True)

        # 稳定性测试
        if self.stability_test:
            print("\n进行参数稳定性测试（过拟合检测）...")
            stability = self.test_parameter_stability(best_params)

            if stability['is_stable']:
                print(f" 参数稳定性良好 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")
            else:
                print(f" 参数可能不稳定 (CV={stability['score_coefficient_of_variation']:.3f}, "
                      f"下降比例={stability['score_drop_ratio']:.3f})")

            best_result['stability'] = stability

        self.best_params = best_params
        self.best_score = best_score

        return {
            'method': 'bayesian_optimization',
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'total_evaluations': len(self.evaluation_history)
        }

    def optimize(self, method: str = 'grid_search', **kwargs) -> Dict[str, Any]:
        """
        执行优化

        Args:
            method: 优化方法 ('grid_search', 'random_search', 'bayesian_optimization')
            **kwargs: 方法特定参数

        Returns:
            优化结果
        """
        if method == 'grid_search':
            return self.grid_search(**kwargs)
        elif method == 'random_search':
            return self.random_search(**kwargs)
        elif method == 'bayesian_optimization':
            return self.bayesian_optimization(**kwargs)
        else:
            raise ValueError(f"未知的优化方法: {method}")

    def get_top_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取前N个最佳结果

        Args:
            n: 返回结果数量

        Returns:
            排序后的结果列表
        """
        sorted_results = sorted(self.evaluation_history,
                                key=lambda x: x['score'],
                                reverse=True)
        return sorted_results[:n]

    def export_results(self, filename: Optional[str] = None) -> str:
        """
        导出优化结果到CSV

        Args:
            filename: 输出文件名

        Returns:
            文件路径
        """
        if filename is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'optimization_results_{timestamp_str}.csv'

        if not filename.endswith('.csv'):
            filename += '.csv'

        # 准备数据
        data = []
        for result in self.evaluation_history:
            row = result['params'].copy()
            row['score'] = result['score']
            row['total_return'] = result['total_return']
            row['sharpe_ratio'] = result['sharpe_ratio']
            row['max_drawdown_pct'] = result['max_drawdown_pct']
            row['profit_loss_ratio'] = result['profit_loss_ratio']
            row['total_trades'] = result['total_trades']
            row['profitable_trades'] = result['profitable_trades']
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('score', ascending=False)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"优化结果已导出到: {filename}")
        return filename


def test_parameter_optimization(csv_file_path):
    """
    参数优化测试函数（支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）
    """
    print("=" * 80)
    print("参数优化测试（支持传统方法、ARIMA-GARCH、ECM、Kalman Filter和Copula + DCC-GARCH策略）")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    # 2. 选择价差类型
    print("\n2. 选择价差类型")
    diff_order = select_spread_type()

    # 3. 配置协整监控窗口参数（供RLS使用）
    print("\n3. 配置协整监控窗口参数（用于RLS监控与周期性复检）")
    window_params = configure_rolling_window_parameters()

    # 4. 输入已筛选好的协整币对
    print("\n4. 输入已筛选好的协整币对")
    selected_pairs = input_preselected_pairs(data, default_diff_order=diff_order)

    if not selected_pairs:
        print("未输入任何币对，无法进行优化")
        return

    # 5. 选择Z-score计算策略
    print("\n5. 选择Z-score计算策略")
    z_score_strategy = select_z_score_strategy()
    if z_score_strategy is None:
        print("未选择策略，退出优化")
        return

    # 6. 选择优化方法
    print("\n6. 选择优化方法")
    print("可选方法:")
    print("  1. 网格搜索（粗粒度+细粒度）")
    print("  2. 随机搜索")
    print("  3. 贝叶斯优化")

    method_choice = input("请选择优化方法 (1/2/3): ").strip()

    method_map = {
        '1': 'grid_search',
        '2': 'random_search',
        '3': 'bayesian_optimization'
    }

    method = method_map.get(method_choice, 'grid_search')

    # 7. 选择优化目标
    print("\n7. 选择优化目标")
    print("可选目标:")
    print("  1. 夏普比率 (sharpe_ratio)")
    print("  2. 总收益率 (return)")
    print("  3. 收益率/回撤比 (return_drawdown_ratio)")

    objective_choice = input("请选择优化目标 (1/2/3): ").strip()

    objective_map = {
        '1': 'sharpe_ratio',
        '2': 'return',
        '3': 'return_drawdown_ratio'
    }

    objective = objective_map.get(objective_choice, 'sharpe_ratio')

    # 8. 显示选择的币对详情
    print(f"\n8. 选择的币对详情")
    for pair in selected_pairs:
        pair_diff_order = pair.get('diff_order', diff_order)
        diff_type = '原始价差' if pair_diff_order == 0 else '一阶差分价差'
        print(f"\n币对: {pair['pair_name']}")
        print(f"  价差类型: {diff_type}")
        print(f"  对冲比率: {pair['hedge_ratio']:.6f} (基于整个数据集计算)")
        if pair.get('spread_adf'):
            print(f"  价差ADF P值: {pair['spread_adf']['p_value']:.6f}")
        if 'cointegration_ratio' in pair:
            print(f"  协整比例: {pair['cointegration_ratio'] * 100:.1f}%")

    # 9. 创建优化器
    strategy_name = z_score_strategy.get_strategy_description() if z_score_strategy else "未知"
    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"\n10. 创建优化器 (方法={method}, 目标={objective}, 策略={strategy_name}, 价差类型={diff_type})")
    print(f"  协整检验窗口大小: {window_params['window_size']} 个数据点（与初始筛选一致）")
    optimizer = ParameterOptimizer(
        data=data,
        selected_pairs=selected_pairs,
        initial_capital=10000,
        objective=objective,
        stability_test=True,
        z_score_strategy=z_score_strategy,  # 传入策略对象
        cointegration_window_size=window_params['window_size'],  # 传递用户选择的窗口大小
        diff_order=diff_order  # 传递价差类型
    )

    # 10. 执行优化
    print(f"\n10. 执行优化...")
    if method == 'grid_search':
        result = optimizer.optimize(method='grid_search',
                                    coarse_first=True,
                                    fine_search_around_best=True)
    elif method == 'random_search':
        n_iter = input("请输入随机搜索迭代次数 (默认100): ").strip()
        n_iter = int(n_iter) if n_iter else 100
        result = optimizer.optimize(method='random_search', n_iter=n_iter)
    else:  # bayesian_optimization
        n_calls = input("请输入贝叶斯优化评估次数 (默认50): ").strip()
        n_calls = int(n_calls) if n_calls else 50
        result = optimizer.optimize(method='bayesian_optimization', n_calls=n_calls)

    # 11. 显示结果
    print("\n" + "=" * 80)
    print("优化结果")
    print("=" * 80)

    if result.get('error'):
        print(f"优化失败: {result['error']}")
        return

    print(f"\n最佳参数:")
    for param_name, param_value in result['best_params'].items():
        print(f"  {param_name}: {param_value}")
    strategy_desc = z_score_strategy.get_strategy_description() if z_score_strategy else "未知"
    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"  使用的策略: {strategy_desc}")
    print(f"  价差类型: {diff_type}")

    print(f"\n最佳得分: {result['best_score']:.4f}")
    if result['best_result']:
        print(f"  总收益率: {result['best_result']['total_return']:.2f}%")
        print(f"  夏普比率: {result['best_result']['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {result['best_result']['max_drawdown_pct']:.2f}%")
        print(f"  盈亏比: {result['best_result']['profit_loss_ratio']:.2f}")
        print(f"  总交易次数: {result['best_result']['total_trades']}")

        if 'stability' in result['best_result']:
            stability = result['best_result']['stability']
            print(f"\n稳定性测试:")
            print(f"  稳定性: {'良好' if stability['is_stable'] else '可能不稳定'}")
            print(f"  变异系数: {stability.get('score_coefficient_of_variation', 0):.3f}")
            print(f"  得分下降比例: {stability.get('score_drop_ratio', 0):.3f}")

    print(f"\n总评估次数: {result['total_evaluations']}")

    # 12. 显示前10个最佳结果
    print("\n前10个最佳参数组合:")
    top_results = optimizer.get_top_results(10)
    for i, res in enumerate(top_results, 1):
        print(f"\n{i}. 得分={res['score']:.4f}, 收益率={res['total_return']:.2f}%, "
              f"夏普={res['sharpe_ratio']:.4f}")
        print(f"   参数: {res['params']}")

    # 13. 导出结果
    print("\n13. 导出优化结果...")
    optimizer.export_results()

    return result


def main():
    """
    主函数
    """
    print("协整交易回测与参数优化（已筛选币对版，含Regime-Switching模型）")
    print("核心流程：")
    print("  1. 用户输入已筛选的协整币对及价差类型")
    print("  2. RLS递归最小二乘 + 指定窗口/步长进行动态对冲比率更新")
    print("  3. 回测期间按窗口周期复检协整状态并调整仓位")
    print("  4. 可选参数优化（网格、随机、贝叶斯）与稳定性测试")
    print("  5. 支持多种Z-score计算策略：")
    print("     - 传统方法（均值和标准差）")
    print("     - ARIMA-GARCH模型")
    print("     - ECM误差修正模型（推荐用于协整交易）")
    print("     - Kalman Filter动态价差模型（推荐用于动态市场）")
    print("     - Copula + DCC-GARCH相关性/波动率模型（推荐用于相关性建模）")
    print("     - Regime-Switching市场状态模型（推荐用于状态转换市场）")
    print()

    # 选择模式
    print("请选择运行模式:")
    print("  1. 普通回测模式")
    print("  2. 参数优化模式")

    mode_choice = input("请选择 (1/2): ").strip()

    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()

    if not csv_file_path:
        csv_file_path = "segment_2_data_ccxt_20251113_103652.csv"
        print(f"使用默认路径: {csv_file_path}")

    if mode_choice == '2':
        # 参数优化模式
        test_parameter_optimization(csv_file_path)
    else:
        # 普通回测模式
        test_rolling_window_cointegration_trading(csv_file_path)


if __name__ == "__main__":
    main()