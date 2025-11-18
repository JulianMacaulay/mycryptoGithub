#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级版协整分析+交易流程完整测试文件（带手续费和动态仓位）
完全复制自 test_cointegration_trading_advanced.py，但使用 cointegration_test_windows_optimization_arima_garch.py 的回测逻辑
包含多阶差分检验、手工选择币对、止盈止损、风险指标、收益率曲线图等完整流程
主要差异：
1. 使用 position_ratio 和 leverage 进行资金管理
2. 使用动态仓位计算（基于可用资金）
3. 基于持仓价值变化计算盈亏（而非价差变化）
4. 包含手续费计算和扣除
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 原有协整分析代码（保持不变） ====================

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


def calculate_spread(price1, price2):
    """
    计算价差

    Args:
        price1: 第一个币种的价格序列
        price2: 第二个币种的价格序列

    Returns:
        pd.Series: 价差序列
    """
    print("计算价差...")

    # 确保两个序列长度一致
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]

    # 计算价差
    spread = price1_aligned - price2_aligned

    print(f"价差统计:")
    print(f"  均值: {spread.mean():.6f}")
    print(f"  标准差: {spread.std():.6f}")
    print(f"  最小值: {spread.min():.6f}")
    print(f"  最大值: {spread.max():.6f}")

    return spread


def calculate_hedge_ratio(price1, price2):
    """
    计算对冲比率

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


def advanced_adf_test(series, max_lags=None):
    """
    执行增强的ADF检验
    Args:
        series: 时间序列
        max_lags: 最大滞后阶数

    Returns:
        dict: ADF检验结果
    """
    print("执行ADF检验...")

    try:
        # 执行ADF检验
        adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')

        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        used_lag = adf_result[2]

        print(f"ADF检验结果:")
        print(f"  ADF统计量: {adf_statistic:.6f}")
        print(f"  P值: {p_value:.6f}")
        print(f"  使用的滞后阶数: {used_lag}")
        print(f"  临界值:")
        for level, value in critical_values.items():
            print(f"    {level}: {value:.6f}")

        # 判断是否平稳
        is_stationary = p_value < 0.05
        print(f"  是否平稳: {'是' if is_stationary else '否'}")

        return {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'used_lag': used_lag,
            'is_stationary': is_stationary
        }

    except Exception as e:
        print(f"ADF检验失败: {str(e)}")
        return None


def distribution_test(series):
    """
    执行概率分布检验

    Args:
        series: 时间序列

    Returns:
        dict: 分布检验结果
    """
    print("执行分布检验...")

    try:
        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(series)
        print(f"Shapiro-Wilk正态性检验:")
        print(f"  统计量: {shapiro_stat:.6f}")
        print(f"  P值: {shapiro_p:.6f}")
        print(f"  是否正态分布: {'是' if shapiro_p > 0.05 else '否'}")

        # 偏度和峰度
        skewness = stats.skew(series)
        kurtosis = stats.kurtosis(series)

        print(f"偏度: {skewness:.6f}")
        print(f"峰度: {kurtosis:.6f}")

        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    except Exception as e:
        print(f"分布检验失败: {str(e)}")
        return None


def calculate_z_score(series, window=20):
    """
    计算Z分数

    Args:
        series: 时间序列
        window: 滚动窗口大小

    Returns:
        pd.Series: Z分数序列
    """
    print(f"计算Z分数 (窗口大小: {window})...")

    # 计算滚动均值和标准差
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # 计算Z分数
    z_scores = (series - rolling_mean) / rolling_std

    print(f"Z分数统计:")
    print(f"  均值: {z_scores.mean():.6f}")
    print(f"  标准差: {z_scores.std():.6f}")
    print(f"  最小值: {z_scores.min():.6f}")
    print(f"  最大值: {z_scores.max():.6f}")

    return z_scores


# ==================== 增强版协整分析代码 ====================

def enhanced_cointegration_test(price1, price2, symbol1, symbol2, max_diff_order=2):
    """
    增强的协整检验，包括多阶差分

    Args:
        price1: 第一个价格序列
        price2: 第二个价格序列
        symbol1: 第一个币种名称
        symbol2: 第二个币种名称
        max_diff_order: 最大差分阶数

    Returns:
        dict: 检验结果
    """
    print(f"\n开始增强协整检验: {symbol1}/{symbol2}")

    results = {
        'pair_name': f"{symbol1}/{symbol2}",
        'symbol1': symbol1,
        'symbol2': symbol2,
        'original_spread': None,
        'spread_adf': None,
        'diff_tests': [],
        'cointegration_found': False,
        'final_spread': None,
        'diff_order': 0,
        'hedge_ratio': None,
        'best_test': None
    }

    # 1. 计算原始价差
    spread = calculate_spread(price1, price2)
    results['original_spread'] = spread

    # 2. 对原始价差进行ADF检验
    print("\n--- 检验原始价差 ---")
    spread_adf = advanced_adf_test(spread)
    results['spread_adf'] = spread_adf

    if spread_adf and spread_adf['is_stationary']:
        # 原始价差就是平稳的
        results['cointegration_found'] = True
        results['final_spread'] = spread
        results['diff_order'] = 0
        results['best_test'] = {
            'type': 'original',
            'adf_result': spread_adf,
            'spread': spread
        }
        print("原始价差协整检验通过")
        return results

    # 3. 如果原始价差不平稳，尝试差分
    current_spread = spread
    for diff_order in range(1, max_diff_order + 1):
        print(f"\n--- 检验{diff_order}阶差分 ---")

        # 计算差分
        diff_spread = current_spread.diff().dropna()

        if len(diff_spread) < 50:  # 数据点太少
            print(f" {diff_order}阶差分数据点不足({len(diff_spread)}个)，跳过")
            break

        # 对差分序列进行ADF检验
        diff_adf = advanced_adf_test(diff_spread)

        diff_result = {
            'diff_order': diff_order,
            'spread': diff_spread,
            'adf_result': diff_adf,
            'is_stationary': diff_adf and diff_adf['is_stationary']
        }
        results['diff_tests'].append(diff_result)

        if diff_adf and diff_adf['is_stationary']:
            # 找到了平稳的差分序列
            results['cointegration_found'] = True
            results['final_spread'] = diff_spread
            results['diff_order'] = diff_order

            # 使用差分数据计算对冲比率
            diff_price1 = price1.diff().dropna()
            diff_price2 = price2.diff().dropna()
            hedge_ratio = calculate_hedge_ratio(diff_price1, diff_price2)
            results['hedge_ratio'] = hedge_ratio

            results['best_test'] = {
                'type': f'diff_{diff_order}',
                'adf_result': diff_adf,
                'spread': diff_spread
            }
            print(f"{diff_order}阶差分协整检验通过")
            break
        else:
            print(f"{diff_order}阶差分协整检验未通过")

        current_spread = diff_spread

    return results


def enhanced_find_cointegrated_pairs(data, max_diff_order=2):
    """
    增强的协整对寻找，包括多阶差分检验

    Args:
        data: 包含各币对数据的字典
        max_diff_order: 最大差分阶数

    Returns:
        list: 协整对列表
    """
    print("=" * 80)
    print("开始增强协整对寻找（包含多阶差分检验）")
    print("=" * 80)

    symbols = list(data.keys())
    all_candidates = []

    print(f"分析 {len(symbols)} 个币对...")

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]
            print(f"\n{'=' * 60}")
            print(f"分析币对: {symbol1}/{symbol2}")
            print(f"{'=' * 60}")

            try:
                # 执行增强的协整检验
                coint_result = enhanced_cointegration_test(
                    data[symbol1],
                    data[symbol2],
                    symbol1,
                    symbol2,
                    max_diff_order
                )

                if coint_result['cointegration_found']:
                    # 如果还没有计算对冲比率（原始协整情况），则计算
                    if coint_result['hedge_ratio'] is None:
                        hedge_ratio = calculate_hedge_ratio(data[symbol1], data[symbol2])
                        coint_result['hedge_ratio'] = hedge_ratio

                    # 添加到候选列表
                    all_candidates.append(coint_result)
                    print(f"{symbol1}/{symbol2} 协整检验通过")
                else:
                    print(f"{symbol1}/{symbol2} 协整检验未通过")

            except Exception as e:
                print(f"分析 {symbol1}/{symbol2} 时出错: {str(e)}")

    return all_candidates


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
        'trading_fee_rate': 0.000275
    }

    print("当前默认参数:")
    print(f"  1. 回看期: {default_params['lookback_period']}")
    print(f"  2. Z-score开仓阈值: {default_params['z_threshold']}")
    print(f"  3. Z-score平仓阈值: {default_params['z_exit_threshold']}")
    print(f"  4. 止盈百分比: {default_params['take_profit_pct'] * 100:.1f}%")
    print(f"  5. 止损百分比: {default_params['stop_loss_pct'] * 100:.1f}%")
    print(f"  6. 最大持仓时间: {default_params['max_holding_hours']}小时")
    print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}%")
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
            except ValueError:
                print(f"输入无效，使用默认值: {default_params['position_ratio'] * 100:.1f}%")

        # 杠杆
        leverage_input = input(f"杠杆倍数 (默认: {default_params['leverage']}): ").strip()
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
        print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}%")
        print(f"  8. 杠杆: {default_params['leverage']}倍")
        print(f"  9. 交易手续费率: {default_params['trading_fee_rate'] * 100:.4f}%")

    return default_params


def display_candidates_for_selection(candidates):
    """
    显示候选币对供手工选择

    Args:
        candidates: 候选币对列表

    Returns:
        list: 用户选择的币对列表
    """
    if not candidates:
        print("没有找到任何协整对")
        return []

    print("\n" + "=" * 80)
    print("协整对候选列表")
    print("=" * 80)

    for i, candidate in enumerate(candidates, 1):
        pair_name = candidate['pair_name']
        diff_order = candidate['diff_order']
        hedge_ratio = candidate['hedge_ratio']
        adf_p = candidate['best_test']['adf_result']['p_value']

        print(f"\n{i}. 币对: {pair_name}")
        print(f"   差分阶数: {diff_order}")
        print(f"   对冲比率: {hedge_ratio:.6f}")
        print(f"   ADF P值: {adf_p:.6f}")
        print(f"   协整类型: {'原始价差' if diff_order == 0 else f'{diff_order}阶差分'}")

        # 显示统计信息
        spread = candidate['final_spread']
        print(f"   价差统计:")
        print(f"     均值: {spread.mean():.6f}")
        print(f"     标准差: {spread.std():.6f}")
        print(f"     数据点: {len(spread)}")

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
                if 0 <= idx < len(candidates):
                    valid_selection.append(candidates[idx])
                else:
                    print(f"序号 {idx + 1} 超出范围")

            if valid_selection:
                print(f"\n已选择 {len(valid_selection)} 个币对:")
                for selected in valid_selection:
                    diff_type = '原始价差' if selected['diff_order'] == 0 else f"{selected['diff_order']}阶差分"
                    print(f"   - {selected['pair_name']} ({diff_type})")
                return valid_selection
            else:
                print("没有有效的选择，请重新输入")

        except ValueError:
            print("输入格式错误，请输入数字序号，用逗号分隔")
        except KeyboardInterrupt:
            print("\n用户取消选择")
            return []


# ==================== 高级交易流程代码 ====================

class AdvancedCointegrationTrading:
    """高级协整交易策略类（带手续费和动态仓位）"""

    def __init__(self, lookback_period=60, z_threshold=2.0, z_exit_threshold=0.5,
                 take_profit_pct=0.15, stop_loss_pct=0.08, max_holding_hours=168,
                 position_ratio=0.5, leverage=5, trading_fee_rate=0.000275):
        """
        初始化高级协整交易策略

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
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.z_exit_threshold = z_exit_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_hours = max_holding_hours
        self.position_ratio = position_ratio
        self.leverage = leverage
        self.trading_fee_rate = trading_fee_rate
        self.positions = {}  # 当前持仓
        self.trades = []  # 交易记录

    def calculate_current_spread(self, price1, price2, hedge_ratio):
        """计算当前价差"""
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

    def calculate_z_score(self, current_spread, historical_spreads):
        """计算当前Z-score"""
        if len(historical_spreads) < 2:
            return 0

        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)

        if spread_std == 0:
            return 0

        return (current_spread - spread_mean) / spread_std

    def generate_trading_signal(self, z_score, diff_order=0):
        """生成交易信号"""
        if z_score > self.z_threshold:
            return {
                'action': 'SHORT_LONG',
                'description': f'Z-score过高({z_score:.3f})，做空价差',
                'confidence': min(abs(z_score) / 3.0, 1.0),
                'diff_order': diff_order
            }
        elif z_score < -self.z_threshold:
            return {
                'action': 'LONG_SHORT',
                'description': f'Z-score过低({z_score:.3f})，做多价差',
                'confidence': min(abs(z_score) / 3.0, 1.0),
                'diff_order': diff_order
            }
        else:
            return {
                'action': 'HOLD',
                'description': f'Z-score正常({z_score:.3f})，观望',
                'confidence': 0.0,
                'diff_order': diff_order
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
        open_fee = total_capital_used * self.trading_fee_rate

        # 创建持仓记录
        position = {
            'pair': f"{symbol1}_{symbol2}",
            'symbol1': symbol1,
            'symbol2': symbol2,
            'symbol1_size': symbol1_size,
            'symbol2_size': symbol2_size,
            'entry_prices': {symbol1: price1, symbol2: price2},
            'entry_spread': current_spread,
            'hedge_ratio': hedge_ratio,
            'entry_time': timestamp,
            'signal': signal,
            'capital_used': total_capital_used,
            'open_fee': open_fee
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
            'fee': open_fee
        }
        self.trades.append(trade)

        diff_info = f"({signal.get('diff_order', 0)}阶差分)" if signal.get('diff_order', 0) > 0 else "(原始价差)"
        print(f"开仓: {pair_info['pair_name']} {diff_info}")
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

        # 计算实际盈亏（基于价差变化，与固定仓位模式等价）
        # 原版使用：total_pnl = -spread_change 或 total_pnl = spread_change（绝对金额）
        # 新版需要：total_pnl = position_size_multiplier * spread_change（保持等价）
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread
        
        # 计算仓位倍数（相对于固定仓位±1.0的倍数）
        # 固定仓位模式下：symbol1_size = ±1.0
        # 动态仓位模式下：symbol1_size = ±symbol1_size_abs
        # 仓位倍数 = abs(symbol1_size) / 1.0 = abs(symbol1_size)
        position_size_multiplier = abs(position['symbol1_size'])
        
        # 根据交易方向计算盈亏（与固定仓位模式等价）
        if position['signal']['action'] == 'SHORT_LONG':
            # 做空价差，价差减少时盈利
            total_pnl = -spread_change * position_size_multiplier
        else:  # LONG_SHORT
            # 做多价差，价差增加时盈利
            total_pnl = spread_change * position_size_multiplier

        # 计算投入资金（基于原始价格）
        entry_value = abs(position['symbol1_size'] * entry_price1) + \
                      abs(position['symbol2_size'] * entry_price2)

        # 计算手续费（用于止盈止损判断）
        close_fee = (abs(position['symbol1_size']) * price1 + abs(
            position['symbol2_size']) * price2) * self.trading_fee_rate
        open_fee = position.get('open_fee', 0)

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

        # 计算最终盈亏（基于价差变化，与固定仓位模式等价）
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread
        
        # 计算仓位倍数（相对于固定仓位±1.0的倍数）
        position_size_multiplier = abs(position['symbol1_size'])
        
        # 根据交易方向计算盈亏（与固定仓位模式等价）
        if position['signal']['action'] == 'SHORT_LONG':
            # 做空价差，价差减少时盈利
            total_pnl = -spread_change * position_size_multiplier
        else:  # LONG_SHORT
            # 做多价差，价差增加时盈利
            total_pnl = spread_change * position_size_multiplier

        # 计算平仓手续费
        close_fee = (abs(position['symbol1_size']) * price1 + abs(
            position['symbol2_size']) * price2) * self.trading_fee_rate

        # 计算总手续费
        open_fee = position.get('open_fee', 0)
        total_fee = open_fee + close_fee

        # 扣除手续费后的净盈亏
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
            'pnl': net_pnl,
            'gross_pnl': total_pnl,
            'open_fee': open_fee,
            'close_fee': close_fee,
            'total_fee': total_fee,
            'holding_hours': (timestamp - position['entry_time']).total_seconds() / 3600
        }
        self.trades.append(trade)

        diff_info = f"({position.get('signal', {}).get('diff_order', 0)}阶差分)" if position.get('signal', {}).get(
            'diff_order', 0) > 0 else "(原始价差)"
        print(f"平仓: {pair_name} {diff_info}")
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

        # 计算夏普比率（自动检测数据频率）
        if len(returns) > 0 and np.std(returns) > 0:
            # 自动检测数据频率
            time_diff = (capital_curve[-1]['timestamp'] - capital_curve[0]['timestamp']).total_seconds()
            periods_per_year = len(returns) * (365.25 * 24 * 3600) / time_diff if time_diff > 0 else 252
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
        """回测协整交易策略（带手续费和动态仓位）"""
        print("\n" + "=" * 60)
        print("开始高级协整交易回测（带手续费和动态仓位）")
        print("=" * 60)

        # 初始化
        # 确定投入资金：capital = initial_capital * position_ratio
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

        # 回测循环
        for i, timestamp in enumerate(all_timestamps):
            # 获取当前价格
            current_prices = {}
            for symbol in data.keys():
                if timestamp in data[symbol].index:
                    current_prices[symbol] = data[symbol].loc[timestamp]

            # 检查每个选择的币对
            for pair_info in selected_pairs:
                symbol1, symbol2 = pair_info['pair_name'].split('/')

                if symbol1 not in current_prices or symbol2 not in current_prices:
                    continue

                # 根据差分阶数计算价差
                if pair_info['diff_order'] == 0:
                    # 原始协整：使用原始价格
                    current_spread = self.calculate_current_spread(
                        current_prices[symbol1],
                        current_prices[symbol2],
                        pair_info['hedge_ratio']
                    )

                    # 获取历史价差数据
                    historical_spreads = []
                    for j in range(max(0, i - self.lookback_period), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index):
                                hist_spread = self.calculate_current_spread(
                                    data[symbol1].loc[hist_timestamp],
                                    data[symbol2].loc[hist_timestamp],
                                    pair_info['hedge_ratio']
                                )
                                historical_spreads.append(hist_spread)
                else:
                    # 差分协整：根据差分阶数使用相应的差分价格
                    diff_order = pair_info['diff_order']

                    if diff_order == 1:
                        # 一阶差分
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
                        for j in range(max(1, i - self.lookback_period), i):
                            if j < len(all_timestamps):
                                hist_timestamp = all_timestamps[j]
                                prev_hist_timestamp = all_timestamps[j - 1]
                                if (hist_timestamp in data[symbol1].index and
                                        hist_timestamp in data[symbol2].index and
                                        prev_hist_timestamp in data[symbol1].index and
                                        prev_hist_timestamp in data[symbol2].index):
                                    hist_diff1 = data[symbol1].loc[hist_timestamp] - data[symbol1].loc[
                                        prev_hist_timestamp]
                                    hist_diff2 = data[symbol2].loc[hist_timestamp] - data[symbol2].loc[
                                        prev_hist_timestamp]
                                    hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                    historical_spreads.append(hist_spread)

                    elif diff_order == 2:
                        # 二阶差分
                        if i > 1:  # 确保有前两个时间点
                            prev_timestamp = all_timestamps[i - 1]
                            prev2_timestamp = all_timestamps[i - 2]
                            if (prev_timestamp in data[symbol1].index and
                                    prev_timestamp in data[symbol2].index and
                                    prev2_timestamp in data[symbol1].index and
                                    prev2_timestamp in data[symbol2].index):
                                current_diff1 = (current_prices[symbol1] -
                                                 2 * data[symbol1].loc[prev_timestamp] +
                                                 data[symbol1].loc[prev2_timestamp])
                                current_diff2 = (current_prices[symbol2] -
                                                 2 * data[symbol2].loc[prev_timestamp] +
                                                 data[symbol2].loc[prev2_timestamp])
                                current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                            else:
                                current_spread = 0
                        else:
                            current_spread = 0

                        # 获取历史二阶差分价差数据
                        historical_spreads = []
                        for j in range(max(2, i - self.lookback_period), i):
                            if j < len(all_timestamps):
                                hist_timestamp = all_timestamps[j]
                                prev_hist_timestamp = all_timestamps[j - 1]
                                prev2_hist_timestamp = all_timestamps[j - 2]
                                if (hist_timestamp in data[symbol1].index and
                                        hist_timestamp in data[symbol2].index and
                                        prev_hist_timestamp in data[symbol1].index and
                                        prev_hist_timestamp in data[symbol2].index and
                                        prev2_hist_timestamp in data[symbol1].index and
                                        prev2_hist_timestamp in data[symbol2].index):
                                    hist_diff1 = (data[symbol1].loc[hist_timestamp] -
                                                  2 * data[symbol1].loc[prev_hist_timestamp] +
                                                  data[symbol1].loc[prev2_hist_timestamp])
                                    hist_diff2 = (data[symbol2].loc[hist_timestamp] -
                                                  2 * data[symbol2].loc[prev_hist_timestamp] +
                                                  data[symbol2].loc[prev2_hist_timestamp])
                                    hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                    historical_spreads.append(hist_spread)

                    else:
                        # 不支持其他差分阶数
                        current_spread = 0
                        historical_spreads = []

                current_z_score = self.calculate_z_score(current_spread, historical_spreads)

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
                    signal = self.generate_trading_signal(current_z_score, pair_info['diff_order'])
                    signal['z_score'] = current_z_score

                    if signal['action'] != 'HOLD':
                        # 计算可用资金（只乘以杠杆，不再乘以position_ratio）
                        available_capital = capital * self.leverage
                        position = self.execute_trade(pair_info, current_prices, signal, timestamp, current_spread,
                                                      available_capital)
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

        # 计算最终结果
        total_trades = len(results['trades'])
        profitable_trades = len([t for t in results['trades'] if t.get('pnl', 0) > 0])

        # 计算总手续费
        total_fees = sum([t.get('total_fee', 0) for t in results['trades'] if t.get('action') == 'CLOSE'])

        # 计算收益率：基于投入资金（capital的初始值）
        initial_invested_capital = initial_capital * self.position_ratio
        final_return = (capital - initial_invested_capital) / initial_invested_capital * 100

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(results['capital_curve'])

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

        # 绘制收益率曲线图
        print(f"\n正在生成收益率曲线图...")
        self.plot_equity_curve(results['capital_curve'])

        return results


def test_advanced_cointegration_trading(csv_file_path):
    """
    高级版协整分析+交易测试

    Args:
        csv_file_path: CSV文件路径
    """
    print("=" * 80)
    print("高级版协整分析+交易测试（带手续费和动态仓位）")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")

    # 2. 寻找协整对（包含多阶差分检验）
    print("\n2. 寻找协整对（包含多阶差分检验）")
    all_candidates = enhanced_find_cointegrated_pairs(data, max_diff_order=2)

    if not all_candidates:
        print("未找到任何协整对，无法进行交易")
        return

    print(f"找到 {len(all_candidates)} 个协整对候选")

    # 3. 循环测试币对
    print("\n3. 币对测试循环")
    test_count = 0

    while True:
        test_count += 1
        print(f"\n{'=' * 80}")
        print(f"第 {test_count} 次测试")
        print(f"{'=' * 80}")

        # 询问是否继续测试
        continue_choice = input("是否继续测试？(y/n): ").strip().lower()

        if continue_choice != 'y':
            print("测试结束，退出程序")
            break

        # 显示候选币对供选择
        selected_pairs = display_candidates_for_selection(all_candidates)

        if not selected_pairs:
            print("未选择任何币对")
            continue_choice = input("是否继续测试其他币对？(y/n): ").strip().lower()
            if continue_choice != 'y':
                break
            continue

        # 4. 显示选择的币对详情
        print(f"\n第 {test_count} 次测试 - 选择的币对详情")
        for pair in selected_pairs:
            print(f"\n币对: {pair['pair_name']}")
            print(f"  差分阶数: {pair['diff_order']}")
            print(f"  对冲比率: {pair['hedge_ratio']:.6f}")
            print(f"  ADF P值: {pair['best_test']['adf_result']['p_value']:.6f}")
            diff_type = '原始价差' if pair['diff_order'] == 0 else f"{pair['diff_order']}阶差分"
            print(f"  协整类型: {diff_type}")

        # 5. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()

        # 6. 执行交易回测
        print(f"\n第 {test_count} 次测试 - 执行交易回测")
        trading_strategy = AdvancedCointegrationTrading(
            lookback_period=trading_params['lookback_period'],
            z_threshold=trading_params['z_threshold'],
            z_exit_threshold=trading_params['z_exit_threshold'],
            take_profit_pct=trading_params['take_profit_pct'],
            stop_loss_pct=trading_params['stop_loss_pct'],
            max_holding_hours=trading_params['max_holding_hours'],
            position_ratio=trading_params['position_ratio'],
            leverage=trading_params['leverage'],
            trading_fee_rate=trading_params['trading_fee_rate']
        )

        results = trading_strategy.backtest_cointegration_trading(
            data,
            selected_pairs,
            initial_capital=10000
        )

        # 7. 显示交易详情
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
                    if trade['action'] == 'CLOSE':
                        pnl_info = f", 净盈亏: {trade.get('pnl', 0):.2f}, 总手续费: {trade.get('total_fee', 0):.4f}"
                    else:
                        pnl_info = f", 开仓手续费: {trade.get('fee', 0):.4f}"
                    diff_info = f"({trade.get('diff_order', 0)}阶差分)" if trade.get('diff_order', 0) > 0 else "(原始价差)"
                    print(f"  {trade['timestamp']}: {action} {trade['symbol1']}={trade['symbol1_price']:.2f}, {trade['symbol2']}={trade['symbol2_price']:.2f}{pnl_info} {diff_info}")
        else:
            print("本次测试无交易记录")

        print(f"\n第 {test_count} 次测试完成")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("高级版协整分析+交易测试完成")
    print(f"总共进行了 {test_count} 次测试")
    print("=" * 80)


def main():
    """
    主函数
    """
    print("高级版协整分析+交易流程完整测试（带手续费和动态仓位）")
    print("支持多阶差分检验、手工选择币对、止盈止损、风险指标、收益率曲线图")
    print("使用 cointegration_test_windows_optimization_arima_garch.py 的回测逻辑")
    print()

    # 示例用法
    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()

    if not csv_file_path:
        # 默认路径
        csv_file_path = "segment_2_data_ccxt_20251113_103652.csv"
        print(f"使用默认路径: {csv_file_path}")

    # 执行高级版测试
    test_advanced_cointegration_trading(csv_file_path)


if __name__ == "__main__":
    main()
