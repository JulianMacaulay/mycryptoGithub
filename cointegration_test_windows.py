#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级版协整分析+交易流程完整测试文件（滚动窗口版本）
使用滚动窗口进行协整检验：
1. 使用固定大小的滚动窗口（如1000条数据）
2. 在每个窗口内进行完整的EG两阶段协整检验
3. 汇总所有窗口的检验结果，识别协整关系的时变特性
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

def enhanced_cointegration_test(price1, price2, symbol1, symbol2, verbose=True):
    """
    正确的协整检验（Engle-Granger方法）

    步骤：
    1. 检验原序列price1和price2的平稳性
    2. 如果原序列不平稳，检验它们的一阶差分是否平稳
    3. 只有当两个原序列都是I(1)时，才能进行协整检验
    4. 先计算最优对冲比率（OLS）
    5. 计算原序列的价差（price1 - β*price2）
    6. 检验原价差的平稳性
    7. 如果原价差平稳，才是真正的协整

    Args:
        price1: 第一个价格序列
        price2: 第二个价格序列
        symbol1: 第一个币种名称
        symbol2: 第二个币种名称
        verbose: 是否打印详细信息

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
        'best_test': None
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

    # 步骤5: 计算最优对冲比率（OLS回归）
    if verbose:
        print(f"\n--- 步骤3: 计算最优对冲比率（OLS回归） ---")
    hedge_ratio = calculate_hedge_ratio(price1, price2)
    results['hedge_ratio'] = hedge_ratio

    # 步骤6: 计算原序列的价差（残差）
    if verbose:
        print(f"\n--- 步骤4: 计算原序列价差（残差） ---")
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]

    spread = price1_aligned - hedge_ratio * price2_aligned
    results['spread'] = spread

    if verbose:
        print(f"价差统计:")
        print(f"  均值: {spread.mean():.6f}")
        print(f"  标准差: {spread.std():.6f}")
        print(f"  最小值: {spread.min():.6f}")
        print(f"  最大值: {spread.max():.6f}")

    # 步骤7: 检验原价差的平稳性（协整检验）
    if verbose:
        print(f"\n--- 步骤5: 检验原价差的平稳性（协整检验） ---")
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
            print(f"\n✓ 协整检验通过！{symbol1} 和 {symbol2} 存在协整关系")
            print(f"  价差是平稳的（I(0)），ADF P值: {spread_adf['p_value']:.6f}")
    else:
        if verbose:
            print(f"\n✗ 协整检验未通过")
            print(f"  价差不平稳，ADF P值: {spread_adf['p_value']:.6f if spread_adf else 'N/A'}")

    return results


# ==================== 滚动窗口协整检验代码 ====================

def rolling_window_cointegration_test(price1, price2, symbol1, symbol2, window_size=500, step_size=100):
    """
    滚动窗口协整检验

    Args:
        price1: 第一个价格序列
        price2: 第二个价格序列
        symbol1: 第一个币种名称
        symbol2: 第二个币种名称
        window_size: 窗口大小（数据条数）
        step_size: 步长（每次移动的数据条数）

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
                verbose=False  # 窗口检验时不打印详细信息
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
                print(f"  ✓ 协整检验通过 (P值: {coint_result['spread_adf']['p_value']:.6f})")
                all_candidates.append(coint_result)
            else:
                print(f"  ✗ 协整检验未通过")

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


def rolling_window_find_cointegrated_pairs(data, window_size=500, step_size=100):
    """
    滚动窗口寻找协整对

    Args:
        data: 包含各币对数据的字典
        window_size: 窗口大小（数据条数）
        step_size: 步长（每次移动的数据条数）

    Returns:
        list: 协整对汇总结果列表
    """
    print("=" * 80)
    print("滚动窗口协整检验")
    print("=" * 80)
    print(f"窗口大小: {window_size}")
    print(f"步长: {step_size}")

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
                    step_size=step_size
                )

                all_summaries.append(summary)

            except Exception as e:
                print(f"分析 {symbol1}/{symbol2} 时出错: {str(e)}")

    return all_summaries


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


def display_rolling_window_candidates(summaries, min_cointegration_ratio=0.5):
    """
    显示滚动窗口协整对候选列表

    Args:
        summaries: 滚动窗口检验结果汇总列表
        min_cointegration_ratio: 最小协整比例阈值（只显示协整比例>=此值的币对）

    Returns:
        list: 用户选择的币对列表
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

        # 显示最佳窗口的信息（P值最小的窗口）
        if summary['all_candidates']:
            best_window = min(summary['all_candidates'],
                              key=lambda x: x['spread_adf']['p_value'] if x['spread_adf'] else 1.0)
            print(f"   最佳窗口:")
            print(f"     时间范围: {best_window['window_start_time']} 到 {best_window['window_end_time']}")
            print(f"     对冲比率: {best_window['hedge_ratio']:.6f}")
            print(f"     价差ADF P值: {best_window['spread_adf']['p_value']:.6f}")

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
                    # 使用最佳窗口的结果作为交易参数
                    summary = filtered_summaries[idx]
                    if summary['all_candidates']:
                        best_window = min(summary['all_candidates'],
                                          key=lambda x: x['spread_adf']['p_value'] if x['spread_adf'] else 1.0)
                        valid_selection.append(best_window)
                    else:
                        print(f"序号 {idx + 1} 没有协整窗口，跳过")
                else:
                    print(f"序号 {idx + 1} 超出范围")

            if valid_selection:
                print(f"\n已选择 {len(valid_selection)} 个币对:")
                for selected in valid_selection:
                    print(
                        f"   - {selected['pair_name']} (窗口: {selected['window_start_time']} 到 {selected['window_end_time']})")
                return valid_selection
            else:
                print("没有有效的选择，请重新输入")

        except ValueError:
            print("输入格式错误，请输入数字序号，用逗号分隔")
        except KeyboardInterrupt:
            print("\n用户取消选择")
            return []


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
        'position_ratio': 0.5
    }

    print("当前默认参数:")
    print(f"  1. 回看期: {default_params['lookback_period']}")
    print(f"  2. Z-score开仓阈值: {default_params['z_threshold']}")
    print(f"  3. Z-score平仓阈值: {default_params['z_exit_threshold']}")
    print(f"  4. 止盈百分比: {default_params['take_profit_pct'] * 100:.1f}%")
    print(f"  5. 止损百分比: {default_params['stop_loss_pct'] * 100:.1f}%")
    print(f"  6. 最大持仓时间: {default_params['max_holding_hours']}小时")
    print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1-default_params['position_ratio'])*100:.1f}%作为安全垫)")

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

        print("\n修改后的参数:")
        print(f"  1. 回看期: {default_params['lookback_period']}")
        print(f"  2. Z-score开仓阈值: {default_params['z_threshold']}")
        print(f"  3. Z-score平仓阈值: {default_params['z_exit_threshold']}")
        print(f"  4. 止盈百分比: {default_params['take_profit_pct'] * 100:.1f}%")
        print(f"  5. 止损百分比: {default_params['stop_loss_pct'] * 100:.1f}%")
        print(f"  6. 最大持仓时间: {default_params['max_holding_hours']}小时")
        print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1-default_params['position_ratio'])*100:.1f}%作为安全垫)")

    return default_params


# ==================== 高级交易流程代码 ====================

class AdvancedCointegrationTrading:
    """高级协整交易策略类"""

    def __init__(self, lookback_period=60, z_threshold=2.0, z_exit_threshold=0.5,
                 take_profit_pct=0.15, stop_loss_pct=0.08, max_holding_hours=168,
                 position_ratio=0.5):
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
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.z_exit_threshold = z_exit_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_hours = max_holding_hours
        self.position_ratio = position_ratio
        self.positions = {}  # 当前持仓
        self.trades = []  # 交易记录

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

    def calculate_z_score(self, current_spread, historical_spreads):
        """计算当前Z-score"""
        if len(historical_spreads) < 2:
            return 0

        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)

        if spread_std == 0:
            return 0

        return (current_spread - spread_mean) / spread_std

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
            'capital_used': total_capital_used  # 记录使用的资金
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
            'capital_used': total_capital_used
        }
        self.trades.append(trade)

        print(f"开仓: {pair_info['pair_name']}")
        print(f"   信号: {signal['description']}")
        print(f"   价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
        print(f"   价差: {current_spread:.6f}")
        print(f"   仓位: {symbol1}={position['symbol1_size']:.6f}, {symbol2}={position['symbol2_size']:.6f}")
        print(f"   使用资金: {total_capital_used:.2f} / {available_capital:.2f}")

        return position

    def check_exit_conditions(self, pair_info, current_prices, current_z_score, timestamp, current_spread):
        """检查平仓条件（包含止盈止损）"""
        pair_name = pair_info['pair_name']
        if pair_name not in self.positions:
            return False, ""

        position = self.positions[pair_name]
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]

        # 计算当前盈亏（基于价差变化）
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread

        # 根据交易方向计算盈亏
        if position['signal']['action'] == 'SHORT_LONG':
            # 做空价差，价差减少时盈利
            total_pnl = -spread_change
        else:  # LONG_SHORT
            # 做多价差，价差增加时盈利
            total_pnl = spread_change

        # 计算投入资金（基于原始价格）
        entry_value = abs(position['symbol1_size'] * position['entry_prices'][symbol1]) + \
                      abs(position['symbol2_size'] * position['entry_prices'][symbol2])

        # 条件1: Z-score回归到均值附近
        if abs(current_z_score) < self.z_exit_threshold:
            return True, f"Z-score回归到{current_z_score:.3f}，平仓获利"

        # 条件2: 持仓时间过长
        holding_hours = (timestamp - position['entry_time']).total_seconds() / 3600
        if holding_hours > self.max_holding_hours:
            return True, f"持仓时间过长({holding_hours:.1f}小时)，强制平仓"

        # 条件3: 止盈条件
        if entry_value > 0:
            pnl_percentage = total_pnl / entry_value
            if total_pnl > 0 and pnl_percentage > self.take_profit_pct:
                return True, f"止盈触发({pnl_percentage * 100:.1f}%)，平仓获利"

        # 条件4: 止损条件
        if entry_value > 0:
            pnl_percentage = total_pnl / entry_value
            if total_pnl < 0 and pnl_percentage < -self.stop_loss_pct:
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

        # 计算最终盈亏（基于价差变化）
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread

        # 根据交易方向计算盈亏
        if position['signal']['action'] == 'SHORT_LONG':
            # 做空价差，价差减少时盈利
            total_pnl = -spread_change
        else:  # LONG_SHORT
            # 做多价差，价差增加时盈利
            total_pnl = spread_change

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
            'pnl': total_pnl,
            'holding_hours': (timestamp - position['entry_time']).total_seconds() / 3600
        }
        self.trades.append(trade)

        print(f"平仓: {pair_name}")
        print(f"   平仓原因: {reason}")
        print(f"   盈亏: {total_pnl:.2f}")
        print(f"   持仓时间: {trade['holding_hours']:.1f}小时")
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

        # 计算夏普比率
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # 年化
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
        """回测协整交易策略（使用原序列价差）"""
        print("\n" + "=" * 60)
        print("开始高级协整交易回测")
        print("=" * 60)

        # 初始化
        capital = initial_capital
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
        print(f"  可用资金: {initial_capital * self.position_ratio:,.2f} (留{(1-self.position_ratio)*100:.1f}%作为安全垫)")
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
                symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']

                if symbol1 not in current_prices or symbol2 not in current_prices:
                    continue

                # 计算原序列的价差（协整检验通过的是原序列价差）
                current_spread = self.calculate_current_spread(
                    current_prices[symbol1],
                    current_prices[symbol2],
                    pair_info['hedge_ratio']
                )

                # 获取历史价差数据（原序列）
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

                current_z_score = self.calculate_z_score(current_spread, historical_spreads)

                # 检查平仓条件
                if pair_info['pair_name'] in self.positions:
                    should_close, close_reason = self.check_exit_conditions(
                        pair_info, current_prices, current_z_score, timestamp, current_spread
                    )

                    if should_close:
                        trade = self.close_position(pair_info, current_prices, close_reason, timestamp, current_spread)
                        if trade:
                            capital += trade['pnl']
                            results['trades'].append(trade)

                # 检查开仓条件（只有在没有持仓时才开仓）
                elif len(self.positions) == 0:  # 单边持仓模式
                    signal = self.generate_trading_signal(current_z_score)
                    signal['z_score'] = current_z_score

                    if signal['action'] != 'HOLD':
                        # 计算可用资金（根据当前资金和仓位比例）
                        available_capital = capital * self.position_ratio
                        position = self.execute_trade(pair_info, current_prices, signal, timestamp, current_spread, available_capital)
                        if position:
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

        final_return = (capital - initial_capital) / initial_capital * 100

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(results['capital_curve'])

        print(f"\n回测结果:")
        print(f"  初始资金: {initial_capital:,.2f}")
        print(f"  最终资金: {capital:,.2f}")
        print(f"  总收益率: {final_return:.2f}%")
        print(f"  总交易次数: {total_trades / 2}")
        print(f"  盈利交易: {profitable_trades}")
        print(f"  胜率: {profitable_trades / (total_trades / 2) * 100:.1f}%" if total_trades > 0 else "  胜率: 0%")

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


def test_rolling_window_cointegration_trading(csv_file_path):
    """
    滚动窗口协整分析+交易测试

    Args:
        csv_file_path: CSV文件路径
    """
    print("=" * 80)
    print("滚动窗口协整分析+交易测试")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")

    # 2. 配置滚动窗口参数
    print("\n2. 配置滚动窗口参数")
    window_params = configure_rolling_window_parameters()

    # 3. 滚动窗口寻找协整对
    print("\n3. 滚动窗口寻找协整对")
    all_summaries = rolling_window_find_cointegrated_pairs(
        data,
        window_size=window_params['window_size'],
        step_size=window_params['step_size']
    )

    if not all_summaries:
        print("未找到任何协整对，无法进行交易")
        return

    print(f"\n找到 {len(all_summaries)} 个币对的滚动窗口检验结果")

    # 4. 显示并选择协整对
    print("\n4. 选择协整对")
    selected_pairs = display_rolling_window_candidates(all_summaries, min_cointegration_ratio=0.5)

    if not selected_pairs:
        print("未选择任何币对，无法进行交易")
        return

    # 5. 循环测试币对
    print("\n5. 币对测试循环")
    test_count = 0

    while True:
        test_count += 1
        print(f"\n{'=' * 80}")
        print(f"第 {test_count} 次测试")
        print(f"{'=' * 80}")

        continue_choice = input("是否继续测试？(y/n): ").strip().lower()

        if continue_choice != 'y':
            print("测试结束，退出程序")
            break

        # 6. 显示选择的币对详情
        print(f"\n第 {test_count} 次测试 - 选择的币对详情")
        for pair in selected_pairs:
            print(f"\n币对: {pair['pair_name']}")
            print(f"  窗口时间范围: {pair['window_start_time']} 到 {pair['window_end_time']}")
            print(
                f"  积分阶数: {pair['symbol1']}=I({pair['price1_order']}), {pair['symbol2']}=I({pair['price2_order']})")
            print(f"  对冲比率: {pair['hedge_ratio']:.6f}")
            print(f"  价差ADF P值: {pair['spread_adf']['p_value']:.6f}")

        # 7. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()

        # 8. 执行交易回测
        print(f"\n第 {test_count} 次测试 - 执行交易回测")
        trading_strategy = AdvancedCointegrationTrading(
            lookback_period=trading_params['lookback_period'],
            z_threshold=trading_params['z_threshold'],
            z_exit_threshold=trading_params['z_exit_threshold'],
            take_profit_pct=trading_params['take_profit_pct'],
            stop_loss_pct=trading_params['stop_loss_pct'],
            max_holding_hours=trading_params['max_holding_hours'],
            position_ratio=trading_params['position_ratio']
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

    print("\n" + "=" * 80)
    print("滚动窗口协整分析+交易测试完成")
    print(f"总共进行了 {test_count} 次测试")
    print("=" * 80)


def main():
    """
    主函数
    """
    print("滚动窗口协整分析+交易流程完整测试")
    print("使用滚动窗口进行协整检验：")
    print("  1. 使用固定大小的滚动窗口（如1000条数据）")
    print("  2. 在每个窗口内进行完整的EG两阶段协整检验")
    print("  3. 汇总所有窗口的检验结果，识别协整关系的时变特性")
    print()

    # 示例用法
    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()

    if not csv_file_path:
        # 默认路径
        csv_file_path = "segment_2_data_ccxt_20251106_195714.csv"
        print(f"使用默认路径: {csv_file_path}")

    # 执行滚动窗口测试
    test_rolling_window_cointegration_trading(csv_file_path)


if __name__ == "__main__":
    main()