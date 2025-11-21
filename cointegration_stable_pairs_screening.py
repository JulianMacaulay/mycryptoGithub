#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定协整币对筛选工具
功能：
1. 使用样本内数据筛选协整比率较高且对冲比率合理的币对
2. 使用样本外数据验证筛选出的币对
3. 计算两个集合的交集，找出稳定的协整币对
4. 支持多窗口参数组合测试
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 基础函数 ====================

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


def calculate_hedge_ratio(price1, price2, silent=False):
    """
    计算对冲比率（使用OLS回归）

    Args:
        price1: 第一个币种的价格序列
        price2: 第二个币种的价格序列
        silent: 是否静默模式（不打印信息）

    Returns:
        float: 对冲比率
    """
    if not silent:
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

    if not silent:
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


def enhanced_cointegration_test(price1, price2, symbol1, symbol2, verbose=True, diff_order=0):
    """
    正确的协整检验（Engle-Granger方法）

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
        'diff_order': diff_order
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
        # 原始价差
        if verbose:
            print(f"\n--- 步骤3: 计算最优对冲比率（OLS回归，原始价格） ---")
        hedge_ratio = calculate_hedge_ratio(price1_aligned, price2_aligned, silent=not verbose)
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
        # 一阶差分价差
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
        hedge_ratio = calculate_hedge_ratio(diff_price1_aligned, diff_price2_aligned, silent=not verbose)
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
            print(f"\n✓ 协整检验通过！{symbol1} 和 {symbol2} 存在协整关系")
            print(f"  价差是平稳的（I(0)），ADF P值: {spread_adf['p_value']:.6f}")
    else:
        if verbose:
            print(f"\n✗ 协整检验未通过")
            print(f"  价差不平稳，ADF P值: {spread_adf['p_value']:.6f if spread_adf else 'N/A'}")

    return results


def rolling_window_cointegration_test(price1, price2, symbol1, symbol2, window_size=500, step_size=100, diff_order=0, verbose=False):
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
        verbose: 是否打印详细信息

    Returns:
        dict: 滚动窗口检验结果
    """
    if verbose:
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

    if verbose:
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

        if verbose:
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
                diff_order=diff_order
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
                if verbose:
                    print(f"   ✓ 协整检验通过 (P值: {coint_result['spread_adf']['p_value']:.6f})")
                all_candidates.append(coint_result)
            else:
                if verbose:
                    print(f"   ✗ 协整检验未通过")

        except Exception as e:
            if verbose:
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
        'all_candidates': all_candidates,
        'window_size': window_size,
        'step_size': step_size,
        'diff_order': diff_order
    }

    if verbose:
        print(f"\n汇总结果:")
        print(f"  总窗口数: {num_windows}")
        print(f"  协整窗口数: {len(all_candidates)}")
        print(f"  协整比例: {summary['cointegration_ratio'] * 100:.1f}%")

    return summary


def rolling_window_find_cointegrated_pairs(data, window_size=500, step_size=100, diff_order=0, verbose=False):
    """
    滚动窗口寻找协整对

    Args:
        data: 包含各币对数据的字典
        window_size: 窗口大小（数据条数）
        step_size: 步长（每次移动的数据条数）
        diff_order: 价差类型，0=原始价差，1=一阶差分价差
        verbose: 是否打印详细信息

    Returns:
        list: 协整对汇总结果列表
    """
    if verbose:
        print("=" * 80)
        print("滚动窗口协整检验")
        print("=" * 80)
        print(f"窗口大小: {window_size}")
        print(f"步长: {step_size}")
        diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
        print(f"价差类型: {diff_type}")

    symbols = list(data.keys())
    all_summaries = []

    if verbose:
        print(f"\n分析 {len(symbols)} 个币对...")

    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    current_pair = 0

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]
            current_pair += 1

            if verbose:
                print(f"\n[{current_pair}/{total_pairs}] 分析币对: {symbol1}/{symbol2}")

            try:
                # 执行滚动窗口协整检验
                summary = rolling_window_cointegration_test(
                    data[symbol1],
                    data[symbol2],
                    symbol1,
                    symbol2,
                    window_size=window_size,
                    step_size=step_size,
                    diff_order=diff_order,
                    verbose=verbose
                )

                all_summaries.append(summary)

            except Exception as e:
                if verbose:
                    print(f"分析 {symbol1}/{symbol2} 时出错: {str(e)}")

    return all_summaries


# ==================== 筛选函数 ====================

def filter_pairs_by_criteria(summaries, min_cointegration_ratio=0.2, max_hedge_ratio=100.0):
    """
    根据协整比率和对冲比率筛选币对

    Args:
        summaries: 滚动窗口检验结果列表
        min_cointegration_ratio: 最小协整比率（默认0.2，即20%）
        max_hedge_ratio: 最大对冲比率（默认100.0）

    Returns:
        list: 筛选后的币对列表
    """
    filtered_pairs = []

    for summary in summaries:
        pair_name = summary['pair_name']
        cointegration_ratio = summary.get('cointegration_ratio', 0)
        
        # 计算整体对冲比率（使用所有协整窗口的平均值）
        hedge_ratios = []
        for window_result in summary.get('window_results', []):
            if window_result.get('cointegration_found', False):
                hedge_ratio = window_result.get('hedge_ratio')
                if hedge_ratio is not None:
                    hedge_ratios.append(hedge_ratio)
        
        if len(hedge_ratios) == 0:
            continue
        
        avg_hedge_ratio = np.mean(hedge_ratios)
        abs_hedge_ratio = abs(avg_hedge_ratio)
        
        # 筛选条件
        if cointegration_ratio >= min_cointegration_ratio and abs_hedge_ratio <= max_hedge_ratio:
            filtered_pairs.append({
                'pair_name': pair_name,
                'symbol1': summary['symbol1'],
                'symbol2': summary['symbol2'],
                'cointegration_ratio': cointegration_ratio,
                'hedge_ratio': avg_hedge_ratio,
                'abs_hedge_ratio': abs_hedge_ratio,
                'total_windows': summary['total_windows'],
                'cointegration_windows': summary['cointegration_windows'],
                'window_size': summary.get('window_size'),
                'step_size': summary.get('step_size'),
                'diff_order': summary.get('diff_order', 0)
            })

    return filtered_pairs


# ==================== 导出函数 ====================

def export_results_to_csv(results, output_dir="./"):
    """
    将筛选结果导出为CSV文件
    
    Args:
        results: screen_stable_cointegrated_pairs函数返回的结果字典
        output_dir: 输出目录路径（默认当前目录）
    
    Returns:
        list: 导出的文件路径列表
    """
    import os
    from datetime import datetime
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exported_files = []
    
    # 1. 导出样本内筛选结果
    in_sample_data = []
    for (window_size, step_size), result in results['all_results'].items():
        for pair in result['in_sample_filtered']:
            in_sample_data.append({
                '币对名称': pair['pair_name'],
                '币种1': pair['symbol1'],
                '币种2': pair['symbol2'],
                '协整比率': f"{pair['cointegration_ratio']:.4f}",
                '对冲比率': f"{pair['hedge_ratio']:.6f}",
                '对冲比率绝对值': f"{pair['abs_hedge_ratio']:.6f}",
                '总窗口数': pair['total_windows'],
                '协整窗口数': pair['cointegration_windows'],
                '窗口大小': pair['window_size'],
                '步长': pair['step_size'],
                '价差类型': '原始价差' if pair.get('diff_order', 0) == 0 else '一阶差分价差'
            })
    
    if in_sample_data:
        in_sample_df = pd.DataFrame(in_sample_data)
        in_sample_file = os.path.join(output_dir, f"样本内筛选结果_{timestamp}.csv")
        in_sample_df.to_csv(in_sample_file, index=False, encoding='utf-8-sig')
        exported_files.append(in_sample_file)
        print(f"\n✓ 样本内筛选结果已导出: {in_sample_file}")
        print(f"  共 {len(in_sample_data)} 条记录")
    
    # 2. 导出样本外验证结果
    out_sample_data = []
    for (window_size, step_size), result in results['all_results'].items():
        for pair in result['out_sample_filtered']:
            out_sample_data.append({
                '币对名称': pair['pair_name'],
                '币种1': pair['symbol1'],
                '币种2': pair['symbol2'],
                '样本内协整比率': f"{pair['in_sample_cointegration_ratio']:.4f}",
                '样本内对冲比率': f"{pair['in_sample_hedge_ratio']:.6f}",
                '样本外协整比率': f"{pair['out_sample_cointegration_ratio']:.4f}",
                '样本外对冲比率': f"{pair['out_sample_hedge_ratio']:.6f}",
                '窗口大小': pair['window_size'],
                '步长': pair['step_size'],
                '价差类型': '原始价差' if pair.get('diff_order', 0) == 0 else '一阶差分价差'
            })
    
    if out_sample_data:
        out_sample_df = pd.DataFrame(out_sample_data)
        out_sample_file = os.path.join(output_dir, f"样本外验证结果_{timestamp}.csv")
        out_sample_df.to_csv(out_sample_file, index=False, encoding='utf-8-sig')
        exported_files.append(out_sample_file)
        print(f"\n✓ 样本外验证结果已导出: {out_sample_file}")
        print(f"  共 {len(out_sample_data)} 条记录")
    
    # 3. 导出交集结果（稳定币对）
    intersection_data = []
    for (window_size, step_size), result in results['all_results'].items():
        for pair in result['intersection']:
            intersection_data.append({
                '币对名称': pair['pair_name'],
                '币种1': pair['symbol1'],
                '币种2': pair['symbol2'],
                '样本内协整比率': f"{pair['in_sample_cointegration_ratio']:.4f}",
                '样本内对冲比率': f"{pair['in_sample_hedge_ratio']:.6f}",
                '样本外协整比率': f"{pair['out_sample_cointegration_ratio']:.4f}",
                '样本外对冲比率': f"{pair['out_sample_hedge_ratio']:.6f}",
                '窗口大小': pair['window_size'],
                '步长': pair['step_size'],
                '价差类型': '原始价差' if pair.get('diff_order', 0) == 0 else '一阶差分价差'
            })
    
    if intersection_data:
        intersection_df = pd.DataFrame(intersection_data)
        intersection_file = os.path.join(output_dir, f"交集结果_稳定币对_{timestamp}.csv")
        intersection_df.to_csv(intersection_file, index=False, encoding='utf-8-sig')
        exported_files.append(intersection_file)
        print(f"\n✓ 交集结果（稳定币对）已导出: {intersection_file}")
        print(f"  共 {len(intersection_data)} 条记录")
    
    # 4. 导出所有窗口参数组合的汇总结果
    summary_data = []
    for pair_name, occurrences in results['all_intersection_pairs'].items():
        for occ in occurrences:
            pair_info = occ['pair_info']
            summary_data.append({
                '币对名称': pair_name,
                '币种1': pair_info['symbol1'],
                '币种2': pair_info['symbol2'],
                '出现的窗口参数组合数': len(occurrences),
                '窗口大小': occ['window_size'],
                '步长': occ['step_size'],
                '样本内协整比率': f"{pair_info['in_sample_cointegration_ratio']:.4f}",
                '样本内对冲比率': f"{pair_info['in_sample_hedge_ratio']:.6f}",
                '样本外协整比率': f"{pair_info['out_sample_cointegration_ratio']:.4f}",
                '样本外对冲比率': f"{pair_info['out_sample_hedge_ratio']:.6f}",
                '价差类型': '原始价差' if pair_info.get('diff_order', 0) == 0 else '一阶差分价差'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f"所有窗口参数组合汇总结果_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        exported_files.append(summary_file)
        print(f"\n✓ 所有窗口参数组合汇总结果已导出: {summary_file}")
        print(f"  共 {len(summary_data)} 条记录")
        print(f"  唯一币对数: {len(results['all_intersection_pairs'])}")
    
    return exported_files


# ==================== 主函数 ====================

def screen_stable_cointegrated_pairs(
    in_sample_file,
    out_sample_file,
    window_params_list,
    min_cointegration_ratio=0.2,
    max_hedge_ratio=100.0,
    diff_order=0,
    verbose=True,
    export_csv=False,
    output_dir="./"
):
    """
    筛选稳定的协整币对

    Args:
        in_sample_file: 样本内数据文件路径
        out_sample_file: 样本外数据文件路径
        window_params_list: 窗口参数列表，格式：[(window_size, step_size), ...]
        min_cointegration_ratio: 最小协整比率
        max_hedge_ratio: 最大对冲比率
        diff_order: 价差类型，0=原始价差，1=一阶差分价差
        verbose: 是否打印详细信息
        export_csv: 是否自动导出CSV文件（默认False）
        output_dir: CSV文件输出目录（默认当前目录）

    Returns:
        dict: 包含所有结果的字典
    """
    print("=" * 100)
    print("稳定协整币对筛选工具")
    print("=" * 100)
    print(f"样本内数据: {in_sample_file}")
    print(f"样本外数据: {out_sample_file}")
    print(f"窗口参数组合数: {len(window_params_list)}")
    print(f"最小协整比率: {min_cointegration_ratio}")
    print(f"最大对冲比率: {max_hedge_ratio}")
    print(f"价差类型: {'原始价差' if diff_order == 0 else '一阶差分价差'}")
    print("=" * 100)

    # 加载数据
    print("\n1. 加载数据...")
    in_sample_data = load_csv_data(in_sample_file)
    if in_sample_data is None:
        print("样本内数据加载失败！")
        return None

    out_sample_data = load_csv_data(out_sample_file)
    if out_sample_data is None:
        print("样本外数据加载失败！")
        return None

    # 存储所有窗口参数组合的结果
    all_results = {}

    # 对每个窗口参数组合进行测试
    for window_idx, (window_size, step_size) in enumerate(window_params_list):
        print(f"\n{'=' * 100}")
        print(f"窗口参数组合 {window_idx + 1}/{len(window_params_list)}: window_size={window_size}, step_size={step_size}")
        print(f"{'=' * 100}")

        # ========== 步骤1: 样本内数据筛选 ==========
        print(f"\n步骤1: 样本内数据筛选 (window_size={window_size}, step_size={step_size})")
        print("-" * 100)

        in_sample_summaries = rolling_window_find_cointegrated_pairs(
            in_sample_data,
            window_size=window_size,
            step_size=step_size,
            diff_order=diff_order,
            verbose=verbose
        )

        in_sample_filtered = filter_pairs_by_criteria(
            in_sample_summaries,
            min_cointegration_ratio=min_cointegration_ratio,
            max_hedge_ratio=max_hedge_ratio
        )

        print(f"\n样本内筛选结果:")
        print(f"  总币对数: {len(in_sample_summaries)}")
        print(f"  筛选后币对数: {len(in_sample_filtered)}")
        print(f"\n筛选出的币对:")
        for pair in in_sample_filtered:
            print(f"  {pair['pair_name']}: 协整比率={pair['cointegration_ratio']:.2%}, 对冲比率={pair['hedge_ratio']:.6f}")

        # ========== 步骤2: 样本外数据验证 ==========
        print(f"\n步骤2: 样本外数据验证 (window_size={window_size}, step_size={step_size})")
        print("-" * 100)

        # 只对样本内筛选出的币对进行样本外验证
        out_sample_filtered = []
        for pair in in_sample_filtered:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']

            # 检查样本外数据中是否有这两个币对
            if symbol1 not in out_sample_data or symbol2 not in out_sample_data:
                if verbose:
                    print(f"  跳过 {pair['pair_name']}: 样本外数据中缺少该币对")
                continue

            # 对样本外数据进行滚动窗口检验
            try:
                out_sample_summary = rolling_window_cointegration_test(
                    out_sample_data[symbol1],
                    out_sample_data[symbol2],
                    symbol1,
                    symbol2,
                    window_size=window_size,
                    step_size=step_size,
                    diff_order=diff_order,
                    verbose=verbose
                )

                # 筛选样本外结果
                out_sample_pairs = filter_pairs_by_criteria(
                    [out_sample_summary],
                    min_cointegration_ratio=min_cointegration_ratio,
                    max_hedge_ratio=max_hedge_ratio
                )

                if len(out_sample_pairs) > 0:
                    out_pair = out_sample_pairs[0]
                    out_sample_filtered.append({
                        'pair_name': pair['pair_name'],
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'in_sample_cointegration_ratio': pair['cointegration_ratio'],
                        'in_sample_hedge_ratio': pair['hedge_ratio'],
                        'out_sample_cointegration_ratio': out_pair['cointegration_ratio'],
                        'out_sample_hedge_ratio': out_pair['hedge_ratio'],
                        'window_size': window_size,
                        'step_size': step_size,
                        'diff_order': diff_order
                    })

            except Exception as e:
                if verbose:
                    print(f"  样本外验证 {pair['pair_name']} 时出错: {str(e)}")
                continue

        print(f"\n样本外验证结果:")
        print(f"  验证币对数: {len(in_sample_filtered)}")
        print(f"  通过验证币对数: {len(out_sample_filtered)}")

        # ========== 步骤3: 计算交集 ==========
        print(f"\n步骤3: 计算交集")
        print("-" * 100)

        # 交集就是样本外验证通过的币对
        intersection = out_sample_filtered

        print(f"\n交集结果 (样本内筛选且样本外验证通过):")
        print(f"  交集币对数: {len(intersection)}")
        if len(intersection) > 0:
            print(f"\n交集币对详情:")
            for pair in intersection:
                print(f"  {pair['pair_name']}:")
                print(f"    样本内协整比率: {pair['in_sample_cointegration_ratio']:.2%}")
                print(f"    样本内对冲比率: {pair['in_sample_hedge_ratio']:.6f}")
                print(f"    样本外协整比率: {pair['out_sample_cointegration_ratio']:.2%}")
                print(f"    样本外对冲比率: {pair['out_sample_hedge_ratio']:.6f}")
                print(f"    窗口参数: window_size={pair['window_size']}, step_size={pair['step_size']}")

        # 保存当前窗口参数组合的结果
        all_results[(window_size, step_size)] = {
            'in_sample_filtered': in_sample_filtered,
            'out_sample_filtered': out_sample_filtered,
            'intersection': intersection
        }

    # ========== 汇总所有窗口参数组合的结果 ==========
    print(f"\n{'=' * 100}")
    print("所有窗口参数组合结果汇总")
    print(f"{'=' * 100}")

    # 统计所有窗口参数组合的交集
    all_intersection_pairs = {}
    for (window_size, step_size), result in all_results.items():
        for pair in result['intersection']:
            pair_key = pair['pair_name']
            if pair_key not in all_intersection_pairs:
                all_intersection_pairs[pair_key] = []
            all_intersection_pairs[pair_key].append({
                'window_size': window_size,
                'step_size': step_size,
                'pair_info': pair
            })

    print(f"\n所有窗口参数组合下的稳定币对:")
    print(f"  唯一币对数: {len(all_intersection_pairs)}")
    for pair_name, occurrences in all_intersection_pairs.items():
        print(f"\n  {pair_name}:")
        print(f"    出现的窗口参数组合数: {len(occurrences)}")
        for occ in occurrences:
            pair_info = occ['pair_info']
            print(f"      窗口参数: window_size={occ['window_size']}, step_size={occ['step_size']}")
            print(f"        样本内协整比率: {pair_info['in_sample_cointegration_ratio']:.2%}")
            print(f"        样本内对冲比率: {pair_info['in_sample_hedge_ratio']:.6f}")
            print(f"        样本外协整比率: {pair_info['out_sample_cointegration_ratio']:.2%}")
            print(f"        样本外对冲比率: {pair_info['out_sample_hedge_ratio']:.6f}")

    result_dict = {
        'all_results': all_results,
        'all_intersection_pairs': all_intersection_pairs,
        'window_params_list': window_params_list,
        'min_cointegration_ratio': min_cointegration_ratio,
        'max_hedge_ratio': max_hedge_ratio,
        'diff_order': diff_order
    }
    
    # 如果设置了自动导出，则导出CSV文件
    if export_csv:
        print(f"\n{'=' * 100}")
        print("自动导出结果到CSV文件")
        print(f"{'=' * 100}")
        try:
            exported_files = export_results_to_csv(result_dict, output_dir=output_dir)
            if exported_files:
                print(f"\n所有结果已成功导出到目录: {output_dir}")
        except Exception as e:
            print(f"\n导出CSV文件时出错: {str(e)}")
    
    return result_dict


# ==================== 主程序入口 ====================

def main():
    """
    主函数
    """
    print("稳定协整币对筛选工具")
    print("=" * 100)

    # 输入文件路径
    in_sample_file = input("请输入样本内数据文件路径 (例如: 1h_4000_样本内_28.csv): ").strip()
    if not in_sample_file:
        print("使用默认样本内数据文件: 1h_4000_样本内_28.csv")
        in_sample_file = "1h_4000_样本内_28.csv"

    out_sample_file = input("请输入样本外数据文件路径 (例如: 1h_1000_样本外_28.csv): ").strip()
    if not out_sample_file:
        print("使用默认样本外数据文件: 1h_1000_样本外_28.csv")
        out_sample_file = "1h_1000_样本外_28.csv"

    # 选择价差类型
    print("\n请选择价差类型:")
    print("  0. 原始价差")
    print("  1. 一阶差分价差")
    diff_order_input = input("请选择 (0/1, 默认0): ").strip()
    diff_order = 0 if diff_order_input == '' or diff_order_input == '0' else 1

    # 筛选参数
    print("\n请输入筛选参数:")
    min_ratio_input = input("最小协整比率 (默认0.2, 即20%): ").strip()
    min_cointegration_ratio = float(min_ratio_input) if min_ratio_input else 0.2

    max_hedge_input = input("最大对冲比率 (默认100.0): ").strip()
    max_hedge_ratio = float(max_hedge_input) if max_hedge_input else 100.0

    # 窗口参数组合
    window_params_list = [
        (500, 100),
        (720, 120),
        (240, 120),
        (200, 50),
        (1000, 500)
    ]

    print(f"\n将使用以下窗口参数组合:")
    for i, (ws, ss) in enumerate(window_params_list, 1):
        print(f"  {i}. window_size={ws}, step_size={ss}")

    # 是否详细输出
    verbose_input = input("\n是否显示详细输出? (y/n, 默认n): ").strip().lower()
    verbose = verbose_input == 'y'

    # 执行筛选
    results = screen_stable_cointegrated_pairs(
        in_sample_file=in_sample_file,
        out_sample_file=out_sample_file,
        window_params_list=window_params_list,
        min_cointegration_ratio=min_cointegration_ratio,
        max_hedge_ratio=max_hedge_ratio,
        diff_order=diff_order,
        verbose=verbose
    )

    if results:
        print("\n筛选完成！")
        
        # 导出结果到CSV文件
        print("\n" + "=" * 100)
        print("导出结果到CSV文件")
        print("=" * 100)
        
        output_dir_input = input("\n请输入输出目录路径 (直接回车使用当前目录): ").strip()
        output_dir = output_dir_input if output_dir_input else "./"
        
        try:
            exported_files = export_results_to_csv(results, output_dir=output_dir)
            if exported_files:
                print(f"\n{'=' * 100}")
                print("所有结果已成功导出！")
                print(f"{'=' * 100}")
                print("\n导出的文件列表:")
                for i, file_path in enumerate(exported_files, 1):
                    print(f"  {i}. {file_path}")
            else:
                print("\n警告: 没有数据可导出")
        except Exception as e:
            print(f"\n导出CSV文件时出错: {str(e)}")
    else:
        print("\n筛选失败！")


if __name__ == "__main__":
    main()


