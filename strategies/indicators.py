"""
技术指标计算模块
提供常用的技术指标计算函数
"""

import pandas as pd
import numpy as np


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算ATR（平均真实波幅）
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期
        
    Returns:
        ATR序列
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_donchian_high(high: pd.Series, period: int, shift: int = 1) -> pd.Series:
    """
    计算唐奇安通道上轨
    
    Args:
        high: 最高价序列
        period: 周期
        shift: 延后周期（默认1，表示使用前一个周期的数据）
        
    Returns:
        唐奇安通道上轨序列
    """
    if shift > 0:
        high_shifted = high.shift(shift)
        donchian_high = high_shifted.rolling(window=period).max()
    else:
        donchian_high = high.rolling(window=period).max()
    
    return donchian_high


def calculate_donchian_low(low: pd.Series, period: int, shift: int = 1) -> pd.Series:
    """
    计算唐奇安通道下轨
    
    Args:
        low: 最低价序列
        period: 周期
        shift: 延后周期（默认1，表示使用前一个周期的数据）
        
    Returns:
        唐奇安通道下轨序列
    """
    if shift > 0:
        low_shifted = low.shift(shift)
        donchian_low = low_shifted.rolling(window=period).min()
    else:
        donchian_low = low.rolling(window=period).min()
    
    return donchian_low


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """
    计算简单移动平均线
    
    Args:
        close: 收盘价序列
        period: 周期
        
    Returns:
        SMA序列
    """
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """
    计算指数移动平均线
    
    Args:
        close: 收盘价序列
        period: 周期
        
    Returns:
        EMA序列
    """
    return close.ewm(span=period, adjust=False).mean()


def detect_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    检测金叉（快线上穿慢线）
    
    Args:
        fast: 快线序列
        slow: 慢线序列
        
    Returns:
        布尔序列，True表示发生金叉
    """
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def detect_crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    检测死叉（快线下穿慢线）
    
    Args:
        fast: 快线序列
        slow: 慢线序列
        
    Returns:
        布尔序列，True表示发生死叉
    """
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))

