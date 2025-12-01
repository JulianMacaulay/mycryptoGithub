"""
多周期均线缠绕突破策略
基于日线趋势判断和小时线均线缠绕突破的交易策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

# 尝试导入talib，如果失败则使用自定义函数
try:
    import talib
    TALIB_AVAILABLE = False
except ImportError:
    TALIB_AVAILABLE = False
    from .indicators import (
        calculate_ema,
        calculate_sma
    )


class FinalMultiplePeriodStrategy(BaseStrategy):
    """
    多周期均线缠绕突破策略
    
    策略逻辑：
    1. 日线级别：使用25日EMA判断趋势方向
    2. 小时线级别：使用4条EMA（5, 10, 20, 30）检测缠绕
    3. 入场：顺应日线趋势，小时线突破缠绕 + 成交量放大 + 阳线/阴线确认
    4. 退出：止盈（3%）或趋势跌破（ema1 < ema2 或 ema1 > ema2）
    5. 观察窗口：平仓后5根K线内观察反向开仓机会
    """
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典，包含：
                - ema_lens: EMA周期列表（默认[5, 10, 20, 30]）
                - ma_len_daily: 日线趋势EMA周期（默认25）
                - tp_pct: 止盈百分比（默认3.0）
                - vol_factor: 成交量放大因子（默认1.2）
                - watch_bars: 观察窗口K线数（默认5）
        """
        super().__init__(params)
        
        # 策略参数
        self.ema_lens = params.get('ema_lens', [5, 10, 20, 30])
        self.ma_len_daily = params.get('ma_len_daily', 25)
        self.tp_pct = params.get('tp_pct', 3.0)
        self.vol_factor = params.get('vol_factor', 1.2)
        self.watch_bars = params.get('watch_bars', 5)
        
        # 策略状态
        self.exit_bar_long = None  # 平多时的bar索引
        self.exit_bar_short = None  # 平空时的bar索引
        self.watch_short = False  # 是否观察开空
        self.watch_long = False  # 是否观察开多
        
        # 指标缓存
        self.ema1 = None
        self.ema2 = None
        self.ema3 = None
        self.ema4 = None
        self.ema_max = None
        self.ema_min = None
        self.ema_range = None
        self.boll_upper = None
        self.boll_lower = None
        self.boll_width = None
        self.vol_ma = None
        self.daily_close = None  # 日线收盘价
        self.daily_ema = None  # 日线EMA
        
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算所有指标
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        # 转换为numpy数组（talib需要）
        close_array = data['close'].values
        volume_array = data['volume'].values
        
        # 计算小时线EMA（使用talib）
        if TALIB_AVAILABLE:
            self.ema1 = pd.Series(talib.EMA(close_array, timeperiod=self.ema_lens[0]), index=data.index)
            self.ema2 = pd.Series(talib.EMA(close_array, timeperiod=self.ema_lens[1]), index=data.index)
            self.ema3 = pd.Series(talib.EMA(close_array, timeperiod=self.ema_lens[2]), index=data.index)
            self.ema4 = pd.Series(talib.EMA(close_array, timeperiod=self.ema_lens[3]), index=data.index)
        else:
            from .indicators import calculate_ema
            self.ema1 = calculate_ema(data['close'], self.ema_lens[0])
            self.ema2 = calculate_ema(data['close'], self.ema_lens[1])
            self.ema3 = calculate_ema(data['close'], self.ema_lens[2])
            self.ema4 = calculate_ema(data['close'], self.ema_lens[3])
        
        # 计算EMA的最大值和最小值
        ema_df = pd.DataFrame({
            'ema1': self.ema1,
            'ema2': self.ema2,
            'ema3': self.ema3,
            'ema4': self.ema4
        })
        self.ema_max = ema_df.max(axis=1)
        self.ema_min = ema_df.min(axis=1)
        self.ema_range = self.ema_max - self.ema_min
        
        # 计算布林带（使用talib）
        if TALIB_AVAILABLE:
            # talib.BBANDS返回(上轨, 中轨, 下轨)
            boll_upper_array, boll_middle_array, boll_lower_array = talib.BBANDS(
                close_array,
                timeperiod=10,
                nbdevup=2,      # 上轨标准差倍数
                nbdevdn=2,      # 下轨标准差倍数
                matype=0        # 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
            )
            self.boll_upper = pd.Series(boll_upper_array, index=data.index)
            self.boll_lower = pd.Series(boll_lower_array, index=data.index)
            self.boll_width = self.boll_upper - self.boll_lower
        else:
            from .indicators import calculate_sma
            basis = calculate_sma(data['close'], 10)
            dev = data['close'].rolling(window=10).std()
            self.boll_upper = basis + dev * 2
            self.boll_lower = basis - dev * 2
            self.boll_width = self.boll_upper - self.boll_lower
        
        # 计算成交量均线（使用talib）
        if TALIB_AVAILABLE:
            self.vol_ma = pd.Series(talib.SMA(volume_array, timeperiod=13), index=data.index)
        else:
            from .indicators import calculate_sma
            self.vol_ma = calculate_sma(data['volume'], 13)
        
        # 计算日线数据（将小时线数据resample为日线）
        daily_data = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # 计算日线EMA（使用talib）
        daily_close_array = daily_data['close'].values
        if TALIB_AVAILABLE:
            daily_ema_array = talib.EMA(daily_close_array, timeperiod=self.ma_len_daily)
            daily_ema = pd.Series(daily_ema_array, index=daily_data.index)
        else:
            from .indicators import calculate_ema
            daily_ema = calculate_ema(daily_data['close'], self.ma_len_daily)
        
        # 将日线数据对齐到小时线数据（前向填充）
        # 创建日期索引对齐
        daily_ema_aligned = daily_ema.reindex(data.index, method='ffill')
        daily_close_aligned = daily_data['close'].reindex(data.index, method='ffill')
        
        self.daily_ema = daily_ema_aligned
        self.daily_close = daily_close_aligned
        
    def generate_signals(self, data: pd.DataFrame, current_idx: int, position_size: float = 0) -> Dict:
        """
        生成交易信号
        
        Args:
            data: 历史数据DataFrame
            current_idx: 当前数据索引
            position_size: 当前持仓大小（正数=多头，负数=空头，0=无持仓）
            
        Returns:
            信号字典，包含：
            - signal: 'long', 'short', 'close_long', 'close_short', 'hold'
            - price: 信号价格
            - reason: 信号原因
        """
        # 检查数据是否足够
        min_period = max(self.ema_lens) + self.ma_len_daily
        if current_idx < min_period:
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '数据不足'}
        
        current_bar = data.iloc[current_idx]
        
        # 获取当前指标值
        daily_close = self.daily_close.iloc[current_idx]
        daily_ema = self.daily_ema.iloc[current_idx]
        ema1 = self.ema1.iloc[current_idx]
        ema2 = self.ema2.iloc[current_idx]
        ema3 = self.ema3.iloc[current_idx]
        ema4 = self.ema4.iloc[current_idx]
        ema_max = self.ema_max.iloc[current_idx]
        ema_min = self.ema_min.iloc[current_idx]
        ema_range = self.ema_range.iloc[current_idx]
        boll_upper = self.boll_upper.iloc[current_idx]
        boll_lower = self.boll_lower.iloc[current_idx]
        boll_width = self.boll_width.iloc[current_idx]
        vol_ma = self.vol_ma.iloc[current_idx]
        
        # 检查是否有有效值
        if pd.isna(daily_close) or pd.isna(daily_ema) or pd.isna(ema1) or pd.isna(ema2):
            return {'signal': 'hold', 'price': current_bar['close'], 'reason': '指标未就绪'}
        
        # 判断日线趋势
        is_long_trend = daily_close > daily_ema
        is_short_trend = daily_close < daily_ema
        
        # 检测缠绕（EMA范围小于布林带宽度的80%）
        tight_threshold = boll_width * 0.8 if not pd.isna(boll_width) else 0
        is_tight = ema_range < tight_threshold if not pd.isna(ema_range) and not pd.isna(tight_threshold) else False
        
        # 检测突破
        break_up = current_bar['close'] > ema_max and is_tight
        break_down = current_bar['close'] < ema_min and is_tight
        
        # 检测成交量放大
        vol_up = current_bar['volume'] > vol_ma * self.vol_factor if not pd.isna(vol_ma) else False
        
        # 检测趋势跌破（用于退出）
        exit_trend_broken = ema1 < ema2  # 多头退出
        exit_trend_broken_short = ema1 > ema2  # 空头退出
        
        # 计算正常入场条件（用于反向开仓检查）
        long_condition = is_long_trend and break_up and vol_up and current_bar['close'] > current_bar['open']
        short_condition = is_short_trend and break_down and vol_up and current_bar['close'] < current_bar['open']
        
        # 如果有持仓，检查退出条件和反向开仓
        if position_size > 0:  # 多头持仓
            # 检查趋势跌破
            if exit_trend_broken:
                self.exit_bar_long = current_idx
                self.watch_short = True
                return {
                    'signal': 'close_long',
                    'price': current_bar['close'],
                    'reason': '多头趋势跌破出场'
                }
            
            # 检查反向开仓（空头信号）：先平多再开空
            if short_condition:
                return {
                    'signal': 'close_long',
                    'price': current_bar['close'],
                    'reason': '反向开空：平多开空',
                    'reverse_signal': 'short',  # 标记需要反向开仓
                    'reverse_price': current_bar['close']
                }
        
        if position_size < 0:  # 空头持仓
            # 检查趋势升破
            if exit_trend_broken_short:
                self.exit_bar_short = current_idx
                self.watch_long = True
                return {
                    'signal': 'close_short',
                    'price': current_bar['close'],
                    'reason': '空头趋势升破出场'
                }
            
            # 检查反向开仓（多头信号）：先平空再开多
            if long_condition:
                return {
                    'signal': 'close_short',
                    'price': current_bar['close'],
                    'reason': '反向开多：平空开多',
                    'reverse_signal': 'long',  # 标记需要反向开仓
                    'reverse_price': current_bar['close']
                }
        
        # 如果没有持仓，检查入场信号或观察窗口
        if position_size == 0:
            # 检查观察窗口开仓
            if self.watch_short and current_bar['close'] < current_bar['open']:
                # 平多后观察开空：需要阴线且跌破布林下轨
                if not pd.isna(boll_lower) and current_bar['low'] < boll_lower:
                    self.watch_short = False
                    self.exit_bar_long = None
                    return {
                        'signal': 'short',
                        'price': current_bar['close'],
                        'reason': '观察窗口开空'
                    }
                # 超过观察窗口，取消观察
                elif self.exit_bar_long is not None and (current_idx - self.exit_bar_long) >= self.watch_bars:
                    self.watch_short = False
                    self.exit_bar_long = None
            
            if self.watch_long and current_bar['close'] > current_bar['open']:
                # 平空后观察开多：需要阳线且突破布林上轨
                if not pd.isna(boll_upper) and current_bar['high'] > boll_upper:
                    self.watch_long = False
                    self.exit_bar_short = None
                    return {
                        'signal': 'long',
                        'price': current_bar['close'],
                        'reason': '观察窗口开多'
                    }
                # 超过观察窗口，取消观察
                elif self.exit_bar_short is not None and (current_idx - self.exit_bar_short) >= self.watch_bars:
                    self.watch_long = False
                    self.exit_bar_short = None
            
            # 正常入场信号（顺应日线趋势）
            if long_condition:
                return {
                    'signal': 'long',
                    'price': current_bar['close'],
                    'reason': '多头入场（日线趋势向上+小时线突破缠绕+成交量放大+阳线）'
                }
            
            if short_condition:
                return {
                    'signal': 'short',
                    'price': current_bar['close'],
                    'reason': '空头入场（日线趋势向下+小时线突破缠绕+成交量放大+阴线）'
                }
        
        return {'signal': 'hold', 'price': current_bar['close'], 'reason': '无信号'}
    
    def check_stop_loss(self, data: pd.DataFrame, current_idx: int,
                       position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查止盈止损
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            止盈信号字典或None
        """
        if last_entry_price <= 0:
            return None
        
        current_bar = data.iloc[current_idx]
        
        # 多头止盈
        if position_size > 0:
            profit_pct = (current_bar['close'] - last_entry_price) / last_entry_price * 100
            if profit_pct >= self.tp_pct:
                self.exit_bar_long = current_idx
                return {
                    'signal': 'close_long',
                    'price': current_bar['close'],
                    'reason': '多头盈利止盈'
                }
        
        # 空头止盈
        elif position_size < 0:
            profit_pct = (last_entry_price - current_bar['close']) / last_entry_price * 100
            if profit_pct >= self.tp_pct:
                self.exit_bar_short = current_idx
                return {
                    'signal': 'close_short',
                    'price': current_bar['close'],
                    'reason': '空头盈利止盈'
                }
        
        return None
    
    def check_add_position(self, data: pd.DataFrame, current_idx: int,
                          position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查是否需要加仓（本策略不支持加仓）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            None（本策略不支持加仓）
        """
        return None
    
    def update_trade_result(self, profit: float):
        """
        更新交易结果
        
        Args:
            profit: 上次交易的盈亏（正数=盈利，负数=亏损）
        """
        pass

