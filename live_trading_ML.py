#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘交易系统（支持机器学习市场状态检测）
基于 backtest_system_ML.py 的核心交易逻辑
结合实时数据获取和交易执行

功能：
1. 实时数据获取和管理
2. 机器学习市场状态检测（趋势/震荡）
3. 根据市场状态动态选择策略
4. 实时交易执行和监控
5. Web监控界面（可选）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import threading
import json
import requests
import hmac
import hashlib
import urllib.parse
from urllib.parse import urlencode
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import os
import sys
import importlib
import importlib.util
import inspect

warnings.filterwarnings('ignore')

# 添加策略模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.turtle_strategy import TurtleStrategy
from strategies.base_strategy import BaseStrategy

# 尝试导入机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装，Random Forest功能将不可用。可以使用: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost未安装，XGBoost功能将不可用。可以使用: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: tensorflow未安装，LSTM和CNN功能将不可用。可以使用: pip install tensorflow")

# 尝试导入Flask（用于Web监控界面）
try:
    from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
    from flask_cors import CORS
    from werkzeug.security import generate_password_hash, check_password_hash
    import sqlite3
    from functools import wraps
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("警告: Flask未安装，Web监控界面将不可用。可以使用: pip install flask flask-cors")

# 导入回测系统中的市场状态检测器和技术指标
# 这里需要从backtest_system_ML.py导入相关类和函数
# 为了简化，我们直接复制必要的代码

# ==================== 币安API配置 ====================

# API配置（请修改为您的实际API密钥）
API_KEY = "SdTSZxmdf61CFsze3udgLRWq0aCaVyyFjsrYKMUOWIfMkm7q3sGRkzSk6QSbM5Qk"
SECRET_KEY = "9HZ04wgrKTy5kDPF5Kman4WSmS9D7YlTscPA7FtX2YLK7vTbpORFNB2jTABQY6HY"
BASE_URL = "https://testnet.binancefuture.com"  # 测试网，实盘请改为 "https://fapi.binance.com"


# ==================== 技术指标计算函数（从backtest_system_ML.py复制） ====================

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ADX（平均趋向指标）"""
    # 计算真实波幅（TR）
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算方向移动（+DM和-DM）
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # 平滑TR、+DM、-DM
    atr = tr.rolling(window=period).mean()
    # 防止除零
    atr = atr.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr).fillna(0)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr).fillna(0)
    
    # 计算DX（防止除零）
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / denominator
    dx = dx.fillna(0)
    
    # 计算ADX
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI（相对强弱指标）"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 防止除零
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rs = rs.fillna(100)  # loss=0时，RS=inf，RSI=100
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bb_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band


def calculate_ma_slope(close: pd.Series, period: int = 20, lookback: int = 5) -> pd.Series:
    """计算均线斜率"""
    ma = close.rolling(window=period).mean()
    slope = (ma - ma.shift(lookback)) / ma.shift(lookback) * 100
    return slope


# ==================== 币安API客户端 ====================

class BinanceAPI:
    """币安API客户端"""

    def __init__(self, api_key=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL):
        """
        初始化币安API客户端

        Args:
            api_key: API密钥
            secret_key: 密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url

    def _generate_signature(self, query_string):
        """生成签名"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self):
        """获取请求头"""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }

    def get_current_price(self, symbol):
        """获取当前价格"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/price"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                print(f"获取价格失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"获取价格异常: {str(e)}")
            return None

    def get_klines(self, symbol, interval='1h', limit=100):
        """获取K线数据"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                klines = []
                for kline in data:
                    klines.append({
                        'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                return klines
            else:
                print(f"获取K线数据失败: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"获取K线数据异常: {str(e)}")
            return []

    def place_order(self, symbol, side, quantity, order_type='MARKET'):
        """下单"""
        try:
            url = f"{self.base_url}/fapi/v1/order"

            # 构建参数
            params = {
                'symbol': symbol,
                'side': side,  # 'BUY' or 'SELL'
                'type': order_type,
                'quantity': str(quantity),
                'timestamp': int(time.time() * 1000)
            }

            # 生成签名
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature

            # 发送请求
            response = requests.post(url, params=params, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"下单成功: {symbol} {side} {quantity} - OrderID: {data.get('orderId')}")
                return data
            else:
                print(f"下单失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"下单异常: {str(e)}")
            return None

    def get_account_info(self):
        """获取账户信息"""
        try:
            url = f"{self.base_url}/fapi/v2/account"

            # 构建参数
            params = {
                'timestamp': int(time.time() * 1000)
            }

            # 生成签名
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature

            # 发送请求
            response = requests.get(url, params=params, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"获取账户信息失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"获取账户信息异常: {str(e)}")
            return None

    def get_position_info(self, symbol=None):
        """获取持仓信息"""
        try:
            url = f"{self.base_url}/fapi/v2/positionRisk"

            # 构建参数
            params = {
                'timestamp': int(time.time() * 1000)
            }
            if symbol:
                params['symbol'] = symbol

            # 生成签名
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature

            # 发送请求
            response = requests.get(url, params=params, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"获取持仓信息失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"获取持仓信息异常: {str(e)}")
            return None

    def get_symbol_precision(self, symbol):
        """获取交易对的精度信息"""
        try:
            # 使用币安主网API获取精度信息，无需认证
            url = "https://demo-fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                for symbol_info in data['symbols']:
                    if symbol_info['symbol'] == symbol:
                        for filter_info in symbol_info['filters']:
                            if filter_info['filterType'] == 'LOT_SIZE':
                                step_size = float(filter_info['stepSize'])
                                return step_size
            return 0.001  # 默认精度
        except Exception as e:
            print(f"获取 {symbol} 精度信息异常: {str(e)}")
            return 0.001


# ==================== 实时数据管理器 ====================

class RealTimeDataManager:
    """实时数据管理器"""

    def __init__(self, binance_api):
        self.binance_api = binance_api
        self.data_cache = {}  # 存储完整的OHLCV数据
        self.running = False
        self.update_thread = None
        self.symbols = []
        self.interval = '1h'

    def start_data_collection(self, symbols, interval='1h'):
        """开始数据收集"""
        self.symbols = symbols
        self.interval = interval
        self.running = True

        # 启动数据更新线程
        self.update_thread = threading.Thread(target=self._update_data_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        print(f"开始收集实时数据: {symbols}")

    def stop_data_collection(self):
        """停止数据收集"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        print("停止数据收集")

    def _update_data_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                for symbol in self.symbols:
                    klines = self.binance_api.get_klines(symbol, self.interval, 500)
                    if klines:
                        df = pd.DataFrame(klines)
                        df.set_index('timestamp', inplace=True)
                        # 存储完整的OHLCV数据
                        self.data_cache[symbol] = df

                # 根据interval确定更新频率
                if self.interval == '1m':
                    time.sleep(60)
                elif self.interval == '5m':
                    time.sleep(300)
                elif self.interval == '15m':
                    time.sleep(1500)
                elif self.interval == '30m':
                    time.sleep(3000)
                elif self.interval == '1h':
                    time.sleep(3600)
                elif self.interval == '4h':
                    time.sleep(14400)
                elif self.interval == '1d':
                    time.sleep(86400)
                else:
                    time.sleep(60)
            except Exception as e:
                print(f"数据更新异常: {str(e)}")
                time.sleep(10)

    def get_current_data(self):
        """获取当前数据（完整OHLCV）"""
        return self.data_cache.copy()

    def get_current_prices(self):
        """获取当前价格（实时价格，用于监控）"""
        prices = {}
        for symbol in self.symbols:
            price = self.binance_api.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices

    def get_latest_closed_kline_prices(self):
        """获取最新已收盘K线的收盘价（用于交易决策）"""
        prices = {}
        for symbol in self.symbols:
            if symbol in self.data_cache and len(self.data_cache[symbol]) > 0:
                # 获取最后一个K线的收盘价（已收盘的K线）
                prices[symbol] = self.data_cache[symbol]['close'].iloc[-1]
        return prices

    def get_latest_closed_kline_timestamp(self):
        """获取最新已收盘K线的时间戳"""
        latest_timestamp = None
        for symbol in self.symbols:
            if symbol in self.data_cache and len(self.data_cache[symbol]) > 0:
                timestamp = self.data_cache[symbol].index[-1]
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
        return latest_timestamp

    def collect_warmup_data(self, symbols, interval='1h', warmup_period=2000):
        """
        收集预热数据

        Args:
            symbols: 交易对列表
            interval: K线间隔
            warmup_period: 预热期长度（数据点数量）

        Returns:
            dict: 预热数据字典（完整OHLCV）
        """
        print(f"\n开始收集预热数据（需要 {warmup_period} 个数据点）...")
        warmup_data = {}

        for symbol in symbols:
            print(f"  收集 {symbol} 的预热数据...")
            klines = self.binance_api.get_klines(symbol, interval, warmup_period)
            if klines:
                df = pd.DataFrame(klines)
                df.set_index('timestamp', inplace=True)
                warmup_data[symbol] = df
                print(f"     {symbol}: {len(warmup_data[symbol])} 个数据点")
            else:
                print(f"     {symbol}: 数据收集失败")

        # 检查数据是否足够
        min_length = min([len(data) for data in warmup_data.values()]) if warmup_data else 0
        if min_length < warmup_period:
            print(f"警告: 预热数据不足，只收集到 {min_length} 个数据点（需要 {warmup_period} 个）")
        else:
            print(f" 预热数据收集完成，每个交易对 {min_length} 个数据点")

        return warmup_data


# ==================== 市场状态检测器（从backtest_system_ML.py导入） ====================

try:
    # 尝试从backtest_system_ML导入市场状态检测器
    # 需要确保backtest_system_ML.py在同一目录
    import importlib.util
    spec = importlib.util.spec_from_file_location("backtest_system_ML", 
                                                  os.path.join(os.path.dirname(__file__), "backtest_system_ML.py"))
    if spec and spec.loader:
        backtest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backtest_module)
        MarketRegimeMLDetector = backtest_module.MarketRegimeMLDetector
        OnlineRegimeSegmenter = getattr(backtest_module, 'OnlineRegimeSegmenter', None)
        ML_DETECTOR_AVAILABLE = True
    else:
        raise ImportError("无法加载backtest_system_ML模块")
except (ImportError, AttributeError) as e:
    ML_DETECTOR_AVAILABLE = False
    print(f"警告: 无法从backtest_system_ML导入市场状态检测器: {e}")
    print("将使用简化版本，请确保backtest_system_ML.py在同一目录")
    
    # 创建简化版本（仅用于占位，实际功能不可用）
    class MarketRegimeMLDetector:
        def __init__(self, *args, **kwargs):
            self.is_trained = False
            print("警告: 使用简化版市场状态检测器")
        
        def train(self, data):
            self.is_trained = False
            print("警告: 简化版检测器无法训练")
        
        def predict(self, data, return_confidence=False):
            if return_confidence:
                return pd.Series(['ranging'] * len(data), index=data.index), pd.Series([0.0] * len(data), index=data.index)
            return pd.Series(['ranging'] * len(data), index=data.index)
    
    OnlineRegimeSegmenter = None


# ==================== 实盘交易引擎 ====================

class LiveTradingEngine:
    """
    实盘交易引擎
    负责执行实盘交易，记录交易，计算收益
    """

    def __init__(self, binance_api: BinanceAPI, symbol: str = 'BTCUSDT',
                 initial_capital: float = 10000, leverage: float = 1.0, 
                 position_ratio: float = 0.5, commission_rate: float = 0.000275):
        """
        初始化实盘交易引擎

        Args:
            binance_api: 币安API客户端
            symbol: 交易对符号
            initial_capital: 初始资金
            leverage: 杠杆倍数
            position_ratio: 仓位比例
            commission_rate: 手续费率
        """
        self.binance_api = binance_api
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate

        # 计算可用资金（考虑仓位比例）
        self.available_capital = initial_capital * position_ratio

        # 账户状态
        self.balance = self.available_capital
        self.equity = self.available_capital

        # 多持仓管理
        self.positions = []  # 持仓列表
        self.position_value = 0.0

        # 交易记录
        self.trades = []
        self.equity_curve = []
        self.signals = []

    def reset(self):
        """重置交易引擎"""
        self.balance = self.available_capital
        self.equity = self.available_capital
        self.positions = []
        self.position_value = 0.0
        self.trades = []
        self.equity_curve = []
        self.signals = []

    def get_total_position_size(self) -> float:
        """获取总持仓数量"""
        return sum(pos['size'] for pos in self.positions)

    def has_positions(self) -> bool:
        """检查是否有持仓"""
        return len(self.positions) > 0

    def get_position_by_id(self, position_id: str = 'default'):
        """根据position_id获取持仓"""
        for pos in self.positions:
            if pos['position_id'] == position_id:
                return pos
        return None

    def get_all_position_ids(self) -> list:
        """获取所有持仓ID列表"""
        return [pos['position_id'] for pos in self.positions]

    def update_equity(self, current_price: float):
        """更新权益（基于当前价格）"""
        # 计算持仓价值
        position_value = 0.0
        for pos in self.positions:
            position_value += pos['size'] * current_price

        self.position_value = position_value
        self.equity = self.balance + position_value
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': self.equity,
            'balance': self.balance,
            'position_value': position_value
        })

    def open_position(self, signal: str, price: float, size: float, 
                     timestamp: datetime, reason: str, position_id: str = None,
                     strategy: BaseStrategy = None):
        """
        开仓（实盘）

        Args:
            signal: 信号类型 ('long' or 'short')
            price: 开仓价格
            size: 开仓数量
            timestamp: 时间戳
            reason: 开仓原因
            position_id: 持仓ID（默认为'default'）
            strategy: 策略对象
        """
        if position_id is None:
            position_id = 'default'

        # 检查是否已有相同ID的持仓
        existing_pos = self.get_position_by_id(position_id)
        if existing_pos:
            print(f"警告: 持仓ID {position_id} 已存在，无法重复开仓")
            return None

        # 计算所需资金
        required_capital = abs(size) * price / self.leverage
        if required_capital > self.balance:
            print(f"资金不足: 需要 {required_capital:.2f}，可用 {self.balance:.2f}")
            return None

        # 实盘下单
        side = 'BUY' if signal == 'long' else 'SELL'
        order = self.binance_api.place_order(
            symbol=self.symbol,
            side=side,
            quantity=abs(size)
        )

        if order and order.get('orderId'):
            # 创建持仓记录
            position = {
                'position_id': position_id,
                'size': size if signal == 'long' else -size,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_idx': len(self.trades),
                'entry_count': 1,
                'strategy': strategy,
                'order_id': order.get('orderId')
            }
            self.positions.append(position)

            # 扣除资金
            self.balance -= required_capital

            # 记录交易
            trade = {
                'timestamp': timestamp,
                'action': 'OPEN',
                'signal': signal,
                'price': price,
                'size': size,
                'reason': reason,
                'position_id': position_id,
                'order_id': order.get('orderId')
            }
            self.trades.append(trade)

            print(f"实盘开仓: {signal} {size:.6f} @ {price:.2f} - {reason}")
            return position
        else:
            print(f"实盘开仓失败: 订单未成功")
            return None

    def close_position(self, price: float, timestamp: datetime, reason: str, position_id: str = None):
        """
        平仓（实盘）

        Args:
            price: 平仓价格
            timestamp: 时间戳
            reason: 平仓原因
            position_id: 持仓ID

        Returns:
            盈亏金额
        """
        if position_id is None:
            position_id = 'default'

        position = self.get_position_by_id(position_id)
        if not position:
            return None

        # 计算盈亏
        pnl = position['size'] * (price - position['entry_price'])

        # 实盘平仓（反向操作）
        side = 'SELL' if position['size'] > 0 else 'BUY'
        order = self.binance_api.place_order(
            symbol=self.symbol,
            side=side,
            quantity=abs(position['size'])
        )

        if order and order.get('orderId'):
            # 计算手续费
            trade_value = abs(position['size']) * price
            fee = trade_value * self.commission_rate

            # 更新资金
            self.balance += abs(position['size']) * position['entry_price'] / self.leverage + pnl - fee

            # 移除持仓
            self.positions = [p for p in self.positions if p['position_id'] != position_id]

            # 记录交易
            trade = {
                'timestamp': timestamp,
                'action': 'CLOSE',
                'price': price,
                'size': position['size'],
                'pnl': pnl,
                'fee': fee,
                'reason': reason,
                'position_id': position_id,
                'order_id': order.get('orderId')
            }
            self.trades.append(trade)

            print(f"实盘平仓: {position_id} @ {price:.2f} - PnL: {pnl:.2f} - {reason}")
            return pnl - fee
        else:
            print(f"实盘平仓失败: 订单未成功")
            return None

    def add_position(self, signal: str, price: float, size: float,
                    timestamp: datetime, reason: str, position_id: str):
        """加仓"""
        position = self.get_position_by_id(position_id)
        if not position:
            return None

        # 计算所需资金
        required_capital = abs(size) * price / self.leverage
        if required_capital > self.balance:
            print(f"资金不足: 需要 {required_capital:.2f}，可用 {self.balance:.2f}")
            return None

        # 实盘下单
        side = 'BUY' if signal == 'long' else 'SELL'
        order = self.binance_api.place_order(
            symbol=self.symbol,
            side=side,
            quantity=abs(size)
        )

        if order and order.get('orderId'):
            # 更新持仓（计算平均价格）
            total_size = position['size'] + (size if signal == 'long' else -size)
            total_value = position['size'] * position['entry_price'] + size * price
            avg_price = total_value / total_size if total_size != 0 else price

            position['size'] = total_size
            position['entry_price'] = avg_price
            position['entry_count'] += 1

            # 扣除资金
            self.balance -= required_capital

            # 记录交易
            trade = {
                'timestamp': timestamp,
                'action': 'ADD',
                'signal': signal,
                'price': price,
                'size': size,
                'reason': reason,
                'position_id': position_id,
                'order_id': order.get('orderId')
            }
            self.trades.append(trade)

            print(f"实盘加仓: {position_id} {signal} {size:.6f} @ {price:.2f} - {reason}")
            return position
        else:
            print(f"实盘加仓失败: 订单未成功")
            return None


# ==================== 实盘交易系统 ====================

class LiveTradingSystem:
    """
    实盘交易系统（支持机器学习市场状态检测）
    """

    def __init__(self, binance_api: BinanceAPI, data_manager: RealTimeDataManager,
                 symbol: str = 'BTCUSDT',
                 strategy: BaseStrategy = None, strategies: Dict[str, BaseStrategy] = None,
                 initial_capital: float = 10000, leverage: float = 1.0, position_ratio: float = 0.5,
                 market_detector: MarketRegimeMLDetector = None,
                 regime_detection_method: str = 'per_kline'):
        """
        初始化实盘交易系统

        Args:
            binance_api: 币安API客户端
            data_manager: 实时数据管理器
            symbol: 交易对符号
            strategy: 单一策略实例（向后兼容）
            strategies: 策略字典 {'trending': 趋势策略, 'ranging': 震荡策略}
            initial_capital: 初始资金
            leverage: 杠杆倍数
            position_ratio: 仓位比例
            market_detector: 市场状态检测器
            regime_detection_method: 市场状态检测方法
        """
        self.binance_api = binance_api
        self.data_manager = data_manager
        self.symbol = symbol

        # 策略管理
        if strategies is not None:
            self.strategies = strategies
            self.strategy = None
        elif strategy is not None:
            self.strategy = strategy
            self.strategies = None
        else:
            raise ValueError("必须提供strategy或strategies参数")

        self.market_detector = market_detector
        self.use_market_regime = market_detector is not None
        self.regime_detection_method = regime_detection_method

        # 创建实盘交易引擎
        self.engine = LiveTradingEngine(
            binance_api=binance_api,
            symbol=symbol,
            initial_capital=initial_capital,
            leverage=leverage,
            position_ratio=position_ratio
        )

        # 市场状态相关
        self.market_regimes = None
        self.market_confidences = None
        self.entry_strategy = None
        self.regime_switch_count = 0
        self.strategy_usage_count = {'trending': 0, 'ranging': 0, 'none': 0}

        # 前瞻性预测参数
        self.future_lookahead = 30

        # 在线分段参数
        self.min_segment_length = 20
        self.change_threshold = 0.7

        # 运行状态
        self.running = False

        # 在线分段器（如果使用segmentation方法）
        self.online_segmenter = None
        if self.use_market_regime and self.regime_detection_method == 'segmentation':
            if ML_DETECTOR_AVAILABLE and OnlineRegimeSegmenter is not None:
                self.online_segmenter = OnlineRegimeSegmenter(
                    detector=self.market_detector,
                    min_segment_length=self.min_segment_length,
                    change_threshold=self.change_threshold
                )

        # 资金曲线
        self.capital_curve = []

    # def initialize_strategies(self, warmup_data: Dict[str, pd.DataFrame]):

    def initialize_strategies(self, warmup_data: Dict[str, pd.DataFrame]):
        """
        初始化策略（使用预热数据）

        Args:
            warmup_data: 预热数据字典
        """
        # 合并所有symbol的数据为单一DataFrame（用于策略初始化）
        # 这里简化处理，假设只有一个交易对
        if len(warmup_data) > 0:
            first_symbol = list(warmup_data.keys())[0]
            data = warmup_data[first_symbol]

            if self.strategies is not None:
                # 多策略模式
                print("\n初始化策略...")
                for regime_type, strategy in self.strategies.items():
                    print(f"  初始化{regime_type}策略: {strategy.name}")
                    strategy.initialize(data)
                    strategy.engine = self.engine
            else:
                # 单策略模式
                self.strategy.initialize(data)
                self.strategy.engine = self.engine

    def get_current_regime(self, current_data: Dict[str, pd.DataFrame], current_idx: int = None):
        """
        获取当前市场状态

        Args:
            current_data: 当前数据字典
            current_idx: 当前索引（用于前瞻性预测）

        Returns:
            (市场状态, 置信度)
        """
        if not self.use_market_regime or not self.market_detector.is_trained:
            return 'ranging', 1.0

        # 合并数据为单一DataFrame
        if len(current_data) > 0:
            first_symbol = list(current_data.keys())[0]
            data = current_data[first_symbol]

            if self.regime_detection_method == 'per_kline':
                # 每根K线都预测
                if len(data) > 0:
                    # 使用最后的数据点进行预测
                    recent_data = data.iloc[-50:] if len(data) >= 50 else data
                    regime, confidence = self.market_detector.predict(
                        recent_data, return_confidence=True
                    )
                    if isinstance(regime, pd.Series):
                        regime = regime.iloc[-1] if len(regime) > 0 else 'ranging'
                    if isinstance(confidence, pd.Series):
                        confidence = confidence.iloc[-1] if len(confidence) > 0 else 0.0
                    return regime, confidence
                return 'ranging', 0.0

            elif self.regime_detection_method == 'future_looking':
                # 前瞻性预测
                if current_idx is not None and len(data) > current_idx:
                    regime, confidence = self.market_detector.predict_future_regime(
                        data, current_idx=current_idx, lookahead=self.future_lookahead,
                        return_confidence=True
                    )
                    return regime, confidence
                return 'ranging', 0.0

            elif self.regime_detection_method == 'segmentation':
                # 在线分段
                if self.online_segmenter:
                    regime, confidence = self.online_segmenter.update(data, len(data) - 1)
                    return regime, confidence
                return 'ranging', 0.0

        return 'ranging', 0.0

    def select_strategy_by_regime(self, current_regime: str, current_confidence: float):
        """
        根据市场状态选择策略

        Args:
            current_regime: 当前市场状态
            current_confidence: 当前置信度

        Returns:
            选择的策略对象或None
        """
        if not self.use_market_regime or self.strategies is None:
            return self.strategy

        # 检查置信度
        avg_confidence = self.market_confidences.mean() if self.market_confidences is not None else 1.0
        use_confidence_threshold = avg_confidence >= 0.3

        if not use_confidence_threshold or current_confidence >= self.market_detector.confidence_threshold:
            # 高置信度或平均置信度很低：直接使用预测结果
            if current_regime == 'trending' and 'trending' in self.strategies:
                self.strategy_usage_count['trending'] += 1
                return self.strategies['trending']
            elif current_regime == 'ranging' and 'ranging' in self.strategies:
                self.strategy_usage_count['ranging'] += 1
                return self.strategies['ranging']
            else:
                self.strategy_usage_count['none'] += 1
                return None
        else:
            # 低置信度：仍然使用预测结果
            if current_regime == 'trending' and 'trending' in self.strategies:
                self.strategy_usage_count['trending'] += 1
                return self.strategies['trending']
            elif current_regime == 'ranging' and 'ranging' in self.strategies:
                self.strategy_usage_count['ranging'] += 1
                return self.strategies['ranging']
            else:
                self.strategy_usage_count['none'] += 1
                return None

    def start_trading(self, max_entries: int = 3):
        """
        开始实盘交易

        Args:
            max_entries: 最大加仓次数
        """
        if not self.data_manager.running:
            print("错误: 数据管理器未启动")
            return

        self.running = True
        last_processed_timestamp = None
        last_status_output = 0  # 上次输出状态的时间戳

        print("\n开始实盘交易...")
        print(f"初始资金: {self.engine.initial_capital}")
        print(f"杠杆倍数: {self.engine.leverage}x")
        print(f"仓位比例: {self.engine.position_ratio * 100:.1f}%")

        while self.running:
            try:
                # 获取当前数据
                current_data = self.data_manager.get_current_data()
                current_prices = self.data_manager.get_current_prices()

                if not current_data or not current_prices:
                    time.sleep(10)
                    continue

                # 每10秒输出一次持仓和交易信息
                current_time = time.time()
                if current_time - last_status_output >= 10:
                    print(f"\n{'=' * 60}")
                    print(f"交易状态监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'=' * 60}")
                    
                    # 输出账户信息
                    print(f"账户信息:")
                    print(f"  初始资金: {self.engine.initial_capital:.2f}")
                    print(f"  当前余额: {self.engine.balance:.2f}")
                    print(f"  持仓价值: {self.engine.position_value:.2f}")
                    print(f"  总权益: {self.engine.equity:.2f}")
                    print(f"  收益率: {(self.engine.equity / self.engine.initial_capital - self.engine.initial_capital * self.engine.position_ratio / self.engine.initial_capital) * 100:.2f}%")
                    
                    # 输出当前价格
                    if len(current_prices) > 0:
                        print(f"\n当前价格:")
                        for symbol, price in current_prices.items():
                            print(f"  {symbol}: {price:.4f}")
                    
                    # 输出持仓信息
                    if self.engine.has_positions():
                        print(f"\n持仓信息:")
                        for pos in self.engine.positions:
                            position_id = pos['position_id']
                            entry_price = pos['entry_price']
                            size = pos['size']
                            entry_time = pos['entry_time']
                            
                            # 计算当前盈亏
                            if len(current_prices) > 0:
                                first_symbol = list(current_prices.keys())[0]
                                current_price = current_prices[first_symbol]
                                pnl = size * (current_price - entry_price)
                                pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                                
                                holding_time = (datetime.now() - entry_time).total_seconds() / 3600
                                direction = "多头" if size > 0 else "空头"
                                
                                print(f"  持仓ID: {position_id}")
                                print(f"    方向: {direction}")
                                print(f"    数量: {abs(size):.6f}")
                                print(f"    开仓价: {entry_price:.4f}")
                                print(f"    当前价: {current_price:.4f}")
                                print(f"    盈亏: {pnl:.2f} ({pnl_pct:+.2f}%)")
                                print(f"    持仓时间: {holding_time:.1f} 小时")
                                print(f"    加仓次数: {pos.get('entry_count', 1)}")
                                
                                # 显示使用的策略
                                strategy_name = "未知"
                                if pos.get('strategy'):
                                    strategy_name = pos['strategy'].name if hasattr(pos['strategy'], 'name') else type(pos['strategy']).__name__
                                print(f"    策略: {strategy_name}")
                            else:
                                print(f"  持仓ID: {position_id}, 数量: {size:.6f}, 开仓价: {entry_price:.4f}")
                    else:
                        print(f"\n持仓信息: 无持仓")
                    
                    # 输出市场状态（如果使用市场状态检测）
                    if self.use_market_regime and self.market_detector.is_trained:
                        if len(current_data) > 0:
                            first_symbol = list(current_data.keys())[0]
                            data = current_data[first_symbol]
                            current_idx = len(data) - 1
                            current_regime, current_confidence = self.get_current_regime(
                                current_data, current_idx
                            )
                            print(f"\n市场状态:")
                            print(f"  状态: {current_regime}")
                            print(f"  置信度: {current_confidence:.2%}")
                            
                            # 显示当前使用的策略
                            if self.strategy:
                                strategy_name = self.strategy.name if hasattr(self.strategy, 'name') else type(self.strategy).__name__
                                print(f"  当前策略: {strategy_name}")
                            else:
                                print(f"  当前策略: 无可用策略")
                    
                    # 输出交易统计
                    if len(self.engine.trades) > 0:
                        print(f"\n交易统计:")
                        open_trades = [t for t in self.engine.trades if t.get('action') == 'OPEN']
                        close_trades = [t for t in self.engine.trades if t.get('action') == 'CLOSE']
                        print(f"  总交易次数: {len(self.engine.trades)}")
                        print(f"  开仓次数: {len(open_trades)}")
                        print(f"  平仓次数: {len(close_trades)}")
                        
                        if close_trades:
                            total_pnl = sum([t.get('pnl', 0) for t in close_trades])
                            print(f"  总盈亏: {total_pnl:.2f}")
                    
                    print(f"{'=' * 60}\n")
                    
                    last_status_output = current_time

                # 检查是否有新的K线收盘
                latest_timestamp = self.data_manager.get_latest_closed_kline_timestamp()
                has_new_kline = (latest_timestamp is not None and
                               latest_timestamp != last_processed_timestamp)

                if not has_new_kline:
                    time.sleep(10)
                    continue

                # 获取最新已收盘K线的数据
                kline_data = {}
                kline_prices = {}
                for symbol in self.data_manager.symbols:
                    if symbol in current_data and len(current_data[symbol]) > 0:
                        kline_data[symbol] = current_data[symbol]
                        kline_prices[symbol] = current_data[symbol]['close'].iloc[-1]

                # 合并为单一DataFrame（简化处理，假设只有一个交易对）
                if len(kline_data) > 0:
                    first_symbol = list(kline_data.keys())[0]
                    data = kline_data[first_symbol]
                    current_price = kline_prices[first_symbol]
                    current_idx = len(data) - 1

                    # 更新权益
                    self.engine.update_equity(current_price)
                    
                    # 更新资金曲线
                    self.update_capital_curve()

                    # 获取当前市场状态
                    if self.use_market_regime and self.market_detector.is_trained:
                        current_regime, current_confidence = self.get_current_regime(
                            kline_data, current_idx
                        )
                        # 选择策略
                        self.strategy = self.select_strategy_by_regime(
                            current_regime, current_confidence
                        )
                    else:
                        current_regime = 'ranging'
                        current_confidence = 1.0

                    # 如果有持仓，使用开仓时的策略管理
                    if self.engine.has_positions():
                        positions_to_check = list(self.engine.positions)
                        for pos in positions_to_check:
                            position_id = pos['position_id']
                            pos_strategy = pos.get('strategy') or self.entry_strategy or self.strategy

                            if pos_strategy is None:
                                # 强制平仓
                                pnl = self.engine.close_position(
                                    current_price, datetime.now(), "策略不可用，强制平仓", position_id
                                )
                                continue

                            # 检查止损
                            stop_signal = pos_strategy.check_stop_loss(
                                data, current_idx, pos['size'], pos['entry_price']
                            )
                            if stop_signal:
                                pnl = self.engine.close_position(
                                    stop_signal['price'], datetime.now(),
                                    stop_signal['reason'], position_id
                                )
                                if pnl is not None:
                                    pos_strategy.update_trade_result(pnl)
                                continue

                            # 检查信号
                            signal = pos_strategy.generate_signals(data, current_idx, pos['size'])
                            if signal['signal'] in ['close_long', 'close_short']:
                                pnl = self.engine.close_position(
                                    current_price, datetime.now(), signal['reason'], position_id
                                )
                                if pnl is not None:
                                    pos_strategy.update_trade_result(pnl)
                                continue

                            # 检查加仓
                            if pos['entry_count'] < max_entries:
                                add_signal = pos_strategy.check_add_position(
                                    data, current_idx, pos['size'], pos['entry_price']
                                )
                                if add_signal:
                                    add_size = pos_strategy.get_position_size(
                                        self.engine.balance, add_signal['price'], self.engine.leverage
                                    )
                                    self.engine.add_position(
                                        add_signal['signal'], add_signal['price'], add_size,
                                        datetime.now(), add_signal['reason'], position_id
                                    )

                    # 如果没有可用策略，跳过开仓
                    if self.strategy is None:
                        last_processed_timestamp = latest_timestamp
                        continue

                    # 检查入场信号
                    default_pos = self.engine.get_position_by_id('default')
                    is_grid_strategy = (self.strategy and hasattr(self.strategy, '__class__') and
                                      self.strategy.__class__.__name__ == 'GridStrategy')
                    can_open_new = default_pos is None or is_grid_strategy

                    if can_open_new:
                        signal = self.strategy.generate_signals(
                            data, current_idx, self.engine.get_total_position_size()
                        )

                        if signal['signal'] == 'long':
                            position_size = self.strategy.get_position_size(
                                self.engine.balance, signal['price'], self.engine.leverage
                            )
                            position_id = signal.get('position_id', 'default')
                            position = self.engine.open_position(
                                'long', signal['price'], position_size,
                                datetime.now(), signal['reason'], position_id, self.strategy
                            )
                            if position and (position_id == 'default' or position_id is None):
                                self.entry_strategy = self.strategy

                        elif signal['signal'] == 'short':
                            position_size = self.strategy.get_position_size(
                                self.engine.balance, signal['price'], self.engine.leverage
                            )
                            position_id = signal.get('position_id', 'default')
                            position = self.engine.open_position(
                                'short', signal['price'], position_size,
                                datetime.now(), signal['reason'], position_id, self.strategy
                            )
                            if position and (position_id == 'default' or position_id is None):
                                self.entry_strategy = self.strategy

                last_processed_timestamp = latest_timestamp
                time.sleep(10)  # 每10秒检查一次

            except KeyboardInterrupt:
                print("\n收到停止信号，正在停止交易...")
                self.running = False
                break
            except Exception as e:
                print(f"交易循环异常: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

        # 平仓所有持仓
        if self.engine.has_positions():
            print("\n平仓所有持仓...")
            positions_to_close = list(self.engine.positions)
            for pos in positions_to_close:
                current_price = self.data_manager.get_current_prices().get(self.symbol, 0)
                if current_price > 0:
                    self.engine.close_position(
                        current_price, datetime.now(), "系统停止，强制平仓", pos['position_id']
                    )

        print("\n实盘交易已停止")

    def update_capital_curve(self):
        """更新资金曲线"""
        self.capital_curve.append({
            'timestamp': datetime.now(),
            'capital': self.engine.equity,
            'balance': self.engine.balance,
            'position_value': self.engine.position_value,
            'positions_count': len(self.engine.positions)
        })

    def get_trading_status(self):
        """获取交易状态"""
        return {
            'running': self.running,
            'current_capital': self.engine.equity,
            'initial_capital': self.engine.initial_capital,
            'balance': self.engine.balance,
            'position_value': self.engine.position_value,
            'total_return': (self.engine.equity / self.engine.initial_capital - 1) * 100 if self.engine.initial_capital > 0 else 0,
            'positions_count': len(self.engine.positions),
            'total_trades': len(self.engine.trades),
            'positions': {pos['position_id']: {
                'position_id': pos['position_id'],
                'size': pos['size'],
                'entry_price': pos['entry_price'],
                'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                'entry_count': pos.get('entry_count', 1)
            } for pos in self.engine.positions},
            'recent_trades': self.engine.trades[-5:] if self.engine.trades else []
        }


# ==================== 辅助函数 ====================

def load_strategy_from_module(module_name: str, strategy_class_name: str, params: Dict = None) -> BaseStrategy:
    """
    从strategies文件夹动态加载策略
    
    Args:
        module_name: 模块名称（不含.py扩展名，如 'turtle_strategy'）
        strategy_class_name: 策略类名称（如 'TurtleStrategy'）
        params: 策略参数字典
        
    Returns:
        策略实例
    """
    try:
        # 动态导入模块
        module = importlib.import_module(f'strategies.{module_name}')
        
        # 获取策略类
        strategy_class = getattr(module, strategy_class_name)
        
        # 检查是否是BaseStrategy的子类
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class_name} 不是BaseStrategy的子类")
        
        # 创建策略实例
        if params is None:
            params = {}
        strategy = strategy_class(params)
        
        return strategy
        
    except ImportError as e:
        raise ImportError(f"无法导入策略模块 {module_name}: {str(e)}")
    except AttributeError as e:
        raise AttributeError(f"策略类 {strategy_class_name} 不存在: {str(e)}")
    except Exception as e:
        raise Exception(f"加载策略失败: {str(e)}")


def list_available_strategies() -> List[str]:
    """
    列出strategies文件夹中所有可用的策略
    
    Returns:
        策略类名称列表（格式: 'module_name.ClassName'）
    """
    strategies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')
    strategies = []
    
    if not os.path.exists(strategies_dir):
        print(f"警告: strategies目录不存在: {strategies_dir}")
        return strategies
    
    for filename in os.listdir(strategies_dir):
        if filename.endswith('_strategy.py') and not filename.endswith('_strategy_bt.py'):
            module_name = filename[:-3]  # 移除.py
            try:
                module = importlib.import_module(f'strategies.{module_name}')
                # 查找所有BaseStrategy的子类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy and 
                        obj.__module__ == module.__name__):
                        strategies.append(f"{module_name}.{name}")
            except Exception as e:
                # 静默忽略导入错误
                pass
    
    return strategies


def get_strategy_params(strategy_class_name: str) -> Dict:
    """
    获取策略参数（用户输入）
    
    Args:
        strategy_class_name: 策略类名称
        
    Returns:
        参数字典
    """
    print(f"\n配置 {strategy_class_name} 参数:")
    print("  直接回车使用默认值")
    
    params = {}
    
    # 根据策略类型设置默认参数
    if 'Turtle' in strategy_class_name:
        params = {
            'n_entries': 3,
            'atr_length': 20,
            'bo_length': 20,
            'fs_length': 55,
            'te_length': 10,
            'use_filter': False,
            'mas': 10,
            'mal': 20
        }
        # 允许用户修改参数
        n_entries_input = input("加仓次数 (默认3): ").strip()
        if n_entries_input:
            try:
                params['n_entries'] = int(n_entries_input)
            except:
                pass
    elif 'FinalMultiplePeriod' in strategy_class_name:
        # FinalMultiplePeriodStrategy的默认参数
        params = {
            'ema_lens': [5, 10, 20, 30],
            'ma_len_daily': 25,
            'tp_pct': 3.0,
            'vol_factor': 1.2,
            'watch_bars': 5
        }
    else:
        # 其他策略使用空参数字典（策略会使用自己的默认值）
        params = {}
    
    return params


def select_strategies() -> Dict[str, BaseStrategy]:
    """
    让用户选择策略（趋势类和/或震荡类）
    
    Returns:
        策略字典 {'trending': 趋势策略, 'ranging': 震荡策略} 或 None
    """
    print("\n" + "=" * 60)
    print("选择交易策略")
    print("=" * 60)
    
    # 列出可用策略
    available_strategies = list_available_strategies()
    if not available_strategies:
        print("未找到可用策略")
        return None
    
    print("\n可用策略:")
    for i, strategy_name in enumerate(available_strategies, 1):
        print(f"  {i}. {strategy_name}")
    
    strategies = {}
    
    # 选择趋势类策略
    print("\n选择趋势类策略（用于趋势市场）:")
    print("  输入策略编号，或直接回车跳过（不选择趋势策略）")
    trend_choice = input("请选择: ").strip()
    
    if trend_choice:
        try:
            trend_idx = int(trend_choice) - 1
            if 0 <= trend_idx < len(available_strategies):
                strategy_name = available_strategies[trend_idx]
                module_name, class_name = strategy_name.split('.')
                
                # 获取策略参数
                params = get_strategy_params(class_name)
                
                strategies['trending'] = load_strategy_from_module(module_name, class_name, params)
                print(f" 已选择趋势策略: {strategy_name}")
            else:
                print("无效选择")
        except Exception as e:
            print(f"加载趋势策略失败: {str(e)}")
    
    # 选择震荡类策略
    print("\n选择震荡类策略（用于震荡市场）:")
    print("  输入策略编号，或直接回车跳过（不选择震荡策略）")
    range_choice = input("请选择: ").strip()
    
    if range_choice:
        try:
            range_idx = int(range_choice) - 1
            if 0 <= range_idx < len(available_strategies):
                strategy_name = available_strategies[range_idx]
                module_name, class_name = strategy_name.split('.')
                
                # 获取策略参数
                params = get_strategy_params(class_name)
                
                strategies['ranging'] = load_strategy_from_module(module_name, class_name, params)
                print(f" 已选择震荡策略: {strategy_name}")
            else:
                print("无效选择")
        except Exception as e:
            print(f"加载震荡策略失败: {str(e)}")
    
    if not strategies:
        print("\n警告: 未选择任何策略，将退出")
        return None
    
    return strategies


def select_regime_detection_method() -> str:
    """选择市场状态检测方法"""
    print("\n选择市场状态检测方法:")
    print("  1. per_kline - 每根K线都预测（原始方法）")
    print("  2. future_looking - 前瞻性预测（预测未来N根K线）")
    print("  3. segmentation - 在线分段（将数据分成多个时间段）")

    choice = input("请选择 (1-3，默认1): ").strip()
    if choice == '2':
        return 'future_looking'
    elif choice == '3':
        return 'segmentation'
    else:
        return 'per_kline'


# ==================== Flask Web服务器 ====================

if FLASK_AVAILABLE:
    class LiveTradingServer:
        """实盘交易Web服务器（简化版，仅监控功能）"""

        def __init__(self, trading_system):
            self.trading_system = trading_system
            self.app = Flask(__name__)
            self.app.secret_key = 'your-secret-key-change-in-production'  # 生产环境请更改
            CORS(self.app)
            self.setup_routes()

        def setup_routes(self):
            """设置路由"""

            @self.app.route('/')
            def index():
                """首页 - 重定向到监控页面"""
                return redirect(url_for('monitor'))

            @self.app.route('/monitor')
            def monitor():
                """实时监控页面"""
                return render_template('monitor.html') if hasattr(self.app, 'send_static_file') else "监控页面（需要templates/monitor.html）"

            @self.app.route('/api/status')
            def get_status():
                """获取交易状态"""
                return jsonify(self.trading_system.get_trading_status())

            @self.app.route('/api/positions')
            def get_positions():
                """获取当前持仓"""
                positions_dict = {}
                for pos in self.trading_system.engine.positions:
                    positions_dict[pos['position_id']] = {
                        'position_id': pos['position_id'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                        'entry_count': pos.get('entry_count', 1),
                        'strategy': pos.get('strategy').name if pos.get('strategy') and hasattr(pos.get('strategy'), 'name') else 'Unknown'
                    }
                return jsonify(positions_dict)

            @self.app.route('/api/trades')
            def get_trades():
                """获取交易记录"""
                # 转换datetime对象为字符串
                trades_list = []
                for trade in self.trading_system.engine.trades:
                    trade_dict = trade.copy()
                    if 'timestamp' in trade_dict and isinstance(trade_dict['timestamp'], datetime):
                        trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                    trades_list.append(trade_dict)
                return jsonify(trades_list)

            @self.app.route('/api/capital_curve')
            def get_capital_curve():
                """获取资金曲线"""
                # 转换datetime对象为字符串
                curve_list = []
                for point in self.trading_system.capital_curve:
                    point_dict = point.copy()
                    if 'timestamp' in point_dict and isinstance(point_dict['timestamp'], datetime):
                        point_dict['timestamp'] = point_dict['timestamp'].isoformat()
                    curve_list.append(point_dict)
                return jsonify(curve_list)

            @self.app.route('/api/monitor/data')
            def monitor_data():
                """获取监控数据API"""
                # 从币安API实时获取账户信息
                account_info = None
                try:
                    account_info = self.trading_system.binance_api.get_account_info()
                except Exception as e:
                    print(f"获取账户信息失败: {str(e)}")

                # 解析账户信息
                balance = 0.0
                equity = 0.0
                available = 0.0

                if account_info:
                    balance = float(account_info.get('totalWalletBalance', 0))
                    available = float(account_info.get('availableBalance', 0))
                    unrealized_profit = float(account_info.get('totalUnrealizedProfit', 0))
                    equity = balance + unrealized_profit

                status = self.trading_system.get_trading_status()
                return jsonify({
                    'status': 'running' if self.trading_system.running else 'stopped',
                    'symbol': self.trading_system.symbol,
                    'positions': {pos['position_id']: {
                        'position_id': pos['position_id'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time']),
                        'entry_count': pos.get('entry_count', 1)
                    } for pos in self.trading_system.engine.positions},
                    'account': {
                        'balance': balance if balance > 0 else self.trading_system.engine.balance,
                        'equity': equity if equity > 0 else self.trading_system.engine.equity,
                        'available': available if available > 0 else self.trading_system.engine.balance
                    },
                    'trading_status': status
                })

            @self.app.route('/api/start_trading', methods=['POST'])
            def start_trading():
                """开始交易"""
                if not self.trading_system.running:
                    self.trading_system.running = True
                    return jsonify({'status': 'success', 'message': '交易已开始'})
                else:
                    return jsonify({'status': 'error', 'message': '交易已在运行中'})

            @self.app.route('/api/stop_trading', methods=['POST'])
            def stop_trading():
                """停止交易"""
                if self.trading_system.running:
                    self.trading_system.running = False
                    return jsonify({'status': 'success', 'message': '交易已停止'})
                else:
                    return jsonify({'status': 'error', 'message': '交易未在运行'})

        def run(self, host='0.0.0.0', port=5000, debug=False):
            """运行服务器"""
            print(f"启动实盘交易Web服务器: http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
else:
    # Flask不可用时的占位类
    class LiveTradingServer:
        def __init__(self, trading_system):
            print("警告: Flask未安装，Web服务器不可用")
        
        def run(self, host='0.0.0.0', port=5000, debug=False):
            print("警告: Flask未安装，无法启动Web服务器")


# ==================== 主函数 ====================

def main():
    """主函数：实盘交易系统"""
    print("=" * 80)
    print("基于机器学习的市场状态检测实盘交易系统")
    print("=" * 80)

    # 1. 配置API密钥
    print("\n1. 配置币安API")
    api_key = input("请输入API密钥 (或回车使用默认): ").strip()
    if not api_key:
        api_key = API_KEY
    secret_key = input("请输入Secret密钥 (或回车使用默认): ").strip()
    if not secret_key:
        secret_key = SECRET_KEY

    base_url = input("请输入API基础URL (测试网/实盘，默认测试网): ").strip()
    if not base_url:
        base_url = BASE_URL

    binance_api = BinanceAPI(api_key=api_key, secret_key=secret_key, base_url=base_url)

    # 2. 选择是否使用市场状态检测
    print("\n2. 选择是否使用机器学习市场状态检测？")
    print("  y: 使用（根据市场状态选择策略）")
    print("  n: 不使用（使用单一策略）")
    use_ml_input = input("请选择 (y/n, 默认y): ").strip().lower()
    use_ml = use_ml_input != 'n'

    # 3. 选择策略
    if use_ml:
        strategies = select_strategies()
        if not strategies:
            print("未选择策略，退出程序")
            return

        print(f"\n策略配置:")
        if 'trending' in strategies:
            print(f"  趋势策略: {strategies['trending'].name}")
        if 'ranging' in strategies:
            print(f"  震荡策略: {strategies['ranging'].name}")
    else:
        # 单策略模式
        available_strategies = list_available_strategies()
        if not available_strategies:
            print("未找到可用策略")
            return

        print("\n可用策略:")
        for i, strategy_name in enumerate(available_strategies, 1):
            print(f"  {i}. {strategy_name}")

        choice = input(f"请选择策略 (1-{len(available_strategies)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_strategies):
                strategy_name = available_strategies[idx]
                module_name, class_name = strategy_name.split('.')
                
                # 获取策略参数
                params = get_strategy_params(class_name)
                
                strategy = load_strategy_from_module(module_name, class_name, params)
                strategies = None
                print(f" 已选择策略: {strategy_name}")
            else:
                print("无效选择")
                return
        except Exception as e:
            print(f"加载策略失败: {str(e)}")
            return

    # 4. 配置交易参数
    print("\n3. 配置交易参数")
    initial_capital = input("初始资金 (默认10000): ").strip()
    initial_capital = float(initial_capital) if initial_capital else 10000

    leverage = input("杠杆倍数 (默认1.0): ").strip()
    leverage = float(leverage) if leverage else 1.0

    position_ratio = input("仓位比例 (默认0.5，即50%): ").strip()
    position_ratio = float(position_ratio) if position_ratio else 0.5

    # 5. 选择交易对
    print("\n4. 选择交易对")
    symbol = input("请输入交易对 (如BTCUSDT): ").strip().upper()
    if not symbol:
        symbol = 'BTCUSDT'

    # 6. 选择K线周期
    print("\n5. 选择K线周期")
    print("  1. 1m (1分钟)")
    print("  2. 5m (5分钟)")
    print("  3. 15m (15分钟)")
    print("  4. 30m (30分钟)")
    print("  5. 1h (1小时)")
    print("  6. 4h (4小时)")
    print("  7. 1d (1天)")

    interval_choice = input("请选择 (1-6，默认4): ").strip()
    interval_map = {'1': '1m', '2': '5m', '3': '15m', '4': '30m','5': '1h', '6': '4h', '7': '1d'}
    interval = interval_map.get(interval_choice, '1h')

    # 7. 初始化数据管理器
    print("\n6. 初始化数据管理器")
    data_manager = RealTimeDataManager(binance_api)

    # 8. 收集预热数据
    print("\n7. 收集预热数据")
    warmup_period = 1000
    warmup_data = data_manager.collect_warmup_data([symbol], interval=interval, warmup_period=warmup_period)
    data_manager.data_cache = warmup_data

    # 9. 训练市场状态检测器（如果使用）
    market_detector = None
    regime_detection_method = 'per_kline'

    if use_ml:
        print("\n8. 训练市场状态检测器")
        if len(warmup_data) > 0:
            first_symbol = list(warmup_data.keys())[0]
            training_data = warmup_data[first_symbol]

            if ML_DETECTOR_AVAILABLE:
                model_type = input("模型类型 (xgboost/random_forest/lstm/cnn，默认xgboost): ").strip()
                model_type = model_type if model_type else 'xgboost'

                market_detector = MarketRegimeMLDetector(
                    model_type=model_type,
                    train_ratio=0.7,
                    label_method='adx_slope',  # 实盘使用无滞后方法
                    smooth_window=5,
                    confidence_threshold=0.6
                )

                print("正在训练模型...")
                market_detector.train(training_data)
                print("模型训练完成")

                regime_detection_method = select_regime_detection_method()
            else:
                print("警告: 市场状态检测器不可用，将使用单一策略")
                use_ml = False

    # 10. 初始化交易系统
    print("\n9. 初始化交易系统")
    if use_ml and strategies:
        trading_system = LiveTradingSystem(
            binance_api=binance_api,
            data_manager=data_manager,
            symbol=symbol,
            strategies=strategies,
            initial_capital=initial_capital,
            leverage=leverage,
            position_ratio=position_ratio,
            market_detector=market_detector,
            regime_detection_method=regime_detection_method
        )
    else:
        trading_system = LiveTradingSystem(
            binance_api=binance_api,
            data_manager=data_manager,
            symbol=symbol,
            strategy=strategy,
            initial_capital=initial_capital,
            leverage=leverage,
            position_ratio=position_ratio
        )

    # 初始化策略
    trading_system.initialize_strategies(warmup_data)

    # 11. 启动实时数据收集
    print("\n10. 启动实时数据收集")
    data_manager.start_data_collection([symbol], interval=interval)

    # 12. 启动Web服务器
    print("\n11. 启动Web服务器")
    server = LiveTradingServer(trading_system)

    # 13. 启动交易循环
    print("\n12. 启动交易循环")
    trading_system.running = True

    def trading_loop():
        """交易循环（在后台线程中运行）"""
        trading_system.start_trading(max_entries=3)

    # 启动交易循环线程
    trading_thread = threading.Thread(target=trading_loop)
    trading_thread.daemon = True
    trading_thread.start()

    try:
        # 启动Web服务器（阻塞主线程）
        server.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n停止实盘交易...")
        trading_system.running = False
        data_manager.stop_data_collection()
        print("实盘交易已停止")


if __name__ == "__main__":
    main()

