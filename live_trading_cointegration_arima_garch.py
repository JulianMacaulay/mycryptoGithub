#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘协整交易系统（支持ARIMA-GARCH）
基于 cointegration_test_windows_optimization_arima_garch.py 的核心交易逻辑
结合 test_live_trading.py 的实时交易基础设施

功能：
1. 实时数据获取和管理
2. 协整交易策略执行（支持传统方法和ARIMA-GARCH模型）
3. 实时交易执行和监控
4. Web监控界面
5. 预热数据收集
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
from flask import Flask, jsonify, request
from typing import Dict, List, Tuple, Any, Optional
from decimal import Decimal, ROUND_HALF_UP

warnings.filterwarnings('ignore')

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
    from strategies import TraditionalZScoreStrategy, ArimaGarchZScoreStrategy, BaseZScoreStrategy
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    print("警告: 策略模块导入失败，将使用内置方法")

# ==================== 币安API配置 ====================

# API配置（请修改为您的实际API密钥）
API_KEY = "SdTSZxmdf61CFsze3udgLRWq0aCaVyyFjsrYKMUOWIfMkm7q3sGRkzSk6QSbM5Qk"
SECRET_KEY = "9HZ04wgrKTy5kDPF5Kman4WSmS9D7YlTscPA7FtX2YLK7vTbpORFNB2jTABQY6HY"
BASE_URL = "https://testnet.binancefuture.com"  # 测试网，实盘请改为 "https://fapi.binance.com"

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
    
    def get_exchange_info(self):
        """获取交易对信息（使用主网API）"""
        try:
            # 使用币安主网API获取精度信息，无需认证
            url = "https://demo-fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"获取交易对信息失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"获取交易对信息异常: {str(e)}")
            return None
    
    def get_symbol_precision(self, symbol):
        """获取交易对的精度信息"""
        try:
            exchange_info = self.get_exchange_info()
            if exchange_info:
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        for filter_info in symbol_info['filters']:
                            if filter_info['filterType'] == 'LOT_SIZE':
                                step_size = float(filter_info['stepSize'])
                                return step_size
            return 0.001  # 默认精度
        except Exception as e:
            print(f"获取 {symbol} 精度信息异常: {str(e)}")
            return 0.001
    
    def get_order_status(self, order_id, symbol):
        """查询订单状态"""
        try:
            url = f"{self.base_url}/fapi/v1/order"
            
            # 构建参数
            params = {
                'symbol': symbol,
                'orderId': order_id,
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
                print(f"查询订单状态失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"查询订单状态异常: {str(e)}")
            return None

# ==================== 实时数据管理 ====================

class RealTimeDataManager:
    """实时数据管理器"""
    
    def __init__(self, binance_api):
        self.binance_api = binance_api
        self.data_cache = {}
        self.running = False
        self.update_thread = None
    
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
                    klines = self.binance_api.get_klines(symbol, self.interval, 100)
                    if klines:
                        df = pd.DataFrame(klines)
                        df.set_index('timestamp', inplace=True)
                        self.data_cache[symbol] = df['close']
                
                # 根据interval确定更新频率
                if self.interval == '1m':
                    time.sleep(60)
                elif self.interval == '5m':
                    time.sleep(300)
                elif self.interval == '1h':
                    time.sleep(3600)
                else:
                    time.sleep(60)
            except Exception as e:
                print(f"数据更新异常: {str(e)}")
                time.sleep(10)
    
    def get_current_data(self):
        """获取当前数据"""
        return self.data_cache.copy()
    
    def get_current_prices(self):
        """获取当前价格"""
        prices = {}
        for symbol in self.symbols:
            price = self.binance_api.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def collect_warmup_data(self, symbols, interval='1h', warmup_period=70):
        """
        收集预热数据
        
        Args:
            symbols: 交易对列表
            interval: K线间隔
            warmup_period: 预热期长度（数据点数量）
        
        Returns:
            dict: 预热数据字典
        """
        print(f"\n开始收集预热数据（需要 {warmup_period} 个数据点）...")
        warmup_data = {}
        
        for symbol in symbols:
            print(f"  收集 {symbol} 的预热数据...")
            klines = self.binance_api.get_klines(symbol, interval, warmup_period)
            if klines:
                df = pd.DataFrame(klines)
                df.set_index('timestamp', inplace=True)
                warmup_data[symbol] = df['close']
                print(f"    ✓ {symbol}: {len(warmup_data[symbol])} 个数据点")
            else:
                print(f"    ✗ {symbol}: 数据收集失败")
        
        # 检查数据是否足够
        min_length = min([len(data) for data in warmup_data.values()]) if warmup_data else 0
        if min_length < warmup_period:
            print(f"警告: 预热数据不足，只收集到 {min_length} 个数据点（需要 {warmup_period} 个）")
        else:
            print(f"✓ 预热数据收集完成，每个交易对 {min_length} 个数据点")
        
        return warmup_data

# ==================== 高级交易流程代码 ====================

class AdvancedCointegrationTrading:
    """高级协整交易策略类（支持策略模式）"""
    
    def __init__(self, binance_api, lookback_period=60, z_threshold=2.0, z_exit_threshold=0.5,
                 take_profit_pct=0.15, stop_loss_pct=0.08, max_holding_hours=168,
                 position_ratio=0.5, leverage=5, trading_fee_rate=0.000275,
                 z_score_strategy=None, use_arima_garch=False, arima_order=(1, 0, 1), garch_order=(1, 1)):
        """
        初始化高级协整交易策略
        
        Args:
            binance_api: 币安API客户端
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
        """
        self.binance_api = binance_api
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
        self.capital_curve = []  # 资金曲线
        self.running = False
        
        # 设置Z-score策略
        if z_score_strategy is not None:
            self.z_score_strategy = z_score_strategy
            self.use_arima_garch = isinstance(z_score_strategy, ArimaGarchZScoreStrategy) if STRATEGIES_AVAILABLE else False
        elif use_arima_garch and STRATEGIES_AVAILABLE and ARIMA_AVAILABLE and GARCH_AVAILABLE:
            try:
                self.z_score_strategy = ArimaGarchZScoreStrategy(arima_order=arima_order, garch_order=garch_order)
                self.use_arima_garch = True
            except Exception as e:
                print(f"警告: ARIMA-GARCH策略初始化失败: {str(e)}，使用传统策略")
                self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None
                self.use_arima_garch = False
        else:
            self.z_score_strategy = TraditionalZScoreStrategy() if STRATEGIES_AVAILABLE else None
            self.use_arima_garch = False
        
        # 向后兼容：保留旧属性
        self.arima_order = arima_order
        self.garch_order = garch_order
        
        # 初始化账户信息
        self._initialize_account()
    
    def _initialize_account(self):
        """初始化账户信息"""
        try:
            print("正在获取账户信息...")
            account_info = self.binance_api.get_account_info()
            
            if account_info:
                self.initial_capital = float(account_info.get('totalWalletBalance', 0))
                self.current_capital = self.initial_capital
                print(f"✓ 账户初始化成功")
                print(f"  总资产: {self.initial_capital:.2f} USDT")
                print(f"  可用余额: {float(account_info.get('availableBalance', 0)):.2f} USDT")
            else:
                print("✗ 无法获取账户信息，使用默认值")
                self.initial_capital = 10000
                self.current_capital = self.initial_capital
        except Exception as e:
            print(f"账户初始化失败: {str(e)}")
            self.initial_capital = 10000
            self.current_capital = self.initial_capital
    
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
        """
        计算当前Z-score（使用策略对象）
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
        
        Returns:
            float: Z-score值
        """
        # 如果使用了策略对象，调用策略的方法
        if self.z_score_strategy is not None:
            return self.z_score_strategy.calculate_z_score(current_spread, historical_spreads)
        
        # 向后兼容：如果没有策略对象，使用传统方法
        if len(historical_spreads) < 2:
            return 0.0
        
        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)
        
        if spread_std == 0:
            return 0.0
        
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
        执行交易（实盘下单）
        
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
        
        # 获取精度信息
        step_size1 = self.binance_api.get_symbol_precision(symbol1)
        step_size2 = self.binance_api.get_symbol_precision(symbol2)
        
        # 根据stepSize确定小数位数
        def get_decimal_places(step_size):
            if step_size >= 1:
                return 0
            elif step_size >= 0.1:
                return 1
            elif step_size >= 0.01:
                return 2
            elif step_size >= 0.001:
                return 3
            else:
                return 4
        
        # 使用Decimal避免浮点数精度问题
        quantity1_raw = Decimal(str(abs(symbol1_size)))
        quantity2_raw = Decimal(str(abs(symbol2_size)))
        step_size1_decimal = Decimal(str(step_size1))
        step_size2_decimal = Decimal(str(step_size2))
        
        # 计算到stepSize的倍数
        quantity1_multiple = (quantity1_raw / step_size1_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        quantity2_multiple = (quantity2_raw / step_size2_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        # 转换为最终数量
        quantity1 = float(quantity1_multiple * step_size1_decimal)
        quantity2 = float(quantity2_multiple * step_size2_decimal)
        
        # 初始化订单变量
        order1 = None
        order2 = None
        
        if signal['action'] == 'SHORT_LONG':
            # 做空价差：做空symbol1，做多symbol2
            print(f"  下单计划: {symbol1} SELL {quantity1}, {symbol2} BUY {quantity2}")
            order1 = self.binance_api.place_order(symbol1, 'SELL', quantity1)
            order2 = self.binance_api.place_order(symbol2, 'BUY', quantity2)
            
        elif signal['action'] == 'LONG_SHORT':
            # 做多价差：做多symbol1，做空symbol2
            print(f"  下单计划: {symbol1} BUY {quantity1}, {symbol2} SELL {quantity2}")
            order1 = self.binance_api.place_order(symbol1, 'BUY', quantity1)
            order2 = self.binance_api.place_order(symbol2, 'SELL', quantity2)
        
        # 检查下单结果
        if order1 and order2 and order1.get('orderId') and order2.get('orderId'):
            # 等待订单成交
            success, final_status1, final_status2 = self.wait_for_orders_completion(
                order1, order2, symbol1, symbol2
            )
            
            if success:
                # 创建持仓记录
                position = {
                    'pair': f"{symbol1}_{symbol2}",
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'symbol1_size': -quantity1 if signal['action'] == 'SHORT_LONG' else quantity1,
                    'symbol2_size': quantity2 if signal['action'] == 'SHORT_LONG' else -quantity2,
                    'entry_prices': {symbol1: price1, symbol2: price2},
                    'entry_spread': current_spread,
                    'hedge_ratio': hedge_ratio,
                    'entry_time': timestamp,
                    'signal': signal,
                    'capital_used': total_capital_used,
                    'orders': [order1, order2]
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
                
                print(f"实盘开仓: {pair_info['pair_name']}")
                print(f"   信号: {signal['description']}")
                print(f"   价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
                print(f"   价差: {current_spread:.6f}")
                print(f"   仓位: {symbol1}={position['symbol1_size']:.6f}, {symbol2}={position['symbol2_size']:.6f}")
                
                return position
            else:
                print(f"订单未完全成交，不保存持仓")
                return None
        else:
            print(f"下单失败: {symbol1} 或 {symbol2} 订单未成功提交")
            return None
    
    def check_exit_conditions(self, pair_info, current_prices, current_z_score, timestamp, current_spread):
        """检查平仓条件（包含止盈止损）"""
        pair_name = pair_info['pair_name']
        if pair_name not in self.positions:
            return False, ""
        
        position = self.positions[pair_name]
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]
        
        # 计算实际盈亏
        entry_price1 = position['entry_prices'][symbol1]
        entry_price2 = position['entry_prices'][symbol2]
        
        if position['signal']['action'] == 'SHORT_LONG':
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        else:  # LONG_SHORT
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        
        # 计算投入资金
        entry_value = abs(position['symbol1_size'] * entry_price1) + \
                      abs(position['symbol2_size'] * entry_price2)
        
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
        
        # 计算最终盈亏
        entry_price1 = position['entry_prices'][symbol1]
        entry_price2 = position['entry_prices'][symbol2]
        
        if position['signal']['action'] == 'SHORT_LONG':
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        else:  # LONG_SHORT
            pnl_symbol1 = position['symbol1_size'] * (price1 - entry_price1)
            pnl_symbol2 = position['symbol2_size'] * (price2 - entry_price2)
            total_pnl = pnl_symbol1 + pnl_symbol2
        
        # 执行平仓订单
        close_order1 = self.binance_api.place_order(
            symbol1, 
            'BUY' if position['symbol1_size'] < 0 else 'SELL', 
            abs(position['symbol1_size'])
        )
        close_order2 = self.binance_api.place_order(
            symbol2, 
            'BUY' if position['symbol2_size'] < 0 else 'SELL', 
            abs(position['symbol2_size'])
        )
        
        if close_order1 and close_order2:
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
            
            print(f"实盘平仓: {pair_name}")
            print(f"   平仓原因: {reason}")
            print(f"   盈亏: {total_pnl:.2f}")
            print(f"   持仓时间: {trade['holding_hours']:.1f}小时")
            
            # 移除持仓
            del self.positions[pair_name]
            
            return trade
        
        return None
    
    def wait_for_orders_completion(self, order1, order2, symbol1, symbol2, max_wait=30):
        """等待订单成交"""
        for i in range(max_wait):
            try:
                status1 = self.binance_api.get_order_status(order1['orderId'], symbol1)
                status2 = self.binance_api.get_order_status(order2['orderId'], symbol2)
                
                if status1 and status2:
                    status1_str = status1.get('status', 'UNKNOWN')
                    status2_str = status2.get('status', 'UNKNOWN')
                    
                    if status1_str in ['FILLED', 'PARTIALLY_FILLED'] and \
                       status2_str in ['FILLED', 'PARTIALLY_FILLED']:
                        return True, status1, status2
                    
                    elif status1_str in ['CANCELED', 'REJECTED', 'EXPIRED'] or \
                         status2_str in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        return False, status1, status2
                
                time.sleep(1)
            except Exception as e:
                print(f"查询订单状态异常: {str(e)}")
                time.sleep(1)
        
        return False, None, None
    
    def update_capital_curve(self):
        """更新资金曲线"""
        try:
            account_info = self.binance_api.get_account_info()
            if account_info:
                self.current_capital = float(account_info.get('totalWalletBalance', self.current_capital))
        except Exception as e:
            print(f"更新资金曲线时获取账户信息失败: {str(e)}")
        
        self.capital_curve.append({
            'timestamp': datetime.now(),
            'capital': self.current_capital,
            'positions_count': len(self.positions)
        })
    
    def get_trading_status(self):
        """获取交易状态"""
        return {
            'running': self.running,
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'positions_count': len(self.positions),
            'total_trades': len(self.trades),
            'positions': self.positions,
            'recent_trades': self.trades[-5:] if self.trades else []
        }

# ==================== Flask Web服务器 ====================

class LiveTradingServer:
    """实盘交易Web服务器"""
    
    def __init__(self, trading_strategy):
        self.trading_strategy = trading_strategy
        self.app = Flask(__name__, static_folder='templates', static_url_path='')
        self.setup_routes()
    
    def setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页面"""
            return self.app.send_static_file('live_trading_monitor.html')
        
        @self.app.route('/api/status')
        def get_status():
            """获取交易状态"""
            return jsonify(self.trading_strategy.get_trading_status())
        
        @self.app.route('/api/positions')
        def get_positions():
            """获取当前持仓"""
            return jsonify(self.trading_strategy.positions)
        
        @self.app.route('/api/trades')
        def get_trades():
            """获取交易记录"""
            return jsonify(self.trading_strategy.trades)
        
        @self.app.route('/api/capital_curve')
        def get_capital_curve():
            """获取资金曲线"""
            return jsonify(self.trading_strategy.capital_curve)
        
        @self.app.route('/api/start_trading', methods=['POST'])
        def start_trading():
            """开始交易"""
            if not self.trading_strategy.running:
                self.trading_strategy.running = True
                return jsonify({'status': 'success', 'message': '交易已开始'})
            else:
                return jsonify({'status': 'error', 'message': '交易已在运行中'})
        
        @self.app.route('/api/stop_trading', methods=['POST'])
        def stop_trading():
            """停止交易"""
            if self.trading_strategy.running:
                self.trading_strategy.running = False
                return jsonify({'status': 'success', 'message': '交易已停止'})
            else:
                return jsonify({'status': 'error', 'message': '交易未在运行'})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """运行服务器"""
        print(f"启动实盘交易Web服务器: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# ==================== 策略选择函数 ====================

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
    
    if ARIMA_AVAILABLE and GARCH_AVAILABLE:
        print("  2. ARIMA-GARCH模型")
        max_choice = 2
    else:
        print("  2. ARIMA-GARCH模型（不可用：缺少必要的库）")
        max_choice = 1
    
    print("  0. 退出程序")
    
    while True:
        try:
            choice = input(f"请选择 (0-{max_choice}): ").strip()
            
            if choice == '0':
                return None
            
            if choice == '1':
                strategy = TraditionalZScoreStrategy()
                print(f"已选择: {strategy.get_strategy_description()}")
                return strategy
            
            if choice == '2' and ARIMA_AVAILABLE and GARCH_AVAILABLE:
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
            
            print(f"无效选择，请输入 0-{max_choice} 之间的数字")
            
        except KeyboardInterrupt:
            print("\n用户取消选择")
            return None
        except Exception as e:
            print(f"选择失败: {str(e)}，请重新选择")

# ==================== 币对配置 ====================

def get_pairs_config():
    """获取币对配置（用户输入）"""
    print("\n" + "=" * 80)
    print("币对配置")
    print("=" * 80)
    
    pairs_config = []
    
    print("请配置要交易的币对（从回测结果中获得的对冲比率等信息）")
    print("可以配置多个币对，输入空行结束配置")
    
    pair_count = 0
    while True:
        pair_count += 1
        print(f"\n--- 配置第 {pair_count} 个币对 ---")
        
        # 输入symbol1
        symbol1 = input("请输入第一个交易对（如: BNBUSDT）: ").strip().upper()
        if not symbol1:
            if pair_count == 1:
                print("至少需要配置一个币对")
                continue
            else:
                break
        
        # 输入symbol2
        symbol2 = input("请输入第二个交易对（如: SOLUSDT）: ").strip().upper()
        if not symbol2:
            print("第二个交易对不能为空，请重新输入")
            pair_count -= 1
            continue
        
        # 选择是否使用差分数据
        print("\n请选择价差计算方式（必须与回测时使用的方法一致）:")
        print("  0. 原始数据（原始价差）")
        print("  1. 一阶差分（一阶差分价差）")
        print("  2. 二阶差分（二阶差分价差）")
        print("\n注意：")
        print("  - 如果选择原始数据，对冲比率应该从原始价格计算得出")
        print("  - 如果选择一阶差分，对冲比率应该从一阶差分价格计算得出")
        print("  - 如果选择二阶差分，对冲比率应该从二阶差分价格计算得出")
        
        while True:
            diff_choice = input("请选择 (0/1/2，默认0): ").strip()
            if not diff_choice:
                diff_order = 0
                break
            elif diff_choice in ['0', '1', '2']:
                diff_order = int(diff_choice)
                break
            else:
                print("无效选择，请输入 0、1 或 2")
        
        # 根据选择的价差类型提示对冲比率来源
        if diff_order == 0:
            hedge_ratio_hint = "原始价格计算的对冲比率"
        elif diff_order == 1:
            hedge_ratio_hint = "一阶差分价格计算的对冲比率"
        else:
            hedge_ratio_hint = "二阶差分价格计算的对冲比率"
        
        # 输入对冲比率
        print(f"\n请输入对冲比率（从回测结果中获得，使用{hedge_ratio_hint}）")
        while True:
            hedge_ratio_input = input(f"对冲比率（如: 1.787595）: ").strip()
            if hedge_ratio_input:
                try:
                    hedge_ratio = float(hedge_ratio_input)
                    break
                except ValueError:
                    print("输入无效，请输入数字")
            else:
                print("对冲比率不能为空")
        
        # 构建币对配置
        pair_name = f"{symbol1}/{symbol2}"
        pair_config = {
            'pair_name': pair_name,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'diff_order': diff_order,
            'cointegration_found': True,
        }
        
        pairs_config.append(pair_config)
        
        print(f"\n✓ 已添加币对: {pair_name}")
        print(f"  对冲比率: {hedge_ratio:.6f}")
        diff_type = '原始价差' if diff_order == 0 else f"{diff_order}阶差分价差"
        print(f"  价差类型: {diff_type}")
        
        # 询问是否继续添加
        continue_input = input("\n是否继续添加币对？(y/n，默认n): ").strip().lower()
        if continue_input != 'y':
            break
    
    if not pairs_config:
        print("未配置任何币对")
        return []
    
    print("\n" + "=" * 80)
    print("已配置的币对:")
    print("=" * 80)
    for i, pair in enumerate(pairs_config, 1):
        diff_type = '原始价差' if pair['diff_order'] == 0 else f"{pair['diff_order']}阶差分价差"
        print(f"{i}. {pair['pair_name']}")
        print(f"   对冲比率: {pair['hedge_ratio']:.6f}")
        print(f"   价差类型: {diff_type}")
    
    return pairs_config

# ==================== 交易参数配置 ====================

def configure_trading_parameters():
    """配置交易参数"""
    print("\n" + "=" * 80)
    print("交易参数配置")
    print("=" * 80)
    
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
    print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1 - default_params['position_ratio']) * 100:.1f}%作为安全垫)")
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
        print(f"  7. 仓位比例: {default_params['position_ratio'] * 100:.1f}% (留{(1 - default_params['position_ratio']) * 100:.1f}%作为安全垫)")
        print(f"  8. 杠杆: {default_params['leverage']}倍")
        print(f"  9. 交易手续费率: {default_params['trading_fee_rate'] * 100:.4f}%")
    
    return default_params

# ==================== 预热数据参数配置 ====================

def configure_warmup_parameters(lookback_period):
    """配置预热数据参数"""
    print("\n" + "=" * 80)
    print("预热数据参数配置")
    print("=" * 80)
    
    # 默认参数
    default_warmup_params = {
        'lookback_period': lookback_period,
        'interval': '1h',  # K线周期
        'warmup_safety_margin': 10  # 安全余量（数据点数量）
    }
    
    print("当前默认参数:")
    print(f"  回看期: {default_warmup_params['lookback_period']} (从交易参数中获取)")
    print(f"  K线周期: {default_warmup_params['interval']}")
    print(f"  安全余量: {default_warmup_params['warmup_safety_margin']} 个数据点")
    print(f"  预热期总长度: {default_warmup_params['lookback_period'] + default_warmup_params['warmup_safety_margin']} 个数据点")
    
    print("\n是否要修改预热数据参数？")
    print("输入 'y' 修改参数，直接回车使用默认参数")
    
    modify_choice = input("请选择: ").strip().lower()
    
    if modify_choice == 'y':
        print("\n请输入新的参数值（直接回车保持默认值）:")
        
        # K线周期
        print("\n可选K线周期:")
        print("  1m - 1分钟")
        print("  5m - 5分钟")
        print("  15m - 15分钟")
        print("  30m - 30分钟")
        print("  1h - 1小时")
        print("  4h - 4小时")
        print("  1d - 1天")
        
        interval_input = input(f"K线周期 (默认: {default_warmup_params['interval']}): ").strip().lower()
        if interval_input:
            valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if interval_input in valid_intervals:
                default_warmup_params['interval'] = interval_input
            else:
                print(f"无效的周期，使用默认值: {default_warmup_params['interval']}")
        
        # 安全余量
        safety_margin_input = input(f"安全余量（数据点数量，默认: {default_warmup_params['warmup_safety_margin']}）: ").strip()
        if safety_margin_input:
            try:
                default_warmup_params['warmup_safety_margin'] = int(safety_margin_input)
                if default_warmup_params['warmup_safety_margin'] < 0:
                    print("安全余量不能为负数，使用默认值")
                    default_warmup_params['warmup_safety_margin'] = 10
            except ValueError:
                print(f"输入无效，使用默认值: {default_warmup_params['warmup_safety_margin']}")
        
        # 更新回看期（从交易参数中获取，但允许用户确认）
        print(f"\n回看期: {default_warmup_params['lookback_period']} (与交易参数一致)")
        lookback_confirm = input("是否要修改回看期？(y/n，默认n): ").strip().lower()
        if lookback_confirm == 'y':
            lookback_input = input(f"回看期 (当前: {default_warmup_params['lookback_period']}): ").strip()
            if lookback_input:
                try:
                    default_warmup_params['lookback_period'] = int(lookback_input)
                except ValueError:
                    print(f"输入无效，保持当前值: {default_warmup_params['lookback_period']}")
        
        print("\n修改后的参数:")
        print(f"  回看期: {default_warmup_params['lookback_period']}")
        print(f"  K线周期: {default_warmup_params['interval']}")
        print(f"  安全余量: {default_warmup_params['warmup_safety_margin']} 个数据点")
        print(f"  预热期总长度: {default_warmup_params['lookback_period'] + default_warmup_params['warmup_safety_margin']} 个数据点")
    
    return default_warmup_params

# ==================== 实盘交易主函数 ====================

def test_live_trading():
    """实盘交易测试主函数"""
    print("=" * 80)
    print("实盘协整交易系统（支持ARIMA-GARCH）")
    print("=" * 80)
    
    # 1. 选择Z-score策略
    z_score_strategy = select_z_score_strategy()
    if z_score_strategy is None:
        print("未选择策略，退出程序")
        return
    
    # 2. 初始化币安API
    print("\n1. 初始化币安API")
    binance_api = BinanceAPI()
    
    # 3. 获取币对配置
    print("\n2. 获取币对配置")
    pairs_config = get_pairs_config()
    
    if not pairs_config:
        print("未配置任何币对，无法进行交易")
        return
    
    # 4. 配置交易参数
    print("\n3. 配置交易参数")
    trading_params = configure_trading_parameters()
    
    # 5. 配置预热数据参数
    print("\n4. 配置预热数据参数")
    warmup_params = configure_warmup_parameters(trading_params['lookback_period'])
    
    # 6. 初始化数据管理器
    print("\n5. 初始化数据管理器")
    symbols = []
    for pair in pairs_config:
        symbols.extend([pair['symbol1'], pair['symbol2']])
    symbols = list(set(symbols))  # 去重
    
    data_manager = RealTimeDataManager(binance_api)
    
    # 7. 收集预热数据
    print("\n6. 收集预热数据")
    warmup_period = warmup_params['lookback_period'] + warmup_params['warmup_safety_margin']
    warmup_data = data_manager.collect_warmup_data(
        symbols, 
        interval=warmup_params['interval'], 
        warmup_period=warmup_period
    )
    
    # 将预热数据加载到数据缓存
    data_manager.data_cache = warmup_data
    
    # 8. 启动实时数据收集
    print("\n7. 启动实时数据收集")
    data_manager.start_data_collection(symbols, interval=warmup_params['interval'])
    
    # 9. 初始化交易策略
    print("\n8. 初始化交易策略")
    trading_strategy = AdvancedCointegrationTrading(
        binance_api=binance_api,
        lookback_period=trading_params['lookback_period'],
        z_threshold=trading_params['z_threshold'],
        z_exit_threshold=trading_params['z_exit_threshold'],
        take_profit_pct=trading_params['take_profit_pct'],
        stop_loss_pct=trading_params['stop_loss_pct'],
        max_holding_hours=trading_params['max_holding_hours'],
        position_ratio=trading_params['position_ratio'],
        leverage=trading_params['leverage'],
        trading_fee_rate=trading_params['trading_fee_rate'],
        z_score_strategy=z_score_strategy
    )
    
    print(f"\n策略参数:")
    print(f"  回看期: {trading_params['lookback_period']}")
    print(f"  Z-score开仓阈值: {trading_params['z_threshold']}")
    print(f"  Z-score平仓阈值: {trading_params['z_exit_threshold']}")
    print(f"  止盈百分比: {trading_params['take_profit_pct'] * 100:.1f}%")
    print(f"  止损百分比: {trading_params['stop_loss_pct'] * 100:.1f}%")
    print(f"  最大持仓时间: {trading_params['max_holding_hours']}小时")
    print(f"  仓位比例: {trading_params['position_ratio'] * 100:.1f}%")
    print(f"  杠杆: {trading_params['leverage']}倍")
    print(f"  Z-score策略: {z_score_strategy.get_strategy_description()}")
    print(f"  K线周期: {warmup_params['interval']}")
    
    # 8. 启动Web服务器
    print("\n7. 启动Web服务器")
    server = LiveTradingServer(trading_strategy)
    
    # 9. 启动交易循环
    print("\n8. 启动交易循环")
    trading_strategy.running = True
    
    def trading_loop():
        """交易循环"""
        last_spread_output = 0
        last_position_sync = 0
        
        while trading_strategy.running:
            try:
                # 获取当前数据
                current_data = data_manager.get_current_data()
                current_prices = data_manager.get_current_prices()
                
                if not current_data or not current_prices:
                    time.sleep(10)
                    continue
                
                # 每10秒输出一次价差数据
                current_time = time.time()
                if current_time - last_spread_output >= 10:
                    print(f"\n{'='*60}")
                    print(f"价差监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")
                    
                    for pair_info in pairs_config:
                        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
                        
                        if symbol1 not in current_prices or symbol2 not in current_prices:
                            continue
                        
                        if symbol1 not in current_data or symbol2 not in current_data:
                            continue
                        
                        # 计算价差和Z-score（根据diff_order选择计算方式）
                        diff_order = pair_info.get('diff_order', 0)
                        data1 = current_data[symbol1]
                        data2 = current_data[symbol2]
                        
                        if diff_order == 0:
                            # 原始价差
                            current_spread = trading_strategy.calculate_current_spread(
                                current_prices[symbol1], 
                                current_prices[symbol2], 
                                pair_info['hedge_ratio']
                            )
                            
                            # 获取历史价差数据
                            historical_spreads = []
                            for i in range(max(0, len(data1) - trading_strategy.lookback_period), len(data1)):
                                if i < len(data2):
                                    hist_spread = trading_strategy.calculate_current_spread(
                                        data1.iloc[i], data2.iloc[i], pair_info['hedge_ratio']
                                    )
                                    historical_spreads.append(hist_spread)
                        elif diff_order == 1:
                            # 一阶差分价差
                            if len(data1) > 1 and len(data2) > 1:
                                # 当前一阶差分：当前价格 - 前一个价格
                                current_diff1 = current_prices[symbol1] - data1.iloc[-1]
                                current_diff2 = current_prices[symbol2] - data2.iloc[-1]
                                # 一阶差分价差 = diff1 - hedge_ratio * diff2
                                # 注意：hedge_ratio应该从一阶差分价格计算得出
                                current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                                
                                # 获取历史一阶差分价差数据
                                historical_spreads = []
                                for i in range(max(1, len(data1) - trading_strategy.lookback_period), len(data1)):
                                    if i < len(data2) and i > 0:
                                        hist_diff1 = data1.iloc[i] - data1.iloc[i-1]
                                        hist_diff2 = data2.iloc[i] - data2.iloc[i-1]
                                        hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                        historical_spreads.append(hist_spread)
                            else:
                                current_spread = 0
                                historical_spreads = []
                        elif diff_order == 2:
                            # 二阶差分价差
                            if len(data1) > 2 and len(data2) > 2:
                                # 当前二阶差分：price[t] - 2*price[t-1] + price[t-2]
                                current_diff1 = (current_prices[symbol1] - 
                                               2 * data1.iloc[-1] + 
                                               data1.iloc[-2])
                                current_diff2 = (current_prices[symbol2] - 
                                               2 * data2.iloc[-1] + 
                                               data2.iloc[-2])
                                # 二阶差分价差 = diff2_1 - hedge_ratio * diff2_2
                                # 注意：hedge_ratio应该从二阶差分价格计算得出
                                current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                                
                                # 获取历史二阶差分价差数据
                                historical_spreads = []
                                for i in range(max(2, len(data1) - trading_strategy.lookback_period), len(data1)):
                                    if i < len(data2) and i > 1:
                                        hist_diff1 = (data1.iloc[i] - 
                                                     2 * data1.iloc[i-1] + 
                                                     data1.iloc[i-2])
                                        hist_diff2 = (data2.iloc[i] - 
                                                     2 * data2.iloc[i-1] + 
                                                     data2.iloc[i-2])
                                        hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                        historical_spreads.append(hist_spread)
                            else:
                                current_spread = 0
                                historical_spreads = []
                        else:
                            # 不支持其他差分阶数，使用原始价差
                            current_spread = trading_strategy.calculate_current_spread(
                                current_prices[symbol1], 
                                current_prices[symbol2], 
                                pair_info['hedge_ratio']
                            )
                            historical_spreads = []
                        
                        current_z_score = trading_strategy.calculate_z_score(current_spread, historical_spreads)
                        
                        # 输出价差信息
                        diff_type = '原始价差' if diff_order == 0 else f"{diff_order}阶差分价差"
                        print(f"币对: {pair_info['pair_name']} ({diff_type})")
                        print(f"  价格: {symbol1}={current_prices[symbol1]:.4f}, {symbol2}={current_prices[symbol2]:.4f}")
                        print(f"  当前价差: {current_spread:.8f}")
                        print(f"  Z-score: {current_z_score:.4f}")
                        print(f"  历史价差数量: {len(historical_spreads)}")
                        
                        # 显示交易信号
                        signal = trading_strategy.generate_trading_signal(current_z_score)
                        signal_color = "🟢" if signal['action'] == 'HOLD' else "🔴"
                        print(f"  交易信号: {signal_color} {signal['description']}")
                        
                        # 显示持仓状态
                        if pair_info['pair_name'] in trading_strategy.positions:
                            position = trading_strategy.positions[pair_info['pair_name']]
                            holding_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
                            print(f"  持仓状态: 🔵 已持仓 {holding_hours:.1f} 小时")
                        else:
                            print(f"  持仓状态: ⚪ 无持仓")
                        
                        print()
                    
                    last_spread_output = current_time
                
                # 检查每个币对（交易逻辑）
                for pair_info in pairs_config:
                    symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
                    
                    if symbol1 not in current_prices or symbol2 not in current_prices:
                        continue
                    
                    if symbol1 not in current_data or symbol2 not in current_data:
                        continue
                    
                    # 计算价差和Z-score（根据diff_order选择计算方式）
                    diff_order = pair_info.get('diff_order', 0)
                    data1 = current_data[symbol1]
                    data2 = current_data[symbol2]
                    
                    if diff_order == 0:
                        # 原始价差
                        current_spread = trading_strategy.calculate_current_spread(
                            current_prices[symbol1], 
                            current_prices[symbol2], 
                            pair_info['hedge_ratio']
                        )
                        
                        # 获取历史价差数据
                        historical_spreads = []
                        for i in range(max(0, len(data1) - trading_strategy.lookback_period), len(data1)):
                            if i < len(data2):
                                hist_spread = trading_strategy.calculate_current_spread(
                                    data1.iloc[i], data2.iloc[i], pair_info['hedge_ratio']
                                )
                                historical_spreads.append(hist_spread)
                    elif diff_order == 1:
                        # 一阶差分价差
                        if len(data1) > 1 and len(data2) > 1:
                            # 当前一阶差分：当前价格 - 前一个价格
                            current_diff1 = current_prices[symbol1] - data1.iloc[-1]
                            current_diff2 = current_prices[symbol2] - data2.iloc[-1]
                            # 一阶差分价差 = diff1 - hedge_ratio * diff2
                            # 注意：hedge_ratio应该从一阶差分价格计算得出
                            current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                            
                            # 获取历史一阶差分价差数据
                            historical_spreads = []
                            for i in range(max(1, len(data1) - trading_strategy.lookback_period), len(data1)):
                                if i < len(data2) and i > 0:
                                    hist_diff1 = data1.iloc[i] - data1.iloc[i-1]
                                    hist_diff2 = data2.iloc[i] - data2.iloc[i-1]
                                    hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                    historical_spreads.append(hist_spread)
                        else:
                            current_spread = 0
                            historical_spreads = []
                    elif diff_order == 2:
                        # 二阶差分价差
                        if len(data1) > 2 and len(data2) > 2:
                            # 当前二阶差分：price[t] - 2*price[t-1] + price[t-2]
                            current_diff1 = (current_prices[symbol1] - 
                                           2 * data1.iloc[-1] + 
                                           data1.iloc[-2])
                            current_diff2 = (current_prices[symbol2] - 
                                           2 * data2.iloc[-1] + 
                                           data2.iloc[-2])
                            # 二阶差分价差 = diff2_1 - hedge_ratio * diff2_2
                            # 注意：hedge_ratio应该从二阶差分价格计算得出
                            current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                            
                            # 获取历史二阶差分价差数据
                            historical_spreads = []
                            for i in range(max(2, len(data1) - trading_strategy.lookback_period), len(data1)):
                                if i < len(data2) and i > 1:
                                    hist_diff1 = (data1.iloc[i] - 
                                                 2 * data1.iloc[i-1] + 
                                                 data1.iloc[i-2])
                                    hist_diff2 = (data2.iloc[i] - 
                                                 2 * data2.iloc[i-1] + 
                                                 data2.iloc[i-2])
                                    hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                    historical_spreads.append(hist_spread)
                        else:
                            current_spread = 0
                            historical_spreads = []
                    else:
                        # 不支持其他差分阶数，使用原始价差
                        current_spread = trading_strategy.calculate_current_spread(
                            current_prices[symbol1], 
                            current_prices[symbol2], 
                            pair_info['hedge_ratio']
                        )
                        historical_spreads = []
                    
                    current_z_score = trading_strategy.calculate_z_score(current_spread, historical_spreads)
                    
                    # 检查平仓条件
                    if pair_info['pair_name'] in trading_strategy.positions:
                        should_close, close_reason = trading_strategy.check_exit_conditions(
                            pair_info, current_prices, current_z_score, datetime.now(), current_spread
                        )
                        
                        if should_close:
                            trading_strategy.close_position(pair_info, current_prices, close_reason, datetime.now(), current_spread)
                    
                    # 检查开仓条件
                    elif len(trading_strategy.positions) == 0:
                        signal = trading_strategy.generate_trading_signal(current_z_score)
                        signal['z_score'] = current_z_score
                        
                        if signal['action'] != 'HOLD':
                            # 获取可用资金
                            try:
                                account_info = binance_api.get_account_info()
                                if account_info:
                                    available_balance = float(account_info.get('availableBalance', 0))
                                    available_capital = available_balance * trading_strategy.position_ratio * trading_strategy.leverage
                                else:
                                    available_capital = trading_strategy.current_capital * trading_strategy.position_ratio * trading_strategy.leverage
                            except:
                                available_capital = trading_strategy.current_capital * trading_strategy.position_ratio * trading_strategy.leverage
                            
                            trading_strategy.execute_trade(pair_info, current_prices, signal, datetime.now(), current_spread, available_capital)
                
                # 更新资金曲线
                trading_strategy.update_capital_curve()
                
                # 每5分钟检查一次
                time.sleep(5)
                
            except Exception as e:
                print(f"交易循环异常: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(60)
    
    # 启动交易循环线程
    trading_thread = threading.Thread(target=trading_loop)
    trading_thread.daemon = True
    trading_thread.start()
    
    try:
        # 启动Web服务器
        server.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n停止实盘交易...")
        trading_strategy.running = False
        data_manager.stop_data_collection()
        print("实盘交易已停止")

def main():
    """主函数"""
    print("实盘协整交易系统（支持ARIMA-GARCH）")
    print("基于 cointegration_test_windows_optimization_arima_garch.py 的核心交易逻辑")
    print("结合 test_live_trading.py 的实时交易基础设施")
    print()
    
    # 执行实盘交易
    test_live_trading()

if __name__ == "__main__":
    main()

