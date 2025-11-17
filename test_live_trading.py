#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘交易测试
结合实时数据获取、实时交易执行、实时监控和记录
基于test_out_of_sample_trading.py，添加实时交易功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import time
import threading
import json
import requests
from flask import Flask, jsonify, request
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 币安API配置 ====================

import hmac
import hashlib
import urllib.parse
from urllib.parse import urlencode

# API配置
API_KEY = "SdTSZxmdf61CFsze3udgLRWq0aCaVyyFjsrYKMUOWIfMkm7q3sGRkzSk6QSbM5Qk"
SECRET_KEY = "9HZ04wgrKTy5kDPF5Kman4WSmS9D7YlTscPA7FtX2YLK7vTbpORFNB2jTABQY6HY"
BASE_URL = "https://testnet.binancefuture.com"

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
            print("''=======================下单精度====================='''")
            print(params)
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
                                print(f"  {symbol} 数量精度: {step_size}")
                                return step_size
            print(f"  无法获取 {symbol} 精度信息，使用默认值")
            return 0.001  # 默认精度
        except Exception as e:
            print(f"获取 {symbol} 精度信息异常: {str(e)}")
            return 0.001
    
    def debug_precision_info(self, symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT','PEPEUSDT']):
        """调试精度信息"""
        print("\n" + "="*60)
        print("精度信息调试")
        print("="*60)
        
        try:
            exchange_info = self.get_exchange_info()
            if exchange_info:
                for symbol in symbols:
                    for symbol_info in exchange_info['symbols']:
                        if symbol_info['symbol'] == symbol:
                            print(f"\n{symbol}:")
                            print(f"  baseAsset: {symbol_info.get('baseAsset')}")
                            print(f"  quoteAsset: {symbol_info.get('quoteAsset')}")
                            print(f"  baseAssetPrecision: {symbol_info.get('baseAssetPrecision')}")
                            print(f"  quoteAssetPrecision: {symbol_info.get('quoteAssetPrecision')}")
                            
                            for filter_info in symbol_info['filters']:
                                if filter_info['filterType'] == 'LOT_SIZE':
                                    print(f"  LOT_SIZE过滤器:")
                                    print(f"    minQty: {filter_info.get('minQty')}")
                                    print(f"    maxQty: {filter_info.get('maxQty')}")
                                    print(f"    stepSize: {filter_info.get('stepSize')}")
                                    break
                            break
            else:
                print("无法获取交易对信息")
        except Exception as e:
            print(f"调试精度信息异常: {str(e)}")
        
        print("="*60)
    
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
    
    def start_data_collection(self, symbols, interval='1m'):
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
                
                # 每60秒更新一次
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

# ==================== 实盘交易策略 ====================

class LiveTradingStrategy:
    """实盘交易策略类"""
    
    def __init__(self, binance_api, data_manager, lookback_period=60, z_threshold=2.0, 
                 z_exit_threshold=0.5, take_profit_pct=0.15, stop_loss_pct=0.08, 
                 max_holding_hours=168):
        """
        初始化实盘交易策略
        
        Args:
            binance_api: 币安API客户端
            data_manager: 数据管理器
            lookback_period: 回看期
            z_threshold: Z-score开仓阈值
            z_exit_threshold: Z-score平仓阈值
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止损百分比
            max_holding_hours: 最大持仓时间（小时）
        """
        self.binance_api = binance_api
        self.data_manager = data_manager
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.z_exit_threshold = z_exit_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_hours = max_holding_hours
        self.positions = {}  # 当前持仓
        self.trades = []     # 交易记录
        self.capital_curve = []  # 资金曲线
        self.running = False
        
        # 获取真实账户信息
        self._initialize_account()
        
        # 调试精度信息
        self.binance_api.debug_precision_info()
    
    def _initialize_account(self):
        """初始化账户信息"""
        try:
            print("正在获取真实账户信息...")
            account_info = self.binance_api.get_account_info()
            
            if account_info:
                self.initial_capital = float(account_info.get('totalWalletBalance', 0))
                self.current_capital = self.initial_capital
                print(f"✓ 账户初始化成功")
                print(f"  总资产: {self.initial_capital:.2f} USDT")
                print(f"  可用余额: {float(account_info.get('availableBalance', 0)):.2f} USDT")
                print(f"  未实现盈亏: {float(account_info.get('totalUnrealizedProfit', 0)):.2f} USDT")
                
                # 同步现有持仓
                self._sync_positions()
            else:
                print("✗ 无法获取账户信息，使用默认值")
                self.initial_capital = 10000
                self.current_capital = self.initial_capital
                
        except Exception as e:
            print(f" 账户初始化失败: {str(e)}")
            self.initial_capital = 10000
            self.current_capital = self.initial_capital
    
    def _sync_positions(self):
        """同步现有持仓"""
        try:
            print("正在同步现有持仓...")
            positions_info = self.binance_api.get_position_info()
            
            if positions_info:
                active_positions = [pos for pos in positions_info if float(pos['positionAmt']) != 0]
                print(f"  发现 {len(active_positions)} 个活跃持仓")
                
                for pos in active_positions:
                    symbol = pos['symbol']
                    position_amt = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    unrealized_pnl = float(pos['unRealizedProfit'])
                    
                    print(f"    {symbol}: {position_amt} @ {entry_price} (未实现盈亏: {unrealized_pnl:.2f})")
                    
                    # 这里可以根据需要将现有持仓添加到self.positions中
                    # 但需要确定这些持仓是否与我们的交易策略相关
            else:
                print("  无现有持仓")
                
        except Exception as e:
            print(f"✗ 持仓同步失败: {str(e)}")
        
    def calculate_current_spread(self, price1, price2, hedge_ratio):
        """计算当前价差"""
        return price1 - hedge_ratio * price2
    
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
    
    def execute_trade(self, pair_info, current_prices, signal, timestamp, current_spread):
        """执行交易"""
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        hedge_ratio = pair_info['hedge_ratio']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]
        
        # 获取当前真实账户余额
        try:
            account_info = self.binance_api.get_account_info()
            if account_info:
                available_balance = float(account_info.get('availableBalance', 0))
                print(f"  可用余额: {available_balance:.2f} USDT")
            else:
                available_balance = self.current_capital
        except Exception as e:
            print(f"  获取账户余额失败: {str(e)}")
            available_balance = self.current_capital
        
        # 计算交易量（使用可用余额的10%）
        trade_amount = available_balance * 0.1
        print(f"  计划交易金额: {trade_amount:.2f} USDT")
        
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
        
        decimal_places1 = get_decimal_places(step_size1)
        decimal_places2 = get_decimal_places(step_size2)
        
        # 使用Decimal避免浮点数精度问题
        from decimal import Decimal, ROUND_HALF_UP
        
        # 计算原始数量
        quantity1_raw = Decimal(str(trade_amount)) / Decimal(str(price1))
        quantity2_raw = quantity1_raw * Decimal(str(hedge_ratio))
        
        # 使用stepSize进行精确计算
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
            
            # 执行下单
            order1 = self.binance_api.place_order(symbol1, 'SELL', quantity1)
            order2 = self.binance_api.place_order(symbol2, 'BUY', quantity2)
            
            # 检查下单结果
            if order1 and order2 and order1.get('orderId') and order2.get('orderId'):
                # 等待订单成交
                success, final_status1, final_status2 = self.wait_for_orders_completion(
                    order1, order2, symbol1, symbol2
                )
                
                if success:
                    position = {
                        'pair': f"{symbol1}_{symbol2}",
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'symbol1_size': -quantity1,  # 做空
                        'symbol2_size': quantity2,  # 做多
                        'entry_prices': {symbol1: price1, symbol2: price2},
                        'entry_spread': current_spread,
                        'hedge_ratio': hedge_ratio,
                        'entry_time': timestamp,
                        'signal': signal,
                        'diff_order': signal.get('diff_order', 0),
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
                    'diff_order': signal.get('diff_order', 0),
                    'entry_spread': current_spread
                }
                self.trades.append(trade)
                
                diff_info = f"({signal.get('diff_order', 0)}阶差分)" if signal.get('diff_order', 0) > 0 else "(原始价差)"
                print(f"实盘开仓: {pair_info['pair_name']} {diff_info}")
                print(f"   信号: {signal['description']}")
                print(f"   价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
                print(f"   价差: {current_spread:.6f}")
                print(f"   仓位: {symbol1}={position['symbol1_size']:.2f}, {symbol2}={position['symbol2_size']:.2f}")
                
                return position
        
        elif signal['action'] == 'LONG_SHORT':
            # 做多价差：做多symbol1，做空symbol2
            print(f"  下单计划: {symbol1} BUY {quantity1}, {symbol2} SELL {quantity2}")
            
            # 执行下单
            order1 = self.binance_api.place_order(symbol1, 'BUY', quantity1)
            order2 = self.binance_api.place_order(symbol2, 'SELL', quantity2)
            
            # 检查下单结果
            if order1 and order2 and order1.get('orderId') and order2.get('orderId'):
                # 等待订单成交
                success, final_status1, final_status2 = self.wait_for_orders_completion(
                    order1, order2, symbol1, symbol2
                )
                
                if success:
                    position = {
                        'pair': f"{symbol1}_{symbol2}",
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'symbol1_size': quantity1,  # 做多
                        'symbol2_size': -quantity2,  # 做空
                        'entry_prices': {symbol1: price1, symbol2: price2},
                        'entry_spread': current_spread,
                        'hedge_ratio': hedge_ratio,
                        'entry_time': timestamp,
                        'signal': signal,
                        'diff_order': signal.get('diff_order', 0),
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
                    'diff_order': signal.get('diff_order', 0),
                    'entry_spread': current_spread
                }
                self.trades.append(trade)
                
                diff_info = f"({signal.get('diff_order', 0)}阶差分)" if signal.get('diff_order', 0) > 0 else "(原始价差)"
                print(f"实盘开仓: {pair_info['pair_name']} {diff_info}")
                print(f"   信号: {signal['description']}")
                print(f"   价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
                print(f"   价差: {current_spread:.6f}")
                print(f"   仓位: {symbol1}={position['symbol1_size']:.2f}, {symbol2}={position['symbol2_size']:.2f}")
                
                return position
            else:
                print(f" 订单未成交，不保存持仓")
                return None
        elif order1 and order1.get('orderId') and (not order2 or not order2.get('orderId')):
            # 第一个订单成功，第二个订单失败
            print(f" 配对交易失败: {symbol1} 成功，{symbol2} 失败")
            print(f"  正在紧急平仓 {symbol1}...")
            
            # 紧急平仓第一个订单
            close_success = self.emergency_close_position(
                symbol1, 'SELL', quantity1, f"配对交易失败，{symbol2}下单失败"
            )
            
            if close_success:
                print(f"✓ 紧急平仓成功，风险已控制")
            else:
                print(f"✗ 紧急平仓失败，请手动处理 {symbol1} 仓位")
            
            return None
        elif order2 and order2.get('orderId') and (not order1 or not order1.get('orderId')):
            # 第二个订单成功，第一个订单失败
            print(f" 配对交易失败: {symbol2} 成功，{symbol1} 失败")
            print(f"  正在紧急平仓 {symbol2}...")
            
            # 紧急平仓第二个订单
            close_success = self.emergency_close_position(
                symbol2, 'BUY', quantity2, f"配对交易失败，{symbol1}下单失败"
            )
            
            if close_success:
                print(f"✓ 紧急平仓成功，风险已控制")
            else:
                print(f"✗ 紧急平仓失败，请手动处理 {symbol2} 仓位")
            
            return None
        else:
            print(f" 下单失败: {symbol1} 或 {symbol2} 订单未成功提交")
            if order1:
                print(f"  {symbol1} 订单: {order1}")
            if order2:
                print(f"  {symbol2} 订单: {order2}")
            return None
    
    def check_exit_conditions(self, pair_info, current_prices, current_z_score, timestamp, current_spread):
        """检查平仓条件"""
        pair_name = pair_info['pair_name']
        if pair_name not in self.positions:
            return False, ""
        
        position = self.positions[pair_name]
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        price1, price2 = current_prices[symbol1], current_prices[symbol2]
        
        # 计算当前盈亏
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread
        
        if position['signal']['action'] == 'SHORT_LONG':
            total_pnl = -spread_change
        else:  # LONG_SHORT
            total_pnl = spread_change
        
        # 计算投入资金
        entry_value = abs(position['symbol1_size'] * position['entry_prices'][symbol1]) + \
                     abs(position['symbol2_size'] * position['entry_prices'][symbol2])
        
        # 条件1: Z-score回归
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
                return True, f"止盈触发({pnl_percentage*100:.1f}%)，平仓获利"
        
        # 条件4: 止损条件
        if entry_value > 0:
            pnl_percentage = total_pnl / entry_value
            if total_pnl < 0 and pnl_percentage < -self.stop_loss_pct:
                return True, f"止损触发({pnl_percentage*100:.1f}%)，平仓止损"
        
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
        entry_spread = position['entry_spread']
        spread_change = current_spread - entry_spread
        
        if position['signal']['action'] == 'SHORT_LONG':
            total_pnl = -spread_change
        else:  # LONG_SHORT
            total_pnl = spread_change
        
        # 执行平仓订单
        close_order1 = self.binance_api.place_order(symbol1, 'BUY' if position['symbol1_size'] < 0 else 'SELL', abs(position['symbol1_size']))
        close_order2 = self.binance_api.place_order(symbol2, 'BUY' if position['symbol2_size'] < 0 else 'SELL', abs(position['symbol2_size']))
        
        if close_order1 and close_order2:
            # 更新资金
            self.current_capital += total_pnl
            
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
                'holding_hours': (timestamp - position['entry_time']).total_seconds() / 3600,
                'diff_order': position.get('diff_order', 0)
            }
            self.trades.append(trade)
            
            diff_info = f"({position.get('diff_order', 0)}阶差分)" if position.get('diff_order', 0) > 0 else "(原始价差)"
            print(f"实盘平仓: {pair_name} {diff_info}")
            print(f"   平仓原因: {reason}")
            print(f"   盈亏: {total_pnl:.2f}")
            print(f"   持仓时间: {trade['holding_hours']:.1f}小时")
            print(f"   平仓价格: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
            print(f"   价差变化: {entry_spread:.6f} -> {current_spread:.6f} (变化: {spread_change:.6f})")
            print(f"   当前资金: {self.current_capital:.2f}")
            
            # 移除持仓
            del self.positions[pair_name]
            
            return trade
        
        return None
    
    def update_capital_curve(self):
        """更新资金曲线"""
        # 获取当前真实账户余额
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
    
    def sync_positions_from_api(self):
        """从API同步持仓信息"""
        try:
            positions_info = self.binance_api.get_position_info()
            if positions_info:
                # 更新本地持仓信息
                for pos in positions_info:
                    symbol = pos['symbol']
                    position_amt = float(pos['positionAmt'])
                    
                    if position_amt != 0:
                        # 检查这个持仓是否在我们的交易策略中
                        for pair_name, local_pos in self.positions.items():
                            if symbol in [local_pos['symbol1'], local_pos['symbol2']]:
                                # 更新持仓数量
                                if symbol == local_pos['symbol1']:
                                    local_pos['symbol1_size'] = position_amt
                                else:
                                    local_pos['symbol2_size'] = position_amt
                                print(f"同步持仓: {symbol} = {position_amt}")
                                break
        except Exception as e:
            print(f"同步持仓失败: {str(e)}")
    
    def wait_for_orders_completion(self, order1, order2, symbol1, symbol2, max_wait=30):
        """等待订单成交"""
        print(f"  等待订单成交...")
        
        for i in range(max_wait):
            try:
                # 查询两个订单的状态
                status1 = self.binance_api.get_order_status(order1['orderId'], symbol1)
                status2 = self.binance_api.get_order_status(order2['orderId'], symbol2)
                
                if status1 and status2:
                    status1_str = status1.get('status', 'UNKNOWN')
                    status2_str = status2.get('status', 'UNKNOWN')
                    
                    print(f"    订单状态: {symbol1}={status1_str}, {symbol2}={status2_str}")
                    
                    # 检查是否都成交了
                    if status1_str in ['FILLED', 'PARTIALLY_FILLED'] and \
                       status2_str in ['FILLED', 'PARTIALLY_FILLED']:
                        print(f"   订单成交完成")
                        return True, status1, status2
                    
                    # 检查是否有订单失败
                    elif status1_str in ['CANCELED', 'REJECTED', 'EXPIRED'] or \
                         status2_str in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        print(f"  ✗ 订单失败: {symbol1}={status1_str}, {symbol2}={status2_str}")
                        return False, status1, status2
                
                # 等待1秒后重试
                time.sleep(1)
                
            except Exception as e:
                print(f"  查询订单状态异常: {str(e)}")
                time.sleep(1)
        
        print(f"   等待超时，订单可能未完全成交")
        return False, None, None
    
    def wait_for_single_order_completion(self, order, symbol, max_wait=30):
        """等待单个订单成交"""
        print(f"  等待订单成交: {symbol}")
        
        for i in range(max_wait):
            try:
                status = self.binance_api.get_order_status(order['orderId'], symbol)
                
                if status:
                    status_str = status.get('status', 'UNKNOWN')
                    print(f"    订单状态: {symbol}={status_str}")
                    
                    # 检查是否成交了
                    if status_str in ['FILLED', 'PARTIALLY_FILLED']:
                        print(f"   订单成交完成: {symbol}")
                        return True, status
                    
                    # 检查是否失败
                    elif status_str in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        print(f"   订单失败: {symbol}={status_str}")
                        return False, status
                
                # 等待1秒后重试
                time.sleep(1)
                
            except Exception as e:
                print(f"  查询订单状态异常: {str(e)}")
                time.sleep(1)
        
        print(f"   等待超时，订单可能未完全成交: {symbol}")
        return False, None
    
    def emergency_close_position(self, symbol, side, quantity, reason="紧急平仓"):
        """紧急平仓单个仓位"""
        try:
            print(f" 紧急平仓: {symbol} {side} {quantity} - 原因: {reason}")
            
            # 执行平仓订单
            order = self.binance_api.place_order(symbol, side, quantity)
            
            if order and order.get('orderId'):
                print(f" 紧急平仓订单已提交: {symbol} {side} {quantity}")
                
                # 等待平仓订单成交
                success, final_status, _ = self.wait_for_orders_completion(
                    order, None, symbol, None, max_wait=10
                )
                
                if success:
                    print(f" 紧急平仓成功: {symbol}")
                    return True
                else:
                    print(f" 紧急平仓失败: {symbol}")
                    return False
            else:
                print(f" 紧急平仓订单提交失败: {symbol}")
                return False
                
        except Exception as e:
            print(f"✗ 紧急平仓异常: {symbol} - {str(e)}")
            return False
    
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

# ==================== 币对配置 ====================

def get_pairs_config():
    """获取币对配置"""
    print("\n" + "=" * 80)
    print("币对配置")
    print("=" * 80)
    
    pairs_config = [
        {
            'pair_name': 'BNBUSDT/SOLUSDT',
            'symbol1': 'BNBUSDT',
            'symbol2': 'SOLUSDT',
            'hedge_ratio': 1.787595,  # 对冲比例（从样本内测试获得）
            'diff_order': 1,         # 差分阶数 (0=原始, 1=一阶差分, 2=二阶差分)
            'cointegration_found': True,
            'final_spread': None,
            'best_test': {
                'type': 'manual_input',
                'adf_result': {'p_value': 0.001},
                'spread': None
            }
        }
    ]
    
    print("当前配置的币对:")
    for i, pair in enumerate(pairs_config, 1):
        diff_type = '原始价差' if pair['diff_order'] == 0 else f"{pair['diff_order']}阶差分"
        print(f"{i}. {pair['pair_name']} - 对冲比例: {pair['hedge_ratio']}, 差分阶数: {pair['diff_order']} ({diff_type})")
    
    return pairs_config

# ==================== 实盘交易主函数 ====================

def test_live_trading():
    """实盘交易测试主函数"""
    print("=" * 80)
    print("实盘交易测试")
    print("=" * 80)
    
    # 1. 初始化币安API
    print("\n1. 初始化币安API")
    binance_api = BinanceAPI()  # 使用配置的API密钥和测试网
    
    # 2. 获取币对配置
    print("\n2. 获取币对配置")
    pairs_config = get_pairs_config()
    
    if not pairs_config:
        print("未配置任何币对，无法进行交易")
        return
    
    # 3. 初始化数据管理器
    print("\n3. 初始化数据管理器")
    symbols = []
    for pair in pairs_config:
        symbols.extend([pair['symbol1'], pair['symbol2']])
    symbols = list(set(symbols))  # 去重
    
    data_manager = RealTimeDataManager(binance_api)
    data_manager.start_data_collection(symbols, interval='1m')
    
    # 4. 初始化交易策略
    print("\n4. 初始化交易策略")
    trading_strategy = LiveTradingStrategy(
        binance_api=binance_api,
        data_manager=data_manager,
        lookback_period=60,
        z_threshold=0.5,
        z_exit_threshold=0.6,
        take_profit_pct=0.15,
        stop_loss_pct=0.08,
        max_holding_hours=168
    )
    
    # 5. 启动Web服务器
    print("\n5. 启动Web服务器")
    server = LiveTradingServer(trading_strategy)
    
    # 6. 启动交易循环
    print("\n6. 启动交易循环")
    trading_strategy.running = True
    
    def trading_loop():
        """交易循环"""
        last_spread_output = 0  # 上次输出价差的时间戳
        last_position_sync = 0  # 上次同步持仓的时间戳
        
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
                count = 0
                if current_time - last_spread_output >= 10:
                    print(f"\n{'='*60}")
                    print(f"价差监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")
                    count+=1
                    print(count)
                    for pair_info in pairs_config:
                        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
                        
                        if symbol1 not in current_prices or symbol2 not in current_prices:
                            continue
                        
                        if symbol1 not in current_data or symbol2 not in current_data:
                            continue
                        
                        # 计算价差和Z-score
                        if pair_info['diff_order'] == 0:
                            # 原始协整
                            current_spread = trading_strategy.calculate_current_spread(
                                current_prices[symbol1], 
                                current_prices[symbol2], 
                                pair_info['hedge_ratio']
                            )
                            
                            # 获取历史价差数据
                            historical_spreads = []
                            data1 = current_data[symbol1]
                            data2 = current_data[symbol2]
                            
                            for i in range(max(0, len(data1) - trading_strategy.lookback_period), len(data1)):
                                if i < len(data2):
                                    hist_spread = trading_strategy.calculate_current_spread(
                                        data1.iloc[i], data2.iloc[i], pair_info['hedge_ratio']
                                    )
                                    historical_spreads.append(hist_spread)
                        
                        elif pair_info['diff_order'] == 1:
                            # 一阶差分
                            data1 = current_data[symbol1]
                            data2 = current_data[symbol2]
                            
                            if len(data1) > 1 and len(data2) > 1:
                                current_diff1 = current_prices[symbol1] - data1.iloc[-2]
                                current_diff2 = current_prices[symbol2] - data2.iloc[-2]
                                current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                                
                                # 获取历史一阶差分价差数据
                                historical_spreads = []
                                for i in range(max(1, len(data1) - trading_strategy.lookback_period), len(data1)):
                                    if i < len(data2):
                                        hist_diff1 = data1.iloc[i] - data1.iloc[i-1]
                                        hist_diff2 = data2.iloc[i] - data2.iloc[i-1]
                                        hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                        historical_spreads.append(hist_spread)
                            else:
                                current_spread = 0
                                historical_spreads = []
                        
                        else:
                            current_spread = 0
                            historical_spreads = []
                        
                        current_z_score = trading_strategy.calculate_z_score(current_spread, historical_spreads)
                        
                        # 输出价差信息
                        diff_type = '原始价差' if pair_info['diff_order'] == 0 else f"{pair_info['diff_order']}阶差分"
                        print(f"币对: {pair_info['pair_name']} ({diff_type})")
                        print(f"  价格: {symbol1}={current_prices[symbol1]:.4f}, {symbol2}={current_prices[symbol2]:.4f}")
                        print(f"  对冲比例: {pair_info['hedge_ratio']:.6f}")
                        print(f"  当前价差: {current_spread:.8f}")
                        print(f"  Z-score: {current_z_score:.4f}")
                        print(f"  历史价差数量: {len(historical_spreads)}")
                        
                        if len(historical_spreads) > 0:
                            spread_mean = np.mean(historical_spreads)
                            spread_std = np.std(historical_spreads)
                            print(f"  价差均值: {spread_mean:.8f}")
                            print(f"  价差标准差: {spread_std:.8f}")
                        
                        # 显示交易信号
                        signal = trading_strategy.generate_trading_signal(current_z_score, pair_info['diff_order'])
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
                    
                    # 计算价差和Z-score（与上面相同的逻辑）
                    if pair_info['diff_order'] == 0:
                        # 原始协整
                        current_spread = trading_strategy.calculate_current_spread(
                            current_prices[symbol1], 
                            current_prices[symbol2], 
                            pair_info['hedge_ratio']
                        )
                        
                        # 获取历史价差数据
                        historical_spreads = []
                        data1 = current_data[symbol1]
                        data2 = current_data[symbol2]
                        
                        for i in range(max(0, len(data1) - trading_strategy.lookback_period), len(data1)):
                            if i < len(data2):
                                hist_spread = trading_strategy.calculate_current_spread(
                                    data1.iloc[i], data2.iloc[i], pair_info['hedge_ratio']
                                )
                                historical_spreads.append(hist_spread)
                    
                    elif pair_info['diff_order'] == 1:
                        # 一阶差分
                        data1 = current_data[symbol1]
                        data2 = current_data[symbol2]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            current_diff1 = current_prices[symbol1] - data1.iloc[-2]
                            current_diff2 = current_prices[symbol2] - data2.iloc[-2]
                            current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                            
                            # 获取历史一阶差分价差数据
                            historical_spreads = []
                            for i in range(max(1, len(data1) - trading_strategy.lookback_period), len(data1)):
                                if i < len(data2):
                                    hist_diff1 = data1.iloc[i] - data1.iloc[i-1]
                                    hist_diff2 = data2.iloc[i] - data2.iloc[i-1]
                                    hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                    historical_spreads.append(hist_spread)
                        else:
                            current_spread = 0
                            historical_spreads = []
                    
                    else:
                        current_spread = 0
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
                        signal = trading_strategy.generate_trading_signal(current_z_score, pair_info['diff_order'])
                        signal['z_score'] = current_z_score
                        
                        if signal['action'] != 'HOLD':
                            trading_strategy.execute_trade(pair_info, current_prices, signal, datetime.now(), current_spread)
                
                # 定期同步持仓（每2分钟）
                if current_time - last_position_sync >= 10:
                    trading_strategy.sync_positions_from_api()
                    last_position_sync = current_time
                
                # 更新资金曲线
                trading_strategy.update_capital_curve()
                
                # 每5分钟检查一次
                time.sleep(5)
                
            except Exception as e:
                print(f"交易循环异常: {str(e)}")
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
    print("实盘交易测试")
    print("结合实时数据获取、实时交易执行、实时监控和记录")
    print()
    
    # 执行实盘交易测试
    test_live_trading()

if __name__ == "__main__":
    main()
