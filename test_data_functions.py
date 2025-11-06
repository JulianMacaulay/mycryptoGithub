#!/usr/bin/env python3
"""
独立的数据获取和下载功能测试文件
直接测试从币安获取真实价格数据和下载功能
不依赖Flask服务器
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_binance_data_fetch():
    """测试从币安获取真实价格数据"""
    print("=" * 60)
    print("测试1: 从币安获取真实价格数据")
    print("=" * 60)
    
    try:
        # 测试币对
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in symbols:
            print(f"\n获取 {symbol} 数据...")
            
            # 从币安API获取K线数据
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   成功获取 {len(data)} 条K线数据")
                
                # 解析K线数据
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
                
                # 转换为DataFrame
                df = pd.DataFrame(klines)
                df.set_index('timestamp', inplace=True)
                
                print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
                print(f"   价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
                print(f"   平均成交量: {df['volume'].mean():.2f}")
                
                # 保存为CSV文件
                csv_filename = f"{symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_filename)
                print(f"   数据已保存到: {csv_filename}")
                
            else:
                print(f"   获取失败: HTTP {response.status_code}")
                print(f"   错误信息: {response.text}")
                
    except Exception as e:
        print(f"币安数据获取失败: {str(e)}")

def test_multiple_symbols():
    """测试获取多个币对数据 - 支持分段保存"""
    print("\n" + "=" * 60)
    print("测试2: 获取多个币对数据")
    print("=" * 60)
    
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'AVAXUSDT', 'SOLUSDT','AAVEUSDT','LTCUSDT']
        all_data = {}
        
        for symbol in symbols:
            print(f"\n获取 {symbol} 数据...")
            
            # 币安API单次请求最多1000条，需要分页获取更多数据
            all_klines = []
            limit = 1000  # 币安API限制
            total_limit = 3000  # 期望获取的总数据量
            max_requests = (total_limit + limit - 1) // limit  # 计算需要的请求次数
            
            # 使用币安API官方分页方式
            start_time = None  # 用于分页的时间戳
            
            for i in range(max_requests):
                print(f"   第 {i+1} 次请求...")
                
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': '1h',
                    'limit': limit
                }
                
                # 如果不是第一次请求，设置endTime参数来获取更早的数据
                if start_time is not None:
                    params['endTime'] = start_time - 1  # 获取endTime之前的数据
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        print(f"   没有更多数据，停止获取")
                        break
                    
                    all_klines.extend(data)
                    print(f"   获取到 {len(data)} 条数据，总计 {len(all_klines)} 条")
                    
                    # 更新start_time为当前批次第一条数据的时间戳，用于下次请求
                    start_time = data[0][0]  # 第一条数据的时间戳
                    
                    # 避免请求过于频繁
                    time.sleep(0.1)
                    
                else:
                    print(f"   请求失败: HTTP {response.status_code}")
                    break
            
            if all_klines:
                # 解析数据
                klines = []
                for kline in all_klines:
                    klines.append({
                        'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                df = pd.DataFrame(klines)
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()  # 按时间排序
                all_data[symbol] = df
                
                print(f"   成功获取 {len(df)} 条数据")
                print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
                print(f"   最新价格: {df['close'].iloc[-1]:.4f}")
                
            else:
                print(f"   获取失败: 没有获取到任何数据")
        
        # 保存所有数据到CSV文件
        if all_data:
            # 保存完整数据
            csv_filename = f"all_symbols_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # 合并所有数据到一个DataFrame
            all_dataframes = []
            for symbol, df in all_data.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                all_dataframes.append(df_copy)
            
            combined_df = pd.concat(all_dataframes, ignore_index=False)
            combined_df.to_csv(csv_filename)
            
            print(f"\n所有数据已保存到: {csv_filename}")
            print(f"包含 {len(all_data)} 个币对的数据")
            print(f"总数据行数: {len(combined_df)}")
            
            # 分段保存数据
            print(f"\n--- 分段保存数据 ---")
            segments = [2000, 1000]  # 第一段1000条，第二段2000条
            
            for segment_index, segment_limit in enumerate(segments):
                print(f"\n保存第 {segment_index + 1} 段数据（{segment_limit} 条）...")
                
                segment_dataframes = []
                for symbol, df in all_data.items():
                    # 按时间排序，取最新的数据
                    df_sorted = df.sort_index()
                    
                    # 计算当前段的起始位置
                    start_idx = sum(segments[:segment_index])  # 前面所有段的总和
                    end_idx = start_idx + segment_limit
                    
                    # 取当前段的数据
                    df_segment = df_sorted.iloc[start_idx:end_idx].copy()
                    df_segment['symbol'] = symbol
                    df_segment['segment'] = f'segment_{segment_index + 1}'
                    segment_dataframes.append(df_segment)
                
                if segment_dataframes:
                    segment_combined_df = pd.concat(segment_dataframes, ignore_index=False)
                    segment_filename = f"segment_{segment_index + 1}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    segment_combined_df.to_csv(segment_filename)
                    
                    print(f"   第 {segment_index + 1} 段数据已保存到: {segment_filename}")
                    print(f"   包含 {len(all_data)} 个币对的数据")
                    print(f"   总数据行数: {len(segment_combined_df)}")
                    print(f"   时间范围: {segment_combined_df.index[0]} 到 {segment_combined_df.index[-1]}")
            
    except Exception as e:
        print(f"多币对数据获取失败: {str(e)}")

def test_different_timeframes():
    """测试不同时间周期的数据获取"""
    print("\n" + "=" * 60)
    print("测试3: 不同时间周期数据获取")
    print("=" * 60)
    
    try:
        symbol = 'BTCUSDT'
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for timeframe in timeframes:
            print(f"\n获取 {symbol} {timeframe} 数据...")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 解析数据
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
                
                df = pd.DataFrame(klines)
                df.set_index('timestamp', inplace=True)
                
                print(f"   成功获取 {len(df)} 条数据")
                print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
                print(f"   价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
                
                # 保存数据
                csv_filename = f"BTCUSDT_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_filename)
                print(f"   数据已保存到: {csv_filename}")
                
            else:
                print(f"   获取失败: HTTP {response.status_code}")
                
    except Exception as e:
        print(f"不同时间周期数据获取失败: {str(e)}")

def test_large_data_fetch():
    """测试获取大量数据"""
    print("\n" + "=" * 60)
    print("测试4: 获取大量数据")
    print("=" * 60)
    
    try:
        symbol = 'BTCUSDT'
        print(f"获取 {symbol} 大量数据...")
        
        # 分批次获取数据
        all_klines = []
        limit = 1000  # 每次最多1000条
        max_requests = 5  # 最多5次请求
        
        for i in range(max_requests):
            print(f"   第 {i+1} 次请求...")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': limit
            }
            
            # 如果不是第一次请求，设置endTime参数
            if i > 0 and all_klines:
                last_timestamp = all_klines[-1][0]
                params['endTime'] = last_timestamp - 1
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if not data:
                    print(f"   没有更多数据，停止获取")
                    break
                
                all_klines.extend(data)
                print(f"   获取到 {len(data)} 条数据，总计 {len(all_klines)} 条")
                
                # 避免请求过于频繁
                time.sleep(0.1)
                
            else:
                print(f"   请求失败: HTTP {response.status_code}")
                break
        
        if all_klines:
            print(f"\n总共获取到 {len(all_klines)} 条数据")
            
            # 解析所有数据
            klines = []
            for kline in all_klines:
                klines.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(klines)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # 按时间排序
            
            print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
            
            # 保存数据
            csv_filename = f"BTCUSDT_large_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_filename)
            print(f"大量数据已保存到: {csv_filename}")
            
    except Exception as e:
        print(f"大量数据获取失败: {str(e)}")

def test_data_validation():
    """测试数据验证"""
    print("\n" + "=" * 60)
    print("测试5: 数据验证")
    print("=" * 60)
    
    try:
        symbol = 'BTCUSDT'
        print(f"验证 {symbol} 数据质量...")
        
        # 获取数据
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': 100
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # 解析数据
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
            
            df = pd.DataFrame(klines)
            df.set_index('timestamp', inplace=True)
            
            print(f"数据点数量: {len(df)}")
            print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 验证OHLC关系
            ohlc_valid = (df['high'] >= df['low']) & (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['low'] <= df['open']) & (df['low'] <= df['close'])
            invalid_ohlc = df[~ohlc_valid]
            if len(invalid_ohlc) > 0:
                print(f"OHLC关系异常: {len(invalid_ohlc)} 个")
            else:
                print("OHLC关系正常")
            
            # 验证价格数据
            print(f"价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
            print(f"价格平均值: {df['close'].mean():.4f}")
            print(f"价格标准差: {df['close'].std():.4f}")
            
            # 检查异常值
            price_std = df['close'].std()
            price_mean = df['close'].mean()
            outliers = df[abs(df['close'] - price_mean) > 3 * price_std]
            if len(outliers) > 0:
                print(f"发现异常价格: {len(outliers)} 个")
            else:
                print("价格数据正常")
            
            # 验证成交量
            print(f"成交量范围: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
            print(f"平均成交量: {df['volume'].mean():.2f}")
            
            # 检查时间间隔
            time_diffs = df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                avg_interval = time_diffs.mean().total_seconds() / 3600
                print(f"平均时间间隔: {avg_interval:.2f} 小时")
            
        else:
            print(f"数据获取失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"数据验证失败: {str(e)}")

def test_performance():
    """测试性能"""
    print("\n" + "=" * 60)
    print("测试6: 性能测试")
    print("=" * 60)
    
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in symbols:
            print(f"\n测试 {symbol} 性能...")
            
            start_time = time.time()
            
            # 获取数据
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"   响应时间: {duration:.2f} 秒")
                print(f"   数据量: {len(data)} 条")
                print(f"   处理速度: {len(data)/duration:.2f} 条/秒")
            else:
                print(f"   请求失败: HTTP {response.status_code}")
                
    except Exception as e:
        print(f"性能测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("开始币安数据获取和下载功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 运行测试
    # test_binance_data_fetch()
    test_multiple_symbols()
    # test_different_timeframes()
    # test_large_data_fetch()
    # test_data_validation()
    # test_performance()
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()