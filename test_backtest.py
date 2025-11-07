"""
回测系统测试脚本
快速测试回测系统是否正常工作
"""

import sys
import os
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.turtle_strategy import TurtleStrategy
from backtest_system import BacktestSystem


def test_backtest_with_sample_data():
    """使用示例数据测试回测系统"""
    print("=" * 60)
    print("回测系统测试")
    print("=" * 60)
    
    # 创建策略
    strategy_params = {
        'n_entries': 3,
        'risk_ratio': 1.0,
        'atr_length': 20,
        'bo_length': 20,
        'fs_length': 55,
        'te_length': 10,
        'use_filter': False,
        'mas': 10,
        'mal': 20
    }
    
    strategy = TurtleStrategy(strategy_params)
    print(f"策略创建成功: {strategy.name}")
    
    # 创建回测系统
    backtest = BacktestSystem(strategy, initial_capital=10000)
    print("回测系统创建成功")
    
    # 尝试从CSV文件加载数据
    csv_files = [
        "all_symbols_data_ccxt_20251106_195714.csv",
        "segment_1_data_ccxt_20251106_195714.csv",
        "segment_2_data_ccxt_20251106_195714.csv"
    ]
    
    data_loaded = False
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                print(f"\n尝试从 {csv_file} 加载数据...")
                backtest.load_data_from_csv(csv_file, symbol='BTCUSDT')
                data_loaded = True
                break
            except Exception as e:
                print(f"加载失败: {e}")
                continue
    
    if not data_loaded:
        print("\n未找到CSV文件，尝试从币安API获取数据...")
        try:
            backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=500)
            data_loaded = True
        except Exception as e:
            print(f"从币安获取数据失败: {e}")
            print("\n创建模拟数据进行测试...")
            # 创建模拟数据
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            np.random.seed(42)
            price = 50000
            data = []
            for i in range(200):
                price += np.random.randn() * 100
                data.append({
                    'open': price,
                    'high': price + abs(np.random.randn() * 50),
                    'low': price - abs(np.random.randn() * 50),
                    'close': price,
                    'volume': np.random.uniform(100, 1000)
                })
            
            df = pd.DataFrame(data, index=dates)
            backtest.data = df
            print(f"创建模拟数据: {len(df)} 条记录")
            data_loaded = True
    
    if data_loaded:
        print("\n开始运行回测...")
        try:
            backtest.run_backtest(max_entries=3, risk_ratio=1.0)
            backtest.print_report()
            print("\n回测完成！")
            
            # 询问是否绘制图表
            try:
                backtest.plot_results(save_path='backtest_result.png')
            except Exception as e:
                print(f"绘图失败（可能缺少matplotlib）: {e}")
                print("可以安装matplotlib来查看图表: pip install matplotlib")
        except Exception as e:
            print(f"回测运行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("无法加载数据，测试终止")


if __name__ == "__main__":
    test_backtest_with_sample_data()

