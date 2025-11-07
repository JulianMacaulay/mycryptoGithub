# 回测系统使用说明

## 系统概述

这是一个完整的量化交易回测系统，包含：
1. **策略模块** (`strategies/`): 包含策略实现
2. **回测引擎** (`backtest_system.py`): 完整的回测系统
3. **技术指标** (`strategies/indicators.py`): 常用技术指标计算

## 文件结构

```
.
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # 策略基类
│   ├── turtle_strategy.py    # 海龟交易策略
│   └── indicators.py         # 技术指标计算
├── backtest_system.py        # 完整的回测系统
├── test_backtest.py          # 测试脚本
└── README_BACKTEST.md        # 使用说明
```

## 快速开始

### 1. 基本使用

```python
from strategies.turtle_strategy import TurtleStrategy
from backtest_system import BacktestSystem

# 创建策略
strategy_params = {
    'n_entries': 3,          # 最大加仓次数
    'risk_ratio': 1.0,       # 风险比例%
    'atr_length': 20,        # ATR周期
    'bo_length': 20,          # 短周期突破
    'fs_length': 55,          # 长周期突破
    'te_length': 10,          # 移动止盈周期
    'use_filter': False,      # 是否使用上次盈利过滤
    'mas': 10,               # 短周期均线
    'mal': 20                # 长周期均线
}

strategy = TurtleStrategy(strategy_params)

# 创建回测系统
backtest = BacktestSystem(strategy, initial_capital=10000)

# 从CSV文件加载数据
backtest.load_data_from_csv('your_data.csv', symbol='BTCUSDT')

# 或从币安API加载数据
backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=1000)

# 运行回测
backtest.run_backtest(max_entries=3, risk_ratio=1.0)

# 查看报告
backtest.print_report()

# 绘制结果
backtest.plot_results(save_path='backtest_result.png')
```

### 2. 运行测试

```bash
python test_backtest.py
```

## 策略说明

### 海龟交易策略 (TurtleStrategy)

基于唐奇安通道突破的交易策略，主要特点：

1. **入场信号**:
   - 20日唐奇安通道突破（主要信号）
   - 55日唐奇安通道突破（Failsafe，仅在无持仓时触发）

2. **加仓逻辑**:
   - 每0.5N（N=ATR）加一次仓
   - 最多加仓3次

3. **退出信号**:
   - 均线交叉退出（短周期下穿长周期=平多，上穿=平空）
   - 移动止盈（10日高低点）

4. **可选过滤**:
   - 上次交易盈利过滤（可选）

## 数据格式

CSV文件应包含以下列：
- `timestamp`: 时间戳（或作为索引）
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

如果CSV包含多个币种，可以指定 `symbol` 参数筛选。

## 回测报告指标

- **初始资金**: 回测开始时的资金
- **最终权益**: 回测结束时的权益
- **总收益率**: 总收益百分比
- **最大回撤**: 最大回撤百分比
- **夏普比率**: 风险调整后收益
- **总交易次数**: 完成的交易数量
- **胜率**: 盈利交易占比
- **平均盈利/亏损**: 平均每笔交易的盈亏
- **盈亏比**: 平均盈利/平均亏损

## 注意事项

1. **数据质量**: 确保数据完整且按时间排序
2. **手续费**: 默认手续费率为0.1%，可在BacktestEngine中调整
3. **滑点**: 当前系统使用信号价格成交，实际交易会有滑点
4. **资金管理**: 策略使用风险比例计算仓位，确保有足够资金

## 扩展策略

要创建新策略，继承 `BaseStrategy` 类：

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def initialize(self, data):
        # 计算指标
        pass
    
    def generate_signals(self, data, current_idx, position_size=0):
        # 生成交易信号
        return {'signal': 'hold', 'price': 0, 'reason': ''}
```

## 依赖库

```bash
pip install pandas numpy matplotlib ccxt requests
```

## 问题排查

1. **数据加载失败**: 检查CSV文件格式或网络连接
2. **指标计算错误**: 确保数据量足够（至少需要最长的指标周期）
3. **绘图失败**: 安装matplotlib: `pip install matplotlib`

