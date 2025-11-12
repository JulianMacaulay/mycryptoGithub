# cointegration_test_windows.py 运行逻辑说明文档

## 概述

`cointegration_test_windows.py` 是一个使用**滚动窗口**进行协整检验和交易回测的程序。与一次性使用全部数据不同，它使用固定大小的窗口在时间序列上滑动，对每个窗口进行协整检验，从而识别协整关系的时变特性。

## 核心特点

1. **滚动窗口检验**：使用固定大小的窗口（如1000条数据）在时间序列上滑动
2. **完整的EG两阶段检验**：每个窗口内都进行完整的Engle-Granger两阶段协整检验
3. **时变特性识别**：可以发现协整关系在不同时间段的存在情况
4. **汇总分析**：统计所有窗口中协整检验通过的比例

---

## 运行流程（按执行顺序）

### 第一步：程序启动

**代码位置：** 第1364-1365行

```python
if __name__ == "__main__":
    main()
```

**执行逻辑：**
- 程序入口，调用 `main()` 函数

---

### 第二步：main() 函数 - 初始化

**代码位置：** 第1341-1361行

```python
def main():
    print("滚动窗口协整分析+交易流程完整测试")
    print("使用滚动窗口进行协整检验：")
    print("  1. 使用固定大小的滚动窗口（如1000条数据）")
    print("  2. 在每个窗口内进行完整的EG两阶段协整检验")
    print("  3. 汇总所有窗口的检验结果，识别协整关系的时变特性")
    print()
    
    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()
    
    if not csv_file_path:
        csv_file_path = "segment_2_data_ccxt_20251106_195714.csv"
        print(f"使用默认路径: {csv_file_path}")
    
    test_rolling_window_cointegration_trading(csv_file_path)
```

**执行逻辑：**
1. 显示程序说明
2. 提示用户输入CSV文件路径
3. 如果用户未输入，使用默认路径
4. 调用 `test_rolling_window_cointegration_trading()` 开始主流程

---

### 第三步：test_rolling_window_cointegration_trading() - 主测试流程

**代码位置：** 第1216-1338行

#### 3.1 加载数据

**代码位置：** 第1227-1234行

```python
# 1. 加载数据
print("\n1. 加载数据")
data = load_csv_data(csv_file_path)
if data is None:
    print("数据加载失败")
    return

print(f"成功加载 {len(data)} 个币对的数据")
```

**执行逻辑：**
- 调用 `load_csv_data()` 加载CSV文件
- 返回格式：`{symbol: price_series}` 字典

**相关函数：** `load_csv_data()` (第31-71行)
- 读取CSV文件
- 按symbol分组
- 提取close价格序列
- 设置时间戳索引

---

#### 3.2 配置滚动窗口参数

**代码位置：** 第1236-1238行

```python
# 2. 配置滚动窗口参数
print("\n2. 配置滚动窗口参数")
window_params = configure_rolling_window_parameters()
```

**执行逻辑：**
- 调用 `configure_rolling_window_parameters()` 配置窗口参数
- 返回参数：`{'window_size': 1000, 'step_size': 100}`

**相关函数：** `configure_rolling_window_parameters()` (第503-547行)
- 默认窗口大小：1000条数据
- 默认步长：100条数据
- 用户可以选择修改这些参数

**参数说明：**
- `window_size`：每个窗口包含的数据条数
- `step_size`：每次窗口移动的数据条数（步长）

---

#### 3.3 滚动窗口寻找协整对

**代码位置：** 第1240-1252行

```python
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
```

**执行逻辑：**
- 调用 `rolling_window_find_cointegrated_pairs()` 对所有币对进行滚动窗口检验
- 返回所有币对的汇总结果

**相关函数：** `rolling_window_find_cointegrated_pairs()` (第457-500行)

**执行流程：**
1. 遍历所有币对组合（C(n,2)）
2. 对每个币对调用 `rolling_window_cointegration_test()`
3. 汇总所有币对的检验结果

---

#### 3.4 滚动窗口协整检验核心逻辑

**相关函数：** `rolling_window_cointegration_test()` (第347-454行)

**代码位置：** 第380-435行（核心循环）

```python
# 滚动窗口
num_windows = (min_length - window_size) // step_size + 1

for window_idx in range(num_windows):
    start_idx = window_idx * step_size
    end_idx = start_idx + window_size
    
    if end_idx > min_length:
        end_idx = min_length
        start_idx = end_idx - window_size
    
    # 提取窗口数据
    window_price1 = price1_aligned.iloc[start_idx:end_idx]
    window_price2 = price2_aligned.iloc[start_idx:end_idx]
    
    # 对当前窗口进行协整检验
    coint_result = enhanced_cointegration_test(
        window_price1,
        window_price2,
        symbol1,
        symbol2,
        verbose=False
    )
    
    # 添加窗口信息
    coint_result['window_idx'] = window_idx
    coint_result['window_start_time'] = window_start_time
    coint_result['window_end_time'] = window_end_time
    # ...
    
    window_results.append(coint_result)
    
    if coint_result['cointegration_found']:
        all_candidates.append(coint_result)
```

**执行逻辑：**
1. **计算窗口数量**：`num_windows = (总数据长度 - 窗口大小) // 步长 + 1`
2. **循环每个窗口**：
   - 计算窗口的起始和结束索引
   - 提取窗口内的价格数据
   - 对窗口数据进行协整检验
   - 记录检验结果
3. **汇总结果**：
   - 统计协整检验通过的窗口数量
   - 计算协整比例 = 协整窗口数 / 总窗口数

**示例：**
```
总数据：2000条
窗口大小：1000条
步长：100条

窗口1：索引 0-999
窗口2：索引 100-1099
窗口3：索引 200-1199
...
窗口11：索引 1000-1999
```

---

#### 3.5 单个窗口的协整检验

**相关函数：** `enhanced_cointegration_test()` (第204-342行)

**执行步骤：**

**步骤1：检验price1的积分阶数**
```python
price1_order = determine_integration_order(price1, max_order=2)
```
- 调用 `determine_integration_order()` 确定积分阶数
- 如果price1不是I(1)，跳过协整检验

**步骤2：检验price2的积分阶数**
```python
price2_order = determine_integration_order(price2, max_order=2)
```
- 同样确定price2的积分阶数
- 如果price2不是I(1)，跳过协整检验

**步骤3：检查是否同阶单整**
```python
if price1_order != price2_order:
    return results  # 积分阶数不同，不能协整
```

**步骤4：计算对冲比率**
```python
hedge_ratio = calculate_hedge_ratio(price1, price2)
```
- 使用OLS回归计算：`price1 = α + β * price2 + ε`
- 对冲比率 = 回归系数β

**步骤5：计算价差**
```python
spread = price1_aligned - hedge_ratio * price2_aligned
```
- 价差 = price1 - β * price2

**步骤6：检验价差的平稳性**
```python
spread_adf = advanced_adf_test(spread)
```
- 对价差进行ADF检验
- 如果P值 < 0.05，价差平稳，协整关系成立

**相关辅助函数：**
- `determine_integration_order()` (第165-199行)：确定积分阶数
- `calculate_hedge_ratio()` (第74-110行)：计算对冲比率
- `advanced_adf_test()` (第113-162行)：ADF检验

---

#### 3.6 显示并选择协整对

**代码位置：** 第1254-1260行

```python
# 4. 显示并选择协整对
print("\n4. 选择协整对")
selected_pairs = display_rolling_window_candidates(all_summaries, min_cointegration_ratio=0.5)

if not selected_pairs:
    print("未选择任何币对，无法进行交易")
    return
```

**执行逻辑：**
- 调用 `display_rolling_window_candidates()` 显示候选币对
- 只显示协整比例 >= 50% 的币对
- 用户选择要交易的币对

**相关函数：** `display_rolling_window_candidates()` (第550-634行)

**显示内容：**
- 币对名称
- 总窗口数
- 协整窗口数
- 协整比例
- 最佳窗口信息（P值最小的窗口）

**选择逻辑：**
- 用户输入序号（如：1,3,5）
- 返回选择的最佳窗口结果（用于后续交易）

---

#### 3.7 交易回测循环

**代码位置：** 第1262-1333行

```python
# 5. 循环测试币对
test_count = 0

while True:
    test_count += 1
    continue_choice = input("是否继续测试？(y/n): ").strip().lower()
    
    if continue_choice != 'y':
        break
    
    # 显示选择的币对详情
    # 配置交易参数
    # 执行交易回测
    # 显示交易详情
```

**执行逻辑：**
1. 用户可以多次测试不同的参数组合
2. 每次测试包括：
   - 显示币对详情
   - 配置交易参数
   - 执行回测
   - 显示交易结果

---

#### 3.8 配置交易参数

**代码位置：** 第1288-1290行

```python
# 7. 配置交易参数
trading_params = configure_trading_parameters()
```

**相关函数：** `configure_trading_parameters()` (第637-725行)

**默认参数：**
- `lookback_period`: 60（回看期）
- `z_threshold`: 1.5（Z-score开仓阈值）
- `z_exit_threshold`: 0.6（Z-score平仓阈值）
- `take_profit_pct`: 0.15（止盈15%）
- `stop_loss_pct`: 0.08（止损8%）
- `max_holding_hours`: 168（最大持仓168小时）

---

#### 3.9 执行交易回测

**代码位置：** 第1292-1307行

```python
# 8. 执行交易回测
trading_strategy = AdvancedCointegrationTrading(
    lookback_period=trading_params['lookback_period'],
    z_threshold=trading_params['z_threshold'],
    z_exit_threshold=trading_params['z_exit_threshold'],
    take_profit_pct=trading_params['take_profit_pct'],
    stop_loss_pct=trading_params['stop_loss_pct'],
    max_holding_hours=trading_params['max_holding_hours']
)

results = trading_strategy.backtest_cointegration_trading(
    data,
    selected_pairs,
    initial_capital=10000
)
```

**执行逻辑：**
- 创建 `AdvancedCointegrationTrading` 实例
- 调用 `backtest_cointegration_trading()` 执行回测

**相关类：** `AdvancedCointegrationTrading` (第730-1213行)

---

#### 3.10 交易回测核心逻辑

**相关方法：** `backtest_cointegration_trading()` (第1085-1213行)

**代码位置：** 第1117-1175行（核心循环）

```python
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
        
        # 计算当前价差
        current_spread = self.calculate_current_spread(
            current_prices[symbol1],
            current_prices[symbol2],
            pair_info['hedge_ratio']
        )
        
        # 获取历史价差数据
        historical_spreads = []
        for j in range(max(0, i-self.lookback_period), i):
            # 计算历史价差...
            historical_spreads.append(hist_spread)
        
        # 计算Z-score
        current_z_score = self.calculate_z_score(current_spread, historical_spreads)
        
        # 检查平仓条件
        if pair_info['pair_name'] in self.positions:
            should_close, close_reason = self.check_exit_conditions(...)
            if should_close:
                trade = self.close_position(...)
        
        # 检查开仓条件
        elif len(self.positions) == 0:
            signal = self.generate_trading_signal(current_z_score)
            if signal['action'] != 'HOLD':
                position = self.execute_trade(...)
```

**执行逻辑：**
1. **遍历所有时间点**
2. **对每个币对**：
   - 计算当前价差：`spread = price1 - hedge_ratio * price2`
   - 获取历史价差数据（用于计算Z-score）
   - 计算Z-score：`(当前价差 - 历史均值) / 历史标准差`
3. **检查平仓条件**：
   - Z-score回归到均值附近
   - 持仓时间过长
   - 止盈触发
   - 止损触发
4. **检查开仓条件**：
   - Z-score > 阈值：做空价差
   - Z-score < -阈值：做多价差
   - 否则：观望

**关键方法：**
- `calculate_current_spread()` (第755-757行)：计算当前价差
- `calculate_z_score()` (第759-770行)：计算Z-score
- `generate_trading_signal()` (第772-791行)：生成交易信号
- `check_exit_conditions()` (第858-905行)：检查平仓条件
- `execute_trade()` (第793-856行)：执行开仓
- `close_position()` (第907-957行)：执行平仓

---

#### 3.11 显示交易详情

**代码位置：** 第1309-1330行

```python
# 9. 显示交易详情
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
            print(f"  {trade['timestamp']}: {action} ...")
```

**执行逻辑：**
- 按币对分组显示所有交易记录
- 显示开仓/平仓信息
- 显示盈亏情况

---

#### 3.12 计算风险指标

**相关方法：** `calculate_risk_metrics()` (第959-1012行)

**代码位置：** 第1190-1207行

```python
# 计算风险指标
risk_metrics = self.calculate_risk_metrics(results['capital_curve'])

print(f"\n回测结果:")
print(f"  初始资金: {initial_capital:,.2f}")
print(f"  最终资金: {capital:,.2f}")
print(f"  总收益率: {final_return:.2f}%")
print(f"  总交易次数: {total_trades / 2}")
print(f"  盈利交易: {profitable_trades}")
print(f"  胜率: {profitable_trades / (total_trades / 2) * 100:.1f}%")

print(f"\n风险指标:")
print(f"  最大回撤: {risk_metrics.get('max_drawdown', 0):,.2f}")
print(f"  最大回撤百分比: {risk_metrics.get('max_drawdown_pct', 0):.2f}%")
print(f"  盈亏比: {risk_metrics.get('profit_loss_ratio', 0):.2f}")
print(f"  夏普比率: {risk_metrics.get('sharpe_ratio', 0):.2f}")
```

**计算的风险指标：**
- 最大回撤
- 最大回撤百分比
- 盈亏比
- 夏普比率
- 平均盈利/亏损

---

#### 3.13 绘制收益率曲线

**代码位置：** 第1209-1211行

```python
# 绘制收益率曲线图
print(f"\n正在生成收益率曲线图...")
self.plot_equity_curve(results['capital_curve'])
```

**相关方法：** `plot_equity_curve()` (第1014-1083行)

**执行逻辑：**
- 绘制资金曲线
- 绘制收益率曲线
- 保存或显示图表

---

## 关键数据结构

### 1. 数据加载结果

```python
data = {
    'BTCUSDT': pd.Series([价格序列], index=[时间戳]),
    'ETHUSDT': pd.Series([价格序列], index=[时间戳]),
    ...
}
```

### 2. 滚动窗口检验结果

```python
summary = {
    'pair_name': 'BTCUSDT/ETHUSDT',
    'symbol1': 'BTCUSDT',
    'symbol2': 'ETHUSDT',
    'total_windows': 11,  # 总窗口数
    'cointegration_windows': 8,  # 协整窗口数
    'cointegration_ratio': 0.727,  # 协整比例
    'window_results': [...],  # 所有窗口的检验结果
    'all_candidates': [...]  # 通过协整检验的窗口
}
```

### 3. 单个窗口检验结果

```python
coint_result = {
    'pair_name': 'BTCUSDT/ETHUSDT',
    'symbol1': 'BTCUSDT',
    'symbol2': 'ETHUSDT',
    'price1_order': 1,  # I(1)
    'price2_order': 1,  # I(1)
    'hedge_ratio': 0.05,  # 对冲比率
    'spread': pd.Series([价差序列]),
    'spread_adf': {
        'p_value': 0.02,
        'is_stationary': True
    },
    'cointegration_found': True,
    'window_idx': 0,
    'window_start_time': '2024-01-01',
    'window_end_time': '2024-01-10',
    ...
}
```

### 4. 交易记录

```python
trade = {
    'timestamp': '2024-01-15 10:00:00',
    'pair': 'BTCUSDT/ETHUSDT',
    'action': 'OPEN' or 'CLOSE',
    'symbol1': 'BTCUSDT',
    'symbol2': 'ETHUSDT',
    'symbol1_price': 50000.0,
    'symbol2_price': 3000.0,
    'hedge_ratio': 0.05,
    'z_score': 2.5,
    'pnl': 100.0,  # 仅平仓时有
    ...
}
```

---

## 关键算法说明

### 1. 滚动窗口计算

```python
num_windows = (min_length - window_size) // step_size + 1

for window_idx in range(num_windows):
    start_idx = window_idx * step_size
    end_idx = start_idx + window_size
```

**示例：**
- 总数据：2000条
- 窗口大小：1000条
- 步长：100条
- 窗口数：(2000 - 1000) // 100 + 1 = 11个窗口

### 2. Z-score计算

```python
spread_mean = np.mean(historical_spreads)
spread_std = np.std(historical_spreads)
z_score = (current_spread - spread_mean) / spread_std
```

**交易信号：**
- Z-score > 阈值：做空价差（价差过高，预期回归）
- Z-score < -阈值：做多价差（价差过低，预期回归）
- 否则：观望

### 3. 盈亏计算

```python
# 做空价差
if position['signal']['action'] == 'SHORT_LONG':
    total_pnl = -spread_change  # 价差减少时盈利

# 做多价差
else:  # LONG_SHORT
    total_pnl = spread_change  # 价差增加时盈利
```

---

## 与原始版本的区别

| 特性 | cointegration_test_original.py | cointegration_test_windows.py |
|------|--------------------------------|-------------------------------|
| **数据使用方式** | 一次性使用全部数据 | 使用滚动窗口 |
| **协整检验** | 对整个数据集检验一次 | 对每个窗口检验一次 |
| **结果输出** | 单个检验结果 | 多个窗口的检验结果汇总 |
| **时变特性** | 无法识别 | 可以识别协整关系的时变特性 |
| **适用场景** | 数据量较小，关系稳定 | 数据量较大，关系可能变化 |

---

## 使用建议

1. **窗口大小选择**：
   - 太小（<500）：检验结果不稳定
   - 太大（>2000）：可能包含结构性断点
   - 推荐：1000-1500条数据

2. **步长选择**：
   - 太小（<50）：窗口重叠太多，计算量大
   - 太大（>200）：可能遗漏重要时间段
   - 推荐：100-200条数据

3. **协整比例阈值**：
   - 推荐：>= 50%
   - 如果阈值太高，可能找不到币对
   - 如果阈值太低，可能选择不稳定的币对

4. **交易参数调整**：
   - 根据回测结果调整Z-score阈值
   - 根据市场波动调整止盈止损
   - 根据持仓周期调整最大持仓时间

---

## 总结

`cointegration_test_windows.py` 通过滚动窗口的方式，能够：
1. **识别协整关系的时变特性**：发现哪些时间段存在协整关系
2. **提高检验的稳定性**：避免因数据长度变化导致的结果不一致
3. **更符合实际交易场景**：使用固定窗口的历史数据进行检验和交易

程序按照：**加载数据 → 配置参数 → 滚动窗口检验 → 选择币对 → 交易回测 → 显示结果** 的顺序执行，每个步骤都有详细的输出和用户交互。

