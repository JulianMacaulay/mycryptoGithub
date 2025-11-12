# cointegration_test_original.py 详细说明文档

## 概述

本文件实现了**完整的Engle-Granger两阶段协整检验**和**协整交易策略回测**。这是正确的协整检验方法，严格按照EG两阶段流程进行。

## 核心特点

1. **完整的EG两阶段协整检验**：
   - 先检验原序列的积分阶数
   - 只有当两个原序列都是I(1)时，才进行协整检验
   - 检验原序列价差的平稳性（不是差分的价差）

2. **使用原始数据进行协整检验**：
   - 价差计算使用原始价格序列
   - 不进行一阶差分处理

3. **完整的交易回测系统**：
   - 基于Z-score的交易信号生成
   - 止盈止损机制
   - 完整的风险指标计算

---

## 程序运行流程（按执行顺序）

### 第一步：程序入口 - main() 函数

**代码位置：** 第1130-1154行

**功能：** 程序入口，获取CSV文件路径并调用测试函数

**执行流程：**
```python
def main():
    print("高级版协整分析+交易流程完整测试（修正版）")
    print("使用正确的协整检验逻辑：")
    print("  1. 先检验原序列的平稳性")
    print("  2. 只有当两个原序列都是I(1)时，才进行协整检验")
    print("  3. 检验原序列价差的平稳性（不是差分的价差）")
    print()
    
    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()
    
    if not csv_file_path:
        csv_file_path = "segment_2_data_ccxt_20251106_195714.csv"
        print(f"使用默认路径: {csv_file_path}")
    
    test_advanced_cointegration_trading(csv_file_path)
```

**关键点：**
- 用户可以选择输入CSV文件路径，或使用默认路径
- 调用 `test_advanced_cointegration_trading()` 开始测试流程

---

### 第二步：主测试函数 - test_advanced_cointegration_trading() 函数

**代码位置：** 第1015-1128行

**功能：** 执行完整的协整分析+交易测试流程

**执行流程：**
```python
def test_advanced_cointegration_trading(csv_file_path):
    # 1. 加载数据
    data = load_csv_data(csv_file_path)
    
    # 2. 寻找协整对
    all_candidates = enhanced_find_cointegrated_pairs(data)
    
    # 3. 循环测试币对
    while True:
        # 3.1 显示候选币对供选择
        selected_pairs = display_candidates_for_selection(all_candidates)
        
        # 3.2 配置交易参数
        trading_params = configure_trading_parameters()
        
        # 3.3 执行交易回测
        trading_strategy = AdvancedCointegrationTrading(...)
        results = trading_strategy.backtest_cointegration_trading(...)
        
        # 3.4 显示交易详情
        # ...
```

**关键点：**
- 这是主流程控制函数
- 按顺序调用各个功能模块
- 支持循环测试多个币对

---

### 第三步：加载数据 - load_csv_data() 函数

**代码位置：** 第29-69行

**功能：** 从CSV文件加载多个币对的价格数据

**执行流程：**
```python
def load_csv_data(csv_file_path):
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 2. 检查数据格式
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 3. 如果有symbol列，按币对分组
    if 'symbol' in df.columns:
        data = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
            symbol_data.set_index('timestamp', inplace=True)
            data[symbol] = symbol_data['close']  # 只提取收盘价
            print(f"币对 {symbol}: {len(symbol_data)} 条数据")
        return data
    else:
        # 假设是单个币对的数据
        data = {'BTCUSDT': df['close']}
        return data
```

**返回格式：**
- `{symbol: price_series}` 字典
- 其中 `price_series` 是 pandas Series，索引是时间戳，值是收盘价

**关键点：**
- 只提取收盘价（close）作为价格序列
- 自动处理时间戳索引
- 支持多币对和单币对数据

---

### 第四步：寻找协整对 - enhanced_find_cointegrated_pairs() 函数

**代码位置：** 第319-364行

**功能：** 遍历所有币对组合，寻找协整对

**执行流程：**
```python
def enhanced_find_cointegrated_pairs(data):
    symbols = list(data.keys())
    all_candidates = []
    
    # 遍历所有币对组合
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]
            
            # 对每个币对执行协整检验
            coint_result = enhanced_cointegration_test(
                data[symbol1], 
                data[symbol2], 
                symbol1,
                symbol2
            )
            
            # 如果找到协整关系，添加到候选列表
            if coint_result['cointegration_found']:
                all_candidates.append(coint_result)
    
    return all_candidates
```

**关键点：**
- 遍历所有币对组合（避免重复）
- 对每个币对调用 `enhanced_cointegration_test()` 进行协整检验
- 返回所有找到的协整对候选列表

---

### 第五步：协整检验核心 - enhanced_cointegration_test() 函数

**代码位置：** 第198-317行

**功能：** 执行完整的Engle-Granger两阶段协整检验

**执行流程（按顺序）：**

#### 5.1 步骤1：检验 price1 的积分阶数

```python
# 步骤1: 检验price1的积分阶数
print(f"\n--- 步骤1: 检验 {symbol1} 的积分阶数 ---")
price1_order = determine_integration_order(price1, max_order=2)
```

**使用函数：** `determine_integration_order()`（第160-194行）

**执行逻辑：**
1. 使用 `advanced_adf_test()` 检验原序列是否平稳
2. 如果平稳，返回 0（I(0)）
3. 如果不平稳，检验一阶差分是否平稳
4. 如果一阶差分平稳，返回 1（I(1)）
5. 如果一阶差分不平稳，检验二阶差分
6. 如果二阶差分平稳，返回 2（I(2)）

**关键点：**
- 使用 `advanced_adf_test()` 检验平稳性
- 如果 price1 是 I(0)，不能进行协整检验
- 如果 price1 的积分阶数无法确定，跳过协整检验

#### 5.2 步骤2：检验 price2 的积分阶数

```python
# 步骤2: 检验price2的积分阶数
print(f"\n--- 步骤2: 检验 {symbol2} 的积分阶数 ---")
price2_order = determine_integration_order(price2, max_order=2)
```

**执行逻辑：** 与步骤1相同

**关键点：**
- 如果 price2 是 I(0)，不能进行协整检验
- 如果 price2 的积分阶数无法确定，跳过协整检验

#### 5.3 步骤3：检查两个序列是否同阶单整

```python
# 步骤3: 检查两个序列是否同阶单整
if price1_order != price2_order:
    print(f"积分阶数不同，不能协整")
    return results
```

**关键点：**
- 两个序列必须同阶单整才能进行协整检验
- 如果积分阶数不同，不能协整

#### 5.4 步骤4：确保两个序列都是I(1)

```python
# 步骤4: 只有当两个序列都是I(1)时，才进行协整检验
if price1_order != 1:
    print(f"当前只支持I(1)序列的协整检验")
    return results
```

**关键点：**
- 当前实现只支持I(1)序列的协整检验
- 如果两个序列都是I(2)或其他，跳过协整检验

#### 5.5 步骤5：计算最优对冲比率（OLS回归）

```python
# 步骤5: 计算最优对冲比率（OLS回归）
print(f"\n--- 步骤3: 计算最优对冲比率（OLS回归） ---")
hedge_ratio = calculate_hedge_ratio(price1, price2)
```

**使用函数：** `calculate_hedge_ratio()`（第71-107行）

**执行逻辑：**
```python
def calculate_hedge_ratio(price1, price2):
    # 1. 确保两个序列长度一致
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]
    
    # 2. 使用OLS回归计算对冲比率
    X = price2_aligned.values.reshape(-1, 1)
    y = price1_aligned.values
    X_with_const = add_constant(X)
    
    # 3. 执行回归: price1 = α + β × price2
    model = OLS(y, X_with_const).fit()
    hedge_ratio = model.params[1]  # 斜率系数β
    
    return hedge_ratio
```

**关键点：**
- 使用OLS回归计算两个价格序列的关系
- 返回斜率系数作为对冲比率
- 对冲比率反映两个币种之间的协整关系

#### 5.6 步骤6：计算原序列的价差

```python
# 步骤6: 计算原序列的价差（残差）
print(f"\n--- 步骤4: 计算原序列价差（残差） ---")
min_length = min(len(price1), len(price2))
price1_aligned = price1.iloc[:min_length]
price2_aligned = price2.iloc[:min_length]

spread = price1_aligned - hedge_ratio * price2_aligned
```

**公式：**
```
spread = price1 - hedge_ratio × price2
```

**关键点：**
- 使用原始价格序列，不是差分
- 价差 = price1 - 对冲比率 × price2
- 这是协整检验的核心：检验这个价差是否平稳

#### 5.7 步骤7：检验原价差的平稳性（协整检验的关键步骤）

```python
# 步骤7: 检验原价差的平稳性（协整检验）
print(f"\n--- 步骤5: 检验原价差的平稳性（协整检验） ---")
spread_adf = advanced_adf_test(spread)
```

**使用函数：** `advanced_adf_test()`（第109-158行）

**执行逻辑：**
```python
def advanced_adf_test(series, max_lags=None, verbose=True):
    # 1. 执行ADF检验（使用AIC准则自动选择最优滞后阶数）
    adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')
    
    # 2. 提取检验结果
    adf_statistic = adf_result[0]  # ADF统计量
    p_value = adf_result[1]        # P值
    critical_values = adf_result[4]  # 临界值
    used_lag = adf_result[2]      # 使用的滞后阶数
    
    # 3. 判断是否平稳（P值 < 0.05 认为平稳）
    is_stationary = p_value < 0.05
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'used_lag': used_lag,
        'is_stationary': is_stationary
    }
```

**关键点：**
- **这是协整检验的核心步骤**：检验价差是否平稳
- 使用AIC准则自动选择最优滞后阶数
- 判断标准：P值 < 0.05 认为序列平稳
- 如果价差平稳（I(0)），协整关系成立

**为什么需要 advanced_adf_test() 函数？**

虽然之前计算了协整关系（通过OLS回归得到对冲比率），但**协整检验的核心就是检验价差的平稳性**。`advanced_adf_test()` 就是用来做这个检验的工具。

**在协整检验中的使用位置：**

1. **在 determine_integration_order() 中**（第172、181、190行）：
   - 检验原序列是否平稳
   - 检验一阶差分是否平稳
   - 检验二阶差分是否平稳

2. **在 enhanced_cointegration_test() 中**（第300行）：
   - 检验价差的平稳性（这是协整检验的关键步骤）
   - 如果价差平稳（I(0)），协整关系成立

#### 5.8 步骤8：判断协整检验结果

```python
if spread_adf and spread_adf['is_stationary']:
    # 原价差平稳，协整关系成立！
    results['cointegration_found'] = True
    print(f"\n✓ 协整检验通过！{symbol1} 和 {symbol2} 存在协整关系")
    print(f"  价差是平稳的（I(0)），ADF P值: {spread_adf['p_value']:.6f}")
else:
    print(f"\n✗ 协整检验未通过")
    print(f"  价差不平稳，ADF P值: {spread_adf['p_value']:.6f if spread_adf else 'N/A'}")
```

**关键点：**
- 如果价差平稳，协整关系成立
- 如果价差不平稳，协整关系不成立

---

### 第六步：显示候选币对 - display_candidates_for_selection() 函数

**代码位置：** 第456-526行

**功能：** 显示所有找到的协整对候选，供用户手工选择

**执行流程：**
```python
def display_candidates_for_selection(candidates):
    # 1. 显示所有候选币对
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. 币对: {candidate['pair_name']}")
        print(f"   积分阶数: {candidate['symbol1']}=I({price1_order}), {candidate['symbol2']}=I({price2_order})")
        print(f"   对冲比率: {hedge_ratio:.6f}")
        print(f"   价差ADF P值: {adf_p:.6f}")
    
    # 2. 用户输入选择的币对序号
    selection_input = input("请输入选择的币对序号: ")
    
    # 3. 返回用户选择的币对列表
    return valid_selection
```

**关键点：**
- 显示所有协整对候选的详细信息
- 用户可以选择一个或多个币对（用逗号分隔）
- 返回用户选择的币对列表

---

### 第七步：配置交易参数 - configure_trading_parameters() 函数

**代码位置：** 第366-454行

**功能：** 配置交易策略参数

**可配置参数：**
- `lookback_period`: 回看期（默认60）
- `z_threshold`: Z-score开仓阈值（默认1.5）
- `z_exit_threshold`: Z-score平仓阈值（默认0.6）
- `take_profit_pct`: 止盈百分比（默认15%）
- `stop_loss_pct`: 止损百分比（默认8%）
- `max_holding_hours`: 最大持仓时间（默认168小时）

**执行流程：**
```python
def configure_trading_parameters():
    # 1. 显示默认参数
    print("当前默认参数:")
    print(f"  1. 回看期: {default_params['lookback_period']}")
    # ... 其他参数
    
    # 2. 询问是否修改参数
    modify_choice = input("是否要修改参数？(y/n): ")
    
    # 3. 如果选择修改，逐个输入新参数
    if modify_choice == 'y':
        # 用户输入新参数值
        # ...
    
    # 4. 返回参数字典
    return default_params
```

---

### 第八步：执行交易回测 - AdvancedCointegrationTrading.backtest_cointegration_trading() 方法

**代码位置：** 第885-1013行

**功能：** 执行完整的交易回测

**执行流程：**

#### 8.1 初始化

```python
def backtest_cointegration_trading(self, data, selected_pairs, initial_capital=10000):
    # 1. 初始化资金
    capital = initial_capital
    
    # 2. 获取所有时间点
    all_timestamps = sorted(list(set(...)))
```

#### 8.2 回测循环

```python
# 3. 遍历每个时间点
for i, timestamp in enumerate(all_timestamps):
    # 3.1 获取当前价格
    current_prices = {...}
    
    # 3.2 对每个选择的币对
    for pair_info in selected_pairs:
        # 3.2.1 计算当前价差
        current_spread = self.calculate_current_spread(...)
        
        # 3.2.2 获取历史价差数据
        historical_spreads = [...]
        
        # 3.2.3 计算Z-score
        current_z_score = self.calculate_z_score(current_spread, historical_spreads)
        
        # 3.2.4 检查平仓条件
        if pair_info['pair_name'] in self.positions:
            should_close, close_reason = self.check_exit_conditions(...)
            if should_close:
                trade = self.close_position(...)
                capital += trade['pnl']
        
        # 3.2.5 检查开仓条件
        elif len(self.positions) == 0:
            signal = self.generate_trading_signal(current_z_score)
            if signal['action'] != 'HOLD':
                position = self.execute_trade(...)
    
    # 3.3 记录资金曲线
    results['capital_curve'].append({
        'timestamp': timestamp,
        'capital': capital,
        'positions_count': len(self.positions)
    })
```

#### 8.3 计算最终结果

```python
# 4. 计算最终结果
final_return = (capital - initial_capital) / initial_capital * 100
risk_metrics = self.calculate_risk_metrics(results['capital_curve'])

# 5. 绘制收益率曲线图
self.plot_equity_curve(results['capital_curve'])
```

---

## 核心函数详解

### 1. advanced_adf_test() 函数

**代码位置：** 第109-158行

**功能：** 执行增强的ADF（Augmented Dickey-Fuller）检验，判断时间序列是否平稳

**为什么需要这个函数？**

虽然之前计算了协整关系（通过OLS回归得到对冲比率），但**协整检验的核心就是检验价差的平稳性**。`advanced_adf_test()` 就是用来做这个检验的工具。

**在协整检验中的作用：**

1. **在 determine_integration_order() 中**：
   - 检验原序列是否平稳
   - 检验一阶差分是否平稳
   - 检验二阶差分是否平稳

2. **在 enhanced_cointegration_test() 中**：
   - 检验价差的平稳性（这是协整检验的关键步骤）
   - 如果价差平稳（I(0)），协整关系成立

**执行逻辑：**
```python
def advanced_adf_test(series, max_lags=None, verbose=True):
    # 1. 执行ADF检验（使用AIC准则自动选择最优滞后阶数）
    adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')
    
    # 2. 提取检验结果
    adf_statistic = adf_result[0]  # ADF统计量
    p_value = adf_result[1]        # P值
    critical_values = adf_result[4]  # 临界值
    used_lag = adf_result[2]      # 使用的滞后阶数
    
    # 3. 判断是否平稳（P值 < 0.05 认为平稳）
    is_stationary = p_value < 0.05
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'used_lag': used_lag,
        'is_stationary': is_stationary
    }
```

**关键点：**
- 使用AIC准则自动选择最优滞后阶数
- 判断标准：P值 < 0.05 认为序列平稳
- 返回完整的检验统计信息

---

### 2. determine_integration_order() 函数

**代码位置：** 第160-194行

**功能：** 确定时间序列的积分阶数（I(0), I(1), I(2)）

**执行逻辑：**
```python
def determine_integration_order(series, max_order=2):
    # 1. 检验原序列（使用 advanced_adf_test）
    adf_result = advanced_adf_test(series, verbose=False)
    if adf_result and adf_result['is_stationary']:
        return 0  # I(0) - 平稳序列
    
    # 2. 检验一阶差分（使用 advanced_adf_test）
    diff1 = series.diff().dropna()
    adf_result = advanced_adf_test(diff1, verbose=False)
    if adf_result and adf_result['is_stationary']:
        return 1  # I(1) - 一阶差分后平稳
    
    # 3. 检验二阶差分（使用 advanced_adf_test）
    diff2 = series.diff().diff().dropna()
    adf_result = advanced_adf_test(diff2, verbose=False)
    if adf_result and adf_result['is_stationary']:
        return 2  # I(2) - 二阶差分后平稳
    
    return None  # 无法确定
```

**关键点：**
- 使用 `advanced_adf_test()` 检验平稳性
- 依次检验原序列、一阶差分、二阶差分
- 返回积分阶数（0=I(0), 1=I(1), 2=I(2), None=无法确定）

---

### 3. calculate_hedge_ratio() 函数

**代码位置：** 第71-107行

**功能：** 使用OLS回归计算对冲比率

**执行逻辑：**
```python
def calculate_hedge_ratio(price1, price2):
    # 1. 确保两个序列长度一致
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]
    
    # 2. 使用OLS回归计算对冲比率
    X = price2_aligned.values.reshape(-1, 1)
    y = price1_aligned.values
    X_with_const = add_constant(X)
    
    # 3. 执行回归: price1 = α + β × price2
    model = OLS(y, X_with_const).fit()
    hedge_ratio = model.params[1]  # 斜率系数β
    
    return hedge_ratio
```

**关键点：**
- 使用OLS回归计算两个价格序列的关系
- 返回斜率系数作为对冲比率
- 对冲比率反映两个币种之间的协整关系

---

### 4. enhanced_cointegration_test() 函数

**代码位置：** 第198-317行

**功能：** 执行完整的Engle-Granger两阶段协整检验

**完整流程：**
1. 步骤1：检验 price1 的积分阶数（使用 `determine_integration_order()`）
2. 步骤2：检验 price2 的积分阶数（使用 `determine_integration_order()`）
3. 步骤3：检查两个序列是否同阶单整
4. 步骤4：确保两个序列都是I(1)
5. 步骤5：计算最优对冲比率（使用 `calculate_hedge_ratio()`）
6. 步骤6：计算原序列的价差（`spread = price1 - hedge_ratio × price2`）
7. 步骤7：检验原价差的平稳性（使用 `advanced_adf_test()`）← **关键步骤**
8. 步骤8：判断协整检验结果

---

### 5. AdvancedCointegrationTrading 类的主要方法

#### 5.1 calculate_current_spread() 方法

**代码位置：** 第555-557行

**功能：** 计算当前价差

**公式：**
```python
def calculate_current_spread(self, price1, price2, hedge_ratio):
    return price1 - hedge_ratio * price2
```

#### 5.2 calculate_z_score() 方法

**代码位置：** 第559-570行

**功能：** 计算当前Z-score

**公式：**
```python
def calculate_z_score(self, current_spread, historical_spreads):
    spread_mean = np.mean(historical_spreads)
    spread_std = np.std(historical_spreads)
    return (current_spread - spread_mean) / spread_std
```

#### 5.3 generate_trading_signal() 方法

**代码位置：** 第572-591行

**功能：** 根据Z-score生成交易信号

**规则：**
- Z-score > z_threshold：做空价差（SHORT_LONG）
- Z-score < -z_threshold：做多价差（LONG_SHORT）
- 其他情况：观望（HOLD）

#### 5.4 execute_trade() 方法

**代码位置：** 第593-656行

**功能：** 执行交易

**开仓数量（Beta中性方法）：**
- SHORT_LONG（做空价差）：
  - symbol1_size = -1.0（做空）
  - symbol2_size = hedge_ratio（做多）
- LONG_SHORT（做多价差）：
  - symbol1_size = 1.0（做多）
  - symbol2_size = -hedge_ratio（做空）

**注意：** 当前实现使用固定的数量（1.0和hedge_ratio），不根据可用资金计算。

#### 5.5 check_exit_conditions() 方法

**代码位置：** 第658-705行

**功能：** 检查平仓条件

**平仓条件：**
1. Z-score回归到均值附近（|Z-score| < z_exit_threshold）
2. 持仓时间过长（超过max_holding_hours）
3. 止盈触发（盈利超过take_profit_pct）
4. 止损触发（亏损超过stop_loss_pct）

#### 5.6 close_position() 方法

**代码位置：** 第707-757行

**功能：** 平仓

**平仓数量：**
- symbol1_size = -position['symbol1_size']（反向操作）
- symbol2_size = -position['symbol2_size']（反向操作）

#### 5.7 calculate_risk_metrics() 方法

**代码位置：** 第759-812行

**功能：** 计算风险指标

**计算的指标：**
- 最大回撤（Max Drawdown）
- 最大回撤百分比
- 盈亏比（Profit/Loss Ratio）
- 夏普比率（Sharpe Ratio）
- 平均盈利/亏损

#### 5.8 plot_equity_curve() 方法

**代码位置：** 第814-883行

**功能：** 绘制收益率曲线图

**输出：**
- 资金曲线图
- 收益率曲线图

---

## 程序完整执行流程图

```
main()
  │
  ├─> test_advanced_cointegration_trading()
  │     │
  │     ├─> 1. load_csv_data() - 加载数据
  │     │
  │     ├─> 2. enhanced_find_cointegrated_pairs() - 寻找协整对
  │     │       │
  │     │       └─> enhanced_cointegration_test() - 完整EG两阶段协整检验
  │     │             │
  │     │             ├─> determine_integration_order() - 确定积分阶数
  │     │             │     │
  │     │             │     └─> advanced_adf_test() - 检验平稳性
  │     │             │
  │     │             ├─> calculate_hedge_ratio() - 计算对冲比率
  │     │             │
  │     │             └─> advanced_adf_test() - 检验价差平稳性（关键步骤）
  │     │
  │     ├─> 3. display_candidates_for_selection() - 显示并选择协整对
  │     │
  │     ├─> 4. configure_trading_parameters() - 配置交易参数
  │     │
  │     └─> 5. AdvancedCointegrationTrading.backtest_cointegration_trading() - 执行回测
  │             │
  │             ├─> calculate_current_spread() - 计算当前价差
  │             │
  │             ├─> calculate_z_score() - 计算Z-score
  │             │
  │             ├─> generate_trading_signal() - 生成交易信号
  │             │
  │             ├─> execute_trade() - 执行交易
  │             │
  │             ├─> check_exit_conditions() - 检查平仓条件
  │             │
  │             ├─> close_position() - 平仓
  │             │
  │             ├─> calculate_risk_metrics() - 计算风险指标
  │             │
  │             └─> plot_equity_curve() - 绘制收益率曲线
```

---

## 关键概念说明

### 1. 协整检验 vs ADF检验

- **协整检验**：检验两个非平稳序列是否存在长期均衡关系
- **ADF检验**：检验单个时间序列是否平稳
- **关系**：协整检验的核心就是检验价差的平稳性，使用ADF检验来实现

### 2. EG两阶段协整检验流程

**阶段一：确定积分阶数**
- 检验 price1 的积分阶数（使用 `advanced_adf_test()`）
- 检验 price2 的积分阶数（使用 `advanced_adf_test()`）
- 确保两个序列都是I(1)

**阶段二：协整检验**
- 计算对冲比率（OLS回归）
- 计算原序列价差：`spread = price1 - β × price2`
- 检验价差的平稳性（使用 `advanced_adf_test()`）← **关键步骤**
- 如果价差平稳（I(0)），协整关系成立

### 3. advanced_adf_test() 在协整检验中的关键作用

**使用位置：**
1. **在 determine_integration_order() 中**（第172、181、190行）：
   - 检验原序列是否平稳
   - 检验一阶差分是否平稳
   - 检验二阶差分是否平稳

2. **在 enhanced_cointegration_test() 中**（第300行）：
   - 检验价差的平稳性（这是协整检验的关键步骤）
   - 如果价差平稳（I(0)），协整关系成立

**为什么需要这个函数？**

虽然之前计算了协整关系（通过OLS回归得到对冲比率），但**协整检验的核心就是检验价差的平稳性**。`advanced_adf_test()` 就是用来做这个检验的工具。

---

## 使用示例

### 运行程序

```bash
python cointegration_test_original.py
```

### 交互流程

1. **输入CSV文件路径**（或使用默认路径）
2. **等待协整检验完成**（自动寻找所有协整对）
3. **选择要交易的币对**（从候选列表中选择，可以多选）
4. **配置交易参数**（或使用默认参数）
5. **查看回测结果**（包括收益率、风险指标、交易记录等）
6. **选择是否继续测试其他币对**

---

## 注意事项

1. **数据要求**：
   - CSV文件必须包含 `symbol`、`timestamp`、`close` 列
   - 时间戳格式需要能被pandas识别
   - 需要足够的数据点（建议至少500条）

2. **协整检验**：
   - 需要至少50个数据点才能进行积分阶数检验
   - 两个序列必须都是I(1)才能进行协整检验
   - 协整关系可能随时间变化，需要定期重新检验

3. **交易参数**：
   - Z-score阈值需要根据市场情况调整
   - 止盈止损百分比需要根据币对波动性调整

4. **风险控制**：
   - 建议设置合理的止损比例
   - 注意最大持仓时间限制
   - **注意：当前实现使用固定的开仓数量（1.0和hedge_ratio），不根据可用资金计算，可能导致资金不足的问题**

---

**文档版本**: 2.0  
**更新日期**: 2024  
**适用文件**: `cointegration_test_original.py`
