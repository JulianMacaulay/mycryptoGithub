# cointegration_test_windows_ECM.py 使用说明文档

## 文件概述

这是一个高级版协整分析+交易流程完整测试文件，支持滚动窗口协整检验、参数优化和ECM误差修正模型。该文件完全复制自 `cointegration_test_windows_optimization_arima_garch.py`，并添加了ECM误差修正模型功能。

### 主要特性

1. **滚动窗口协整检验**：使用固定大小的滚动窗口进行协整检验，识别协整关系的时变特性
2. **价差类型选择**：支持原始价差和一阶差分价差
3. **多种Z-score计算策略**：
   - 传统方法（均值和标准差）
   - ARIMA-GARCH模型
   - ECM误差修正模型（推荐用于协整交易）
4. **参数优化功能**：
   - 网格搜索（粗粒度+细粒度分层搜索）
   - 贝叶斯优化
   - 随机搜索
   - 过拟合检测（参数稳定性测试）
5. **完整交易回测**：包含手续费、动态仓位、止盈止损等

---

## 运行流程（按执行顺序）

### 第一步：程序启动（main函数）

**位置**：`main()` 函数（3056-3091行）

**功能**：
- 显示程序标题和功能介绍
- 让用户选择运行模式
- 提示用户输入CSV文件路径

**用户交互**：
```
请选择运行模式:
  1. 普通回测模式
  2. 参数优化模式

请选择 (1/2): 

请输入CSV文件路径 (或按回车使用默认路径): 
```

**说明**：
- 如果用户直接回车，使用默认路径：`segment_2_data_ccxt_20251113_103652.csv`
- 根据用户选择调用不同的测试函数

---

## 模式一：普通回测模式

### 第二步：加载数据

**位置**：`test_rolling_window_cointegration_trading()` 函数（2095-2245行），第1步

**功能**：
- 调用 `load_csv_data()` 函数加载CSV文件
- 解析数据，按币对（symbol）分组
- 返回包含各币对价格序列的字典

**数据格式要求**：
- CSV文件需包含列：`timestamp`, `symbol`, `close`
- 或包含多个币对的数据

**输出示例**：
```
成功加载 5 个币对的数据
```

---

### 第三步：选择价差类型

**位置**：`test_rolling_window_cointegration_trading()` 函数，第2步

**功能**：
- 调用 `select_spread_type()` 函数（601-641行）
- 让用户选择使用原始价差还是一阶差分价差

**用户交互**：
```
请选择用于协整检验和回测的价差类型:
  0. 原始价差（原始价格计算的价差）
  1. 一阶差分价差（一阶差分价格计算的价差）

请选择 (0/1，默认0): 
```

**说明**：
- **原始价差**：`spread = price1 - β * price2`，对冲比率从原始价格计算
- **一阶差分价差**：`spread = diff1 - β * diff2`，对冲比率从一阶差分价格计算
- 选择会影响后续的协整检验和交易回测

---

### 第四步：配置滚动窗口参数

**位置**：`test_rolling_window_cointegration_trading()` 函数，第3步

**功能**：
- 调用 `configure_rolling_window_parameters()` 函数（644-688行）
- 配置滚动窗口的大小和步长

**参数**：
1. **窗口大小（window_size）**：默认1000
   - 每个窗口包含的数据条数
   - 窗口越大，检验结果越稳定，但可能错过短期协整关系
   
2. **步长（step_size）**：默认100
   - 每次移动窗口的数据条数
   - 步长越小，窗口重叠越多，检验更细致但计算量更大

**用户交互**：
```
是否要修改参数？(y/n): 
```

---

### 第五步：滚动窗口寻找协整对

**位置**：`test_rolling_window_cointegration_trading()` 函数，第4步

**功能**：
- 调用 `rolling_window_find_cointegrated_pairs()` 函数（547-598行）
- 对所有币对组合进行滚动窗口协整检验

#### 滚动窗口协整检验流程

**5.1 对每个币对组合进行滚动窗口检验**

**位置**：`rolling_window_cointegration_test()` 函数（439-537行）

**流程**：
1. 将数据分成多个滚动窗口
2. 对每个窗口执行完整的协整检验：
   - 检验两个序列的积分阶数
   - 计算对冲比率（根据价差类型选择原始价格或差分价格）
   - 计算价差（根据价差类型）
   - 对价差进行ADF平稳性检验
3. 汇总所有窗口的检验结果

**5.2 汇总结果**

**统计信息**：
- 总窗口数
- 协整窗口数（通过协整检验的窗口数）
- 协整比例 = 协整窗口数 / 总窗口数

**输出示例**：
```
找到 3 个币对的滚动窗口检验结果
```

---

### 第六步：显示并选择协整对

**位置**：`test_rolling_window_cointegration_trading()` 函数，第5步

**功能**：
- 调用 `display_rolling_window_candidates()` 函数（691-836行）
- 显示所有找到的协整对候选
- 让用户选择要使用的币对

**显示信息**：
- 币对名称
- 总窗口数和协整窗口数
- 协整比例
- 整体数据对冲比率（基于整个数据集重新计算）
- 最佳窗口的参考信息

**用户交互**：
```
请选择要使用的币对（输入序号，用逗号分隔，如: 1,3,5）
请输入选择的币对序号: 
```

**说明**：
- 只显示协整比例 >= 30% 的币对（可配置）
- 选择币对后，会使用整个数据集重新计算对冲比率和协整检验结果
- 最终使用的对冲比率是基于整个数据集计算的，而不是窗口内的

---

### 第七步：策略选择和回测循环

**位置**：`test_rolling_window_cointegration_trading()` 函数，第6步

**功能**：
- 支持多次测试不同的策略和参数
- 每次测试可以重新选择Z-score计算策略

**用户交互**：
```
是否继续测试？(y/n): 
```

---

### 第八步：选择Z-score计算策略

**位置**：`select_z_score_strategy()` 函数（839-975行）

**功能**：
- 让用户选择Z-score计算策略
- 支持三种策略：传统方法、ARIMA-GARCH、ECM

**策略选项**：

**选项1：传统方法（TraditionalZScoreStrategy）**
- 使用均值和标准差计算Z-score
- 公式：`z_score = (current_spread - mean) / std`
- 无需额外参数

**选项2：ARIMA-GARCH模型（ArimaGarchZScoreStrategy）**
- 使用ARIMA模型预测价差的均值
- 使用GARCH模型预测价差的波动率
- 需要配置ARIMA和GARCH的阶数

**选项3：ECM误差修正模型（EcmZScoreStrategy）** ⭐推荐
- 使用误差修正模型预测价差的均值回归
- 基于协整关系的长期均衡和短期偏离
- 需要配置误差修正项的滞后阶数和最小数据长度

**用户交互**：
```
请选择Z-score计算策略:
  1. 传统方法（均值和标准差）
  2. ARIMA-GARCH模型
  3. ECM误差修正模型（推荐用于协整交易）

请选择 (0-3): 
```

**ECM策略参数配置**：
```
配置ECM误差修正模型参数:
  直接回车使用默认值: 滞后阶数=1, 最小数据长度=30

误差修正项滞后阶数 (默认1): 
最小数据长度 (默认30): 
```

---

### 第九步：显示选择的币对详情

**位置**：`test_rolling_window_cointegration_trading()` 函数，第7步

**功能**：
- 显示已选择币对的详细信息
- 包括价差类型、对冲比率、协整比例等

---

### 第十步：配置交易参数

**位置**：`test_rolling_window_cointegration_trading()` 函数，第8步

**功能**：
- 调用 `configure_trading_parameters()` 函数（927-1106行）
- 配置交易策略的所有参数

**参数列表**：

1. **回看期（lookback_period）**：默认60
   - 用于计算Z-score的历史数据窗口大小

2. **Z-score开仓阈值（z_threshold）**：默认1.5
   - 当Z-score超过此值时开仓

3. **Z-score平仓阈值（z_exit_threshold）**：默认0.6
   - 当Z-score回归到此值以下时平仓

4. **止盈百分比（take_profit_pct）**：默认15%
   - 盈利达到此百分比时平仓

5. **止损百分比（stop_loss_pct）**：默认8%
   - 亏损达到此百分比时平仓

6. **最大持仓时间（max_holding_hours）**：默认168小时（7天）
   - 超过此时间强制平仓

7. **仓位比例（position_ratio）**：默认50%（0.5）
   - 使用多少比例的初始资金进行交易

8. **杠杆倍数（leverage）**：默认5倍
   - 可用资金 = 投入资金 × 杠杆

9. **交易手续费率（trading_fee_rate）**：默认0.0275%（0.000275）

**用户交互**：
```
是否要修改参数？(y/n): 
```

---

### 第十一步：执行交易回测

**位置**：`test_rolling_window_cointegration_trading()` 函数，第9步

**功能**：
- 创建交易策略实例
- 执行完整的回测流程
- 记录所有交易和资金曲线

#### 11.1 初始化策略

**位置**：`AdvancedCointegrationTrading.__init__()` 方法（1114-1172行）

**功能**：
- 保存所有交易参数
- 初始化Z-score计算策略对象
- 初始化持仓字典和交易记录列表

#### 11.2 回测主循环

**位置**：`backtest_cointegration_trading()` 方法（1765-2092行）

**流程**：

**11.2.1 遍历所有时间点**
- 获取当前时间点的所有币对价格

**11.2.2 根据价差类型计算价差**

**原始价差（diff_order=0）**：
```python
current_spread = price1 - hedge_ratio * price2
```

**一阶差分价差（diff_order=1）**：
```python
current_diff1 = price1[t] - price1[t-1]
current_diff2 = price2[t] - price2[t-1]
current_spread = current_diff1 - hedge_ratio * current_diff2
```

**11.2.3 计算Z-score**

**位置**：`calculate_z_score()` 方法（1223-1240行）

**策略调用**：
- 如果使用了策略对象（`self.z_score_strategy`），调用策略的 `calculate_z_score()` 方法
- 如果没有策略对象，使用向后兼容的内置方法

**传统方法**：
```python
z_score = (current_spread - mean) / std
```

**ARIMA-GARCH方法**：
- 使用ARIMA模型预测均值
- 使用GARCH模型预测波动率
- `z_score = (current_spread - predicted_mean) / predicted_volatility`

**ECM方法**（详见下方"ECM误差修正模型详解"部分）：
- 计算长期均值（均衡值）
- 计算误差修正项 ECM = spread_{t-1} - mean
- 使用OLS回归估计误差修正系数
- 预测价差的均值回归
- `z_score = (current_spread - predicted_mean) / historical_std`

**11.2.4 检查平仓条件**

**位置**：`check_exit_conditions()` 方法（1400-1465行）

**平仓条件**：
1. **Z-score回归**：`|z_score| < z_exit_threshold`
2. **持仓时间过长**：`holding_hours > max_holding_hours`
3. **止盈触发**：`net_pnl > 0 且 pnl_percentage > take_profit_pct`
4. **止损触发**：`net_pnl < 0 且 pnl_percentage < -stop_loss_pct`

**盈亏计算**（统一方法）：
```python
# 无论使用哪种价差类型，都使用相同方法
pnl_symbol1 = symbol1_size * (price1_current - price1_entry)
pnl_symbol2 = symbol2_size * (price2_current - price2_entry)
total_pnl = pnl_symbol1 + pnl_symbol2
net_pnl = total_pnl - close_fee  # 扣除平仓手续费
```

**11.2.5 检查开仓条件**

**位置**：`generate_trading_signal()` 方法（1302-1318行）

**开仓条件**：
- `z_score > z_threshold` → 做空价差（SHORT_LONG）
- `z_score < -z_threshold` → 做多价差（LONG_SHORT）

**执行开仓**：

**位置**：`execute_trade()` 方法（1323-1398行）

**仓位计算**：
```python
# 计算可用资金
available_capital = capital * leverage

# 基于Beta中性计算仓位
symbol1_size, symbol2_size = calculate_position_size_beta_neutral(
    available_capital, price1, price2, hedge_ratio, signal
)
```

**仓位计算公式**：
```python
capital_coefficient = price1 + hedge_ratio * price2
symbol1_size_abs = available_capital / capital_coefficient
symbol2_size_abs = hedge_ratio * symbol1_size_abs
```

**11.2.6 记录资金曲线**
- 每个时间点记录当前资金和持仓数量

#### 11.3 计算风险指标

**位置**：`calculate_risk_metrics()` 方法（1580-1680行）

**指标**：
1. **最大回撤（Max Drawdown）**
2. **最大回撤百分比（Max Drawdown %）**
3. **盈亏比（Profit/Loss Ratio）**
4. **夏普比率（Sharpe Ratio）**
5. **平均盈利/亏损**
6. **总交易次数、盈利交易数、亏损交易数**

#### 11.4 绘制收益率曲线图

**位置**：`plot_equity_curve()` 方法（1682-1763行）

**功能**：
- 绘制资金曲线图
- 绘制收益率曲线图
- 显示图表或保存为文件

---

### 第十二步：显示交易详情

**位置**：`test_rolling_window_cointegration_trading()` 函数，第9步

**功能**：
- 显示所有交易记录
- 按币对分组显示
- 显示每笔交易的关键信息

**输出示例**：
```
BTCUSDT/ETHUSDT 交易记录:
  2024-01-01 10:00:00: 开仓 BTCUSDT=50000.00, ETHUSDT=3000.00
  2024-01-02 14:30:00: 平仓 BTCUSDT=50100.00, ETHUSDT=2995.00, 盈亏: 150.25
```

---

### 第十三步：询问是否继续

**位置**：`test_rolling_window_cointegration_trading()` 函数

**功能**：
- 询问用户是否继续测试
- 如果继续，可以重新选择策略和参数

**用户交互**：
```
请选择下一步操作:
  1. 继续测试（重新选择策略）
  0. 退出程序

请选择 (0/1): 
```

---

## 模式二：参数优化模式

### 第二步：加载数据

**位置**：`test_parameter_optimization()` 函数（2876-3053行），第1步

**功能**：与普通回测模式相同

---

### 第三步：选择价差类型

**位置**：`test_parameter_optimization()` 函数，第2步

**功能**：与普通回测模式相同

---

### 第四步：配置滚动窗口参数

**位置**：`test_parameter_optimization()` 函数，第3步

**功能**：与普通回测模式相同

---

### 第五步：滚动窗口寻找协整对

**位置**：`test_parameter_optimization()` 函数，第4步

**功能**：与普通回测模式相同

---

### 第六步：选择协整对

**位置**：`test_parameter_optimization()` 函数，第5步

**功能**：与普通回测模式相同

---

### 第七步：选择Z-score计算策略

**位置**：`test_parameter_optimization()` 函数，第6步

**功能**：与普通回测模式相同

---

### 第八步：选择优化方法

**位置**：`test_parameter_optimization()` 函数，第7步

**功能**：
- 让用户选择参数优化方法

**优化方法**：

**方法1：网格搜索（Grid Search）**
- 粗粒度搜索：在较大的参数范围内搜索
- 细粒度搜索：在最佳参数附近进行精细搜索
- 优点：全面，能找到全局最优
- 缺点：计算量大

**方法2：随机搜索（Random Search）**
- 在参数空间中随机采样
- 优点：计算量相对较小
- 缺点：可能错过最优解

**方法3：贝叶斯优化（Bayesian Optimization）**
- 使用高斯过程模型指导搜索
- 优点：效率高，能找到较好的解
- 缺点：需要安装 scikit-optimize 库

**用户交互**：
```
请选择优化方法 (1/2/3): 
```

---

### 第九步：选择优化目标

**位置**：`test_parameter_optimization()` 函数，第8步

**功能**：
- 让用户选择参数优化的目标函数

**优化目标**：

**目标1：夏普比率（sharpe_ratio）**
- 衡量风险调整后的收益
- 公式：`sharpe = mean(returns) / std(returns) * sqrt(periods_per_year)`
- 推荐用于大多数情况

**目标2：总收益率（return）**
- 只考虑收益，不考虑风险
- 公式：`return = (final_capital - initial_capital) / initial_capital`

**目标3：收益率/回撤比（return_drawdown_ratio）**
- 考虑收益和最大回撤的平衡
- 公式：`ratio = total_return / max_drawdown_pct`

**用户交互**：
```
请选择优化目标 (1/2/3): 
```

---

### 第十步：显示选择的币对详情

**位置**：`test_parameter_optimization()` 函数，第9步

**功能**：显示已选择币对的详细信息

---

### 第十一步：创建优化器

**位置**：`test_parameter_optimization()` 函数，第10步

**功能**：
- 创建 `ParameterOptimizer` 实例（2250-2293行）
- 传入数据、币对、初始资金、优化目标、策略对象等

**参数搜索空间**（在 `ParameterOptimizer.__init__()` 中定义）：
- `lookback_period`: [30, 60, 90, 120]（粗粒度），步长10（细粒度）
- `z_threshold`: [1.0, 1.5, 2.0, 2.5, 3.0]（粗粒度），步长0.1（细粒度）
- `z_exit_threshold`: [0.3, 0.5, 0.7, 0.9]（粗粒度），步长0.1（细粒度）
- `take_profit_pct`: [0.05, 0.10, 0.15, 0.20, 0.25]（粗粒度），步长0.02（细粒度）
- `stop_loss_pct`: [0.05, 0.08, 0.10, 0.12, 0.15]（粗粒度），步长0.01（细粒度）

---

### 第十二步：执行优化

**位置**：`test_parameter_optimization()` 函数，第11步

**功能**：
- 调用优化器的 `optimize()` 方法
- 根据选择的优化方法执行参数搜索

#### 12.1 网格搜索流程

**位置**：`ParameterOptimizer.grid_search()` 方法

**流程**：
1. **粗粒度搜索**：在较大的参数范围内搜索
2. **细粒度搜索**：在最佳参数附近进行精细搜索
3. **稳定性测试**：对最佳参数进行扰动，检查结果稳定性（过拟合检测）

#### 12.2 随机搜索流程

**位置**：`ParameterOptimizer.random_search()` 方法

**流程**：
1. 在参数空间中随机采样指定次数
2. 评估每个参数组合
3. 返回最佳参数

#### 12.3 贝叶斯优化流程

**位置**：`ParameterOptimizer.bayesian_optimization()` 方法

**流程**：
1. 使用高斯过程模型建模目标函数
2. 使用采集函数指导下一步搜索
3. 迭代优化，找到最佳参数

**用户交互**（随机搜索和贝叶斯优化）：
```
请输入随机搜索迭代次数 (默认100): 
请输入贝叶斯优化评估次数 (默认50): 
```

---

### 第十三步：显示优化结果

**位置**：`test_parameter_optimization()` 函数，第12步

**功能**：
- 显示最佳参数组合
- 显示最佳得分和风险指标
- 显示稳定性测试结果
- 显示前10个最佳参数组合

**输出示例**：
```
最佳参数:
  lookback_period: 60
  z_threshold: 1.5
  z_exit_threshold: 0.6
  take_profit_pct: 0.15
  stop_loss_pct: 0.08
  使用的策略: ECM误差修正模型 (滞后阶数=1)
  价差类型: 一阶差分价差

最佳得分: 2.3456
  总收益率: 25.50%
  夏普比率: 2.3456
  最大回撤: 8.50%
  盈亏比: 1.85
  总交易次数: 45

稳定性测试:
  稳定性: 良好
  变异系数: 0.123
  得分下降比例: 0.045
```

---

### 第十四步：导出优化结果

**位置**：`test_parameter_optimization()` 函数，第12步

**功能**：
- 调用 `ParameterOptimizer.export_results()` 方法
- 将所有评估结果导出到CSV文件

**输出文件**：
- 文件名格式：`optimization_results_YYYYMMDD_HHMMSS.csv`
- 包含所有参数组合及其评估结果

---

## 关键函数说明

### 1. load_csv_data() - 加载CSV数据

**位置**：82-130行

**功能**：从CSV文件加载价格数据

**返回**：字典，键为币对名称，值为价格序列（pd.Series）

---

### 2. select_spread_type() - 选择价差类型

**位置**：601-641行

**功能**：让用户选择使用原始价差还是一阶差分价差

**返回**：`diff_order`（0=原始价差，1=一阶差分价差）

---

### 3. enhanced_cointegration_test() - 协整检验

**位置**：259-433行

**功能**：执行完整的Engle-Granger两阶段协整检验

**流程**：
1. 检验两个序列的积分阶数
2. 检查是否同阶单整（当前只支持I(1)）
3. 根据价差类型计算对冲比率：
   - 原始价差：使用原始价格计算OLS
   - 一阶差分价差：使用一阶差分价格计算OLS
4. 计算价差
5. 对价差进行ADF平稳性检验

---

### 4. rolling_window_cointegration_test() - 滚动窗口协整检验

**位置**：439-537行

**功能**：对币对进行滚动窗口协整检验

**流程**：
1. 将数据分成多个滚动窗口
2. 对每个窗口执行协整检验
3. 汇总所有窗口的检验结果

---

### 5. select_z_score_strategy() - 选择Z-score策略

**位置**：839-975行

**功能**：让用户选择Z-score计算策略

**支持的策略**：
- 传统方法
- ARIMA-GARCH模型
- ECM误差修正模型

---

### 6. AdvancedCointegrationTrading类 - 交易策略类

**主要方法**：

- **`calculate_z_score()`**（1223-1240行）
  - 计算Z-score（调用策略对象的方法）

- **`generate_trading_signal()`**（1302-1318行）
  - 根据Z-score生成交易信号

- **`execute_trade()`**（1323-1398行）
  - 执行开仓操作

- **`check_exit_conditions()`**（1400-1465行）
  - 检查平仓条件

- **`close_position()`**（1467-1567行）
  - 执行平仓操作

- **`backtest_cointegration_trading()`**（1765-2092行）
  - 执行完整回测

---

### 7. ParameterOptimizer类 - 参数优化器

**主要方法**：

- **`evaluate_params()`**（2318-2380行）
  - 评估参数组合

- **`grid_search()`**（2382-2450行）
  - 网格搜索优化

- **`random_search()`**（2452-2480行）
  - 随机搜索优化

- **`bayesian_optimization()`**（2482-2550行）
  - 贝叶斯优化

- **`export_results()`**（2838-2873行）
  - 导出优化结果到CSV

---

## 重要概念说明

### 1. 价差类型（diff_order）

- **0 = 原始价差**：`spread = price1 - β * price2`
- **1 = 一阶差分价差**：`spread = diff1 - β * diff2`，其中 `diff = price[t] - price[t-1]`

**对冲比率计算**：
- 原始价差：使用原始价格进行OLS回归
- 一阶差分价差：使用一阶差分价格进行OLS回归

### 2. 滚动窗口协整检验

**目的**：
- 识别协整关系的时变特性
- 评估协整关系的稳定性

**流程**：
1. 将数据分成多个滚动窗口
2. 对每个窗口进行协整检验
3. 统计协整窗口的比例

**协整比例**：
- 协整比例 = 协整窗口数 / 总窗口数
- 协整比例越高，说明协整关系越稳定

### 3. Z-score计算策略

**传统方法**：
- 使用历史价差的均值和标准差
- 简单快速，适合大多数情况

**ARIMA-GARCH方法**：
- 使用ARIMA模型预测均值
- 使用GARCH模型预测波动率
- 适合波动率变化较大的情况

**ECM方法**（推荐用于协整交易）：
- 使用误差修正模型预测均值回归
- 基于协整关系的长期均衡和短期偏离
- 特别适合协整交易，因为ECM模型专门描述协整关系的均值回归特性
- **详细原理和公式见下方"ECM误差修正模型详解"部分**

### 4. 盈亏计算（统一方法）

**无论使用哪种价差类型，盈亏计算都相同**：

```python
pnl_symbol1 = symbol1_size * (price1_current - price1_entry)
pnl_symbol2 = symbol2_size * (price2_current - price2_entry)
total_pnl = pnl_symbol1 + pnl_symbol2
net_pnl = total_pnl - close_fee
```

**原因**：
- 实际交易的是价格本身，不是价差
- 盈亏来自价格变化，而不是价差变化
- 价差（原始或差分）只用于信号生成（Z-score计算）

### 5. 参数优化

**优化参数**：
- 回看期（lookback_period）
- Z-score开仓阈值（z_threshold）
- Z-score平仓阈值（z_exit_threshold）
- 止盈百分比（take_profit_pct）
- 止损百分比（stop_loss_pct）

**注意**：
- Z-score计算策略不在优化范围内，需要单独选择
- 价差类型不在优化范围内，需要在优化前选择

**过拟合检测**：
- 对最佳参数进行扰动
- 检查结果稳定性
- 避免选择不稳定的参数组合

---

## 使用示例

### 普通回测模式完整流程

```bash
python cointegration_test_windows_ECM.py
```

**交互流程**：

1. 选择运行模式：`1`（普通回测模式）
2. 输入CSV文件路径（或回车使用默认）
3. 选择价差类型：`0`（原始价差）或 `1`（一阶差分价差）
4. 配置滚动窗口参数（或使用默认）
5. 等待滚动窗口协整检验完成
6. 选择要测试的币对（如：1,3,5）
7. 选择Z-score计算策略：
   - `1`：传统方法
   - `2`：ARIMA-GARCH模型
   - `3`：ECM误差修正模型（推荐）
8. 如果选择ECM，配置ECM参数（或使用默认）
9. 配置交易参数（或使用默认）
10. 查看回测结果和图表
11. 选择是否继续测试其他策略

### 参数优化模式完整流程

```bash
python cointegration_test_windows_ECM.py
```

**交互流程**：

1. 选择运行模式：`2`（参数优化模式）
2. 输入CSV文件路径（或回车使用默认）
3. 选择价差类型：`0` 或 `1`
4. 配置滚动窗口参数
5. 等待滚动窗口协整检验完成
6. 选择要优化的币对
7. 选择Z-score计算策略
8. 选择优化方法：`1`（网格搜索）、`2`（随机搜索）、`3`（贝叶斯优化）
9. 选择优化目标：`1`（夏普比率）、`2`（总收益率）、`3`（收益率/回撤比）
10. 等待优化完成（可能需要较长时间）
11. 查看优化结果
12. 查看导出的CSV文件

---

## 注意事项

1. **价差类型选择**：选择原始价差还是一阶差分价差会影响对冲比率的计算和后续交易
2. **策略选择**：ECM误差修正模型特别适合协整交易，推荐使用
3. **滚动窗口参数**：窗口大小和步长会影响协整检验的结果和计算时间
4. **参数优化**：优化过程可能需要较长时间，建议先用较小的参数空间测试
5. **过拟合检测**：优化结果包含稳定性测试，注意检查稳定性指标
6. **单边持仓**：同一时间只能持有一个币对的仓位

---

## 输出文件

- **收益率曲线图**：自动显示或保存为图片文件
- **优化结果CSV**：`optimization_results_YYYYMMDD_HHMMSS.csv`

---

---

## ECM误差修正模型详解

### 一、ECM模型理论基础

#### 1.1 协整关系与误差修正

**协整关系**：
如果两个非平稳时间序列 `X_t` 和 `Y_t` 都是 I(1)（一阶单整），但它们的线性组合 `Z_t = X_t - βY_t` 是平稳的（I(0)），则称 `X_t` 和 `Y_t` 是协整的。

**长期均衡关系**：
对于协整的序列，存在长期均衡关系：
```
X_t = α + βY_t + ε_t
```
其中 `ε_t` 是平稳的误差项，表示短期偏离。

**误差修正机制**：
当价差偏离长期均衡时，存在一种"误差修正"机制，使价差回归到均衡值。这就是误差修正模型（ECM）的核心思想。

#### 1.2 为什么需要ECM模型？

在协整交易中：
- **传统方法**：假设价差的均值是常数，使用历史均值计算Z-score
- **问题**：价差可能偏离均值，传统方法无法捕捉均值回归的动态过程
- **ECM模型**：专门描述协整关系的均值回归特性，能够预测价差如何回归到均衡值

#### 1.3 ECM模型的基本形式

**一阶ECM模型**（最常用）：
```
Δspread_t = α + β*ECM_{t-1} + ε_t
```

其中：
- `Δspread_t = spread_t - spread_{t-1}`：价差的一阶差分（变化量）
- `ECM_{t-1} = spread_{t-1} - μ`：误差修正项（前一期的偏离）
- `μ`：长期均值（均衡值）
- `β`：误差修正系数（通常为负，表示均值回归）
- `α`：常数项
- `ε_t`：误差项

**误差修正系数的含义**：
- `β < 0`：表示均值回归，偏离越大，回归速度越快
- `β = 0`：表示没有均值回归
- `β > 0`：表示发散（不常见，可能表示协整关系破裂）

---

### 二、ECM模型的数学推导

#### 2.1 从协整关系推导ECM模型

**协整关系**：
```
spread_t = μ + ε_t
```
其中 `ε_t` 是平稳的误差项。

**误差修正项**：
```
ECM_t = spread_t - μ = ε_t
```

**一阶差分**：
```
Δspread_t = spread_t - spread_{t-1}
          = (μ + ε_t) - (μ + ε_{t-1})
          = ε_t - ε_{t-1}
```

**误差修正模型**：
由于 `ε_t` 是平稳的，可以表示为：
```
ε_t = ρ*ε_{t-1} + u_t
```
其中 `|ρ| < 1`（平稳性要求），`u_t` 是白噪声。

因此：
```
Δspread_t = ε_t - ε_{t-1}
          = (ρ*ε_{t-1} + u_t) - ε_{t-1}
          = (ρ - 1)*ε_{t-1} + u_t
          = (ρ - 1)*ECM_{t-1} + u_t
```

**ECM模型**：
```
Δspread_t = α + β*ECM_{t-1} + ε_t
```
其中 `β = ρ - 1 < 0`（因为 `|ρ| < 1`），表示均值回归。

#### 2.2 ECM模型的参数估计

**使用OLS回归估计参数**：

**回归方程**：
```
Δspread_t = α + β*ECM_{t-1} + ε_t
```

**数据准备**：
- 因变量：`y = [Δspread_1, Δspread_2, ..., Δspread_T]`
- 自变量：`X = [1, ECM_0], [1, ECM_1], ..., [1, ECM_{T-1}]`

**OLS估计**：
```
[α, β]' = (X'X)^{-1}X'y
```

**参数解释**：
- `α`：常数项，表示价差的平均变化
- `β`：误差修正系数，表示均值回归的速度
  - `β` 越接近 -1，回归速度越快
  - `β` 越接近 0，回归速度越慢

#### 2.3 使用ECM模型预测

**预测价差变化**：
```
Δspread_t_predicted = α + β*ECM_{t-1}
```

**预测价差均值**：
```
spread_t_predicted = spread_{t-1} + Δspread_t_predicted
                   = spread_{t-1} + α + β*ECM_{t-1}
                   = spread_{t-1} + α + β*(spread_{t-1} - μ)
                   = spread_{t-1} + α + β*spread_{t-1} - β*μ
                   = (1 + β)*spread_{t-1} + α - β*μ
```

**简化形式**（忽略常数项，关注均值回归）：
```
spread_t_predicted ≈ μ - β*ECM_{t-1}
                   = μ - β*(spread_{t-1} - μ)
                   = μ - β*spread_{t-1} + β*μ
                   = (1 + β)*μ - β*spread_{t-1}
```

**当 `β < 0` 时**：
- 如果 `spread_{t-1} > μ`（价差高于均值），则 `ECM_{t-1} > 0`，`β*ECM_{t-1} < 0`
- 因此 `spread_t_predicted < spread_{t-1}`，价差下降，回归均值
- 如果 `spread_{t-1} < μ`（价差低于均值），则 `ECM_{t-1} < 0`，`β*ECM_{t-1} > 0`
- 因此 `spread_t_predicted > spread_{t-1}`，价差上升，回归均值

---

### 三、代码实现详解

**代码位置**：`strategies/ecm_zscore_strategy.py`

#### 3.1 初始化

```23:43:strategies/ecm_zscore_strategy.py
    def __init__(self, ecm_lag: int = 1, min_data_length: int = 30, **kwargs):
        """
        初始化ECM策略
        
        Args:
            ecm_lag: 误差修正项的滞后阶数（默认1，即使用前一期）
            min_data_length: 最小数据长度要求
            **kwargs: 其他策略参数
        """
        super().__init__(ecm_lag=ecm_lag, min_data_length=min_data_length, **kwargs)
        self.name = "ECM误差修正模型"
        self.ecm_lag = ecm_lag
        self.min_data_length = min_data_length
        
        # 检查库是否可用
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels库不可用，无法使用ECM策略")
        
        # 存储模型参数缓存
        self._ecm_params_cache = {}
        self._max_cache_size = 10
```

**参数说明**：
- **ecm_lag**：误差修正项的滞后阶数，默认1（使用前一期）
- **min_data_length**：最小数据长度，默认30
- **缓存机制**：缓存ECM系数，避免重复计算

#### 3.2 计算长期均值

```79:80:strategies/ecm_zscore_strategy.py
            # 计算长期均值（均衡值）
            long_term_mean = np.mean(spreads_array)
```

**公式**：
```
μ = (1/T) * Σ_{t=1}^T spread_t
```

**说明**：使用历史价差的均值作为长期均衡值。

#### 3.3 计算误差修正项

```96:97:strategies/ecm_zscore_strategy.py
                # 计算误差修正项序列
                ecm_terms = spreads_array[:-1] - long_term_mean  # ECM_{t-1}
```

**公式**：
```
ECM_{t-1} = spread_{t-1} - μ
```

**说明**：计算每个时刻的误差修正项（偏离长期均值的程度）。

#### 3.4 计算价差的一阶差分

```99:100:strategies/ecm_zscore_strategy.py
                # 计算价差的一阶差分
                diff_spreads = np.diff(spreads_array)  # Δspread_t
```

**公式**：
```
Δspread_t = spread_t - spread_{t-1}
```

**说明**：计算价差的变化量。

#### 3.5 估计ECM系数

```108:126:strategies/ecm_zscore_strategy.py
                    # 构建ECM模型：Δspread_t = α + β*ECM_{t-1} + ε_t
                    ecm_terms_aligned = ecm_terms[-min_len:]
                    diff_spreads_aligned = diff_spreads[-min_len:]
                    
                    # 使用OLS回归估计ECM系数
                    X = add_constant(ecm_terms_aligned.reshape(-1, 1))
                    y = diff_spreads_aligned
                    
                    try:
                        model = OLS(y, X).fit()
                        ecm_coefficient = model.params[1]  # β系数（误差修正系数）
                        
                        # 验证系数合理性（应该在-1到0之间，表示均值回归）
                        if ecm_coefficient > 0 or ecm_coefficient < -1:
                            # 系数不合理，使用默认值
                            ecm_coefficient = -0.1
                    except Exception:
                        # 回归失败，使用默认值
                        ecm_coefficient = -0.1
```

**回归方程**：
```
Δspread_t = α + β*ECM_{t-1} + ε_t
```

**OLS估计**：
- `X = [1, ECM_0], [1, ECM_1], ..., [1, ECM_{T-1}]`（添加常数项）
- `y = [Δspread_1, Δspread_2, ..., Δspread_T]`
- `[α, β]' = (X'X)^{-1}X'y`
- `ecm_coefficient = β`（误差修正系数）

**系数验证**：
- 合理的ECM系数应该在 `[-1, 0]` 之间
- 如果系数不合理，使用默认值 `-0.1`

#### 3.6 预测价差的均值回归

```136:147:strategies/ecm_zscore_strategy.py
            # 计算当前误差修正项
            if len(historical_spreads) > self.ecm_lag:
                ecm_current = historical_spreads[-self.ecm_lag] - long_term_mean
            else:
                ecm_current = historical_spreads[-1] - long_term_mean
            
            # 使用ECM模型预测价差的均值回归
            # 预测的价差变化：Δspread_predicted = β*ECM_{t-1}
            predicted_change = ecm_coefficient * ecm_current
            
            # 预测的价差均值（基于均值回归）
            predicted_mean = long_term_mean - predicted_change
```

**公式推导**：

1. **计算当前误差修正项**：
   ```
   ECM_{t-1} = spread_{t-1} - μ
   ```

2. **预测价差变化**：
   ```
   Δspread_t_predicted = β*ECM_{t-1}
   ```

3. **预测价差均值**（基于均值回归）：
   ```
   spread_t_predicted = spread_{t-1} + Δspread_t_predicted
                      = spread_{t-1} + β*ECM_{t-1}
                      = spread_{t-1} + β*(spread_{t-1} - μ)
                      = spread_{t-1} + β*spread_{t-1} - β*μ
                      = (1 + β)*spread_{t-1} - β*μ
   ```

4. **简化形式**（代码中的实现）：
   ```
   predicted_mean = μ - β*ECM_{t-1}
                  = μ - β*(spread_{t-1} - μ)
                  = μ - β*spread_{t-1} + β*μ
                  = (1 + β)*μ - β*spread_{t-1}
   ```

**说明**：
- 代码中使用 `predicted_mean = long_term_mean - predicted_change`
- 这等价于 `predicted_mean = μ - β*ECM_{t-1}`
- 当 `β < 0` 时，如果 `ECM_{t-1} > 0`（价差高于均值），则 `predicted_mean < μ`，价差下降
- 当 `β < 0` 时，如果 `ECM_{t-1} < 0`（价差低于均值），则 `predicted_mean > μ`，价差上升

#### 3.7 计算Z-score

```154:155:strategies/ecm_zscore_strategy.py
            # 计算Z-score
            z_score = (current_spread - predicted_mean) / historical_std
```

**公式**：
```
Z = (spread_t - spread_t_predicted) / σ
```

其中：
- `spread_t`：当前价差
- `spread_t_predicted`：ECM模型预测的价差均值
- `σ`：历史波动率（标准差）

**与传统方法的区别**：
- **传统方法**：`Z = (spread_t - μ) / σ`，使用固定的长期均值
- **ECM方法**：`Z = (spread_t - spread_t_predicted) / σ`，使用动态预测的均值

---

### 四、ECM模型在回测中的应用

#### 4.1 策略调用流程

在回测主循环中，ECM策略的调用流程如下：

```644:645:cointegration_test_windows_ECM.py
                current_z_score = self.calculate_z_score(current_spread, historical_spreads,
                                                        historical_prices1, historical_prices2)
```

**调用链**：
1. `backtest_cointegration_trading()` 方法收集历史价差数据
2. 调用 `calculate_z_score()` 方法（`AdvancedCointegrationTrading` 类）
3. `calculate_z_score()` 方法调用 `z_score_strategy.calculate_z_score()`（策略对象）
4. `EcmZScoreStrategy.calculate_z_score()` 执行以下步骤：
   - 计算长期均值
   - 计算误差修正项
   - 估计ECM系数
   - 预测价差的均值回归
   - 计算Z-score

#### 4.2 模型缓存机制

为了提高计算效率，ECM策略实现了参数缓存：

```89:134:strategies/ecm_zscore_strategy.py
            # 创建缓存键（基于数据长度和最后几个值）
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._ecm_params_cache:
                ecm_coefficient = self._ecm_params_cache[cache_key]
            else:
                # 计算误差修正项序列
                ecm_terms = spreads_array[:-1] - long_term_mean  # ECM_{t-1}
                
                # 计算价差的一阶差分
                diff_spreads = np.diff(spreads_array)  # Δspread_t
                
                # 确保长度匹配
                min_len = min(len(ecm_terms), len(diff_spreads))
                if min_len < 10:
                    # 数据不足，使用简单方法
                    ecm_coefficient = -0.1  # 默认的均值回归系数
                else:
                    # 构建ECM模型：Δspread_t = α + β*ECM_{t-1} + ε_t
                    ecm_terms_aligned = ecm_terms[-min_len:]
                    diff_spreads_aligned = diff_spreads[-min_len:]
                    
                    # 使用OLS回归估计ECM系数
                    X = add_constant(ecm_terms_aligned.reshape(-1, 1))
                    y = diff_spreads_aligned
                    
                    try:
                        model = OLS(y, X).fit()
                        ecm_coefficient = model.params[1]  # β系数（误差修正系数）
                        
                        # 验证系数合理性（应该在-1到0之间，表示均值回归）
                        if ecm_coefficient > 0 or ecm_coefficient < -1:
                            # 系数不合理，使用默认值
                            ecm_coefficient = -0.1
                    except Exception:
                        # 回归失败，使用默认值
                        ecm_coefficient = -0.1
                
                # 缓存参数（限制缓存大小）
                if len(self._ecm_params_cache) >= self._max_cache_size:
                    # 清除最旧的缓存
                    oldest_key = next(iter(self._ecm_params_cache))
                    del self._ecm_params_cache[oldest_key]
                
                self._ecm_params_cache[cache_key] = ecm_coefficient
```

**缓存机制**：
- **缓存键**：基于数据长度和最后5个数据点
- **缓存内容**：ECM系数（β）
- **缓存大小**：最多缓存10个参数（`_max_cache_size = 10`）
- **缓存策略**：FIFO（先进先出），当缓存满时删除最旧的参数

#### 4.3 容错机制

ECM策略实现了多层容错机制：

1. **数据不足**：如果历史数据少于最小要求，返回0
2. **波动率为0**：如果历史波动率为0，返回0
3. **回归失败**：如果OLS回归失败，使用默认ECM系数 `-0.1`
4. **系数不合理**：如果ECM系数不在合理范围内（`[-1, 0]`），使用默认值
5. **预测结果无效**：如果预测结果为NaN，使用长期均值

---

### 五、ECM模型参数说明

#### 5.1 初始化参数

```23:43:strategies/ecm_zscore_strategy.py
    def __init__(self, ecm_lag: int = 1, min_data_length: int = 30, **kwargs):
        """
        初始化ECM策略
        
        Args:
            ecm_lag: 误差修正项的滞后阶数（默认1，即使用前一期）
            min_data_length: 最小数据长度要求
            **kwargs: 其他策略参数
        """
        super().__init__(ecm_lag=ecm_lag, min_data_length=min_data_length, **kwargs)
        self.name = "ECM误差修正模型"
        self.ecm_lag = ecm_lag
        self.min_data_length = min_data_length
        
        # 检查库是否可用
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels库不可用，无法使用ECM策略")
        
        # 存储模型参数缓存
        self._ecm_params_cache = {}
        self._max_cache_size = 10
```

**参数说明**：
- **ecm_lag**：误差修正项的滞后阶数，默认1
  - `ecm_lag = 1`：使用前一期（`ECM_{t-1}`）
  - `ecm_lag = 2`：使用前两期（`ECM_{t-2}`）
  - 通常使用1阶即可
- **min_data_length**：最小数据长度，默认30
  - 如果历史数据少于这个长度，使用传统方法
  - 建议值：至少是 `ecm_lag + 10`

#### 5.2 参数选择建议

**误差修正项滞后阶数（ecm_lag）**：
- **1**（推荐）：适合大多数情况，使用前一期
- **2**：如果价差回归有延迟，可以使用前两期
- **更多阶数**：通常不推荐，模型复杂度高

**最小数据长度（min_data_length）**：
- **30**（默认）：适合大多数情况
- **50**：如果数据充足，可以提高稳定性
- **20**：最小推荐值，低于此值可能不稳定

---

### 六、ECM模型数学公式总结

#### 6.1 完整模型公式

**ECM模型**：
```
Δspread_t = α + β*ECM_{t-1} + ε_t
```

其中：
- `Δspread_t = spread_t - spread_{t-1}`：价差的一阶差分
- `ECM_{t-1} = spread_{t-1} - μ`：误差修正项
- `μ = mean(historical_spreads)`：长期均值（均衡值）
- `β`：误差修正系数（通常为负，表示均值回归）
- `α`：常数项
- `ε_t`：误差项

#### 6.2 参数估计公式

**OLS回归**：
```
[α, β]' = (X'X)^{-1}X'y
```

其中：
- `X = [1, ECM_0], [1, ECM_1], ..., [1, ECM_{T-1}]`
- `y = [Δspread_1, Δspread_2, ..., Δspread_T]`

#### 6.3 预测公式

**预测价差变化**：
```
Δspread_t_predicted = β*ECM_{t-1}
```

**预测价差均值**：
```
spread_t_predicted = μ - β*ECM_{t-1}
                  = μ - β*(spread_{t-1} - μ)
                  = (1 + β)*μ - β*spread_{t-1}
```

#### 6.4 Z-score计算公式

**ECM Z-score**：
```
Z = (spread_t - spread_t_predicted) / σ
```

其中：
- `spread_t`：当前价差
- `spread_t_predicted`：ECM模型预测的价差均值
- `σ`：历史波动率（标准差）

---

### 七、ECM策略的优势和局限性

#### 7.1 优势

1. **理论基础扎实**：
   - 基于协整理论和误差修正模型，有坚实的统计学基础
   - 专门描述协整关系的均值回归特性

2. **动态预测**：
   - 能够预测价差的均值回归，而不是使用固定的历史均值
   - 根据当前偏离程度调整预测

3. **适合协整交易**：
   - ECM模型专门用于协整关系，特别适合协整交易
   - 能够捕捉短期偏离和长期均衡的动态过程

4. **计算效率高**：
   - 虽然有OLS回归，但计算相对简单
   - 有缓存机制，避免重复计算

#### 7.2 局限性

1. **需要足够的数据**：
   - 至少需要 `min_data_length` 个数据点
   - 需要足够的数据来估计ECM系数

2. **假设协整关系稳定**：
   - 如果协整关系破裂，ECM模型可能失效
   - 需要定期重新检验协整关系

3. **参数选择敏感**：
   - 滞后阶数的选择影响模型性能
   - 需要根据市场特性调整参数

4. **可能过拟合**：
   - 如果数据不足，OLS回归可能过拟合
   - 需要验证ECM系数的合理性

#### 7.3 适用场景

**推荐使用ECM策略的场景**：
1. **协整关系稳定**：两个资产之间存在稳定的协整关系
2. **数据充足**：有足够的历史数据（至少30个数据点）
3. **均值回归明显**：价差有明显的均值回归特性
4. **对准确性要求高**：需要更准确的Z-score计算

**不推荐使用ECM策略的场景**：
1. **数据不足**：历史数据少于30个数据点
2. **协整关系不稳定**：协整关系经常破裂
3. **价差发散**：价差没有均值回归特性

---

### 八、ECM模型与传统方法的对比

#### 8.1 传统方法

**Z-score计算公式**：
```
Z_traditional = (spread_t - μ) / σ
```

其中：
- `μ = mean(historical_spreads)`：固定的历史均值
- `σ = std(historical_spreads)`：历史标准差

**特点**：
- 使用固定的历史均值
- 不考虑均值回归的动态过程
- 计算简单快速

#### 8.2 ECM方法

**Z-score计算公式**：
```
Z_ecm = (spread_t - spread_t_predicted) / σ
```

其中：
- `spread_t_predicted = μ - β*ECM_{t-1}`：动态预测的均值
- `σ = std(historical_spreads)`：历史标准差

**特点**：
- 使用动态预测的均值
- 考虑均值回归的动态过程
- 根据当前偏离程度调整预测

#### 8.3 对比示例

**场景**：当前价差 `spread_t = 10`，长期均值 `μ = 5`，历史标准差 `σ = 2`

**传统方法**：
```
Z_traditional = (10 - 5) / 2 = 2.5
```

**ECM方法**（假设 `β = -0.2`，`ECM_{t-1} = 3`）：
```
spread_t_predicted = 5 - (-0.2) * 3 = 5 + 0.6 = 5.6
Z_ecm = (10 - 5.6) / 2 = 2.2
```

**说明**：
- ECM方法预测价差会回归到 `5.6`（接近均值 `5`）
- 因此Z-score略低于传统方法（`2.2 < 2.5`）
- 这反映了ECM模型对均值回归的预期

---

### 九、总结

ECM误差修正模型是一个强大的Z-score计算策略，能够：

1. **捕捉均值回归**：使用误差修正模型预测价差的均值回归
2. **动态调整预测**：根据当前偏离程度调整预测的均值
3. **理论基础扎实**：基于协整理论和误差修正模型
4. **适合协整交易**：专门用于协整关系，特别适合协整交易

在协整交易中，ECM策略特别适合：
- 协整关系稳定的场景
- 价差有明显均值回归特性的场景
- 需要更准确Z-score计算的场景

通过合理选择参数（滞后阶数、最小数据长度等），ECM策略能够显著提高协整交易的性能。

---

## 版本信息

- **文件版本**：cointegration_test_windows_ECM.py
- **主要特性**：滚动窗口协整检验、参数优化、ECM误差修正模型
- **最后更新**：添加了ECM误差修正模型支持，策略部分放在strategies文件夹下单独调用


