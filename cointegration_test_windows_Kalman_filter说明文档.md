# cointegration_test_windows_Kalman_filter.py 详细说明文档

## 文件概述

本文件是一个高级协整分析+交易流程完整测试程序，支持滚动窗口协整检验、参数优化和多种Z-score计算策略（包括新增的Kalman Filter动态价差模型）。

### 主要功能

1. **滚动窗口协整检验**：使用固定大小的滚动窗口进行协整关系检验，识别时变特性
2. **参数优化**：支持网格搜索、随机搜索、贝叶斯优化三种方法
3. **多种Z-score策略**：
   - 传统方法（均值和标准差）
   - ARIMA-GARCH模型
   - ECM误差修正模型
   - **Kalman Filter动态价差模型**（新增）
4. **完整交易回测**：包含开仓、平仓、止盈止损、手续费计算等

---

## 程序运行流程

### 阶段一：程序启动和初始化

#### 1. 程序入口（第3165-3166行）

```python
if __name__ == "__main__":
    main()
```

- **执行时机**：直接运行脚本时
- **功能**：调用主函数 `main()`

#### 2. 主函数初始化（第3127-3142行）

```3127:3142:cointegration_test_windows_Kalman_filter.py
def main():
    """
    主函数
    """
    print("滚动窗口协整分析+交易流程完整测试（带参数优化+Kalman Filter动态价差模型）")
    print("使用滚动窗口进行协整检验：")
    print("  1. 使用固定大小的滚动窗口（如1000条数据）")
    print("  2. 在每个窗口内进行完整的EG两阶段协整检验")
    print("  3. 汇总所有窗口的检验结果，识别协整关系的时变特性")
    print("  4. 支持参数优化（网格搜索、随机搜索、贝叶斯优化）")
    print("  5. 支持多种Z-score计算策略：")
    print("     - 传统方法（均值和标准差）")
    print("     - ARIMA-GARCH模型")
    print("     - ECM误差修正模型（推荐用于协整交易）")
    print("     - Kalman Filter动态价差模型（推荐用于动态市场）")
    print()
```

- **功能**：显示程序介绍和功能说明
- **输出**：程序标题和功能列表

#### 3. 模式选择（第3144-3155行）

```3144:3155:cointegration_test_windows_Kalman_filter.py
    # 选择模式
    print("请选择运行模式:")
    print("  1. 普通回测模式")
    print("  2. 参数优化模式")

    mode_choice = input("请选择 (1/2): ").strip()

    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()

    if not csv_file_path:
        csv_file_path = "segment_2_data_ccxt_20251113_103652.csv"
        print(f"使用默认路径: {csv_file_path}")
```

- **功能**：
  - 让用户选择运行模式（普通回测或参数优化）
  - 输入CSV文件路径，或使用默认路径
- **用户输入**：
  - 模式选择：1 或 2
  - CSV文件路径（可选）

#### 4. 模式分支（第3157-3162行）

```3157:3162:cointegration_test_windows_Kalman_filter.py
    if mode_choice == '2':
        # 参数优化模式
        test_parameter_optimization(csv_file_path)
    else:
        # 普通回测模式
        test_rolling_window_cointegration_trading(csv_file_path)
```

- **功能**：根据用户选择调用不同的测试函数
- **分支**：
  - 模式1：调用 `test_rolling_window_cointegration_trading()`（普通回测）
  - 模式2：调用 `test_parameter_optimization()`（参数优化）

---

## 模式一：普通回测模式流程

### 步骤1：数据加载（第2177-2184行）

```2177:2184:cointegration_test_windows_Kalman_filter.py
    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")
```

- **调用函数**：`load_csv_data(csv_file_path)`（第86-125行）
- **功能**：
  - 从CSV文件读取价格数据
  - 解析时间戳和价格列
  - 返回字典格式：`{symbol: DataFrame}`
- **数据结构**：每个DataFrame包含 `timestamp` 和 `close` 列

### 步骤2：选择价差类型（第2186-2188行）

```2186:2188:cointegration_test_windows_Kalman_filter.py
    # 2. 选择价差类型
    print("\n2. 选择价差类型")
    diff_order = select_spread_type()
```

- **调用函数**：`select_spread_type()`（第601-643行）
- **功能**：让用户选择使用原始价差还是差分价差
- **选项**：
  - `0`：原始价差（spread = price1 - hedge_ratio * price2）
  - `1`：一阶差分价差（spread_diff = spread[t] - spread[t-1]）
- **返回值**：`diff_order`（0 或 1）

### 步骤3：配置滚动窗口参数（第2190-2192行）

```2190:2192:cointegration_test_windows_Kalman_filter.py
    # 3. 配置滚动窗口参数
    print("\n3. 配置滚动窗口参数")
    window_params = configure_rolling_window_parameters()
```

- **调用函数**：`configure_rolling_window_parameters()`（第644-690行）
- **功能**：配置滚动窗口的大小和步长
- **参数**：
  - `window_size`：窗口大小（默认1000）
  - `step_size`：步长（默认100）
- **返回值**：字典 `{'window_size': int, 'step_size': int}`

### 步骤4：滚动窗口寻找协整对（第2194-2207行）

```2194:2207:cointegration_test_windows_Kalman_filter.py
    # 4. 滚动窗口寻找协整对
    print("\n4. 滚动窗口寻找协整对")
    all_summaries = rolling_window_find_cointegrated_pairs(
        data,
        window_size=window_params['window_size'],
        step_size=window_params['step_size'],
        diff_order=diff_order  # 传递价差类型
    )

    if not all_summaries:
        print("未找到任何协整对，无法进行交易")
        return

    print(f"\n找到 {len(all_summaries)} 个币对的滚动窗口检验结果")
```

- **调用函数**：`rolling_window_find_cointegrated_pairs()`（第551-600行）
- **功能**：
  - 对所有币对组合进行滚动窗口协整检验
  - 汇总每个币对的协整检验结果
- **内部流程**：
  1. 遍历所有币对组合（第551-600行）
  2. 对每个币对调用 `rolling_window_cointegration_test()`（第439-550行）
  3. 在每个窗口内调用 `enhanced_cointegration_test()`（第259-438行）
  4. 汇总所有窗口的检验结果

#### 4.1 滚动窗口协整检验详细流程

**函数**：`rolling_window_cointegration_test()`（第439-550行）

1. **数据对齐**（第439-474行）
   - 对齐两个价格序列的时间戳
   - 确保数据长度一致

2. **窗口划分**（第475-498行）
   - 根据 `window_size` 和 `step_size` 划分窗口
   - 计算窗口数量：`num_windows = (min_length - window_size) // step_size + 1`

3. **逐个窗口检验**（第478-529行）
   - 对每个窗口提取数据
   - 调用 `enhanced_cointegration_test()` 进行协整检验
   - 记录每个窗口的检验结果

4. **结果汇总**（第531-550行）
   - 计算协整比例（通过检验的窗口数 / 总窗口数）
   - 返回汇总结果

#### 4.2 增强协整检验详细流程

**函数**：`enhanced_cointegration_test()`（第259-438行）

1. **计算对冲比率**（第259-265行）
   - 调用 `calculate_hedge_ratio()`（第129-167行）
   - 使用OLS回归：`price1 = α + β * price2`
   - 返回对冲比率 `β`

2. **计算价差**（第266-275行）
   - 原始价差：`spread = price1 - hedge_ratio * price2`
   - 如果 `diff_order > 0`，计算差分价差

3. **ADF检验**（第276-290行）
   - 调用 `advanced_adf_test()`（第168-219行）
   - 检验价差的平稳性
   - 返回ADF统计量、P值等

4. **确定积分阶数**（第291-300行）
   - 调用 `determine_integration_order()`（第220-258行）
   - 确定价格序列的积分阶数

5. **协整判断**（第301-437行）
   - 根据ADF检验结果判断是否协整
   - 返回详细的检验结果字典

### 步骤5：显示并选择协整对（第2209-2220行）

```2209:2220:cointegration_test_windows_Kalman_filter.py
    # 5. 显示并选择协整对
    print("\n5. 选择协整对")
    selected_pairs = display_rolling_window_candidates(
        all_summaries,
        data,  # 传递原始数据用于重新计算对冲比率
        diff_order,  # 传递价差类型
        min_cointegration_ratio=0.3
    )

    if not selected_pairs:
        print("未选择任何币对，无法进行交易")
        return
```

- **调用函数**：`display_rolling_window_candidates()`（第691-837行）
- **功能**：
  - 显示所有找到的协整对及其统计信息
  - 让用户选择要交易的币对
  - 重新计算基于整个数据集的对冲比率
- **显示信息**：
  - 币对名称
  - 协整比例（通过检验的窗口比例）
  - 平均对冲比率
  - 平均ADF P值
- **筛选条件**：`min_cointegration_ratio=0.3`（至少30%的窗口通过检验）

### 步骤6：策略选择和回测循环（第2222-2316行）

```2222:2316:cointegration_test_windows_Kalman_filter.py
    # 6. 策略选择和回测循环
    print("\n6. 策略选择和回测循环")
    test_count = 0

    while True:
        test_count += 1
        print(f"\n{'=' * 80}")
        print(f"第 {test_count} 次测试")
        print(f"{'=' * 80}")

        # 选择Z-score计算策略
        z_score_strategy = select_z_score_strategy()
        if z_score_strategy is None:
            print("测试结束，退出程序")
            break

        # 7. 显示选择的币对详情
        print(f"\n第 {test_count} 次测试 - 选择的币对详情")
        for pair in selected_pairs:
            diff_order = pair.get('diff_order', 0)
            diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
            print(f"\n币对: {pair['pair_name']}")
            print(f"  价差类型: {diff_type}")
            print(f"  对冲比率: {pair['hedge_ratio']:.6f} (基于整个数据集计算)")
            if pair.get('spread_adf'):
                print(f"  价差ADF P值: {pair['spread_adf']['p_value']:.6f}")
            if 'cointegration_ratio' in pair:
                print(f"  协整比例: {pair['cointegration_ratio'] * 100:.1f}%")

        # 8. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()

        # 9. 执行交易回测
        print(f"\n第 {test_count} 次测试 - 执行交易回测")
        print(f"使用的策略: {z_score_strategy.get_strategy_description()}")
        trading_strategy = AdvancedCointegrationTrading(
            lookback_period=trading_params['lookback_period'],
            z_threshold=trading_params['z_threshold'],
            z_exit_threshold=trading_params['z_exit_threshold'],
            take_profit_pct=trading_params['take_profit_pct'],
            stop_loss_pct=trading_params['stop_loss_pct'],
            max_holding_hours=trading_params['max_holding_hours'],
            position_ratio=trading_params['position_ratio'],
            leverage=trading_params['leverage'],
            trading_fee_rate=trading_params.get('trading_fee_rate', 0.000275),
            z_score_strategy=z_score_strategy  # 使用策略对象
        )

        results = trading_strategy.backtest_cointegration_trading(
            data,
            selected_pairs,
            initial_capital=10000
        )

        # 9. 显示交易详情
        print(f"\n第 {test_count} 次测试 - 交易详情")
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
                    print(
                        f"  {trade['timestamp']}: {action} {trade['symbol1']}={trade['symbol1_price']:.2f}, {trade['symbol2']}={trade['symbol2_price']:.2f}{pnl_info}")
        else:
            print("本次测试无交易记录")

        print(f"\n第 {test_count} 次测试完成")
        print("=" * 80)

        # 询问是否继续
        print("\n请选择下一步操作:")
        print("  1. 继续测试（重新选择策略）")
        print("  0. 退出程序")
        continue_choice = input("请选择 (0/1): ").strip()

        if continue_choice != '1':
            print("测试结束，退出程序")
            break
```

#### 6.1 选择Z-score计算策略（第2233-2236行）

- **调用函数**：`select_z_score_strategy()`（第839-1043行）
- **功能**：让用户选择Z-score计算策略
- **选项**：
  1. **传统方法**：使用均值和标准差计算Z-score
  2. **ARIMA-GARCH模型**：使用ARIMA预测均值，GARCH预测波动率
  3. **ECM误差修正模型**：使用误差修正模型预测均值回归
  4. **Kalman Filter动态价差模型**（新增）：使用Kalman Filter动态估计价差的均值和方差

**Kalman Filter策略配置**（第978-1033行）：
- 过程噪声方差（默认0.01）
- 观测噪声方差（默认0.1）
- 最小数据长度（默认30）

#### 6.2 配置交易参数（第2251-2253行）

- **调用函数**：`configure_trading_parameters()`（第1044-1173行）
- **功能**：配置交易策略参数
- **参数**：
  - `lookback_period`：回看期（默认60）
  - `z_threshold`：Z-score开仓阈值（默认1.5）
  - `z_exit_threshold`：Z-score平仓阈值（默认0.6）
  - `take_profit_pct`：止盈百分比（默认0.15）
  - `stop_loss_pct`：止损百分比（默认0.08）
  - `max_holding_hours`：最大持仓时间（默认168小时）
  - `position_ratio`：仓位比例（默认0.5）
  - `leverage`：杠杆倍数（默认5）
  - `trading_fee_rate`：交易手续费率（默认0.000275）

#### 6.3 创建交易策略对象（第2258-2269行）

- **类**：`AdvancedCointegrationTrading`（第1175-2164行）
- **功能**：初始化交易策略对象
- **关键属性**：
  - 交易参数（阈值、止盈止损等）
  - Z-score策略对象（`z_score_strategy`）
  - 策略类型标志（`use_arima_garch`、`use_ecm`、`use_kalman_filter`）

#### 6.4 执行回测（第2271-2275行）

- **调用方法**：`backtest_cointegration_trading()`（第1831-1830行）
- **功能**：执行完整的交易回测

### 步骤7：回测执行详细流程

**方法**：`backtest_cointegration_trading()`（第1831-2050行）

#### 7.1 初始化（第1842-1852行）

```1842:1852:cointegration_test_windows_Kalman_filter.py
        # 初始化
        # 确定投入资金：capital = initial_capital * position_ratio
        # position_ratio只在初始化时使用一次，之后不再使用
        capital = initial_capital * self.position_ratio
        results = {
            'capital_curve': [],
            'trades': [],
            'daily_returns': [],
            'positions': {},
            'pair_details': {}
        }
```

- **功能**：初始化回测数据结构
- **资金计算**：`capital = initial_capital * position_ratio`

#### 7.2 获取所有时间点（第1854-1858行）

- **功能**：收集所有币对的时间戳，合并并排序
- **用途**：按时间顺序遍历所有数据点

#### 7.3 回测主循环（第1885-2049行）

对每个时间点执行以下步骤：

##### 7.3.1 获取当前价格（第1886-1892行）

- 从数据中提取当前时间点的所有币对价格

##### 7.3.2 检查现有持仓（第1894-1950行）

对每个持仓执行：

1. **获取持仓信息**（第1894-1900行）
   - 持仓币对、开仓价格、持仓数量等

2. **计算当前价差**（第1902-1905行）
   - 使用对冲比率计算当前价差

3. **计算Z-score**（第1907-1920行）
   - 获取历史价差序列
   - 调用Z-score策略计算当前Z-score
   - **Kalman Filter策略**：使用动态估计的均值和方差计算Z-score

4. **检查退出条件**（第1922-1950行）
   - Z-score平仓阈值
   - 止盈止损
   - 最大持仓时间
   - 如果满足条件，执行平仓

##### 7.3.3 检查开仓条件（第1952-2020行）

对每个选择的币对执行：

1. **检查是否已有持仓**（第1954-1957行）
   - 如果已有持仓，跳过

2. **计算当前价差**（第1959-1962行）
   - 使用对冲比率计算价差

3. **获取历史价差**（第1964-1980行）
   - 从历史数据中提取价差序列
   - 长度 = `lookback_period`

4. **计算Z-score**（第1982-1995行）
   - 调用Z-score策略计算Z-score
   - **Kalman Filter策略**：
     - 使用Kalman Filter动态更新价差的估计值
     - 基于估计的均值和方差计算Z-score

5. **生成交易信号**（第1997-2018行）
   - `z_score >= z_threshold`：做空价差（SHORT_LONG）
   - `z_score <= -z_threshold`：做多价差（LONG_SHORT）

6. **执行开仓**（第2020-2029行）
   - 计算持仓数量（Beta中性）
   - 记录开仓信息
   - 扣除手续费

##### 7.3.4 更新资金曲线（第2031-2049行）

- 计算当前总资产
- 记录到 `capital_curve`
- 计算日收益率

#### 7.4 回测结果分析（第2051-2164行）

1. **计算统计指标**（第2051-2120行）
   - 总收益率
   - 年化收益率
   - 最大回撤
   - 夏普比率
   - 胜率等

2. **显示结果**（第2122-2140行）
   - 打印所有统计指标
   - 显示资金曲线图

3. **导出交易记录**（第2142-2164行）
   - 导出为CSV文件

---

## 模式二：参数优化模式流程

### 步骤1-5：与普通回测模式相同

- 数据加载
- 选择价差类型
- 配置滚动窗口参数
- 滚动窗口寻找协整对
- 选择协整对

### 步骤6：选择Z-score计算策略（第2996-3001行）

```2996:3001:cointegration_test_windows_Kalman_filter.py
    # 6. 选择Z-score计算策略
    print("\n6. 选择Z-score计算策略")
    z_score_strategy = select_z_score_strategy()
    if z_score_strategy is None:
        print("未选择策略，退出优化")
        return
```

- **功能**：选择用于优化的Z-score策略
- **注意**：优化过程中策略固定，只优化交易参数

### 步骤7：选择优化方法（第3003-3030行）

```3003:3030:cointegration_test_windows_Kalman_filter.py
    # 7. 选择优化方法
    print("\n7. 选择优化方法")
    print("可选方法:")
    print("  1. 网格搜索（粗粒度+细粒度）")
    print("  2. 随机搜索")
    print("  3. 贝叶斯优化")

    method_choice = input("请选择优化方法 (1/2/3): ").strip()

    # 创建优化器
    optimizer = ParameterOptimizer(
        data,
        selected_pairs,
        initial_capital=10000,
        objective='sharpe_ratio',  # 优化目标：夏普比率
        stability_test=True,  # 启用稳定性测试
        z_score_strategy=z_score_strategy  # 使用选择的策略
    )
```

- **优化方法**：
  1. **网格搜索**：系统遍历所有参数组合
  2. **随机搜索**：随机采样参数组合
  3. **贝叶斯优化**：使用高斯过程优化（需要scikit-optimize）

### 步骤8：执行优化（第3032-3119行）

#### 8.1 网格搜索（第3032-3055行）

- **流程**：
  1. 粗粒度搜索：遍历粗粒度参数组合
  2. 细粒度搜索：在最佳参数附近进行细粒度搜索
  3. 稳定性测试：对最佳参数进行扰动测试

#### 8.2 随机搜索（第3057-3075行）

- **流程**：
  1. 随机生成参数组合
  2. 评估每个组合
  3. 选择最佳组合

#### 8.3 贝叶斯优化（第3077-3119行）

- **流程**：
  1. 定义参数搜索空间
  2. 使用高斯过程建模目标函数
  3. 迭代优化，选择下一个评估点
  4. 收敛后返回最佳参数

### 步骤9：导出优化结果（第3120-3124行）

```3120:3124:cointegration_test_windows_Kalman_filter.py
    # 12. 导出结果
    print("\n12. 导出优化结果...")
    optimizer.export_results()

    return result
```

- **功能**：将优化结果导出为CSV文件
- **内容**：最佳参数、评估历史、统计指标等

---

## Kalman Filter策略详解

### 策略类：KalmanFilterZScoreStrategy

**文件位置**：`strategies/kalman_filter_zscore_strategy.py`

### 核心方法：calculate_z_score()

#### 1. 输入验证（第128-131行）

- 检查历史数据长度是否足够
- 不足时返回0.0（中性信号）

#### 2. Kalman Filter初始化（第137-150行）

- 如果状态未初始化，使用前一半数据初始化
- 初始化状态均值：历史价差的均值
- 初始化状态协方差：历史价差的方差

#### 3. Kalman Filter更新（第151-167行）

- 对历史数据进行滤波
- 使用当前观测更新状态估计
- 执行预测步骤和更新步骤

#### 4. 计算动态标准差（第169-184行）

- 方法1：使用估计值的标准差（如果有足够数据）
- 方法2：使用协方差的平方根
- 如果无效，回退到历史数据的标准差

#### 5. 计算Z-score（第186-197行）

```python
z_score = (current_spread - self._state_mean) / self._estimated_std
```

- 使用Kalman Filter估计的均值
- 使用动态估计的标准差

#### 6. 容错机制（第189-212行）

- 验证Z-score有效性（NaN/Inf检查）
- 如果无效，回退到传统方法
- 如果传统方法也失败，返回0.0

### Kalman Filter数学模型

**状态方程**：
```
x_t = x_{t-1} + w_t  (w_t ~ N(0, Q))
```

**观测方程**：
```
y_t = x_t + v_t  (v_t ~ N(0, R))
```

**预测步骤**：
```
x_{t|t-1} = x_{t-1|t-1}
P_{t|t-1} = P_{t-1|t-1} + Q
```

**更新步骤**：
```
K = P_{t|t-1} / (P_{t|t-1} + R)
x_{t|t} = x_{t|t-1} + K * (y_t - x_{t|t-1})
P_{t|t} = (1 - K) * P_{t|t-1}
```

其中：
- `x_t`：价差的真实值（状态）
- `y_t`：观测到的价差
- `Q`：过程噪声方差（`process_variance`）
- `R`：观测噪声方差（`observation_variance`）
- `K`：Kalman增益

---

## 关键数据结构

### 1. 协整检验结果

```python
{
    'pair_name': str,  # 币对名称
    'symbol1': str,    # 第一个币种
    'symbol2': str,    # 第二个币种
    'hedge_ratio': float,  # 对冲比率
    'cointegration_found': bool,  # 是否协整
    'spread_adf': {
        'statistic': float,  # ADF统计量
        'p_value': float,    # P值
        'critical_values': dict  # 临界值
    },
    'cointegration_ratio': float,  # 协整比例
    'window_results': list  # 各窗口的检验结果
}
```

### 2. 交易记录

```python
{
    'timestamp': str,  # 时间戳
    'pair': str,       # 币对名称
    'action': str,     # 'OPEN' 或 'CLOSE'
    'symbol1': str,    # 第一个币种
    'symbol2': str,    # 第二个币种
    'symbol1_price': float,  # symbol1价格
    'symbol2_price': float,  # symbol2价格
    'symbol1_size': float,   # symbol1数量
    'symbol2_size': float,   # symbol2数量
    'spread': float,   # 价差
    'z_score': float,  # Z-score
    'pnl': float,      # 盈亏（仅平仓时）
    'signal': dict     # 交易信号
}
```

### 3. 回测结果

```python
{
    'capital_curve': list,      # 资金曲线
    'trades': list,             # 交易记录
    'daily_returns': list,      # 日收益率
    'positions': dict,          # 持仓信息
    'pair_details': dict,       # 币对详情
    'total_return': float,      # 总收益率
    'annual_return': float,     # 年化收益率
    'max_drawdown': float,      # 最大回撤
    'sharpe_ratio': float,      # 夏普比率
    'win_rate': float           # 胜率
}
```

---

## 注意事项

1. **数据要求**：
   - CSV文件必须包含 `timestamp` 和 `close` 列
   - 时间戳格式需要能被pandas解析

2. **Kalman Filter参数**：
   - `process_variance`：过程噪声方差，控制状态转移的不确定性
   - `observation_variance`：观测噪声方差，控制观测的不确定性
   - 参数需要根据数据特性调整

3. **计算性能**：
   - 滚动窗口检验可能耗时较长
   - 参数优化需要大量回测，建议先用小数据集测试

4. **策略选择**：
   - 传统方法：简单快速，适合稳定市场
   - ARIMA-GARCH：适合有趋势和波动聚集的市场
   - ECM：适合协整关系明显的市场
   - Kalman Filter：适合动态变化的市场，能自适应调整

---

## 总结

本程序提供了一个完整的协整交易回测框架，支持：

1. **滚动窗口协整检验**：识别时变的协整关系
2. **多种Z-score策略**：包括新增的Kalman Filter动态价差模型
3. **完整交易回测**：包含开仓、平仓、止盈止损、手续费等
4. **参数优化**：支持多种优化方法，寻找最佳参数组合

Kalman Filter策略通过动态估计价差的均值和方差，能够更好地适应市场变化，提高交易信号的准确性。


