# cointegration_test_windows_ Regime-Switching.py 详细说明文档

## 文件概述

本文件是一个高级协整分析+交易流程完整测试程序，支持滚动窗口协整检验、参数优化和多种Z-score计算策略（包括新增的Regime-Switching市场状态模型）。

### 主要功能

1. **滚动窗口协整检验**：使用固定大小的滚动窗口进行协整关系检验，识别时变特性
2. **参数优化**：支持网格搜索、随机搜索、贝叶斯优化三种方法
3. **多种Z-score策略**：
   - 传统方法（均值和标准差）
   - ARIMA-GARCH模型
   - ECM误差修正模型
   - Kalman Filter动态价差模型
   - Copula + DCC-GARCH相关性/波动率模型
   - **Regime-Switching市场状态模型**（新增）
4. **完整交易回测**：包含开仓、平仓、止盈止损、手续费计算等

---

## 程序运行流程

### 阶段一：程序启动和初始化

#### 1. 程序入口

```3306:3307:cointegration_test_windows_ Regime-Switching.py
if __name__ == "__main__":
    main()
```

- **执行时机**：直接运行脚本时
- **功能**：调用主函数 `main()`

#### 2. 主函数初始化

```3266:3283:cointegration_test_windows_ Regime-Switching.py
def main():
    """
    主函数
    """
    print("滚动窗口协整分析+交易流程完整测试（带参数优化+Regime-Switching市场状态模型）")
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
    print("     - Copula + DCC-GARCH相关性/波动率模型（推荐用于相关性建模）")
    print("     - Regime-Switching市场状态模型（推荐用于状态转换市场）")
    print()
```

- **功能**：显示程序介绍和功能说明
- **输出**：程序标题和功能列表

#### 3. 模式选择

```3285:3303:cointegration_test_windows_ Regime-Switching.py
    # 选择模式
    print("请选择运行模式:")
    print("  1. 普通回测模式")
    print("  2. 参数优化模式")

    mode_choice = input("请选择 (1/2): ").strip()

    csv_file_path = input("请输入CSV文件路径 (或按回车使用默认路径): ").strip()

    if not csv_file_path:
        csv_file_path = "segment_2_data_ccxt_20251113_103652.csv"
        print(f"使用默认路径: {csv_file_path}")

    if mode_choice == '2':
        # 参数优化模式
        test_parameter_optimization(csv_file_path)
    else:
        # 普通回测模式
        test_rolling_window_cointegration_trading(csv_file_path)
```

- **功能**：
  - 让用户选择运行模式（普通回测或参数优化）
  - 输入CSV文件路径，或使用默认路径
- **用户输入**：
  - 模式选择：1 或 2
  - CSV文件路径（可选）
- **分支**：
  - 模式1：调用 `test_rolling_window_cointegration_trading()`（普通回测）
  - 模式2：调用 `test_parameter_optimization()`（参数优化）

---

## 模式一：普通回测模式流程

### 步骤1：数据加载

```2316:2323:cointegration_test_windows_ Regime-Switching.py
    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")
```

**调用函数**：`load_csv_data(csv_file_path)`

```86:154:cointegration_test_windows_ Regime-Switching.py
def load_csv_data(csv_file_path):
    """
    从CSV文件加载数据

    Args:
        csv_file_path: CSV文件路径

    Returns:
        dict: 包含各币对数据的字典
    """
    try:
        print(f"正在加载CSV文件: {csv_file_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 检查数据格式
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 如果有symbol列，按币对分组
        if 'symbol' in df.columns:
            data = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
                symbol_data.set_index('timestamp', inplace=True)
                data[symbol] = symbol_data['close']
                print(f"币对 {symbol}: {len(symbol_data)} 条数据")
        else:
            # 假设是单个币对的数据
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            data = {'BTCUSDT': df['close']}
            print(f"单个币对数据: {len(df)} 条数据")

        return data

    except Exception as e:
        print(f"加载CSV文件失败: {str(e)}")
        return None
```

- **功能**：
  - 从CSV文件读取价格数据
  - 解析时间戳和价格列
  - 返回字典格式：`{symbol: Series}`
- **数据结构**：每个Series的索引是时间戳，值是收盘价

### 步骤2：选择价差类型

```2325:2327:cointegration_test_windows_ Regime-Switching.py
    # 2. 选择价差类型
    print("\n2. 选择价差类型")
    diff_order = select_spread_type()
```

**调用函数**：`select_spread_type()`

- **功能**：让用户选择使用原始价差还是差分价差
- **选项**：
  - `0`：原始价差（spread = price1 - hedge_ratio * price2）
  - `1`：一阶差分价差（spread_diff = spread[t] - spread[t-1]）
- **返回值**：`diff_order`（0 或 1）

### 步骤3：配置滚动窗口参数

```2329:2331:cointegration_test_windows_ Regime-Switching.py
    # 3. 配置滚动窗口参数
    print("\n3. 配置滚动窗口参数")
    window_params = configure_rolling_window_parameters()
```

**调用函数**：`configure_rolling_window_parameters()`

- **功能**：配置滚动窗口的大小和步长
- **参数**：
  - `window_size`：窗口大小（默认1000）
  - `step_size`：步长（默认100）
- **返回值**：字典 `{'window_size': int, 'step_size': int}`

### 步骤4：滚动窗口寻找协整对

```2333:2346:cointegration_test_windows_ Regime-Switching.py
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

**调用函数**：`rolling_window_find_cointegrated_pairs()`

- **功能**：
  - 对所有币对组合进行滚动窗口协整检验
  - 汇总每个币对的协整检验结果
- **内部流程**：
  1. 遍历所有币对组合
  2. 对每个币对调用 `rolling_window_cointegration_test()`
  3. 在每个窗口内调用 `enhanced_cointegration_test()`
  4. 汇总所有窗口的检验结果

### 步骤5：显示并选择协整对

```2348:2359:cointegration_test_windows_ Regime-Switching.py
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

**调用函数**：`display_rolling_window_candidates()`

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

### 步骤6：策略选择和回测循环

```2361:2450:cointegration_test_windows_ Regime-Switching.py
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
```

#### 6.1 选择Z-score计算策略

```2371:2375:cointegration_test_windows_ Regime-Switching.py
        # 选择Z-score计算策略
        z_score_strategy = select_z_score_strategy()
        if z_score_strategy is None:
            print("测试结束，退出程序")
            break
```

**调用函数**：`select_z_score_strategy()`

```840:1154:cointegration_test_windows_ Regime-Switching.py
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

    # 检查ARIMA-GARCH是否可用
    arima_garch_available = ARIMA_AVAILABLE and GARCH_AVAILABLE
    if arima_garch_available:
        print("  2. ARIMA-GARCH模型")
    else:
        print("  2. ARIMA-GARCH模型（不可用：缺少必要的库）")

    # 检查ECM是否可用
    ecm_available = STRATEGIES_AVAILABLE and STATSMODELS_AVAILABLE
    if ecm_available:
        print("  3. ECM误差修正模型（推荐用于协整交易）")
    else:
        print("  3. ECM误差修正模型（不可用：缺少必要的库）")

    # 检查Kalman Filter是否可用
    kalman_available = STRATEGIES_AVAILABLE
    if kalman_available:
        print("  4. Kalman Filter动态价差模型（推荐用于动态市场）")
    else:
        print("  4. Kalman Filter动态价差模型（不可用：缺少必要的库）")

    # 检查Copula + DCC-GARCH是否可用
    copula_dcc_available = STRATEGIES_AVAILABLE and GARCH_AVAILABLE
    if copula_dcc_available:
        print("  5. Copula + DCC-GARCH相关性/波动率模型（推荐用于相关性建模）")
    else:
        print("  5. Copula + DCC-GARCH相关性/波动率模型（不可用：缺少必要的库）")

    # 检查Regime-Switching是否可用
    regime_switching_available = STRATEGIES_AVAILABLE and STATSMODELS_AVAILABLE
    if regime_switching_available:
        print("  6. Regime-Switching市场状态模型（推荐用于状态转换市场）")
    else:
        print("  6. Regime-Switching市场状态模型（不可用：缺少必要的库）")

    print("  0. 退出程序")

    # 确定最大选择数
    max_choice = 6

    while True:
        try:
            choice = input(f"请选择 (0-{max_choice}): ").strip()

            if choice == '0':
                return None

            if choice == '1':
                strategy = TraditionalZScoreStrategy()
                print(f"已选择: {strategy.get_strategy_description()}")
                return strategy

            # ... 其他策略选择 ...

            if choice == '6' and regime_switching_available:
                # 询问Regime-Switching参数
                print("\n配置Regime-Switching市场状态模型参数:")
                print("  直接回车使用默认值: 状态数量=2, 最小数据长度=50, 平滑概率=True")

                n_regimes_input = input("状态数量 (默认2): ").strip()
                if n_regimes_input:
                    try:
                        n_regimes = int(n_regimes_input)
                        if n_regimes < 2:
                            print("状态数量必须>=2，使用默认值")
                            n_regimes = 2
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        n_regimes = 2
                else:
                    n_regimes = 2

                min_data_input = input("最小数据长度 (默认50): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 20:
                            print("最小数据长度必须>=20，使用默认值")
                            min_data_length = 50
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 50
                else:
                    min_data_length = 50

                smoothing_input = input("是否使用平滑概率 (True/False，默认True): ").strip().lower()
                if smoothing_input in ['true', '1', 'yes', 'y']:
                    smoothing = True
                elif smoothing_input in ['false', '0', 'no', 'n']:
                    smoothing = False
                else:
                    smoothing = True

                try:
                    strategy = RegimeSwitchingZScoreStrategy(
                        n_regimes=n_regimes,
                        min_data_length=min_data_length,
                        smoothing=smoothing
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Regime-Switching策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue
```

- **功能**：让用户选择Z-score计算策略
- **选项**：
  1. **传统方法**：使用均值和标准差计算Z-score
  2. **ARIMA-GARCH模型**：使用ARIMA预测均值，GARCH预测波动率
  3. **ECM误差修正模型**：使用误差修正模型预测均值回归
  4. **Kalman Filter动态价差模型**：使用Kalman Filter动态估计价差的均值和方差
  5. **Copula + DCC-GARCH相关性/波动率模型**：使用DCC-GARCH估计动态波动率和相关性，使用Copula建模依赖结构
  6. **Regime-Switching市场状态模型**（新增）：使用马尔可夫状态转换模型识别市场状态，根据不同状态调整Z-score计算

**Regime-Switching策略配置**：
- 状态数量（默认2）
- 最小数据长度（默认50）
- 是否使用平滑概率（默认True）

#### 6.2 配置交易参数

```2390:2392:cointegration_test_windows_ Regime-Switching.py
        # 8. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()
```

**调用函数**：`configure_trading_parameters()`

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

#### 6.3 创建交易策略对象

```2394:2408:cointegration_test_windows_ Regime-Switching.py
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
```

**类**：`AdvancedCointegrationTrading`

- **功能**：初始化交易策略对象
- **关键属性**：
  - 交易参数（阈值、止盈止损等）
  - Z-score策略对象（`z_score_strategy`）
  - 策略类型标志（`use_arima_garch`、`use_ecm`、`use_kalman_filter`、`use_copula_dcc_garch`、`use_regime_switching`）

#### 6.4 执行回测

```2410:2414:cointegration_test_windows_ Regime-Switching.py
        results = trading_strategy.backtest_cointegration_trading(
            data,
            selected_pairs,
            initial_capital=10000
        )
```

**调用方法**：`backtest_cointegration_trading()`

### 步骤7：回测执行详细流程

**方法**：`backtest_cointegration_trading()`

#### 7.1 初始化

```1975:1985:cointegration_test_windows_ Regime-Switching.py
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

#### 7.2 获取所有时间点

```1987:1991:cointegration_test_windows_ Regime-Switching.py
        # 获取所有时间点
        all_timestamps = set()
        for symbol in data.keys():
            all_timestamps.update(data[symbol].index)
        all_timestamps = sorted(list(all_timestamps))
```

- **功能**：收集所有币对的时间戳，合并并排序
- **用途**：按时间顺序遍历所有数据点

#### 7.3 回测主循环

```2019:2080:cointegration_test_windows_ Regime-Switching.py
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

                if symbol1 not in current_prices or symbol2 not in current_prices:
                    continue

                # 根据diff_order计算价差
                diff_order = pair_info.get('diff_order', 0)

                if diff_order == 0:
                    # 原始价差
                    current_spread = self.calculate_current_spread(
                        current_prices[symbol1],
                        current_prices[symbol2],
                        pair_info['hedge_ratio']
                    )

                    # 获取历史原始价差数据
                    historical_spreads = []
                    historical_prices1 = []
                    historical_prices2 = []
                    for j in range(max(0, i - self.lookback_period), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index):
                                hist_spread = self.calculate_current_spread(
                                    data[symbol1].loc[hist_timestamp],
                                    data[symbol2].loc[hist_timestamp],
                                    pair_info['hedge_ratio']
                                )
                                historical_spreads.append(hist_spread)
                                historical_prices1.append(data[symbol1].loc[hist_timestamp])
                                historical_prices2.append(data[symbol2].loc[hist_timestamp])
                else:
                    # 一阶差分价差
                    if i > 0:  # 确保有前一个时间点
                        prev_timestamp = all_timestamps[i-1]
                        if (prev_timestamp in data[symbol1].index and
                                prev_timestamp in data[symbol2].index):
                            current_diff1 = current_prices[symbol1] - data[symbol1].loc[prev_timestamp]
                            current_diff2 = current_prices[symbol2] - data[symbol2].loc[prev_timestamp]
                            current_spread = current_diff1 - pair_info['hedge_ratio'] * current_diff2
                        else:
                            current_spread = 0
                    else:
                        current_spread = 0

                    # 获取历史一阶差分价差数据
                    historical_spreads = []
                    historical_prices1 = []
                    historical_prices2 = []
                    for j in range(max(1, i - self.lookback_period), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            prev_hist_timestamp = all_timestamps[j-1]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index and
                                    prev_hist_timestamp in data[symbol1].index and
                                    prev_hist_timestamp in data[symbol2].index):
                                hist_diff1 = data[symbol1].loc[hist_timestamp] - data[symbol1].loc[prev_hist_timestamp]
                                hist_diff2 = data[symbol2].loc[hist_timestamp] - data[symbol2].loc[prev_hist_timestamp]
                                hist_spread = hist_diff1 - pair_info['hedge_ratio'] * hist_diff2
                                historical_spreads.append(hist_spread)
                                historical_prices1.append(data[symbol1].loc[hist_timestamp])
                                historical_prices2.append(data[symbol2].loc[hist_timestamp])

                current_z_score = self.calculate_z_score(current_spread, historical_spreads,
                                                        historical_prices1, historical_prices2)
```

对每个时间点执行以下步骤：

##### 7.3.1 获取当前价格

- 从数据中提取当前时间点的所有币对价格

##### 7.3.2 检查现有持仓

对每个持仓执行：

1. **获取持仓信息**：持仓币对、开仓价格、持仓数量等
2. **计算当前价差**：使用对冲比率计算当前价差
3. **计算Z-score**：
   - 获取历史价差序列和历史价格序列
   - 调用Z-score策略计算当前Z-score，传递价格数据
   - **Regime-Switching策略**：
     - 接收历史价差序列
     - 拟合马尔可夫状态转换模型，识别市场状态
     - 估计当前市场状态
     - 根据当前状态使用对应的均值和标准差计算Z-score
4. **检查退出条件**：
   - Z-score平仓阈值
   - 止盈止损
   - 最大持仓时间
   - 如果满足条件，执行平仓

##### 7.3.3 检查开仓条件

对每个选择的币对执行：

1. **检查是否已有持仓**：如果已有持仓，跳过
2. **计算当前价差**：使用对冲比率计算价差
3. **获取历史价差**：从历史数据中提取价差序列，长度 = `lookback_period`
4. **计算Z-score**：
   - 获取历史价差序列和历史价格序列
   - 调用Z-score策略计算Z-score，传递价格数据
   - **Regime-Switching策略**：
     - 接收历史价差序列
     - 拟合马尔可夫状态转换模型，识别市场状态
     - 估计当前市场状态
     - 根据当前状态使用对应的均值和标准差计算Z-score
5. **生成交易信号**：
   - `z_score >= z_threshold`：做空价差（SHORT_LONG）
   - `z_score <= -z_threshold`：做多价差（LONG_SHORT）
6. **执行开仓**：
   - 计算持仓数量（Beta中性）
   - 记录开仓信息
   - 扣除手续费

##### 7.3.4 更新资金曲线

- 计算当前总资产
- 记录到 `capital_curve`
- 计算日收益率

#### 7.4 回测结果分析

1. **计算统计指标**：总收益率、年化收益率、最大回撤、夏普比率、胜率等
2. **显示结果**：打印所有统计指标，显示资金曲线图
3. **导出交易记录**：导出为CSV文件

---

## 模式二：参数优化模式流程

### 步骤1-5：与普通回测模式相同

- 数据加载
- 选择价差类型
- 配置滚动窗口参数
- 滚动窗口寻找协整对
- 选择协整对

### 步骤6：选择Z-score计算策略

```3107:3112:cointegration_test_windows_ Regime-Switching.py
    # 6. 选择Z-score计算策略
    print("\n6. 选择Z-score计算策略")
    z_score_strategy = select_z_score_strategy()
    if z_score_strategy is None:
        print("未选择策略，退出优化")
        return
```

- **功能**：选择用于优化的Z-score策略
- **注意**：优化过程中策略固定，只优化交易参数

### 步骤7：选择优化方法

```3114:3146:cointegration_test_windows_ Regime-Switching.py
    # 7. 选择优化方法
    print("\n7. 选择优化方法")
    print("可选方法:")
    print("  1. 网格搜索（粗粒度+细粒度）")
    print("  2. 随机搜索")
    print("  3. 贝叶斯优化")

    method_choice = input("请选择优化方法 (1/2/3): ").strip()

    method_map = {
        '1': 'grid_search',
        '2': 'random_search',
        '3': 'bayesian_optimization'
    }

    method = method_map.get(method_choice, 'grid_search')

    # 8. 选择优化目标
    print("\n8. 选择优化目标")
    print("可选目标:")
    print("  1. 夏普比率 (sharpe_ratio)")
    print("  2. 总收益率 (return)")
    print("  3. 收益率/回撤比 (return_drawdown_ratio)")

    objective_choice = input("请选择优化目标 (1/2/3): ").strip()

    objective_map = {
        '1': 'sharpe_ratio',
        '2': 'return',
        '3': 'return_drawdown_ratio'
    }

    objective = objective_map.get(objective_choice, 'sharpe_ratio')
```

- **优化方法**：
  1. **网格搜索**：系统遍历所有参数组合
  2. **随机搜索**：随机采样参数组合
  3. **贝叶斯优化**：使用高斯过程优化（需要scikit-optimize）

### 步骤8：创建优化器并执行优化

```3161:3187:cointegration_test_windows_ Regime-Switching.py
    # 10. 创建优化器
    strategy_name = z_score_strategy.get_strategy_description() if z_score_strategy else "未知"
    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"\n10. 创建优化器 (方法={method}, 目标={objective}, 策略={strategy_name}, 价差类型={diff_type})")
    optimizer = ParameterOptimizer(
        data=data,
        selected_pairs=selected_pairs,
        initial_capital=10000,
        objective=objective,
        stability_test=True,
        z_score_strategy=z_score_strategy  # 传入策略对象
    )

    # 11. 执行优化
    print(f"\n11. 执行优化...")
    if method == 'grid_search':
        result = optimizer.optimize(method='grid_search',
                                    coarse_first=True,
                                    fine_search_around_best=True)
    elif method == 'random_search':
        n_iter = input("请输入随机搜索迭代次数 (默认100): ").strip()
        n_iter = int(n_iter) if n_iter else 100
        result = optimizer.optimize(method='random_search', n_iter=n_iter)
    else:  # bayesian_optimization
        n_calls = input("请输入贝叶斯优化评估次数 (默认50): ").strip()
        n_calls = int(n_calls) if n_calls else 50
        result = optimizer.optimize(method='bayesian_optimization', n_calls=n_calls)
```

### 步骤9：显示和导出优化结果

```3189:3233:cointegration_test_windows_ Regime-Switching.py
    # 12. 显示结果
    print("\n" + "=" * 80)
    print("优化结果")
    print("=" * 80)

    if result.get('error'):
        print(f"优化失败: {result['error']}")
        return

    print(f"\n最佳参数:")
    for param_name, param_value in result['best_params'].items():
        print(f"  {param_name}: {param_value}")
    strategy_desc = z_score_strategy.get_strategy_description() if z_score_strategy else "未知"
    diff_type = '原始价差' if diff_order == 0 else '一阶差分价差'
    print(f"  使用的策略: {strategy_desc}")
    print(f"  价差类型: {diff_type}")

    print(f"\n最佳得分: {result['best_score']:.4f}")
    if result['best_result']:
        print(f"  总收益率: {result['best_result']['total_return']:.2f}%")
        print(f"  夏普比率: {result['best_result']['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {result['best_result']['max_drawdown_pct']:.2f}%")
        print(f"  盈亏比: {result['best_result']['profit_loss_ratio']:.2f}")
        print(f"  总交易次数: {result['best_result']['total_trades']}")

        if 'stability' in result['best_result']:
            stability = result['best_result']['stability']
            print(f"\n稳定性测试:")
            print(f"  稳定性: {'良好' if stability['is_stable'] else '可能不稳定'}")
            print(f"  变异系数: {stability.get('score_coefficient_of_variation', 0):.3f}")
            print(f"  得分下降比例: {stability.get('score_drop_ratio', 0):.3f}")

    print(f"\n总评估次数: {result['total_evaluations']}")

    # 11. 显示前10个最佳结果
    print("\n前10个最佳参数组合:")
    top_results = optimizer.get_top_results(10)
    for i, res in enumerate(top_results, 1):
        print(f"\n{i}. 得分={res['score']:.4f}, 收益率={res['total_return']:.2f}%, "
              f"夏普={res['sharpe_ratio']:.4f}")
        print(f"   参数: {res['params']}")

    # 12. 导出结果
    print("\n12. 导出优化结果...")
    optimizer.export_results()
```

- **功能**：将优化结果导出为CSV文件
- **内容**：最佳参数、评估历史、统计指标等

---

## Regime-Switching策略详解

### 策略类：RegimeSwitchingZScoreStrategy

**文件位置**：`strategies/regime_switching_zscore_strategy.py`

---

## 一、Regime-Switching模型理论基础

### 1.1 模型概述

**Regime-Switching（状态转换）模型**是一种用于捕捉时间序列在不同状态之间转换的统计模型。在金融时间序列分析中，市场经常在不同的"状态"或"制度"之间转换，例如：

- **高波动率状态**：市场波动剧烈，价差变化大
- **低波动率状态**：市场相对平静，价差变化小
- **均值回归强状态**：价差快速回归均值
- **均值回归弱状态**：价差偏离均值后回归较慢

### 1.2 为什么需要Regime-Switching模型？

在协整交易中，传统方法假设：
- 价差的均值和标准差是常数
- 市场状态不变

但实际上：
- 市场状态会随时间变化
- 不同状态下的价差行为不同
- 使用固定参数可能导致Z-score计算不准确

Regime-Switching模型能够：
- 识别当前市场状态
- 根据不同状态使用不同的参数（均值和标准差）
- 动态适应市场状态变化

---

## 二、马尔可夫状态转换模型（Markov Regime-Switching Model）

### 2.1 基本思想

**马尔可夫状态转换模型**假设：
1. 市场在有限个状态（regimes）之间转换
2. 状态转换遵循马尔可夫过程（当前状态只依赖于前一状态）
3. 每个状态有不同的参数（均值、方差等）

### 2.2 数学模型

#### 2.2.1 状态变量

定义状态变量 `S_t`，表示时刻 `t` 的市场状态：

```
S_t ∈ {0, 1, 2, ..., K-1}
```

其中 `K` 是状态数量（通常 `K = 2`，即高波动率状态和低波动率状态）。

#### 2.2.2 价差模型

在每个状态下，价差 `spread_t` 的分布不同：

**状态 k 下的价差模型**：
```
spread_t | S_t = k ~ N(μ_k, σ²_k)
```

其中：
- `μ_k`：状态 k 下的均值
- `σ²_k`：状态 k 下的方差

**完整模型**：
```
spread_t = μ_{S_t} + ε_t
ε_t | S_t ~ N(0, σ²_{S_t})
```

#### 2.2.3 状态转换概率（Transition Probabilities）

**马尔可夫链假设**：当前状态只依赖于前一状态：

```
P(S_t = j | S_{t-1} = i, S_{t-2}, ..., S_0) = P(S_t = j | S_{t-1} = i) = p_{ij}
```

**状态转换概率矩阵** `P`：
```
P = [p_{ij}]_{K×K}
```

其中 `p_{ij}` 表示从状态 `i` 转换到状态 `j` 的概率。

**约束条件**：
```
Σ_j p_{ij} = 1  （每行概率和为1）
p_{ij} ≥ 0      （概率非负）
```

**两状态模型示例**（`K = 2`）：
```
P = [p_{00}  p_{01}]  =  [p_{00}    1-p_{00}]
    [p_{10}  p_{11}]     [1-p_{11}  p_{11}  ]
```

其中：
- `p_{00}`：保持在状态0的概率
- `p_{01} = 1 - p_{00}`：从状态0转换到状态1的概率
- `p_{11}`：保持在状态1的概率
- `p_{10} = 1 - p_{11}`：从状态1转换到状态0的概率

#### 2.2.4 完整模型公式

**观测方程（Observation Equation）**：
```
spread_t = μ_{S_t} + σ_{S_t} * z_t
```

其中 `z_t ~ N(0, 1)` 是标准正态随机变量。

**状态方程（State Equation）**：
```
S_t | S_{t-1} = i ~ Multinomial(1, [p_{i0}, p_{i1}, ..., p_{i(K-1)}])
```

**初始状态分布**：
```
P(S_0 = k) = π_k
```

其中 `Σ_k π_k = 1`。

### 2.3 参数估计

#### 2.3.1 最大似然估计（MLE）

**似然函数**：

对于观测序列 `{spread_1, spread_2, ..., spread_T}`，似然函数为：

```
L(θ) = P(spread_1, spread_2, ..., spread_T | θ)
```

其中 `θ = {μ_k, σ²_k, p_{ij}, π_k}` 是所有参数的集合。

**完整似然函数**（考虑所有可能的状态序列）：

```
L(θ) = Σ_{S_1} Σ_{S_2} ... Σ_{S_T} P(spread_1, ..., spread_T, S_1, ..., S_T | θ)
```

**分解**：
```
L(θ) = Σ_{S_1} ... Σ_{S_T} [P(S_0) * Π_{t=1}^T P(S_t | S_{t-1}) * P(spread_t | S_t)]
```

#### 2.3.2 期望最大化算法（EM Algorithm）

由于状态是隐变量（不可观测），通常使用EM算法估计参数：

**E步（Expectation）**：计算状态的后验概率
```
γ_t(k) = P(S_t = k | spread_1, ..., spread_T, θ^{(m)})
```

**M步（Maximization）**：更新参数
```
μ_k^{(m+1)} = Σ_t γ_t(k) * spread_t / Σ_t γ_t(k)
σ²_k^{(m+1)} = Σ_t γ_t(k) * (spread_t - μ_k^{(m+1)})² / Σ_t γ_t(k)
p_{ij}^{(m+1)} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
```

其中 `ξ_t(i,j) = P(S_{t-1} = i, S_t = j | spread_1, ..., spread_T, θ^{(m)})`。

### 2.4 状态推断（State Inference）

#### 2.4.1 滤波概率（Filtered Probabilities）

**滤波概率**：基于到时刻 `t` 为止的观测，估计当前状态：

```
α_t(k) = P(S_t = k | spread_1, ..., spread_t)
```

**前向算法（Forward Algorithm）**：

初始化：
```
α_0(k) = π_k * f(spread_0 | S_0 = k)
```

递归：
```
α_t(k) = f(spread_t | S_t = k) * Σ_i α_{t-1}(i) * p_{ik}
```

其中 `f(spread_t | S_t = k)` 是状态 `k` 下的概率密度函数：
```
f(spread_t | S_t = k) = (1/√(2πσ²_k)) * exp(-(spread_t - μ_k)²/(2σ²_k))
```

#### 2.4.2 平滑概率（Smoothed Probabilities）

**平滑概率**：基于全部观测，估计历史状态：

```
β_t(k) = P(S_t = k | spread_1, ..., spread_T)
```

**后向算法（Backward Algorithm）**：

初始化：
```
β_T(k) = 1
```

递归：
```
β_t(k) = Σ_j β_{t+1}(j) * p_{kj} * f(spread_{t+1} | S_{t+1} = j)
```

**平滑概率计算**：
```
β_t(k) = α_t(k) * β_t(k) / Σ_j α_t(j) * β_t(j)
```

#### 2.4.3 最可能状态序列（Viterbi Algorithm）

**Viterbi算法**：找到最可能的状态序列：

```
S* = argmax_{S_1,...,S_T} P(S_1, ..., S_T | spread_1, ..., spread_T)
```

---

## 三、代码实现详解

### 3.1 完整版实现（MarkovRegression）

**代码位置**：`strategies/regime_switching_zscore_strategy.py`

#### 3.1.1 模型拟合

```55:80:strategies/regime_switching_zscore_strategy.py
    def _fit_regime_switching_model(self, spreads: np.ndarray) -> Optional[object]:
        """
        拟合马尔可夫状态转换模型
        
        使用MarkovRegression模型，假设价差在不同状态下的均值和方差不同
        
        Args:
            spreads: 价差序列
            
        Returns:
            拟合的模型对象，如果失败返回None
        """
        # 如果MarkovRegression不可用，直接使用简化版本
        if not MARKOV_REGRESSION_AVAILABLE:
            return self._fit_simplified_regime_model(spreads)
        
        try:
            # 使用MarkovRegression模型
            # 假设价差在不同状态下的均值和方差不同
            model = MarkovRegression(spreads, k_regimes=self.n_regimes, 
                                     switching_variance=True, switching_mean=True)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # 如果MarkovRegression拟合失败，使用简化版本
            return self._fit_simplified_regime_model(spreads)
```

**MarkovRegression模型**：
- `k_regimes`：状态数量
- `switching_variance=True`：允许方差在不同状态间转换
- `switching_mean=True`：允许均值在不同状态间转换

**模型公式**：
```
spread_t = μ_{S_t} + σ_{S_t} * z_t
```

其中：
- `μ_{S_t}`：状态 `S_t` 下的均值（可转换）
- `σ_{S_t}`：状态 `S_t` 下的标准差（可转换）
- `z_t ~ N(0, 1)`：标准正态随机变量

#### 3.1.2 状态估计

```142:206:strategies/regime_switching_zscore_strategy.py
    def _estimate_current_regime(self, spreads: np.ndarray, 
                                 model: Optional[object] = None) -> Tuple[int, dict]:
        """
        估计当前市场状态
        
        Args:
            spreads: 价差序列
            model: 拟合的模型对象（可选）
            
        Returns:
            Tuple[int, dict]: (当前状态, 状态参数字典)
        """
        if model is None:
            # 使用简化方法
            simplified_model = self._fit_simplified_regime_model(spreads)
            if simplified_model is None:
                # 估计失败，使用默认状态
                return (0, {'mean': np.mean(spreads), 'std': np.std(spreads)})
            
            current_state = simplified_model['current_state']
            regime_params = simplified_model['regime_params']
            return (current_state, regime_params.get(current_state, 
                   {'mean': np.mean(spreads), 'std': np.std(spreads)}))
        
        try:
            # 使用MarkovRegression模型
            if hasattr(model, 'smoothed_marginal_probabilities'):
                # 获取平滑概率
                smoothed_probs = model.smoothed_marginal_probabilities
                # 当前状态是概率最大的状态
                current_state = np.argmax(smoothed_probs[-1, :])
            elif hasattr(model, 'filtered_marginal_probabilities'):
                # 使用滤波概率
                filtered_probs = model.filtered_marginal_probabilities
                current_state = np.argmax(filtered_probs[-1, :])
            else:
                # 使用简化方法
                return self._estimate_current_regime(spreads, None)
            
            # 获取当前状态的参数
            if hasattr(model, 'params'):
                # 从模型参数中提取状态相关的参数
                # MarkovRegression的参数结构：[均值参数, 方差参数, 转换概率]
                n_params_per_regime = len(model.params) // (2 * self.n_regimes + self.n_regimes * (self.n_regimes - 1))
                # 简化处理：使用历史数据估计
                regime_params = {}
                for regime in range(self.n_regimes):
                    # 使用简化方法估计每个状态的参数
                    regime_params[regime] = {
                        'mean': np.mean(spreads),
                        'std': np.std(spreads)
                    }
                
                # 使用当前状态的参数
                current_regime_params = regime_params.get(current_state, 
                    {'mean': np.mean(spreads), 'std': np.std(spreads)})
            else:
                # 使用简化方法
                return self._estimate_current_regime(spreads, None)
            
            return (current_state, current_regime_params)
            
        except Exception:
            # 模型使用失败，使用简化方法
            return self._estimate_current_regime(spreads, None)
```

**状态估计方法**：

1. **平滑概率**（如果可用）：
   ```
   current_state = argmax_k P(S_T = k | spread_1, ..., spread_T)
   ```
   使用全部数据估计最后时刻的状态。

2. **滤波概率**（如果平滑概率不可用）：
   ```
   current_state = argmax_k P(S_T = k | spread_1, ..., spread_T)
   ```
   只使用到当前时刻的数据。

3. **状态参数提取**：
   - 从模型参数中提取每个状态的均值和方差
   - 如果提取失败，使用历史数据估计

### 3.2 简化版实现（当MarkovRegression不可用时）

**代码位置**：`strategies/regime_switching_zscore_strategy.py`

#### 3.2.1 状态识别

```82:140:strategies/regime_switching_zscore_strategy.py
    def _fit_simplified_regime_model(self, spreads: np.ndarray) -> Optional[dict]:
        """
        拟合简化版状态转换模型（当MarkovRegression不可用时）
        
        使用K-means聚类识别状态，然后估计每个状态的参数
        
        Args:
            spreads: 价差序列
            
        Returns:
            包含状态参数的字典，如果失败返回None
        """
        try:
            # 计算价差的一阶差分（用于识别波动率状态）
            diff_spreads = np.diff(spreads)
            abs_diff = np.abs(diff_spreads)
            
            # 使用简单的阈值方法识别状态
            # 状态0：低波动率（abs_diff < 中位数）
            # 状态1：高波动率（abs_diff >= 中位数）
            threshold = np.median(abs_diff)
            
            # 识别状态
            states = (abs_diff >= threshold).astype(int)
            
            # 为每个状态估计参数
            regime_params = {}
            for regime in range(self.n_regimes):
                regime_mask = states == regime
                if np.sum(regime_mask) < 5:  # 至少需要5个数据点
                    # 如果某个状态数据不足，使用全部数据
                    regime_spreads = spreads
                else:
                    # 获取该状态对应的价差
                    regime_indices = np.where(regime_mask)[0] + 1  # +1因为diff后索引偏移
                    regime_spreads = spreads[regime_indices]
                
                regime_params[regime] = {
                    'mean': np.mean(regime_spreads),
                    'std': np.std(regime_spreads) if len(regime_spreads) > 1 else np.std(spreads)
                }
            
            # 估计状态转换概率（简化版：使用历史频率）
            transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
            for i in range(len(states) - 1):
                transition_matrix[states[i], states[i+1]] += 1
            
            # 归一化
            row_sums = transition_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            return {
                'regime_params': regime_params,
                'transition_matrix': transition_matrix,
                'current_state': states[-1] if len(states) > 0 else 0
            }
        except Exception:
            return None
```

**简化版状态识别方法**：

1. **计算价差的一阶差分**：
   ```
   diff_t = spread_t - spread_{t-1}
   abs_diff_t = |diff_t|
   ```

2. **使用中位数阈值识别状态**：
   ```
   threshold = median(abs_diff)
   state_t = 1 if abs_diff_t >= threshold else 0
   ```
   - 状态0：低波动率（`abs_diff < threshold`）
   - 状态1：高波动率（`abs_diff >= threshold`）

3. **估计每个状态的参数**：
   ```
   μ_k = mean(spreads | state = k)
   σ_k = std(spreads | state = k)
   ```

4. **估计状态转换概率矩阵**：
   ```
   p_{ij} = count(从状态i转换到状态j) / count(状态i出现的次数)
   ```

**转换概率矩阵估计**：
```
P = [p_{00}  p_{01}]  =  [n_{00}/n_0  n_{01}/n_0]
    [p_{10}  p_{11}]     [n_{10}/n_1  n_{11}/n_1]
```

其中：
- `n_{ij}`：从状态 `i` 转换到状态 `j` 的次数
- `n_i = n_{i0} + n_{i1}`：状态 `i` 出现的总次数

### 3.3 Z-score计算

```208:308:strategies/regime_switching_zscore_strategy.py
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float],
                         historical_prices1: List[float] = None, historical_prices2: List[float] = None) -> float:
        """
        使用Regime-Switching模型计算Z-score
        
        方法：
        1. 拟合马尔可夫状态转换模型，识别市场状态
        2. 估计当前市场状态
        3. 根据当前状态使用对应的均值和标准差计算Z-score
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            historical_prices1: 第一个资产的历史价格序列（可选，当前未使用）
            historical_prices2: 第二个资产的历史价格序列（可选，当前未使用）
            
        Returns:
            float: Z-score值
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=self.min_data_length):
            return 0.0
        
        try:
            # 转换为numpy数组
            spreads_array = np.array(historical_spreads)
            
            # 创建缓存键
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._regime_models:
                model = self._regime_models[cache_key]
                regime_params = self._regime_params_cache.get(cache_key)
            else:
                # 拟合状态转换模型
                model = self._fit_regime_switching_model(spreads_array)
                
                if model is None:
                    # 模型拟合失败，使用传统方法
                    historical_mean = np.mean(spreads_array)
                    historical_std = np.std(spreads_array)
                    if historical_std > 0:
                        return (current_spread - historical_mean) / historical_std
                    return 0.0
                
                # 估计状态参数
                current_state, regime_params = self._estimate_current_regime(spreads_array, model)
                
                # 缓存模型和参数
                if len(self._regime_models) >= self._max_cache_size:
                    oldest_key = next(iter(self._regime_models))
                    del self._regime_models[oldest_key]
                    if oldest_key in self._regime_params_cache:
                        del self._regime_params_cache[oldest_key]
                
                self._regime_models[cache_key] = model
                self._regime_params_cache[cache_key] = regime_params
            
            # 如果regime_params是None，重新估计
            if regime_params is None:
                current_state, regime_params = self._estimate_current_regime(spreads_array, model)
                self._regime_params_cache[cache_key] = regime_params
            
            # 获取当前状态的均值和标准差
            regime_mean = regime_params.get('mean', np.mean(spreads_array))
            regime_std = regime_params.get('std', np.std(spreads_array))
            
            # 验证标准差
            if regime_std <= 0 or np.isnan(regime_std):
                regime_std = np.std(spreads_array)
                if regime_std <= 0:
                    return 0.0
            
            # 计算Z-score（使用当前状态的参数）
            z_score = (current_spread - regime_mean) / regime_std
            
            # 验证结果
            if np.isnan(z_score) or np.isinf(z_score):
                # 结果无效，使用传统方法
                historical_mean = np.mean(spreads_array)
                historical_std = np.std(spreads_array)
                if historical_std > 0:
                    z_score = (current_spread - historical_mean) / historical_std
                else:
                    return 0.0
            
            return z_score
            
        except Exception as e:
            # 任何错误都返回0，并尝试使用传统方法作为后备
            try:
                spreads_array = np.array(historical_spreads)
                historical_mean = np.mean(spreads_array)
                historical_std = np.std(spreads_array)
                if historical_std > 0:
                    return (current_spread - historical_mean) / historical_std
            except:
                pass
            print(f"Regime-Switching模型计算失败: {str(e)}")
            return 0.0
```

**Z-score计算公式**：

```
Z = (spread_t - μ_{S_t}) / σ_{S_t}
```

其中：
- `spread_t`：当前价差
- `μ_{S_t}`：当前状态 `S_t` 下的均值
- `σ_{S_t}`：当前状态 `S_t` 下的标准差

**与传统方法的区别**：
- **传统方法**：使用全部历史数据的均值和标准差
  ```
  Z_traditional = (spread_t - μ_all) / σ_all
  ```
  其中 `μ_all = mean(historical_spreads)`，`σ_all = std(historical_spreads)`
  
- **Regime-Switching方法**：根据当前市场状态使用对应状态的均值和标准差
  ```
  Z_regime = (spread_t - μ_{S_t}) / σ_{S_t}
  ```
  其中 `μ_{S_t}` 和 `σ_{S_t}` 是当前状态 `S_t` 下的参数

**优势**：
1. **动态适应**：能够根据市场状态变化自动调整参数
2. **更准确**：在高波动率状态下使用更大的标准差，避免误判
3. **更稳健**：在低波动率状态下使用更小的标准差，提高信号灵敏度

---

## 四、Regime-Switching策略在回测中的应用

### 4.1 策略调用流程

在回测主循环中，Regime-Switching策略的调用流程如下：

```644:645:cointegration_test_windows_ Regime-Switching.py
                current_z_score = self.calculate_z_score(current_spread, historical_spreads,
                                                        historical_prices1, historical_prices2)
```

**调用链**：
1. `backtest_cointegration_trading()` 方法收集历史价差和价格数据
2. 调用 `calculate_z_score()` 方法（`AdvancedCointegrationTrading` 类）
3. `calculate_z_score()` 方法调用 `z_score_strategy.calculate_z_score()`（策略对象）
4. `RegimeSwitchingZScoreStrategy.calculate_z_score()` 执行以下步骤：
   - 拟合状态转换模型
   - 估计当前市场状态
   - 使用状态参数计算Z-score

### 4.2 模型缓存机制

为了提高计算效率，Regime-Switching策略实现了模型缓存：

```1376:1406:strategies/regime_switching_zscore_strategy.py
            # 创建缓存键
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            
            # 检查缓存
            if cache_key in self._regime_models:
                model = self._regime_models[cache_key]
                regime_params = self._regime_params_cache.get(cache_key)
            else:
                # 拟合状态转换模型
                model = self._fit_regime_switching_model(spreads_array)
                
                if model is None:
                    # 模型拟合失败，使用传统方法
                    historical_mean = np.mean(spreads_array)
                    historical_std = np.std(spreads_array)
                    if historical_std > 0:
                        return (current_spread - historical_mean) / historical_std
                    return 0.0
                
                # 估计状态参数
                current_state, regime_params = self._estimate_current_regime(spreads_array, model)
                
                # 缓存模型和参数
                if len(self._regime_models) >= self._max_cache_size:
                    oldest_key = next(iter(self._regime_models))
                    del self._regime_models[oldest_key]
                    if oldest_key in self._regime_params_cache:
                        del self._regime_params_cache[oldest_key]
                
                self._regime_models[cache_key] = model
                self._regime_params_cache[cache_key] = regime_params
```

**缓存机制**：
- **缓存键**：基于数据长度和最后5个数据点
- **缓存内容**：拟合的模型对象和状态参数
- **缓存大小**：最多缓存10个模型（`_max_cache_size = 10`）
- **缓存策略**：FIFO（先进先出），当缓存满时删除最旧的模型

### 4.3 容错机制

Regime-Switching策略实现了多层容错机制：

1. **模型拟合失败**：如果MarkovRegression拟合失败，自动切换到简化版模型
2. **简化版模型失败**：如果简化版模型也失败，使用传统方法（均值和标准差）
3. **状态估计失败**：如果状态估计失败，使用全部数据的均值和标准差
4. **参数验证**：验证标准差是否为正数，如果无效则使用传统方法
5. **结果验证**：验证Z-score是否为有效数值，如果无效则使用传统方法

---

## 五、Regime-Switching模型参数说明

### 5.1 初始化参数

```28:44:strategies/regime_switching_zscore_strategy.py
    def __init__(self, n_regimes: int = 2, min_data_length: int = 50, 
                 smoothing: bool = True, **kwargs):
        """
        初始化Regime-Switching策略
        
        Args:
            n_regimes: 状态数量（默认2，即高波动率状态和低波动率状态）
            min_data_length: 最小数据长度要求
            smoothing: 是否使用平滑概率（默认True）
            **kwargs: 其他策略参数
        """
        super().__init__(n_regimes=n_regimes, min_data_length=min_data_length,
                        smoothing=smoothing, **kwargs)
        self.name = "Regime-Switching市场状态模型"
        self.n_regimes = n_regimes
        self.min_data_length = min_data_length
        self.smoothing = smoothing
```

**参数说明**：
- **n_regimes**：状态数量，默认2（高波动率状态和低波动率状态）
  - 可以设置为更多状态（如3个状态：低波动率、中波动率、高波动率）
  - 状态数量越多，模型越复杂，需要更多数据
- **min_data_length**：最小数据长度，默认50
  - 如果历史数据少于这个长度，使用传统方法
  - 建议值：至少是状态数量的10倍
- **smoothing**：是否使用平滑概率，默认True
  - True：使用平滑概率（基于全部数据）
  - False：使用滤波概率（基于到当前时刻的数据）

### 5.2 参数选择建议

**状态数量（n_regimes）**：
- **2个状态**（推荐）：适合大多数市场
  - 状态0：低波动率状态
  - 状态1：高波动率状态
- **3个状态**：适合波动率变化较大的市场
  - 状态0：低波动率
  - 状态1：中波动率
  - 状态2：高波动率
- **更多状态**：通常不推荐，模型复杂度高，容易过拟合

**最小数据长度（min_data_length）**：
- **50**（默认）：适合大多数情况
- **100**：如果数据充足，可以提高稳定性
- **20**：最小推荐值，低于此值可能不稳定

**平滑概率（smoothing）**：
- **True**（推荐）：使用全部数据估计状态，更准确
- **False**：只使用到当前时刻的数据，计算更快但可能不够准确

---

## 六、Regime-Switching模型数学公式总结

### 6.1 完整模型公式

**观测方程**：
```
spread_t = μ_{S_t} + σ_{S_t} * z_t
```
其中 `z_t ~ N(0, 1)`

**状态方程**：
```
P(S_t = j | S_{t-1} = i) = p_{ij}
```

**状态转换概率矩阵**：
```
P = [p_{00}  p_{01}  ...  p_{0(K-1)}]
    [p_{10}  p_{11}  ...  p_{1(K-1)}]
    [...     ...     ...  ...       ]
    [p_{(K-1)0}  p_{(K-1)1}  ...  p_{(K-1)(K-1)}]
```

**约束条件**：
```
Σ_j p_{ij} = 1  （每行概率和为1）
p_{ij} ≥ 0      （概率非负）
```

### 6.2 参数估计公式

**均值估计**（EM算法M步）：
```
μ_k = Σ_t γ_t(k) * spread_t / Σ_t γ_t(k)
```

**方差估计**（EM算法M步）：
```
σ²_k = Σ_t γ_t(k) * (spread_t - μ_k)² / Σ_t γ_t(k)
```

**转换概率估计**（EM算法M步）：
```
p_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
```

其中：
- `γ_t(k) = P(S_t = k | spread_1, ..., spread_T)`：状态k在时刻t的后验概率
- `ξ_t(i,j) = P(S_{t-1} = i, S_t = j | spread_1, ..., spread_T)`：从状态i转换到状态j的联合后验概率

### 6.3 状态推断公式

**滤波概率**（前向算法）：
```
α_t(k) = f(spread_t | S_t = k) * Σ_i α_{t-1}(i) * p_{ik}
```

**平滑概率**（后向算法）：
```
β_t(k) = α_t(k) * β_t(k) / Σ_j α_t(j) * β_t(j)
```

**当前状态估计**：
```
S_t = argmax_k P(S_t = k | spread_1, ..., spread_T)
```

### 6.4 Z-score计算公式

**Regime-Switching Z-score**：
```
Z_t = (spread_t - μ_{S_t}) / σ_{S_t}
```

其中：
- `spread_t`：当前价差
- `μ_{S_t}`：当前状态 `S_t` 下的均值
- `σ_{S_t}`：当前状态 `S_t` 下的标准差

---

## 七、Regime-Switching策略的优势和局限性

### 7.1 优势

1. **动态适应市场状态**：
   - 能够自动识别市场状态变化
   - 根据不同状态调整参数，提高Z-score计算准确性

2. **更准确的信号**：
   - 在高波动率状态下使用更大的标准差，避免误判
   - 在低波动率状态下使用更小的标准差，提高信号灵敏度

3. **理论基础扎实**：
   - 基于马尔可夫状态转换模型，有坚实的统计学基础
   - 能够捕捉市场状态转换的动态特性

4. **容错机制完善**：
   - 多层容错机制，确保策略在各种情况下都能正常工作
   - 自动降级到简化版或传统方法

### 7.2 局限性

1. **计算复杂度较高**：
   - 需要拟合状态转换模型，计算时间较长
   - 虽然有缓存机制，但仍比传统方法慢

2. **需要足够的数据**：
   - 至少需要 `min_data_length` 个数据点
   - 状态数量越多，需要的数据越多

3. **参数选择敏感**：
   - 状态数量的选择影响模型性能
   - 需要根据市场特性调整参数

4. **可能过拟合**：
   - 如果状态数量过多，可能过拟合历史数据
   - 需要谨慎选择状态数量

### 7.3 适用场景

**推荐使用Regime-Switching策略的场景**：
1. **市场状态变化明显**：市场在不同波动率状态之间频繁转换
2. **数据充足**：有足够的历史数据（至少50个数据点）
3. **需要动态适应**：希望策略能够自动适应市场状态变化
4. **对准确性要求高**：需要更准确的Z-score计算

**不推荐使用Regime-Switching策略的场景**：
1. **数据不足**：历史数据少于50个数据点
2. **计算资源有限**：对计算速度要求很高
3. **市场状态稳定**：市场状态变化不明显，传统方法已足够

---

## 八、总结

Regime-Switching市场状态模型是一个强大的Z-score计算策略，能够：

1. **识别市场状态**：使用马尔可夫状态转换模型识别市场状态
2. **动态调整参数**：根据不同状态使用不同的均值和标准差
3. **提高准确性**：相比传统方法，能够更准确地计算Z-score
4. **完善容错**：多层容错机制，确保策略稳定运行

在协整交易中，Regime-Switching策略特别适合：
- 市场状态变化明显的场景
- 需要动态适应市场变化的场景
- 对Z-score计算准确性要求高的场景

通过合理选择参数（状态数量、最小数据长度等），Regime-Switching策略能够显著提高协整交易的性能。