# cointegration_test_windows_Copula_DCC-GARCH.py 详细说明文档

## 文件概述

本文件是一个高级协整分析+交易流程完整测试程序，支持滚动窗口协整检验、参数优化和多种Z-score计算策略（包括新增的Copula + DCC-GARCH相关性/波动率模型）。

### 主要功能

1. **滚动窗口协整检验**：使用固定大小的滚动窗口进行协整关系检验，识别时变特性
2. **参数优化**：支持网格搜索、随机搜索、贝叶斯优化三种方法
3. **多种Z-score策略**：
   - 传统方法（均值和标准差）
   - ARIMA-GARCH模型
   - ECM误差修正模型
   - Kalman Filter动态价差模型
   - **Copula + DCC-GARCH相关性/波动率模型**（新增）
4. **完整交易回测**：包含开仓、平仓、止盈止损、手续费计算等

---

## 程序运行流程

### 阶段一：程序启动和初始化

#### 1. 程序入口

```3232:3233:cointegration_test_windows_Copula_DCC-GARCH.py
if __name__ == "__main__":
    main()
```

- **执行时机**：直接运行脚本时
- **功能**：调用主函数 `main()`

#### 2. 主函数初始化

```3193:3209:cointegration_test_windows_Copula_DCC-GARCH.py
def main():
    """
    主函数
    """
    print("滚动窗口协整分析+交易流程完整测试（带参数优化+Copula + DCC-GARCH相关性/波动率模型）")
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
    print()
```

- **功能**：显示程序介绍和功能说明
- **输出**：程序标题和功能列表

#### 3. 模式选择

```3211:3229:cointegration_test_windows_Copula_DCC-GARCH.py
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

```2243:2250:cointegration_test_windows_Copula_DCC-GARCH.py
    # 1. 加载数据
    print("\n1. 加载数据")
    data = load_csv_data(csv_file_path)
    if data is None:
        print("数据加载失败")
        return

    print(f"成功加载 {len(data)} 个币对的数据")
```

**调用函数**：`load_csv_data(csv_file_path)`

```87:127:cointegration_test_windows_Copula_DCC-GARCH.py
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

```2252:2254:cointegration_test_windows_Copula_DCC-GARCH.py
    # 2. 选择价差类型
    print("\n2. 选择价差类型")
    diff_order = select_spread_type()
```

**调用函数**：`select_spread_type()`（第602-644行）

- **功能**：让用户选择使用原始价差还是差分价差
- **选项**：
  - `0`：原始价差（spread = price1 - hedge_ratio * price2）
  - `1`：一阶差分价差（spread_diff = spread[t] - spread[t-1]）
- **返回值**：`diff_order`（0 或 1）

### 步骤3：配置滚动窗口参数

```2256:2258:cointegration_test_windows_Copula_DCC-GARCH.py
    # 3. 配置滚动窗口参数
    print("\n3. 配置滚动窗口参数")
    window_params = configure_rolling_window_parameters()
```

**调用函数**：`configure_rolling_window_parameters()`（第645-691行）

- **功能**：配置滚动窗口的大小和步长
- **参数**：
  - `window_size`：窗口大小（默认1000）
  - `step_size`：步长（默认100）
- **返回值**：字典 `{'window_size': int, 'step_size': int}`

### 步骤4：滚动窗口寻找协整对

```2260:2273:cointegration_test_windows_Copula_DCC-GARCH.py
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

**调用函数**：`rolling_window_find_cointegrated_pairs()`（第552-601行）

- **功能**：
  - 对所有币对组合进行滚动窗口协整检验
  - 汇总每个币对的协整检验结果
- **内部流程**：
  1. 遍历所有币对组合
  2. 对每个币对调用 `rolling_window_cointegration_test()`（第440-551行）
  3. 在每个窗口内调用 `enhanced_cointegration_test()`（第260-439行）
  4. 汇总所有窗口的检验结果

#### 4.1 滚动窗口协整检验详细流程

**函数**：`rolling_window_cointegration_test()`（第440-551行）

1. **数据对齐**：对齐两个价格序列的时间戳，确保数据长度一致
2. **窗口划分**：根据 `window_size` 和 `step_size` 划分窗口
3. **逐个窗口检验**：对每个窗口提取数据，调用 `enhanced_cointegration_test()` 进行协整检验
4. **结果汇总**：计算协整比例（通过检验的窗口数 / 总窗口数）

#### 4.2 增强协整检验详细流程

**函数**：`enhanced_cointegration_test()`（第260-439行）

1. **计算对冲比率**：调用 `calculate_hedge_ratio()`（第130-168行），使用OLS回归
2. **计算价差**：原始价差或差分价差
3. **ADF检验**：调用 `advanced_adf_test()`（第169-220行），检验价差的平稳性
4. **确定积分阶数**：调用 `determine_integration_order()`（第221-259行）
5. **协整判断**：根据ADF检验结果判断是否协整

### 步骤5：显示并选择协整对

```2275:2286:cointegration_test_windows_Copula_DCC-GARCH.py
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

**调用函数**：`display_rolling_window_candidates()`（第692-839行）

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

```2288:2382:cointegration_test_windows_Copula_DCC-GARCH.py
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

    print("\n" + "=" * 80)
    print("滚动窗口协整分析+交易测试完成")
    print(f"总共进行了 {test_count} 次测试")
    print("=" * 80)
```

#### 6.1 选择Z-score计算策略

```2298:2302:cointegration_test_windows_Copula_DCC-GARCH.py
        # 选择Z-score计算策略
        z_score_strategy = select_z_score_strategy()
        if z_score_strategy is None:
            print("测试结束，退出程序")
            break
```

**调用函数**：`select_z_score_strategy()`（第840-1103行）

```840:1103:cointegration_test_windows_Copula_DCC-GARCH.py
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

    print("  0. 退出程序")

    # 确定最大选择数
    max_choice = 5

    while True:
        try:
            choice = input(f"请选择 (0-{max_choice}): ").strip()

            if choice == '0':
                return None

            if choice == '1':
                strategy = TraditionalZScoreStrategy()
                print(f"已选择: {strategy.get_strategy_description()}")
                return strategy

            if choice == '2' and arima_garch_available:
                # 询问ARIMA和GARCH参数
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

            if choice == '3' and ecm_available:
                # 询问ECM参数
                print("\n配置ECM误差修正模型参数:")
                print("  直接回车使用默认值: 滞后阶数=1, 最小数据长度=30")

                ecm_lag_input = input("误差修正项滞后阶数 (默认1): ").strip()
                if ecm_lag_input:
                    try:
                        ecm_lag = int(ecm_lag_input)
                        if ecm_lag < 1:
                            print("滞后阶数必须>=1，使用默认值")
                            ecm_lag = 1
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        ecm_lag = 1
                else:
                    ecm_lag = 1

                min_data_input = input("最小数据长度 (默认30): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 10:
                            print("最小数据长度必须>=10，使用默认值")
                            min_data_length = 30
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 30
                else:
                    min_data_length = 30

                try:
                    strategy = EcmZScoreStrategy(ecm_lag=ecm_lag, min_data_length=min_data_length)
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"ECM策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '4' and kalman_available:
                # 询问Kalman Filter参数
                print("\n配置Kalman Filter动态价差模型参数:")
                print("  直接回车使用默认值: 过程方差=0.01, 观测方差=0.1, 最小数据长度=30")

                process_var_input = input("过程噪声方差 (默认0.01): ").strip()
                if process_var_input:
                    try:
                        process_variance = float(process_var_input)
                        if process_variance <= 0:
                            print("过程方差必须>0，使用默认值")
                            process_variance = 0.01
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        process_variance = 0.01
                else:
                    process_variance = 0.01

                obs_var_input = input("观测噪声方差 (默认0.1): ").strip()
                if obs_var_input:
                    try:
                        observation_variance = float(obs_var_input)
                        if observation_variance <= 0:
                            print("观测方差必须>0，使用默认值")
                            observation_variance = 0.1
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        observation_variance = 0.1
                else:
                    observation_variance = 0.1

                min_data_input = input("最小数据长度 (默认30): ").strip()
                if min_data_input:
                    try:
                        min_data_length = int(min_data_input)
                        if min_data_length < 10:
                            print("最小数据长度必须>=10，使用默认值")
                            min_data_length = 30
                    except ValueError:
                        print("输入格式错误，使用默认值")
                        min_data_length = 30
                else:
                    min_data_length = 30

                try:
                    strategy = KalmanFilterZScoreStrategy(
                        process_variance=process_variance,
                        observation_variance=observation_variance,
                        min_data_length=min_data_length
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Kalman Filter策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            if choice == '5' and copula_dcc_available:
                # 询问Copula + DCC-GARCH参数
                print("\n配置Copula + DCC-GARCH相关性/波动率模型参数:")
                print("  直接回车使用默认值: GARCH(1,1), Copula类型=高斯, 最小数据长度=50")

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

                copula_input = input("Copula类型 (gaussian/student，默认gaussian): ").strip().lower()
                if copula_input in ['gaussian', 'student']:
                    copula_type = copula_input
                else:
                    print("输入格式错误，使用默认值")
                    copula_type = 'gaussian'

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

                try:
                    strategy = CopulaDccGarchZScoreStrategy(
                        garch_order=garch_order,
                        copula_type=copula_type,
                        min_data_length=min_data_length
                    )
                    print(f"已选择: {strategy.get_strategy_description()}")
                    return strategy
                except Exception as e:
                    print(f"Copula + DCC-GARCH策略初始化失败: {str(e)}")
                    print("请重新选择")
                    continue

            print(f"无效选择，请输入 0-{max_choice} 之间的数字")

        except KeyboardInterrupt:
            print("\n用户取消选择")
            return None
        except Exception as e:
            print(f"选择失败: {str(e)}，请重新选择")
```

- **功能**：让用户选择Z-score计算策略
- **选项**：
  1. **传统方法**：使用均值和标准差计算Z-score
  2. **ARIMA-GARCH模型**：使用ARIMA预测均值，GARCH预测波动率
  3. **ECM误差修正模型**：使用误差修正模型预测均值回归
  4. **Kalman Filter动态价差模型**：使用Kalman Filter动态估计价差的均值和方差
  5. **Copula + DCC-GARCH相关性/波动率模型**（新增）：使用DCC-GARCH估计动态波动率和相关性，使用Copula建模依赖结构

**Copula + DCC-GARCH策略配置**（第1043-1094行）：
- GARCH阶数（默认(1,1)）
- Copula类型（'gaussian'或'student'，默认'gaussian'）
- 最小数据长度（默认50）

#### 6.2 配置交易参数

```2317:2319:cointegration_test_windows_Copula_DCC-GARCH.py
        # 8. 配置交易参数
        print(f"\n第 {test_count} 次测试 - 配置交易参数")
        trading_params = configure_trading_parameters()
```

**调用函数**：`configure_trading_parameters()`（第1105-1173行）

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

```2321:2335:cointegration_test_windows_Copula_DCC-GARCH.py
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

**类**：`AdvancedCointegrationTrading`（第1236-2230行）

- **功能**：初始化交易策略对象
- **关键属性**：
  - 交易参数（阈值、止盈止损等）
  - Z-score策略对象（`z_score_strategy`）
  - 策略类型标志（`use_arima_garch`、`use_ecm`、`use_kalman_filter`、`use_copula_dcc_garch`）

#### 6.4 执行回测

```2337:2341:cointegration_test_windows_Copula_DCC-GARCH.py
        results = trading_strategy.backtest_cointegration_trading(
            data,
            selected_pairs,
            initial_capital=10000
        )
```

**调用方法**：`backtest_cointegration_trading()`（第1894-2230行）

### 步骤7：回测执行详细流程

**方法**：`backtest_cointegration_trading()`（第1894-2230行）

#### 7.1 初始化

```1911:1921:cointegration_test_windows_Copula_DCC-GARCH.py
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

```1923:1927:cointegration_test_windows_Copula_DCC-GARCH.py
        # 获取所有时间点
        all_timestamps = set()
        for symbol in data.keys():
            all_timestamps.update(data[symbol].index)
        all_timestamps = sorted(list(all_timestamps))
```

- **功能**：收集所有币对的时间戳，合并并排序
- **用途**：按时间顺序遍历所有数据点

#### 7.3 回测主循环

```1955:2230:cointegration_test_windows_Copula_DCC-GARCH.py
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
                else:
                    # 一阶差分价差
                    # 需要先计算原始价差，再计算差分
                    current_spread_raw = self.calculate_current_spread(
                        current_prices[symbol1],
                        current_prices[symbol2],
                        pair_info['hedge_ratio']
                    )

                    # 获取历史原始价差数据
                    historical_spreads_raw = []
                    for j in range(max(0, i - self.lookback_period - 1), i):
                        if j < len(all_timestamps):
                            hist_timestamp = all_timestamps[j]
                            if (hist_timestamp in data[symbol1].index and
                                    hist_timestamp in data[symbol2].index):
                                hist_spread = self.calculate_current_spread(
                                    data[symbol1].loc[hist_timestamp],
                                    data[symbol2].loc[hist_timestamp],
                                    pair_info['hedge_ratio']
                                )
                                historical_spreads_raw.append(hist_spread)

                    # 计算一阶差分
                    if len(historical_spreads_raw) >= 2:
                        historical_spreads = np.diff(historical_spreads_raw)
                        current_spread = current_spread_raw - historical_spreads_raw[-1]
                    else:
                        continue

                # 检查现有持仓
                pair_name = pair_info['pair_name']
                if pair_name in self.positions:
                    # 已有持仓，检查退出条件
                    position = self.positions[pair_name]
                    # ... 检查退出条件并平仓 ...

                # 检查开仓条件
                if pair_name not in self.positions:
                    # 计算Z-score
                    if self.z_score_strategy is not None:
                        z_score = self.z_score_strategy.calculate_z_score(
                            current_spread, historical_spreads
                        )
                    else:
                        # 使用传统方法
                        z_score = self.calculate_z_score_traditional(
                            current_spread, historical_spreads
                        )

                    # 生成交易信号
                    if z_score >= self.z_threshold:
                        # 做空价差（SHORT_LONG）
                        signal = {
                            'action': 'SHORT_LONG',
                            'z_score': z_score,
                            'spread': current_spread
                        }
                        # 执行开仓
                        # ...
                    elif z_score <= -self.z_threshold:
                        # 做多价差（LONG_SHORT）
                        signal = {
                            'action': 'LONG_SHORT',
                            'z_score': z_score,
                            'spread': current_spread
                        }
                        # 执行开仓
                        # ...

            # 更新资金曲线
            # ...
```

对每个时间点执行以下步骤：

##### 7.3.1 获取当前价格

- 从数据中提取当前时间点的所有币对价格

##### 7.3.2 检查现有持仓

对每个持仓执行：

1. **获取持仓信息**：持仓币对、开仓价格、持仓数量等
2. **计算当前价差**：使用对冲比率计算当前价差
3. **计算Z-score**：
   - 获取历史价差序列
   - 调用Z-score策略计算当前Z-score
   - **Copula + DCC-GARCH策略**：使用DCC-GARCH估计动态波动率，使用Copula估计依赖结构，计算Z-score
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
   - 调用Z-score策略计算Z-score
   - **Copula + DCC-GARCH策略**：
     - 将价差序列转换为收益率序列
     - 使用GARCH模型估计动态波动率
     - 使用DCC模型估计动态相关性
     - 使用Copula估计依赖结构
     - 基于动态波动率和相关性计算价差的动态方差
     - 计算Z-score
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

```3062:3067:cointegration_test_windows_Copula_DCC-GARCH.py
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

```3069:3101:cointegration_test_windows_Copula_DCC-GARCH.py
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

```3116:3142:cointegration_test_windows_Copula_DCC-GARCH.py
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

#### 8.1 网格搜索

- **流程**：
  1. 粗粒度搜索：遍历粗粒度参数组合
  2. 细粒度搜索：在最佳参数附近进行细粒度搜索
  3. 稳定性测试：对最佳参数进行扰动测试

#### 8.2 随机搜索

- **流程**：
  1. 随机生成参数组合
  2. 评估每个组合
  3. 选择最佳组合

#### 8.3 贝叶斯优化

- **流程**：
  1. 定义参数搜索空间
  2. 使用高斯过程建模目标函数
  3. 迭代优化，选择下一个评估点
  4. 收敛后返回最佳参数

### 步骤9：显示和导出优化结果

```3144:3188:cointegration_test_windows_Copula_DCC-GARCH.py
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

## Copula + DCC-GARCH策略详解

### 策略类：CopulaDccGarchZScoreStrategy

**文件位置**：`strategies/copula_dcc_garch_zscore_strategy.py`

### 核心方法：calculate_z_score()

#### 1. 输入验证

- 检查历史数据长度是否足够（默认最小50）
- 不足时返回0.0（中性信号）

#### 2. 计算价差收益率

- 将价差序列转换为收益率序列
- 使用简单收益率：`r_t = (spread_t - spread_{t-1}) / spread_{t-1}`

#### 3. 拟合GARCH模型

- 使用GARCH模型估计价差收益率的动态波动率
- 预测下一期的波动率

#### 4. 估计DCC相关性

- 使用滚动窗口估计动态相关性（简化版DCC）
- 计算两个资产之间的动态相关系数

#### 5. 估计Copula参数

- 对于高斯Copula，参数是相关系数
- 使用经验分布函数转换为标准正态分布
- 估计Copula参数

#### 6. 计算动态标准差

- 使用GARCH预测的波动率
- 使用Copula调整波动率（根据相关性调整）
- 将收益率波动率转换为价差波动率

#### 7. 计算Z-score

```python
z_score = (current_spread - historical_mean) / spread_std
```

- 使用历史均值
- 使用动态估计的标准差

#### 8. 容错机制

- 验证Z-score有效性（NaN/Inf检查）
- 如果无效，回退到传统方法
- 如果传统方法也失败，返回0.0

### Copula + DCC-GARCH数学模型

**GARCH模型**：
```
σ²_t = ω + α₁*ε²_{t-1} + β₁*σ²_{t-1}
```

**DCC模型**（简化版）：
```
ρ_t = corr(returns1[t-w:t], returns2[t-w:t])
```

**高斯Copula**：
```
C(u₁, u₂; ρ) = Φ_ρ(Φ⁻¹(u₁), Φ⁻¹(u₂))
```

其中：
- `σ²_t`：条件方差（波动率的平方）
- `ρ_t`：动态相关系数
- `u₁, u₂`：边际分布的累积分布函数值
- `Φ_ρ`：二元标准正态分布的累积分布函数

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

2. **Copula + DCC-GARCH参数**：
   - `garch_order`：GARCH模型阶数，默认(1,1)
   - `copula_type`：Copula类型，'gaussian'或'student'，默认'gaussian'
   - `min_data_length`：最小数据长度，默认50（需要更多数据用于GARCH和Copula估计）

3. **计算性能**：
   - 滚动窗口检验可能耗时较长
   - 参数优化需要大量回测，建议先用小数据集测试
   - Copula + DCC-GARCH策略需要更多计算资源

4. **策略选择**：
   - 传统方法：简单快速，适合稳定市场
   - ARIMA-GARCH：适合有趋势和波动聚集的市场
   - ECM：适合协整关系明显的市场
   - Kalman Filter：适合动态变化的市场，能自适应调整
   - **Copula + DCC-GARCH**：适合需要建模相关性和波动率动态变化的市场

---

## 总结

本程序提供了一个完整的协整交易回测框架，支持：

1. **滚动窗口协整检验**：识别时变的协整关系
2. **多种Z-score策略**：包括新增的Copula + DCC-GARCH相关性/波动率模型
3. **完整交易回测**：包含开仓、平仓、止盈止损、手续费等
4. **参数优化**：支持多种优化方法，寻找最佳参数组合

Copula + DCC-GARCH策略通过动态估计波动率和相关性，使用Copula建模依赖结构，能够更好地捕捉市场动态，提高交易信号的准确性。

