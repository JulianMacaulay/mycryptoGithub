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
   - 获取历史价差序列和历史价格序列
   - 调用Z-score策略计算当前Z-score，传递价格数据
   - **Copula + DCC-GARCH策略（完整版）**：
     - 接收两个资产的历史价格序列（`historical_prices1` 和 `historical_prices2`）
     - 计算两个资产的收益率序列
     - 对每个资产分别拟合GARCH模型，估计动态波动率
     - 使用DCC模型估计两个资产之间的动态相关性
     - 使用Copula建模两个资产之间的依赖结构
     - 基于动态波动率和相关性计算价差的动态方差
     - 计算Z-score
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
   - **Copula + DCC-GARCH策略（完整版）**：
     - 接收两个资产的历史价格序列（`historical_prices1` 和 `historical_prices2`）
     - 计算两个资产的收益率序列（对数收益率）
     - 对每个资产分别拟合GARCH模型，估计各自的动态波动率
     - 使用DCC模型估计两个资产之间的动态相关性
     - 使用Copula建模两个资产之间的依赖结构
     - 基于两个资产的动态波动率和DCC相关性计算价差的动态方差
     - 使用Copula参数调整价差方差（考虑尾部依赖）
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

## Copula + DCC-GARCH策略详解（完整版）

### 策略类：CopulaDccGarchZScoreStrategy

**文件位置**：`strategies/copula_dcc_garch_zscore_strategy.py`

**重要说明**：本策略是完整版实现，需要两个资产的价格序列，而不是只有价差序列。

---

## 一、Copula + DCC-GARCH模型理论基础

### 1.1 模型概述

Copula + DCC-GARCH模型是一个多层次的金融时间序列模型，用于：
- **GARCH模型**：对每个资产分别建模，捕捉单个资产收益率的**波动率聚集效应**（volatility clustering）
- **DCC模型**：捕捉两个资产之间的**动态条件相关性**（Dynamic Conditional Correlation）
- **Copula函数**：建模两个资产之间的**依赖结构**（dependence structure），特别是尾部依赖

### 1.2 为什么需要Copula + DCC-GARCH？

在协整交易中，我们需要：
1. **动态波动率**：每个资产的波动率不是常数，而是时变的
2. **动态相关性**：两个资产之间的相关性会随时间变化
3. **非线性依赖**：资产之间的依赖关系可能不是线性的，特别是在极端情况下（尾部依赖）
4. **价差方差**：基于两个资产的动态波动率和动态相关性，准确计算价差的动态方差

传统方法假设：
- 波动率是常数
- 相关性是常数
- 依赖关系是线性的
- 价差方差 = 常数

Copula + DCC-GARCH模型能够：
- 动态估计每个资产的波动率
- 动态估计两个资产之间的相关性
- 捕捉非线性依赖结构
- 基于动态波动率和相关性计算价差的动态方差

---

## 二、GARCH模型详解

### 2.1 GARCH模型原理

**GARCH** = **G**eneralized **A**uto**R**egressive **C**onditional **H**eteroskedasticity（广义自回归条件异方差模型）

#### 2.1.1 基本思想

金融时间序列的波动率具有**聚集效应**（volatility clustering）：
- 高波动率时期往往连续出现
- 低波动率时期也往往连续出现
- 波动率本身是时变的，不是常数

#### 2.1.2 GARCH(p, q)模型数学表达式

**均值方程**：
```
r_t = μ + ε_t
```

其中：
- `r_t`：时刻t的收益率
- `μ`：均值（通常假设为0或常数）
- `ε_t`：误差项

**误差项分解**：
```
ε_t = σ_t * z_t
```

其中：
- `σ_t`：时刻t的条件标准差（波动率）
- `z_t`：标准化残差，通常假设为标准正态分布：`z_t ~ N(0, 1)`

**条件方差方程（GARCH核心）**：
```
σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
```

展开形式（GARCH(p, q)）：
```
σ²_t = ω + α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + αₚ*ε²_{t-p}
              + β₁*σ²_{t-1} + β₂*σ²_{t-2} + ... + β_q*σ²_{t-q}
```

其中：
- `σ²_t`：时刻t的条件方差（波动率的平方）
- `ω > 0`：常数项（长期平均波动率）
- `α_i ≥ 0`：ARCH项系数（残差平方的滞后项系数）
- `β_j ≥ 0`：GARCH项系数（条件方差的滞后项系数）
- `ε²_{t-i}`：滞后i期的残差平方（捕捉"冲击"的影响）
- `σ²_{t-j}`：滞后j期的条件方差（捕捉波动率的持续性）

#### 2.1.3 GARCH(1,1)模型（最常用）

**数学表达式**：
```
σ²_t = ω + α₁*ε²_{t-1} + β₁*σ²_{t-1}
```

**参数含义**：
- `ω`：长期平均波动率
- `α₁`：短期冲击的影响（残差平方的系数）
- `β₁`：波动率的持续性（条件方差的系数）

**约束条件**：
```
α₁ + β₁ < 1  （确保模型平稳）
```

**解释**：
- 当前波动率依赖于：
  - 上一期的残差平方（`ε²_{t-1}`）：捕捉"冲击"的影响
  - 上一期的波动率（`σ²_{t-1}`）：捕捉波动率的持续性

#### 2.1.4 在价差Z-score计算中的应用

**代码实现**（第76-92行）：

```76:92:strategies/copula_dcc_garch_zscore_strategy.py
    def _fit_garch_model(self, returns: np.ndarray) -> Optional[object]:
        """
        拟合GARCH模型
        
        Args:
            returns: 收益率序列
            
        Returns:
            拟合的GARCH模型对象，如果失败返回None
        """
        try:
            garch_model = arch_model(returns, vol='Garch', 
                                    p=self.garch_order[0], q=self.garch_order[1])
            garch_fitted = garch_model.fit(disp='off')
            return garch_fitted
        except Exception:
            return None
```

**步骤**（完整版）：
1. **获取两个资产的价格序列**：从主程序传递 `historical_prices1` 和 `historical_prices2`
2. **计算两个资产的收益率序列**：使用对数收益率 `r_t = log(P_t / P_{t-1})`
3. **对每个资产分别拟合GARCH模型**：
   - 资产1：`garch_fitted1 = fit_garch(returns1)`
   - 资产2：`garch_fitted2 = fit_garch(returns2)`
4. **预测下一期的条件方差**：
   - `σ²₁_{t+1}` = GARCH预测（资产1）
   - `σ²₂_{t+1}` = GARCH预测（资产2）
5. **计算预测的波动率**：
   - `σ₁_{t+1} = √(σ²₁_{t+1})`
   - `σ₂_{t+1} = √(σ²₂_{t+1})`

**预测波动率**（第390-405行）：

```390:405:strategies/copula_dcc_garch_zscore_strategy.py
            # 步骤4: 预测动态波动率
            try:
                garch_forecast1 = garch_fitted1.forecast(horizon=1)
                garch_forecast2 = garch_fitted2.forecast(horizon=1)
                predicted_vol1 = np.sqrt(garch_forecast1.variance.values[-1, 0])
                predicted_vol2 = np.sqrt(garch_forecast2.variance.values[-1, 0])
            except Exception:
                # 预测失败，使用历史波动率
                predicted_vol1 = np.std(returns1)
                predicted_vol2 = np.std(returns2)
            
            # 验证波动率
            if predicted_vol1 <= 0 or np.isnan(predicted_vol1):
                predicted_vol1 = np.std(returns1)
            if predicted_vol2 <= 0 or np.isnan(predicted_vol2):
                predicted_vol2 = np.std(returns2)
```

---

## 三、DCC模型详解

### 3.1 DCC模型原理

**DCC** = **D**ynamic **C**onditional **C**orrelation（动态条件相关性）

#### 3.1.1 基本思想

在协整交易中，两个资产之间的相关性不是常数，而是**时变的**：
- 市场平静时，相关性可能较低
- 市场波动时，相关性可能急剧上升（相关性崩溃，correlation breakdown）
- 相关性本身具有聚集效应

#### 3.1.2 完整DCC-GARCH模型

对于两个资产，完整的DCC-GARCH模型包括：

**资产1的GARCH模型**：
```
r₁_t = μ₁ + ε₁_t
ε₁_t = σ₁_t * z₁_t
σ²₁_t = ω₁ + α₁₁*ε²₁_{t-1} + β₁₁*σ²₁_{t-1}
```

**资产2的GARCH模型**：
```
r₂_t = μ₂ + ε₂_t
ε₂_t = σ₂_t * z₂_t
σ²₂_t = ω₂ + α₂₂*ε²₂_{t-1} + β₂₂*σ²₂_{t-1}
```

**标准化残差**：
```
u₁_t = ε₁_t / σ₁_t = z₁_t
u₂_t = ε₂_t / σ₂_t = z₂_t
```

**DCC模型（动态相关性）**：
```
Q_t = (1 - a - b) * Q̄ + a * u_{t-1} * u'_{t-1} + b * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
```

其中：
- `Q_t`：准相关矩阵（quasi-correlation matrix）
- `R_t`：动态相关矩阵（dynamic correlation matrix）
- `Q̄`：标准化残差的样本相关矩阵
- `a, b`：DCC参数，满足 `a + b < 1`
- `u_t = [u₁_t, u₂_t]'`：标准化残差向量

**动态相关系数**：
```
ρ_t = R_t[1,2]  （资产1和资产2之间的动态相关系数）
```

#### 3.1.3 完整DCC模型实现

本策略实现了**完整DCC模型**，包括DCC参数估计和动态相关性计算。

**步骤1：估计DCC参数**（第94-180行）

**代码实现**：`_estimate_dcc_parameters()`

```94:180:strategies/copula_dcc_garch_zscore_strategy.py
    def _estimate_dcc_parameters(self, returns1: np.ndarray, returns2: np.ndarray,
                                 garch_fitted1: object, garch_fitted2: object) -> Tuple[float, float]:
        """
        估计DCC（动态条件相关性）模型参数
        
        DCC模型：
        Q_t = (1 - α - β) * Q_bar + α * (ε_{t-1} * ε_{t-1}') + β * Q_{t-1}
        R_t = (diag(Q_t))^{-1/2} * Q_t * (diag(Q_t))^{-1/2}
        
        其中：
        - Q_t: 条件协方差矩阵
        - R_t: 条件相关系数矩阵
        - ε_t: 标准化残差
        - α, β: DCC参数（α + β < 1）
        
        简化实现：使用MLE估计α和β
        """
```

**DCC参数估计方法**：
1. **获取标准化残差**：
   ```
   u₁_t = ε₁_t / σ₁_t  （资产1的标准化残差）
   u₂_t = ε₂_t / σ₂_t  （资产2的标准化残差）
   ```
   其中 `σ₁_t` 和 `σ₂_t` 来自GARCH模型的条件波动率

2. **计算无条件协方差矩阵**：
   ```
   Q̄ = E[u_t * u_t']
   ```
   其中 `u_t = [u₁_t, u₂_t]'`

3. **使用网格搜索估计DCC参数**：
   - 在 `α ∈ [0.01, 0.15]` 和 `β ∈ [0.80, 0.95]` 范围内搜索
   - 约束条件：`α + β < 1`
   - 目标：最大化对数似然函数

**步骤2：计算动态相关系数**（第182-249行）

**代码实现**：`_estimate_dcc_correlation()`

```182:249:strategies/copula_dcc_garch_zscore_strategy.py
    def _estimate_dcc_correlation(self, returns1: np.ndarray, returns2: np.ndarray,
                                  garch_fitted1: object, garch_fitted2: object,
                                  dcc_alpha: float, dcc_beta: float) -> float:
        """
        使用DCC模型估计当前时刻的动态相关系数
        
        Args:
            returns1: 第一个资产的收益率序列
            returns2: 第二个资产的收益率序列
            garch_fitted1: 第一个资产的GARCH模型
            garch_fitted2: 第二个资产的GARCH模型
            dcc_alpha: DCC参数α
            dcc_beta: DCC参数β
            
        Returns:
            float: 当前时刻的动态相关系数
        """
        # 获取标准化残差
        conditional_vol1 = garch_fitted1.conditional_volatility
        conditional_vol2 = garch_fitted2.conditional_volatility
        
        standardized_residuals1 = returns1 / (conditional_vol1 + 1e-8)
        standardized_residuals2 = returns2 / (conditional_vol2 + 1e-8)
        
        # 计算无条件协方差矩阵
        Q_bar = np.cov(standardized_residuals1, standardized_residuals2)
        q_bar_12 = Q_bar[0, 1]
        
        # 初始化Q矩阵
        Q_t = np.array([[1.0, q_bar_12], [q_bar_12, 1.0]])
        
        # 递归更新Q矩阵
        for t in range(1, len(returns1)):
            epsilon_t = np.array([standardized_residuals1[t-1], 
                                standardized_residuals2[t-1]])
            Q_t = (1 - dcc_alpha - dcc_beta) * np.array([[1.0, q_bar_12], 
                                                          [q_bar_12, 1.0]]) + \
                  dcc_alpha * np.outer(epsilon_t, epsilon_t) + \
                  dcc_beta * Q_t
        
        # 计算条件相关系数矩阵 R_t
        diag_inv = 1.0 / np.sqrt(np.diag(Q_t))
        R_t = np.diag(diag_inv) @ Q_t @ np.diag(diag_inv)
        
        # 返回相关系数
        correlation = R_t[0, 1]
```

**完整DCC公式**：
```
Q_t = (1 - α - β) * Q̄ + α * (u_{t-1} * u'_{t-1}) + β * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
ρ_t = R_t[1,2]
```

其中：
- `Q_t`：准相关矩阵（quasi-correlation matrix）
- `R_t`：条件相关系数矩阵
- `ρ_t`：动态相关系数
- `α, β`：DCC参数（通过MLE估计）

**优点**：
- 完整实现DCC模型
- 能够准确捕捉相关性的时变特性
- 考虑了波动率的影响（通过标准化残差）

**与简化版的区别**：
- 简化版：使用滚动窗口计算相关系数
- 完整版：使用DCC模型递归更新，考虑历史相关性的持续性

---

## 四、Copula函数详解

### 4.1 Copula函数原理

#### 4.1.1 基本思想

**Copula函数**是连接多个随机变量的边际分布和联合分布的函数。

**Sklar定理**（Copula理论的基础）：
对于n个随机变量 `X₁, X₂, ..., Xₙ`，如果它们的边际分布函数分别为 `F₁, F₂, ..., Fₙ`，联合分布函数为 `F`，则存在一个Copula函数 `C`，使得：

```
F(x₁, x₂, ..., xₙ) = C(F₁(x₁), F₂(x₂), ..., Fₙ(xₙ))
```

**关键洞察**：
- Copula函数**分离**了边际分布和依赖结构
- 边际分布描述单个变量的特征
- Copula函数描述变量之间的依赖关系

#### 4.1.2 高斯Copula（Gaussian Copula）

**数学定义**：

对于两个随机变量 `X₁, X₂`，高斯Copula定义为：

```
C(u₁, u₂; ρ) = Φ_ρ(Φ⁻¹(u₁), Φ⁻¹(u₂))
```

其中：
- `u₁ = F₁(x₁)`：第一个变量的累积分布函数值
- `u₂ = F₂(x₂)`：第二个变量的累积分布函数值
- `Φ⁻¹()`：标准正态分布的逆累积分布函数（分位数函数）
- `Φ_ρ()`：相关系数为 `ρ` 的二元标准正态分布的累积分布函数
- `ρ`：Copula参数（对于高斯Copula，就是相关系数）

**二元标准正态分布**：

如果 `(Z₁, Z₂) ~ N₂(0, Σ)`，其中：
```
Σ = [1   ρ]
    [ρ   1]
```

则：
```
Φ_ρ(z₁, z₂) = P(Z₁ ≤ z₁, Z₂ ≤ z₂)
```

**密度函数**：

高斯Copula的密度函数为：
```
c(u₁, u₂; ρ) = (1/√(1-ρ²)) * exp(-(z₁² + z₂² - 2ρz₁z₂)/(2(1-ρ²)) + (z₁² + z₂²)/2)
```

其中 `z₁ = Φ⁻¹(u₁)`, `z₂ = Φ⁻¹(u₂)`

#### 4.1.3 在价差Z-score计算中的应用

**代码实现**（第251-308行）：

```251:308:strategies/copula_dcc_garch_zscore_strategy.py
    def _estimate_copula_parameter(self, returns1: np.ndarray, returns2: np.ndarray) -> Tuple[float, Optional[float]]:
        """
        估计Copula参数
        
        对于高斯Copula，参数是相关系数
        对于t-Copula，需要估计相关系数和自由度
        
        Args:
            returns1: 第一个资产的收益率序列
            returns2: 第二个资产的收益率序列
            
        Returns:
            Tuple[float, Optional[float]]: (相关系数, 自由度) 对于t-Copula，自由度不为None
        """
        if len(returns1) != len(returns2) or len(returns1) < 10:
            return (0.0, None)
        
        try:
            # 转换为标准正态分布（使用经验分布函数）
            # 使用排序位置估计边际分布
            ranks1 = np.argsort(np.argsort(returns1)) / (len(returns1) - 1)
            ranks2 = np.argsort(np.argsort(returns2)) / (len(returns2) - 1)
            
            # 转换为标准正态分布
            u1 = norm.ppf(np.clip(ranks1, 0.001, 0.999))
            u2 = norm.ppf(np.clip(ranks2, 0.001, 0.999))
            
            # 估计相关系数（高斯Copula参数）
            correlation = np.corrcoef(u1, u2)[0, 1]
            
            if np.isnan(correlation) or np.isinf(correlation):
                correlation = 0.0
            correlation = np.clip(correlation, -0.99, 0.99)
            
            # 如果是t-Copula，还需要估计自由度
            if self.copula_type == 'student':
                # 使用最大似然估计自由度（简化版）
                # 尝试不同的自由度值，选择使对数似然最大的
                best_df = 5.0
                best_loglik = -np.inf
                
                for df in [3, 4, 5, 6, 7, 8, 9, 10]:
                    try:
                        # 计算t-Copula的对数似然（简化版）
                        # 实际应该使用完整的t-Copula密度函数
                        loglik = -0.5 * (df + 2) * np.sum(np.log(1 + (u1**2 + u2**2 - 2*correlation*u1*u2) / (df * (1 - correlation**2))))
                        if loglik > best_loglik:
                            best_loglik = loglik
                            best_df = float(df)
                    except:
                        continue
                
                return (correlation, best_df)
            else:
                return (correlation, None)
```

**步骤详解**：

1. **估计边际分布**（使用经验分布函数）：
   ```
   u₁ = rank(returns1) / (n - 1)
   u₂ = rank(returns2) / (n - 1)
   ```
   其中 `rank()` 是排序位置

2. **转换为标准正态分布**：
   ```
   z₁ = Φ⁻¹(u₁)
   z₂ = Φ⁻¹(u₂)
   ```

3. **估计Copula参数**：
   - **高斯Copula**：`ρ = corr(z₁, z₂)`
   - **t-Copula**：`(ρ, ν)`，其中 `ν` 是自由度（通过最大似然估计）

**为什么使用经验分布函数？**

- 不需要假设边际分布的具体形式
- 对异常值更稳健
- 能够捕捉非正态的边际分布
- 能够捕捉尾部依赖（特别是t-Copula）

---

## 五、Copula + DCC-GARCH完整流程（完整版）

### 5.1 计算Z-score的完整步骤

**核心方法**：`calculate_z_score()`（第310-463行）

**重要**：完整版需要两个资产的价格序列（`historical_prices1` 和 `historical_prices2`），而不是只有价差序列。

#### 步骤1：输入验证和数据准备

```332:355:strategies/copula_dcc_garch_zscore_strategy.py
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=self.min_data_length):
            return 0.0
        
        # 检查是否有价格数据
        if historical_prices1 is None or historical_prices2 is None:
            # 如果没有价格数据，回退到简化版本（基于价差收益率）
            return self._calculate_z_score_simplified(current_spread, historical_spreads)
        
        if len(historical_prices1) != len(historical_prices2) or \
           len(historical_prices1) != len(historical_spreads):
            # 数据长度不匹配
            return self._calculate_z_score_simplified(current_spread, historical_spreads)
        
        try:
            # 转换为numpy数组
            prices1_array = np.array(historical_prices1)
            prices2_array = np.array(historical_prices2)
            spreads_array = np.array(historical_spreads)
            
            # 计算两个资产的收益率序列
            returns1 = self._calculate_returns(prices1_array)
            returns2 = self._calculate_returns(prices2_array)
```

**收益率计算公式**（对数收益率）：
```
r₁_t = log(P₁_t / P₁_{t-1})  （资产1的收益率）
r₂_t = log(P₂_t / P₂_{t-1})  （资产2的收益率）
```

#### 步骤2：对每个资产拟合GARCH模型

```367:388:strategies/copula_dcc_garch_zscore_strategy.py
                # 步骤1: 对每个资产拟合GARCH模型
                garch_fitted1 = self._fit_garch_model(returns1)
                garch_fitted2 = self._fit_garch_model(returns2)
                
                if garch_fitted1 is None or garch_fitted2 is None:
                    # GARCH拟合失败，使用传统方法
                    return self._calculate_z_score_simplified(current_spread, historical_spreads)
                
                # 步骤2: 估计DCC参数
                dcc_alpha, dcc_beta = self._estimate_dcc_parameters(returns1, returns2, 
                                                                    garch_fitted1, garch_fitted2)
                
                # 步骤3: 估计Copula参数
                copula_param, copula_df = self._estimate_copula_parameter(returns1, returns2)
```

**GARCH模型拟合**：
- 资产1：`garch_fitted1 = fit_garch(returns1)`，估计 `σ²₁_t`
- 资产2：`garch_fitted2 = fit_garch(returns2)`，估计 `σ²₂_t`
- 使用最大似然估计（MLE）估计参数

#### 步骤3：估计DCC参数

**调用**：`_estimate_dcc_parameters(returns1, returns2, garch_fitted1, garch_fitted2)`

**方法**：
1. 获取标准化残差：`u₁_t = r₁_t / σ₁_t`，`u₂_t = r₂_t / σ₂_t`
2. 计算无条件协方差矩阵：`Q̄ = E[u_t * u_t']`
3. 使用网格搜索估计DCC参数 `(α, β)`，最大化对数似然函数

#### 步骤4：估计Copula参数

**调用**：`_estimate_copula_parameter(returns1, returns2)`

**方法**：
1. 使用经验分布函数估计边际分布
2. 转换为标准正态分布
3. 估计Copula参数 `ρ`（高斯Copula）或 `(ρ, ν)`（t-Copula）

#### 步骤5：预测动态波动率

```390:405:strategies/copula_dcc_garch_zscore_strategy.py
            # 步骤4: 预测动态波动率
            try:
                garch_forecast1 = garch_fitted1.forecast(horizon=1)
                garch_forecast2 = garch_fitted2.forecast(horizon=1)
                predicted_vol1 = np.sqrt(garch_forecast1.variance.values[-1, 0])
                predicted_vol2 = np.sqrt(garch_forecast2.variance.values[-1, 0])
            except Exception:
                # 预测失败，使用历史波动率
                predicted_vol1 = np.std(returns1)
                predicted_vol2 = np.std(returns2)
            
            # 验证波动率
            if predicted_vol1 <= 0 or np.isnan(predicted_vol1):
                predicted_vol1 = np.std(returns1)
            if predicted_vol2 <= 0 or np.isnan(predicted_vol2):
                predicted_vol2 = np.std(returns2)
```

**预测公式**：
```
σ²₁_{t+1} = ω₁ + α₁₁*ε²₁_t + β₁₁*σ²₁_t
σ²₂_{t+1} = ω₂ + α₂₂*ε²₂_t + β₂₂*σ²₂_t
σ₁_{t+1} = √(σ²₁_{t+1})
σ₂_{t+1} = √(σ²₂_{t+1})
```

#### 步骤6：估计动态相关系数（DCC）

```410:413:strategies/copula_dcc_garch_zscore_strategy.py
            # 步骤5: 估计动态相关系数（使用DCC模型）
            dcc_correlation = self._estimate_dcc_correlation(returns1, returns2, 
                                                            garch_fitted1, garch_fitted2,
                                                            dcc_alpha, dcc_beta)
```

**DCC公式**：
```
Q_t = (1 - α - β) * Q̄ + α * (u_{t-1} * u'_{t-1}) + β * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
ρ_t = R_t[1,2]
```

#### 步骤7：计算价差的动态方差

```415:446:strategies/copula_dcc_garch_zscore_strategy.py
            # 步骤6: 计算价差的动态方差
            # 假设价差 = price1 - hedge_ratio * price2
            # 价差的方差 = Var(price1) + hedge_ratio^2 * Var(price2) - 2 * hedge_ratio * Cov(price1, price2)
            # 对于收益率：Var(spread_return) = Var(return1) + hedge_ratio^2 * Var(return2) - 2 * hedge_ratio * Corr * Std(return1) * Std(return2)
            
            # 估计hedge_ratio（使用历史价差和价格）
            # 简化：使用OLS回归估计hedge_ratio
            try:
                # price1 = hedge_ratio * price2 + spread
                # 使用价格序列估计hedge_ratio
                X = prices2_array.reshape(-1, 1)
                y = prices1_array
                hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
            except:
                # 估计失败，使用1.0作为默认值
                hedge_ratio = 1.0
            
            # 计算价差的动态方差
            spread_variance = predicted_vol1**2 + (hedge_ratio**2) * predicted_vol2**2 - \
                            2 * hedge_ratio * dcc_correlation * predicted_vol1 * predicted_vol2
            
            # 使用Copula调整方差（考虑尾部依赖）
            if self.copula_type == 'student' and copula_df is not None:
                # t-Copula有尾部依赖，可能需要调整方差
                # 简化处理：使用Copula参数作为调整因子
                copula_adjustment = 1.0 + 0.1 * abs(copula_param)
            else:
                # 高斯Copula
                copula_adjustment = 1.0 + 0.05 * abs(copula_param)
            
            adjusted_spread_variance = spread_variance * copula_adjustment
            spread_std = np.sqrt(max(adjusted_spread_variance, 1e-8))
```

**价差方差公式**（完整版）：
```
Var(spread_return) = Var(return1) + hedge_ratio² * Var(return2) 
                     - 2 * hedge_ratio * ρ_t * σ₁_t * σ₂_t
```

其中：
- `Var(return1) = σ²₁_t`（资产1的动态方差）
- `Var(return2) = σ²₂_t`（资产2的动态方差）
- `ρ_t`：DCC动态相关系数
- `hedge_ratio`：对冲比率（通过OLS回归估计）

**Copula调整**：
```
σ²_spread_adjusted = σ²_spread * (1 + c * |ρ_copula|)
```

其中：
- `c = 0.1`（t-Copula）或 `c = 0.05`（高斯Copula）
- `ρ_copula`：Copula参数

#### 步骤8：计算Z-score

```448:458:strategies/copula_dcc_garch_zscore_strategy.py
            # 步骤7: 计算价差的动态均值
            historical_mean = np.mean(spreads_array)
            
            # 步骤8: 计算Z-score
            z_score = (current_spread - historical_mean) / spread_std
            
            # 验证结果
            if np.isnan(z_score) or np.isinf(z_score):
                return self._calculate_z_score_simplified(current_spread, historical_spreads)
            
            return z_score
```

**Z-score公式**：
```
Z = (spread_t - μ_spread) / σ_spread
```

其中：
- `spread_t`：当前价差
- `μ_spread`：价差的历史均值
- `σ_spread`：价差的动态标准差（基于两个资产的动态波动率、DCC相关性和Copula调整）

---

## 六、数学模型总结（完整版）

### 6.1 完整模型框架

**资产收益率模型**：
```
r₁_t = log(P₁_t / P₁_{t-1})  （资产1的对数收益率）
r₂_t = log(P₂_t / P₂_{t-1})  （资产2的对数收益率）
```

**GARCH模型（每个资产的波动率）**：
```
σ²₁_t = ω₁ + α₁₁*ε²₁_{t-1} + β₁₁*σ²₁_{t-1}  （资产1）
σ²₂_t = ω₂ + α₂₂*ε²₂_{t-1} + β₂₂*σ²₂_{t-1}  （资产2）
```

**标准化残差**：
```
u₁_t = r₁_t / σ₁_t
u₂_t = r₂_t / σ₂_t
```

**DCC模型（动态相关性）**：
```
Q_t = (1 - α - β) * Q̄ + α * (u_{t-1} * u'_{t-1}) + β * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
ρ_t = R_t[1,2]
```

**Copula模型（依赖结构）**：
```
u₁ = F₁(r₁_t)  （资产1的经验分布函数值）
u₂ = F₂(r₂_t)  （资产2的经验分布函数值）
z₁ = Φ⁻¹(u₁)
z₂ = Φ⁻¹(u₂)
ρ_copula = corr(z₁, z₂)
```

**价差方差（基于两个资产的动态波动率和DCC相关性）**：
```
Var(spread_return) = σ²₁_t + hedge_ratio² * σ²₂_t 
                     - 2 * hedge_ratio * ρ_t * σ₁_t * σ₂_t
```

**Copula调整**：
```
σ²_spread_adjusted = σ²_spread * (1 + c * |ρ_copula|)
```

其中：
- `c = 0.1`（t-Copula）或 `c = 0.05`（高斯Copula）

**价差标准差**：
```
σ_spread = √(σ²_spread_adjusted)
```

**Z-score**：
```
Z = (spread_t - μ_spread) / σ_spread
```

### 6.2 模型优势（完整版）

1. **动态波动率**：对每个资产分别建模GARCH，能够捕捉各自波动率的时变特性
2. **动态相关性**：使用完整DCC模型，能够准确捕捉两个资产之间相关性的时变特性
3. **非线性依赖**：Copula函数能够捕捉非线性依赖结构，特别是尾部依赖
4. **价差方差**：基于两个资产的动态波动率和动态相关性，准确计算价差的动态方差
5. **稳健性**：使用经验分布函数，对异常值更稳健

### 6.3 模型局限性

1. **计算复杂度**：需要更多数据（最小50个数据点）和计算资源
2. **参数调整**：Copula调整因子（0.05或0.1）是经验值，可能需要根据数据调整
3. **假设限制**：仍然假设标准化残差服从标准正态分布（GARCH模型）
4. **DCC参数估计**：使用网格搜索，可能不如完整MLE精确

---

## 七、容错机制

### 7.1 多层容错设计

**第一层：输入验证**
- 检查数据长度是否足够
- 不足时返回0.0

**第二层：GARCH拟合失败**
- 如果GARCH拟合失败，回退到传统方法

**第三层：波动率预测失败**
- 如果波动率预测失败，使用历史波动率

**第四层：Z-score验证**
- 验证Z-score有效性（NaN/Inf检查）
- 如果无效，回退到传统方法

**第五层：全局异常捕获**
- 捕获所有异常，尝试传统方法作为后备

### 7.2 容错代码实现

```285:296:strategies/copula_dcc_garch_zscore_strategy.py
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
            print(f"Copula + DCC-GARCH模型计算失败: {str(e)}")
            return 0.0
```

---

## 八、参数说明

### 8.1 策略参数

- **garch_order**：GARCH模型阶数 `(p, q)`，默认 `(1, 1)`
  - `p`：ARCH项阶数（残差平方的滞后项数）
  - `q`：GARCH项阶数（条件方差的滞后项数）

- **copula_type**：Copula类型，可选：
  - `'gaussian'`：高斯Copula（默认）
  - `'student'`：t-Copula（未来扩展）

- **min_data_length**：最小数据长度，默认50
  - GARCH模型需要至少20个数据点
  - Copula估计需要更多数据点

### 8.2 参数选择建议

1. **GARCH阶数**：
   - 对于大多数金融时间序列，GARCH(1,1)已经足够
   - 如果数据有复杂的波动率模式，可以尝试GARCH(2,1)或GARCH(1,2)

2. **Copula类型**：
   - 高斯Copula：适合大多数情况
   - t-Copula：适合有尾部依赖的情况（未来实现）

3. **最小数据长度**：
   - 至少50个数据点
   - 如果数据质量好，可以降低到40
   - 如果数据质量差，建议增加到60-80

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

2. **Copula + DCC-GARCH参数**（完整版）：
   - `garch_order`：GARCH模型阶数，默认(1,1)
   - `copula_type`：Copula类型，'gaussian'或'student'，默认'gaussian'
   - `min_data_length`：最小数据长度，默认50（需要更多数据用于GARCH、DCC和Copula估计）
   - **重要**：需要两个资产的价格序列（`historical_prices1` 和 `historical_prices2`），主程序会自动传递

3. **计算性能**：
   - 滚动窗口检验可能耗时较长
   - 参数优化需要大量回测，建议先用小数据集测试
   - Copula + DCC-GARCH策略需要更多计算资源

4. **策略选择**：
   - 传统方法：简单快速，适合稳定市场
   - ARIMA-GARCH：适合有趋势和波动聚集的市场
   - ECM：适合协整关系明显的市场
   - Kalman Filter：适合动态变化的市场，能自适应调整
   - **Copula + DCC-GARCH（完整版）**：适合需要建模两个资产之间相关性和波动率动态变化的市场，能够准确计算价差的动态方差

---

## 总结

本程序提供了一个完整的协整交易回测框架，支持：

1. **滚动窗口协整检验**：识别时变的协整关系
2. **多种Z-score策略**：包括新增的Copula + DCC-GARCH相关性/波动率模型
3. **完整交易回测**：包含开仓、平仓、止盈止损、手续费等
4. **参数优化**：支持多种优化方法，寻找最佳参数组合

Copula + DCC-GARCH策略（完整版）通过：
1. 对每个资产分别建模GARCH，动态估计各自的波动率
2. 使用完整DCC模型，动态估计两个资产之间的相关性
3. 使用Copula建模两个资产之间的依赖结构
4. 基于动态波动率和相关性，准确计算价差的动态方差

能够更好地捕捉市场动态，提高交易信号的准确性。

