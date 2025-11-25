# backtest_system_ML.py 详细说明文档

## 目录
1. [系统概述](#系统概述)
2. [程序运行流程](#程序运行流程)
3. [机器学习模型详解](#机器学习模型详解)
4. [机器学习信号在回测中的应用](#机器学习信号在回测中的应用)
5. [核心类和方法说明](#核心类和方法说明)

---

## 系统概述

`backtest_system_ML.py` 是一个完整的回测系统，支持基于机器学习的市场状态检测（趋势/震荡），并根据市场状态动态选择相应的交易策略。

### 主要功能
- 从币安API或CSV文件加载数据
- 使用机器学习模型检测市场状态（趋势/震荡）
- 根据市场状态选择不同的交易策略
- 执行完整的回测并生成报告

---

## 程序运行流程

### 1. 主函数入口

程序从 `main()` 函数开始执行：

```2666:2267:backtest_system_ML.py
if __name__ == "__main__":
    main()
```

### 2. 初始化阶段

#### 2.1 选择是否使用市场状态检测

```2114:2125:backtest_system_ML.py
    # 1. 选择是否使用市场状态检测
    print("\n是否使用机器学习市场状态检测？")
    print("  y: 使用（根据市场状态选择策略）")
    print("  n: 不使用（使用单一策略）")
    use_ml_input = input("请选择 (y/n, 默认y): ").strip().lower()
    use_ml = use_ml_input != 'n'
```

#### 2.2 选择交易策略

如果使用市场状态检测，需要选择趋势策略和震荡策略：

```2128:2146:backtest_system_ML.py
    # 2. 选择策略
    if use_ml:
        strategies = select_strategies()
        if not strategies:
            print("未选择策略，退出程序")
            return
        
        # 检查策略配置
        has_trending = 'trending' in strategies
        has_ranging = 'ranging' in strategies
        
        if not has_trending and not has_ranging:
            print("错误: 必须至少选择一个策略")
            return
        
        print(f"\n策略配置:")
        if has_trending:
            print(f"  趋势策略: {strategies['trending'].name}")
        if has_ranging:
            print(f"  震荡策略: {strategies['ranging'].name}")
```

#### 2.3 加载数据

```2179:2192:backtest_system_ML.py
    # 3. 加载数据
    print("\n" + "=" * 60)
    print("加载数据")
    print("=" * 60)
    
    csv_file = input("请输入CSV文件路径（直接回车使用默认）: ").strip()
    if not csv_file:
        csv_file = "segment_1_data_ccxt_20251106_195714.csv"
    
    symbol = None
    if os.path.exists(csv_file):
        symbol_input = input("请输入币种符号（如ETHUSDT，直接回车跳过）: ").strip()
        if symbol_input:
            symbol = symbol_input
```

数据加载方法：

```1075:1119:backtest_system_ML.py
    def load_data_from_csv(self, filepath: str, symbol: str = None):
        """
        从CSV文件加载数据

        Args:
            filepath: CSV文件路径
            symbol: 如果CSV包含多个币种，指定要使用的币种
        """
        df = pd.read_csv(filepath)

        # 如果包含symbol列，筛选特定币种
        if 'symbol' in df.columns and symbol:
            # 支持多种格式：ETHUSDT, ETH/USDT, ETH-USDT
            symbol_variants = [symbol, symbol.replace('USDT', '/USDT'), symbol.replace('/', 'USDT')]
            df_filtered = df[df['symbol'].isin(symbol_variants)].copy()
            if len(df_filtered) > 0:
                df = df_filtered
                print(f"筛选币种: {symbol}，找到 {len(df)} 条记录")
            else:
                print(f"警告: 未找到币种 {symbol}，使用所有数据")

        # 确保时间列为datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            # 尝试将第一列作为时间索引
            df.index = pd.to_datetime(df.index)

        # 确保有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")

        # 按时间排序
        df = df.sort_index()

        self.data = df
        print(f"成功加载数据: {len(df)} 条记录")
        print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

        return df
```

#### 2.4 创建市场状态检测器

如果使用机器学习，需要配置检测器：

```2194:2220:backtest_system_ML.py
    # 4. 创建回测系统
    if use_ml:
        # 选择标签生成方法
        label_method = select_label_method()
        
        # 选择机器学习模型
        model_type = select_ml_model()
        if not model_type:
            print("无法继续，退出程序")
            return
        
        # 获取训练数据比例
        train_ratio = get_train_ratio()
        
        # 创建市场状态检测器
        market_detector = MarketRegimeMLDetector(
            model_type=model_type, 
            train_ratio=train_ratio,
            label_method=label_method
        )
        
        # 创建回测系统
        backtest = BacktestSystem(
            strategies=strategies,
            initial_capital=10000,
            market_detector=market_detector
        )
```

### 3. 模型训练阶段

```2238:2243:backtest_system_ML.py
    # 6. 训练模型（如果使用市场状态检测）
    if use_ml and backtest.market_detector:
        print("\n" + "=" * 60)
        print("训练市场状态检测模型")
        print("=" * 60)
        backtest.market_detector.train(backtest.data)
```

### 4. 回测执行阶段

```2245:2252:backtest_system_ML.py
    # 7. 运行回测
    print("\n" + "=" * 60)
    print("开始回测")
    print("=" * 60)
    max_entries_input = input("最大加仓次数 (默认3): ").strip()
    max_entries = int(max_entries_input) if max_entries_input else 3
    
    backtest.run_backtest(max_entries=max_entries)
```

### 5. 结果输出阶段

```2254:2263:backtest_system_ML.py
    # 8. 生成报告
    backtest.print_report()

    # 9. 绘制结果
    save_path = input("\n保存图表路径（直接回车使用默认）: ").strip()
    if not save_path:
        save_path = 'backtest_result_ML.png'
    backtest.plot_results(save_path=save_path)
    
    print(f"\n回测完成！结果已保存到: {save_path}")
```

---

## 机器学习模型详解

### 1. 市场状态检测器类：MarketRegimeMLDetector

#### 1.1 初始化

```187:216:backtest_system_ML.py
    def __init__(self, model_type: str = 'xgboost', train_ratio: float = 0.5,
                 label_method: str = 'forward_looking'):
        """
        初始化市场状态检测器
        
        Args:
            model_type: 模型类型 ('xgboost', 'random_forest', 'lstm', 'cnn')
            train_ratio: 训练数据比例（默认0.5，即50%用于训练，50%用于回测）
            label_method: 标签生成方法 ('forward_looking', 'unsupervised', 'adx_slope')
                - 'forward_looking': 前瞻性标签（使用未来信息）
                - 'unsupervised': 无监督学习（K-means聚类）
                - 'adx_slope': ADX+均线斜率（传统方法，有滞后性）
        """
        self.model_type = model_type
        self.train_ratio = train_ratio
        self.label_method = label_method
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_names = []
        
        # 无监督学习相关
        self.cluster_model = None  # 聚类模型（用于无监督学习）
        self.cluster_labels_map = None  # 聚类标签到市场状态的映射
```

### 2. 特征提取

#### 2.1 特征提取方法

系统提取多种技术指标作为特征：

```445:522:backtest_system_ML.py
    def _extract_features(self, data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        提取特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            lookback: 回看期（用于时间序列特征）
            
        Returns:
            特征DataFrame
        """
        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data['open']
        volume = data['volume'] if 'volume' in data.columns else pd.Series([0] * len(data))
        
        features = pd.DataFrame(index=data.index)
        
        # 1. 价格特征
        features['price_change'] = close.pct_change()
        features['high_low_ratio'] = (high - low) / close
        features['open_close_ratio'] = (close - open_price) / open_price
        
        # 2. 技术指标特征
        # ADX
        adx = calculate_adx(high, low, close, period=14)
        features['adx'] = adx
        
        # RSI
        rsi = calculate_rsi(close, period=14)
        features['rsi'] = rsi
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram
        
        # 布林带
        bb_upper, bb_middle, bb_lower = calculate_bb_bands(close, period=20)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        from strategies.indicators import calculate_atr
        atr = calculate_atr(high, low, close, period=14)
        features['atr'] = atr / close  # 归一化ATR
        
        # 均线
        ma5 = close.rolling(window=5).mean()
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()
        features['ma5'] = ma5 / close - 1
        features['ma10'] = ma10 / close - 1
        features['ma20'] = ma20 / close - 1
        
        # 均线斜率
        ma_slope = calculate_ma_slope(close, period=20, lookback=5)
        features['ma_slope'] = ma_slope / close  # 归一化斜率
        
        # 3. 时间序列特征（最近N期的统计量）
        for period in [5, 10, 20]:
            features[f'price_std_{period}'] = close.rolling(window=period).std() / close
            features[f'price_mean_{period}'] = close.rolling(window=period).mean() / close - 1
            features[f'volume_mean_{period}'] = volume.rolling(window=period).mean() if volume.sum() > 0 else 0
        
        # 4. 滞后特征
        for lag in [1, 2, 3, 5]:
            features[f'price_change_lag_{lag}'] = close.pct_change().shift(lag)
            features[f'adx_lag_{lag}'] = adx.shift(lag)
        
        # 填充NaN值
        features = features.bfill().fillna(0)
        
        self.feature_names = features.columns.tolist()
        
        return features
```

**特征说明：**
- **价格特征**：价格变化率、高低比、开收比
- **技术指标**：ADX、RSI、MACD、布林带、ATR、均线、均线斜率
- **时间序列特征**：不同周期的价格标准差、均值、成交量均值
- **滞后特征**：过去1、2、3、5期的价格变化和ADX

### 3. 标签生成方法

#### 3.1 前瞻性标签（Forward Looking）

```243:302:backtest_system_ML.py
    def _generate_forward_looking_labels(self, data: pd.DataFrame, is_training: bool = True) -> pd.Series:
        """
        生成前瞻性标签（使用未来信息）
        
        方法：使用未来N期的价格表现来判断当前市场状态
        - 如果未来有明显趋势 → 当前是趋势市场 (1)
        - 如果未来是震荡 → 当前是震荡市场 (0)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            is_training: 是否用于训练
            
        Returns:
            标签序列（1=趋势，0=震荡）
        """
        if not is_training:
            # 回测时无法使用前瞻性标签，返回全0（震荡）
            print("警告: 回测时无法使用前瞻性标签，返回默认标签")
            return pd.Series([0] * len(data), index=data.index)
        
        close = data['close']
        lookahead = 5  # 前瞻期数（可以调整）
        
        # 计算未来N期的价格变化
        future_price = close.shift(-lookahead)
        future_return = (future_price / close - 1).abs()
        
        # 计算未来N期的价格方向性（趋势强度）
        # 方法1：计算未来N期的累计收益率
        future_cumulative_return = abs(close.shift(-lookahead) / close - 1)
        
        # 方法2：计算未来N期的价格波动（低波动=趋势，高波动=震荡）
        future_volatility = close.rolling(window=lookahead).std().shift(-lookahead) / close
        
        # 方法3：计算未来N期的价格方向一致性
        future_returns = close.pct_change().shift(-lookahead)
        future_directional_strength = abs(future_returns.rolling(window=lookahead).sum())
        
        # 组合判断：
        # 1. 未来累计收益率大（有明显价格变化）
        # 2. 未来方向一致性强（价格朝一个方向移动）
        # 3. 未来波动率相对较低（不是来回震荡）
        
        return_threshold = 0.02  # 未来5期累计收益率阈值（2%）
        direction_threshold = 0.015  # 方向一致性阈值
        volatility_threshold = 0.03  # 波动率阈值（相对价格）
        
        # 趋势市场：收益率大 且 方向一致 且 波动率不太高
        is_trending = (
            (future_cumulative_return > return_threshold) &
            (future_directional_strength > direction_threshold) &
            (future_volatility < volatility_threshold)
        )
        
        labels = is_trending.astype(int)
        
        # 填充最后N期的NaN值（因为无法知道未来）
        labels.iloc[-lookahead:] = 0  # 默认标记为震荡
        
        return labels
```

**公式说明：**
- 未来累计收益率：`future_cumulative_return = |close[t+5] / close[t] - 1|`
- 方向一致性：`future_directional_strength = |Σ(return[t+i])|`，i=1到5
- 波动率：`future_volatility = std(close[t:t+5]) / close[t]`
- 趋势判断：`is_trending = (future_cumulative_return > 0.02) & (future_directional_strength > 0.015) & (future_volatility < 0.03)`

#### 3.2 无监督学习标签（K-means聚类）

```304:411:backtest_system_ML.py
    def _generate_unsupervised_labels(self, data: pd.DataFrame, is_training: bool = True) -> pd.Series:
        """
        使用无监督学习生成标签（K-means聚类）
        
        方法：
        1. 提取特征（价格变化、波动率、方向性等）
        2. 使用K-means聚类将市场状态分为2类
        3. 根据聚类结果的特征判断哪类是趋势、哪类是震荡
        
        Args:
            data: 包含OHLCV数据的DataFrame
            is_training: 是否用于训练
            
        Returns:
            标签序列（1=趋势，0=震荡）
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("无监督学习需要scikit-learn库")
        
        from sklearn.cluster import KMeans
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 提取用于聚类的特征
        cluster_features = pd.DataFrame(index=data.index)
        
        # 1. 价格变化特征
        cluster_features['return'] = close.pct_change()
        cluster_features['return_abs'] = abs(cluster_features['return'])
        cluster_features['return_std_5'] = close.pct_change().rolling(5).std()
        cluster_features['return_std_10'] = close.pct_change().rolling(10).std()
        
        # 2. 波动率特征
        cluster_features['high_low_range'] = (high - low) / close
        cluster_features['volatility_5'] = close.rolling(5).std() / close
        cluster_features['volatility_10'] = close.rolling(10).std() / close
        
        # 3. 方向性特征
        cluster_features['direction_5'] = (close - close.shift(5)) / close.shift(5)
        cluster_features['direction_10'] = (close - close.shift(10)) / close.shift(10)
        cluster_features['direction_consistency'] = abs(cluster_features['direction_5'] - cluster_features['direction_10'])
        
        # 4. 趋势强度特征
        ma_slope = calculate_ma_slope(close, period=20, lookback=5)
        cluster_features['ma_slope'] = abs(ma_slope) / close
        
        # 填充NaN值
        cluster_features = cluster_features.fillna(0)
        
        if is_training:
            # 训练时：使用K-means聚类
            print("使用K-means聚类进行无监督学习...")
            
            # 标准化特征
            cluster_features_scaled = self.scaler.fit_transform(cluster_features) if self.scaler else cluster_features.values
            
            # K-means聚类（2个类别：趋势和震荡）
            self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(cluster_features_scaled)
            
            # 分析每个聚类的特征，判断哪类是趋势、哪类是震荡
            # 趋势类特征：方向一致性强、波动率相对较低、有明确方向
            # 震荡类特征：方向一致性弱、波动率相对较高、来回波动
            
            cluster_centers = self.cluster_model.cluster_centers_
            feature_names = cluster_features.columns.tolist()
            
            # 计算每个聚类的平均特征值
            cluster_0_features = cluster_features[cluster_labels == 0].mean()
            cluster_1_features = cluster_features[cluster_labels == 1].mean()
            
            # 判断趋势强度（使用方向一致性和波动率）
            # 方向一致性高、波动率低 → 趋势市场
            direction_idx = feature_names.index('direction_consistency') if 'direction_consistency' in feature_names else 0
            volatility_idx = feature_names.index('volatility_5') if 'volatility_5' in feature_names else 1
            
            cluster_0_trend_score = cluster_centers[0][direction_idx] - cluster_centers[0][volatility_idx]
            cluster_1_trend_score = cluster_centers[1][direction_idx] - cluster_centers[1][volatility_idx]
            
            # 趋势分数高的类别标记为趋势市场(1)，低的标记为震荡市场(0)
            if cluster_0_trend_score > cluster_1_trend_score:
                self.cluster_labels_map = {0: 1, 1: 0}  # 类别0是趋势，类别1是震荡
            else:
                self.cluster_labels_map = {0: 0, 1: 1}  # 类别0是震荡，类别1是趋势
            
            print(f"聚类分析:")
            print(f"  类别0特征: 方向一致性={cluster_0_features.get('direction_consistency', 0):.4f}, "
                  f"波动率={cluster_0_features.get('volatility_5', 0):.4f}")
            print(f"  类别1特征: 方向一致性={cluster_1_features.get('direction_consistency', 0):.4f}, "
                  f"波动率={cluster_1_features.get('volatility_5', 0):.4f}")
            print(f"  类别0 → {'趋势' if self.cluster_labels_map[0] == 1 else '震荡'}")
            print(f"  类别1 → {'趋势' if self.cluster_labels_map[1] == 1 else '震荡'}")
            
            # 映射聚类标签到市场状态
            labels = pd.Series([self.cluster_labels_map[label] for label in cluster_labels], index=data.index)
            
        else:
            # 回测时：使用训练好的聚类模型
            if self.cluster_model is None or self.cluster_labels_map is None:
                raise ValueError("聚类模型未训练，无法生成标签")
            
            cluster_features_scaled = self.scaler.transform(cluster_features) if self.scaler else cluster_features.values
            cluster_labels = self.cluster_model.predict(cluster_features_scaled)
            labels = pd.Series([self.cluster_labels_map[label] for label in cluster_labels], index=data.index)
        
        return labels
```

**K-means聚类公式：**
- 目标函数：`J = Σ||x_i - μ_c_i||²`，其中 `μ_c_i` 是样本 `x_i` 所属簇的中心
- 趋势分数：`trend_score = direction_consistency - volatility`
- 根据趋势分数判断聚类类别对应的市场状态

#### 3.3 ADX+均线斜率标签（传统方法）

```413:443:backtest_system_ML.py
    def _generate_adx_slope_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        生成标签（使用ADX + 均线斜率，传统方法，有滞后性）
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            标签序列（1=趋势，0=震荡）
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 计算ADX
        adx = calculate_adx(high, low, close, period=14)
        
        # 计算均线斜率
        ma_slope = calculate_ma_slope(close, period=20, lookback=5)
        ma_slope_abs = abs(ma_slope)
        
        # 归一化均线斜率（相对于价格）
        normalized_slope = ma_slope_abs / close
        
        # 组合判断：ADX > 25 且 归一化斜率 > 0.001
        adx_threshold = 25
        slope_threshold = 0.001
        
        labels = ((adx > adx_threshold) & (normalized_slope > slope_threshold)).astype(int)
        
        return labels
```

**公式说明：**
- ADX计算：通过真实波幅(TR)、方向移动(+DM/-DM)计算
- 均线斜率：`ma_slope = (MA[t] - MA[t-5]) / 5`
- 归一化斜率：`normalized_slope = |ma_slope| / close[t]`
- 趋势判断：`is_trending = (ADX > 25) & (normalized_slope > 0.001)`

### 4. 机器学习模型

#### 4.1 XGBoost模型

**模型公式：**
XGBoost使用梯度提升决策树（GBDT）算法：

```
ŷ = Σ(k=1 to K) f_k(x)

其中：
- f_k 是第k棵树
- K 是树的数量
- 目标函数：L(θ) = Σl(y_i, ŷ_i) + ΣΩ(f_k)
```

**代码实现：**

```580:587:backtest_system_ML.py
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
```

**参数说明：**
- `n_estimators=100`：树的数量
- `max_depth=5`：树的最大深度
- `learning_rate=0.1`：学习率
- `random_state=42`：随机种子

#### 4.2 Random Forest模型

**模型公式：**
随机森林是多个决策树的集成：

```
ŷ = (1/K) * Σ(k=1 to K) T_k(x)

其中：
- T_k 是第k棵决策树
- K 是树的数量（默认100）
- 每棵树使用随机采样的特征和样本
```

**代码实现：**

```589:596:backtest_system_ML.py
        elif self.model_type == 'random_forest' and SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
```

**参数说明：**
- `n_estimators=100`：树的数量
- `max_depth=10`：树的最大深度
- `n_jobs=-1`：使用所有CPU核心

#### 4.3 LSTM模型

**模型公式：**
LSTM（长短期记忆网络）用于处理时间序列：

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # 遗忘门
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # 输入门
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # 候选值
C_t = f_t * C_{t-1} + i_t * C̃_t  # 细胞状态
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # 输出门
h_t = o_t * tanh(C_t)  # 隐藏状态
```

**代码实现：**

```598:608:backtest_system_ML.py
        elif self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
            self.model = self._build_lstm_model(X_train_scaled.shape[1])
            # 准备LSTM输入（需要3D数组）
            X_train_lstm = self._prepare_lstm_input(X_train_scaled)
            X_test_lstm = self._prepare_lstm_input(X_test_scaled)
            # 对齐标签（LSTM输入会减少sequence_length个样本）
            y_train_lstm = y_train.iloc[10:] if isinstance(y_train, pd.Series) else y_train[10:]
            y_test_lstm = y_test.iloc[10:] if isinstance(y_test, pd.Series) else y_test[10:]
            self.model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
            X_test_scaled = X_test_lstm
            y_test = y_test_lstm
```

**模型结构：**

```648:659:backtest_system_ML.py
    def _build_lstm_model(self, input_dim: int):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(10, input_dim)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
```

**参数说明：**
- 第一层LSTM：50个单元，返回序列
- Dropout：0.2（防止过拟合）
- 第二层LSTM：50个单元，不返回序列
- 全连接层：25个神经元，ReLU激活
- 输出层：1个神经元，Sigmoid激活（二分类）

**时间序列输入准备：**

```676:681:backtest_system_ML.py
    def _prepare_lstm_input(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """准备LSTM/CNN输入（时间序列格式）"""
        X_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
        return np.array(X_seq)
```

#### 4.4 CNN模型

**模型公式：**
CNN（卷积神经网络）用于模式识别：

```
卷积层：y = ReLU(Conv1D(x, W) + b)
池化层：y = MaxPooling(x)
全连接层：y = ReLU(W · x + b)
输出层：y = Sigmoid(W · x + b)
```

**代码实现：**

```610:620:backtest_system_ML.py
        elif self.model_type == 'cnn' and TENSORFLOW_AVAILABLE:
            self.model = self._build_cnn_model(X_train_scaled.shape[1])
            # 准备CNN输入（需要3D数组）
            X_train_cnn = self._prepare_lstm_input(X_train_scaled)
            X_test_cnn = self._prepare_lstm_input(X_test_scaled)
            # 对齐标签（CNN输入会减少sequence_length个样本）
            y_train_cnn = y_train.iloc[10:] if isinstance(y_train, pd.Series) else y_train[10:]
            y_test_cnn = y_test.iloc[10:] if isinstance(y_test, pd.Series) else y_test[10:]
            self.model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32, verbose=0)
            X_test_scaled = X_test_cnn
            y_test = y_test_cnn
```

**模型结构：**

```661:674:backtest_system_ML.py
    def _build_cnn_model(self, input_dim: int):
        """构建CNN模型"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, input_dim)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
```

**参数说明：**
- 第一层卷积：64个滤波器，核大小3
- 最大池化：池化大小2
- 第二层卷积：32个滤波器，核大小3
- 最大池化：池化大小2
- 全连接层：50个神经元，ReLU激活
- 输出层：1个神经元，Sigmoid激活

### 5. 模型训练流程

```524:646:backtest_system_ML.py
    def train(self, data: pd.DataFrame):
        """
        训练模型
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        print(f"\n开始训练市场状态检测模型（{self.model_type}）...")
        
        # 划分训练集和测试集
        train_size = int(len(data) * self.train_ratio)
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
        
        print(f"训练集大小: {len(train_data)} ({self.train_ratio*100:.1f}%)")
        print(f"测试集大小: {len(test_data)} ({(1-self.train_ratio)*100:.1f}%)")
        
        # 提取特征和标签
        print("提取特征...")
        X_train = self._extract_features(train_data)
        y_train = self._generate_labels(train_data, is_training=True)
        
        X_test = self._extract_features(test_data)
        # 测试集标签：如果是前瞻性标签，回测时无法使用，使用模型预测
        if self.label_method == 'forward_looking':
            # 前瞻性标签在回测时无法使用，使用默认标签（仅用于评估，实际预测时不用）
            y_test = self._generate_labels(test_data, is_training=False)
            print("注意: 前瞻性标签在回测时无法使用，测试集标签仅用于参考")
        else:
            y_test = self._generate_labels(test_data, is_training=False)
        
        # 移除包含NaN的行
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"有效训练样本: {len(X_train)}")
        print(f"有效测试样本: {len(X_test)}")
        print(f"训练集中趋势市场比例: {y_train.mean():.2%}")
        print(f"测试集中趋势市场比例: {y_test.mean():.2%}")
        
        # 标准化特征
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # 训练模型
        print(f"\n训练 {self.model_type} 模型...")
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif self.model_type == 'random_forest' and SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
            self.model = self._build_lstm_model(X_train_scaled.shape[1])
            # 准备LSTM输入（需要3D数组）
            X_train_lstm = self._prepare_lstm_input(X_train_scaled)
            X_test_lstm = self._prepare_lstm_input(X_test_scaled)
            # 对齐标签（LSTM输入会减少sequence_length个样本）
            y_train_lstm = y_train.iloc[10:] if isinstance(y_train, pd.Series) else y_train[10:]
            y_test_lstm = y_test.iloc[10:] if isinstance(y_test, pd.Series) else y_test[10:]
            self.model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
            X_test_scaled = X_test_lstm
            y_test = y_test_lstm
            
        elif self.model_type == 'cnn' and TENSORFLOW_AVAILABLE:
            self.model = self._build_cnn_model(X_train_scaled.shape[1])
            # 准备CNN输入（需要3D数组）
            X_train_cnn = self._prepare_lstm_input(X_train_scaled)
            X_test_cnn = self._prepare_lstm_input(X_test_scaled)
            # 对齐标签（CNN输入会减少sequence_length个样本）
            y_train_cnn = y_train.iloc[10:] if isinstance(y_train, pd.Series) else y_train[10:]
            y_test_cnn = y_test.iloc[10:] if isinstance(y_test, pd.Series) else y_test[10:]
            self.model.fit(X_train_cnn, y_test_cnn, epochs=50, batch_size=32, verbose=0)
            X_test_scaled = X_test_cnn
            y_test = y_test_cnn
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type} 或相关库未安装")
        
        # 评估模型
        if self.model_type in ['lstm', 'cnn']:
            # LSTM/CNN输出连续值，需要转换为0/1
            y_pred_proba = self.model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = self.model.predict(X_test_scaled)
        
        # 确保y_test和y_pred都是整数类型
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型训练完成！")
        print(f"测试集准确率: {accuracy:.2%}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['震荡', '趋势']))
        
        self.is_trained = True
```

**训练步骤：**
1. 数据划分：按 `train_ratio` 划分训练集和测试集
2. 特征提取：调用 `_extract_features()` 提取特征
3. 标签生成：调用 `_generate_labels()` 生成标签
4. 数据清洗：移除NaN值
5. 特征标准化：使用 `StandardScaler` 标准化
6. 模型训练：根据模型类型选择相应的训练方法
7. 模型评估：计算准确率和分类报告

### 6. 模型预测

```683:725:backtest_system_ML.py
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        预测市场状态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            预测结果序列（'trending' 或 'ranging'）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 提取特征
        X = self._extract_features(data)
        
        # 移除NaN行
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            return pd.Series(['ranging'] * len(data), index=data.index)
        
        # 标准化
        if self.scaler:
            X_scaled = self.scaler.transform(X_valid)
        else:
            X_scaled = X_valid.values
        
        # 预测
        if self.model_type in ['lstm', 'cnn']:
            X_seq = self._prepare_lstm_input(X_scaled)
            predictions = (self.model.predict(X_seq) > 0.5).astype(int).flatten()
            # 填充前面的序列长度
            predictions = np.concatenate([[0] * 10, predictions])
        else:
            predictions = self.model.predict(X_scaled)
        
        # 转换为字符串
        results = pd.Series(['ranging'] * len(data), index=data.index)
        results.loc[valid_mask] = ['trending' if p == 1 else 'ranging' for p in predictions[:len(X_valid)]]
        
        return results
```

**预测步骤：**
1. 提取特征
2. 数据清洗
3. 特征标准化
4. 模型预测（LSTM/CNN需要特殊处理）
5. 结果转换（0/1 → 'ranging'/'trending'）

---

## 机器学习信号在回测中的应用

### 1. 市场状态预测

在回测开始前，系统会预测整个数据集的市场状态：

```1200:1207:backtest_system_ML.py
        # 如果使用市场状态检测，先预测市场状态
        if self.use_market_regime and self.market_detector.is_trained:
            print("\n预测市场状态...")
            self.market_regimes = self.market_detector.predict(self.data)
            print(f"趋势市场占比: {(self.market_regimes == 'trending').sum() / len(self.market_regimes):.2%}")
            print(f"震荡市场占比: {(self.market_regimes == 'ranging').sum() / len(self.market_regimes):.2%}")
        else:
            self.market_regimes = None
```

### 2. 策略选择

在每个K线，系统根据预测的市场状态选择相应的策略：

```1233:1252:backtest_system_ML.py
            # 获取当前市场状态
            current_regime = None
            if self.use_market_regime and self.market_regimes is not None:
                current_regime = self.market_regimes.iloc[idx]
            
            # 根据市场状态选择策略（如果使用市场状态检测）
            if self.use_market_regime and self.market_regimes is not None and self.strategies is not None:
                # 根据市场状态选择策略
                if current_regime == 'trending' and 'trending' in self.strategies:
                    self.strategy = self.strategies['trending']
                elif current_regime == 'ranging' and 'ranging' in self.strategies:
                    self.strategy = self.strategies['ranging']
                else:
                    # 如果当前市场状态没有对应策略，不交易
                    self.strategy = None
            elif self.use_market_regime and self.market_regimes is not None and self.strategy is not None:
                # 单策略模式：检查市场状态是否匹配
                # 如果策略是趋势策略但市场是震荡，或反之，则不交易
                # 这里需要知道策略类型，暂时允许交易（可以在策略类中添加类型属性）
                pass
```

**策略选择逻辑：**
- 如果市场状态是 `'trending'` 且存在趋势策略 → 使用趋势策略
- 如果市场状态是 `'ranging'` 且存在震荡策略 → 使用震荡策略
- 如果市场状态没有对应策略 → 不交易（`self.strategy = None`）

### 3. 开仓信号生成

当没有持仓时，系统根据当前选择的策略生成开仓信号：

```1332:1376:backtest_system_ML.py
            # 如果没有可用策略，跳过开仓逻辑
            if self.strategy is None:
                continue

            # 检查入场信号
            if self.engine.position_size == 0:
                signal = self.strategy.generate_signals(self.data, idx, 0)

                if signal['signal'] == 'long':
                    # 计算仓位大小
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        position_size = self.strategy.get_position_size(
                            self.engine.balance,
                            atr_value,
                            signal['price'],
                            self.engine.leverage
                        )
                        self.engine.open_position(
                            'long',
                            signal['price'],
                            position_size,
                            idx,
                            signal['reason'],
                            current_regime
                        )
                        # 记录开仓时使用的策略
                        self.entry_strategy = self.strategy

                elif signal['signal'] == 'short':
                    # 计算仓位大小
                    atr_value = self.strategy.atr.iloc[idx]
                    if not pd.isna(atr_value) and atr_value > 0:
                        position_size = self.strategy.get_position_size(
                            self.engine.balance,
                            atr_value,
                            signal['price'],
                            self.engine.leverage
                        )
                        self.engine.open_position(
                            'short',
                            signal['price'],
                            position_size,
                            idx,
                            signal['reason'],
                            current_regime
                        )
                        # 记录开仓时使用的策略
                        self.entry_strategy = self.strategy
```

**关键点：**
- 开仓时记录市场状态：`current_regime` 被传递给 `open_position()`
- 记录开仓策略：`self.entry_strategy = self.strategy`，用于后续持仓管理

### 4. 持仓管理

当有持仓时，系统使用开仓时记录的策略进行持仓管理（即使市场状态变化）：

```1257:1326:backtest_system_ML.py
            # 如果有持仓，使用开仓时的策略（即使当前市场状态变化）
            if self.engine.position_size != 0:
                # 使用开仓时记录的策略
                current_strategy = self.entry_strategy if self.entry_strategy is not None else self.strategy
                
                if current_strategy is None:
                    # 如果开仓策略也不存在，强制平仓
                    pnl = self.engine.close_position(current_price, idx, "策略不可用，强制平仓")
                    if pnl is not None and self.entry_strategy:
                        self.entry_strategy.update_trade_result(pnl)
                    continue
                
                # 检查止损（移动止盈）
                stop_signal = current_strategy.check_stop_loss(
                    self.data, idx,
                    self.engine.position_size,
                    self.engine.entry_price
                )
                if stop_signal:
                    pnl = self.engine.close_position(
                        stop_signal['price'],
                        idx,
                        stop_signal['reason'],
                        current_regime
                    )
                    if pnl is not None:
                        current_strategy.update_trade_result(pnl)
                    self.entry_strategy = None  # 清除开仓策略记录
                    continue

                # 检查均线交叉退出
                signal = current_strategy.generate_signals(self.data, idx, self.engine.position_size)
                if signal['signal'] in ['close_long', 'close_short']:
                    pnl = self.engine.close_position(current_price, idx, signal['reason'], current_regime)
                    if pnl is not None:
                        current_strategy.update_trade_result(pnl)
                    self.entry_strategy = None  # 清除开仓策略记录
                    continue

                # 检查加仓
                if self.engine.entry_count < max_entries:
                    add_signal = current_strategy.check_add_position(
                        self.data, idx,
                        self.engine.position_size,
                        self.engine.entry_price
                    )
                    if add_signal:
                        # 计算加仓数量
                        atr_value = current_strategy.atr.iloc[idx]
                        if not pd.isna(atr_value) and atr_value > 0:
                            add_size = current_strategy.get_position_size(
                                self.engine.balance,
                                atr_value,
                                add_signal['price'],
                                self.engine.leverage
                            )
                            # 限制加仓数量不超过最大可开仓位
                            max_add_value = self.engine.balance * self.engine.leverage
                            max_add_size = max_add_value / add_signal['price'] if add_signal['price'] > 0 else 0
                            actual_add_size = min(add_size, max_add_size) if max_add_size > 0 else 0
                            if actual_add_size > 0:
                                self.engine.add_position(
                                    add_signal['signal'],
                                    add_signal['price'],
                                    actual_add_size,
                                    idx,
                                    add_signal['reason'],
                                    current_regime
                                )
                    continue
```

**持仓管理逻辑：**
- 使用开仓时的策略：`current_strategy = self.entry_strategy`
- 即使市场状态变化，仍使用开仓时的策略管理持仓
- 平仓时记录市场状态：`current_regime` 被传递给 `close_position()`

### 5. 交易记录中的市场状态

系统在每笔交易中记录市场状态：

```819:828:backtest_system_ML.py
            self.trades.append({
                'type': 'open_long',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason,
                'market_regime': market_regime  # 记录市场状态
            })
```

在导出CSV时，市场状态会被转换为中文：

```1724:1731:backtest_system_ML.py
            # 市场状态：记录交易时的市场状态（趋势/震荡）
            market_regime = trade.get('market_regime', 'N/A')
            if market_regime == 'trending':
                trade_record['市场状态'] = '趋势'
            elif market_regime == 'ranging':
                trade_record['市场状态'] = '震荡'
            else:
                trade_record['市场状态'] = market_regime
```

### 6. 完整流程总结

**机器学习信号参与回测的完整流程：**

1. **训练阶段**：
   - 提取特征（技术指标、价格特征等）
   - 生成标签（趋势/震荡）
   - 训练模型（XGBoost/Random Forest/LSTM/CNN）

2. **预测阶段**：
   - 对整个数据集预测市场状态
   - 生成 `market_regimes` 序列（每个K线对应一个状态）

3. **回测阶段**：
   - 遍历每个K线
   - 获取当前市场状态：`current_regime = self.market_regimes.iloc[idx]`
   - 根据市场状态选择策略：`self.strategy = self.strategies['trending']` 或 `self.strategies['ranging']`
   - 生成交易信号：`signal = self.strategy.generate_signals(...)`
   - 执行交易：`self.engine.open_position(..., current_regime)`
   - 记录市场状态：每笔交易都记录 `market_regime`

4. **结果分析**：
   - 在交易记录CSV中可以看到每笔交易的市场状态
   - 可以分析不同市场状态下的交易表现

---

## 核心类和方法说明

### 1. MarketRegimeMLDetector（市场状态检测器）

**主要方法：**
- `__init__()`: 初始化检测器
- `_extract_features()`: 提取特征
- `_generate_labels()`: 生成标签
- `train()`: 训练模型
- `predict()`: 预测市场状态

### 2. BacktestEngine（回测引擎）

**主要方法：**
- `open_position()`: 开仓
- `add_position()`: 加仓
- `close_position()`: 平仓
- `update_equity()`: 更新权益

### 3. BacktestSystem（回测系统）

**主要方法：**
- `load_data_from_csv()`: 从CSV加载数据
- `load_data_from_binance()`: 从币安API加载数据
- `run_backtest()`: 运行回测
- `generate_report()`: 生成报告
- `plot_results()`: 绘制结果
- `export_trades_to_csv()`: 导出交易记录

---

## 总结

`backtest_system_ML.py` 系统通过以下方式将机器学习信号应用到回测中：

1. **市场状态检测**：使用机器学习模型预测每个K线的市场状态（趋势/震荡）
2. **策略选择**：根据市场状态动态选择相应的交易策略
3. **信号生成**：使用选定的策略生成交易信号
4. **交易执行**：执行交易并记录市场状态
5. **结果分析**：在交易记录中可以看到每笔交易的市场状态，便于分析不同市场状态下的表现

这种设计使得系统能够根据市场状态自适应地选择策略，提高回测的准确性和实用性。

