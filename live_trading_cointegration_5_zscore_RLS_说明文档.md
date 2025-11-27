# 实盘协整交易系统详细说明文档

## 目录
1. [系统概述](#系统概述)
2. [程序运行流程](#程序运行流程)
3. [核心模型和算法](#核心模型和算法)
4. [代码详解](#代码详解)

---

## 系统概述

本系统是一个基于协整关系的实盘交易系统，支持：
- **6种Z-score计算策略**：传统方法、ARIMA-GARCH、ECM、Kalman Filter、Copula+DCC-GARCH、Regime-Switching
- **RLS动态对冲比率**：使用递归最小二乘法动态更新对冲比率
- **定期协整检验**：在使用RLS时，定期检验协整关系是否仍然存在，防止协整关系破裂导致的风险
- **实时数据管理**：自动收集和管理实时价格数据
- **Web监控界面**：实时监控交易状态

---

## 程序运行流程

### 1. 程序入口 (`if __name__ == "__main__"`)

**代码位置**：第2294行

```python
if __name__ == "__main__":
    main()
```

**执行流程**：
- 调用 `main()` 函数
- `main()` 函数调用 `test_live_trading()` 开始实盘交易

---

### 2. 策略选择 (`select_z_score_strategy()`)

**代码位置**：第1240-1434行

**功能**：让用户选择Z-score计算策略

**可选策略**：
1. **传统方法**：使用均值和标准差
2. **ARIMA-GARCH模型**：时间序列预测
3. **ECM误差修正模型**：协整误差修正
4. **Kalman Filter动态价差模型**：动态状态估计
5. **Copula + DCC-GARCH模型**：相关性建模
6. **Regime-Switching市场状态模型**：状态转换模型

**代码示例**：
```python
def select_z_score_strategy():
    """选择Z-score计算策略"""
    if not STRATEGIES_AVAILABLE:
        print("警告: 策略模块不可用，将使用传统方法")
        return None
    
    print("请选择Z-score计算策略:")
    print("  1. 传统方法（均值和标准差）")
    print("  2. ARIMA-GARCH模型")
    print("  3. ECM误差修正模型")
    print("  4. Kalman Filter动态价差模型")
    print("  5. Copula + DCC-GARCH相关性/波动率模型")
    print("  6. Regime-Switching市场状态模型")
    
    choice = input("请选择 (0-6): ").strip()
    # ... 根据选择创建对应策略对象
```

---

### 3. 币安API初始化 (`BinanceAPI`)

**代码位置**：第77-312行

**功能**：初始化币安API客户端，用于获取实时价格和执行交易

**关键方法**：
- `get_current_price(symbol)`: 获取当前价格
- `get_klines(symbol, interval, limit)`: 获取K线数据
- `place_order(...)`: 下单
- `get_account_info()`: 获取账户信息

**代码示例**：
```python
class BinanceAPI:
    def __init__(self, api_key=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
    
    def _generate_signature(self, query_string):
        """生成API签名"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
```

---

### 4. 币对配置 (`get_pairs_config()`)

**代码位置**：第1439-1553行

**功能**：配置要交易的币对信息

**配置内容**：
- 币对名称（symbol1, symbol2）
- 对冲比率（hedge_ratio）
- 差分阶数（diff_order）
- 协整状态（cointegration_found）

**代码示例**：
```python
def get_pairs_config():
    """获取币对配置（用户输入）"""
    pairs_config = []
    
    while True:
        symbol1 = input("请输入第一个币种 (如: BTCUSDT): ").strip().upper()
        symbol2 = input("请输入第二个币种 (如: ETHUSDT): ").strip().upper()
        hedge_ratio = float(input("请输入对冲比率: ").strip())
        
        pair_info = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'pair_name': f"{symbol1}_{symbol2}",
            'diff_order': 0,  # 0=原始价差, 1=一阶差分, 2=二阶差分
            'cointegration_found': True
        }
        pairs_config.append(pair_info)
```

---

### 5. 交易参数配置 (`configure_trading_parameters()`)

**代码位置**：第1554-1684行

**功能**：配置交易策略参数

**参数列表**：
- `lookback_period`: 回看期（用于计算Z-score的历史数据长度）
- `z_threshold`: Z-score开仓阈值
- `z_exit_threshold`: Z-score平仓阈值
- `take_profit_pct`: 止盈百分比
- `stop_loss_pct`: 止损百分比
- `max_holding_hours`: 最大持仓时间
- `position_ratio`: 仓位比例
- `leverage`: 杠杆倍数
- `trading_fee_rate`: 交易手续费率

---

### 6. 预热数据收集 (`RealTimeDataManager.collect_warmup_data()`)

**代码位置**：第313-416行

**功能**：在开始交易前收集足够的历史数据

**流程**：
1. 计算需要的预热数据量：`warmup_period = lookback_period + warmup_safety_margin`
2. 从币安API获取历史K线数据
3. 存储到数据缓存中

**代码示例**：
```python
def collect_warmup_data(self, symbols, interval='1h', warmup_period=100):
    """收集预热数据"""
    warmup_data = {}
    
    for symbol in symbols:
        # 获取历史K线数据
        klines = self.binance_api.get_klines(
            symbol=symbol,
            interval=interval,
            limit=warmup_period
        )
        
        # 转换为pandas Series
        prices = pd.Series([float(k[4]) for k in klines])  # 收盘价
        warmup_data[symbol] = prices
    
    return warmup_data
```

---

### 7. RLS参数配置和初始化

**代码位置**：第2258-2371行

**功能**：配置并初始化RLS（递归最小二乘）算法和定期协整检验

**配置流程**：

#### 7.1 选择是否使用RLS
```python
use_rls_input = input("是否使用RLS动态对冲比率? (y/n, 默认y): ").strip().lower()
use_rls = use_rls_input != 'n'
```

#### 7.2 配置协整监控窗口参数（如果使用RLS）
如果选择使用RLS，系统会提示配置协整检验窗口大小：

```python
if use_rls:
    print("\n配置协整监控窗口参数（用于定期协整检验）")
    print("该窗口大小用于定期协整检验，验证协整关系是否仍然存在")
    window_input = input(f"协整检验窗口大小 (默认500): ").strip()
    cointegration_window_size = int(window_input) if window_input else 500
    
    # 使用窗口大小作为协整检验间隔
    cointegration_check_interval = cointegration_window_size
    print(f"  协整检验窗口大小: {cointegration_window_size} 个数据点")
    print(f"  协整检验间隔: {cointegration_check_interval} 个数据点（使用窗口大小）")
```

**重要说明**：
- **协整检验窗口大小**：用于定期协整检验时使用的数据窗口大小（默认500个数据点）
- **协整检验间隔**：每隔多少数据点进行一次协整检验（等于窗口大小）
- 例如：窗口大小为500，则每500个数据点进行一次协整检验，每次检验使用最近500个数据点

#### 7.3 配置RLS参数
```python
if use_rls:
    rls_lambda = float(input("RLS遗忘因子 (默认0.99): ").strip() or "0.99")
    rls_max_change_rate = float(input("RLS最大变化率 (默认0.2): ").strip() or "0.2")
```

**RLS参数说明**：
- `use_rls`: 是否使用RLS（默认True）
- `rls_lambda`: 遗忘因子（0 < λ ≤ 1，默认0.99）
- `rls_max_change_rate`: 最大变化率（防止突变，默认0.2）

#### 7.4 RLS初始化
```python
if use_rls:
    print("\n初始化RLS...")
    for pair_info in pairs_config:
        symbol1, symbol2 = pair_info['symbol1'], pair_info['symbol2']
        pair_key = f"{symbol1}_{symbol2}"
        
        # 使用预热数据初始化RLS
        init_price1 = warmup_data[symbol1]
        init_price2 = warmup_data[symbol2]
        
        trading_strategy.initialize_rls_for_pair(pair_key, init_price1, init_price2)
```

**初始化协整状态**：
在初始化RLS时，系统会同时初始化协整状态：
```python
self.cointegration_status[pair_key] = {
    'is_cointegrated': True,  # 初始假设协整
    'last_check_index': 0,  # 上次检验的数据点索引
    'cointegration_ratio': 1.0,  # 协整比率（1.0=完全协整，0.0=完全破裂）
    'last_hedge_ratio': rls.get_hedge_ratio(),
    'consecutive_failures': 0  # 连续失败计数
}
```

---

### 8. 交易策略初始化 (`AdvancedCointegrationTrading`)

**代码位置**：第879-1000行

**功能**：初始化协整交易策略对象

**初始化参数**：
```python
# 获取价差类型（从第一个币对获取，假设所有币对使用相同的价差类型）
diff_order = pairs_config[0].get('diff_order', 0) if pairs_config else 0

trading_strategy = AdvancedCointegrationTrading(
    binance_api=binance_api,
    lookback_period=trading_params['lookback_period'],
    z_threshold=trading_params['z_threshold'],
    z_exit_threshold=trading_params['z_exit_threshold'],
    take_profit_pct=trading_params['take_profit_pct'],
    stop_loss_pct=trading_params['stop_loss_pct'],
    max_holding_hours=trading_params['max_holding_hours'],
    position_ratio=trading_params['position_ratio'],
    leverage=trading_params['leverage'],
    trading_fee_rate=trading_params['trading_fee_rate'],
    z_score_strategy=z_score_strategy,
    use_rls=use_rls,
    rls_lambda=rls_lambda,
    rls_max_change_rate=rls_max_change_rate,
    cointegration_window_size=cointegration_window_size,  # 协整检验窗口大小
    cointegration_check_interval=cointegration_check_interval,  # 协整检验间隔
    diff_order=diff_order  # 价差类型
)
```

**新增参数说明**：
- `cointegration_window_size`: 协整检验窗口大小（默认500，与用户输入一致）
- `cointegration_check_interval`: 协整检验间隔（默认500，等于窗口大小）
- `diff_order`: 价差类型（0=原始价差，1=一阶差分价差）

**协整状态跟踪**：
策略类内部维护协整状态：
```python
self.cointegration_status = {}  # {pair_key: {'is_cointegrated': bool, 'last_check_index': int, 'cointegration_ratio': float}}
self.data_point_count = {}  # {pair_key: count}  # 数据点计数器（用于实盘交易）
```

---

### 9. 实时交易循环 (`trading_loop()`)

**代码位置**：第1959-2251行

**功能**：实时监控市场并执行交易

**循环流程**：

#### 9.1 获取当前数据
```python
current_data = data_manager.get_current_data()
current_prices = data_manager.get_current_prices()
```

#### 9.2 更新RLS对冲比率（如果启用）
```python
if trading_strategy.use_rls and pair_key in trading_strategy.rls_instances:
    # 更新RLS对冲比率
    current_hedge_ratio = trading_strategy.update_rls_for_pair(
        pair_key, current_prices[symbol1], current_prices[symbol2]
    )
    if current_hedge_ratio is None:
        current_hedge_ratio = pair_info.get('hedge_ratio', 1.0)
    
    # 更新数据点计数器（用于定期协整检验）
    if pair_key not in trading_strategy.data_point_count:
        trading_strategy.data_point_count[pair_key] = 0
    trading_strategy.data_point_count[pair_key] += 1
    
    # 定期协整检验
    if pair_key in trading_strategy.cointegration_status:
        data1 = current_data[symbol1]
        data2 = current_data[symbol2]
        
        coint_check_result = trading_strategy.check_cointegration_periodically(
            pair_key, data1, data2, symbol1, symbol2
        )
        
        # 如果协整关系破裂，根据协整比率决定是否交易
        cointegration_ratio = coint_check_result.get('cointegration_ratio', 1.0)
        if cointegration_ratio <= 0:
            # 完全暂停交易
            continue
        # 如果协整比率很低（<0.2），也暂停交易
        elif cointegration_ratio < 0.2:
            continue
else:
    # 使用静态对冲比率
    current_hedge_ratio = pair_info.get('hedge_ratio', 1.0)
```

#### 9.3 定期协整检验（如果使用RLS）

**代码位置**：第2575-2595行

**功能**：在使用RLS时，定期检验协整关系是否仍然存在

**检验流程**：

1. **更新数据点计数器**：
```python
# 更新数据点计数器（用于定期协整检验）
if pair_key not in trading_strategy.data_point_count:
    trading_strategy.data_point_count[pair_key] = 0
trading_strategy.data_point_count[pair_key] += 1
```

2. **执行定期协整检验**：
```python
# 定期协整检验
if pair_key in trading_strategy.cointegration_status:
    data1 = current_data[symbol1]
    data2 = current_data[symbol2]
    
    coint_check_result = trading_strategy.check_cointegration_periodically(
        pair_key, data1, data2, symbol1, symbol2
    )
```

3. **根据协整比率决定是否交易**：
```python
# 如果协整关系破裂，根据协整比率决定是否交易
cointegration_ratio = coint_check_result.get('cointegration_ratio', 1.0)
if cointegration_ratio <= 0:
    # 完全暂停交易
    continue
# 如果协整比率很低（<0.2），也暂停交易
elif cointegration_ratio < 0.2:
    continue
```

**重要说明**：
- 定期协整检验只在**使用RLS**时执行
- 检验间隔 = 协整检验窗口大小（用户输入）
- 检验窗口大小 = 协整检验窗口大小（用户输入）
- 如果协整关系破裂，系统会渐进式暂停交易，避免不必要的损失

#### 9.4 计算价差
根据 `diff_order` 选择计算方式：

**原始价差**（diff_order=0）：
```python
current_spread = price1 - hedge_ratio * price2
```

**一阶差分价差**（diff_order=1）：
```python
diff1 = price1[t] - price1[t-1]
diff2 = price2[t] - price2[t-1]
current_spread = diff1 - hedge_ratio * diff2
```

**二阶差分价差**（diff_order=2）：
```python
diff2_1 = price1[t] - 2*price1[t-1] + price1[t-2]
diff2_2 = price2[t] - 2*price2[t-1] + price2[t-2]
current_spread = diff2_1 - hedge_ratio * diff2_2
```

#### 9.5 计算Z-score
```python
current_z_score = trading_strategy.calculate_z_score(
    current_spread, 
    historical_spreads,
    historical_prices1=historical_prices1,
    historical_prices2=historical_prices2
)
```

#### 9.6 生成交易信号
```python
signal = trading_strategy.generate_trading_signal(current_z_score)
```

**信号规则**：
- `Z-score > z_threshold` → `SHORT_LONG`（做空价差）
- `Z-score < -z_threshold` → `LONG_SHORT`（做多价差）
- 否则 → `HOLD`（观望）

#### 9.7 检查平仓条件
```python
if pair_info['pair_name'] in trading_strategy.positions:
    should_close, close_reason = trading_strategy.check_exit_conditions(
        pair_info, current_prices, current_z_score, datetime.now(), current_spread
    )
    
    if should_close:
        trading_strategy.close_position(...)
```

**平仓条件**：
1. Z-score回归到 `[-z_exit_threshold, z_exit_threshold]` 区间
2. 达到止盈百分比
3. 达到止损百分比
4. 超过最大持仓时间

#### 9.8 执行开仓
```python
if signal['action'] != 'HOLD' and len(trading_strategy.positions) == 0:
    trading_strategy.execute_trade(
        pair_info_with_rls, 
        current_prices, 
        signal, 
        datetime.now(),
        current_spread, 
        available_capital
    )
```

**部分成交处理**（新增功能）：

当配对交易出现部分成交时（一个订单成功，另一个订单失败），系统会自动平掉已成交的订单，避免单边持仓风险。

**处理流程**：

1. **检测部分成交**：
   - 如果`order1`成功但`order2`失败
   - 如果`order2`成功但`order1`失败

2. **自动紧急平仓**：
   ```python
   # 第一个订单成功，第二个订单失败
   elif order1 and order1.get('orderId') and (not order2 or not order2.get('orderId')):
       print(f"配对交易失败: {symbol1} 成功，{symbol2} 失败")
       print(f"  正在紧急平仓 {symbol1}...")
       
       # 根据信号方向确定平仓方向
       if signal['action'] == 'SHORT_LONG':
           close_side = 'BUY'  # 做空symbol1，需要买入平仓
       else:  # LONG_SHORT
           close_side = 'SELL'  # 做多symbol1，需要卖出平仓
       
       # 紧急平仓第一个订单
       close_success = self.emergency_close_position(
           symbol1, close_side, quantity1, f"配对交易失败，{symbol2}下单失败"
       )
   ```

3. **紧急平仓方法**：
   ```python
   def emergency_close_position(self, symbol, side, quantity, reason="紧急平仓"):
       """紧急平仓单个仓位"""
       # 执行平仓订单
       order = self.binance_api.place_order(symbol, side, quantity)
       
       # 等待平仓订单成交
       success, final_status, _ = self.wait_for_orders_completion(
           order, None, symbol, None, max_wait=10
       )
       
       return success
   ```

**风险控制**：
- 如果紧急平仓成功，系统会提示"✓ 紧急平仓成功，风险已控制"
- 如果紧急平仓失败，系统会提示"✗ 紧急平仓失败，请手动处理"，需要手动处理仓位

**重要性**：
配对交易的核心是对冲风险，如果只有一个币种成交，就会形成单边持仓，暴露方向性风险。自动平仓机制可以及时控制这种风险。

---

### 10. 实时交易服务器框架

**代码位置**：第1728-1788行（`LiveTradingServer`类）

**功能**：提供Web监控界面和API接口，实时监控交易状态

#### 10.1 系统架构

系统采用**多线程架构**，包含以下组件：

```
┌─────────────────────────────────────────────────────────┐
│                   主进程 (Main Process)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  Flask Web服务器  │  │   交易循环线程    │            │
│  │   (主线程)        │  │  (trading_loop)  │            │
│  │                  │  │                  │            │
│  │  - 提供Web界面   │  │  - 监控市场      │            │
│  │  - API接口       │  │  - 执行交易      │            │
│  │  - 端口: 5000    │  │  - 检查平仓条件  │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                          │
│  ┌──────────────────┐                                  │
│  │  数据收集线程      │                                  │
│  │  (update_thread)  │                                  │
│  │                    │                                  │
│  │  - 定期获取K线数据 │                                  │
│  │  - 更新数据缓存    │                                  │
│  └──────────────────┘                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**线程说明**：

1. **Flask Web服务器（主线程）**：
   - 运行Flask应用，监听HTTP请求
   - 提供Web监控界面和RESTful API
   - 阻塞主线程，直到程序退出

2. **交易循环线程（`trading_loop`）**：
   - 独立线程，后台运行
   - 定期检查市场数据
   - 执行交易决策（开仓、平仓）
   - 只在K线收盘时执行交易

3. **数据收集线程（`update_thread`）**：
   - 独立线程，后台运行
   - 定期从币安API获取最新K线数据
   - 更新数据缓存（`data_cache`）

#### 10.2 Flask Web服务器

**初始化**：
```python
class LiveTradingServer:
    def __init__(self, trading_strategy):
        self.trading_strategy = trading_strategy
        self.app = Flask(__name__, static_folder='templates', static_url_path='')
        self.setup_routes()
```

**路由设置**：
```python
def setup_routes(self):
    """设置路由"""
    
    @self.app.route('/')
    def index():
        """主页面"""
        return self.app.send_static_file('live_trading_monitor.html')
    
    @self.app.route('/api/status')
    def get_status():
        """获取交易状态"""
        return jsonify(self.trading_strategy.get_trading_status())
    
    @self.app.route('/api/positions')
    def get_positions():
        """获取当前持仓"""
        return jsonify(self.trading_strategy.positions)
    
    @self.app.route('/api/trades')
    def get_trades():
        """获取交易记录"""
        return jsonify(self.trading_strategy.trades)
    
    @self.app.route('/api/capital_curve')
    def get_capital_curve():
        """获取资金曲线"""
        return jsonify(self.trading_strategy.capital_curve)
    
    @self.app.route('/api/start_trading', methods=['POST'])
    def start_trading():
        """开始交易"""
        if not self.trading_strategy.running:
            self.trading_strategy.running = True
            return jsonify({'status': 'success', 'message': '交易已开始'})
        else:
            return jsonify({'status': 'error', 'message': '交易已在运行中'})
    
    @self.app.route('/api/stop_trading', methods=['POST'])
    def stop_trading():
        """停止交易"""
        if self.trading_strategy.running:
            self.trading_strategy.running = False
            return jsonify({'status': 'success', 'message': '交易已停止'})
        else:
            return jsonify({'status': 'error', 'message': '交易未在运行'})
```

#### 10.3 API接口说明

**GET接口**：

| 路径 | 说明 | 返回数据 |
|------|------|---------|
| `/` | Web监控界面 | HTML页面 |
| `/api/status` | 获取交易状态 | 运行状态、资金、持仓数、交易数等 |
| `/api/positions` | 获取当前持仓 | 所有持仓的详细信息 |
| `/api/trades` | 获取交易记录 | 所有历史交易记录 |
| `/api/capital_curve` | 获取资金曲线 | 资金曲线数据（时间序列） |

**POST接口**：

| 路径 | 说明 | 请求体 | 返回数据 |
|------|------|--------|---------|
| `/api/start_trading` | 开始交易 | 无 | `{'status': 'success', 'message': '交易已开始'}` |
| `/api/stop_trading` | 停止交易 | 无 | `{'status': 'success', 'message': '交易已停止'}` |

**状态接口返回示例**：
```json
{
    "running": true,
    "current_capital": 10000.0,
    "initial_capital": 10000.0,
    "total_return": 0.0,
    "positions_count": 1,
    "total_trades": 5,
    "positions": {
        "BTCUSDT_ETHUSDT": {
            "pair": "BTCUSDT_ETHUSDT",
            "symbol1": "BTCUSDT",
            "symbol2": "ETHUSDT",
            "symbol1_size": -0.1,
            "symbol2_size": 1.5,
            "entry_prices": {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0},
            "entry_spread": 100.0,
            "entry_time": "2024-01-01T10:00:00"
        }
    },
    "recent_trades": [...]
}
```

#### 10.4 服务器启动流程

**代码位置**：第2491-2550行

```python
# 12. 启动Web服务器
print("\n11. 启动Web服务器")
server = LiveTradingServer(trading_strategy)

# 13. 启动交易循环
print("\n12. 启动交易循环")
trading_strategy.running = True

# 获取预热数据中的最后一个K线时间戳，作为基准时间戳
initial_kline_timestamp = data_manager.get_latest_closed_kline_timestamp()

# 定义交易循环函数
def trading_loop():
    """交易循环（只在K线收盘时执行交易决策）"""
    last_processed_kline_timestamp = initial_kline_timestamp
    
    while trading_strategy.running:
        # 交易逻辑...
        pass

# 启动交易循环线程
trading_thread = threading.Thread(target=trading_loop)
trading_thread.daemon = True
trading_thread.start()

# 启动Web服务器（阻塞主线程）
try:
    server.run(host='0.0.0.0', port=5000, debug=False)
except KeyboardInterrupt:
    print("\n停止实盘交易...")
    trading_strategy.running = False
    data_manager.stop_data_collection()
    print("实盘交易已停止")
```

**启动顺序**：

1. 创建`LiveTradingServer`实例
2. 设置`trading_strategy.running = True`
3. 启动交易循环线程（后台运行）
4. 启动Flask Web服务器（阻塞主线程，监听HTTP请求）

**线程生命周期**：

- **主线程**：运行Flask服务器，直到程序退出（Ctrl+C）
- **交易循环线程**：后台运行，检查`trading_strategy.running`标志
- **数据收集线程**：后台运行，由`RealTimeDataManager`管理

#### 10.5 数据流架构

```
币安API
   │
   ├─→ RealTimeDataManager (数据收集线程)
   │      │
   │      └─→ data_cache (数据缓存)
   │             │
   │             ├─→ trading_loop (交易循环线程)
   │             │      │
   │             │      └─→ AdvancedCointegrationTrading
   │             │             │
   │             │             ├─→ 计算价差和Z-score
   │             │             ├─→ 生成交易信号
   │             │             └─→ 执行交易
   │             │
   │             └─→ LiveTradingServer (Web服务器)
   │                    │
   │                    └─→ API接口 → Web界面
```

**数据同步**：

- 所有线程共享`trading_strategy`对象
- 数据收集线程更新`data_cache`
- 交易循环线程读取`data_cache`进行交易决策
- Web服务器读取`trading_strategy`状态提供API

#### 10.6 访问Web界面

启动程序后，可以通过浏览器访问：

```
http://localhost:5000
```

或者从其他设备访问（如果在服务器上运行）：

```
http://服务器IP:5000
```

**Web界面功能**：

- 实时显示交易状态
- 显示当前持仓
- 显示交易记录
- 显示资金曲线
- 可以启动/停止交易

---

## 核心模型和算法

### 1. RLS（递归最小二乘）算法

**代码位置**：第417-591行

**模型公式**：

**线性回归模型**：
```
price1_t = β₀ + β₁ × price2_t + ε_t
```

其中：
- `β₀`: 截距
- `β₁`: 对冲比率（斜率）
- `ε_t`: 误差项

**RLS更新公式**：

1. **特征向量**：
```
x_t = [1, price2_t]ᵀ
```

2. **预测误差**：
```
e_t = price1_t - x_tᵀ × β_{t-1}
```

3. **Kalman增益**：
```
K_t = P_{t-1} × x_t / (λ + x_tᵀ × P_{t-1} × x_t)
```

其中：
- `P_{t-1}`: 协方差矩阵
- `λ`: 遗忘因子（0 < λ ≤ 1）

4. **参数更新**：
```
β_t = β_{t-1} + K_t × e_t
```

5. **协方差矩阵更新**：
```
P_t = (P_{t-1} - K_t × x_tᵀ × P_{t-1}) / λ
```

**代码实现**：
```python
def update(self, price1_t, price2_t):
    """更新对冲比率（RLS更新步骤）"""
    # 特征向量
    x_t = np.array([1.0, price2_t])
    y_t = price1_t
    
    # 预测误差
    prediction = np.dot(x_t, self.beta)
    error = y_t - prediction
    
    # Kalman增益
    denominator = self.lambda_forgetting + np.dot(x_t, np.dot(self.P, x_t))
    K_t = np.dot(self.P, x_t) / denominator
    
    # 更新参数
    beta_new = self.beta + K_t * error
    
    # 限制变化率（防止突变）
    if len(self.beta_history) > 0:
        beta_old = self.beta_history[-1]
        change_rate = abs((beta_new[1] - beta_old[1]) / (beta_old[1] + 1e-8))
        
        if change_rate > self.max_change_rate:
            max_change = self.max_change_rate * abs(beta_old[1])
            if beta_new[1] > beta_old[1]:
                beta_new[1] = beta_old[1] + max_change
            else:
                beta_new[1] = beta_old[1] - max_change
    
    # 更新协方差矩阵
    self.P = (self.P - np.outer(K_t, np.dot(self.P, x_t))) / self.lambda_forgetting
    
    # 更新状态
    self.beta = beta_new
    self.beta_history.append(self.beta.copy())
    
    return self.beta[1]  # 返回对冲比率
```

---

### 2. 定期协整检验

**代码位置**：第1125-1230行（`check_cointegration_periodically`方法）

**功能**：在使用RLS时，定期检验协整关系是否仍然存在，防止协整关系破裂导致的风险

**检验流程**：

#### 2.1 检验间隔判断
```python
# 检查是否需要重新检验（每N个数据点检验一次）
current_index = self.data_point_count.get(pair_key, 0)
last_check = status['last_check_index']

if current_index - last_check < self.cointegration_check_interval:
    # 不需要检验，返回当前状态
    return {'is_cointegrated': status['is_cointegrated'], 'cointegration_ratio': status.get('cointegration_ratio', 1.0)}
```

**说明**：
- 每隔 `cointegration_check_interval` 个数据点进行一次协整检验
- 协整检验间隔 = 协整检验窗口大小（用户输入）

#### 2.2 确定检验窗口大小
```python
# 使用与初始筛选相同的窗口大小进行协整检验
max_window_size = min(len(price1_series), len(price2_series))
target_window_size = self.cointegration_window_size  # 用户输入的窗口大小

# 如果可用数据少于目标窗口大小，使用可用数据的80%
if max_window_size < target_window_size:
    window_size = max(100, int(max_window_size * 0.8))  # 至少100个数据点
else:
    window_size = target_window_size
```

**说明**：
- 优先使用用户输入的窗口大小（`cointegration_window_size`）
- 如果可用数据不足，使用可用数据的80%，但至少需要100个数据点

#### 2.3 执行协整检验
```python
# 获取最近的数据
recent_price1 = price1_series.iloc[-window_size:]
recent_price2 = price2_series.iloc[-window_size:]

# 执行协整检验
coint_result = enhanced_cointegration_test(
    recent_price1, recent_price2, symbol1, symbol2,
    verbose=False, diff_order=self.diff_order
)

is_cointegrated = coint_result.get('cointegration_found', False)
```

**协整检验方法**：
使用 `enhanced_cointegration_test` 函数，该方法：
1. 检验两个价格序列的积分阶数
2. 计算最优对冲比率（OLS回归）
3. 计算价差（根据diff_order选择原始价差或差分价差）
4. 使用ADF检验价差的平稳性
5. 如果价差平稳，则协整关系成立

#### 2.4 协整关系破裂处理（渐进式暂停机制）

**连续失败次数与协整比率**：

| 连续失败次数 | 协整比率 | 仓位调整 | 说明 |
|------------|---------|---------|------|
| 0（通过） | 100% | 100% | 正常交易 |
| 1次失败 | 50% | 50% | 减少交易仓位 |
| 2次失败 | 20% | 20% | 大幅减少交易仓位 |
| 3次及以上 | 0% | 0% | 完全暂停交易 |

**代码实现**：
```python
if is_cointegrated:
    # 协整检验通过
    status['cointegration_ratio'] = 1.0
    status['consecutive_failures'] = 0  # 重置连续失败计数
else:
    # 协整检验失败
    consecutive_failures = status.get('consecutive_failures', 0) + 1
    status['consecutive_failures'] = consecutive_failures
    
    # 根据连续失败次数决定协整比率
    if consecutive_failures == 1:
        status['cointegration_ratio'] = 0.5
    elif consecutive_failures == 2:
        status['cointegration_ratio'] = 0.2
    else:
        status['cointegration_ratio'] = 0.0  # 完全暂停交易
```

#### 2.5 交易决策
```python
# 如果协整关系破裂，根据协整比率决定是否交易
cointegration_ratio = coint_check_result.get('cointegration_ratio', 1.0)
if cointegration_ratio <= 0:
    # 完全暂停交易
    continue
elif cointegration_ratio < 0.2:
    # 协整比率很低，也暂停交易
    continue
```

**说明**：
- 如果协整比率 ≤ 0，完全暂停该币对的交易
- 如果协整比率 < 0.2，也暂停交易
- 只有在协整比率 ≥ 0.2 时才允许交易

---

### 3. Z-score计算

**代码位置**：第819-850行

**传统方法公式**：
```
Z-score = (当前价差 - 历史价差均值) / 历史价差标准差
```

```
Z_t = (spread_t - μ) / σ
```

其中：
- `spread_t`: 当前价差
- `μ`: 历史价差均值
- `σ`: 历史价差标准差

**代码实现**：
```python
def calculate_z_score(self, current_spread, historical_spreads, 
                     historical_prices1=None, historical_prices2=None):
    """计算当前Z-score（使用策略对象）"""
    if self.z_score_strategy is not None:
        return self.z_score_strategy.calculate_z_score(
            current_spread, 
            historical_spreads,
            historical_prices1=historical_prices1,
            historical_prices2=historical_prices2
        )
    
    # 传统方法
    if len(historical_spreads) < 2:
        return 0.0
    
    spread_mean = np.mean(historical_spreads)
    spread_std = np.std(historical_spreads)
    
    if spread_std == 0:
        return 0.0
    
    return (current_spread - spread_mean) / spread_std
```

---

### 4. 交易信号生成

**代码位置**：第851-870行

**信号规则**：

```
信号 = {
    'SHORT_LONG':  如果 Z-score > z_threshold
    'LONG_SHORT':  如果 Z-score < -z_threshold
    'HOLD':        其他情况
}
```

**代码实现**：
```python
def generate_trading_signal(self, z_score):
    """生成交易信号"""
    if z_score > self.z_threshold:
        return {
            'action': 'SHORT_LONG',
            'description': f'Z-score过高({z_score:.3f})，做空价差',
            'confidence': min(abs(z_score) / 3.0, 1.0)
        }
    elif z_score < -self.z_threshold:
        return {
            'action': 'LONG_SHORT',
            'description': f'Z-score过低({z_score:.3f})，做多价差',
            'confidence': min(abs(z_score) / 3.0, 1.0)
        }
    else:
        return {
            'action': 'HOLD',
            'description': f'Z-score正常({z_score:.3f})，观望',
            'confidence': 0.0
        }
```

---

### 5. 仓位计算（Beta中性）

**代码位置**：第712-754行

**模型公式**：

**目标**：构建Beta中性的投资组合，使得：
```
ΔPortfolio = ΔPrice1 - β × ΔPrice2 = 0
```

**资金分配**：

1. **总资金占用系数**：
```
C = price1 + β × price2
```

2. **Symbol1数量**：
```
Q1 = AvailableCapital / C
```

3. **Symbol2数量**：
```
Q2 = β × Q1
```

4. **实际资金占用**：
```
TotalCapital = |Q1| × price1 + |Q2| × price2
```

**代码实现**：
```python
def calculate_position_size_beta_neutral(self, available_capital, price1, price2, hedge_ratio, signal):
    """基于Beta中性计算开仓数量"""
    # 计算总资金占用系数
    capital_coefficient = price1 + hedge_ratio * price2
    
    # 计算symbol1的数量（绝对值）
    symbol1_size_abs = available_capital / capital_coefficient
    
    # 计算symbol2的数量（绝对值）
    symbol2_size_abs = hedge_ratio * symbol1_size_abs
    
    # 根据信号方向确定正负
    if signal['action'] == 'SHORT_LONG':
        symbol1_size = -symbol1_size_abs  # 做空
        symbol2_size = +symbol2_size_abs  # 做多
    elif signal['action'] == 'LONG_SHORT':
        symbol1_size = +symbol1_size_abs  # 做多
        symbol2_size = -symbol2_size_abs  # 做空
    
    # 计算实际资金占用
    total_capital_used = abs(symbol1_size) * price1 + abs(symbol2_size) * price2
    
    return symbol1_size, symbol2_size, total_capital_used
```

---

### 6. 平仓条件检查

**代码位置**：第1000-1100行（`check_exit_conditions`方法）

**平仓条件**：

1. **Z-score回归**：
```
|Z-score| ≤ z_exit_threshold
```

2. **止盈**：
```
收益率 ≥ take_profit_pct
```

3. **止损**：
```
收益率 ≤ -stop_loss_pct
```

4. **最大持仓时间**：
```
持仓时间 ≥ max_holding_hours
```

**收益率计算**：
```
收益率 = (当前价差 - 开仓价差) / 开仓价差
```

对于 `SHORT_LONG` 信号（做空价差）：
```
收益率 = (开仓价差 - 当前价差) / 开仓价差
```

对于 `LONG_SHORT` 信号（做多价差）：
```
收益率 = (当前价差 - 开仓价差) / 开仓价差
```

---

### 7. 各策略模型公式

#### 6.1 传统方法

**公式**：
```
Z_t = (spread_t - μ) / σ
```

其中：
- `μ = mean(historical_spreads)`
- `σ = std(historical_spreads)`

---

#### 6.2 ARIMA-GARCH模型

**ARIMA模型**：
```
(1 - φ₁B - ... - φₚBᵖ)(1 - B)ᵈ spread_t = (1 + θ₁B + ... + θₚBᵖ)ε_t
```

**GARCH模型**：
```
σ²_t = ω + Σᵢ αᵢε²_{t-i} + Σⱼ βⱼσ²_{t-j}
```

**Z-score计算**：
```
Z_t = (spread_t - μ_t) / σ_t
```

其中：
- `μ_t`: ARIMA预测的均值
- `σ_t`: GARCH预测的标准差

---

#### 6.3 ECM（误差修正模型）

**协整关系**：
```
price1_t = α + β × price2_t + ε_t
```

**误差修正项**：
```
ECM_t = price1_t - α - β × price2_t
```

**ECM模型**：
```
Δspread_t = γ × ECM_{t-1} + Σᵢ φᵢΔspread_{t-i} + ε_t
```

**Z-score计算**：
```
Z_t = ECM_t / σ_ECM
```

---

#### 6.4 Kalman Filter

**状态方程**：
```
x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
```

**观测方程**：
```
y_t = x_t + v_t,  v_t ~ N(0, R)
```

**Kalman Filter更新**：

1. **预测步骤**：
```
x_{t|t-1} = x_{t-1|t-1}
P_{t|t-1} = P_{t-1|t-1} + Q
```

2. **更新步骤**：
```
K_t = P_{t|t-1} / (P_{t|t-1} + R)
x_{t|t} = x_{t|t-1} + K_t × (y_t - x_{t|t-1})
P_{t|t} = (1 - K_t) × P_{t|t-1}
```

**Z-score计算**：
```
Z_t = (spread_t - x_{t|t}) / σ_t
```

---

#### 6.5 Copula + DCC-GARCH

**DCC-GARCH模型**：

**GARCH(1,1)波动率**：
```
σ²_{i,t} = ω_i + α_i × ε²_{i,t-1} + β_i × σ²_{i,t-1}
```

**标准化残差**：
```
z_{i,t} = ε_{i,t} / σ_{i,t}
```

**动态相关系数**：
```
Q_t = (1 - a - b) × Q̄ + a × z_{t-1}z'_{t-1} + b × Q_{t-1}
R_t = diag(Q_t)^{-1/2} × Q_t × diag(Q_t)^{-1/2}
```

**Copula模型**：
```
C(u₁, u₂) = Φ_2(Φ⁻¹(u₁), Φ⁻¹(u₂); R_t)
```

其中：
- `u_i = F_i(price_i)`: 边际分布
- `Φ_2`: 二元正态分布
- `R_t`: 动态相关系数矩阵

**Z-score计算**：
```
Z_t = (spread_t - μ_t) / σ_t
```

其中 `σ_t` 由DCC-GARCH模型估计。

---

#### 6.6 Regime-Switching模型

**状态转换模型**：
```
spread_t | S_t = k ~ N(μ_k, σ²_k)
```

**状态转换概率**（马尔可夫链）：
```
P(S_t = j | S_{t-1} = i) = p_{ij}
```

**状态转换概率矩阵**：
```
P = [p_{00}  p_{01}]  =  [p_{00}     1-p_{00}]
    [p_{10}  p_{11}]     [1-p_{11}  p_{11}  ]
```

**滤波概率**：
```
P(S_t = k | spread_1, ..., spread_t)
```

**平滑概率**：
```
P(S_t = k | spread_1, ..., spread_T),  T > t
```

**Z-score计算**：
```
Z_t = (spread_t - μ_{S_t}) / σ_{S_t}
```

其中 `S_t` 是当前估计的市场状态。

---

## 代码详解

### 关键类和方法

#### 0. 协整检验辅助函数

**代码位置**：第593-788行

**函数列表**：

##### `calculate_hedge_ratio(price1, price2)`
计算对冲比率（使用OLS回归）

```python
def calculate_hedge_ratio(price1, price2):
    """计算对冲比率（使用OLS回归）"""
    min_length = min(len(price1), len(price2))
    price1_aligned = price1.iloc[:min_length]
    price2_aligned = price2.iloc[:min_length]
    
    X = price2_aligned.values.reshape(-1, 1)
    y = price1_aligned.values
    X_with_const = add_constant(X)
    
    model = OLS(y, X_with_const).fit()
    hedge_ratio = model.params[1]  # 斜率系数
    
    return hedge_ratio
```

##### `advanced_adf_test(series, max_lags=None, verbose=True)`
执行增强的ADF检验，判断序列是否平稳

```python
def advanced_adf_test(series, max_lags=None, verbose=True):
    """执行增强的ADF检验"""
    adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')
    
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    used_lag = adf_result[2]
    
    is_stationary = p_value < 0.05
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'used_lag': used_lag,
        'is_stationary': is_stationary
    }
```

##### `determine_integration_order(series, max_order=2)`
确定序列的积分阶数（I(0), I(1), I(2)）

```python
def determine_integration_order(series, max_order=2):
    """确定序列的积分阶数"""
    # 检验原序列
    adf_result = advanced_adf_test(series, verbose=False)
    if adf_result and adf_result['is_stationary']:
        return 0  # I(0)
    
    # 检验一阶差分
    if max_order >= 1:
        diff1 = series.diff().dropna()
        adf_result = advanced_adf_test(diff1, verbose=False)
        if adf_result and adf_result['is_stationary']:
            return 1  # I(1)
    
    # 检验二阶差分
    if max_order >= 2:
        diff2 = series.diff().diff().dropna()
        adf_result = advanced_adf_test(diff2, verbose=False)
        if adf_result and adf_result['is_stationary']:
            return 2  # I(2)
    
    return None  # 无法确定
```

##### `enhanced_cointegration_test(price1, price2, symbol1, symbol2, verbose=True, diff_order=0)`
完整的协整检验（Engle-Granger方法）

**检验步骤**：
1. 检验price1和price2的积分阶数
2. 检查两个序列是否同阶单整（必须都是I(1)）
3. 根据diff_order计算对冲比率和价差
4. 使用ADF检验价差的平稳性
5. 如果价差平稳，则协整关系成立

**返回值**：
```python
{
    'cointegration_found': bool,  # 是否找到协整关系
    'spread_adf': dict,  # ADF检验结果
    'hedge_ratio': float,  # 对冲比率
    'spread': Series,  # 价差序列
    'p_value': float  # ADF P值
}
```

---

#### 1. `RecursiveLeastSquares` 类

**初始化方法**：
```python
def __init__(self, lambda_forgetting=0.99, initial_covariance=1000.0, max_change_rate=0.2):
    self.lambda_forgetting = lambda_forgetting  # 遗忘因子
    self.initial_covariance = initial_covariance  # 初始协方差
    self.max_change_rate = max_change_rate  # 最大变化率
    self.beta = None  # 对冲比率 [截距, 斜率]
    self.P = None  # 协方差矩阵
    self.initialized = False
```

**初始化RLS**：
```python
def initialize(self, initial_price1, initial_price2):
    """使用OLS估计初始对冲比率"""
    # 使用OLS回归
    X = price2_aligned.values.reshape(-1, 1)
    y = price1_aligned.values
    X_with_const = add_constant(X)
    
    model = OLS(y, X_with_const).fit()
    
    # 初始化参数
    self.beta = np.array([model.params[0], model.params[1]])
    self.P = np.eye(2) * self.initial_covariance
    self.initialized = True
```

---

#### 2. `AdvancedCointegrationTrading` 类

**初始化RLS**：
```python
def initialize_rls_for_pair(self, pair_key, initial_price1, initial_price2):
    """为币对初始化RLS"""
    if not self.use_rls:
        return
    
    rls = RecursiveLeastSquares(
        lambda_forgetting=self.rls_lambda,
        max_change_rate=self.rls_max_change_rate
    )
    rls.initialize(initial_price1, initial_price2)
    self.rls_instances[pair_key] = rls
```

**更新RLS**：
```python
def update_rls_for_pair(self, pair_key, price1_t, price2_t):
    """更新币对的RLS对冲比率"""
    if not self.use_rls or pair_key not in self.rls_instances:
        return None
    
    rls = self.rls_instances[pair_key]
    hedge_ratio = rls.update(price1_t, price2_t)
    
    # 更新价格历史
    if pair_key in self.price_history:
        self.price_history[pair_key]['price1'].append(price1_t)
        self.price_history[pair_key]['price2'].append(price2_t)
        # 保持历史长度不超过lookback_period
        if len(self.price_history[pair_key]['price1']) > self.lookback_period * 2:
            self.price_history[pair_key]['price1'] = self.price_history[pair_key]['price1'][-self.lookback_period:]
            self.price_history[pair_key]['price2'] = self.price_history[pair_key]['price2'][-self.lookback_period:]
    
    return hedge_ratio
```

**紧急平仓**：
```python
def emergency_close_position(self, symbol, side, quantity, reason="紧急平仓"):
    """紧急平仓单个仓位（用于部分成交时平掉已成交的订单）"""
    # 执行平仓订单
    order = self.binance_api.place_order(symbol, side, quantity)
    
    if order and order.get('orderId'):
        # 等待平仓订单成交
        success, final_status, _ = self.wait_for_orders_completion(
            order, None, symbol, None, max_wait=10
        )
        return success
    return False
```

**执行交易（包含部分成交处理）**：
```python
def execute_trade(self, pair_info, current_prices, signal, timestamp, current_spread, available_capital):
    """执行交易（实盘下单）"""
    # 1. 计算开仓数量
    symbol1_size, symbol2_size, total_capital_used = self.calculate_position_size_beta_neutral(...)
    
    # 2. 下单
    order1 = self.binance_api.place_order(symbol1, 'SELL', quantity1)
    order2 = self.binance_api.place_order(symbol2, 'BUY', quantity2)
    
    # 3. 检查下单结果
    if order1 and order2 and order1.get('orderId') and order2.get('orderId'):
        # 两个订单都成功提交，等待成交
        success, final_status1, final_status2 = self.wait_for_orders_completion(...)
        
        if success:
            # 创建持仓记录
            self.positions[pair_info['pair_name']] = position
            self.trades.append(trade)
            return position
        else:
            return None
    elif order1 and order1.get('orderId') and (not order2 or not order2.get('orderId')):
        # 第一个订单成功，第二个订单失败 - 紧急平仓
        close_success = self.emergency_close_position(...)
        return None
    elif order2 and order2.get('orderId') and (not order1 or not order1.get('orderId')):
        # 第二个订单成功，第一个订单失败 - 紧急平仓
        close_success = self.emergency_close_position(...)
        return None
    else:
        # 两个订单都失败
        return None
```

**紧急平仓**：
```python
def emergency_close_position(self, symbol, side, quantity, reason="紧急平仓"):
    """紧急平仓单个仓位（用于部分成交时平掉已成交的订单）"""
    try:
        print(f"  紧急平仓: {symbol} {side} {quantity} - 原因: {reason}")
        
        # 执行平仓订单
        order = self.binance_api.place_order(symbol, side, quantity)
        
        if order and order.get('orderId'):
            print(f"  紧急平仓订单已提交: {symbol} {side} {quantity}")
            
            # 等待平仓订单成交
            success, final_status, _ = self.wait_for_orders_completion(
                order, None, symbol, None, max_wait=10
            )
            
            if success:
                print(f"  ✓ 紧急平仓成功: {symbol}")
                return True
            else:
                print(f"  ✗ 紧急平仓失败: {symbol}")
                return False
        else:
            print(f"  ✗ 紧急平仓订单提交失败: {symbol}")
            return False
    except Exception as e:
        print(f"✗ 紧急平仓异常: {symbol} - {str(e)}")
        return False
```

**定期协整检验**：
```python
def check_cointegration_periodically(self, pair_key, price1_series, price2_series, symbol1, symbol2):
    """
    定期进行协整检验（实盘交易版本）
    
    功能：
    1. 检查是否需要重新检验（根据cointegration_check_interval）
    2. 使用窗口大小（cointegration_window_size）进行协整检验
    3. 根据检验结果更新协整状态和协整比率
    4. 如果协整关系破裂，渐进式降低协整比率
    
    参数：
    - pair_key: 币对标识
    - price1_series: 价格序列1（pandas Series或list）
    - price2_series: 价格序列2（pandas Series或list）
    - symbol1: 币种1名称
    - symbol2: 币种2名称
    
    返回：
    - dict: 协整检验结果，包含is_cointegrated和cointegration_ratio
    """
    # 1. 检查检验间隔
    current_index = self.data_point_count.get(pair_key, 0)
    last_check = status['last_check_index']
    
    if current_index - last_check < self.cointegration_check_interval:
        # 不需要检验，返回当前状态
        return {
            'is_cointegrated': status['is_cointegrated'],
            'cointegration_ratio': status.get('cointegration_ratio', 1.0)
        }
    
    # 2. 确定检验窗口大小
    max_window_size = min(len(price1_series), len(price2_series))
    target_window_size = self.cointegration_window_size  # 用户输入的窗口大小
    
    if max_window_size < target_window_size:
        window_size = max(100, int(max_window_size * 0.8))  # 至少100个数据点
    else:
        window_size = target_window_size
    
    # 3. 获取最近的数据
    recent_price1 = price1_series.iloc[-window_size:]
    recent_price2 = price2_series.iloc[-window_size:]
    
    # 4. 执行协整检验
    coint_result = enhanced_cointegration_test(
        recent_price1, recent_price2, symbol1, symbol2,
        verbose=False, diff_order=self.diff_order
    )
    
    is_cointegrated = coint_result.get('cointegration_found', False)
    
    # 5. 更新协整状态
    status['is_cointegrated'] = is_cointegrated
    status['last_check_index'] = current_index
    
    if is_cointegrated:
        # 协整检验通过
        status['cointegration_ratio'] = 1.0
        status['consecutive_failures'] = 0
    else:
        # 协整检验失败，渐进式降低协整比率
        consecutive_failures = status.get('consecutive_failures', 0) + 1
        status['consecutive_failures'] = consecutive_failures
        
        if consecutive_failures == 1:
            status['cointegration_ratio'] = 0.5
        elif consecutive_failures == 2:
            status['cointegration_ratio'] = 0.2
        else:
            status['cointegration_ratio'] = 0.0  # 完全暂停交易
    
    return {
        'is_cointegrated': is_cointegrated,
        'cointegration_ratio': status.get('cointegration_ratio', 0.0),
        'coint_result': coint_result
    }
```

---

#### 3. `RealTimeDataManager` 类

**收集预热数据**：
```python
def collect_warmup_data(self, symbols, interval='1h', warmup_period=100):
    """收集预热数据"""
    warmup_data = {}
    
    for symbol in symbols:
        klines = self.binance_api.get_klines(
            symbol=symbol,
            interval=interval,
            limit=warmup_period
        )
        
        prices = pd.Series([float(k[4]) for k in klines])
        warmup_data[symbol] = prices
    
    return warmup_data
```

**实时数据收集**：
```python
def start_data_collection(self, symbols, interval='1h'):
    """启动实时数据收集线程"""
    self.collection_thread = threading.Thread(
        target=self._data_collection_loop,
        args=(symbols, interval),
        daemon=True
    )
    self.collection_thread.start()
```

---

## 总结

本系统实现了完整的协整交易流程：

1. **策略选择**：支持6种Z-score计算策略
2. **数据管理**：自动收集和管理实时价格数据
3. **RLS动态对冲**：使用递归最小二乘法动态更新对冲比率
4. **定期协整检验**：在使用RLS时，定期检验协整关系是否仍然存在，防止协整关系破裂导致的风险
5. **交易执行**：基于Z-score生成交易信号并执行交易
6. **风险控制**：止盈、止损、最大持仓时间等风险控制机制
7. **实时监控**：Web界面实时监控交易状态

**定期协整检验的重要性**：

在使用RLS动态更新对冲比率时，系统假设协整关系仍然存在。但如果协整关系破裂，继续使用RLS更新对冲比率就没有意义了。因此，系统会：

1. **定期检验**：每隔N个数据点（N=协整检验窗口大小）进行一次协整检验
2. **渐进式暂停**：如果协整检验失败，根据连续失败次数渐进式降低协整比率，最终暂停交易
3. **自动恢复**：如果协整关系恢复，系统会自动恢复正常交易

这样可以在协整关系破裂时及时停止交易，避免不必要的损失。

**部分成交风险控制**：

配对交易的核心是对冲风险，如果只有一个币种成交，就会形成单边持仓，暴露方向性风险。系统实现了自动紧急平仓机制：

1. **检测部分成交**：实时检测订单状态，发现部分成交立即处理
2. **自动平仓**：自动平掉已成交的订单，避免单边持仓
3. **风险提示**：如果紧急平仓失败，会提示手动处理

系统设计灵活，可根据市场情况选择最适合的策略和参数。




