# 多周期均线缠绕突破策略详细说明文档

## 策略概述

**策略名称**：FinalMultiplePeriodStrategy（多周期均线缠绕突破策略）

**策略类型**：技术指标趋势类策略

**核心思想**：
- 使用日线级别判断大趋势方向（顺势交易）
- 使用小时线级别检测均线缠绕状态，寻找突破机会
- 结合成交量确认和K线形态过滤，提高入场质量
- 使用止盈和趋势跌破双重退出机制
- 支持反向开仓：有持仓时如果出现反向信号，会先平仓再反向开仓

---

## 一、策略初始化阶段（__init__）

### 1.1 参数设置

策略支持以下可配置参数：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `ema_lens` | [5, 10, 20, 30] | 小时线EMA周期列表，用于检测缠绕 |
| `ma_len_daily` | 25 | 日线趋势EMA周期 |
| `tp_pct` | 3.0 | 止盈百分比（3%） |
| `vol_factor` | 1.2 | 成交量放大因子（成交量需大于均量的1.2倍） |
| `watch_bars` | 5 | 观察窗口K线数（平仓后观察5根K线） |

### 1.2 状态变量初始化

```python
# 策略状态变量
self.exit_bar_long = None      # 记录平多时的bar索引
self.exit_bar_short = None      # 记录平空时的bar索引
self.watch_short = False        # 是否在观察开空机会
self.watch_long = False         # 是否在观察开多机会
```

这些变量用于跟踪策略状态，特别是观察窗口机制。

---

## 二、指标计算阶段（initialize）

在回测开始前，策略会一次性计算所有需要的技术指标。

### 2.1 小时线EMA计算

计算4条EMA均线：
- **EMA1**：5周期EMA（最快）
- **EMA2**：10周期EMA
- **EMA3**：20周期EMA
- **EMA4**：30周期EMA（最慢）

```python
self.ema1 = calculate_ema(data['close'], 5)
self.ema2 = calculate_ema(data['close'], 10)
self.ema3 = calculate_ema(data['close'], 20)
self.ema4 = calculate_ema(data['close'], 30)
```

### 2.2 EMA缠绕检测指标

计算4条EMA的最大值、最小值和范围：
- **ema_max**：4条EMA中的最大值
- **ema_min**：4条EMA中的最小值
- **ema_range**：ema_max - ema_min（缠绕程度）

当`ema_range`很小时，说明4条EMA缠绕在一起，市场处于震荡状态。

### 2.3 布林带计算

用于判断缠绕阈值和观察窗口开仓条件：
- **basis**：10周期SMA（中轨）
- **dev**：10周期标准差
- **boll_upper**：上轨 = basis + 2*dev
- **boll_lower**：下轨 = basis - 2*dev
- **boll_width**：布林带宽度 = boll_upper - boll_lower

### 2.4 成交量均线

计算13周期成交量SMA，用于判断成交量是否放大：
```python
self.vol_ma = calculate_sma(data['volume'], 13)
```

### 2.5 日线数据生成（关键步骤）

**重要**：策略需要日线数据来判断大趋势，但输入数据可能是小时线。

**处理方式**：使用`resample('D')`将小时线数据重采样为日线数据

```python
daily_data = data.resample('D').agg({
    'open': 'first',    # 取当天第一根K线的开盘价
    'high': 'max',      # 取当天最高价
    'low': 'min',       # 取当天最低价
    'close': 'last',    # 取当天最后一根K线的收盘价
    'volume': 'sum'     # 当天成交量总和
})
```

**示例**：
- 输入：24根小时线K线（1天的数据）
- 输出：1根日线K线

### 2.6 日线EMA计算

计算日线25周期EMA：
```python
daily_ema = calculate_ema(daily_data['close'], 25)
```

### 2.7 日线数据对齐到小时线

由于日线数据只有每天一根K线，而小时线数据有24根，需要将日线数据对齐到小时线：

```python
daily_ema_aligned = daily_ema.reindex(data.index, method='ffill')
daily_close_aligned = daily_data['close'].reindex(data.index, method='ffill')
```

**前向填充（ffill）**：同一天的24根小时线K线，都使用当天的日线数据。

**示例**：
```
2024-01-01 00:00:00  -> 使用 2024-01-01 的日线EMA
2024-01-01 01:00:00  -> 使用 2024-01-01 的日线EMA
...
2024-01-01 23:00:00 -> 使用 2024-01-01 的日线EMA
2024-01-02 00:00:00  -> 使用 2024-01-02 的日线EMA
```

### 2.8 ATR计算（系统需要）

虽然策略本身不使用ATR，但回测系统需要ATR来计算仓位大小，所以也计算了14周期ATR。

---

## 三、每个K线周期的处理流程

回测系统会遍历每一根K线，按以下顺序调用策略方法：

### 流程概览

```
每个K线周期开始
    ↓
1. check_stop_loss() - 检查止盈
    ↓ (如果有持仓)
2. generate_signals() - 生成交易信号
    ↓ (如果有持仓，检查退出；如果无持仓，检查入场)
3. check_add_position() - 检查加仓（本策略不支持）
    ↓
下一个K线周期
```

---

## 四、止盈检查（check_stop_loss）

**调用时机**：每个K线周期，系统首先调用此方法检查止盈

**处理逻辑**：

### 4.1 多头止盈

```python
if position_size > 0:  # 有多头持仓
    profit_pct = (当前收盘价 - 入场价格) / 入场价格 * 100
    if profit_pct >= 3.0:  # 盈利达到3%
        return {
            'signal': 'close_long',
            'price': 当前收盘价,
            'reason': '多头盈利止盈'
        }
```

**示例**：
- 入场价格：100
- 当前价格：103
- 盈利百分比：(103-100)/100*100 = 3%
- **触发止盈**

### 4.2 空头止盈

```python
if position_size < 0:  # 有空头持仓
    profit_pct = (入场价格 - 当前收盘价) / 入场价格 * 100
    if profit_pct >= 3.0:  # 盈利达到3%
        return {
            'signal': 'close_short',
            'price': 当前收盘价,
            'reason': '空头盈利止盈'
        }
```

**示例**：
- 入场价格：100
- 当前价格：97
- 盈利百分比：(100-97)/100*100 = 3%
- **触发止盈**

### 4.3 状态更新

当触发止盈时，会记录平仓时的bar索引：
- 多头止盈：`self.exit_bar_long = current_idx`
- 空头止盈：`self.exit_bar_short = current_idx`

**注意**：止盈后不会进入观察窗口（代码中注释掉了`watch_short`和`watch_long`的设置）。

---

## 五、信号生成（generate_signals）

**调用时机**：每个K线周期，在止盈检查之后

**处理流程**：

### 5.1 数据验证

```python
# 检查数据是否足够
min_period = max([5, 10, 20, 30]) + 25 = 55
if current_idx < 55:
    return {'signal': 'hold', 'reason': '数据不足'}
```

前55根K线数据不足，无法计算所有指标。

### 5.2 获取当前指标值

从预计算的指标序列中获取当前K线的值：
- `daily_close`：当前K线对应的日线收盘价
- `daily_ema`：当前K线对应的日线25周期EMA
- `ema1, ema2, ema3, ema4`：当前K线的4条EMA值
- `ema_max, ema_min, ema_range`：当前K线的EMA缠绕指标
- `boll_upper, boll_lower, boll_width`：当前K线的布林带指标
- `vol_ma`：当前K线的成交量均线

### 5.3 判断日线趋势

```python
is_long_trend = daily_close > daily_ema   # 日线收盘价 > 日线EMA，趋势向上
is_short_trend = daily_close < daily_ema  # 日线收盘价 < 日线EMA，趋势向下
```

**核心逻辑**：只做顺势交易，不做逆势交易。

### 5.4 检测缠绕状态

```python
tight_threshold = boll_width * 0.8  # 缠绕阈值 = 布林带宽度的80%
is_tight = ema_range < tight_threshold  # EMA范围 < 阈值，说明缠绕
```

**逻辑说明**：
- 当4条EMA缠绕在一起时，`ema_range`很小
- 如果`ema_range < boll_width * 0.8`，说明市场处于震荡状态
- 震荡后的突破往往更有力度

### 5.5 检测突破

```python
break_up = (当前收盘价 > ema_max) and is_tight   # 向上突破缠绕
break_down = (当前收盘价 < ema_min) and is_tight  # 向下突破缠绕
```

**条件**：
1. 价格突破EMA的最大值（向上）或最小值（向下）
2. 且处于缠绕状态（`is_tight = True`）

### 5.6 检测成交量放大

```python
vol_up = 当前成交量 > vol_ma * 1.2
```

成交量需要大于13周期均量的1.2倍，确保突破有成交量支撑。

### 5.7 检测趋势跌破（退出信号）

```python
exit_trend_broken = ema1 < ema2      # 5日EMA < 10日EMA，多头趋势跌破
exit_trend_broken_short = ema1 > ema2  # 5日EMA > 10日EMA，空头趋势升破
```

**逻辑**：
- 当快线（EMA1）跌破慢线（EMA2）时，说明短期趋势转弱
- 多头持仓时，如果`ema1 < ema2`，应该退出
- 空头持仓时，如果`ema1 > ema2`，应该退出

---

## 六、持仓状态处理

### 6.1 有多头持仓（position_size > 0）

#### 6.1.1 检查趋势跌破退出

```python
if exit_trend_broken:  # ema1 < ema2
    self.exit_bar_long = current_idx
    self.watch_short = True  # 进入观察开空窗口
    return {
        'signal': 'close_long',
        'reason': '多头趋势跌破出场'
    }
```

**状态更新**：
- 记录平多时的bar索引：`self.exit_bar_long = current_idx`
- 开启观察开空窗口：`self.watch_short = True`

#### 6.1.2 检查反向开仓（空头信号）

**重要功能**：如果有多头持仓但遇到空头入场信号，会先平多再开空。

```python
if short_condition:  # 满足空头入场条件
    return {
        'signal': 'close_long',
        'price': current_bar['close'],
        'reason': '反向开空：平多开空',
        'reverse_signal': 'short',  # 标记需要反向开仓
        'reverse_price': current_bar['close']
    }
```

**逻辑说明**：
- 当有多头持仓时，如果同时满足空头入场条件（日线趋势向下 + 小时线突破缠绕 + 成交量放大 + 阴线）
- 策略会返回平多信号，并标记需要反向开空
- 回测系统会先平多仓，然后立即开空仓
- 这确保了策略能够及时捕捉反向机会，与TradingView的PineScript逻辑一致

**示例场景**：
```
K线1: 持有多头，入场价格=100
K线2: 日线趋势转为向下，小时线出现空头信号
  → 先平多仓（价格=98）
  → 立即开空仓（价格=98）
```

### 6.2 有空头持仓（position_size < 0）

#### 6.2.1 检查趋势升破退出

```python
if exit_trend_broken_short:  # ema1 > ema2
    self.exit_bar_short = current_idx
    self.watch_long = True  # 进入观察开多窗口
    return {
        'signal': 'close_short',
        'reason': '空头趋势升破出场'
    }
```

**状态更新**：
- 记录平空时的bar索引：`self.exit_bar_short = current_idx`
- 开启观察开多窗口：`self.watch_long = True`

#### 6.2.2 检查反向开仓（多头信号）

**重要功能**：如果有空头持仓但遇到多头入场信号，会先平空再开多。

```python
if long_condition:  # 满足多头入场条件
    return {
        'signal': 'close_short',
        'price': current_bar['close'],
        'reason': '反向开多：平空开多',
        'reverse_signal': 'long',  # 标记需要反向开仓
        'reverse_price': current_bar['close']
    }
```

**逻辑说明**：
- 当有空头持仓时，如果同时满足多头入场条件（日线趋势向上 + 小时线突破缠绕 + 成交量放大 + 阳线）
- 策略会返回平空信号，并标记需要反向开多
- 回测系统会先平空仓，然后立即开多仓
- 这确保了策略能够及时捕捉反向机会

**示例场景**：
```
K线1: 持有空头，入场价格=100
K线2: 日线趋势转为向上，小时线出现多头信号
  → 先平空仓（价格=102）
  → 立即开多仓（价格=102）
```

### 6.3 无持仓（position_size == 0）

#### 6.3.1 检查观察窗口开仓

**平多后观察开空**：
```python
if self.watch_short and 当前是阴线:
    if 最低价 < 布林下轨:
        self.watch_short = False
        return {
            'signal': 'short',
            'reason': '观察窗口开空'
        }
    elif 超过5根K线:
        self.watch_short = False  # 取消观察
```

**条件**：
1. 处于观察开空状态（`watch_short = True`）
2. 当前K线是阴线（`close < open`）
3. 最低价跌破布林下轨（`low < boll_lower`）
4. 在5根K线内（`current_idx - exit_bar_long < 5`）

**平空后观察开多**：
```python
if self.watch_long and 当前是阳线:
    if 最高价 > 布林上轨:
        self.watch_long = False
        return {
            'signal': 'long',
            'reason': '观察窗口开多'
        }
    elif 超过5根K线:
        self.watch_long = False  # 取消观察
```

**条件**：
1. 处于观察开多状态（`watch_long = True`）
2. 当前K线是阳线（`close > open`）
3. 最高价突破布林上轨（`high > boll_upper`）
4. 在5根K线内（`current_idx - exit_bar_short < 5`）

#### 6.3.2 正常入场信号

**多头入场条件**（全部满足）：
```python
long_condition = (
    is_long_trend and      # 1. 日线趋势向上
    break_up and           # 2. 小时线向上突破缠绕
    vol_up and             # 3. 成交量放大
    close > open           # 4. 当前是阳线
)
```

**空头入场条件**（全部满足）：
```python
short_condition = (
    is_short_trend and     # 1. 日线趋势向下
    break_down and         # 2. 小时线向下突破缠绕
    vol_up and             # 3. 成交量放大
    close < open           # 4. 当前是阴线
)
```

**入场信号返回**：
```python
if long_condition:
    return {
        'signal': 'long',
        'price': 当前收盘价,
        'reason': '多头入场（日线趋势向上+小时线突破缠绕+成交量放大+阳线）'
    }

if short_condition:
    return {
        'signal': 'short',
        'price': 当前收盘价,
        'reason': '空头入场（日线趋势向下+小时线突破缠绕+成交量放大+阴线）'
    }
```

---

## 七、加仓检查（check_add_position）

**本策略不支持加仓**，此方法始终返回`None`。

```python
def check_add_position(...):
    return None  # 不支持加仓
```

---

## 八、交易结果更新（update_trade_result）

**调用时机**：每次平仓后，系统会调用此方法

**当前实现**：空实现，不做任何处理

```python
def update_trade_result(self, profit: float):
    pass
```

---

## 九、完整交易流程示例

### 场景1：正常多头交易流程

```
K线1: 无持仓
  - 日线趋势：向上 ✓
  - 缠绕状态：是 ✓
  - 向上突破：是 ✓
  - 成交量放大：是 ✓
  - 阳线：是 ✓
  → 开多仓，入场价格=100

K线2-10: 持有多头
  - 检查止盈：盈利未达3%，继续持有
  - 检查趋势：ema1 > ema2，趋势正常，继续持有
  - 检查反向开仓：无空头信号，继续持有

K线11: 持有多头
  - 检查止盈：盈利达到3%
  → 止盈平仓，价格=103

K线12-16: 无持仓
  - 无入场信号，等待
```

### 场景1.5：反向开仓流程（平多开空）

```
K线1: 开多仓，入场价格=100

K线2-20: 持有多头
  - 盈利未达3%，继续持有

K线21: 持有多头
  - 日线趋势：转为向下 ✓
  - 缠绕状态：是 ✓
  - 向下突破：是 ✓
  - 成交量放大：是 ✓
  - 阴线：是 ✓
  → 满足空头入场条件
  → 先平多仓，价格=98
  → 立即开空仓，价格=98

K线22-30: 持有空头
  - 检查止盈和趋势
```

### 场景2：趋势跌破退出流程

```
K线1: 开多仓，入场价格=100

K线2-20: 持有多头
  - 盈利未达3%，继续持有

K线21: 持有多头
  - 检查趋势：ema1 < ema2，趋势跌破
  → 平多仓，价格=98
  → 开启观察开空窗口（watch_short = True）

K线22: 无持仓，观察开空
  - 当前是阴线 ✓
  - 最低价 < 布林下轨 ✓
  → 开空仓，入场价格=97

K线23-30: 持有空头
  - 检查止盈和趋势
```

### 场景2.5：反向开仓流程（平空开多）

```
K线1: 开空仓，入场价格=100

K线2-20: 持有空头
  - 盈利未达3%，继续持有

K线21: 持有空头
  - 日线趋势：转为向上 ✓
  - 缠绕状态：是 ✓
  - 向上突破：是 ✓
  - 成交量放大：是 ✓
  - 阳线：是 ✓
  → 满足多头入场条件
  → 先平空仓，价格=102
  → 立即开多仓，价格=102

K线22-30: 持有多头
  - 检查止盈和趋势
```

### 场景3：观察窗口超时

```
K线1: 平多仓，开启观察开空窗口

K线2-5: 观察中
  - 未满足开空条件（未跌破布林下轨）

K线6: 超过5根K线
  → 取消观察（watch_short = False）
  → 恢复正常入场逻辑
```

---

## 十、策略特点总结

### 10.1 优势

1. **多周期确认**：日线判断趋势，小时线寻找入场点，提高胜率
2. **缠绕突破**：只在震荡后的突破时入场，避免假突破
3. **成交量确认**：要求成交量放大，确保突破有效
4. **K线形态过滤**：阳线做多，阴线做空，符合市场心理
5. **双重退出机制**：止盈保护利润，趋势跌破及时止损
6. **观察窗口**：平仓后观察反向机会，提高资金利用率
7. **反向开仓机制**：有持仓时如果出现反向信号，会先平仓再反向开仓，及时捕捉趋势转换

### 10.2 注意事项

1. **数据周期要求**：输入数据最好是小时线，策略会自动转换为日线
2. **参数敏感性**：缠绕阈值、止盈百分比等参数需要根据市场调整
3. **震荡市场**：在长期震荡市场中，可能频繁开平仓
4. **观察窗口**：观察窗口机制可能错过一些机会，但也能避免反向开仓的风险

### 10.3 适用市场

- **趋势市场**：策略表现较好，能抓住趋势机会
- **震荡市场**：可能频繁交易，需要优化参数
- **波动较大的市场**：止盈机制能保护利润

---

## 十一、参数优化建议

### 11.1 EMA周期调整

- **快周期（5, 10）**：调整可以改变趋势跌破的敏感度
- **慢周期（20, 30）**：调整可以改变缠绕检测的周期

### 11.2 止盈百分比

- **3%**：适合波动较大的市场
- **2%**：适合波动较小的市场，更早止盈
- **5%**：适合强趋势市场，让利润奔跑

### 11.3 成交量因子

- **1.2**：默认值，要求成交量放大20%
- **1.5**：更严格，要求成交量放大50%
- **1.0**：不要求成交量放大（可能产生更多假信号）

### 11.4 观察窗口

- **5根K线**：默认值
- **3根K线**：更短，可能错过机会
- **10根K线**：更长，可能增加反向开仓风险

---

## 十二、与PineScript原策略的对应关系

| PineScript代码 | Python实现 | 说明 |
|---------------|-----------|------|
| `request.security(..., "D", ...)` | `data.resample('D')` | 获取日线数据 |
| `ta.ema(close, 5/10/20/30)` | `calculate_ema(..., 5/10/20/30)` | 计算EMA |
| `ta.sma(close, 10)` | `calculate_sma(..., 10)` | 计算SMA（布林带中轨） |
| `ta.stdev(close, 10)` | `data['close'].rolling(10).std()` | 计算标准差 |
| `isTight = ema_range < tight_threshold` | `is_tight = ema_range < boll_width * 0.8` | 缠绕检测 |
| `strategy.position_size > 0` | `position_size > 0` | 多头持仓判断 |
| `strategy.opentrades.entry_price(0)` | `last_entry_price`（系统传入） | 入场价格 |
| `bar_index - exitBarLong >= 5` | `current_idx - exit_bar_long >= 5` | 观察窗口超时 |
| `if (shortCondition and strategy.position_size >=0) { if (strategy.position_size > 0) strategy.close("Long"); strategy.entry("Short", ...) }` | 返回`{'signal': 'close_long', 'reverse_signal': 'short', ...}` | 反向开仓机制（平多开空） |
| `if (longCondition and strategy.position_size <=0) { if (strategy.position_size < 0) strategy.close("Short"); strategy.entry("Long", ...) }` | 返回`{'signal': 'close_short', 'reverse_signal': 'long', ...}` | 反向开仓机制（平空开多） |

---

## 十三、使用示例

### 13.1 创建策略实例

```python
from strategies.final_multiple_period_strategy import FinalMultiplePeriodStrategy

# 使用默认参数
strategy = FinalMultiplePeriodStrategy({})

# 自定义参数
params = {
    'ema_lens': [5, 10, 20, 30],
    'ma_len_daily': 25,
    'tp_pct': 3.0,
    'vol_factor': 1.2,
    'watch_bars': 5
}
strategy = FinalMultiplePeriodStrategy(params)
```

### 13.2 在回测系统中使用

```python
from backtest_system_ML import BacktestSystem
from strategies.final_multiple_period_strategy import FinalMultiplePeriodStrategy

# 创建策略
strategy = FinalMultiplePeriodStrategy({})

# 创建回测系统
backtest = BacktestSystem(
    strategy=strategy,
    initial_capital=10000
)

# 加载数据（建议使用小时线数据）
backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=2000)

# 运行回测
backtest.run_backtest(max_entries=3)

# 生成报告
backtest.print_report()
```

---

## 十四、常见问题

### Q1: 为什么需要日线数据？

A: 日线数据用于判断大趋势方向，确保只做顺势交易，提高胜率。

### Q2: 如果输入数据是日线怎么办？

A: 策略会尝试重采样，但可能不准确。建议使用小时线数据，让策略自动转换为日线。

### Q3: 观察窗口机制的作用是什么？

A: 当趋势跌破退出后，市场可能快速反转，观察窗口可以在5根K线内捕捉反向开仓机会。

### Q4: 为什么需要成交量放大？

A: 成交量放大可以确认突破的有效性，避免假突破。

### Q5: 策略支持加仓吗？

A: 不支持。策略设计为单次开仓，通过止盈和趋势跌破退出。

### Q6: 什么是反向开仓机制？

A: 当有持仓时，如果出现反向入场信号（例如持有多头但出现空头信号），策略会先平掉当前持仓，然后立即开反向仓位。这确保了策略能够及时捕捉趋势转换，与TradingView的PineScript逻辑一致。

---

## 十五、版本信息

- **策略文件**：`strategies/final_multiple_period_strategy.py`
- **创建日期**：2024年
- **基于**：TradingView PineScript策略
- **兼容系统**：`backtest_system_ML.py`

---

**文档结束**

