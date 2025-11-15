# ARIMA-GARCH模型详细说明文档

## 目录
1. [模型概述](#模型概述)
2. [ARIMA模型原理](#arima模型原理)
3. [GARCH模型原理](#garch模型原理)
4. [ARIMA-GARCH组合模型](#arima-garch组合模型)
5. [详细实现步骤](#详细实现步骤)
6. [代码详解](#代码详解)
7. [模型参数说明](#模型参数说明)
8. [应用场景与优势](#应用场景与优势)

---

## 模型概述

ARIMA-GARCH模型是时间序列分析中的经典组合模型，用于同时建模序列的**均值**和**波动率**。

### 核心思想

- **ARIMA模型**：捕捉价差序列的**均值动态**（趋势、周期性等）
- **GARCH模型**：捕捉价差序列的**波动率聚集效应**（volatility clustering）

### 在Z-score计算中的应用

传统Z-score计算：
```
Z = (X - μ) / σ
```
其中 μ 和 σ 是历史数据的均值和标准差（固定值）

ARIMA-GARCH Z-score计算：
```
Z = (X_{t+1} - μ_pred) / σ_pred
```
其中 μ_pred 和 σ_pred 是**动态预测**的均值和波动率

---

## ARIMA模型原理

### 什么是ARIMA模型？

**ARIMA** = **A**uto**R**egressive **I**ntegrated **M**oving **A**verage（自回归积分滑动平均模型）

### ARIMA模型结构

ARIMA(p, d, q) 模型由三部分组成：

#### 1. AR(p) - 自回归部分

**数学表达式**：
```
φ(B) * X_t = ε_t
```

其中：
- X_t：时间序列在时刻t的值
- φ(B) = 1 - φ₁*B - φ₂*B² - ... - φₚ*Bᵖ：自回归多项式
- B：滞后算子（B*X_t = X_{t-1}）
- ε_t：白噪声误差项

**展开形式**：
```
X_t = φ₁*X_{t-1} + φ₂*X_{t-2} + ... + φₚ*X_{t-p} + ε_t
```

**含义**：当前值依赖于前p个历史值的线性组合

#### 2. I(d) - 积分部分（差分）

**数学表达式**：
```
(1-B)^d * X_t = Y_t
```

其中：
- d：差分阶数
- Y_t：差分后的平稳序列

**常见情况**：
- d=0：不差分，直接使用原序列
- d=1：一阶差分，Y_t = X_t - X_{t-1}
- d=2：二阶差分，Y_t = (X_t - X_{t-1}) - (X_{t-1} - X_{t-2})

**含义**：通过差分使非平稳序列变为平稳序列

#### 3. MA(q) - 滑动平均部分

**数学表达式**：
```
X_t = θ(B) * ε_t
```

其中：
- θ(B) = 1 + θ₁*B + θ₂*B² + ... + θ_q*B^q：滑动平均多项式

**展开形式**：
```
X_t = ε_t + θ₁*ε_{t-1} + θ₂*ε_{t-2} + ... + θ_q*ε_{t-q}
```

**含义**：当前值是当前和前q个误差项的线性组合

### 完整的ARIMA(p, d, q)模型

**数学表达式**：
```
φ(B) * (1-B)^d * X_t = θ(B) * ε_t
```

**展开形式**（以ARIMA(1,0,1)为例）：
```
X_t = φ₁*X_{t-1} + ε_t + θ₁*ε_{t-1}
```

### 在价差预测中的应用

```python:strategies/arima_garch_zscore_strategy.py
# 步骤1: 拟合ARIMA模型
arima_model = ARIMA(spreads_array, order=self.arima_order)
arima_fitted = arima_model.fit()
```

**作用**：
- 对历史价差序列建模
- 捕捉价差的时间序列特征（趋势、周期性、自相关性）
- 预测下一期的价差均值

---

## GARCH模型原理

### 什么是GARCH模型？

**GARCH** = **G**eneralized **A**uto**R**egressive **C**onditional **H**eteroskedasticity（广义自回归条件异方差模型）

### 为什么需要GARCH模型？

金融时间序列的波动率具有**聚集效应**（volatility clustering）：
- 高波动率时期往往连续出现
- 低波动率时期也往往连续出现
- 波动率本身是时变的，不是常数

### GARCH模型结构

GARCH(p, q) 模型：

**条件方差方程**：
```
σ²_t = ω + α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + αₚ*ε²_{t-p} + β₁*σ²_{t-1} + β₂*σ²_{t-2} + ... + β_q*σ²_{t-q}
```

其中：
- σ²_t：时刻t的条件方差（波动率的平方）
- ω > 0：常数项
- α_i ≥ 0：ARCH项系数（残差平方的滞后项）
- β_j ≥ 0：GARCH项系数（条件方差的滞后项）
- ε_t：残差项

**约束条件**：
```
α₁ + α₂ + ... + αₚ + β₁ + β₂ + ... + β_q < 1
```
（确保模型平稳）

### GARCH(1,1)模型（最常用）

**数学表达式**：
```
σ²_t = ω + α₁*ε²_{t-1} + β₁*σ²_{t-1}
```

**含义**：
- 当前波动率依赖于：
  - 上一期的残差平方（ε²_{t-1}）：捕捉"冲击"的影响
  - 上一期的波动率（σ²_{t-1}）：捕捉波动率的持续性

### 在价差波动率预测中的应用

```python:strategies/arima_garch_zscore_strategy.py
# 步骤2: 获取ARIMA残差
arima_residuals = arima_fitted.resid

# 步骤3: 拟合GARCH模型
garch_model = arch_model(arima_residuals, vol='Garch', 
                        p=self.garch_order[0], q=self.garch_order[1])
garch_fitted = garch_model.fit(disp='off')
```

**作用**：
- 对ARIMA模型的残差建模
- 捕捉残差的波动率聚集效应
- 预测下一期的波动率

---

## ARIMA-GARCH组合模型

### 组合模型的结构

**完整模型**：

**均值方程**：
```
X_t = μ_t + ε_t
```

**误差分解**：
```
ε_t = σ_t * z_t
```

**方差方程**：
```
σ²_t = ω + α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + αₚ*ε²_{t-p} + β₁*σ²_{t-1} + β₂*σ²_{t-2} + ... + β_q*σ²_{t-q}
```

其中：
- μ_t：由ARIMA模型预测的均值
- σ_t：由GARCH模型预测的波动率
- z_t：标准化残差（通常假设为标准正态分布）

### 建模流程

```
历史价差序列 X_t
    ↓
[步骤1] ARIMA模型 → 预测均值 μ_pred
    ↓
[步骤2] 提取残差 ε_t = X_t - μ_t
    ↓
[步骤3] GARCH模型 → 预测波动率 σ_pred
    ↓
[步骤4] 计算Z-score: Z = (X_{t+1} - μ_pred) / σ_pred
```

---

## 详细实现步骤

### 步骤0：数据验证

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 验证输入数据（ARIMA-GARCH需要更多数据）
min_required_length = max(20, sum(self.arima_order) + 5)
if not self.validate_input(historical_spreads, min_length=min_required_length):
    # 数据不足时返回0，不进行回退
    return 0.0
```

**原理**：
- ARIMA模型需要足够的历史数据来估计参数
- 最小数据量：`max(20, sum(arima_order) + 5)`
- 例如：ARIMA(1,0,1) 需要至少 `max(20, 1+0+1+5) = 20` 个数据点

**为什么需要更多数据？**
- ARIMA模型需要估计多个参数（φ_i, θ_i）
- 需要足够的自由度进行参数估计
- 需要验证模型的稳定性

---

### 步骤1：拟合ARIMA模型

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 转换为numpy数组
spreads_array = np.array(historical_spreads)

# 步骤1: 拟合ARIMA模型
arima_model = ARIMA(spreads_array, order=self.arima_order)
arima_fitted = arima_model.fit()
```

**原理**：

1. **模型选择**：
   - 使用指定的ARIMA阶数 `(p, d, q)`
   - 例如：`(1, 0, 1)` 表示ARIMA(1,0,1)

2. **参数估计**：
   - 使用最大似然估计（MLE）或最小二乘法（OLS）
   - 估计自回归系数 φ_i 和滑动平均系数 θ_i

3. **模型拟合**：
   - 拟合过程会优化参数，使模型最好地描述历史数据
   - 生成拟合模型对象 `arima_fitted`

**数学过程**（以ARIMA(1,0,1)为例）：
```
X_t = φ₁*X_{t-1} + ε_t + θ₁*ε_{t-1}
```

通过历史数据估计 φ₁ 和 θ₁ 的值。

---

### 步骤2：提取ARIMA残差

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 步骤2: 获取ARIMA残差
arima_residuals = arima_fitted.resid

# 确保残差有足够的数据
min_residuals_length = max(10, sum(self.garch_order) + 3)
if len(arima_residuals) < min_residuals_length:
    # 残差数据不足，返回0
    return 0.0
```

**原理**：

1. **残差定义**：
   ```
   ε_t = X_t - X_fitted_t
   ```
   其中 X_fitted_t 是ARIMA模型的拟合值

2. **残差的性质**：
   - 如果ARIMA模型拟合良好，残差应该是**白噪声**
   - 但残差的**波动率**可能不是常数（异方差性）
   - 这正是GARCH模型要解决的问题

3. **为什么需要残差？**
   - GARCH模型对**残差**建模，而不是对原始序列
   - 残差已经去除了均值动态，只保留波动率信息

**示例**：
```
原始价差序列: [1.0, 1.2, 0.9, 1.1, 0.8, ...]
ARIMA拟合值:  [1.0, 1.1, 1.0, 1.0, 0.9, ...]
残差序列:     [0.0, 0.1, -0.1, 0.1, -0.1, ...]  ← 用于GARCH建模
```

---

### 步骤3：拟合GARCH模型

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 步骤3: 拟合GARCH模型
garch_model = arch_model(arima_residuals, vol='Garch', 
                        p=self.garch_order[0], q=self.garch_order[1])
garch_fitted = garch_model.fit(disp='off')
```

**原理**：

1. **模型选择**：
   - 使用指定的GARCH阶数 `(p, q)`
   - 例如：`(1, 1)` 表示GARCH(1,1)

2. **参数估计**：
   - 使用最大似然估计（MLE）
   - 估计参数：ω, α_i, β_j

3. **模型拟合**：
   - 拟合过程会优化参数，使模型最好地描述残差的波动率特征
   - 生成拟合模型对象 `garch_fitted`

**数学过程**（以GARCH(1,1)为例）：
```
σ²_t = ω + α₁*ε²_{t-1} + β₁*σ²_{t-1}
```

通过残差序列估计 ω, α₁, β₁ 的值。

**参数含义**：
- ω：长期平均波动率
- α₁：短期冲击的影响（残差平方的系数）
- β₁：波动率的持续性（条件方差的系数）

---

### 步骤4：预测价差的均值（ARIMA预测）

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 步骤4: 预测当前价差的均值（ARIMA）
arima_forecast = arima_fitted.forecast(steps=1)
# 处理不同的返回格式
if hasattr(arima_forecast, 'iloc'):
    predicted_mean = arima_forecast.iloc[0]
elif isinstance(arima_forecast, (list, np.ndarray)):
    predicted_mean = float(arima_forecast[0])
else:
    predicted_mean = float(arima_forecast)
```

**原理**：

1. **预测方法**：
   - 使用拟合好的ARIMA模型进行一步向前预测
   - `forecast(steps=1)` 表示预测未来1期

2. **预测公式**（以ARIMA(1,0,1)为例）：
   ```
   X_pred_{t+1} = φ₁*X_t + θ₁*ε_est_t
   ```
   其中 ε_est_t 是估计的误差项

3. **预测值**：
   - μ_pred = X_pred_{t+1}：下一期价差的预测均值

**示例**：
```
历史价差: [1.0, 1.2, 0.9, 1.1, 0.8]
ARIMA预测: 0.95  ← 下一期价差的预测均值
```

---

### 步骤5：预测价差的波动率（GARCH预测）

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 步骤5: 预测当前价差的波动率（GARCH）
garch_forecast = garch_fitted.forecast(horizon=1)
predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, 0])
```

**原理**：

1. **预测方法**：
   - 使用拟合好的GARCH模型进行一步向前预测
   - `forecast(horizon=1)` 表示预测未来1期的波动率

2. **预测公式**（以GARCH(1,1)为例）：
   ```
   σ²_pred_{t+1} = ω + α₁*ε²_t + β₁*σ²_t
   ```
   其中：
   - ε_t：当前期的残差（已知）
   - σ²_t：当前期的条件方差（已知）

3. **预测值**：
   - σ_pred_{t+1} = sqrt(σ²_pred_{t+1})：下一期价差的预测波动率

**示例**：
```
当前残差: 0.1
当前波动率: 0.15
GARCH预测: σ_{t+1} = 0.16  ← 下一期价差的预测波动率
```

---

### 步骤6：计算Z-score

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 验证预测结果
if predicted_volatility <= 0 or np.isnan(predicted_volatility) or np.isnan(predicted_mean):
    # 预测结果无效，返回0
    return 0.0

# 步骤6: 计算Z-score
z_score = (current_spread - predicted_mean) / predicted_volatility

return z_score
```

**原理**：

**Z-score公式**：
```
Z = (X_{t+1} - μ_pred) / σ_pred
```

其中：
- X_{t+1}：当前实际价差
- μ_pred：ARIMA预测的均值
- σ_pred：GARCH预测的波动率

**含义**：
- Z-score表示当前价差偏离预测均值的**标准化距离**
- 使用**动态预测**的均值和波动率，而不是历史固定值
- 能够更好地适应市场变化

**示例**：
```
当前价差: 1.2
预测均值: 0.95
预测波动率: 0.16
Z-score = (1.2 - 0.95) / 0.16 = 1.56
```

---

## 代码详解

### 模型缓存机制

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
# 创建缓存键（基于数据长度和最后几个值）
cache_key = (len(spreads_array), tuple(spreads_array[-5:]))

# 检查缓存
if cache_key in self._arima_garch_models:
    arima_fitted, garch_fitted = self._arima_garch_models[cache_key]
else:
    # 拟合新模型...
    # 缓存模型（限制缓存大小）
    if len(self._arima_garch_models) >= self._max_cache_size:
        # 清除最旧的缓存
        oldest_key = next(iter(self._arima_garch_models))
        del self._arima_garch_models[oldest_key]
    
    self._arima_garch_models[cache_key] = (arima_fitted, garch_fitted)
```

**原理**：

1. **缓存键设计**：
   - 基于数据长度和最后5个值
   - 如果数据相同，可以复用已拟合的模型

2. **缓存优势**：
   - 避免重复拟合模型，提高计算效率
   - ARIMA和GARCH模型拟合是计算密集型操作

3. **缓存管理**：
   - 限制缓存大小为10个模型
   - 使用LRU（最近最少使用）策略清除旧缓存

**为什么需要缓存？**
- 在回测过程中，相邻时间点的历史数据可能相同
- 重复拟合相同的模型是浪费的
- 缓存可以显著提高计算速度

---

### 错误处理

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
try:
    # ... 模型拟合和预测 ...
except Exception as e:
    # 任何错误都返回0，不进行回退（策略独立）
    print(f"ARIMA-GARCH模型计算失败: {str(e)}")
    return 0.0
```

**错误情况**：

1. **数据不足**：
   - 历史数据少于最小要求
   - 残差数据不足

2. **模型拟合失败**：
   - ARIMA模型无法收敛
   - GARCH模型无法收敛
   - 参数估计失败

3. **预测失败**：
   - 预测结果无效（NaN或负数）
   - 波动率预测为0或负数

**处理策略**：
- 返回0（表示无法计算Z-score）
- 不进行回退到传统方法（策略独立）
- 打印错误信息便于调试

---

## 模型参数说明

### ARIMA阶数 (p, d, q)

#### p - 自回归项数

**含义**：当前值依赖于前p个历史值

**选择原则**：
- 通常从1开始尝试
- 可以通过ACF（自相关函数）图判断
- 常见值：1, 2, 3

**代码**：
```python
arima_order = (1, 0, 1)  # p=1
```

#### d - 差分阶数

**含义**：使序列平稳所需的差分次数

**选择原则**：
- **d=0**：序列已经是平稳的（本系统使用）
- **d=1**：一阶差分后平稳
- **d=2**：二阶差分后平稳

**重要**：在本系统中，由于价差已经是平稳序列（I(0)），**d应该设为0**

**代码**：
```python
arima_order = (1, 0, 1)  # d=0（价差是平稳序列）
```

#### q - 滑动平均项数

**含义**：当前值依赖于前q个误差项

**选择原则**：
- 通常从1开始尝试
- 可以通过PACF（偏自相关函数）图判断
- 常见值：1, 2

**代码**：
```python
arima_order = (1, 0, 1)  # q=1
```

### GARCH阶数 (p, q)

#### p - ARCH项数

**含义**：条件方差依赖于前p个残差平方

**选择原则**：
- 通常从1开始
- 常见值：1, 2

**代码**：
```python
garch_order = (1, 1)  # p=1
```

#### q - GARCH项数

**含义**：条件方差依赖于前q个条件方差

**选择原则**：
- 通常从1开始
- GARCH(1,1)是最常用的模型
- 常见值：1, 2

**代码**：
```python
garch_order = (1, 1)  # q=1
```

### 默认参数

**代码**：
```python:strategies/arima_garch_zscore_strategy.py
def __init__(self, arima_order: Tuple[int, int, int] = (1, 0, 1), 
             garch_order: Tuple[int, int] = (1, 1), **kwargs):
```

**默认值**：
- ARIMA(1, 0, 1)：简单的ARIMA模型
- GARCH(1, 1)：最常用的GARCH模型

**为什么选择这些默认值？**
- 参数少，计算快
- 通常能捕捉主要的时间序列特征
- 适合大多数金融时间序列

---

## 应用场景与优势

### 适用场景

1. **数据量充足**：
   - 至少需要20个历史数据点
   - 更多数据通常能获得更好的拟合效果

2. **价差序列有明显的时间序列特征**：
   - 存在趋势或周期性
   - 存在自相关性

3. **波动率有明显聚集效应**：
   - 高波动率时期连续出现
   - 低波动率时期连续出现

### 优势

1. **动态预测**：
   - 均值和波动率都是动态预测的
   - 能够适应市场变化

2. **捕捉波动率聚集**：
   - GARCH模型能够捕捉波动率的时变特征
   - 在高波动率时期给出更大的波动率预测

3. **理论上更准确**：
   - 考虑了时间序列的动态特征
   - 考虑了波动率的时变特征

### 劣势

1. **计算复杂度高**：
   - 需要拟合两个模型（ARIMA + GARCH）
   - 计算时间较长

2. **数据要求高**：
   - 需要足够的历史数据
   - 数据不足时无法使用

3. **模型可能失败**：
   - 模型拟合可能不收敛
   - 需要错误处理机制

---

## 完整流程图

```
输入：当前价差 current_spread，历史价差序列 historical_spreads
  ↓
[数据验证]
检查数据量是否足够（≥20个数据点）
  ↓ 否 → 返回 0.0
  ↓ 是
[检查缓存]
基于数据长度和最后5个值创建缓存键
  ↓
缓存中存在？ → 是 → 使用缓存的模型
  ↓ 否
[步骤1：拟合ARIMA模型]
ARIMA(spreads_array, order=(p,d,q))
  ↓
估计参数：φ₁, φ₂, ..., θ₁, θ₂, ...
  ↓
[步骤2：提取残差]
ε_t = X_t - X_fitted_t
  ↓
[步骤3：拟合GARCH模型]
GARCH(arima_residuals, order=(p,q))
  ↓
估计参数：ω, α₁, α₂, ..., β₁, β₂, ...
  ↓
[缓存模型]
保存到 _arima_garch_models 字典
  ↓
[步骤4：预测均值]
μ_pred = ARIMA.forecast(steps=1)
  ↓
[步骤5：预测波动率]
σ_pred = sqrt(GARCH.forecast(horizon=1))
  ↓
[验证预测结果]
检查是否有效（>0, 非NaN）
  ↓ 无效 → 返回 0.0
  ↓ 有效
[步骤6：计算Z-score]
Z = (current_spread - μ_pred) / σ_pred
  ↓
返回 Z-score
```

---

## 数学公式总结

### ARIMA(p, d, q)模型

```
φ(B) * (1-B)^d * X_t = θ(B) * ε_t
```

**展开形式**（ARIMA(1,0,1)）：
```
X_t = φ₁*X_{t-1} + ε_t + θ₁*ε_{t-1}
```

### GARCH(p, q)模型

```
σ²_t = ω + α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + αₚ*ε²_{t-p} + β₁*σ²_{t-1} + β₂*σ²_{t-2} + ... + β_q*σ²_{t-q}
```

**展开形式**（GARCH(1,1)）：
```
σ²_t = ω + α₁*ε²_{t-1} + β₁*σ²_{t-1}
```

### Z-score计算

```
Z = (X_{t+1} - μ_pred) / σ_pred
```

其中：
- μ_pred：ARIMA模型预测的均值
- σ_pred：GARCH模型预测的波动率

---

## 代码文件位置

所有相关代码位于：
- `strategies/arima_garch_zscore_strategy.py`：ARIMA-GARCH策略实现
- `strategies/base_zscore_strategy.py`：策略基类

---

**文档版本**：v1.0  
**最后更新**：2024年  
**作者**：量化交易系统开发团队

