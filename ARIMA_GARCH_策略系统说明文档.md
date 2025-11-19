# ARIMA-GARCH协整交易策略系统说明文档

## 目录
1. [系统概述](#系统概述)
2. [系统架构](#系统架构)
3. [策略模式设计](#策略模式设计)
4. [策略模型详解](#策略模型详解)
5. [核心代码解析](#核心代码解析)
6. [使用流程](#使用流程)
7. [技术细节](#技术细节)

---

## 系统概述

本系统是一个基于协整理论的量化交易策略系统，支持多种Z-score计算方法。系统采用**策略模式（Strategy Pattern）**设计，允许用户灵活选择不同的Z-score计算策略进行回测和交易。

### 核心功能
- **协整检验**：使用Engle-Granger两阶段方法检验币对间的协整关系
- **滚动窗口分析**：通过滚动窗口识别协整关系的时变特性
- **多策略支持**：支持传统方法和ARIMA-GARCH模型两种Z-score计算策略
- **策略循环选择**：回测结束后可重新选择策略进行测试
- **参数优化**：支持网格搜索、随机搜索、贝叶斯优化等多种优化方法（优化器也支持策略对象模式）

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    主程序入口                                │
│          cointegration_test_windows_optimization_arima_garch.py │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼──────────┐
│  协整检验模块   │          │   交易策略模块      │
│                │          │                    │
│ - 滚动窗口检验  │          │ - 信号生成         │
│ - 对冲比率计算  │          │ - 仓位管理         │
│ - ADF检验      │          │ - 风险控制         │
└────────────────┘          └─────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │   Z-score策略模块    │
                           │   strategies/       │
                           │                     │
                           │ - BaseZScoreStrategy│
                           │ - Traditional      │
                           │ - ArimaGarch       │
                           └─────────────────────┘
```

### 目录结构

```
cryptoStudy/
├── cointegration_test_windows_optimization_arima_garch.py  # 主程序
├── strategies/                                              # 策略模块
│   ├── __init__.py
│   ├── base_zscore_strategy.py          # 策略基类
│   ├── traditional_zscore_strategy.py   # 传统策略
│   └── arima_garch_zscore_strategy.py   # ARIMA-GARCH策略
└── ARIMA_GARCH_策略系统说明文档.md       # 本文档
```

---

## 策略模式设计

### 设计理念

系统采用**策略模式（Strategy Pattern）**，将Z-score计算算法封装成独立的策略类。这种设计的优势：

1. **开闭原则**：对扩展开放，对修改关闭
2. **单一职责**：每个策略类只负责一种计算方法
3. **策略独立**：各策略之间完全独立，互不影响
4. **易于扩展**：新增策略只需添加新类，无需修改现有代码

### 策略基类设计

所有Z-score计算策略都继承自 `BaseZScoreStrategy` 基类：

```python:strategies/base_zscore_strategy.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseZScoreStrategy(ABC):
    """Z-score计算策略基类"""
    
    def __init__(self, **kwargs):
        """初始化策略"""
        self.name = self.__class__.__name__
        self.params = kwargs
    
    @abstractmethod
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float]) -> float:
        """
        计算当前Z-score（抽象方法，子类必须实现）
        
        Args:
            current_spread: 当前价差
            historical_spreads: 历史价差序列
            
        Returns:
            float: Z-score值
        """
        pass
    
    def validate_input(self, historical_spreads: List[float], min_length: int = 2) -> bool:
        """验证输入数据"""
        if not historical_spreads or len(historical_spreads) < min_length:
            return False
        return True
```

**关键设计点**：
- 使用 `@abstractmethod` 装饰器确保子类必须实现 `calculate_z_score` 方法
- 提供统一的输入验证方法 `validate_input`
- 所有策略具有统一的接口，便于替换

### 策略注入机制

主交易类通过**依赖注入**的方式使用策略：

```python:cointegration_test_windows_optimization_arima_garch.py
class AdvancedCointegrationTrading:
    def __init__(self, ..., z_score_strategy=None, ...):
        """
        初始化交易策略类
        
        Args:
            z_score_strategy: Z-score计算策略对象（BaseZScoreStrategy实例）
        """
        # 设置Z-score策略
        if z_score_strategy is not None:
            self.z_score_strategy = z_score_strategy
        else:
            # 默认使用传统策略
            self.z_score_strategy = TraditionalZScoreStrategy()
    
    def calculate_z_score(self, current_spread, historical_spreads):
        """计算Z-score（委托给策略对象）"""
        if self.z_score_strategy is not None:
            return self.z_score_strategy.calculate_z_score(current_spread, historical_spreads)
        # 向后兼容代码...
```

---

## 策略模型详解

### 1. 传统Z-score策略（TraditionalZScoreStrategy）

#### 原理

传统方法使用**统计学的Z-score公式**，基于历史数据的均值和标准差计算：

**数学公式**：
\[
Z = \frac{X - \mu}{\sigma}
\]

其中：
- \(X\)：当前价差
- \(\mu\)：历史价差的均值
- \(\sigma\)：历史价差的标准差

#### 实现代码

```python:strategies/traditional_zscore_strategy.py
class TraditionalZScoreStrategy(BaseZScoreStrategy):
    """传统Z-score计算策略（使用均值和标准差）"""
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float]) -> float:
        """
        计算当前Z-score（传统方法：使用均值和标准差）
        """
        # 验证输入数据
        if not self.validate_input(historical_spreads, min_length=2):
            return 0.0
        
        # 计算均值和标准差
        spread_mean = np.mean(historical_spreads)
        spread_std = np.std(historical_spreads)
        
        # 如果标准差为0，返回0
        if spread_std == 0:
            return 0.0
        
        # 计算Z-score
        z_score = (current_spread - spread_mean) / spread_std
        
        return z_score
```

#### 特点

- **优点**：
  - 计算简单快速
  - 不需要额外依赖库
  - 对数据量要求低（最少2个数据点）
  - 结果稳定可靠

- **缺点**：
  - 假设价差服从正态分布
  - 无法捕捉价差的动态变化特征
  - 对波动率聚集现象不敏感

#### 适用场景

- 数据量较少时
- 价差序列相对平稳时
- 需要快速计算时

---

### 2. ARIMA-GARCH策略（ArimaGarchZScoreStrategy）

#### 原理

ARIMA-GARCH策略结合了**ARIMA模型**和**GARCH模型**，分别预测价差的均值和波动率：

**ARIMA模型（自回归积分滑动平均模型）**：
- 用于预测价差的**均值**
- 捕捉价差的时间序列特征（趋势、周期性等）

**GARCH模型（广义自回归条件异方差模型）**：
- 用于预测价差的**波动率**
- 捕捉波动率的聚集效应（volatility clustering）

**计算流程**：

1. **ARIMA建模**：对历史价差序列建立ARIMA模型
   \[
   \text{ARIMA}(p, d, q): \phi(B)(1-B)^d X_t = \theta(B)\epsilon_t
   \]

2. **提取残差**：获取ARIMA模型的残差序列
   \[
   \epsilon_t = X_t - \hat{X}_t
   \]

3. **GARCH建模**：对残差序列建立GARCH模型
   \[
   \sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2
   \]

4. **预测均值和波动率**：
   - 使用ARIMA模型预测下一期的均值：\(\hat{\mu}_{t+1}\)
   - 使用GARCH模型预测下一期的波动率：\(\hat{\sigma}_{t+1}\)

5. **计算Z-score**：
   \[
   Z = \frac{X_{t+1} - \hat{\mu}_{t+1}}{\hat{\sigma}_{t+1}}
   \]

#### 实现代码

```python:strategies/arima_garch_zscore_strategy.py
class ArimaGarchZScoreStrategy(BaseZScoreStrategy):
    """ARIMA-GARCH Z-score计算策略"""
    
    def __init__(self, arima_order: Tuple[int, int, int] = (1, 0, 1), 
                 garch_order: Tuple[int, int] = (1, 1), **kwargs):
        """
        初始化ARIMA-GARCH策略
        
        Args:
            arima_order: ARIMA模型阶数 (p, d, q)
            garch_order: GARCH模型阶数 (p, q)
        """
        self.arima_order = arima_order  # 例如：(1, 0, 1)
        self.garch_order = garch_order  # 例如：(1, 1)
        self._arima_garch_models = {}   # 模型缓存
        self._max_cache_size = 10
    
    def calculate_z_score(self, current_spread: float, historical_spreads: List[float]) -> float:
        """
        使用ARIMA-GARCH模型计算Z-score
        """
        # 验证输入数据（ARIMA-GARCH需要更多数据）
        min_required_length = max(20, sum(self.arima_order) + 5)
        if not self.validate_input(historical_spreads, min_length=min_required_length):
            return 0.0
        
        try:
            spreads_array = np.array(historical_spreads)
            
            # 检查模型缓存
            cache_key = (len(spreads_array), tuple(spreads_array[-5:]))
            if cache_key in self._arima_garch_models:
                arima_fitted, garch_fitted = self._arima_garch_models[cache_key]
            else:
                # 步骤1: 拟合ARIMA模型
                arima_model = ARIMA(spreads_array, order=self.arima_order)
                arima_fitted = arima_model.fit()
                
                # 步骤2: 获取ARIMA残差
                arima_residuals = arima_fitted.resid
                
                # 步骤3: 拟合GARCH模型
                garch_model = arch_model(arima_residuals, vol='Garch', 
                                        p=self.garch_order[0], q=self.garch_order[1])
                garch_fitted = garch_model.fit(disp='off')
                
                # 缓存模型
                if len(self._arima_garch_models) >= self._max_cache_size:
                    oldest_key = next(iter(self._arima_garch_models))
                    del self._arima_garch_models[oldest_key]
                self._arima_garch_models[cache_key] = (arima_fitted, garch_fitted)
            
            # 步骤4: 预测当前价差的均值（ARIMA）
            arima_forecast = arima_fitted.forecast(steps=1)
            predicted_mean = float(arima_forecast.iloc[0] if hasattr(arima_forecast, 'iloc') 
                                  else arima_forecast[0])
            
            # 步骤5: 预测当前价差的波动率（GARCH）
            garch_forecast = garch_fitted.forecast(horizon=1)
            predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, 0])
            
            # 验证预测结果
            if predicted_volatility <= 0 or np.isnan(predicted_volatility) or np.isnan(predicted_mean):
                return 0.0
            
            # 步骤6: 计算Z-score
            z_score = (current_spread - predicted_mean) / predicted_volatility
            
            return z_score
            
        except Exception as e:
            # 任何错误都返回0，不进行回退（策略独立）
            print(f"ARIMA-GARCH模型计算失败: {str(e)}")
            return 0.0
```

#### 关键特性

**1. 模型缓存机制**
- 基于数据长度和最后几个值创建缓存键
- 避免重复拟合模型，提高计算效率
- 限制缓存大小为10，防止内存溢出

**2. 策略独立性**
- **不进行回退**：如果ARIMA-GARCH模型失败，返回0而不是回退到传统方法
- 每个策略完全独立，用户明确知道使用的是哪种方法

**3. 参数配置**
- ARIMA阶数 `(p, d, q)`：
  - `p`：自回归项数
  - `d`：差分阶数
  - `q`：滑动平均项数
- GARCH阶数 `(p, q)`：
  - `p`：ARCH项数（残差平方的滞后项）
  - `q`：GARCH项数（条件方差的滞后项）

#### 特点

- **优点**：
  - 能够捕捉价差的动态特征
  - 对波动率聚集现象敏感
  - 预测更准确（理论上）
  - 适应市场变化

- **缺点**：
  - 计算复杂度高
  - 需要更多历史数据（至少20个数据点）
  - 模型拟合可能失败
  - 需要额外的依赖库（statsmodels, arch）

#### 适用场景

- 数据量充足时（≥20个数据点）
- 价差序列有明显的时间序列特征时
- 波动率有明显聚集效应时
- 需要更精确的预测时

---

## 核心代码解析

### 策略选择机制

系统提供了交互式的策略选择功能：

```python:cointegration_test_windows_optimization_arima_garch.py
def select_z_score_strategy():
    """
    选择Z-score计算策略
    
    Returns:
        BaseZScoreStrategy: 选择的策略对象
    """
    print("请选择Z-score计算策略:")
    print("  1. 传统方法（均值和标准差）")
    print("  2. ARIMA-GARCH模型")
    print("  0. 退出程序")
    
    while True:
        choice = input(f"请选择 (0-2): ").strip()
        
        if choice == '0':
            return None
        
        if choice == '1':
            strategy = TraditionalZScoreStrategy()
            print(f"已选择: {strategy.get_strategy_description()}")
            return strategy
        
        if choice == '2' and ARIMA_AVAILABLE and GARCH_AVAILABLE:
            # 配置ARIMA-GARCH参数
            arima_input = input("ARIMA阶数 (p,d,q，格式如: 1,0,1): ").strip()
            garch_input = input("GARCH阶数 (p,q，格式如: 1,1): ").strip()
            
            # 解析参数...
            strategy = ArimaGarchZScoreStrategy(arima_order=arima_order, garch_order=garch_order)
            return strategy
```

### 策略注入到交易类

**普通回测模式**：

```python:cointegration_test_windows_optimization_arima_garch.py
# 选择策略
z_score_strategy = select_z_score_strategy()

# 创建交易策略实例，注入Z-score策略
trading_strategy = AdvancedCointegrationTrading(
    lookback_period=60,
    z_threshold=1.5,
    z_exit_threshold=0.6,
    # ... 其他参数
    z_score_strategy=z_score_strategy  # 注入策略对象
)
```

**参数优化模式**：

```python:cointegration_test_windows_optimization_arima_garch.py
# 选择策略
z_score_strategy = select_z_score_strategy()

# 创建优化器，传入策略对象
optimizer = ParameterOptimizer(
    data=data,
    selected_pairs=selected_pairs,
    initial_capital=10000,
    objective='sharpe_ratio',
    stability_test=True,
    z_score_strategy=z_score_strategy  # 注入策略对象
)

# 优化器在评估参数时使用这个策略
# 在 evaluate_params 方法中：
strategy = AdvancedCointegrationTrading(
    ...
    z_score_strategy=self.z_score_strategy  # 使用优化器的策略对象
)
```

### Z-score计算调用链

```python:cointegration_test_windows_optimization_arima_garch.py
class AdvancedCointegrationTrading:
    def calculate_z_score(self, current_spread, historical_spreads):
        """计算Z-score（委托给策略对象）"""
        if self.z_score_strategy is not None:
            # 调用策略对象的计算方法
            return self.z_score_strategy.calculate_z_score(current_spread, historical_spreads)
        # 向后兼容代码...
    
    def backtest_cointegration_trading(self, data, selected_pairs, initial_capital=10000):
        """回测协整交易策略"""
        for timestamp in all_timestamps:
            # 计算当前价差
            current_spread = self.calculate_current_spread(price1, price2, hedge_ratio)
            
            # 获取历史价差
            historical_spreads = [...]
            
            # 使用策略计算Z-score
            current_z_score = self.calculate_z_score(current_spread, historical_spreads)
            
            # 生成交易信号
            signal = self.generate_trading_signal(current_z_score)
            # ...
```

---

## 使用流程

### 完整使用流程

系统支持两种运行模式：

#### 模式1：普通回测模式

```
1. 启动程序，选择模式1
   ↓
2. 加载数据（CSV文件）
   ↓
3. 配置滚动窗口参数
   ↓
4. 滚动窗口协整检验
   ↓
5. 选择协整对
   ↓
6. 【循环开始】
   ↓
7. 选择Z-score计算策略
   ├─ 1. 传统方法
   └─ 2. ARIMA-GARCH模型（可配置参数）
   ↓
8. 配置交易参数
   ↓
9. 执行回测
   ↓
10. 显示回测结果
   ↓
11. 询问是否继续
   ├─ 1. 继续测试（返回步骤7）
   └─ 0. 退出程序
```

#### 模式2：参数优化模式

```
1. 启动程序，选择模式2
   ↓
2. 加载数据（CSV文件）
   ↓
3. 配置滚动窗口参数
   ↓
4. 滚动窗口协整检验
   ↓
5. 选择协整对
   ↓
6. 选择Z-score计算策略
   ├─ 1. 传统方法
   └─ 2. ARIMA-GARCH模型（可配置参数）
   ↓
7. 选择优化方法
   ├─ 1. 网格搜索（粗粒度+细粒度）
   ├─ 2. 随机搜索
   └─ 3. 贝叶斯优化
   ↓
8. 选择优化目标
   ├─ 1. 夏普比率
   ├─ 2. 总收益率
   └─ 3. 收益率/回撤比
   ↓
9. 执行参数优化
   ↓
10. 显示优化结果（包含最佳参数和策略信息）
   ↓
11. 导出优化结果到CSV
```

### 策略选择示例

```
请选择Z-score计算策略:
  1. 传统方法（均值和标准差）
  2. ARIMA-GARCH模型
  0. 退出程序
请选择 (0-2): 2

配置ARIMA-GARCH模型参数:
  直接回车使用默认值: ARIMA(1,0,1), GARCH(1,1)
ARIMA阶数 (p,d,q，格式如: 1,0,1): 1,0,1
GARCH阶数 (p,q，格式如: 1,1): 1,1
已选择: ARIMA-GARCH模型 (ARIMA(1, 0, 1), GARCH(1, 1))
```

### 循环测试示例

```
第 1 次测试
============================================================
请选择Z-score计算策略:
  1. 传统方法（均值和标准差）
  2. ARIMA-GARCH模型
  0. 退出程序
请选择 (0-2): 1
已选择: 传统方法（均值和标准差）

... 执行回测 ...

第 1 次测试完成
============================================================

请选择下一步操作:
  1. 继续测试（重新选择策略）
  0. 退出程序
请选择 (0/1): 1

第 2 次测试
============================================================
请选择Z-score计算策略:
  ...
```

---

## 技术细节

### 1. 策略独立性设计

**设计原则**：每个策略完全独立，不相互依赖

**实现方式**：
- ARIMA-GARCH策略失败时返回0，**不回退**到传统方法
- 用户明确知道使用的是哪种策略
- 便于对比不同策略的效果

```python:strategies/arima_garch_zscore_strategy.py
except Exception as e:
    # 任何错误都返回0，不进行回退（策略独立）
    print(f"ARIMA-GARCH模型计算失败: {str(e)}")
    return 0.0
```

### 2. 模型缓存机制

**目的**：提高计算效率，避免重复拟合模型

**实现**：
- 使用数据长度和最后5个值作为缓存键
- 限制缓存大小为10个模型
- 采用LRU（最近最少使用）策略清除旧缓存

```python:strategies/arima_garch_zscore_strategy.py
# 创建缓存键
cache_key = (len(spreads_array), tuple(spreads_array[-5:]))

# 检查缓存
if cache_key in self._arima_garch_models:
    arima_fitted, garch_fitted = self._arima_garch_models[cache_key]
else:
    # 拟合新模型并缓存
    ...
```

### 3. 向后兼容性

系统保留了旧的 `use_arima_garch` 参数，确保旧代码仍能运行：

**交易策略类**：

```python:cointegration_test_windows_optimization_arima_garch.py
def __init__(self, ..., z_score_strategy=None, use_arima_garch=False, ...):
    # 优先使用策略对象
    if z_score_strategy is not None:
        self.z_score_strategy = z_score_strategy
    # 向后兼容：使用旧参数
    elif use_arima_garch:
        self.z_score_strategy = ArimaGarchZScoreStrategy(...)
    else:
        self.z_score_strategy = TraditionalZScoreStrategy()
```

**优化器类**：

```python:cointegration_test_windows_optimization_arima_garch.py
class ParameterOptimizer:
    def __init__(self, ..., z_score_strategy=None):
        # 优先使用策略对象
        if z_score_strategy is not None:
            self.z_score_strategy = z_score_strategy
        else:
            self.z_score_strategy = TraditionalZScoreStrategy()
    
    def set_use_arima_garch(self, use_arima_garch):
        """向后兼容方法"""
        if use_arima_garch:
            self.z_score_strategy = ArimaGarchZScoreStrategy()
        else:
            self.z_score_strategy = TraditionalZScoreStrategy()
```

### 4. 优化器策略对象使用

优化器现在完全支持策略对象模式，与普通回测模式保持一致：

**优化器初始化**：

```python:cointegration_test_windows_optimization_arima_garch.py
# 选择策略
z_score_strategy = select_z_score_strategy()

# 创建优化器，传入策略对象
optimizer = ParameterOptimizer(
    data=data,
    selected_pairs=selected_pairs,
    initial_capital=10000,
    objective='sharpe_ratio',
    stability_test=True,
    z_score_strategy=z_score_strategy  # 传入策略对象
)
```

**优化器评估参数**：

```python:cointegration_test_windows_optimization_arima_garch.py
def evaluate_params(self, params):
    # 创建交易策略实例，使用优化器的策略对象
    strategy = AdvancedCointegrationTrading(
        ...
        z_score_strategy=self.z_score_strategy  # 使用优化器的策略对象
    )
    # 执行回测...
```

**优化器显示信息**：

```python:cointegration_test_windows_optimization_arima_garch.py
def grid_search(self):
    print("开始网格搜索优化")
    if self.z_score_strategy:
        print(f"（使用策略: {self.z_score_strategy.get_strategy_description()}）")
```

### 5. 数据验证

每个策略都有数据验证机制：

```python:strategies/base_zscore_strategy.py
def validate_input(self, historical_spreads: List[float], min_length: int = 2) -> bool:
    """验证输入数据"""
    if not historical_spreads or len(historical_spreads) < min_length:
        return False
    return True
```

- **传统策略**：最少需要2个数据点
- **ARIMA-GARCH策略**：最少需要 `max(20, sum(arima_order) + 5)` 个数据点

### 6. 错误处理

**传统策略**：
- 数据不足：返回0
- 标准差为0：返回0

**ARIMA-GARCH策略**：
- 数据不足：返回0
- 模型拟合失败：返回0并打印错误信息
- 预测结果无效：返回0

---

## 总结

### 系统优势

1. **模块化设计**：策略独立，易于维护和扩展
2. **灵活选择**：用户可以根据数据特点选择合适策略
3. **循环测试**：可以快速对比不同策略的效果
4. **统一接口**：普通回测和参数优化都使用相同的策略对象模式
5. **向后兼容**：保留旧接口，不影响现有代码

### 策略对比

| 特性 | 传统方法 | ARIMA-GARCH |
|------|---------|-------------|
| 计算速度 | 快 | 慢 |
| 数据要求 | 低（≥2） | 高（≥20） |
| 预测精度 | 一般 | 较高 |
| 适用场景 | 数据少、平稳 | 数据多、有特征 |
| 依赖库 | 无 | statsmodels, arch |

### 扩展建议

未来可以添加的策略：
- **EWMA策略**：使用指数加权移动平均
- **Kalman Filter策略**：使用卡尔曼滤波
- **机器学习策略**：使用LSTM、Transformer等模型

只需继承 `BaseZScoreStrategy` 并实现 `calculate_z_score` 方法即可！

---

## 优化器模式详细说明

### 优化器策略对象模式

优化器现在完全支持策略对象模式，与普通回测模式使用相同的策略选择机制。

### 优化器使用示例

```python
# 1. 选择策略
z_score_strategy = select_z_score_strategy()
# 输出：已选择: ARIMA-GARCH模型 (ARIMA(1, 0, 1), GARCH(1, 1))

# 2. 创建优化器
optimizer = ParameterOptimizer(
    data=data,
    selected_pairs=selected_pairs,
    initial_capital=10000,
    objective='sharpe_ratio',
    stability_test=True,
    z_score_strategy=z_score_strategy  # 传入策略对象
)

# 3. 执行优化
result = optimizer.optimize(method='grid_search')

# 4. 查看结果
print(f"使用的策略: {z_score_strategy.get_strategy_description()}")
print(f"最佳参数: {result['best_params']}")
print(f"最佳得分: {result['best_score']}")
```

### 优化器显示信息

优化器在执行优化时会显示使用的策略：

```
开始网格搜索优化
（使用策略: ARIMA-GARCH模型 (ARIMA(1, 0, 1), GARCH(1, 1))）
================================================================================
```

### 优化结果包含策略信息

优化完成后，结果中会显示使用的策略：

```
最佳参数:
  lookback_period: 60
  z_threshold: 1.5
  ...
  使用的策略: ARIMA-GARCH模型 (ARIMA(1, 0, 1), GARCH(1, 1))
```

---

**文档版本**：v1.1  
**最后更新**：2024年（已更新优化器策略对象模式）  
**作者**：量化交易系统开发团队

