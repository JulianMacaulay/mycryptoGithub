"""
完整的回测系统（支持机器学习市场状态检测）
支持从币安API或CSV文件加载数据，运行策略，生成回测报告
支持基于机器学习的市场状态检测（趋势/震荡），根据市场状态选择策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys
import requests
import ccxt
import time
import warnings
import importlib
import inspect

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加策略模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.turtle_strategy import TurtleStrategy
from strategies.base_strategy import BaseStrategy

# 尝试导入机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装，Random Forest功能将不可用。可以使用: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost未安装，XGBoost功能将不可用。可以使用: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: tensorflow未安装，LSTM和CNN功能将不可用。可以使用: pip install tensorflow")


# ==================== 技术指标计算函数 ====================

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算ADX（平均趋向指标）
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期（默认14）
        
    Returns:
        ADX序列
    """
    # 计算真实波幅（TR）
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算方向移动（+DM和-DM）
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # 平滑TR、+DM、-DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # 计算DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # 计算ADX
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI（相对强弱指标）
    
    Args:
        close: 收盘价序列
        period: 周期（默认14）
        
    Returns:
        RSI序列
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标
    
    Args:
        close: 收盘价序列
        fast: 快线周期（默认12）
        slow: 慢线周期（默认26）
        signal: 信号线周期（默认9）
        
    Returns:
        (MACD线, 信号线, 柱状图)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bb_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带
    
    Args:
        close: 收盘价序列
        period: 周期（默认20）
        std_dev: 标准差倍数（默认2.0）
        
    Returns:
        (上轨, 中轨, 下轨)
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def calculate_ma_slope(close: pd.Series, period: int = 20, lookback: int = 5) -> pd.Series:
    """
    计算均线斜率（用于判断趋势强度）
    
    Args:
        close: 收盘价序列
        period: 均线周期（默认20）
        lookback: 回看期（默认5）
        
    Returns:
        均线斜率序列
    """
    ma = close.rolling(window=period).mean()
    slope = (ma - ma.shift(lookback)) / lookback
    return slope


# ==================== 市场状态检测器 ====================

class MarketRegimeMLDetector:
    """
    基于机器学习的市场状态检测器
    判断市场是趋势市场还是震荡市场
    """
    
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
        
    def _generate_labels(self, data: pd.DataFrame, is_training: bool = True) -> pd.Series:
        """
        生成标签（趋势=1，震荡=0）
        
        支持三种方法：
        1. forward_looking: 前瞻性标签（使用未来信息，仅训练时可用）
        2. unsupervised: 无监督学习（K-means聚类）
        3. adx_slope: ADX + 均线斜率（传统方法，有滞后性）
        
        Args:
            data: 包含OHLCV数据的DataFrame
            is_training: 是否用于训练（前瞻性标签仅在训练时可用）
            
        Returns:
            标签序列（1=趋势，0=震荡）
        """
        if self.label_method == 'forward_looking':
            return self._generate_forward_looking_labels(data, is_training)
        elif self.label_method == 'unsupervised':
            return self._generate_unsupervised_labels(data, is_training)
        elif self.label_method == 'adx_slope':
            return self._generate_adx_slope_labels(data)
        else:
            raise ValueError(f"不支持的标签生成方法: {self.label_method}")
    
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
            self.model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32, verbose=0)
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
    
    def _prepare_lstm_input(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """准备LSTM/CNN输入（时间序列格式）"""
        X_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
        return np.array(X_seq)
    
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


class BacktestEngine:
    """
    回测引擎
    负责执行策略回测，记录交易，计算收益
    """

    def __init__(self, initial_capital: float = 10000, commission_rate: float = 0.001,
                 leverage: float = 5.0, position_ratio: float = 0.5):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（默认0.1%）
            leverage: 杠杆倍数（默认5倍）
            position_ratio: 仓位比例（默认1.0，即100%，0.5表示50%）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.leverage = leverage
        self.position_ratio = position_ratio

        # 计算可用资金（考虑仓位比例）
        self.available_capital = initial_capital * position_ratio

        # 账户状态（使用可用资金，因为只有这部分可以交易）
        self.balance = self.available_capital
        self.equity = self.available_capital
        self.position_size = 0.0  # 持仓数量（正数=多头，负数=空头）
        self.position_value = 0.0  # 持仓价值
        self.entry_price = 0.0  # 入场价格
        self.entry_idx = -1  # 入场索引
        self.entry_count = 0  # 加仓次数

        # 交易记录
        self.trades = []  # 所有交易记录
        self.equity_curve = []  # 权益曲线
        self.signals = []  # 所有信号记录

    def reset(self):
        """重置回测引擎"""
        # 重置时使用可用资金（考虑仓位比例）
        self.balance = self.available_capital
        self.equity = self.available_capital
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_count = 0
        self.trades = []
        self.equity_curve = []
        self.signals = []

    def open_position(self, signal: str, price: float, size: float,
                      current_idx: int, reason: str = ""):
        """
        开仓

        Args:
            signal: 'long' 或 'short'
            price: 开仓价格
            size: 开仓数量（绝对值）
            current_idx: 当前索引
            reason: 开仓原因
        """
        # 计算最大可开仓位（基于可用资金和杠杆）
        max_position_value = self.balance * self.leverage  # 最大仓位价值
        max_size = max_position_value / price if price > 0 else 0  # 最大数量

        # 取较小值（策略计算的size和最大可开仓位）
        actual_size = min(size, max_size) if max_size > 0 else 0

        if actual_size <= 0:
            # 无法开仓，直接返回
            return

        if signal == 'long':
            self.position_size = actual_size
            self.entry_price = price
            self.entry_idx = current_idx
            self.entry_count = 1

            # 计算成本（考虑杠杆）
            # 开仓价值 = actual_size * price
            # 保证金 = 开仓价值 / 杠杆倍数
            # 成本 = 保证金 * (1 + 手续费率)
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost

            self.trades.append({
                'type': 'open_long',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })

        elif signal == 'short':
            self.position_size = -actual_size  # 负数表示空头
            self.entry_price = price  # 价格始终是正数
            self.entry_idx = current_idx
            self.entry_count = 1

            # 做空：也需要保证金（类似做多），扣除资金
            # 在期货交易中，做空也需要保证金，只是持仓方向相反
            # 计算成本（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost

            self.trades.append({
                'type': 'open_short',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })

    def add_position(self, signal: str, price: float, size: float,
                     current_idx: int, reason: str = ""):
        """
        加仓

        Args:
            signal: 'add_long' 或 'add_short'
            price: 加仓价格
            size: 加仓数量（绝对值）
            current_idx: 当前索引
            reason: 加仓原因
        """
        # 计算最大可加仓位（基于可用资金和杠杆）
        max_position_value = self.balance * self.leverage  # 最大仓位价值
        max_size = max_position_value / price if price > 0 else 0  # 最大数量

        # 取较小值（策略计算的size和最大可开仓位）
        actual_size = min(size, max_size) if max_size > 0 else 0

        if actual_size <= 0:
            # 无法加仓（资金不足）
            return

        if signal == 'add_long' and self.position_size > 0:
            # 计算新的平均入场价格
            total_size = self.position_size + actual_size
            total_cost = self.position_size * self.entry_price + actual_size * price
            self.entry_price = total_cost / total_size
            self.position_size = total_size
            self.entry_count += 1

            # 扣除成本（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost

            self.trades.append({
                'type': 'add_long',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })

        elif signal == 'add_short' and self.position_size < 0:
            # 计算新的平均入场价格（价格始终是正数）
            current_size = abs(self.position_size)
            total_size = current_size + actual_size
            total_cost = current_size * self.entry_price + actual_size * price
            self.entry_price = total_cost / total_size  # 价格始终是正数
            self.position_size = -total_size  # 负数表示空头
            self.entry_count += 1

            # 做空加仓：也需要扣除保证金（考虑杠杆）
            position_value = actual_size * price
            margin = position_value / self.leverage
            cost = margin * (1 + self.commission_rate)
            self.balance -= cost

            self.trades.append({
                'type': 'add_short',
                'price': price,
                'size': actual_size,
                'idx': current_idx,
                'balance': self.balance,
                'equity': self.equity,  # 记录当前权益
                'reason': reason
            })

    def close_position(self, price: float, current_idx: int, reason: str = ""):
        """
        平仓

        Args:
            price: 平仓价格
            current_idx: 当前索引
            reason: 平仓原因
        """
        if self.position_size == 0:
            return None

        # 计算盈亏
        if self.position_size > 0:  # 平多
            # 多头：买入价 entry_price，卖出价 price
            gross_pnl = (price - self.entry_price) * self.position_size
            # 平仓时：收回保证金 + 盈亏，扣除平仓手续费
            # 开仓时扣除了：margin * (1 + commission_rate) = margin + margin * commission_rate
            # 平仓时收回：margin + gross_pnl - close_cost
            size = self.position_size
            entry_margin = (self.entry_price * size) / self.leverage
            close_cost = (size * price) * self.commission_rate
            # 计算开仓手续费（开仓时已经扣除，但需要在pnl中体现，以便CSV中记录的是净盈亏）
            # 开仓手续费 = (entry_price * size / leverage) * commission_rate
            open_cost = entry_margin * self.commission_rate
            # 净盈亏 = 毛盈亏 - 平仓手续费 - 开仓手续费（用于CSV记录）
            pnl = gross_pnl - close_cost - open_cost
            # 余额计算：收回保证金 + 毛盈亏 - 平仓手续费（开仓手续费已在开仓时扣除）
            self.balance += entry_margin + gross_pnl - close_cost
        else:  # 平空
            # 空头：开仓价 entry_price（正数），平仓价 price
            # 做空时已经扣除了保证金
            # 平仓时：如果价格下跌，盈利；如果价格上涨，亏损
            # 盈亏 = (entry_price - price) * size
            size = abs(self.position_size)  # 获取数量（正数）
            entry_price = self.entry_price  # entry_price已经是正数，不需要abs
            gross_pnl = (entry_price - price) * size

            # 平仓时：收回保证金 + 盈亏，扣除平仓手续费
            # 开仓时扣除了：margin * (1 + commission_rate) = margin + margin * commission_rate
            # 平仓时收回：margin + gross_pnl - close_cost
            entry_margin = (entry_price * size) / self.leverage
            close_cost = (size * price) * self.commission_rate
            # 计算开仓手续费（开仓时已经扣除，但需要在pnl中体现，以便CSV中记录的是净盈亏）
            # 开仓手续费 = (entry_price * size / leverage) * commission_rate
            open_cost = entry_margin * self.commission_rate
            # 净盈亏 = 毛盈亏 - 平仓手续费 - 开仓手续费（用于CSV记录）
            pnl = gross_pnl - close_cost - open_cost
            # 余额计算：收回保证金 + 毛盈亏 - 平仓手续费（开仓手续费已在开仓时扣除）
            self.balance += entry_margin + gross_pnl - close_cost

        # 记录交易（保存平仓前的entry_price，确保是正数）
        trade_type = 'close_long' if self.position_size > 0 else 'close_short'
        # 保存entry_price（确保是正数）
        saved_entry_price = abs(self.entry_price) if self.entry_price < 0 else self.entry_price
        self.trades.append({
            'type': trade_type,
            'price': price,
            'size': abs(self.position_size),
            'idx': current_idx,
            'entry_price': saved_entry_price,  # 确保是正数
            'pnl': pnl,  # 记录净盈亏（已扣除平仓手续费和开仓手续费）
            'balance': self.balance,
            'equity': self.balance,  # 平仓后无持仓，权益=余额
            'reason': reason,
            'entry_count': self.entry_count
        })

        # 重置持仓
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_count = 0

        return pnl

    def update_equity(self, current_price: float):
        """
        更新权益（未实现盈亏）

        Args:
            current_price: 当前价格
        """
        if self.position_size == 0:
            self.equity = self.balance
        elif self.position_size > 0:  # 多头
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            self.equity = self.balance + unrealized_pnl
            self.position_value = self.position_size * current_price
        else:  # 空头
            # 做空：entry_price是正数（开仓价），position_size是负数
            # 未实现盈亏 = (开仓价 - 当前价) * 数量
            size = abs(self.position_size)  # 获取数量（正数）
            entry_price = self.entry_price  # entry_price已经是正数，不需要abs
            unrealized_pnl = (entry_price - current_price) * size
            self.equity = self.balance + unrealized_pnl
            self.position_value = size * current_price

        # 记录权益曲线：交易账户权益 + 未投入资金（与最终权益计算保持一致）
        total_equity = self.equity + (self.initial_capital - self.available_capital)
        self.equity_curve.append(total_equity)


class BacktestSystem:
    """
    完整的回测系统
    """

    def __init__(self, strategy: BaseStrategy = None, strategies: Dict[str, BaseStrategy] = None,
                 initial_capital: float = 10000, leverage: float = 5.0, position_ratio: float = 0.5,
                 market_detector: MarketRegimeMLDetector = None):
        """
        初始化回测系统

        Args:
            strategy: 单一策略实例（向后兼容）
            strategies: 策略字典 {'trending': 趋势策略, 'ranging': 震荡策略}
            initial_capital: 初始资金
            leverage: 杠杆倍数（默认5倍）
            position_ratio: 仓位比例（默认1.0，即100%，0.5表示用50%的资金）
            market_detector: 市场状态检测器
        """
        # 策略管理
        if strategies is not None:
            self.strategies = strategies
            self.strategy = None  # 当前使用的策略
        elif strategy is not None:
            self.strategy = strategy
            self.strategies = None
        else:
            raise ValueError("必须提供strategy或strategies参数")
        
        self.market_detector = market_detector
        self.use_market_regime = market_detector is not None
        
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            leverage=leverage,
            position_ratio=position_ratio
        )
        self.data = None
        self.market_regimes = None  # 存储市场状态预测结果
        self.entry_strategy = None  # 记录开仓时使用的策略（用于持仓管理）

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

    def load_data_from_binance(self, symbol: str, interval: str = '1h',
                               limit: int = 1000, start_time: int = None, end_time: int = None):
        """
        从币安API加载数据

        Args:
            symbol: 交易对（如 'BTCUSDT'）
            interval: K线周期（如 '1h', '1d'）
            limit: 获取数量
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
        """
        print(f"从币安获取 {symbol} 数据...")

        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        all_klines = []
        end_time_param = end_time

        # 分页获取数据
        while len(all_klines) < limit:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    interval,
                    since=start_time,
                    limit=min(1000, limit - len(all_klines)),
                    params={'endTime': end_time_param} if end_time_param else {}
                )

                if not ohlcv:
                    break

                all_klines.extend(ohlcv)

                if len(ohlcv) < 1000 or len(all_klines) >= limit:
                    break

                # 更新end_time用于下次请求
                end_time_param = ohlcv[0][0] - 1
                time.sleep(0.1)

            except Exception as e:
                print(f"获取数据失败: {e}")
                break

        # 转换为DataFrame
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df.sort_index()

        self.data = df
        print(f"成功获取数据: {len(df)} 条记录")
        print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

        return df

    def run_backtest(self, max_entries: int = 3):
        """
        运行回测

        Args:
            max_entries: 最大加仓次数
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        # 重置引擎
        self.engine.reset()

        # 如果使用市场状态检测，先预测市场状态
        if self.use_market_regime and self.market_detector.is_trained:
            print("\n预测市场状态...")
            self.market_regimes = self.market_detector.predict(self.data)
            print(f"趋势市场占比: {(self.market_regimes == 'trending').sum() / len(self.market_regimes):.2%}")
            print(f"震荡市场占比: {(self.market_regimes == 'ranging').sum() / len(self.market_regimes):.2%}")
        else:
            self.market_regimes = None

        # 初始化策略
        if self.strategies is not None:
            # 多策略模式：初始化所有策略
            for regime_type, strategy in self.strategies.items():
                strategy.initialize(self.data)
            self.strategy = None  # 当前策略将在循环中动态选择
        else:
            # 单策略模式
            self.strategy.initialize(self.data)

            print(f"\n开始回测...")
            print(f"初始资金: {self.engine.initial_capital}")
            print(f"数据量: {len(self.data)} 条")
            print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        if self.use_market_regime:
            print(f"使用市场状态检测: 是")
            if self.strategies:
                print(f"可用策略: {list(self.strategies.keys())}")

        # 遍历每个K线
        for idx in range(len(self.data)):
            current_bar = self.data.iloc[idx]
            current_price = current_bar['close']

            # 根据市场状态选择策略（如果使用市场状态检测）
            if self.use_market_regime and self.market_regimes is not None and self.strategies is not None:
                current_regime = self.market_regimes.iloc[idx]
                
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
                current_regime = self.market_regimes.iloc[idx]
                # 如果策略是趋势策略但市场是震荡，或反之，则不交易
                # 这里需要知道策略类型，暂时允许交易（可以在策略类中添加类型属性）
                pass

            # 更新权益
            self.engine.update_equity(current_price)

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
                        stop_signal['reason']
                    )
                    if pnl is not None:
                        current_strategy.update_trade_result(pnl)
                    self.entry_strategy = None  # 清除开仓策略记录
                    continue

                # 检查均线交叉退出
                signal = current_strategy.generate_signals(self.data, idx, self.engine.position_size)
                if signal['signal'] in ['close_long', 'close_short']:
                    pnl = self.engine.close_position(current_price, idx, signal['reason'])
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
                                    add_signal['reason']
                                )
                    continue
            
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
                            signal['reason']
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
                            signal['reason']
                        )
                        # 记录开仓时使用的策略
                        self.entry_strategy = self.strategy

        # 最后平仓（如果有持仓）
        if self.engine.position_size != 0:
            last_price = self.data.iloc[-1]['close']
            # 使用开仓时的策略
            final_strategy = self.entry_strategy if self.entry_strategy is not None else self.strategy
            pnl = self.engine.close_position(last_price, len(self.data) - 1, "回测结束平仓")
            if pnl is not None and final_strategy:
                final_strategy.update_trade_result(pnl)
            self.entry_strategy = None

        # 计算最终权益
        # 交易账户权益（只包含投入交易的部分）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        # 最终总资产 = 交易账户权益 + 未投入资金
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)
        # 收益率计算：用总资产变化除以初始资金
        # 例如：初始10000，用5000赚了1000，最终总资产=14156，收益率 = (14156-10000)/10000 = 41.56%
        total_return = (final_equity - self.engine.initial_capital) / self.engine.initial_capital * 100

        print(f"\n回测完成！")
        print(f"初始资金: {self.engine.initial_capital:,.2f}")
        print(f"可用资金: {self.engine.available_capital:,.2f} (仓位比例: {self.engine.position_ratio * 100:.1f}%)")
        print(f"杠杆倍数: {self.engine.leverage}x")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"总交易次数: {len([t for t in self.engine.trades if 'close' in t['type']])}")

    def generate_report(self) -> Dict:
        """
        生成回测报告

        Returns:
            包含各种统计信息的字典
        """
        # 提取所有平仓交易
        closed_trades = [t for t in self.engine.trades if 'close' in t['type']]

        # 计算最终权益（如果有持仓，使用当前权益；否则使用余额）
        # 交易账户权益（只包含投入交易的部分）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        # 最终总资产 = 交易账户权益 + 未投入资金
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)

        # 计算总收益率：用总资产变化除以初始资金
        # 例如：初始10000，用5000赚了1000，最终总资产=14156，收益率 = (14156-10000)/10000 = 41.56%
        total_return = (final_equity - self.engine.initial_capital) / self.engine.initial_capital * 100

        if not closed_trades:
            return {
                'initial_capital': self.engine.initial_capital,
                'available_capital': self.engine.available_capital,
                'leverage': self.engine.leverage,
                'position_ratio': self.engine.position_ratio,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'total_pnl': final_equity - self.engine.initial_capital
            }

        # 计算统计指标
        pnls = [t['pnl'] for t in closed_trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)

        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(
            losing_trades) != 0 else (float('inf') if winning_trades else 0)

        # 计算最大回撤：最大回撤百分比相对于初始资金（不是峰值）
        equity_array = np.array(self.engine.equity_curve)
        if len(equity_array) > 0:
            peak = equity_array[0]
            max_drawdown = 0
            max_drawdown_pct = 0

            for equity in equity_array:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                # 最大回撤百分比 = 最大回撤 / 初始资金 * 100
                # 例如：初始10000，最大回撤2000，回撤百分比 = 2000/10000 = 20%
                drawdown_pct = drawdown / self.engine.initial_capital if self.engine.initial_capital > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

            max_drawdown_pct = max_drawdown_pct * 100  # 转换为百分比
        else:
            max_drawdown = 0
            max_drawdown_pct = 0

        report = {
            'initial_capital': self.engine.initial_capital,
            'available_capital': self.engine.available_capital,
            'leverage': self.engine.leverage,
            'position_ratio': self.engine.position_ratio,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }

        return report

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率（参考标准计算方式）"""
        if len(self.engine.equity_curve) < 2:
            return 0.0

        # 计算每期收益率
        returns = []
        for i in range(1, len(self.engine.equity_curve)):
            prev_equity = self.engine.equity_curve[i - 1]
            curr_equity = self.engine.equity_curve[i]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # 年化夏普比率（假设数据是小时级别，252*24=6048小时/年）
        # 如果是日线，使用252；如果是小时线，使用6048
        periods_per_year = 252  # 默认按日线计算
        if self.data is not None and len(self.data) > 0:
            time_diff = (self.data.index[-1] - self.data.index[0]).total_seconds() / 3600
            if time_diff > 0:
                data_points = len(self.data)
                hours_per_point = time_diff / data_points
                if hours_per_point < 2:
                    periods_per_year = 252 * 24  # 小时线
                elif hours_per_point < 12:
                    periods_per_year = 252 * 2  # 4小时线
                else:
                    periods_per_year = 252  # 日线

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        return sharpe

    def plot_results(self, save_path: str = None):
        """
        绘制回测结果（基于交易记录中的盈亏数据）

        Args:
            save_path: 保存路径（可选）
        """
        if self.data is None:
            print("没有数据可绘制")
            return

        # 从交易记录中提取盈亏数据，生成权益曲线
        print("正在从交易记录生成权益曲线...")

        # 获取所有平仓交易（只有平仓交易才有pnl）
        closed_trades = [t for t in self.engine.trades if 'close' in t['type']]

        if not closed_trades:
            print("没有交易记录可绘制")
            return

        # 按时间索引排序
        closed_trades.sort(key=lambda x: x['idx'])

        # 计算累计盈亏（基于交易记录中的pnl）
        cumulative_pnl = 0
        equity_points = []  # [(时间索引, 总权益)]

        # 初始权益 = 可用资金
        initial_trading_equity = self.engine.available_capital

        # 构建每个K线的权益值
        trade_idx = 0  # 当前处理的交易索引

        for idx in range(len(self.data)):
            # 累加这个K线之前的所有盈亏
            while trade_idx < len(closed_trades) and closed_trades[trade_idx]['idx'] < idx:
                cumulative_pnl += closed_trades[trade_idx].get('pnl', 0)
                trade_idx += 1

            # 如果这个K线有平仓交易，加上这笔交易的盈亏
            current_pnl = 0
            while trade_idx < len(closed_trades) and closed_trades[trade_idx]['idx'] == idx:
                current_pnl += closed_trades[trade_idx].get('pnl', 0)
                trade_idx += 1

            # 计算当前权益 = 初始可用资金 + 累计盈亏
            current_trading_equity = initial_trading_equity + cumulative_pnl + current_pnl

            # 总权益 = 交易账户权益 + 未投入资金
            total_equity = current_trading_equity + (self.engine.initial_capital - self.engine.available_capital)

            equity_points.append((idx, total_equity))

            # 更新累计盈亏（用于下一个K线）
            cumulative_pnl += current_pnl

        # 提取时间和权益值
        timestamps = [self.data.index[idx] for idx, _ in equity_points]
        equities = [eq for _, eq in equity_points]

        # 计算收益率：相对于初始资金
        initial_equity = self.engine.initial_capital
        returns = [(eq - initial_equity) / self.engine.initial_capital * 100 for eq in equities]

        # 计算最终权益（与报告保持一致）
        trading_equity = self.engine.equity if self.engine.position_size != 0 else self.engine.balance
        final_equity = trading_equity + (self.engine.initial_capital - self.engine.available_capital)

        print("正在创建图表...")

        # 创建图表（参考标准格式：资金曲线 + 收益率曲线）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # 处理多策略模式
        if self.strategies is not None:
            strategy_title = " + ".join([s.name for s in self.strategies.values()])
        elif self.strategy is not None:
            strategy_title = self.strategy.name
        else:
            strategy_title = "ML策略"
        
        fig.suptitle(f'{strategy_title} 回测结果', fontsize=16, fontweight='bold')

        # 1. 权益曲线（资金曲线）
        print("正在绘制权益曲线...")
        ax1.plot(timestamps, equities, linewidth=1.5, color='blue', alpha=0.8, label='权益曲线')
        ax1.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.7, label='初始资金')

        # 添加最终权益标注
        ax1.axhline(y=final_equity, color='red', linestyle='--', alpha=0.5,
                    label=f'最终权益: {final_equity:,.2f}')
        # 在图表上添加文本标注
        ax1.text(timestamps[-1], final_equity, f' {final_equity:,.2f}',
                 verticalalignment='bottom', fontsize=10, color='red')

        ax1.set_title('权益曲线（基于交易记录）', fontsize=14, fontweight='bold')
        ax1.set_ylabel('权益 (USDT)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. 收益率曲线
        print("正在绘制收益率曲线...")
        ax2.plot(timestamps, returns, linewidth=1.5, color='green', alpha=0.8, label='收益率')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='零线')
        ax2.set_title('收益率曲线', fontsize=14, fontweight='bold')
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 格式化时间轴（参考标准方式）
        print("正在格式化时间轴...")
        if len(timestamps) > 100:
            # 数据点多时，减少时间轴标签
            step = max(1, len(timestamps) // 10)
            for ax in [ax1, ax2]:
                ax.set_xticks(timestamps[::step])
                ax.tick_params(axis='x', rotation=45)
        else:
            # 数据点少时，正常格式化
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)

        print("正在调整布局...")
        plt.tight_layout()

        if save_path:
            print(f"正在保存图片到: {save_path}")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"回测结果图已保存到: {save_path}")

        print("正在显示图表...")
        try:
            plt.show(block=True)
            print("图表显示完成")
        except Exception as e:
            print(f"图表显示失败: {str(e)}")
            print("尝试保存图片...")
            backup_path = save_path or 'backtest_result_backup.png'
            plt.savefig(backup_path, dpi=200, bbox_inches='tight')
            print(f"图片已保存为 {backup_path}")
            plt.close()

    def export_trades_to_csv(self, filepath: str = None):
        """
        导出交易记录到CSV文件

        Args:
            filepath: 保存路径，如果为None则自动生成
        """
        if not self.engine.trades:
            print("没有交易记录可导出")
            return

        # 准备交易记录数据
        trades_data = []
        for trade in self.engine.trades:
            trade_record = {
                '交易类型': trade['type'],
                '时间索引': trade['idx'],
                '时间': self.data.index[trade['idx']] if self.data is not None and trade['idx'] < len(
                    self.data) else 'N/A',
                '价格': trade['price'],
                '数量': trade['size'],
                '原因': trade.get('reason', ''),
            }

            # 如果是平仓交易，添加盈亏信息
            if 'close' in trade['type']:
                trade_record['入场价格'] = trade.get('entry_price', 'N/A')
                trade_record['盈亏'] = trade.get('pnl', 0)
                trade_record['加仓次数'] = trade.get('entry_count', 0)

            # 余额：该笔交易后的现金余额（不包括未实现盈亏）
            # 说明：余额是账户中的现金，开仓时会扣除保证金，平仓时会收回
            trade_record['余额'] = trade.get('balance', 0)

            # 权益：该笔交易后的账户总价值（余额 + 未实现盈亏）
            # 说明：如果有持仓，权益 = 余额 + 未实现盈亏；如果无持仓，权益 = 余额
            trade_record['权益'] = trade.get('equity', trade.get('balance', 0))

            trades_data.append(trade_record)

        # 转换为DataFrame
        df = pd.DataFrame(trades_data)

        # 生成文件名
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 处理多策略模式
            if self.strategies is not None:
                strategy_names = "_".join([s.name for s in self.strategies.values()])
                filepath = f"trades_{strategy_names}_{timestamp}.csv"
            elif self.strategy is not None:
                filepath = f"trades_{self.strategy.name}_{timestamp}.csv"
            else:
                filepath = f"trades_ML_{timestamp}.csv"

        # 保存到CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n交易记录已保存到: {filepath}")
        print(f"共 {len(trades_data)} 笔交易记录")

        return filepath

    def print_report(self):
        """打印回测报告（参考标准格式）"""
        report = self.generate_report()

        if not report or report['total_trades'] == 0:
            print("\n" + "=" * 60)
            print("回测报告")
            print("=" * 60)
            print(f"初始资金: {report.get('initial_capital', 0):,.2f}")
            print(f"最终权益: {report.get('final_equity', 0):,.2f}")
            print(f"总收益率: {report.get('total_return', 0):.2f}%")
            print(f"总交易次数: {report.get('total_trades', 0)}")
            print("=" * 60)
            return

        print("\n" + "=" * 60)
        print("回测报告")
        print("=" * 60)
        print(f"初始资金: {report['initial_capital']:,.2f}")
        print(f"可用资金: {report['available_capital']:,.2f} (仓位比例: {report['position_ratio'] * 100:.1f}%)")
        print(f"杠杆倍数: {report['leverage']}x")
        print(f"最终权益: {report['final_equity']:,.2f}")
        print(f"总收益率: {report['total_return']:.2f}%")
        print(f"总交易次数: {report['total_trades']}")
        print(f"盈利交易: {report['winning_trades']}")
        print(f"胜率: {report['win_rate']:.1f}%")

        print(f"\n风险指标:")
        print(f"  最大回撤: {report['max_drawdown']:,.2f}")
        print(f"  最大回撤百分比: {report['max_drawdown_pct']:.2f}%")
        print(f"  盈亏比: {report['profit_factor']:.2f}" if report['profit_factor'] != float('inf') else f"  盈亏比: ∞")
        print(f"  夏普比率: {report['sharpe_ratio']:.2f}")
        print(f"  平均盈利: {report['avg_win']:.2f}")
        print(f"  平均亏损: {report['avg_loss']:.2f}")
        print("=" * 60)

        # 自动导出交易记录
        self.export_trades_to_csv()


# ==================== 策略加载函数 ====================

def load_strategy_from_module(module_name: str, strategy_class_name: str, params: Dict = None) -> BaseStrategy:
    """
    从strategies文件夹动态加载策略
    
    Args:
        module_name: 模块名称（不含.py扩展名，如 'turtle_strategy'）
        strategy_class_name: 策略类名称（如 'TurtleStrategy'）
        params: 策略参数字典
        
    Returns:
        策略实例
    """
    try:
        # 动态导入模块
        module = importlib.import_module(f'strategies.{module_name}')
        
        # 获取策略类
        strategy_class = getattr(module, strategy_class_name)
        
        # 检查是否是BaseStrategy的子类
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class_name} 不是BaseStrategy的子类")
        
        # 创建策略实例
        if params is None:
            params = {}
        strategy = strategy_class(params)
        
        return strategy
        
    except ImportError as e:
        raise ImportError(f"无法导入策略模块 {module_name}: {str(e)}")
    except AttributeError as e:
        raise AttributeError(f"策略类 {strategy_class_name} 不存在: {str(e)}")
    except Exception as e:
        raise Exception(f"加载策略失败: {str(e)}")


def list_available_strategies() -> List[str]:
    """
    列出strategies文件夹中所有可用的策略
    
    Returns:
        策略类名称列表
    """
    strategies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')
    strategies = []
    
    for filename in os.listdir(strategies_dir):
        if filename.endswith('_strategy.py') or filename.endswith('_strategy_bt.py'):
            module_name = filename[:-3]  # 移除.py
            try:
                module = importlib.import_module(f'strategies.{module_name}')
                # 查找所有BaseStrategy的子类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy and 
                        obj.__module__ == module.__name__):
                        strategies.append(f"{module_name}.{name}")
            except:
                pass
    
    return strategies


def select_strategies() -> Dict[str, BaseStrategy]:
    """
    让用户选择策略（趋势类和/或震荡类）
    
    Returns:
        策略字典 {'trending': 趋势策略, 'ranging': 震荡策略} 或 None
    """
    print("\n" + "=" * 60)
    print("选择交易策略")
    print("=" * 60)
    
    # 列出可用策略
    available_strategies = list_available_strategies()
    if not available_strategies:
        print("未找到可用策略")
        return None
    
    print("\n可用策略:")
    for i, strategy_name in enumerate(available_strategies, 1):
        print(f"  {i}. {strategy_name}")
    
    strategies = {}
    
    # 选择趋势类策略
    print("\n选择趋势类策略（用于趋势市场）:")
    print("  输入策略编号，或直接回车跳过（不选择趋势策略）")
    trend_choice = input("请选择: ").strip()
    
    if trend_choice:
        try:
            trend_idx = int(trend_choice) - 1
            if 0 <= trend_idx < len(available_strategies):
                strategy_name = available_strategies[trend_idx]
                module_name, class_name = strategy_name.split('.')
                
                # 获取策略参数
                params = get_strategy_params(class_name)
                
                strategies['trending'] = load_strategy_from_module(module_name, class_name, params)
                print(f" 已选择趋势策略: {strategy_name}")
            else:
                print("无效选择")
        except Exception as e:
            print(f"加载趋势策略失败: {str(e)}")
    
    # 选择震荡类策略
    print("\n选择震荡类策略（用于震荡市场）:")
    print("  输入策略编号，或直接回车跳过（不选择震荡策略）")
    range_choice = input("请选择: ").strip()
    
    if range_choice:
        try:
            range_idx = int(range_choice) - 1
            if 0 <= range_idx < len(available_strategies):
                strategy_name = available_strategies[range_idx]
                module_name, class_name = strategy_name.split('.')
                
                # 获取策略参数
                params = get_strategy_params(class_name)
                
                strategies['ranging'] = load_strategy_from_module(module_name, class_name, params)
                print(f"✓ 已选择震荡策略: {strategy_name}")
            else:
                print("无效选择")
        except Exception as e:
            print(f"加载震荡策略失败: {str(e)}")
    
    if not strategies:
        print("\n警告: 未选择任何策略，将退出")
        return None
    
    return strategies


def get_strategy_params(strategy_class_name: str) -> Dict:
    """
    获取策略参数（用户输入）
    
    Args:
        strategy_class_name: 策略类名称
        
    Returns:
        参数字典
    """
    print(f"\n配置 {strategy_class_name} 参数:")
    print("  直接回车使用默认值")
    
    params = {}
    
    # 根据策略类型设置默认参数
    if 'Turtle' in strategy_class_name:
        params = {
        'n_entries': 3,
        'atr_length': 20,
        'bo_length': 20,
        'fs_length': 55,
        'te_length': 10,
        'use_filter': False,
        'mas': 10,
        'mal': 20
    }

        # 允许用户修改参数
        n_entries_input = input("加仓次数 (默认3): ").strip()
        if n_entries_input:
            try:
                params['n_entries'] = int(n_entries_input)
            except:
                pass
    
    return params


def select_label_method() -> str:
    """
    让用户选择标签生成方法
    
    Returns:
        标签生成方法字符串
    """
    print("\n" + "=" * 60)
    print("选择标签生成方法")
    print("=" * 60)
    print("1. 前瞻性标签（Forward Looking）")
    print("   使用未来N期的市场表现来判断当前状态")
    print("   优点: 标签更准确，反映当前状态的真实结果")
    print("   缺点: 训练时可用，回测时无法使用（但模型可以预测）")
    print()
    print("2. 无监督学习（Unsupervised Learning）")
    print("   使用K-means聚类自动发现市场状态")
    print("   优点: 不依赖人为定义，自动发现市场模式")
    print("   缺点: 需要足够数据，聚类结果需要解释")
    print()
    print("3. ADX + 均线斜率（传统方法）")
    print("   使用ADX和均线斜率判断趋势/震荡")
    print("   优点: 简单直观，易于理解")
    print("   缺点: 有滞后性，可能不够准确")
    
    while True:
        choice = input("\n请选择标签生成方法 (1/2/3, 默认1): ").strip()
        
        if not choice:
            return 'forward_looking'
        
        if choice == '1':
            print("✓ 已选择: 前瞻性标签")
            return 'forward_looking'
        elif choice == '2':
            if not SKLEARN_AVAILABLE:
                print("错误: 无监督学习需要scikit-learn库")
                continue
            print("✓ 已选择: 无监督学习")
            return 'unsupervised'
        elif choice == '3':
            print("✓ 已选择: ADX + 均线斜率")
            return 'adx_slope'
        else:
            print("无效选择，请输入 1、2 或 3")


def select_ml_model() -> str:
    """
    让用户选择机器学习模型
    
    Returns:
        模型类型字符串
    """
    print("\n" + "=" * 60)
    print("选择机器学习模型")
    print("=" * 60)
    
    available_models = []
    
    if XGBOOST_AVAILABLE:
        available_models.append(('1', 'xgboost', 'XGBoost（推荐，速度快，效果好）'))
    else:
        print("  1. XGBoost（不可用：需要安装 xgboost）")
    
    if SKLEARN_AVAILABLE:
        available_models.append(('2', 'random_forest', 'Random Forest（稳定，可解释性强）'))
    else:
        print("  2. Random Forest（不可用：需要安装 scikit-learn）")
    
    if TENSORFLOW_AVAILABLE:
        available_models.append(('3', 'lstm', 'LSTM（适合时间序列，训练较慢）'))
        available_models.append(('4', 'cnn', 'CNN（适合模式识别，训练较慢）'))
    else:
        print("  3. LSTM（不可用：需要安装 tensorflow）")
        print("  4. CNN（不可用：需要安装 tensorflow）")
    
    if not available_models:
        print("\n错误: 没有可用的机器学习库，请安装 xgboost 或 scikit-learn 或 tensorflow")
        return None
    
    print("\n可用模型:")
    for choice, model_type, description in available_models:
        print(f"  {choice}. {description}")
    
    while True:
        user_choice = input(f"\n请选择模型 (1-{len(available_models)}): ").strip()
        
        for choice, model_type, _ in available_models:
            if user_choice == choice:
                print(f"✓ 已选择: {model_type}")
                return model_type
        
        print(f"无效选择，请输入 1-{len(available_models)} 之间的数字")


def get_train_ratio() -> float:
    """
    获取训练数据比例
    
    Returns:
        训练数据比例（0-1之间）
    """
    print("\n" + "=" * 60)
    print("配置数据划分比例")
    print("=" * 60)
    print("格式: [训练比例, 回测比例]，例如: 0.5,0.5 或 0.7,0.3")
    print("  训练比例: 用于训练模型的数据比例")
    print("  回测比例: 用于回测的数据比例")
    print("  两个比例之和应该等于1")
    
    while True:
        ratio_input = input("\n请输入数据划分比例 (默认: 0.5,0.5): ").strip()
        
        if not ratio_input:
            return 0.5
        
        try:
            parts = ratio_input.split(',')
            if len(parts) == 2:
                train_ratio = float(parts[0].strip())
                test_ratio = float(parts[1].strip())
                
                if 0 < train_ratio < 1 and 0 < test_ratio < 1:
                    if abs(train_ratio + test_ratio - 1.0) < 0.01:  # 允许小的浮点误差
                        print(f"✓ 训练比例: {train_ratio:.1%}, 回测比例: {test_ratio:.1%}")
                        return train_ratio
                    else:
                        print("错误: 两个比例之和必须等于1")
                else:
                    print("错误: 比例必须在0和1之间")
            else:
                print("错误: 请输入两个数字，用逗号分隔")
        except ValueError:
            print("错误: 请输入有效的数字")


def main():
    """主函数：支持机器学习市场状态检测的回测系统"""
    print("=" * 80)
    print("基于机器学习的市场状态检测回测系统")
    print("=" * 80)
    
    # 1. 选择是否使用市场状态检测
    print("\n是否使用机器学习市场状态检测？")
    print("  y: 使用（根据市场状态选择策略）")
    print("  n: 不使用（使用单一策略）")
    use_ml_input = input("请选择 (y/n, 默认y): ").strip().lower()
    use_ml = use_ml_input != 'n'
    
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
        
        if not has_trending:
            print("  注意: 未选择趋势策略，在趋势市场将不交易")
        if not has_ranging:
            print("  注意: 未选择震荡策略，在震荡市场将不交易")
    else:
        # 单策略模式
        available_strategies = list_available_strategies()
        if not available_strategies:
            print("未找到可用策略")
            return
        
        print("\n可用策略:")
        for i, strategy_name in enumerate(available_strategies, 1):
            print(f"  {i}. {strategy_name}")
        
        choice = input(f"请选择策略 (1-{len(available_strategies)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_strategies):
                strategy_name = available_strategies[idx]
                module_name, class_name = strategy_name.split('.')
                params = get_strategy_params(class_name)
                strategy = load_strategy_from_module(module_name, class_name, params)
                strategies = None
            else:
                print("无效选择")
                return
        except Exception as e:
            print(f"加载策略失败: {str(e)}")
            return
    
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
    else:
        # 单策略模式
        backtest = BacktestSystem(
            strategy=strategy,
            initial_capital=10000
        )
    
    # 5. 加载数据
    if os.path.exists(csv_file):
        print(f"\n从CSV文件加载数据: {csv_file}")
        backtest.load_data_from_csv(csv_file, symbol=symbol)
    else:
        print(f"\n文件不存在: {csv_file}")
        print("从币安API加载数据...")
        symbol_input = input("请输入交易对（如BTC/USDT）: ").strip() or 'BTC/USDT'
        backtest.load_data_from_binance(symbol_input, interval='1h', limit=2000)
    
    # 6. 训练模型（如果使用市场状态检测）
    if use_ml and backtest.market_detector:
        print("\n" + "=" * 60)
        print("训练市场状态检测模型")
        print("=" * 60)
        backtest.market_detector.train(backtest.data)
    
    # 7. 运行回测
    print("\n" + "=" * 60)
    print("开始回测")
    print("=" * 60)
    max_entries_input = input("最大加仓次数 (默认3): ").strip()
    max_entries = int(max_entries_input) if max_entries_input else 3
    
    backtest.run_backtest(max_entries=max_entries)
    
    # 8. 生成报告
    backtest.print_report()

    # 9. 绘制结果
    save_path = input("\n保存图表路径（直接回车使用默认）: ").strip()
    if not save_path:
        save_path = 'backtest_result_ML.png'
    backtest.plot_results(save_path=save_path)
    
    print(f"\n回测完成！结果已保存到: {save_path}")


if __name__ == "__main__":
    main()

