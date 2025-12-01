"""
网格策略（Grid Trading Strategy）
在价格区间内设置多个买卖网格，支持做多和做空网格
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from .base_strategy import BaseStrategy


class GridStrategy(BaseStrategy):
    """
    网格策略
    
    策略逻辑：
    1. 在价格区间内设置多个网格价位
    2. 做多网格：价格下跌到网格价位时买入，价格上涨到上一级网格时卖出
    3. 做空网格：价格上涨到网格价位时做空，价格下跌到下一级网格时平空
    4. 每个网格独立管理，可以同时持有多个网格的仓位
    5. 适合震荡市场，通过频繁交易赚取差价
    """
    
    def __init__(self, params: Dict):
        """
        初始化策略
        
        Args:
            params: 策略参数字典，包含：
                - grid_count: 网格数量（默认10）
                - grid_spacing_pct: 网格间距百分比（默认0.02，即2%）
                - grid_position_ratio: 每个网格的仓位比例（默认0.1，即10%资金）
                - upper_price: 网格上边界价格（如果为None，则使用历史最高价）
                - lower_price: 网格下边界价格（如果为None，则使用历史最低价）
                - use_dynamic_grid: 是否使用动态网格（根据价格波动调整，默认False）
                - stop_loss_pct: 止损百分比（默认0.10，即10%，超出网格范围时止损）
                - take_profit_pct: 止盈百分比（默认0.05，即5%，每个网格的盈利目标）
                - enable_short_grid: 是否启用做空网格（默认False）
                - short_grid_ratio: 做空网格的比例（默认0.5，即50%的网格做空，下半部分做多，上半部分做空）
        """
        super().__init__(params)
        
        # 策略参数
        self.grid_count = params.get('grid_count', 10)
        self.grid_spacing_pct = params.get('grid_spacing_pct', 0.02)
        self.grid_position_ratio = params.get('grid_position_ratio', 0.1)
        self.upper_price = params.get('upper_price', None)
        self.lower_price = params.get('lower_price', None)
        self.use_dynamic_grid = params.get('use_dynamic_grid', False)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.10)
        self.take_profit_pct = params.get('take_profit_pct', 0.05)
        self.enable_short_grid = params.get('enable_short_grid', False)
        self.short_grid_ratio = params.get('short_grid_ratio', 0.5)  # 做空网格的比例
        
        # 策略状态
        self.grid_levels = []  # 网格价位列表
        self.grid_types = []  # 网格类型列表：'long'（做多）或'short'（做空）
        self.grid_initialized = False  # 网格是否已初始化
        
        # 指标缓存
        self.price_history = []  # 价格历史（用于动态调整网格）
        
    def initialize(self, data: pd.DataFrame):
        """
        初始化策略，计算网格价位
        
        Args:
            data: 历史数据DataFrame，包含open, high, low, close, volume列
        """
        # 确定网格边界
        if self.upper_price is None:
            # 使用历史最高价
            self.upper_price = data['high'].max() * 1.05  # 增加5%缓冲
        
        if self.lower_price is None:
            # 使用历史最低价
            self.lower_price = data['low'].min() * 0.95  # 减少5%缓冲
        
        # 确保上边界大于下边界
        if self.upper_price <= self.lower_price:
            # 使用当前价格范围
            current_price = data['close'].iloc[-1]
            price_range = current_price * 0.2  # 20%的价格范围
            self.upper_price = current_price + price_range / 2
            self.lower_price = current_price - price_range / 2
        
        # 计算网格价位
        self._calculate_grid_levels()
        
        self.grid_initialized = True
        long_count = sum(1 for t in self.grid_types if t == 'long')
        short_count = sum(1 for t in self.grid_types if t == 'short')
        print(f"网格策略初始化完成：")
        print(f"  上边界: {self.upper_price:.4f}")
        print(f"  下边界: {self.lower_price:.4f}")
        print(f"  网格数量: {self.grid_count}")
        print(f"  网格间距: {self.grid_spacing_pct*100:.2f}%")
        print(f"  做多网格: {long_count}个")
        if self.enable_short_grid:
            print(f"  做空网格: {short_count}个")
    
    def _calculate_grid_levels(self):
        """
        计算网格价位和网格类型
        """
        self.grid_levels = []
        self.grid_types = []
        
        # 计算总价格范围
        price_range = self.upper_price - self.lower_price
        
        # 计算网格间距（价格）
        grid_spacing = price_range / (self.grid_count + 1)
        
        # 计算做空网格的数量
        if self.enable_short_grid:
            short_count = int(self.grid_count * self.short_grid_ratio)
            long_count = self.grid_count - short_count
        else:
            short_count = 0
            long_count = self.grid_count
        
        # 生成网格价位（从下往上）
        for i in range(1, self.grid_count + 1):
            level = self.lower_price + grid_spacing * i
            self.grid_levels.append(level)
            
            # 确定网格类型：下半部分做多，上半部分做空
            if self.enable_short_grid:
                if i <= long_count:
                    self.grid_types.append('long')  # 下半部分做多
                else:
                    self.grid_types.append('short')  # 上半部分做空
            else:
                self.grid_types.append('long')  # 全部做多
        
        # 按价格排序（同时保持grid_types的对应关系）
        sorted_indices = sorted(range(len(self.grid_levels)), key=lambda x: self.grid_levels[x])
        self.grid_levels = [self.grid_levels[i] for i in sorted_indices]
        self.grid_types = [self.grid_types[i] for i in sorted_indices]
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int, position_size: float = 0) -> Dict:
        """
        生成交易信号（完全依赖回测系统的持仓管理）
        
        Args:
            data: 历史数据DataFrame
            current_idx: 当前数据索引
            position_size: 当前持仓大小（对于网格策略，这个参数被忽略，直接从回测系统查询）
            
        Returns:
            信号字典（可能返回多个信号，但回测系统一次只能处理一个，所以返回优先级最高的）
        """
        if not self.grid_initialized:
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '网格未初始化'}
        
        # 如果回测引擎不可用，无法查询持仓
        if self.engine is None:
            return {'signal': 'hold', 'price': data.iloc[current_idx]['close'], 'reason': '回测引擎未设置'}
        
        current_bar = data.iloc[current_idx]
        current_price = current_bar['close']
        high_price = current_bar['high']
        low_price = current_bar['low']
        
        # 检查是否需要动态调整网格
        if self.use_dynamic_grid and current_idx > 100:
            # 使用最近100根K线的价格范围
            recent_data = data.iloc[max(0, current_idx-100):current_idx+1]
            new_upper = recent_data['high'].max() * 1.05
            new_lower = recent_data['low'].min() * 0.95
            
            # 如果价格范围变化超过20%，重新计算网格
            if abs(new_upper - self.upper_price) / self.upper_price > 0.2 or \
               abs(new_lower - self.lower_price) / self.lower_price > 0.2:
                self.upper_price = new_upper
                self.lower_price = new_lower
                self._calculate_grid_levels()
        
        signals = []
        
        # 遍历所有网格，检查是否需要交易
        # 完全依赖回测系统的持仓管理，通过position_id查询每个网格的持仓状态
        for i, grid_level in enumerate(self.grid_levels):
            position_id = f'grid_{grid_level:.4f}'
            grid_type = self.grid_types[i]  # 获取网格类型：'long'或'short'
            
            # 从回测系统查询该网格的持仓状态
            grid_pos = self.engine.get_position_by_id(position_id)
            has_position = grid_pos is not None and grid_pos['size'] != 0
            
            # ==================== 做多网格逻辑 ====================
            if grid_type == 'long':
                # 买入网格：价格下跌到网格价位时买入
                if not has_position:  # 该网格无持仓
                    # 检查是否触及买入网格（价格从上方跌到网格价位）
                    if low_price <= grid_level <= high_price:
                        # 检查是否是从上方跌下来的（避免在上涨时买入）
                        if current_idx > 0:
                            prev_price = data.iloc[current_idx - 1]['close']
                            if prev_price > grid_level:  # 从上方跌下来
                                signals.append({
                                    'signal': 'long',
                                    'price': grid_level,
                                    'reason': f'做多网格买入（网格{i+1}，价位{grid_level:.4f}）',
                                    'grid_level': grid_level,
                                    'grid_index': i,
                                    'position_id': position_id,
                                    '_grid_strategy': True
                                })
                
                # 卖出网格：价格上涨到网格价位时卖出（平多）
                elif grid_pos['size'] > 0:  # 该网格有多头持仓
                    entry_price = grid_pos['entry_price']
                    
                    # 检查止盈：价格上涨到上一级网格价位
                    if i < len(self.grid_levels) - 1:
                        upper_grid = self.grid_levels[i + 1]
                        if high_price >= upper_grid:
                            signals.append({
                                'signal': 'close_long',
                                'price': upper_grid,
                                'reason': f'做多网格止盈（网格{i+1}，价位{upper_grid:.4f}）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    else:
                        # 最高网格：使用止盈百分比
                        take_profit_price = entry_price * (1 + self.take_profit_pct)
                        if high_price >= take_profit_price:
                            signals.append({
                                'signal': 'close_long',
                                'price': take_profit_price,
                                'reason': f'做多网格止盈（网格{i+1}，盈利{self.take_profit_pct*100:.1f}%）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    
                    # 检查止损：价格跌破下一级网格价位
                    if i > 0:
                        lower_grid = self.grid_levels[i - 1]
                        if low_price <= lower_grid:
                            signals.append({
                                'signal': 'close_long',
                                'price': lower_grid,
                                'reason': f'做多网格止损（网格{i+1}，价位{lower_grid:.4f}）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    else:
                        # 最低网格：使用止损百分比
                        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                        if low_price <= stop_loss_price:
                            signals.append({
                                'signal': 'close_long',
                                'price': stop_loss_price,
                                'reason': f'做多网格止损（网格{i+1}，亏损{self.stop_loss_pct*100:.1f}%）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
            
            # ==================== 做空网格逻辑 ====================
            elif grid_type == 'short':
                # 做空网格：价格上涨到网格价位时做空
                if not has_position:  # 该网格无持仓
                    # 检查是否触及做空网格（价格从下方涨到网格价位）
                    if low_price <= grid_level <= high_price:
                        # 检查是否是从下方涨上来的（避免在下跌时做空）
                        if current_idx > 0:
                            prev_price = data.iloc[current_idx - 1]['close']
                            if prev_price < grid_level:  # 从下方涨上来
                                signals.append({
                                    'signal': 'short',
                                    'price': grid_level,
                                    'reason': f'做空网格开仓（网格{i+1}，价位{grid_level:.4f}）',
                                    'grid_level': grid_level,
                                    'grid_index': i,
                                    'position_id': position_id,
                                    '_grid_strategy': True
                                })
                
                # 平空网格：价格下跌到网格价位时平空
                elif grid_pos['size'] < 0:  # 该网格有空头持仓
                    entry_price = grid_pos['entry_price']
                    
                    # 检查止盈：价格下跌到下一级网格价位
                    if i > 0:
                        lower_grid = self.grid_levels[i - 1]
                        if low_price <= lower_grid:
                            signals.append({
                                'signal': 'close_short',
                                'price': lower_grid,
                                'reason': f'做空网格止盈（网格{i+1}，价位{lower_grid:.4f}）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    else:
                        # 最低网格：使用止盈百分比
                        take_profit_price = entry_price * (1 - self.take_profit_pct)
                        if low_price <= take_profit_price:
                            signals.append({
                                'signal': 'close_short',
                                'price': take_profit_price,
                                'reason': f'做空网格止盈（网格{i+1}，盈利{self.take_profit_pct*100:.1f}%）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    
                    # 检查止损：价格上涨到上一级网格价位
                    if i < len(self.grid_levels) - 1:
                        upper_grid = self.grid_levels[i + 1]
                        if high_price >= upper_grid:
                            signals.append({
                                'signal': 'close_short',
                                'price': upper_grid,
                                'reason': f'做空网格止损（网格{i+1}，价位{upper_grid:.4f}）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
                    else:
                        # 最高网格：使用止损百分比
                        stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                        if high_price >= stop_loss_price:
                            signals.append({
                                'signal': 'close_short',
                                'price': stop_loss_price,
                                'reason': f'做空网格止损（网格{i+1}，亏损{self.stop_loss_pct*100:.1f}%）',
                                'grid_level': grid_level,
                                'grid_index': i,
                                'position_id': position_id,
                                '_grid_strategy': True
                            })
        
        # 返回优先级最高的信号（优先处理平仓，然后处理开仓）
        if signals:
            # 优先处理平仓信号
            close_signals = [s for s in signals if s['signal'] in ['close_long', 'close_short']]
            if close_signals:
                return close_signals[0]
            # 然后处理开仓信号
            open_signals = [s for s in signals if s['signal'] in ['long', 'short']]
            if open_signals:
                return open_signals[0]
        
        return {'signal': 'hold', 'price': current_price, 'reason': '无信号'}
    
    def get_position_size(self, account_balance: float, entry_price: float, leverage: float = 1.0) -> float:
        """
        计算仓位大小（每个网格使用固定比例的资金）
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            leverage: 杠杆倍数
            
        Returns:
            仓位大小（数量）
        """
        if entry_price <= 0:
            return 0
        
        # 每个网格的仓位价值 = 账户余额 * 网格仓位比例 * 杠杆
        position_value = account_balance * self.grid_position_ratio * leverage
        
        # 仓位数量 = 仓位价值 / 入场价格
        position_size = position_value / entry_price
        
        return position_size
    
    
    def check_stop_loss(self, data: pd.DataFrame, current_idx: int,
                       position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查止损（已在generate_signals中实现）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            止损信号字典或None
        """
        # 止损逻辑已在generate_signals中实现
        return None
    
    def check_add_position(self, data: pd.DataFrame, current_idx: int,
                          position_size: float, last_entry_price: float) -> Optional[Dict]:
        """
        检查是否需要加仓（网格策略通常不加仓，而是通过多个网格独立管理）
        
        Args:
            data: 历史数据
            current_idx: 当前索引
            position_size: 当前持仓大小
            last_entry_price: 上次入场价格
            
        Returns:
            加仓信号字典或None（网格策略通常返回None）
        """
        # 网格策略通常不在单个网格中加仓，而是通过多个网格独立管理
        return None
    
    def update_trade_result(self, profit: float):
        """
        更新交易结果（网格策略可以用于统计或调整参数）
        
        Args:
            profit: 上次交易的盈亏（正数=盈利，负数=亏损）
        """
        # 网格策略可以在这里记录交易结果，用于后续优化
        # 目前不需要特殊处理，但方法必须存在以兼容回测系统
        pass

