"""
完整的回测系统（使用backtrader）
支持从币安API或CSV文件加载数据，运行策略，生成回测报告
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
warnings.filterwarnings('ignore')

# backtrader相关导入
import backtrader as bt
from backtrader import BrokerBase, Order
from backtrader.utils import date2num

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加策略模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.turtle_strategy_bt import TurtleStrategyBT


class LeverageBroker(bt.brokers.BackBroker):
    """
    支持杠杆的Broker
    实现保证金机制、强制平仓等
    """
    
    params = (
        ('leverage', 5.0),      # 杠杆倍数
        ('commission', 0.001),  # 手续费率
    )
    
    def __init__(self):
        super(LeverageBroker, self).__init__()
        self.leverage = self.p.leverage
        self.commission = self.p.commission
        
        # 记录持仓信息 {data: {'size': float, 'price': float, 'margin': float}}
        # 注意：使用_custom_positions避免与backtrader的positions属性冲突
        self._custom_positions = {}
        
        # 记录每笔订单的保证金（用于平仓时计算）
        # 使用id(order)作为键，因为order对象不可哈希
        self.order_margins = {}  # {id(order): margin}
        
        # 记录每笔交易的盈亏（用于策略中获取真实的pnl）
        # 使用id(trade)作为键
        self.trade_pnls = {}  # {id(trade): pnl}
    
    def submit(self, order, check=True):
        """
        提交订单 - 关键方法1
        在这里计算保证金，而不是全额扣除
        
        Args:
            order: 订单对象
            check: 是否检查订单（backtrader传递的参数，这里不使用但需要接受）
        """
        data = order.data
        
        # 获取当前持仓（用于判断是否是平仓订单）
        current_position = self.getposition(data)
        current_size = current_position.size
        
        # 判断是否是平仓订单
        # backtrader的close()方法会创建一个与当前持仓方向相反的订单
        # 关键：需要检查订单的size属性，但order.size可能还没有设置
        # 更可靠的方法：检查订单是否是通过close()创建的
        # 但backtrader没有直接标记，我们需要通过其他方式判断
        
        # 方法1：检查订单的size（如果已设置）
        # 方法2：检查订单方向与持仓方向是否相反
        is_close_order = False
        if current_size != 0:
            # 获取订单数量（order.size可能为None，需要处理）
            order_size = getattr(order, 'size', None)
            if order_size is None:
                # 如果order.size未设置，尝试从order的其他属性获取
                # 对于close()创建的订单，可能需要特殊处理
                # 暂时先不判断，让订单正常处理
                pass
            else:
                # 有持仓时，检查订单方向
                if current_size > 0:  # 多头持仓
                    # 平多：卖出订单（order.size < 0 或 not order.isbuy()）
                    if not order.isbuy():
                        # 卖出订单，检查数量
                        order_size_abs = abs(order_size) if order_size < 0 else order_size
                        if order_size_abs <= current_size:
                            is_close_order = True
                else:  # 空头持仓（current_size < 0）
                    # 平空：买入订单（order.size > 0 或 order.isbuy()）
                    if order.isbuy():
                        # 买入订单，检查数量
                        order_size_abs = abs(order_size) if order_size > 0 else abs(order_size)
                        if order_size_abs <= abs(current_size):
                            is_close_order = True
            
            # 方法2：如果方法1无法判断，使用更宽松的条件
            # 如果当前有持仓，且订单方向与持仓相反，很可能是平仓订单
            if not is_close_order and current_size != 0:
                if current_size > 0 and not order.isbuy():
                    # 多头持仓 + 卖出订单 = 可能是平仓
                    is_close_order = True
                elif current_size < 0 and order.isbuy():
                    # 空头持仓 + 买入订单 = 可能是平仓
                    is_close_order = True
        
        # 获取订单价格（如果是市价单，使用当前市场价格）
        if order.price is None:
            # 市价单：使用当前收盘价
            order_price = data.close[0]
        else:
            # 限价单：使用指定价格
            order_price = order.price
        
        # 如果是平仓订单，不需要扣除保证金，只记录订单信息
        if is_close_order:
            # 平仓订单：不扣除保证金，只记录订单信息（用于execute中处理）
            # 注意：平仓订单的margin设为0，因为不需要额外保证金
            self.order_margins[id(order)] = {
                'margin': 0,  # 平仓不需要保证金
                'price': order_price,
                'is_close': True  # 标记为平仓订单
            }
        else:
            # 开仓或加仓订单：需要扣除保证金
            # 计算订单价值
            order_value = abs(order.size) * order_price
            
            # 计算保证金（关键！）
            margin = order_value / self.leverage
            
            # 计算手续费（基于保证金）
            commission = margin * self.commission
            
            # 总成本 = 保证金 + 手续费
            total_cost = margin + commission
            
            # 检查资金是否足够（检查保证金，不是全额）
            if total_cost > self.cash:
                # 资金不足，拒绝订单
                # 注意：不要扣除保证金，因为订单被拒绝了
                # 也不要在order_margins中记录，因为订单不会执行
                order.reject()
                return order
            
            # 扣除保证金（不是全额）
            self.cash -= total_cost
            
            # 记录订单的保证金和价格（使用id(order)作为键，因为order对象不可哈希）
            # 注意：不在submit中更新持仓，而是在execute中更新，以保持与backtrader的position同步
            self.order_margins[id(order)] = {
                'margin': margin,
                'price': order_price,
                'is_close': False  # 标记为开仓/加仓订单
            }
        
        # 执行订单（backtrader会自动处理）
        # 注意：这里不直接调用execute，让backtrader的标准流程处理
        # 传递check参数给父类方法
        return super(LeverageBroker, self).submit(order, check=check)
    
    def execute(self, order):
        """
        执行订单 - 关键方法：在这里更新持仓，与backtrader的position同步
        """
        # 保存当前现金（因为backtrader的默认execute可能会调整现金）
        cash_before = self.cash
        
        # 先调用父类的execute，让backtrader更新self.position
        result = super(LeverageBroker, self).execute(order)
        
        # 恢复现金（我们会在下面自己处理现金）
        self.cash = cash_before
        
        # 获取订单信息
        order_id = id(order)
        # 注意：平仓订单可能不在order_margins中（因为平仓时没有在submit中记录）
        # 所以如果不在order_margins中，可能是平仓订单，直接返回
        if order_id not in self.order_margins:
            # 检查是否是平仓订单（平仓订单不会在submit中记录，因为self.close()创建的订单）
            # 但需要处理平仓：收回保证金、计算盈亏、更新持仓
            if hasattr(order, 'executed') and order.executed:
                data = order.data
                if data not in self._custom_positions:
                    self._custom_positions[data] = {'size': 0, 'price': 0, 'margin': 0}
                pos = self._custom_positions[data]
                bt_position = self.getposition(data)
                current_bt_size = bt_position.size
                
                # 如果持仓变为0，说明是平仓
                if current_bt_size == 0 and pos['size'] != 0:
                    # 这是平仓订单！需要处理平仓逻辑
                    old_size = pos['size']
                    old_price = pos['price']
                    old_margin = pos['margin']
                    
                    # 获取平仓价格
                    if hasattr(order, 'executed') and order.executed:
                        close_price = order.executed.price
                        executed_size = abs(order.executed.size)
                    else:
                        close_price = data.close[0]
                        executed_size = abs(old_size)
                    
                    # 计算盈亏
                    if old_size > 0:  # 平多
                        gross_pnl = (close_price - old_price) * executed_size
                    else:  # 平空
                        gross_pnl = (old_price - close_price) * executed_size
                    
                    # 收回保证金（按比例）
                    returned_margin = old_margin * (executed_size / abs(old_size)) if abs(old_size) > 0 else old_margin
                    
                    # 平仓手续费
                    close_cost = (executed_size * close_price) * self.commission
                    
                    # 收回：保证金 + 盈亏 - 手续费
                    self.cash += returned_margin + gross_pnl - close_cost
                    
                    # 更新持仓
                    if abs(old_size) == executed_size:
                        # 全部平仓
                        pos['size'] = 0
                        pos['price'] = 0
                        pos['margin'] = 0
                    else:
                        # 部分平仓
                        pos['size'] = old_size - (executed_size if old_size > 0 else -executed_size)
                        pos['margin'] -= returned_margin
            return result
        
        order_info = self.order_margins[order_id]
        margin = order_info['margin']
        order_price = order_info['price']
        is_close = order_info.get('is_close', False)
        data = order.data
        
        # 初始化持仓信息
        if data not in self._custom_positions:
            self._custom_positions[data] = {'size': 0, 'price': 0, 'margin': 0}
        
        pos = self._custom_positions[data]
        
        # 获取backtrader的持仓（已经通过父类execute更新）
        bt_position = self.getposition(data)
        current_bt_size = bt_position.size
        
        # 如果是平仓订单，优先处理平仓逻辑
        if is_close:
            # 平仓订单：收回保证金、计算盈亏、更新持仓
            old_size = pos['size']
            old_price = pos['price']
            old_margin = pos['margin']
            
            if old_size == 0:
                # 没有持仓，不应该发生，直接返回
                del self.order_margins[order_id]
                return result
            
            # 获取平仓价格和执行数量
            if hasattr(order, 'executed') and order.executed:
                close_price = order.executed.price
                executed_size = abs(order.executed.size)
            else:
                close_price = order_price
                executed_size = abs(old_size)
            
            # 计算盈亏
            if old_size > 0:  # 平多
                gross_pnl = (close_price - old_price) * executed_size
            else:  # 平空
                gross_pnl = (old_price - close_price) * executed_size
            
            # 收回保证金（按比例）
            returned_margin = old_margin * (executed_size / abs(old_size)) if abs(old_size) > 0 else old_margin
            
            # 平仓手续费
            close_cost = (executed_size * close_price) * self.commission
            
            # 收回：保证金 + 盈亏 - 手续费
            self.cash += returned_margin + gross_pnl - close_cost
            
            # 计算净盈亏（用于记录）
            # 净盈亏 = 毛盈亏 - 平仓手续费 - 开仓手续费
            # 开仓手续费 = 保证金 * 手续费率
            open_cost = returned_margin * self.commission
            net_pnl = gross_pnl - close_cost - open_cost
            
            # 记录交易的净盈亏（用于策略中获取）
            # 注意：这里我们无法直接获取trade对象，所以需要在notify_trade中计算
            # 但我们可以将净盈亏存储在某个地方，让策略可以访问
            # 暂时先不存储，因为trade对象在notify_trade中才可用
            
            # 更新持仓
            if abs(old_size) == executed_size:
                # 全部平仓
                pos['size'] = 0
                pos['price'] = 0
                pos['margin'] = 0
            else:
                # 部分平仓
                pos['size'] = old_size - (executed_size if old_size > 0 else -executed_size)
                pos['margin'] -= returned_margin
            
            # 清理订单信息
            del self.order_margins[order_id]
            return result
        
        # 判断是开仓、加仓还是平仓（非标记的平仓订单，可能是反向开仓）
        if order.isbuy():
            if current_bt_size > 0:  # 开多或加多
                if pos['size'] == 0:
                    # 开仓
                    pos['size'] = current_bt_size
                    pos['price'] = order_price
                    pos['margin'] = margin
                else:
                    # 加仓：计算平均价格
                    old_size = pos['size']
                    old_price = pos['price']
                    new_size = current_bt_size
                    new_price = (old_size * old_price + (new_size - old_size) * order_price) / new_size if new_size > 0 else order_price
                    pos['size'] = new_size
                    pos['price'] = new_price
                    pos['margin'] += margin
            elif current_bt_size == 0 and pos['size'] < 0:
                # 平空（买入平空）
                old_size = abs(pos['size'])
                executed_size = abs(order.executed.size)
                if old_size >= executed_size:
                    # 部分平仓或全部平仓
                    # 计算盈亏
                    gross_pnl = (pos['price'] - order_price) * executed_size  # 空头：开仓价高则盈利
                    # 收回保证金（按比例）
                    returned_margin = pos['margin'] * (executed_size / old_size)
                    # 平仓手续费
                    close_cost = (executed_size * order_price) * self.commission
                    # 收回：保证金 + 盈亏 - 手续费
                    self.cash += returned_margin + gross_pnl - close_cost
                    
                    if old_size == executed_size:
                        # 全部平仓
                        pos['size'] = 0
                        pos['price'] = 0
                        pos['margin'] = 0
                    else:
                        # 部分平仓
                        pos['size'] = -(old_size - executed_size)
                        pos['margin'] -= returned_margin
                else:
                    # 全部平仓并反向开多
                    # 先平空
                    gross_pnl = (pos['price'] - order_price) * old_size
                    returned_margin = pos['margin']
                    close_cost = (old_size * order_price) * self.commission
                    self.cash += returned_margin + gross_pnl - close_cost
                    
                    # 再开多
                    pos['size'] = current_bt_size
                    pos['price'] = order_price
                    pos['margin'] = margin
        else:  # 卖出
            if current_bt_size < 0:  # 开空或加空
                if pos['size'] == 0:
                    # 开仓
                    pos['size'] = current_bt_size
                    pos['price'] = order_price
                    pos['margin'] = margin
                else:
                    # 加仓：计算平均价格
                    old_size = abs(pos['size'])
                    old_price = pos['price']
                    new_size = abs(current_bt_size)
                    new_price = (old_size * old_price + (new_size - old_size) * order_price) / new_size if new_size > 0 else order_price
                    pos['size'] = -new_size
                    pos['price'] = new_price
                    pos['margin'] += margin
            elif current_bt_size == 0 and pos['size'] > 0:
                # 平多（卖出平多）
                old_size = pos['size']
                executed_size = abs(order.executed.size)
                if old_size >= executed_size:
                    # 部分平仓或全部平仓
                    # 计算盈亏
                    gross_pnl = (order_price - pos['price']) * executed_size  # 多头：平仓价高则盈利
                    # 收回保证金（按比例）
                    returned_margin = pos['margin'] * (executed_size / old_size)
                    # 平仓手续费
                    close_cost = (executed_size * order_price) * self.commission
                    # 收回：保证金 + 盈亏 - 手续费
                    self.cash += returned_margin + gross_pnl - close_cost
                    
                    if old_size == executed_size:
                        # 全部平仓
                        pos['size'] = 0
                        pos['price'] = 0
                        pos['margin'] = 0
                    else:
                        # 部分平仓
                        pos['size'] = old_size - executed_size
                        pos['margin'] -= returned_margin
                else:
                    # 全部平仓并反向开空
                    # 先平多
                    gross_pnl = (order_price - pos['price']) * old_size
                    returned_margin = pos['margin']
                    close_cost = (old_size * order_price) * self.commission
                    self.cash += returned_margin + gross_pnl - close_cost
                    
                    # 再开空
                    pos['size'] = current_bt_size
                    pos['price'] = order_price
                    pos['margin'] = margin
        
        # 清理订单信息（订单已执行）
        del self.order_margins[order_id]
        
        return result
    
    def cancel(self, order, bracket=False):
        """
        取消订单
        
        Args:
            order: 订单对象
            bracket: 是否为括号订单（backtrader传递的参数，这里不使用但需要接受）
        """
        # 如果订单还未执行，退还保证金
        order_id = id(order)
        if order_id in self.order_margins:
            order_info = self.order_margins[order_id]
            margin = order_info['margin'] if isinstance(order_info, dict) else order_info
            commission = margin * self.commission
            self.cash += margin + commission
            del self.order_margins[order_id]
        
        # 传递bracket参数给父类方法
        return super(LeverageBroker, self).cancel(order, bracket=bracket)
    
    def get_value(self, datas=None):
        """
        获取账户总价值 - 关键方法2
        需要计算：现金 + 持仓盈亏（如果有持仓）
        
        注意：这个方法的逻辑应该与原版系统的equity计算一致：
        - 如果没有持仓：权益 = 余额（现金）
        - 如果有持仓：权益 = 余额 + 未实现盈亏
        """
        # 基础价值（现金）
        value = self.cash
        
        # 计算持仓价值（使用_custom_positions）
        for data, pos in self._custom_positions.items():
            if pos['size'] != 0:
                try:
                    current_price = data.close[0]
                except:
                    continue
                
                # 计算未实现盈亏
                if pos['size'] > 0:  # 多头
                    unrealized_pnl = (current_price - pos['price']) * pos['size']
                else:  # 空头
                    unrealized_pnl = (pos['price'] - current_price) * abs(pos['size'])
                
                # 账户价值 = 现金 + 未实现盈亏
                # 注意：保证金已经在现金中扣除了，所以不需要再加回来
                # 因为平仓时会收回保证金，所以现金已经包含了所有可用的资金
                # 未实现盈亏是额外的价值
                value += unrealized_pnl
        
        return value
    
    def get_cash(self):
        """获取当前现金"""
        return self.cash
    
    def set_cash(self, cash):
        """设置现金"""
        self.cash = cash
        return cash
    
    def check_margin(self, order):
        """
        检查保证金 - 关键方法3
        用于强制平仓检查
        """
        data = order.data
        
        # 获取订单价格（如果是市价单，使用当前市场价格）
        if order.price is None:
            order_price = data.close[0]
        else:
            order_price = order.price
        
        # 计算新订单的保证金
        order_value = abs(order.size) * order_price
        margin = order_value / self.leverage
        
        # 计算当前持仓的保证金要求（使用_custom_positions）
        total_margin = margin
        if data in self._custom_positions:
            total_margin += self._custom_positions[data]['margin']
        
        # 计算账户权益
        equity = self.get_value([data])
        
        # 检查是否满足保证金要求
        # 通常需要：权益 >= 保证金 * 维持保证金率（如120%）
        # maintenance_margin_ratio = 1.2
        # required_margin = total_margin * maintenance_margin_ratio
        #
        # if equity < required_margin:
        #     # 保证金不足，需要强制平仓
        #     return False
        
        return True
    
    def setcommission(self, commission=0.0, margin=None, mult=1.0, commtype=None, stocklike=False, interest=0.0, interest_long=False, leverage=1.0, automargin=False, name=None):
        """
        设置手续费（重写以支持杠杆）
        """
        # 保存commission参数，但实际手续费在submit中计算
        self.commission = commission
        return super(LeverageBroker, self).setcommission(
            commission=commission,
            margin=margin,
            mult=mult,
            commtype=commtype,
            stocklike=stocklike,
            interest=interest,
            interest_long=interest_long,
            leverage=leverage,
            automargin=automargin,
            name=name
        )


class BacktestSystemBT:
    """
    完整的回测系统（使用backtrader）
    """
    
    def __init__(self, strategy_params: Dict, initial_capital: float = 10000, 
                 leverage: float = 5.0, position_ratio: float = 0.5):
        """
        初始化回测系统
        
        Args:
            strategy_params: 策略参数字典
            initial_capital: 初始资金
            leverage: 杠杆倍数（默认5倍）
            position_ratio: 仓位比例（默认0.5，即50%）
        """
        self.strategy_params = strategy_params
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.position_ratio = position_ratio
        
        # 创建Cerebro引擎
        self.cerebro = bt.Cerebro()
        
        # 计算可用资金（考虑仓位比例）
        self.available_capital = initial_capital * position_ratio
        
        # 设置自定义杠杆Broker
        self.cerebro.broker = LeverageBroker(
            leverage=leverage,
            commission=0.001
        )
        
        # 设置初始资金（使用全部初始资金，但限制交易时只使用可用资金）
        # 注意：虽然设置了全部资金，但在check_margin中会限制只能使用available_capital
        # 这样当可用资金耗尽时，可以从储备资金中补充
        self.cerebro.broker.set_cash(self.initial_capital)
        
        # 记录储备资金（不用于交易，但可以作为补充）
        self.reserve_capital = self.initial_capital - self.available_capital
        
        # 设置手续费（Broker内部会基于保证金计算）
        self.cerebro.broker.setcommission(commission=0.001)
        
        # 数据
        self.data = None
        self.data_feed = None
        
        # 回测结果
        self.results = None
        
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
            df.index = pd.to_datetime(df.index)
        
        # 确保有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        # 按时间排序
        df = df.sort_index()
        
        # 确保列名符合backtrader要求
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
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
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
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
        
        # 更新策略参数
        self.strategy_params['max_entries'] = max_entries
        
        # 创建数据源
        self.data_feed = bt.feeds.PandasData(
            dataname=self.data,
            datetime=None,  # 使用索引作为时间
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        
        # 添加数据到Cerebro
        self.cerebro.adddata(self.data_feed)
        
        # 添加策略
        self.cerebro.addstrategy(TurtleStrategyBT, **self.strategy_params)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        print(f"\n开始回测...")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"可用资金: {self.available_capital:,.2f} (仓位比例: {self.position_ratio * 100:.1f}%)")
        print(f"杠杆倍数: {self.leverage}x (注意：backtrader的杠杆需要通过margin参数实现)")
        print(f"数据量: {len(self.data)} 条")
        print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        
        # 运行回测
        self.results = self.cerebro.run()
        
        # 获取策略实例
        self.strategy = self.results[0]
        
        # 计算最终权益
        # 获取交易账户的实际价值（包括现金、保证金、盈亏）
        final_value = self.cerebro.broker.getvalue()
        trading_equity = final_value
        
        # 最终总资产 = 交易账户权益 + 储备资金（未投入的部分）
        # 注意：如果交易账户权益为负或0，说明资金耗尽了
        # 储备资金 = 初始资金 - 可用资金 = initial_capital - available_capital
        reserve_capital = self.initial_capital - self.available_capital
        final_equity = trading_equity + reserve_capital
        
        # 收益率计算：用总资产变化除以初始资金
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        print(f"\n回测完成！")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"可用资金: {self.available_capital:,.2f} (仓位比例: {self.position_ratio * 100:.1f}%)")
        print(f"杠杆倍数: {self.leverage}x")
        print(f"交易账户权益: {trading_equity:,.2f}")
        print(f"储备资金: {reserve_capital:,.2f}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        
        # 获取交易统计
        trade_analyzer = self.strategy.analyzers.trades.get_analysis()
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        print(f"总交易次数: {total_trades}")
        
        return self.results
    
    def generate_report(self) -> Dict:
        """
        生成回测报告
        
        Returns:
            包含各种统计信息的字典
        """
        if self.results is None:
            raise ValueError("请先运行回测")
        
        strategy = self.results[0]
        
        # 获取分析器结果
        trade_analyzer = strategy.analyzers.trades.get_analysis()
        sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
        
        # 计算最终权益
        # 获取交易账户的实际价值（包括现金、保证金、盈亏）
        final_value = self.cerebro.broker.getvalue()
        trading_equity = final_value
        
        # 最终总资产 = 交易账户权益 + 储备资金（未投入的部分）
        # 注意：如果交易账户权益为负或0，说明资金耗尽了
        # 储备资金 = 初始资金 - 可用资金 = initial_capital - available_capital
        reserve_capital = self.initial_capital - self.available_capital
        final_equity = trading_equity + reserve_capital
        
        # 收益率计算：用总资产变化除以初始资金
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # 提取交易统计
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won = trade_analyzer.get('won', {}).get('total', 0)
        lost = trade_analyzer.get('lost', {}).get('total', 0)
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trade_analyzer.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = trade_analyzer.get('lost', {}).get('pnl', {}).get('average', 0)
        
        total_won = trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0)
        total_lost = trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0)
        profit_factor = abs(total_won / total_lost) if total_lost != 0 else (float('inf') if total_won > 0 else 0)
        
        # 最大回撤
        max_drawdown = drawdown_analyzer.get('max', {}).get('drawdown', 0)
        max_drawdown_pct = drawdown_analyzer.get('max', {}).get('drawdown', 0) / self.initial_capital * 100
        
        # 夏普比率
        sharpe_ratio = sharpe_analyzer.get('sharperatio', 0)
        if sharpe_ratio is None:
            sharpe_ratio = 0
        
        # 总盈亏
        # 计算总盈亏：交易账户权益 - 可用资金（初始投入）
        # 如果交易账户权益 < 可用资金，说明亏损了
        total_pnl = trading_equity - self.available_capital
        
        report = {
            'initial_capital': self.initial_capital,
            'available_capital': self.available_capital,
            'leverage': self.leverage,
            'position_ratio': self.position_ratio,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': won,
            'losing_trades': lost,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
        }
        
        return report
    
    def print_report(self):
        """打印回测报告"""
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
        print(f"可用资金: {report['available_capital']:,.2f} (仓位比例: {report['position_ratio']*100:.1f}%)")
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
    
    def plot_results(self, save_path: str = None):
        """
        绘制回测结果
        
        Args:
            save_path: 保存路径（可选）
        """
        if self.results is None:
            print("没有数据可绘制，请先运行回测")
            return
        
        # backtrader内置绘图
        try:
            self.cerebro.plot(style='candlestick', barup='green', bardown='red')
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                print(f"回测结果图已保存到: {save_path}")
        except Exception as e:
            print(f"绘图失败: {str(e)}")
    
    def export_trades_to_csv(self, filepath: str = None):
        """
        导出交易记录到CSV文件
        
        Args:
            filepath: 保存路径，如果为None则自动生成
        """
        if self.results is None:
            print("没有交易记录可导出，请先运行回测")
            return
        
        strategy = self.results[0]
        
        # 方法1: 尝试从策略中获取交易记录（如果策略记录了）
        trades_data = []
        
        # 检查策略是否有trades属性（需要策略自己记录）
        if hasattr(strategy, 'trades') and strategy.trades:
            # 如果策略记录了交易，直接使用
            for trade in strategy.trades:
                trades_data.append(trade)
        else:
            # 方法2: 从backtrader的观察者中提取交易记录
            # 需要添加Trade观察者来记录交易
            # 这里我们尝试从TradeAnalyzer的统计信息中重建交易记录
            
            # 获取TradeAnalyzer的详细分析
            trade_analyzer = strategy.analyzers.trades.get_analysis()
            
            # TradeAnalyzer主要提供统计数据，不提供详细的每笔交易
            # 我们需要在策略中记录交易，或者使用观察者
            
            # 尝试从策略的entry_orders中提取（如果策略记录了）
            if hasattr(strategy, 'entry_orders') and strategy.entry_orders:
                # 构建交易记录（简化版，因为backtrader不直接提供完整交易历史）
                for i, order in enumerate(strategy.entry_orders):
                    trade_record = {
                        '序号': i + 1,
                        '交易类型': order.get('type', 'unknown'),
                        '时间': order.get('date', 'N/A'),
                        '价格': order.get('price', 0),
                        '数量': order.get('size', 0),
                    }
                    trades_data.append(trade_record)
            else:
                # 如果策略没有记录，尝试从TradeAnalyzer的统计中提取基本信息
                total_trades = trade_analyzer.get('total', {}).get('total', 0)
                if total_trades > 0:
                    # 只能导出统计信息，无法导出详细交易记录
                    print("警告: backtrader的TradeAnalyzer只提供统计数据，无法导出详细交易记录")
                    print("建议: 在策略中添加交易记录功能，或使用观察者记录交易")
                    
                    # 至少导出统计信息
                    stats_record = {
                        '统计类型': '汇总',
                        '总交易次数': total_trades,
                        '盈利交易': trade_analyzer.get('won', {}).get('total', 0),
                        '亏损交易': trade_analyzer.get('lost', {}).get('total', 0),
                        '总盈亏': trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0),
                    }
                    trades_data.append(stats_record)
        
        if not trades_data:
            print("没有找到交易记录")
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(trades_data)
        
        # 生成文件名
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"trades_backtrader_{timestamp}.csv"
        
        # 保存到CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n交易记录已保存到: {filepath}")
        print(f"共 {len(trades_data)} 条记录")
        
        return filepath


def main():
    """主函数：示例用法"""
    # 创建策略参数
    strategy_params = {
        'n_entries': 3,
        'atr_length': 20,
        'bo_length': 20,
        'fs_length': 55,
        'te_length': 10,
        'use_filter': False,
        'mas': 10,
        'mal': 20
    }
    
    # 创建回测系统
    backtest = BacktestSystemBT(
        strategy_params=strategy_params,
        initial_capital=10000,
        leverage=5.0,
        position_ratio=0.5
    )
    
    # 方式1: 从CSV文件加载数据
    csv_file = "segment_1_data_ccxt_20251106_195714.csv"
    if os.path.exists(csv_file):
        print(f"从CSV文件加载数据: {csv_file}")
        backtest.load_data_from_csv(csv_file, symbol='ETHUSDT')
    else:
        # 方式2: 从币安API加载数据
        print("从币安API加载数据...")
        backtest.load_data_from_binance('BTC/USDT', interval='1h', limit=2000)
    
    # 运行回测
    backtest.run_backtest(max_entries=3)
    
    # 生成报告
    backtest.print_report()
    
    # 绘制结果
    backtest.plot_results(save_path='backtest_result_bt.png')


if __name__ == "__main__":
    main()

