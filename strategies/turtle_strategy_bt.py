"""
海龟交易策略（backtrader版本）
基于唐奇安通道突破的交易策略
"""

import backtrader as bt
from typing import Dict, Optional


class TurtleStrategyBT(bt.Strategy):
    """
    海龟策略（backtrader版本）
    将原有的TurtleStrategy逻辑转换为backtrader的Strategy
    
    策略逻辑：
    1. 使用20日和55日唐奇安通道进行突破交易
    2. 支持加仓（最多3次，每0.5N加一次）
    3. 使用10日高低点进行移动止盈
    4. 使用均线交叉作为退出信号
    5. 可选的上次盈利过滤
    """
    
    params = (
        ('n_entries', 3),           # 最大加仓次数
        ('atr_length', 20),         # ATR周期
        ('bo_length', 20),          # 短周期突破（20日）
        ('fs_length', 55),          # 长周期突破（55日）
        ('te_length', 10),          # 移动止盈周期
        ('use_filter', False),      # 是否使用上次盈利过滤
        ('mas', 10),                # 短周期均线
        ('mal', 20),                # 长周期均线
        ('max_entries', 3),         # 最大加仓次数（与n_entries相同）
    )
    
    def __init__(self):
        # 策略状态
        self.last_trade_loss = True
        self.entry_count = 0
        self.last_entry_price = None
        
        # 临时状态（用于在订单被拒绝时回退）
        self._pending_entry_count = None
        self._pending_last_entry_price = None
        
        # 计算指标
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_length)
        
        # 唐奇安通道（20日）
        self.donchian_hi = bt.indicators.Highest(self.data.high, period=self.p.bo_length)
        self.donchian_lo = bt.indicators.Lowest(self.data.low, period=self.p.bo_length)
        
        # 长周期唐奇安通道（55日）
        self.fs_donchian_hi = bt.indicators.Highest(self.data.high, period=self.p.fs_length)
        self.fs_donchian_lo = bt.indicators.Lowest(self.data.low, period=self.p.fs_length)
        
        # 移动止盈（10日最高/最低）
        self.exit_highest = bt.indicators.Highest(self.data.high, period=self.p.te_length)
        self.exit_lowest = bt.indicators.Lowest(self.data.low, period=self.p.te_length)
        
        # 均线（用于退出）
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.mas)
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.mal)
        
        # 订单跟踪
        self.order = None
        self.entry_orders = []  # 记录所有入场订单
        
        # 交易记录（用于导出CSV）
        self.trades = []  # 记录所有完成的交易（开仓+平仓）
    
    def next(self):
        """每个bar执行一次"""
        # 跳过数据不足的情况
        if len(self.data) < max(self.p.fs_length, self.p.mal):
            return
        
        # 调试：记录每个bar的执行
        if len(self.data) % 100 == 0:  # 每100个bar打印一次
            print(f"Bar {len(self.data)}: position={self.position.size}, order={self.order}, cash={self.broker.get_cash():.2f}")
        
        # 获取当前价格和指标值
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # 获取前一根K线的指标值（避免未来函数）
        prev_idx = len(self.data) - 2
        if prev_idx < 0:
            return
            
        donchian_hi = self.donchian_hi[-1]  # 前一根K线的20日最高
        donchian_lo = self.donchian_lo[-1]  # 前一根K线的20日最低
        fs_donchian_hi = self.fs_donchian_hi[-1]  # 前一根K线的55日最高
        fs_donchian_lo = self.fs_donchian_lo[-1]  # 前一根K线的55日最低
        exit_highest = self.exit_highest[-1]  # 前一根K线的10日最高
        exit_lowest = self.exit_lowest[-1]  # 前一根K线的10日最低
        
        atr_value = self.atr[0]
        ma_short = self.ma_short[0]
        ma_long = self.ma_long[0]
        ma_short_prev = self.ma_short[-1]
        ma_long_prev = self.ma_long[-1]
        
        # 检查是否有未完成的订单
        if self.order:
            return
        
        # 获取当前持仓
        position_size = self.position.size
        
        # 1. 检查止损（移动止盈）
        if position_size != 0:
            if position_size > 0:  # 多头
                if current_low < exit_lowest:
                    # 多头止损：价格跌破10日最低点
                    self.order = self.close(price=exit_lowest)
                    self.entry_count = 0
                    return
            else:  # 空头
                if current_high > exit_highest:
                    # 空头止损：价格涨破10日最高点
                    self.order = self.close(price=exit_highest)
                    self.entry_count = 0
                    return
        
        # 2. 检查均线交叉退出
        if position_size != 0:
            # 检测死叉（短均线下穿长均线）
            crossunder = ma_short_prev > ma_long_prev and ma_short < ma_long
            # 检测金叉（短均线上穿长均线）
            crossover = ma_short_prev < ma_long_prev and ma_short > ma_long
            
            if position_size > 0 and crossunder:
                # 多头：死叉退出
                self.order = self.close()
                self.entry_count = 0
                return
            elif position_size < 0 and crossover:
                # 空头：金叉退出
                self.order = self.close()
                self.entry_count = 0
                return
        
        # 3. 检查加仓
        if position_size != 0 and self.entry_count < self.p.max_entries:
            if self.last_entry_price is not None and atr_value > 0:
                if position_size > 0:  # 多头加仓
                    add_price = self.last_entry_price + 0.5 * atr_value
                    if current_high >= add_price:
                        # 计算加仓数量
                        size = self._calculate_position_size(atr_value, add_price)
                        if size > 0:
                            # 保存当前状态，以便订单被拒绝时回退
                            self._pending_entry_count = self.entry_count + 1
                            self._pending_last_entry_price = add_price
                            self.order = self.buy(size=size, price=add_price)
                            # 注意：不在发出订单时立即更新 entry_count，等订单成功后再更新
                            return
                else:  # 空头加仓
                    add_price = self.last_entry_price - 0.5 * atr_value
                    if current_low <= add_price:
                        # 计算加仓数量
                        size = self._calculate_position_size(atr_value, add_price)
                        if size > 0:
                            # 保存当前状态，以便订单被拒绝时回退
                            self._pending_entry_count = self.entry_count + 1
                            self._pending_last_entry_price = add_price
                            self.order = self.sell(size=size, price=add_price)
                            # 注意：不在发出订单时立即更新 entry_count，等订单成功后再更新
                            return
        
        # 4. 检查入场信号（只在无持仓时）
        if position_size == 0:
            # 判断是否允许突破（过滤条件）
            allow_breakout = not self.p.use_filter or self.last_trade_loss
            
            # 20日突破
            long_entry = allow_breakout and current_high > donchian_hi
            short_entry = allow_breakout and current_low < donchian_lo
            
            # 55日突破（Failsafe）- 只在无持仓时触发（与原版策略一致）
            long_entry_fs = current_high > fs_donchian_hi
            short_entry_fs = current_low < fs_donchian_lo
            
            if long_entry:
                # 20日突破做多
                size = self._calculate_position_size(atr_value, donchian_hi)
                if size > 0:
                    # 保存当前状态，以便订单被拒绝时回退
                    self._pending_entry_count = 1
                    self._pending_last_entry_price = donchian_hi
                    self.order = self.buy(size=size, price=donchian_hi)
                    # 注意：不在发出订单时立即更新状态，等订单成功后再更新
            elif short_entry:
                # 20日突破做空
                size = self._calculate_position_size(atr_value, donchian_lo)
                if size > 0:
                    # 保存当前状态，以便订单被拒绝时回退
                    self._pending_entry_count = 1
                    self._pending_last_entry_price = donchian_lo
                    self.order = self.sell(size=size, price=donchian_lo)
                    # 注意：不在发出订单时立即更新状态，等订单成功后再更新
            elif long_entry_fs:
                # 55日突破做多（Failsafe）
                size = self._calculate_position_size(atr_value, fs_donchian_hi)
                if size > 0:
                    # 保存当前状态，以便订单被拒绝时回退
                    self._pending_entry_count = 1
                    self._pending_last_entry_price = fs_donchian_hi
                    self.order = self.buy(size=size, price=fs_donchian_hi)
                    # 注意：不在发出订单时立即更新状态，等订单成功后再更新
            elif short_entry_fs:
                # 55日突破做空（Failsafe）
                size = self._calculate_position_size(atr_value, fs_donchian_lo)
                if size > 0:
                    # 保存当前状态，以便订单被拒绝时回退
                    self._pending_last_entry_price = fs_donchian_lo
                    self._pending_entry_count = 1
                    self.order = self.sell(size=size, price=fs_donchian_lo)
                    # 注意：不在发出订单时立即更新状态，等订单成功后再更新
    
    def _calculate_position_size(self, atr_value, entry_price):
        """
        计算仓位大小（基于风险比例）
        这是策略层面的风险控制，与杠杆无关
        Broker会负责处理保证金计算
        
        Args:
            atr_value: ATR值（保留参数以兼容调用，但不使用）
            entry_price: 入场价格
            
        Returns:
            仓位大小（数量）
        """
        if entry_price <= 0:
            return 0
        
        # 获取账户权益
        account_equity = self.broker.getvalue()
        
        # 仓位大小 = 账户权益 / 入场价格
        # 使用账户权益的固定比例（可根据需要调整）
        position_size = account_equity / entry_price
        
        return position_size
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # 调试：打印订单状态
        print(f"订单通知: status={order.getstatusname()}, isbuy={order.isbuy()}, size={order.size if hasattr(order, 'size') else 'N/A'}, executed={order.executed.size if hasattr(order, 'executed') and order.executed else 'N/A'}")
        
        if order.status in [order.Completed]:
            # 检查是否是平仓订单
            # 平仓：买入订单但持仓减少，或卖出订单但持仓减少
            current_position = self.position.size
            
            if order.isbuy():
                # 买入订单
                if current_position == 0 and self.last_entry_price is not None:
                    # 这是平空订单（买入平空）
                    self.entry_count = 0
                    self.last_entry_price = None
                    self._pending_entry_count = None
                    self._pending_last_entry_price = None
                elif order.executed.size > 0:
                    # 开多或加多 - 订单成功，更新状态
                    if self._pending_entry_count is not None:
                        self.entry_count = self._pending_entry_count
                        self._pending_entry_count = None
                    if self._pending_last_entry_price is not None:
                        self.last_entry_price = self._pending_last_entry_price
                        self._pending_last_entry_price = None
                self.entry_orders.append({
                    'type': 'buy',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'date': self.data.datetime.date(0)
                })
            elif order.issell():
                # 卖出订单
                if order.executed.size < 0:
                    # 开空 - 订单成功，更新状态
                    if self._pending_entry_count is not None:
                        self.entry_count = self._pending_entry_count
                        self._pending_entry_count = None
                    if self._pending_last_entry_price is not None:
                        self.last_entry_price = self._pending_last_entry_price
                        self._pending_last_entry_price = None
                    self.entry_orders.append({
                        'type': 'sell',
                        'price': order.executed.price,
                        'size': abs(order.executed.size),
                        'date': self.data.datetime.date(0)
                    })
                elif order.executed.size > 0 and current_position == 0 and self.last_entry_price is not None:
                    # 这是平多订单（卖出平多）
                    self.entry_count = 0
                    self.last_entry_price = None
                    self._pending_entry_count = None
                    self._pending_last_entry_price = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单被拒绝/取消，需要清理保证金（如果有）
            order_id = id(order)
            if hasattr(self.broker, 'order_margins') and order_id in self.broker.order_margins:
                order_info = self.broker.order_margins[order_id]
                margin = order_info.get('margin', 0)
                commission = margin * self.broker.commission if hasattr(self.broker, 'commission') else 0
                # 退还保证金和手续费
                self.broker.cash += margin + commission
                # 清理订单记录
                del self.broker.order_margins[order_id]
            
            # 订单被拒绝，回退临时状态（不清除 entry_count 和 last_entry_price，因为它们可能已经正确）
            # 注意：只有在发出新订单时才会设置 _pending_*，所以这里只需要清除它们
            self._pending_entry_count = None
            self._pending_last_entry_price = None
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        # 调试：打印交易状态
        print(f"交易通知: isclosed={trade.isclosed}, size={trade.size}, pnl={trade.pnl if hasattr(trade, 'pnl') else 'N/A'}")
        
        if trade.isclosed:
            # 交易平仓，记录完整交易信息
            # 注意：backtrader的trade.pnl可能不包含我们的保证金逻辑
            # 我们需要从broker的_custom_positions中获取真实的盈亏
            # 但trade.pnl可以作为参考
            
            # 尝试从broker获取真实的盈亏（如果可用）
            real_pnl = trade.pnl  # 默认使用backtrader的pnl
            try:
                # 从broker的_custom_positions中计算真实的盈亏
                # 但这需要知道开仓价和平仓价，而trade对象可能没有这些信息
                # 所以暂时使用trade.pnl，但需要调整
                # 注意：trade.pnl可能已经包含了手续费，但不包含保证金逻辑
                real_pnl = trade.pnl
            except:
                real_pnl = trade.pnl
            
            pnl = real_pnl
            self.last_trade_loss = pnl < 0
            self.entry_count = 0
            self.last_entry_price = None
            
            # 记录交易信息（用于导出CSV）
            try:
                # 获取当前时间
                if hasattr(self.data.datetime, 'datetime'):
                    current_time = self.data.datetime.datetime(0)
                else:
                    current_time = self.data.datetime.date(0)
                
                # 计算平仓价格（从盈亏反推）
                # 对于多头：pnl = (平仓价 - 开仓价) * 数量
                # 对于空头：pnl = (开仓价 - 平仓价) * 数量
                entry_price = trade.price
                position_size = abs(trade.size)
                if trade.size > 0:  # 多头
                    exit_price = entry_price + (pnl / position_size) if position_size > 0 else entry_price
                else:  # 空头
                    exit_price = entry_price - (pnl / position_size) if position_size > 0 else entry_price
                
                # 计算盈亏百分比
                # 注意：这里使用开仓价值（包含杠杆）来计算百分比
                # 开仓价值 = 数量 * 开仓价
                # 但实际投入是保证金，所以百分比应该基于保证金
                # 为了与原版系统一致，我们使用开仓价值
                pnl_pct = (pnl / (entry_price * position_size) * 100) if entry_price * position_size > 0 else 0
                
                trade_record = {
                    '交易类型': 'long' if trade.size > 0 else 'short',
                    '开仓时间': current_time,
                    '平仓时间': current_time,  # 平仓时间就是当前时间
                    '开仓价格': entry_price,
                    '平仓价格': exit_price,
                    '数量': position_size,
                    '盈亏': pnl,
                    '盈亏百分比': pnl_pct,
                    '持仓时长': trade.barlen if hasattr(trade, 'barlen') else 'N/A',
                }
                self.trades.append(trade_record)
            except Exception as e:
                # 如果记录失败，至少记录基本信息
                trade_record = {
                    '交易类型': 'long' if trade.size > 0 else 'short',
                    '盈亏': pnl,
                }
                self.trades.append(trade_record)

