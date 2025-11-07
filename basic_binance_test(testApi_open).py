"""
最基础的币安API测试
只测试连接和认证
"""

import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode

# 配置
API_KEY = "SdTSZxmdf61CFsze3udgLRWq0aCaVyyFjsrYKMUOWIfMkm7q3sGRkzSk6QSbM5Qk"
SECRET_KEY = "9HZ04wgrKTy5kDPF5Kman4WSmS9D7YlTscPA7FtX2YLK7vTbpORFNB2jTABQY6HY"
BASE_URL = "https://testnet.binancefuture.com"

def create_signature(params):
    """创建API签名"""
    query_string = urlencode(params)
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_headers():
    """获取请求头"""
    return {'X-MBX-APIKEY': API_KEY}

def test_1_connection():
    """测试1: 基础连接"""
    print("测试1: 基础连接")
    try:
        response = requests.get(f"{BASE_URL}/fapi/v1/time", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"服务器时间: {data.get('serverTime')}")
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_2_price():
    """测试2: 获取价格"""
    print("\n测试2: 获取ETH价格")
    try:
        response = requests.get(f"{BASE_URL}/fapi/v1/ticker/price?symbol=ETHUSDT", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"ETH价格: {data.get('price')}")
            return True
        else:
            print(f" 响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_3_account():
    """测试3: 获取账户信息（需要认证）"""
    print("\n 测试3: 获取账户信息")
    try:
        # 准备参数
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        # 创建签名
        query_string = urlencode(params)
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        print(f"参数: {params}")
        print(f"签名: {signature[:20]}...")
        
        # 发送请求
        response = requests.get(f"{BASE_URL}/fapi/v2/account", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"账户信息获取成功")
            print(f"总资产: {data.get('totalWalletBalance', 'N/A')}")
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f" 错误: {e}")
        return False

def test_4_set_position_mode():
    """测试4: 设置持仓模式"""
    print("\n测试4: 设置持仓模式")
    try:
        # 设置持仓模式为单向持仓
        params = {
            'dualSidePosition': 'false',  # 单向持仓
            'timestamp': int(time.time() * 1000)
        }
        
        # 创建签名
        query_string = urlencode(params)
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        print(f"设置参数: {params}")
        
        # 发送设置请求
        response = requests.post(f"{BASE_URL}/fapi/v1/positionSide/dual", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        
        if response.status_code == 200:
            print("持仓模式设置成功")
            return True
        else:
            print("持仓模式设置失败")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        return False

def get_position_mode():
    """获取当前持仓模式"""
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/positionSide/dual", params=params, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('dualSidePosition', False)  # True=双向持仓, False=单向持仓
        return False
    except:
        return False

def test_5_simple_order():
    """测试5: 简单下单测试（自动适配单向/双向持仓模式）"""
    print("\n测试5: 简单下单测试")
    try:
        # 先获取ETH价格
        price_response = requests.get(f"{BASE_URL}/fapi/v1/ticker/price?symbol=ETHUSDT", timeout=10)
        if price_response.status_code != 200:
            print(" 无法获取ETH价格")
            return False
        
        eth_price = float(price_response.json()['price'])
        print(f"ETH价格: {eth_price}")
        
        # 计算数量 (500 USDT / 价格) - ETH/USDT精度要求3位小数
        quantity = round(500 / eth_price, 3)
        print(f"下单数量: {quantity} ETH")
        
        # 检测当前持仓模式
        is_dual_side = get_position_mode()
        position_mode = "双向持仓" if is_dual_side else "单向持仓"
        print(f"当前持仓模式: {position_mode}")
        
        # 准备下单参数
        params = {
            'symbol': 'ETHUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        # 如果是双向持仓模式，必须指定 positionSide
        if is_dual_side:
            params['positionSide'] = 'LONG'  # BUY + LONG = 开多
            print(f"双向持仓模式：添加 positionSide='LONG' (开多)")
        
        # 创建签名
        query_string = urlencode(params)
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': API_KEY,
            'Content-Type': 'application/json'
        }
        
        print(f"下单参数: {params}")
        
        # 发送下单请求 - 使用URL参数而不是JSON
        response = requests.post(f"{BASE_URL}/fapi/v1/order", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        
        if response.status_code == 200:
            print("下单成功！")
            return True
        else:
            print("下单失败")
            return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_position_risk():
    """测试6: 获取持仓风险信息（持仓方向、仓位大小、持仓币种等）"""
    print("\n测试6: 获取持仓风险信息")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v2/positionRisk", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            positions = response.json()
            print(f"持仓数量: {len(positions)}")
            
            # 只显示有持仓的币种
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            if active_positions:
                print("\n当前持仓详情:")
                print("-" * 100)
                print(f"{'币种':<12} {'持仓方向':<8} {'持仓数量':<15} {'持仓价值':<15} {'未实现盈亏':<15} {'杠杆':<8}")
                print("-" * 100)
                
                for pos in active_positions:
                    symbol = pos.get('symbol', 'N/A')
                    position_amt = float(pos.get('positionAmt', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    leverage = pos.get('leverage', 'N/A')
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    # 判断持仓方向
                    if position_amt > 0:
                        direction = "多头"
                    elif position_amt < 0:
                        direction = "空头"
                    else:
                        direction = "无持仓"
                    
                    # 计算持仓价值（绝对值）
                    position_value = abs(position_amt * mark_price)
                    
                    print(f"{symbol:<12} {direction:<8} {position_amt:<15.4f} {position_value:<15.2f} {unrealized_pnl:<15.2f} {leverage:<8}")
            else:
                print("当前无持仓")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_get_position_mode():
    """测试7: 查询当前持仓模式（单向/双向）"""
    print("\n测试7: 查询当前持仓模式")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/positionSide/dual", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            dual_side = data.get('dualSidePosition', False)
            mode = "双向持仓" if dual_side else "单向持仓"
            print(f"当前持仓模式: {mode}")
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_8_leverage_bracket():
    """测试8: 查询杠杆倍数信息"""
    print("\n测试8: 查询杠杆倍数信息")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/leverageBracket", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            brackets = response.json()
            print(f"交易对数量: {len(brackets)}")
            
            # 显示前5个交易对的杠杆信息
            print("\n部分交易对杠杆信息:")
            print("-" * 80)
            print(f"{'币种':<12} {'最大杠杆':<10} {'初始保证金率':<15} {'维持保证金率':<15}")
            print("-" * 80)
            
            for i, bracket in enumerate(brackets[:5]):
                symbol = bracket.get('symbol', 'N/A')
                brackets_list = bracket.get('brackets', [])
                if brackets_list:
                    max_leverage = brackets_list[-1].get('initialLeverage', 'N/A')
                    initial_margin = brackets_list[-1].get('initialMarginRate', 'N/A')
                    maint_margin = brackets_list[-1].get('maintMarginRate', 'N/A')
                    print(f"{symbol:<12} {max_leverage:<10} {initial_margin:<15} {maint_margin:<15}")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_9_account_balance():
    """测试9: 获取账户余额详情"""
    print("\n测试9: 获取账户余额详情")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v2/account", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n账户资产概览:")
            print("-" * 60)
            print(f"总资产: {data.get('totalWalletBalance', 'N/A')} USDT")
            print(f"可用余额: {data.get('availableBalance', 'N/A')} USDT")
            print(f"持仓保证金: {data.get('totalPositionInitialMargin', 'N/A')} USDT")
            print(f"订单保证金: {data.get('totalOpenOrderInitialMargin', 'N/A')} USDT")
            print(f"未实现盈亏: {data.get('totalUnrealizedProfit', 'N/A')} USDT")
            print(f"保证金余额: {data.get('totalMarginBalance', 'N/A')} USDT")
            print(f"账户权益: {data.get('totalCrossWalletBalance', 'N/A')} USDT")
            
            # 显示资产详情
            assets = data.get('assets', [])
            if assets:
                print("\n资产详情:")
                print("-" * 80)
                print(f"{'资产':<10} {'可用余额':<15} {'钱包余额':<15} {'未实现盈亏':<15}")
                print("-" * 80)
                for asset in assets:
                    asset_name = asset.get('asset', 'N/A')
                    available = float(asset.get('availableBalance', 0))
                    wallet_balance = float(asset.get('walletBalance', 0))
                    unrealized_pnl = float(asset.get('unrealizedProfit', 0))
                    
                    # 只显示有余额或未实现盈亏的资产
                    if available != 0 or wallet_balance != 0 or unrealized_pnl != 0:
                        print(f"{asset_name:<10} {available:<15.4f} {wallet_balance:<15.4f} {unrealized_pnl:<15.4f}")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_10_open_orders():
    """测试10: 查询当前挂单"""
    print("\n测试10: 查询当前挂单")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/openOrders", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            orders = response.json()
            print(f"当前挂单数量: {len(orders)}")
            
            if orders:
                print("\n挂单详情:")
                print("-" * 100)
                print(f"{'币种':<12} {'方向':<8} {'类型':<12} {'价格':<15} {'数量':<15} {'状态':<10}")
                print("-" * 100)
                
                for order in orders:
                    symbol = order.get('symbol', 'N/A')
                    side = order.get('side', 'N/A')
                    order_type = order.get('type', 'N/A')
                    price = float(order.get('price', 0))
                    orig_qty = float(order.get('origQty', 0))
                    status = order.get('status', 'N/A')
                    
                    print(f"{symbol:<12} {side:<8} {order_type:<12} {price:<15.4f} {orig_qty:<15.4f} {status:<10}")
            else:
                print("当前无挂单")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_11_funding_rate():
    """测试11: 查询资金费率"""
    print("\n测试11: 查询资金费率")
    try:
        # 查询当前资金费率（不需要签名）
        response = requests.get(f"{BASE_URL}/fapi/v1/premiumIndex?symbol=ETHUSDT", timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"币种: ETHUSDT")
            print(f"标记价格: {data.get('markPrice', 'N/A')}")
            print(f"资金费率: {data.get('lastFundingRate', 'N/A')} ({float(data.get('lastFundingRate', 0)) * 100:.4f}%)")
            print(f"下次资金费率时间: {data.get('nextFundingTime', 'N/A')}")
            print(f"资金费率间隔: 8小时")
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_12_trade_history():
    """测试12: 查询最近交易历史"""
    print("\n测试12: 查询最近交易历史")
    try:
        params = {
            'symbol': 'ETHUSDT',
            'limit': 5,  # 只查询最近5笔
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/userTrades", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            trades = response.json()
            print(f"ETHUSDT 最近交易记录数: {len(trades)}")
            
            if trades:
                print("\n最近交易记录:")
                print("-" * 100)
                print(f"{'时间':<20} {'方向':<8} {'价格':<15} {'数量':<15} {'盈亏':<15} {'手续费':<15}")
                print("-" * 100)
                
                for trade in trades[:5]:  # 只显示前5笔
                    trade_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(trade.get('time', 0) / 1000))
                    side = trade.get('side', 'N/A')
                    price = float(trade.get('price', 0))
                    qty = float(trade.get('qty', 0))
                    realized_pnl = float(trade.get('realizedPnl', 0))
                    commission = float(trade.get('commission', 0))
                    
                    print(f"{trade_time:<20} {side:<8} {price:<15.4f} {qty:<15.4f} {realized_pnl:<15.4f} {commission:<15.4f}")
            else:
                print("暂无交易记录")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_13_order_history():
    """测试13: 查询订单历史（所有订单）"""
    print("\n测试13: 查询订单历史")
    try:
        params = {
            'symbol': 'ETHUSDT',
            'limit': 5,  # 只查询最近5笔
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/allOrders", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            orders = response.json()
            print(f"ETHUSDT 订单历史数量: {len(orders)}")
            
            if orders:
                print("\n最近订单历史:")
                print("-" * 120)
                print(f"{'时间':<20} {'方向':<8} {'类型':<12} {'价格':<15} {'数量':<15} {'状态':<12} {'成交数量':<15}")
                print("-" * 120)
                
                for order in orders[:5]:  # 只显示前5笔
                    order_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(order.get('time', 0) / 1000))
                    side = order.get('side', 'N/A')
                    order_type = order.get('type', 'N/A')
                    price = float(order.get('price', 0))
                    orig_qty = float(order.get('origQty', 0))
                    executed_qty = float(order.get('executedQty', 0))
                    status = order.get('status', 'N/A')
                    
                    print(f"{order_time:<20} {side:<8} {order_type:<12} {price:<15.4f} {orig_qty:<15.4f} {status:<12} {executed_qty:<15.4f}")
            else:
                print("暂无订单历史")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_14_income_history():
    """测试14: 查询账户收入历史（资金费用、交易手续费、强平等）"""
    print("\n测试14: 查询账户收入历史")
    try:
        params = {
            'incomeType': 'TRANSFER',  # TRANSFER, COMMISSION, INSURANCE_CLEAR, FUNDING_FEE, REALIZED_PNL, COMMISSION_REBATE
            'limit': 10,
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/income", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            incomes = response.json()
            print(f"收入记录数量: {len(incomes)}")
            
            if incomes:
                print("\n最近收入记录:")
                print("-" * 100)
                print(f"{'时间':<20} {'收入类型':<20} {'资产':<10} {'金额':<15} {'备注':<30}")
                print("-" * 100)
                
                for income in incomes[:10]:
                    income_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(income.get('time', 0) / 1000))
                    income_type = income.get('incomeType', 'N/A')
                    asset = income.get('asset', 'N/A')
                    income_amount = float(income.get('income', 0))
                    info = income.get('info', 'N/A')
                    
                    print(f"{income_time:<20} {income_type:<20} {asset:<10} {income_amount:<15.4f} {str(info)[:30]:<30}")
            else:
                print("暂无收入记录")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_15_commission_rate():
    """测试15: 查询账户手续费率"""
    print("\n测试15: 查询账户手续费率")
    try:
        params = {
            'symbol': 'ETHUSDT',
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/commissionRate", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"币种: {data.get('symbol', 'N/A')}")
            print(f"做市商手续费率: {data.get('makerCommissionRate', 'N/A')} ({float(data.get('makerCommissionRate', 0)) * 100:.4f}%)")
            print(f"吃单手续费率: {data.get('takerCommissionRate', 'N/A')} ({float(data.get('takerCommissionRate', 0)) * 100:.4f}%)")
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_16_api_trading_status():
    """测试16: 查询API交易状态"""
    print("\n测试16: 查询API交易状态")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/apiTradingStatus", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"API交易状态: {data.get('data', {}).get('isLocked', 'N/A')}")
            print(f"触发条件: {data.get('data', {}).get('triggerCondition', 'N/A')}")
            
            # 显示IP限制
            ip_restrict = data.get('data', {}).get('IPRestrict', {})
            if ip_restrict:
                print(f"IP限制: {ip_restrict}")
            
            # 显示交易限制
            trading_restrict = data.get('data', {}).get('tradingRestrict', {})
            if trading_restrict:
                print(f"交易限制: {trading_restrict}")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_17_force_orders():
    """测试17: 查询强制平仓历史"""
    print("\n测试17: 查询强制平仓历史")
    try:
        params = {
            'limit': 5,
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/forceOrders", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            orders = response.json()
            print(f"强制平仓记录数量: {len(orders)}")
            
            if orders:
                print("\n强制平仓记录:")
                print("-" * 120)
                print(f"{'时间':<20} {'币种':<12} {'方向':<8} {'类型':<12} {'价格':<15} {'数量':<15} {'状态':<12}")
                print("-" * 120)
                
                for order in orders[:5]:
                    order_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(order.get('time', 0) / 1000))
                    symbol = order.get('symbol', 'N/A')
                    side = order.get('side', 'N/A')
                    order_type = order.get('type', 'N/A')
                    price = float(order.get('price', 0))
                    orig_qty = float(order.get('origQty', 0))
                    status = order.get('status', 'N/A')
                    
                    print(f"{order_time:<20} {symbol:<12} {side:<8} {order_type:<12} {price:<15.4f} {orig_qty:<15.4f} {status:<12}")
            else:
                print("暂无强制平仓记录")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_18_account_trades():
    """测试18: 查询账户所有交易（不限制币种）"""
    print("\n测试18: 查询账户所有交易")
    try:
        params = {
            'limit': 5,
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = create_signature(params)
        
        response = requests.get(f"{BASE_URL}/fapi/v1/userTrades", params=params, headers=get_headers(), timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            trades = response.json()
            print(f"账户交易记录数量: {len(trades)}")
            
            if trades:
                print("\n最近交易记录:")
                print("-" * 120)
                print(f"{'时间':<20} {'币种':<12} {'方向':<8} {'价格':<15} {'数量':<15} {'盈亏':<15} {'手续费':<15}")
                print("-" * 120)
                
                for trade in trades[:5]:
                    trade_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(trade.get('time', 0) / 1000))
                    symbol = trade.get('symbol', 'N/A')
                    side = trade.get('side', 'N/A')
                    price = float(trade.get('price', 0))
                    qty = float(trade.get('qty', 0))
                    realized_pnl = float(trade.get('realizedPnl', 0))
                    commission = float(trade.get('commission', 0))
                    
                    print(f"{trade_time:<20} {symbol:<12} {side:<8} {price:<15.4f} {qty:<15.4f} {realized_pnl:<15.4f} {commission:<15.4f}")
            else:
                print("暂无交易记录")
            
            return True
        else:
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    print("币安API基础测试")
    print("=" * 50)
    print(f" API Key: {API_KEY[:20]}...")
    print(f" 测试网: {BASE_URL}")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_1_connection,
        test_2_price,
        test_3_account,
        test_4_set_position_mode,
        test_5_simple_order,
        test_6_position_risk,        # 持仓风险信息（持仓方向、仓位大小、持仓币种）
        test_7_get_position_mode,   # 查询持仓模式
        test_8_leverage_bracket,     # 查询杠杆倍数
        test_9_account_balance,     # 账户余额详情
        test_10_open_orders,         # 当前挂单
        test_11_funding_rate,        # 资金费率
        test_12_trade_history,       # 交易历史（指定币种）
        test_13_order_history,       # 订单历史（所有订单）
        test_14_income_history,      # 账户收入历史（资金费用、手续费等）
        test_15_commission_rate,     # 账户手续费率
        # test_16_api_trading_status,  # API交易状态
        test_17_force_orders,        # 强制平仓历史
        test_18_account_trades       # 账户所有交易（不限制币种）
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("所有测试通过！币安API连接正常！")
    else:
        print(" 部分测试失败，请检查配置")

if __name__ == "__main__":
    main()
