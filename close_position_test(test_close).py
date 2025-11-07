"""
币安平仓测试
测试平仓功能
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
    """创建签名"""
    query_string = urlencode(params)
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

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

def test_2_get_positions():
    """测试2: 获取持仓信息"""
    print("\n测试2: 获取持仓信息")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        signature = create_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        response = requests.get(f"{BASE_URL}/fapi/v2/positionRisk", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            positions = response.json()
            print("持仓信息获取成功")
            
            # 查找ETH持仓
            eth_position = None
            for position in positions:
                if position['symbol'] == 'ETHUSDT' and float(position['positionAmt']) != 0:
                    eth_position = position
                    break
            
            if eth_position:
                print(f"找到ETH持仓:")
                print(f"  持仓数量: {eth_position['positionAmt']} ETH")
                print(f"  持仓价值: {eth_position['notional']} USDT")
                print(f"  未实现盈亏: {eth_position['unRealizedProfit']} USDT")
                print(f"  保证金: {eth_position['initialMargin']} USDT")
                print(f"  杠杆: {eth_position['leverage']}x")
                return eth_position
            else:
                print("当前没有ETH持仓")
                return None
        else:
            print(f"获取持仓失败: {response.text}")
            return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def test_3_close_position(position):
    """测试3: 平仓"""
    print("\n测试3: 平仓")
    try:
        if not position:
            print("没有持仓，无法平仓")
            return False
        
        position_amt = float(position['positionAmt'])
        if position_amt == 0:
            print("持仓数量为0，无需平仓")
            return True
        
        # 确定平仓方向
        if position_amt > 0:
            # 持有多头，需要卖出平仓
            side = 'SELL'
            quantity = abs(position_amt)
            print(f"持有多头 {position_amt} ETH，需要卖出平仓")
        else:
            # 持有空头，需要买入平仓
            side = 'BUY'
            quantity = abs(position_amt)
            print(f"持有空头 {abs(position_amt)} ETH，需要买入平仓")
        
        # 准备平仓参数
        params = {
            'symbol': 'ETHUSDT',
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'reduceOnly': 'true',  # 只减仓，不平仓
            'timestamp': int(time.time() * 1000)
        }
        
        # 创建签名
        signature = create_signature(params)
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        print(f"平仓参数: {params}")
        
        # 发送平仓请求
        response = requests.post(f"{BASE_URL}/fapi/v1/order", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        
        if response.status_code == 200:
            print("平仓成功！")
            order_data = response.json()
            print(f"订单ID: {order_data.get('orderId')}")
            print(f"订单状态: {order_data.get('status')}")
            print(f"执行数量: {order_data.get('executedQty')}")
            print(f"执行价格: {order_data.get('avgPrice')}")
            return True
        else:
            print("平仓失败")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_4_verify_close():
    """测试4: 验证平仓结果"""
    print("\n测试4: 验证平仓结果")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        signature = create_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        response = requests.get(f"{BASE_URL}/fapi/v2/positionRisk", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            positions = response.json()
            
            # 查找ETH持仓
            eth_position = None
            for position in positions:
                if position['symbol'] == 'ETHUSDT':
                    eth_position = position
                    break
            
            if eth_position:
                position_amt = float(eth_position['positionAmt'])
                if position_amt == 0:
                    print("平仓成功！ETH持仓已清零")
                    print(f"未实现盈亏: {eth_position['unRealizedProfit']} USDT")
                    return True
                else:
                    print(f"平仓未完成，仍有持仓: {position_amt} ETH")
                    return False
            else:
                print("未找到ETH持仓信息")
                return False
        else:
            print(f"获取持仓失败: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_5_get_account():
    """测试5: 获取账户信息"""
    print("\n测试5: 获取账户信息")
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        signature = create_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        response = requests.get(f"{BASE_URL}/fapi/v2/account", params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            account_data = response.json()
            print("账户信息获取成功")
            print(f"总资产: {account_data.get('totalWalletBalance', 'N/A')} USDT")
            print(f"可用余额: {account_data.get('availableBalance', 'N/A')} USDT")
            print(f"未实现盈亏: {account_data.get('totalUnrealizedProfit', 'N/A')} USDT")
            
            # 显示USDT余额
            for asset in account_data.get('assets', []):
                if asset['asset'] == 'USDT':
                    print(f"USDT余额: {asset['walletBalance']} USDT")
                    break
            
            return True
        else:
            print(f"获取账户信息失败: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    print("币安平仓测试")
    print("=" * 50)
    print(f"API Key: {API_KEY[:20]}...")
    print(f"测试网: {BASE_URL}")
    print("=" * 50)
    
    # 运行测试
    tests = [
        test_1_connection,
        test_2_get_positions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    # 获取持仓信息
    position = None
    try:
        params = {
            'timestamp': int(time.time() * 1000)
        }
        signature = create_signature(params)
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': API_KEY}
        response = requests.get(f"{BASE_URL}/fapi/v2/positionRisk", params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            positions = response.json()
            for pos in positions:
                if pos['symbol'] == 'ETHUSDT' and float(pos['positionAmt']) != 0:
                    position = pos
                    break
    except:
        pass
    
    if position:
        print(f"\n找到ETH持仓: {position['positionAmt']} ETH")
        print("开始平仓测试...")
        
        # 平仓测试
        if test_3_close_position(position):
            passed += 1
            print("-" * 30)
            
            # 等待一下让订单处理
            print("等待订单处理...")
            time.sleep(2)
            
            # 验证平仓结果
            if test_4_verify_close():
                passed += 1
            print("-" * 30)
            
            # 获取账户信息
            if test_5_get_account():
                passed += 1
        else:
            print("平仓失败")
    else:
        print("\n没有找到ETH持仓，无法进行平仓测试")
        print("请先运行 basic_binance_test.py 开仓")
    
    print(f"\n测试结果: {passed}/{total + 3} 通过")
    
    if passed >= total:
        print("平仓测试完成！")
    else:
        print("部分测试失败，请检查配置")

if __name__ == "__main__":
    main()
