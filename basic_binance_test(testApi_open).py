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
API_KEY = "dxubGWbnYJ5QzMdaOwBksMBn7rT6nr2h6JBXAtRiKCrFMe4gVEXyyyEEYjPmSloi"
SECRET_KEY = "xDnOhW5D4dRUVUqKqWigcgOqmGrWg4byxJISSL4eUi0FkRy6o1cP8jelLebXsb75"
BASE_URL = "https://testnet.binancefuture.com"

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

def test_5_simple_order():
    """测试5: 简单下单测试"""
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
        
        # 准备下单参数
        params = {
            'symbol': 'ETHUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': quantity,
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
        test_5_simple_order
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print(" 所有测试通过！币安API连接正常！")
    else:
        print("部分测试失败，请检查配置")

if __name__ == "__main__":
    main()
