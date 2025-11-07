"""
测试 ccxt 库对币安期货账户 API 的支持情况
专门用于检查 ccxt 能调用哪些币安账户相关的 API 接口
"""

import ccxt
import time

# 配置
API_KEY = "SdTSZxmdf61CFsze3udgLRWq0aCaVyyFjsrYKMUOWIfMkm7q3sGRkzSk6QSbM5Qk"
SECRET_KEY = "9HZ04wgrKTy5kDPF5Kman4WSmS9D7YlTscPA7FtX2YLK7vTbpORFNB2jTABQY6HY"

def get_exchange():
    """获取币安期货交易所实例（测试网）"""
    testnet_base = 'https://testnet.binancefuture.com'
    
    # 创建 exchange 实例
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # 期货
            'loadMarketsOnStartup': False,  # 禁用自动加载市场
        }
    })
    
    # 手动设置测试网URL
    testnet_urls = {
        'api': {
            'public': f'{testnet_base}/fapi/v1',
            'private': f'{testnet_base}/fapi/v1',
            'fapiPublic': f'{testnet_base}/fapi/v1',
            'fapiPrivate': f'{testnet_base}/fapi/v1',
            'fapiPublicV2': f'{testnet_base}/fapi/v2',
            'fapiPrivateV2': f'{testnet_base}/fapi/v2',
            'sapi': f'{testnet_base}/fapi/v1',
        }
    }
    
    exchange.urls = testnet_urls
    if hasattr(exchange, '_urls'):
        exchange._urls = testnet_urls
    
    return exchange

def test_ccxt_binance_account_apis():
    """测试 ccxt 提供的标准统一方法（exchange.xxx()）对币安期货账户 API 的支持情况"""
    print("=" * 100)
    print("测试 ccxt 标准统一方法对币安期货账户 API 的支持情况")
    print("=" * 100)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {API_KEY[:20]}...")
    print(f"测试网: 币安期货测试网")
    print("=" * 100)
    print("\n注意：本测试专注于 ccxt 提供的标准统一方法（如 exchange.fetch_balance()）")
    print("而不是币安特有的方法（如 exchange.fapiPrivateGetAccount()）")
    print("=" * 100)
    
    exchange = get_exchange()
    
    # 禁用自动加载市场，避免触发不存在的接口
    exchange.options['loadMarketsOnStartup'] = False
    exchange.options['loadMarkets'] = False
    
    results = {
        'successful': [],
        'failed': [],
        'not_found': []
    }
    
    # ========== 第一部分：检查 exchange 对象有哪些标准方法 ==========
    print("\n【第一部分】检查 exchange 对象中的标准统一方法（fetch_xxx 开头）")
    print("-" * 100)
    
    all_methods = dir(exchange)
    
    # 查找所有以 fetch_ 开头的标准方法
    fetch_methods = [m for m in all_methods if m.startswith('fetch_') and not m.startswith('__')]
    
    # 查找与账户、持仓、订单、交易相关的标准方法
    account_related_keywords = ['balance', 'position', 'order', 'trade', 'funding', 'ticker', 'time']
    
    relevant_fetch_methods = []
    for method in fetch_methods:
        method_lower = method.lower()
        if any(keyword in method_lower for keyword in account_related_keywords):
            relevant_fetch_methods.append(method)
    
    print(f"找到 {len(fetch_methods)} 个 fetch_ 开头的方法")
    print(f"其中与账户相关的方法: {len(relevant_fetch_methods)} 个")
    print("\n所有 fetch_ 开头的方法列表:")
    for i, method in enumerate(sorted(fetch_methods), 1):
        marker = "★" if method in relevant_fetch_methods else " "
        print(f"  {marker} {i:2d}. {method}")
    
    # ========== 第二部分：关于 load_markets 的说明 ==========
    print("\n\n【第二部分】关于 load_markets 的说明")
    print("-" * 100)
    print("""
load_markets() 是 ccxt 库的一个重要方法，它的作用是：
1. 加载交易所的所有市场信息（交易对列表、交易规则、精度等）
2. 对于币安期货，它会调用 /fapi/v1/exchangeInfo 接口
3. 同时还会尝试调用 /fapi/v1/capital/config/getall 来获取货币配置

问题：
- 币安期货测试网不支持 /fapi/v1/capital/config/getall 接口
- 所以当 ccxt 尝试加载市场信息时会失败
- 这导致很多需要市场信息的方法（如 fetch_balance, fetch_orders 等）无法使用

解决方案：
- 尝试手动加载市场信息（只加载 exchangeInfo，跳过 currency config）
- 或者使用币安特有的方法（如 fapiPrivateGetAccount）来绕过这个问题
    """)
    
    # 尝试手动加载市场信息（只加载 exchangeInfo）
    print("\n尝试手动加载市场信息（只加载 exchangeInfo）...")
    try:
        # 尝试直接调用 exchangeInfo，不调用 currency config
        if hasattr(exchange, 'fapiPublicGetExchangeInfo'):
            markets_info = exchange.fapiPublicGetExchangeInfo()
            # 手动构建 markets 字典
            markets = {}
            for market in markets_info.get('symbols', []):
                symbol = market['symbol']
                markets[symbol] = {
                    'id': symbol,
                    'symbol': symbol,
                    'base': market.get('baseAsset', ''),
                    'quote': market.get('quoteAsset', ''),
                    'active': market.get('status') == 'TRADING',
                }
            exchange.markets = markets
            exchange.markets_by_id = {m['id']: m for m in markets.values()}
            print(f"  ✓ 成功加载 {len(markets)} 个交易对的市场信息")
            exchange.markets_loaded = True
        else:
            print("  ✗ 无法找到 fapiPublicGetExchangeInfo 方法")
    except Exception as e:
        print(f"  ✗ 加载市场信息失败: {str(e)[:80]}")
    
    # ========== 第三部分：测试 ccxt 标准统一方法 ==========
    print("\n\n【第三部分】测试 ccxt 标准统一方法（exchange.fetch_xxx()）")
    print("-" * 100)
    print("注意：如果已成功加载市场信息，这些方法应该可以正常工作\n")
    
    # 定义要测试的标准方法
    standard_methods_tests = [
        {
            'name': '获取账户余额',
            'method': 'fetch_balance',
            'params': {'type': 'future'},
            'description': '获取期货账户余额'
        },
        {
            'name': '获取持仓信息',
            'method': 'fetch_positions',
            'params': ['ETH/USDT:USDT'],  # 单个 symbol
            'description': '获取指定交易对的持仓信息'
        },
        {
            'name': '获取持仓信息（所有）',
            'method': 'fetch_positions',
            'params': [],  # 不传参数表示获取所有
            'description': '获取所有持仓信息'
        },
        {
            'name': '获取持仓风险',
            'method': 'fetch_positions_risk',
            'params': ['ETH/USDT:USDT'],  # 单个 symbol
            'description': '获取持仓风险信息'
        },
        {
            'name': '获取当前挂单',
            'method': 'fetch_open_orders',
            'params': ['ETH/USDT:USDT'],
            'description': '获取当前未完成的订单'
        },
        {
            'name': '获取所有订单',
            'method': 'fetch_orders',
            'params': ['ETH/USDT:USDT'],
            'params_dict': {'limit': 5},
            'description': '获取订单历史'
        },
        {
            'name': '获取我的交易记录',
            'method': 'fetch_my_trades',
            'params': ['ETH/USDT:USDT'],
            'params_dict': {'limit': 5},
            'description': '获取我的交易历史'
        },
        {
            'name': '获取资金费率',
            'method': 'fetch_funding_rate',
            'params': ['ETH/USDT:USDT'],
            'description': '获取资金费率'
        },
        {
            'name': '获取资金费率历史',
            'method': 'fetch_funding_rate_history',
            'params': ['ETH/USDT:USDT'],
            'params_dict': {'limit': 5},
            'description': '获取资金费率历史'
        },
        {
            'name': '获取价格',
            'method': 'fetch_ticker',
            'params': ['ETH/USDT:USDT'],
            'description': '获取交易对价格'
        },
        {
            'name': '获取服务器时间',
            'method': 'fetch_time',
            'params': [],
            'description': '获取交易所服务器时间'
        },
    ]
    
    print(f"测试 {len(standard_methods_tests)} 个标准方法...\n")
    
    for test in standard_methods_tests:
        test_name = test['name']
        method_name = test['method']
        description = test.get('description', '')
        
        if not hasattr(exchange, method_name):
            print(f"  ⊗ {test_name:25s} | {method_name:30s} | 方法不存在")
            results['not_found'].append({
                'name': test_name,
                'method': method_name,
                'description': description
            })
            continue
        
        try:
            method = getattr(exchange, method_name)
            
            # 准备参数
            if 'params_dict' in test:
                # 如果有 params_dict，使用它（通常用于 limit 等参数）
                if test.get('params'):
                    if isinstance(test['params'], dict):
                        # 如果是字典参数（如 {'type': 'future'}）
                        result = method(**test['params'], **test['params_dict'])
                    else:
                        result = method(*test['params'], **test['params_dict'])
                else:
                    result = method(**test['params_dict'])
            elif test.get('params'):
                if isinstance(test['params'], dict):
                    # 如果是字典参数（如 {'type': 'future'}）
                    result = method(**test['params'])
                elif isinstance(test['params'], list):
                    # 如果是列表参数
                    if len(test['params']) == 0:
                        result = method()
                    else:
                        result = method(*test['params'])
                else:
                    result = method(test['params'])
            else:
                result = method()
            
            # 成功
            result_type = type(result).__name__
            if isinstance(result, dict):
                result_info = f"字典，{len(result)} 个键"
                if len(result) > 0:
                    keys = list(result.keys())[:5]
                    result_info += f": {keys}"
            elif isinstance(result, list):
                result_info = f"列表，{len(result)} 个元素"
                if len(result) > 0:
                    result_info += f"，元素类型: {type(result[0]).__name__}"
            else:
                result_info = f"类型: {result_type}"
            
            print(f"  ✓ {test_name:25s} | {method_name:30s} | {result_info}")
            results['successful'].append({
                'name': test_name,
                'method': method_name,
                'description': description,
                'result_type': result_type,
                'result_info': result_info
            })
            
        except Exception as e:
            error_msg = str(e)
            # 如果是 load_markets 相关的错误，记录但标记为已知问题
            if 'load_markets' in error_msg.lower() or 'capital/config' in error_msg.lower():
                print(f"  ⚠ {test_name:25s} | {method_name:30s} | 需要 load_markets (已知问题)")
                results['failed'].append({
                    'name': test_name,
                    'method': method_name,
                    'description': description,
                    'error': '需要 load_markets (已知问题)',
                    'error_full': error_msg[:100]
                })
            else:
                print(f"  ✗ {test_name:25s} | {method_name:30s} | 错误: {error_msg[:50]}")
                results['failed'].append({
                    'name': test_name,
                    'method': method_name,
                    'description': description,
                    'error': error_msg[:100]
                })
    
    
    # ========== 第四部分：总结 ==========
    print("\n\n【第四部分】测试总结")
    print("=" * 100)
    
    print(f"\n✓ 成功的方法: {len(results['successful'])} 个")
    print("-" * 100)
    for item in results['successful']:
        desc = item.get('description', '')
        result_info = item.get('result_info', item.get('result_type', 'N/A'))
        print(f"  • {item['name']:25s} | {item['method']:30s} | {result_info}")
        if desc:
            print(f"    {desc}")
    
    print(f"\n✗ 失败的方法: {len(results['failed'])} 个")
    if results['failed']:
        print("-" * 100)
        for item in results['failed'][:15]:  # 显示前15个
            error = item.get('error', 'Unknown error')
            print(f"  • {item['name']:25s} | {item['method']:30s} | {error[:60]}")
        if len(results['failed']) > 15:
            print(f"  ... 还有 {len(results['failed']) - 15} 个失败的方法")
    
    print(f"\n⊗ 方法不存在: {len(results['not_found'])} 个")
    if results['not_found']:
        print("-" * 100)
        for item in results['not_found']:
            desc = item.get('description', '')
            print(f"  • {item['name']:25s} | {item['method']:30s}")
            if desc:
                print(f"    {desc}")
    
    # 统计信息
    total_tested = len(results['successful']) + len(results['failed']) + len(results['not_found'])
    success_rate = (len(results['successful']) / total_tested * 100) if total_tested > 0 else 0
    
    print("\n" + "=" * 100)
    print("【最终统计】")
    print("-" * 100)
    print(f"总计测试: {total_tested} 个标准方法")
    print(f"✓ 成功: {len(results['successful'])} 个 ({success_rate:.1f}%)")
    print(f"✗ 失败: {len(results['failed'])} 个")
    print(f"⊗ 不存在: {len(results['not_found'])} 个")
    print("\n【可用的 ccxt 标准方法列表】")
    print("-" * 100)
    for item in results['successful']:
        print(f"  exchange.{item['method']}()  # {item['name']}")
    print("=" * 100)
    
    return results

if __name__ == "__main__":
    try:
        results = test_ccxt_binance_account_apis()
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

