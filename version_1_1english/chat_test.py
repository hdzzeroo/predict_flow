#!/usr/bin/env python3
"""
交通数据查询助手 - 交互式聊天测试
简化版，专注于对话体验
"""

import os
import sys
from functions import enhanced_generate_file_paths
from config import config, setup_api_key

def chat_interface():
    """
    简化的聊天界面
    """
    print("🤖 交通数据查询助手 (真实LLM版)")
    print("=" * 40)
    
    # 显示当前配置
    llm_status = "真实LLM" if config.is_llm_available() else "本地解析"
    model_info = f"({config.openai_model})" if config.is_llm_available() else ""
    print(f"🔧 当前模式: {llm_status} {model_info}")
    
    print("\n💡 示例查询:")
    print("  • 4月23日関越的情况")
    print("  • 東北道5月数据")
    print("  • 2023年東名道")
    print("  • 令和5年中央道数据")
    print("─" * 40)
    print("输入 'quit' 退出 | 'config' 查看配置")
    print()
    
    while True:
        try:
            # 获取用户输入
            query = input("👤 您: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', '退出']:
                print("🤖 助手: 再见！")
                break
            
            if query.lower() in ['config', '配置']:
                config.print_config()
                continue
            
            # 处理查询
            print("🤖 助手: 正在分析您的查询...")
            
            try:
                # 使用LLM方法解析（传递API密钥）
                paths = enhanced_generate_file_paths(
                    query, 
                    use_llm=config.use_real_llm,
                    api_key=config.get_openai_api_key()
                )
                
                if paths:
                    method = "🤖 LLM解析" if config.is_llm_available() else "🔧 本地解析"
                    print(f"🤖 助手: ({method}) 找到了 {len(paths)} 个相关文件:")
                    for i, path in enumerate(paths, 1):
                        filename = os.path.basename(path)
                        exists = "✅" if os.path.exists(path) else "❌"
                        print(f"       {i}. {filename} {exists}")
                else:
                    print("🤖 助手: 抱歉，没有找到匹配的文件")
                    
            except Exception as e:
                print(f"🤖 助手: 处理出错了: {e}")
            
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n🤖 助手: 再见！")
            break
        except Exception as e:
            print(f"出错了: {e}")

def quick_test():
    """
    快速测试几个案例
    """
    test_queries = [
        "4月23日関越",
        "2023年東北道5月", 
        "東名道数据"
    ]
    
    print("⚡ 快速测试")
    print("=" * 30)
    
    for query in test_queries:
        print(f"\n测试: '{query}'")
        try:
            paths = enhanced_generate_file_paths(query, use_llm=True)
            print(f"结果: {len(paths)} 个文件")
            for path in paths:
                print(f"  📁 {os.path.basename(path)}")
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    print("🚀 交通数据查询助手启动")
    print("=" * 40)
    
    # 首先设置API密钥
    if not config.is_llm_available():
        print("⚠️ 未检测到OpenAI API密钥")
        setup_choice = input("是否要设置API密钥以使用真实LLM? (y/n, 默认y): ").strip().lower()
        if setup_choice != 'n':
            setup_api_key()
    
    print("\n选择模式:")
    print("1. 💬 聊天模式")
    print("2. ⚡ 快速测试")
    print("3. ⚙️ 配置管理")
    
    choice = input("请选择 (1/2/3, 默认1): ").strip() or "1"
    
    if choice == "1":
        chat_interface()
    elif choice == "2":
        quick_test()
    elif choice == "3":
        config.print_config()
        setup_api_key()
    else:
        print("无效选择，启动聊天模式")
        chat_interface() 