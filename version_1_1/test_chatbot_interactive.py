#!/usr/bin/env python3
"""
交互式Chatbot测试工具
让用户可以直接测试不同的chatbot实现方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import json

# 导入不同的chatbot实现
from functions import extract_route_and_time, generate_file_path
from llm_chatbot import llm_chatbot
from openai_chatbot import enhanced_llm_chatbot, LLMConfig

def original_chatbot(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    原始的正则表达式方法（从implementation.py复制）
    """
    print("🔧 使用原始正则表达式方法")
    user_input = state.get("user_input", "")
    
    if not user_input:
        return {
            "file_path": "data/関越2024_cleaned.csv",
            "route": "未指定",
            "ts": "未指定"
        }
    
    print(f"📝 用户输入: {user_input}")
    
    # 提取路名和时间
    extracted_info = extract_route_and_time(user_input)
    route = extracted_info.get('route')
    time = extracted_info.get('time')
    
    print(f"🛣️  提取的路名: {route}")
    print(f"📅 提取的时间: {time}")
    
    # 生成文件路径
    file_path = generate_file_path(route, time)
    
    print(f"📁 生成的文件路径: {file_path}")
    
    return {
        "file_path": file_path,
        "route": route or "未指定",
        "ts": time or "未指定",
        "method": "正则表达式"
    }

def format_result(result: Dict[str, Any], method_name: str) -> str:
    """
    格式化输出结果
    """
    lines = [
        f"\n{'='*50}",
        f"🎯 {method_name} 解析结果",
        f"{'='*50}",
        f"📁 文件路径: {result.get('file_path', 'N/A')}",
        f"🛣️  路线名称: {result.get('route', 'N/A')}",
        f"📅 时间信息: {result.get('ts', 'N/A')}",
    ]
    
    # 添加额外信息（如果有）
    if 'analysis_type' in result:
        lines.append(f"🔍 分析类型: {result['analysis_type']}")
    
    if 'confidence' in result:
        lines.append(f"🎯 置信度: {result['confidence']:.2f}")
    
    if 'route_type' in result:
        lines.append(f"🏗️  路线类型: {result['route_type']}")
    
    if 'llm_provider' in result:
        lines.append(f"🤖 LLM后端: {result['llm_provider']}")
    
    lines.append(f"{'='*50}\n")
    
    return "\n".join(lines)

def test_single_input(user_input: str):
    """
    测试单个输入的所有方法
    """
    print(f"\n🧪 测试输入: '{user_input}'")
    print("─" * 60)
    
    state = {"user_input": user_input}
    
    # 1. 测试原始正则方法
    try:
        result1 = original_chatbot(state)
        print(format_result(result1, "原始正则表达式方法"))
    except Exception as e:
        print(f"❌ 原始方法出错: {e}\n")
    
    # 2. 测试LLM模拟方法
    try:
        result2 = llm_chatbot(state)
        print(format_result(result2, "LLM模拟方法"))
    except Exception as e:
        print(f"❌ LLM模拟方法出错: {e}\n")
    
    # 3. 测试增强LLM方法（降级模式）
    try:
        config = LLMConfig(provider="fallback")  # 使用降级方法
        result3 = enhanced_llm_chatbot(state, config)
        print(format_result(result3, "增强LLM方法（降级模式）"))
    except Exception as e:
        print(f"❌ 增强LLM方法出错: {e}\n")

def batch_test():
    """
    批量测试预定义的测试用例
    """
    test_cases = [
        "请分析関越高速公路2024年的交通情况",
        "我想看看関越道路在令和5年的数据",
        "分析东名高速2023年拥堵情况", 
        "查看中央道2022年的渋滞数据",
        "2021年首都高速的交通状况如何？",
        "関越道平成30年情况",
        "分析関越高速",
        "2024年数据",
        "関越",
        "看看東名的情况"
    ]
    
    print("\n🚀 开始批量测试...")
    print(f"共 {len(test_cases)} 个测试用例")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试用例 {i}/{len(test_cases)}")
        test_single_input(test_case)
        
        if i < len(test_cases):
            input("按 Enter 继续下一个测试用例...")

def interactive_test():
    """
    交互式测试模式
    """
    print("\n🎮 进入交互式测试模式")
    print("输入你的查询，输入 'quit' 或 'exit' 退出")
    print("输入 'batch' 运行批量测试")
    print("─" * 60)
    
    while True:
        try:
            user_input = input("\n💬 请输入查询: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 退出测试，再见！")
                break
            elif user_input.lower() == 'batch':
                batch_test()
                continue
            elif not user_input:
                print("⚠️  输入不能为空，请重试")
                continue
            
            test_single_input(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 收到中断信号，退出测试")
            break
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")

def show_usage():
    """
    显示使用说明
    """
    print("🤖 Chatbot测试工具")
    print("=" * 50)
    print("这个工具可以帮你测试不同的chatbot实现方法：")
    print()
    print("📋 测试方法包括：")
    print("1. 🔧 原始正则表达式方法")
    print("2. 🤖 LLM模拟方法") 
    print("3. 🚀 增强LLM方法")
    print()
    print("🎯 使用方式：")
    print("- 直接运行进入交互模式")
    print("- 输入自然语言查询测试解析效果")
    print("- 输入 'batch' 运行预定义测试用例")
    print("- 输入 'quit' 退出")
    print()
    print("💡 测试示例：")
    print("- '请分析関越高速公路2024年的交通情况'")
    print("- '我想看看関越道路在令和5年的数据'")
    print("- '分析东名高速2023年拥堵情况'")
    print("=" * 50)

def compare_methods():
    """
    方法对比测试
    """
    print("\n📊 方法对比测试")
    print("=" * 60)
    
    test_input = "请分析関越高速公路2024年的交通情况"
    state = {"user_input": test_input}
    
    print(f"📝 测试输入: {test_input}")
    print()
    
    methods = [
        ("🔧 正则表达式", lambda: original_chatbot(state)),
        ("🤖 LLM模拟", lambda: llm_chatbot(state)),
        ("🚀 增强LLM", lambda: enhanced_llm_chatbot(state, LLMConfig(provider="fallback")))
    ]
    
    results = []
    for name, method in methods:
        try:
            print(f"测试 {name}...")
            result = method()
            results.append((name, result))
            print(f"✅ {name} 完成")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results.append((name, {"error": str(e)}))
    
    print("\n📋 对比结果:")
    print("─" * 60)
    
    for name, result in results:
        if "error" in result:
            print(f"{name}: ❌ 错误 - {result['error']}")
        else:
            route = result.get('route', 'N/A')
            ts = result.get('ts', 'N/A') 
            confidence = result.get('confidence', 'N/A')
            print(f"{name}:")
            print(f"  路线: {route}")
            print(f"  时间: {ts}")
            if confidence != 'N/A':
                print(f"  置信度: {confidence}")
            print()

def main():
    """
    主函数
    """
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'batch':
            batch_test()
        elif command == 'compare':
            compare_methods()
        elif command == 'help':
            show_usage()
        else:
            print(f"❌ 未知命令: {command}")
            print("💡 可用命令: batch, compare, help")
    else:
        show_usage()
        interactive_test()

if __name__ == "__main__":
    main() 