#!/usr/bin/env python3
"""
快速Chatbot测试脚本 - 简单易用版本
"""

from functions import extract_route_and_time, generate_file_path

def test_chatbot(user_input: str):
    """
    快速测试chatbot功能
    """
    print(f"\n🧪 测试输入: '{user_input}'")
    print("=" * 60)
    
    # 步骤1: 提取路名和时间
    print("📋 步骤1: 解析用户输入...")
    extracted_info = extract_route_and_time(user_input)
    route = extracted_info.get('route')
    time = extracted_info.get('time')
    
    print(f"  🛣️  路线名称: {route if route else '❌ 未识别'}")
    print(f"  📅 时间信息: {time if time else '❌ 未识别'}")
    
    # 步骤2: 生成文件路径
    print("\n📋 步骤2: 生成文件路径...")
    file_path = generate_file_path(route, time)
    print(f"  📁 文件路径: {file_path}")
    
    # 步骤3: 显示最终结果
    print("\n📋 步骤3: 最终解析结果")
    print("─" * 40)
    print(f"✅ 路线: {route or '関越高速道路 (默认)'}")
    print(f"✅ 时间: {time or '2024 (默认)'}")
    print(f"✅ 数据文件: {file_path}")
    print("=" * 60)

def main():
    """
    主测试函数
    """
    print("🤖 Chatbot快速测试工具")
    print("这个工具测试正则表达式方法解析用户输入的能力")
    print()
    
    # 预定义测试用例
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
    
    print("🎯 选择测试模式:")
    print("1. 快速演示 (运行预定义测试用例)")
    print("2. 交互输入 (手动输入测试)")
    print("3. 单个测试 (测试一个预定义用例)")
    
    try:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            # 快速演示模式
            print(f"\n🚀 运行 {len(test_cases)} 个预定义测试用例...")
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n📋 测试用例 {i}/{len(test_cases)}")
                test_chatbot(test_case)
                if i < len(test_cases):
                    input("按 Enter 继续...")
                    
        elif choice == "2":
            # 交互输入模式
            print("\n🎮 交互输入模式 (输入 'quit' 退出)")
            while True:
                user_input = input("\n💬 请输入查询: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 退出测试")
                    break
                if user_input:
                    test_chatbot(user_input)
                else:
                    print("⚠️  输入不能为空")
                    
        elif choice == "3":
            # 单个测试模式
            print("\n📋 选择一个测试用例:")
            for i, case in enumerate(test_cases, 1):
                print(f"{i}. {case}")
            
            try:
                idx = int(input(f"\n选择测试用例 (1-{len(test_cases)}): ")) - 1
                if 0 <= idx < len(test_cases):
                    test_chatbot(test_cases[idx])
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入数字")
                
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n👋 收到中断信号，退出测试")
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")

if __name__ == "__main__":
    main() 