#!/usr/bin/env python3
"""
完整工作流测试脚本
从chatbot节点到report节点的端到端测试
"""

import os
import sys
import json
sys.path.append(os.path.dirname(__file__))

# 导入所有节点函数
from implementation import chatbot, visualization, cluster, draw_hulls, report
from config import config

def test_complete_workflow(user_input: str):
    """测试完整的工作流程"""
    print("🚀 开始完整工作流测试")
    print("=" * 80)
    print(f"📝 用户输入: {user_input}")
    print("-" * 80)
    
    # 初始化状态
    state = {"user_input": user_input}
    
    try:
        # ===============================
        # 步骤1: Chatbot节点
        # ===============================
        print("\n1️⃣ Chatbot节点 - 解析用户输入")
        print("-" * 40)
        
        chatbot_result = chatbot(state)
        state.update(chatbot_result)
        
        print("✅ Chatbot节点完成")
        print(f"   主文件路径: {state.get('file_path', 'None')}")
        print(f"   多文件路径: {len(state.get('file_paths', []))} 个文件")
        if state.get('file_paths'):
            for i, fp in enumerate(state.get('file_paths', []), 1):
                print(f"      {i}. {fp}")
        print(f"   路线: {state.get('route', 'None')}")
        print(f"   时间: {state.get('ts', 'None')}")
        
        # ===============================
        # 步骤2: Visualization节点  
        # ===============================
        print("\n2️⃣ Visualization节点 - 生成三角形可视化")
        print("-" * 40)
        
        viz_result = visualization(state)
        state.update(viz_result)
        
        triangles = state.get('triangles', [])
        fig_path = state.get('fig_path', '')
        
        print("✅ Visualization节点完成")
        print(f"   生成三角形数量: {len(triangles)}")
        print(f"   图片路径: {fig_path}")
        print(f"   图片文件存在: {os.path.exists(fig_path) if fig_path else False}")
        
        if triangles:
            # 分析三角形来源
            source_stats = {}
            for triangle in triangles:
                source = triangle.get('source_file', 'unknown')
                source_stats[source] = source_stats.get(source, 0) + 1
            
            if len(source_stats) > 1:
                print("   📊 多文件来源分布:")
                for source, count in source_stats.items():
                    print(f"      {source}: {count} 个")
        
        if not triangles:
            print("❌ 没有生成三角形数据，无法继续")
            return False
        
        # ===============================
        # 步骤3: Cluster节点
        # ===============================
        print("\n3️⃣ Cluster节点 - 聚类分析")
        print("-" * 40)
        
        cluster_result = cluster(state)
        state.update(cluster_result)
        
        clusters = state.get('clusters', [])
        cluster_analysis = state.get('cluster_analysis', {})
        
        print("✅ Cluster节点完成")
        print(f"   发现聚类数量: {len(clusters)}")
        print(f"   聚类分析: {cluster_analysis.get('total_clusters', 0)} 个聚类覆盖 {cluster_analysis.get('total_triangles', 0)} 个三角形")
        
        if clusters:
            # 显示前几个聚类的详情
            for i, cluster_stat in enumerate(cluster_analysis.get('cluster_stats', [])[:3]):
                kp_range = cluster_stat['kp_range']
                time_range = cluster_stat['time_range']
                print(f"   聚类 {cluster_stat['cluster_id']}: {cluster_stat['size']} 个三角形")
                print(f"      KP范围: {kp_range['min']:.1f} - {kp_range['max']:.1f} km")
                print(f"      时间范围: {time_range['min']:.0f} - {time_range['max']:.0f} 分钟")
        
        # ===============================
        # 步骤4: Draw Hulls节点
        # ===============================
        print("\n4️⃣ Draw Hulls节点 - 绘制外包大三角形")
        print("-" * 40)
        
        hulls_result = draw_hulls(state)
        state.update(hulls_result)
        
        hulls = state.get('hulls', [])
        hull_fig_path = state.get('hull_fig_path', '')
        
        print("✅ Draw Hulls节点完成")
        print(f"   外包大三角形数量: {len(hulls)}")
        print(f"   带外包图的图片路径: {hull_fig_path}")
        print(f"   外包图文件存在: {os.path.exists(hull_fig_path) if hull_fig_path else False}")
        
        if hulls:
            print("   🔺 外包大三角形详情:")
            for i, hull in enumerate(hulls[:3]):  # 显示前3个
                kp_start, kp_end = hull['kp_range']
                print(f"      大三角形 {hull['cluster_id']}: 覆盖 {hull['cluster_size']} 个原始三角形")
                print(f"         KP范围: {kp_start:.1f} - {kp_end:.1f} km")
                print(f"         面积: {hull['area']:.1f} 平方单位")
        
        # ===============================
        # 步骤5: Report节点
        # ===============================
        print("\n5️⃣ Report节点 - 生成最终报告")
        print("-" * 40)
        
        report_result = report(state)
        state.update(report_result)
        
        final_report = state.get('final_report', '')
        
        print("✅ Report节点完成")
        print(f"   报告长度: {len(final_report)} 字符")
        
        # 显示报告的前几行
        if final_report:
            report_lines = final_report.split('\n')
            print("   📋 报告预览:")
            for line in report_lines[:10]:  # 显示前10行
                if line.strip():
                    print(f"      {line}")
            if len(report_lines) > 10:
                print("      ...")
        
        # ===============================
        # 工作流完成总结
        # ===============================
        print("\n🎉 完整工作流执行成功！")
        print("=" * 80)
        
        print("📊 最终结果统计:")
        print(f"   原始文件数量: {len(state.get('file_paths', []))}")
        print(f"   生成三角形数量: {len(state.get('triangles', []))}")
        print(f"   识别聚类数量: {len(state.get('clusters', []))}")
        print(f"   外包大三角形数量: {len(state.get('hulls', []))}")
        print(f"   生成图片数量: {2 if hull_fig_path and fig_path else (1 if fig_path else 0)}")
        
        print("\n📁 生成的文件:")
        if fig_path and os.path.exists(fig_path):
            print(f"   🖼️ 原始三角形可视化: {fig_path}")
        if hull_fig_path and os.path.exists(hull_fig_path):
            print(f"   🖼️ 带外包大三角形图: {hull_fig_path}")
        
        # 保存最终状态
        save_workflow_state(state, user_input)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 工作流执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_workflow_state(state, user_input):
    """保存工作流状态到文件"""
    try:
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 准备保存的状态（排除不可序列化的内容）
        save_state = {}
        for key, value in state.items():
            if key == 'triangles':
                # 只保存三角形的基本信息
                save_state[key] = [
                    {
                        'id': t.get('id'),
                        'center': t.get('center'),
                        'area': t.get('area'),
                        'kp_range': [t.get('kp_start'), t.get('kp_end')],
                        'source_file': t.get('source_file', 'unknown')
                    }
                    for t in value[:10]  # 只保存前10个作为示例
                ]
            elif key == 'hulls':
                # 保存外包大三角形信息
                save_state[key] = [
                    {
                        'cluster_id': h.get('cluster_id'),
                        'cluster_size': h.get('cluster_size'),
                        'kp_range': h.get('kp_range'),
                        'area': h.get('area')
                    }
                    for h in value
                ]
            elif isinstance(value, (str, int, float, bool, list)) and key != 'cluster_analysis':
                save_state[key] = value
        
        # 添加元信息
        save_state['_metadata'] = {
            'user_input': user_input,
            'workflow_version': 'version_1_1',
            'execution_time': pd.Timestamp.now().isoformat(),
            'total_triangles': len(state.get('triangles', [])),
            'total_clusters': len(state.get('clusters', [])),
            'total_hulls': len(state.get('hulls', []))
        }
        
        # 保存到JSON文件
        output_file = "output/workflow_state.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_state, f, ensure_ascii=False, indent=2)
        
        print(f"   📄 工作流状态已保存: {output_file}")
        
    except Exception as e:
        print(f"   ⚠️ 状态保存失败: {str(e)}")


def run_predefined_tests():
    """运行预定义的测试用例"""
    print("\n🧪 运行预定义测试用例")
    
    test_cases = [
        "4月23日関越道的交通情况",
        "分析関越高速公路2024年的数据", 
        "東北道春季交通拥堵情况",
        "请分析2023年中央道的拥堵模式"
    ]
    
    success_count = 0
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}/{len(test_cases)}")
        print("=" * 80)
        
        success = test_complete_workflow(test_input)
        
        if success:
            success_count += 1
            print(f"✅ 测试用例 {i} 成功")
        else:
            print(f"❌ 测试用例 {i} 失败")
        
        if i < len(test_cases):
            print(f"\n{'='*80}")
            print("等待下一个测试用例...")
            input("按回车键继续...")
    
    # 最终统计
    print(f"\n🏁 预定义测试完成！")
    print(f"   成功: {success_count}/{len(test_cases)} 个测试用例")
    print(f"   成功率: {success_count/len(test_cases)*100:.1f}%")
    
    if success_count == len(test_cases):
        print("🎉 所有预定义测试用例都成功执行！")
    else:
        print("⚠️ 部分预定义测试用例失败，请检查错误信息。")


def run_interactive_test():
    """运行交互式测试"""
    print("\n💬 交互式测试模式")
    print("您可以输入自定义查询来测试chatbot和完整工作流")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 60)
    
    test_count = 0
    success_count = 0
    
    while True:
        try:
            # 获取用户输入
            print(f"\n📝 请输入您的查询 (测试 #{test_count + 1}):")
            user_input = input(">>> ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                break
                
            # 检查空输入
            if not user_input:
                print("⚠️ 请输入有效的查询内容")
                continue
            
            test_count += 1
            print(f"\n🚀 开始测试您的输入: {user_input}")
            print("=" * 80)
            
            # 执行测试
            success = test_complete_workflow(user_input)
            
            if success:
                success_count += 1
                print(f"✅ 您的测试成功！")
            else:
                print(f"❌ 您的测试失败")
            
            # 询问是否继续
            print(f"\n{'='*80}")
            continue_test = input("继续测试？(y/n 或 回车继续): ").strip().lower()
            if continue_test in ['n', 'no', '不']:
                break
                
        except KeyboardInterrupt:
            print("\n\n⏹️ 用户中断测试")
            break
        except Exception as e:
            print(f"\n❌ 交互式测试出错: {str(e)}")
            continue
    
    # 交互式测试统计
    if test_count > 0:
        print(f"\n🏁 交互式测试完成！")
        print(f"   总测试次数: {test_count}")
        print(f"   成功次数: {success_count}")
        print(f"   成功率: {success_count/test_count*100:.1f}%")
    else:
        print("\n📝 没有进行任何测试")


def main():
    """主函数 - 提供交互式和预定义测试选择"""
    print("🏗️ 交通拥堵分析工作流 - 完整测试")
    print("配置状态:")
    print(f"   LLM可用: {config.is_llm_available()}")
    print(f"   OpenAI API密钥: {'已设置' if config.get_openai_api_key() else '未设置'}")
    print("")
    
    # 显示测试选项
    print("请选择测试模式:")
    print("1. 💬 交互式测试 (手动输入查询)")
    print("2. 🧪 预定义测试 (使用内置测试用例)")
    print("3. 🔄 两种模式都运行")
    print("0. ❌ 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1/2/3/0): ").strip()
            
            if choice == '1':
                run_interactive_test()
                break
            elif choice == '2':
                run_predefined_tests()
                break
            elif choice == '3':
                print("\n🔄 先运行交互式测试，然后运行预定义测试")
                run_interactive_test()
                
                print("\n" + "="*80)
                print("交互式测试完成，现在开始预定义测试...")
                input("按回车键继续到预定义测试...")
                
                run_predefined_tests()
                break
            elif choice == '0':
                print("👋 退出测试")
                break
            else:
                print("⚠️ 无效选择，请输入 1、2、3 或 0")
                
        except KeyboardInterrupt:
            print("\n\n👋 退出测试")
            break
        except Exception as e:
            print(f"❌ 输入处理错误: {str(e)}")
            continue


if __name__ == "__main__":
    # 添加缺失的导入
    import pandas as pd
    
    main() 