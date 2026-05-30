#!/usr/bin/env python3
"""
Complete workflow test script
End-to-end testing from chatbot node to report node
"""

import os
import sys
import json
sys.path.append(os.path.dirname(__file__))

# Import all node functions
from implementation import chatbot, visualization, analyze_with_llm, report
from config import config

def test_complete_workflow(user_input: str):
    """Test the complete workflow"""
    print("🚀 Starting complete workflow test")
    print("=" * 80)
    print(f"📝 User input: {user_input}")
    print("-" * 80)
    
    # Initialize state
    state = {"user_input": user_input}
    
    try:
        # ===============================
        # Step 1: Chatbot node
        # ===============================
        print("\n1️⃣ Chatbot node - Parse user input")
        print("-" * 40)
        
        chatbot_result = chatbot(state)
        state.update(chatbot_result)
        
        print("✅ Chatbot node completed")
        print(f"   Main file path: {state.get('file_path', 'None')}")
        print(f"   Multiple file paths: {len(state.get('file_paths', []))} files")
        if state.get('file_paths'):
            for i, fp in enumerate(state.get('file_paths', []), 1):
                print(f"      {i}. {fp}")
        print(f"   Route: {state.get('route', 'None')}")
        print(f"   Time: {state.get('ts', 'None')}")
        
        # ===============================
        # Step 2: Visualization node  
        # ===============================
        print("\n2️⃣ Visualization node - Generate triangle visualization")
        print("-" * 40)
        
        viz_result = visualization(state)
        state.update(viz_result)
        
        triangles = state.get('triangles', [])
        fig_path = state.get('fig_path', '')
        
        print("✅ Visualization node completed")
        print(f"   Generated triangles count: {len(triangles)}")
        print(f"   Image path: {fig_path}")
        print(f"   Image file exists: {os.path.exists(fig_path) if fig_path else False}")
        
        if triangles:
            # Analyze triangle sources
            source_stats = {}
            for triangle in triangles:
                source = triangle.get('source_file', 'unknown')
                source_stats[source] = source_stats.get(source, 0) + 1
            
            if len(source_stats) > 1:
                print("   📊 Multi-file source distribution:")
                for source, count in source_stats.items():
                    print(f"      {source}: {count} triangles")
        
        if not triangles:
            print("❌ No triangle data generated, cannot continue")
            return False
        
        # ===============================
        # Step 3: Analyze with LLM node (替代cluster和draw_hulls)
        # ===============================
        print("\n3️⃣ Analyze with LLM node - LLM-based hotspot analysis")
        print("-" * 40)

        llm_result = analyze_with_llm(state)
        state.update(llm_result)

        llm_analysis = state.get('llm_analysis', {})

        print("✅ LLM analysis node completed")

        # 统计各方向的热点数量
        total_hotspots = 0
        for direction, analysis in llm_analysis.items():
            hotspots = analysis.get('hotspots', [])
            total_hotspots += len(hotspots)
            summary = analysis.get('summary', {})
            confidence = summary.get('analysis_confidence', 0)

            print(f"   {direction} direction: {len(hotspots)} hotspots (confidence: {confidence:.2f})")

            # 显示前3个热点
            for i, hotspot in enumerate(hotspots[:3]):
                kp_range = hotspot['kp_range']
                time_range = hotspot['time_range']
                print(f"      Hotspot {i+1}: KP {kp_range[0]:.1f}-{kp_range[1]:.1f}km, "
                      f"frequency: {hotspot['frequency']}, severity: {hotspot['severity']}")

        print(f"   Total hotspots across all directions: {total_hotspots}")

        # ===============================
        # Step 4: Report node
        # ===============================
        print("\n4️⃣ Report node - Generate final report")
        print("-" * 40)
        
        report_result = report(state)
        state.update(report_result)
        
        final_report = state.get('final_report', '')
        
        print("✅ Report node completed")
        print(f"   Report length: {len(final_report)} characters")
        
        # Show first few lines of the report
        if final_report:
            report_lines = final_report.split('\n')
            print("   📋 Report preview:")
            for line in report_lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"      {line}")
            if len(report_lines) > 10:
                print("      ...")
        
        # ===============================
        # Workflow completion summary
        # ===============================
        print("\n🎉 Complete workflow executed successfully!")
        print("=" * 80)
        
        print("📊 Final result statistics:")
        print(f"   Original file count: {len(state.get('file_paths', []))}")
        print(f"   Generated triangles count: {len(state.get('triangles', []))}")
        print(f"   LLM-identified hotspots count: {total_hotspots}")

        # 显示CSV输出路径
        csv_output_path = state.get('csv_output_path', '')
        if csv_output_path:
            print(f"   CSV prediction output: {csv_output_path}")

        print("\n📁 Generated files:")
        if fig_path and os.path.exists(fig_path):
            print(f"   🖼️ Triangle visualization: {fig_path}")

        # 显示所有图片路径
        fig_paths = state.get('fig_paths', [])
        if fig_paths and len(fig_paths) > 1:
            for i, path in enumerate(fig_paths[1:], 2):
                if os.path.exists(path):
                    print(f"   🖼️ Triangle visualization {i}: {path}")

        if csv_output_path and os.path.exists(csv_output_path):
            print(f"   📄 CSV prediction: {csv_output_path}")
        
        # Save final state
        save_workflow_state(state, user_input)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_workflow_state(state, user_input):
    """Save workflow state to file (包含完整的三角形数据和LLM分析结果)"""
    try:
        # Create output directory
        os.makedirs("output", exist_ok=True)

        # Prepare state for saving (exclude non-serializable content)
        save_state = {}
        for key, value in state.items():
            if key == 'triangles':
                # 🔥 保存所有三角形的完整信息（不再限制为前10个）
                save_state[key] = [
                    {
                        'id': t.get('id'),
                        'shape_type': t.get('shape_type', 'triangle'),
                        'vertices': t.get('vertices', []),  # 包含完整顶点坐标
                        'center': t.get('center'),
                        'area': t.get('area'),
                        'width': t.get('width'),
                        'height': t.get('height'),
                        'kp_start': t.get('kp_start'),
                        'kp_end': t.get('kp_end'),
                        'kp_range': [t.get('kp_start'), t.get('kp_end')],
                        'time_start': t.get('time_start'),
                        'time_end': t.get('time_end'),
                        'time_peak': t.get('time_peak'),
                        'source_file': t.get('source_file', 'unknown')
                    }
                    for t in value  # 保存所有三角形，不再截断
                ]
            elif key == 'direction_data':
                # 🔥 保存按方向分组的完整数据
                save_state[key] = {}
                for direction, data in value.items():
                    save_state[key][direction] = {
                        'triangles': [
                            {
                                'id': t.get('id'),
                                'shape_type': t.get('shape_type', 'triangle'),
                                'vertices': t.get('vertices', []),
                                'center': t.get('center'),
                                'area': t.get('area'),
                                'kp_start': t.get('kp_start'),
                                'kp_end': t.get('kp_end'),
                                'time_start': t.get('time_start'),
                                'time_end': t.get('time_end'),
                                'time_peak': t.get('time_peak'),
                                'source_file': t.get('source_file', 'unknown')
                            }
                            for t in data.get('triangles', [])
                        ],
                        'fig_path': data.get('fig_path', ''),
                        'triangle_count': len(data.get('triangles', []))
                    }
            elif key == 'llm_analysis':
                # 🔥 保存LLM分析结果
                save_state[key] = value  # LLM分析结果已经是可序列化的dict
            elif key == 'hulls':
                # Save convex hull triangle information
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

        # Add metadata
        llm_analysis = state.get('llm_analysis', {})
        total_hotspots = sum(len(analysis.get('hotspots', [])) for analysis in llm_analysis.values())

        save_state['_metadata'] = {
            'user_input': user_input,
            'workflow_version': 'version_1_3',
            'execution_time': pd.Timestamp.now().isoformat(),
            'total_triangles': len(state.get('triangles', [])),
            'total_clusters': len(state.get('clusters', [])),
            'total_hulls': len(state.get('hulls', [])),
            'total_hotspots': total_hotspots,  # 新增：LLM识别的热点总数
            'directions_analyzed': list(state.get('direction_data', {}).keys())  # 新增：分析的方向
        }

        # Save to JSON file
        output_file = "output/workflow_state.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_state, f, ensure_ascii=False, indent=2)

        print(f"   📄 Workflow state saved: {output_file}")
        print(f"      - Total triangles saved: {len(state.get('triangles', []))}")
        print(f"      - Directions saved: {list(state.get('direction_data', {}).keys())}")
        print(f"      - LLM hotspots saved: {total_hotspots}")

    except Exception as e:
        print(f"   ⚠️ State saving failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_predefined_tests():
    """Run predefined test cases"""
    print("\n🧪 Running predefined test cases")
    
    test_cases = [
        "April 23rd Kan-Etsu Expressway traffic conditions",
        "Analyze Kan-Etsu Expressway 2024 data", 
        "Tohoku Expressway spring traffic congestion situation",
        "Please analyze 2023 Chuo Expressway congestion patterns"
    ]
    
    success_count = 0
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 Test case {i}/{len(test_cases)}")
        print("=" * 80)
        
        success = test_complete_workflow(test_input)
        
        if success:
            success_count += 1
            print(f"✅ Test case {i} succeeded")
        else:
            print(f"❌ Test case {i} failed")
        
        if i < len(test_cases):
            print(f"\n{'='*80}")
            print("Waiting for next test case...")
            input("Press Enter to continue...")
    
    # Final statistics
    print(f"\n🏁 Predefined tests completed!")
    print(f"   Success: {success_count}/{len(test_cases)} test cases")
    print(f"   Success rate: {success_count/len(test_cases)*100:.1f}%")
    
    if success_count == len(test_cases):
        print("🎉 All predefined test cases executed successfully!")
    else:
        print("⚠️ Some predefined test cases failed, please check error messages.")


def run_interactive_test():
    """Run interactive test"""
    print("\n💬 Interactive test mode")
    print("You can input custom queries to test chatbot and complete workflow")
    print("Enter 'quit' or 'exit' to exit")
    print("-" * 60)
    
    test_count = 0
    success_count = 0
    
    while True:
        try:
            # Get user input
            print(f"\n📝 Please enter your query (test #{test_count + 1}):")
            user_input = input(">>> ").strip()
            
            # Check exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            # Check empty input
            if not user_input:
                print("⚠️ Please enter valid query content")
                continue
            
            test_count += 1
            print(f"\n🚀 Starting test with your input: {user_input}")
            print("=" * 80)
            
            # Execute test
            success = test_complete_workflow(user_input)
            
            if success:
                success_count += 1
                print(f"✅ Your test succeeded!")
            else:
                print(f"❌ Your test failed")
            
            # Ask whether to continue
            print(f"\n{'='*80}")
            continue_test = input("Continue testing? (y/n or Enter to continue): ").strip().lower()
            if continue_test in ['n', 'no']:
                break
                
        except KeyboardInterrupt:
            print("\n\n⏹️ User interrupted test")
            break
        except Exception as e:
            print(f"\n❌ Interactive test error: {str(e)}")
            continue
    
    # Interactive test statistics
    if test_count > 0:
        print(f"\n🏁 Interactive test completed!")
        print(f"   Total tests: {test_count}")
        print(f"   Successful tests: {success_count}")
        print(f"   Success rate: {success_count/test_count*100:.1f}%")
    else:
        print("\n📝 No tests were conducted")


def main():
    """Main function - provide interactive and predefined test selection"""
    print("🏗️ Traffic Congestion Analysis Workflow - Complete Test")
    print("Configuration Status:")
    print(f"   LLM Available: {config.is_llm_available()}")
    print(f"   OpenAI API Key: {'Set' if config.get_openai_api_key() else 'Not Set'}")
    print("")
    
    # Display test options
    print("Please select test mode:")
    print("1. 💬 Interactive Test (Manual Query Input)")
    print("2. 🧪 Predefined Test (Using Built-in Test Cases)")
    print("3. 🔄 Run Both Modes")
    print("0. ❌ Exit")
    
    while True:
        try:
            choice = input("\nPlease select (1/2/3/0): ").strip()
            
            if choice == '1':
                run_interactive_test()
                break
            elif choice == '2':
                run_predefined_tests()
                break
            elif choice == '3':
                print("\n🔄 First run interactive test, then run predefined test")
                run_interactive_test()
                
                print("\n" + "="*80)
                print("Interactive test completed, now starting predefined test...")
                input("Press Enter to continue to predefined test...")
                
                run_predefined_tests()
                break
            elif choice == '0':
                print("👋 Exit test")
                break
            else:
                print("⚠️ Invalid selection, please enter 1, 2, 3 or 0")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exit test")
            break
        except Exception as e:
            print(f"❌ Input processing error: {str(e)}")
            continue


if __name__ == "__main__":
    # Add missing import
    import pandas as pd
    
    main() 