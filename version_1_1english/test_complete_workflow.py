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
from implementation import chatbot, visualization, cluster, draw_hulls, report
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
        # Step 3: Cluster node
        # ===============================
        print("\n3️⃣ Cluster node - Clustering analysis")
        print("-" * 40)
        
        cluster_result = cluster(state)
        state.update(cluster_result)
        
        clusters = state.get('clusters', [])
        cluster_analysis = state.get('cluster_analysis', {})
        
        print("✅ Cluster node completed")
        print(f"   Discovered clusters count: {len(clusters)}")
        print(f"   Cluster analysis: {cluster_analysis.get('total_clusters', 0)} clusters covering {cluster_analysis.get('total_triangles', 0)} triangles")
        
        if clusters:
            # Show details of the first few clusters
            for i, cluster_stat in enumerate(cluster_analysis.get('cluster_stats', [])[:3]):
                kp_range = cluster_stat['kp_range']
                time_range = cluster_stat['time_range']
                print(f"   Cluster {cluster_stat['cluster_id']}: {cluster_stat['size']} triangles")
                print(f"      KP range: {kp_range['min']:.1f} - {kp_range['max']:.1f} km")
                print(f"      Time range: {time_range['min']:.0f} - {time_range['max']:.0f} minutes")
        
        # ===============================
        # Step 4: Draw Hulls node
        # ===============================
        print("\n4️⃣ Draw Hulls node - Draw convex hull triangles")
        print("-" * 40)
        
        hulls_result = draw_hulls(state)
        state.update(hulls_result)
        
        hulls = state.get('hulls', [])
        hull_fig_path = state.get('hull_fig_path', '')
        
        print("✅ Draw Hulls node completed")
        print(f"   Convex hull triangles count: {len(hulls)}")
        print(f"   Image with hulls path: {hull_fig_path}")
        print(f"   Hull image file exists: {os.path.exists(hull_fig_path) if hull_fig_path else False}")
        
        if hulls:
            print("   🔺 Convex hull triangle details:")
            for i, hull in enumerate(hulls[:3]):  # Show first 3
                kp_start, kp_end = hull['kp_range']
                print(f"      Hull triangle {hull['cluster_id']}: covers {hull['cluster_size']} original triangles")
                print(f"         KP range: {kp_start:.1f} - {kp_end:.1f} km")
                print(f"         Area: {hull['area']:.1f} square units")
        
        # ===============================
        # Step 5: Report node
        # ===============================
        print("\n5️⃣ Report node - Generate final report")
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
        print(f"   Identified clusters count: {len(state.get('clusters', []))}")
        print(f"   Convex hull triangles count: {len(state.get('hulls', []))}")
        print(f"   Generated images count: {2 if hull_fig_path and fig_path else (1 if fig_path else 0)}")
        
        print("\n📁 Generated files:")
        if fig_path and os.path.exists(fig_path):
            print(f"   🖼️ Original triangle visualization: {fig_path}")
        if hull_fig_path and os.path.exists(hull_fig_path):
            print(f"   🖼️ Image with convex hull triangles: {hull_fig_path}")
        
        # Save final state
        save_workflow_state(state, user_input)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_workflow_state(state, user_input):
    """Save workflow state to file"""
    try:
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Prepare state for saving (exclude non-serializable content)
        save_state = {}
        for key, value in state.items():
            if key == 'triangles':
                # Only save basic triangle information
                save_state[key] = [
                    {
                        'id': t.get('id'),
                        'center': t.get('center'),
                        'area': t.get('area'),
                        'kp_range': [t.get('kp_start'), t.get('kp_end')],
                        'source_file': t.get('source_file', 'unknown')
                    }
                    for t in value[:10]  # Only save first 10 as examples
                ]
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
        save_state['_metadata'] = {
            'user_input': user_input,
            'workflow_version': 'version_1_1',
            'execution_time': pd.Timestamp.now().isoformat(),
            'total_triangles': len(state.get('triangles', [])),
            'total_clusters': len(state.get('clusters', [])),
            'total_hulls': len(state.get('hulls', []))
        }
        
        # Save to JSON file
        output_file = "output/workflow_state.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_state, f, ensure_ascii=False, indent=2)
        
        print(f"   📄 Workflow state saved: {output_file}")
        
    except Exception as e:
        print(f"   ⚠️ State saving failed: {str(e)}")


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