#!/usr/bin/env python3
"""
简单测试脚本 - 测试新的LLM分析工作流
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from implementation import compiled_agent
from config import config

def main():
    print("="*70)
    print("🧪 Testing LLM-based Workflow")
    print("="*70)

    # 检查配置
    print("\n📋 Configuration:")
    print(f"   API Key: {'Set ✅' if config.get_openai_api_key() else 'Not set ⚠️'}")
    print(f"   Model: {config.openai_model}")
    print(f"   LLM Available: {config.is_llm_available()}")

    if not config.is_llm_available():
        print("\n⚠️ Warning: No API key configured, will use fallback analysis")

    # 测试输入
    test_input = "请分析関越高速公路上行方向2024年5月3日的交通情况"

    print(f"\n📝 Test Input:")
    print(f"   {test_input}")

    print("\n" + "="*70)
    print("🚀 Running workflow...")
    print("="*70)

    try:
        # 执行工作流
        result = compiled_agent.invoke({"user_input": test_input})

        print("\n" + "="*70)
        print("✅ Workflow completed successfully!")
        print("="*70)

        # 显示结果
        print("\n📊 Results:")

        # LLM分析结果
        llm_analysis = result.get('llm_analysis', {})
        if llm_analysis:
            print(f"\n🤖 LLM Analysis:")
            for direction, analysis in llm_analysis.items():
                hotspots = analysis.get('hotspots', [])
                summary = analysis.get('summary', {})
                confidence = summary.get('analysis_confidence', 0)

                print(f"\n   {direction} direction:")
                print(f"      Hotspots: {len(hotspots)}")
                print(f"      Confidence: {confidence:.2f}")

                for i, hotspot in enumerate(hotspots[:3], 1):
                    kp_range = hotspot['kp_range']
                    print(f"      Hotspot {i}: KP {kp_range[0]:.1f}-{kp_range[1]:.1f}km, "
                          f"frequency: {hotspot['frequency']}")

        # 文件输出
        print(f"\n📁 Generated Files:")

        fig_paths = result.get('fig_paths', [])
        if fig_paths:
            for i, path in enumerate(fig_paths, 1):
                if os.path.exists(path):
                    print(f"   🖼️ Image {i}: {path}")

        csv_path = result.get('csv_output_path', '')
        if csv_path and os.path.exists(csv_path):
            print(f"   📄 CSV: {csv_path}")

        # 报告预览
        final_report = result.get('final_report', '')
        if final_report:
            print(f"\n📄 Report Preview (first 500 chars):")
            print(f"   {final_report[:500]}...")

        print("\n" + "="*70)
        print("🎉 Test completed successfully!")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)