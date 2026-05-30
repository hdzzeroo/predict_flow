#!/usr/bin/env python3
"""
查看LLM思考链的脚本
用于从workflow_state.json中提取并美化显示LLM的推理过程
"""

import json
import sys
from pathlib import Path


def format_thinking_step(step_name: str, content: str, max_width: int = 100) -> str:
    """格式化单个思考步骤"""
    lines = []
    lines.append(f"\n{'='*max_width}")
    lines.append(f"  {step_name}")
    lines.append(f"{'='*max_width}\n")

    # 如果content是字典，格式化为JSON
    if isinstance(content, dict):
        lines.append(json.dumps(content, ensure_ascii=False, indent=2))
    else:
        # 如果是字符串，按照句子分段
        sentences = content.split('。')
        for sentence in sentences:
            if sentence.strip():
                lines.append(sentence.strip() + '。')

    lines.append("")
    return "\n".join(lines)


def display_thinking_chain(workflow_state_path: str = "output/workflow_state.json"):
    """
    显示LLM的完整思考链

    Args:
        workflow_state_path: workflow_state.json文件路径
    """
    # 读取workflow_state.json
    path = Path(workflow_state_path)
    if not path.exists():
        print(f"❌ 错误: 找不到文件 {workflow_state_path}")
        print(f"   请确保已经运行过工作流")
        return

    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    # 获取LLM分析结果
    llm_analysis = state.get("llm_analysis", {})

    if not llm_analysis:
        print("❌ 错误: workflow_state.json中没有LLM分析结果")
        print("   可能是使用了传统聚类方法而非LLM分析")
        return

    print("\n" + "="*100)
    print("  LLM 思考链查看器")
    print("="*100)

    # 遍历每个方向
    for direction, analysis in llm_analysis.items():
        print(f"\n{'#'*100}")
        print(f"  {direction}方向分析")
        print(f"{'#'*100}")

        # 获取thinking字段
        thinking = analysis.get("thinking", {})

        if not thinking:
            print(f"\n⚠️ {direction}方向没有thinking字段")
            continue

        # 显示4个步骤
        steps = [
            ("Step 1: 聚类识别（Clustering）", "step1_clustering"),
            ("Step 2: 边界分析（Boundary Analysis）", "step2_boundaries"),
            ("Step 3: 形状构造（Shape Construction）", "step3_construction"),
            ("Step 4: 质量验证（Validation）", "step4_validation")
        ]

        for step_name, step_key in steps:
            content = thinking.get(step_key)
            if content:
                print(format_thinking_step(step_name, content))
            else:
                print(f"\n⚠️ 缺少 {step_name}")

        # 显示分析摘要
        summary = analysis.get("summary", {})
        hotspots = analysis.get("hotspots", [])

        print(f"\n{'='*100}")
        print(f"  分析结果摘要")
        print(f"{'='*100}")
        print(f"方向: {direction}")
        print(f"识别的热点数量: {len(hotspots)}")
        print(f"分析置信度: {summary.get('analysis_confidence', 0):.2f}")
        print(f"最严重热点ID: {summary.get('most_severe_hotspot_id', 'N/A')}")

        # 显示热点列表
        if hotspots:
            print(f"\n热点列表:")
            for i, hotspot in enumerate(hotspots, 1):
                kp_start, kp_end = hotspot['kp_range']
                time_start, time_end = hotspot['time_range']
                start_hour, start_min = divmod(time_start, 60)
                end_hour, end_min = divmod(time_end, 60)

                print(f"\n  Hotspot {i} (ID: {hotspot['hotspot_id']}):")
                print(f"    位置: KP {kp_start:.1f} - {kp_end:.1f} km")
                print(f"    时间: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}")
                print(f"    频次: {hotspot['frequency']}")
                print(f"    严重程度: {hotspot['severity']}")
                print(f"    描述: {hotspot.get('description', 'N/A')}")
                print(f"    包含的三角形ID: {hotspot['included_triangle_ids']}")

        print("\n")


def display_thinking_summary(workflow_state_path: str = "output/workflow_state.json"):
    """
    显示LLM思考链的简要摘要（不显示详细内容）

    Args:
        workflow_state_path: workflow_state.json文件路径
    """
    path = Path(workflow_state_path)
    if not path.exists():
        print(f"❌ 错误: 找不到文件 {workflow_state_path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    llm_analysis = state.get("llm_analysis", {})

    if not llm_analysis:
        print("❌ 错误: workflow_state.json中没有LLM分析结果")
        return

    print("\n" + "="*80)
    print("  LLM 分析摘要")
    print("="*80)

    for direction, analysis in llm_analysis.items():
        thinking = analysis.get("thinking", {})
        summary = analysis.get("summary", {})
        hotspots = analysis.get("hotspots", [])

        print(f"\n【{direction}方向】")
        print(f"  - 热点数量: {len(hotspots)}")
        print(f"  - 置信度: {summary.get('analysis_confidence', 0):.2f}")
        print(f"  - 思考步骤:")

        if thinking.get("step1_clustering"):
            cluster_text = str(thinking["step1_clustering"])
            cluster_count = cluster_text.count("Cluster")
            print(f"    ✓ Step 1 聚类识别: 识别了{cluster_count}个聚类")

        if thinking.get("step2_boundaries"):
            print(f"    ✓ Step 2 边界分析: 完成")

        if thinking.get("step3_construction"):
            print(f"    ✓ Step 3 形状构造: 完成")

        if thinking.get("step4_validation"):
            print(f"    ✓ Step 4 质量验证: 完成")

    print("\n")


def export_thinking_to_file(
    workflow_state_path: str = "output/workflow_state.json",
    output_path: str = "output/llm_thinking_chain.txt"
):
    """
    将LLM思考链导出到文本文件

    Args:
        workflow_state_path: workflow_state.json文件路径
        output_path: 输出文件路径
    """
    import io
    from contextlib import redirect_stdout

    # 捕获display_thinking_chain的输出
    f = io.StringIO()
    with redirect_stdout(f):
        display_thinking_chain(workflow_state_path)

    output = f.getvalue()

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"✅ LLM思考链已导出到: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="查看LLM的思考链")
    parser.add_argument(
        "--file", "-f",
        default="output/workflow_state.json",
        help="workflow_state.json文件路径 (默认: output/workflow_state.json)"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="只显示摘要，不显示详细内容"
    )
    parser.add_argument(
        "--export", "-e",
        help="导出思考链到指定文件"
    )

    args = parser.parse_args()

    if args.export:
        export_thinking_to_file(args.file, args.export)
    elif args.summary:
        display_thinking_summary(args.file)
    else:
        display_thinking_chain(args.file)
