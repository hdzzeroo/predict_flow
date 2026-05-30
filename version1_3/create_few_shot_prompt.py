#!/usr/bin/env python3
"""
创建适合添加到prompt中的few-shot示例
"""

import json
from typing import Dict, List, Any


def format_time(minutes: int) -> str:
    """将分钟数转换为HH:MM格式"""
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}"


def create_few_shot_prompt(
    workflow_state_path: str = "output/workflow_state.json",
    output_path: str = "output/few_shot_prompt.txt",
    direction: str = "上"  # 选择哪个方向作为示例
):
    """
    创建few-shot prompt示例
    """
    with open(workflow_state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    direction_data = state.get('direction_data', {}).get(direction, {})
    llm_analysis = state.get('llm_analysis', {}).get(direction, {})

    triangles = direction_data.get('triangles', [])
    hotspots = llm_analysis.get('hotspots', [])
    thinking = llm_analysis.get('thinking', {})

    # 构建few-shot示例
    lines = []

    lines.append("<few_shot_example>")
    lines.append("")
    lines.append("以下是一个实际分析案例，供参考：")
    lines.append("")

    # INPUT部分
    lines.append("## 输入数据")
    lines.append(f"方向：{direction}")
    lines.append(f"三角形数量：{len(triangles)}")
    lines.append("")

    # 提取KP和时间范围
    kp_values = []
    time_values = []
    years = set()

    for t in triangles:
        if 'kp_range' in t:
            kp_values.extend(t['kp_range'])
        if 'time_start' in t and 'time_end' in t:
            time_values.extend([t['time_start'], t['time_end']])
        source = t.get('source_file', '')
        if source:
            parts = source.split('_')
            if len(parts) >= 3:
                try:
                    years.add(int(parts[2]))
                except:
                    pass

    if kp_values and time_values:
        lines.append(f"KP范围：{min(kp_values):.1f} - {max(kp_values):.1f} km")
        lines.append(f"时间范围：{format_time(min(time_values))} - {format_time(max(time_values))}")
        lines.append(f"覆盖年份：{sorted(years)}")
    lines.append("")

    # 显示部分三角形（简化版）
    lines.append("三角形数据（示例前10个）：")
    lines.append("```json")
    triangle_samples = []
    for i, t in enumerate(triangles[:10]):
        sample = {
            'id': t.get('id'),
            'shape_type': t.get('shape_type'),
            'kp_range': [round(x, 1) for x in t.get('kp_range', [0, 0])],
            'time_range': [t.get('time_start'), t.get('time_end')],
            'center': [round(x, 1) for x in t.get('center', [0, 0])],
            'source_file': t.get('source_file', '').split('/')[-1]
        }
        triangle_samples.append(sample)

    lines.append(json.dumps(triangle_samples, ensure_ascii=False, indent=2))
    lines.append("```")

    if len(triangles) > 10:
        lines.append(f"（...还有{len(triangles) - 10}个三角形）")
    lines.append("")

    # THINKING部分
    lines.append("## LLM分析过程")
    lines.append("")

    lines.append("### Step 1: 聚类识别")
    step1 = thinking.get('step1_clustering', '')
    if step1:
        # 只显示前800字符
        lines.append(step1[:800] + ("..." if len(step1) > 800 else ""))
    lines.append("")

    lines.append("### Step 2: 边界分析")
    step2 = thinking.get('step2_boundaries', {})
    if isinstance(step2, dict):
        lines.append("```json")
        # 只显示前3个cluster
        sample_boundaries = {}
        for i, (cluster_id, info) in enumerate(list(step2.items())[:3]):
            sample_boundaries[cluster_id] = info
        lines.append(json.dumps(sample_boundaries, ensure_ascii=False, indent=2))
        lines.append("```")
        if len(step2) > 3:
            lines.append(f"（...还有{len(step2) - 3}个cluster的边界分析）")
    lines.append("")

    lines.append("### Step 3: 形状构造")
    step3 = thinking.get('step3_construction', '')
    if step3:
        lines.append(step3[:500] + ("..." if len(step3) > 500 else ""))
    lines.append("")

    lines.append("### Step 4: 质量验证")
    step4 = thinking.get('step4_validation', '')
    if step4:
        lines.append(step4[:500] + ("..." if len(step4) > 500 else ""))
    lines.append("")

    # OUTPUT部分
    lines.append("## 最终输出")
    lines.append("")
    lines.append(f"识别出{len(hotspots)}个热点：")
    lines.append("")

    lines.append("```json")
    output_data = {
        "direction": direction,
        "hotspots": [],
        "summary": llm_analysis.get('summary', {})
    }

    # 格式化hotspots
    for h in hotspots:
        formatted_hotspot = {
            "hotspot_id": h['hotspot_id'],
            "kp_range": [round(x, 1) for x in h['kp_range']],
            "time_range": h['time_range'],
            "included_triangle_ids": h['included_triangle_ids'],
            "frequency": h['frequency'],
            "years_coverage": h.get('years_coverage', []),
            "severity": h['severity'],
            "description": h.get('description', ''),
            "reasoning": h.get('reasoning', ''),
            "prediction_shape": h.get('prediction_shape', {})
        }
        output_data["hotspots"].append(formatted_hotspot)

    lines.append(json.dumps(output_data, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("</few_shot_example>")

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ Few-shot prompt已保存到: {output_path}")
    print(f"   文件大小: {len('\n'.join(lines))} 字符")
    print(f"   约 {len('\n'.join(lines)) // 4} tokens (估算)")

    # 打印统计
    print(f"\n示例包含:")
    print(f"  - 输入: {len(triangles)} 个三角形 (显示前10个)")
    print(f"  - 输出: {len(hotspots)} 个热点")
    print(f"  - Thinking: 4个步骤的推理过程")

    return '\n'.join(lines)


def create_compact_few_shot(
    workflow_state_path: str = "output/workflow_state.json",
    output_path: str = "output/few_shot_compact.txt",
    direction: str = "上"
):
    """
    创建紧凑版few-shot示例（更短，只包含关键信息）
    """
    with open(workflow_state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    direction_data = state.get('direction_data', {}).get(direction, {})
    llm_analysis = state.get('llm_analysis', {}).get(direction, {})

    triangles = direction_data.get('triangles', [])
    hotspots = llm_analysis.get('hotspots', [])
    thinking = llm_analysis.get('thinking', {})

    lines = []

    lines.append("<few_shot_example_compact>")
    lines.append("")

    # 简化的输入
    lines.append(f"输入: {len(triangles)}个三角形 ({direction}方向)")
    lines.append("")

    # 只显示聚类结果（Step 1）
    lines.append("聚类识别:")
    step1 = thinking.get('step1_clustering', '')
    # 提取Cluster信息
    cluster_lines = [line.strip() for line in step1.split('。') if 'Cluster' in line][:3]
    for line in cluster_lines:
        if line:
            lines.append(f"  {line}。")
    lines.append("")

    # 输出热点
    lines.append(f"输出: {len(hotspots)}个热点")
    for h in hotspots:
        kp_start, kp_end = h['kp_range']
        time_start, time_end = h['time_range']
        lines.append(f"  Hotspot {h['hotspot_id']}: KP {kp_start:.1f}-{kp_end:.1f}km, "
                    f"{format_time(time_start)}-{format_time(time_end)}, "
                    f"{h['frequency']}个事件, {h['severity']}")
    lines.append("")

    lines.append("</few_shot_example_compact>")

    # 保存
    content = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 紧凑版few-shot已保存到: {output_path}")
    print(f"   文件大小: {len(content)} 字符")
    print(f"   约 {len(content) // 4} tokens (估算)")

    return content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="创建few-shot prompt示例")
    parser.add_argument(
        "--input", "-i",
        default="output/workflow_state.json",
        help="workflow_state.json文件路径"
    )
    parser.add_argument(
        "--direction", "-d",
        default="上",
        choices=["上", "下"],
        help="选择哪个方向作为示例"
    )
    parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="生成紧凑版"
    )

    args = parser.parse_args()

    # 生成完整版
    create_few_shot_prompt(
        workflow_state_path=args.input,
        output_path=f"output/few_shot_prompt_{args.direction}.txt",
        direction=args.direction
    )

    # 如果需要紧凑版
    if args.compact:
        create_compact_few_shot(
            workflow_state_path=args.input,
            output_path=f"output/few_shot_compact_{args.direction}.txt",
            direction=args.direction
        )

    print("\n提示: 将生成的文件内容添加到prompt_templates.py的system prompt中")
