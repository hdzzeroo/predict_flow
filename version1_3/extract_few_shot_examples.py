#!/usr/bin/env python3
"""
从workflow_state.json中提取LLM聚类结果，用于few-shot学习
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def extract_triangle_summary(triangles: List[Dict]) -> List[Dict]:
    """
    提取三角形的简化摘要信息
    """
    summaries = []
    for t in triangles:
        summary = {
            'id': t.get('id'),
            'shape_type': t.get('shape_type'),
            'kp_range': t.get('kp_range', []),
            'time_range': [t.get('time_start'), t.get('time_end')],
            'center': t.get('center', []),
            'source_file': t.get('source_file', ''),
            'vertices': t.get('vertices', [])
        }
        summaries.append(summary)
    return summaries


def format_time_range(time_range: List[int]) -> str:
    """将分钟数转换为可读的时间范围"""
    start_hour, start_min = divmod(time_range[0], 60)
    end_hour, end_min = divmod(time_range[1], 60)
    return f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}"


def extract_hotspot_info(hotspot: Dict) -> Dict:
    """提取热点的关键信息"""
    return {
        'hotspot_id': hotspot['hotspot_id'],
        'kp_range': hotspot['kp_range'],
        'kp_span': hotspot['kp_range'][1] - hotspot['kp_range'][0],
        'time_range': hotspot['time_range'],
        'time_range_str': format_time_range(hotspot['time_range']),
        'time_span_minutes': hotspot['time_range'][1] - hotspot['time_range'][0],
        'included_triangle_ids': hotspot['included_triangle_ids'],
        'frequency': hotspot['frequency'],
        'years_coverage': hotspot.get('years_coverage', []),
        'severity': hotspot['severity'],
        'description': hotspot.get('description', ''),
        'reasoning': hotspot.get('reasoning', ''),
        'prediction_shape': hotspot.get('prediction_shape', {})
    }


def extract_few_shot_example(
    workflow_state_path: str = "output/workflow_state.json",
    output_path: str = "output/few_shot_example.json"
):
    """
    从workflow_state.json中提取few-shot示例

    Args:
        workflow_state_path: workflow_state.json文件路径
        output_path: 输出文件路径
    """
    # 读取workflow_state.json
    with open(workflow_state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    # 提取基本信息
    user_input = state.get('user_input', '')
    route = state.get('route', '')
    target_date = f"{state.get('target_year')}-{state.get('target_month'):02d}-{state.get('target_day'):02d}"

    # 提取每个方向的数据
    few_shot_data = {
        'metadata': {
            'user_input': user_input,
            'route': route,
            'target_date': target_date,
            'file_paths': state.get('file_paths', [])
        },
        'directions': {}
    }

    direction_data = state.get('direction_data', {})
    llm_analysis = state.get('llm_analysis', {})

    for direction in direction_data.keys():
        print(f"\n提取 {direction} 方向的数据...")

        # 输入：三角形数据
        triangles = direction_data[direction].get('triangles', [])
        triangle_summaries = extract_triangle_summary(triangles)

        # 输出：LLM聚类结果
        analysis = llm_analysis.get(direction, {})
        hotspots = analysis.get('hotspots', [])
        hotspot_infos = [extract_hotspot_info(h) for h in hotspots]

        # thinking过程
        thinking = analysis.get('thinking', {})

        # 统计信息
        kp_values = []
        time_values = []
        for t in triangles:
            if 'kp_range' in t:
                kp_values.extend(t['kp_range'])
            if 'time_start' in t and 'time_end' in t:
                time_values.extend([t['time_start'], t['time_end']])

        direction_example = {
            'input': {
                'direction': direction,
                'triangle_count': len(triangles),
                'kp_range': [min(kp_values), max(kp_values)] if kp_values else [0, 0],
                'time_range': [min(time_values), max(time_values)] if time_values else [0, 0],
                'time_range_str': format_time_range([min(time_values), max(time_values)]) if time_values else "00:00-00:00",
                'triangles': triangle_summaries
            },
            'output': {
                'thinking': thinking,
                'hotspots': hotspot_infos,
                'summary': analysis.get('summary', {})
            }
        }

        few_shot_data['directions'][direction] = direction_example

        print(f"  输入三角形数: {len(triangles)}")
        print(f"  输出热点数: {len(hotspots)}")
        print(f"  KP范围: {direction_example['input']['kp_range']}")
        print(f"  时间范围: {direction_example['input']['time_range_str']}")

    # 保存到文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(few_shot_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Few-shot示例已保存到: {output_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("Few-shot示例摘要")
    print("="*80)
    print(f"路线: {route}")
    print(f"日期: {target_date}")
    print(f"方向数: {len(few_shot_data['directions'])}")

    for direction, data in few_shot_data['directions'].items():
        print(f"\n{direction}方向:")
        print(f"  输入: {data['input']['triangle_count']} 个三角形")
        print(f"  输出: {len(data['output']['hotspots'])} 个热点")
        print(f"  覆盖率: {sum(len(h['included_triangle_ids']) for h in data['output']['hotspots'])} / {data['input']['triangle_count']}")

    return few_shot_data


def create_readable_summary(
    workflow_state_path: str = "output/workflow_state.json",
    output_path: str = "output/few_shot_example_readable.txt"
):
    """
    创建人类可读的few-shot示例文档
    """
    with open(workflow_state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    output = []
    output.append("="*80)
    output.append("FEW-SHOT LEARNING EXAMPLE")
    output.append("="*80)
    output.append("")

    # 基本信息
    output.append(f"用户输入: {state.get('user_input')}")
    output.append(f"路线: {state.get('route')}")
    output.append(f"目标日期: {state.get('target_year')}-{state.get('target_month'):02d}-{state.get('target_day'):02d}")
    output.append("")

    direction_data = state.get('direction_data', {})
    llm_analysis = state.get('llm_analysis', {})

    for direction in direction_data.keys():
        output.append("\n" + "#"*80)
        output.append(f"# {direction}方向示例")
        output.append("#"*80)
        output.append("")

        # 输入数据
        triangles = direction_data[direction].get('triangles', [])
        output.append("## INPUT (输入数据)")
        output.append("")
        output.append(f"三角形总数: {len(triangles)}")
        output.append("")

        # 统计信息
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
                # 提取年份 (例如: "関越道_上_2014_05-04" -> 2014)
                parts = source.split('_')
                if len(parts) >= 3:
                    try:
                        years.add(int(parts[2]))
                    except:
                        pass

        if kp_values and time_values:
            output.append(f"KP范围: {min(kp_values):.1f} - {max(kp_values):.1f} km")
            output.append(f"时间范围: {format_time_range([min(time_values), max(time_values)])}")
            output.append(f"覆盖年份: {sorted(years)}")
        output.append("")

        # 显示前5个三角形作为示例
        output.append("三角形示例 (前5个):")
        for i, t in enumerate(triangles[:5]):
            output.append(f"\n  Triangle {t.get('id')}:")
            output.append(f"    形状: {t.get('shape_type')}")
            output.append(f"    KP: {t.get('kp_range', [0, 0])[0]:.1f} - {t.get('kp_range', [0, 0])[1]:.1f} km")
            output.append(f"    时间: {format_time_range([t.get('time_start', 0), t.get('time_end', 0)])}")
            output.append(f"    来源: {t.get('source_file', 'N/A')}")

        if len(triangles) > 5:
            output.append(f"\n  ... 还有 {len(triangles) - 5} 个三角形")
        output.append("")

        # LLM思考过程
        analysis = llm_analysis.get(direction, {})
        thinking = analysis.get('thinking', {})

        output.append("\n## LLM THINKING (思考过程)")
        output.append("")

        output.append("### Step 1: 聚类识别")
        output.append(thinking.get('step1_clustering', 'N/A')[:500] + "...")
        output.append("")

        output.append("### Step 2: 边界分析")
        boundaries = thinking.get('step2_boundaries', {})
        if isinstance(boundaries, dict):
            for cluster_id, info in list(boundaries.items())[:3]:  # 显示前3个
                output.append(f"\n  {cluster_id}:")
                output.append(f"    front_kp: {info.get('front_kp')}")
                output.append(f"    back_kp: {info.get('back_kp')}")
                output.append(f"    shape_decision: {info.get('shape_decision')}")
        output.append("")

        # 输出结果
        hotspots = analysis.get('hotspots', [])
        output.append("\n## OUTPUT (聚类结果)")
        output.append("")
        output.append(f"识别的热点数: {len(hotspots)}")
        output.append("")

        for hotspot in hotspots:
            output.append(f"\n### Hotspot {hotspot['hotspot_id']}")
            output.append(f"  KP范围: {hotspot['kp_range'][0]:.1f} - {hotspot['kp_range'][1]:.1f} km (跨度: {hotspot['kp_range'][1] - hotspot['kp_range'][0]:.1f} km)")
            output.append(f"  时间范围: {format_time_range(hotspot['time_range'])} (跨度: {hotspot['time_range'][1] - hotspot['time_range'][0]} 分钟)")
            output.append(f"  包含的三角形ID: {hotspot['included_triangle_ids']}")
            output.append(f"  频次: {hotspot['frequency']}")
            output.append(f"  覆盖年份: {hotspot.get('years_coverage', [])}")
            output.append(f"  严重程度: {hotspot['severity']}")
            output.append(f"  描述: {hotspot.get('description', 'N/A')}")

            # 预测形状
            shape = hotspot.get('prediction_shape', {})
            if shape:
                output.append(f"  预测形状类型: {shape.get('shape_type')}")
                vertices = shape.get('vertices', [])
                if vertices:
                    output.append(f"  顶点数: {len(vertices)}")
                    output.append(f"  顶点坐标: {vertices}")

        output.append("")

        # 摘要
        summary = analysis.get('summary', {})
        output.append("\n## SUMMARY (摘要)")
        output.append(f"  总热点数: {summary.get('total_hotspots')}")
        output.append(f"  最严重热点ID: {summary.get('most_severe_hotspot_id')}")
        output.append(f"  分析置信度: {summary.get('analysis_confidence')}")
        output.append("")

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"\n✅ 可读格式已保存到: {output_path}")
    print(f"   可以直接查看该文件了解few-shot示例的详细内容")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取few-shot学习示例")
    parser.add_argument(
        "--input", "-i",
        default="output/workflow_state.json",
        help="workflow_state.json文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/few_shot_example.json",
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--readable", "-r",
        action="store_true",
        help="生成可读的文本格式"
    )

    args = parser.parse_args()

    # 提取JSON格式
    extract_few_shot_example(args.input, args.output)

    # 如果指定，也生成可读格式
    if args.readable:
        readable_path = args.output.replace('.json', '_readable.txt')
        create_readable_summary(args.input, readable_path)
