"""This file was generated using `langgraph-gen` version 0.0.3.

This file provides a placeholder implementation for the corresponding stub.

Replace the placeholder implementation with your own logic.
"""


from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import pandas as pd

from functions import (extract_route_and_time, generate_file_path, process_traffic_data, 
                     cluster_triangles_dbscan, cluster_triangles_custom, analyze_clusters,
                     calculate_all_hull_triangles, visualize_triangles_with_hulls, create_hull_summary_report,
                     enhanced_generate_file_paths, call_llm_for_date_parsing,
                     process_multiple_traffic_data)
from config import config
from stub import CustomAgent


class TrafficState(TypedDict, total=False):
    # —— 用户输入 ——
    user_input: str                     # 原始指令
    # —— Chatbot 节点输出 ——
    file_path: str                      # 主要 CSV 路径（向后兼容）
    file_paths: List[str]              # 多个 CSV 路径列表
    route: str                          # 路段标识（可选）
    ts: str                             # 时间字符串（可选）
    # —— 可视化节点输出 ——
    triangles: List[Dict[str, Any]]     # 小三角形数据
    fig_path: str                       # 图片文件路径
    # —— 聚类节点输出 ——
    clusters: List[List[int]]           # 三角形索引簇
    cluster_analysis: Dict[str, Any]    # 聚类分析结果
    # —— 大三角形节点输出 ——
    hulls: List[Dict[str, Any]]         # 外包大三角形
    hull_fig_path: str                  # 带外包大三角形的图片路径
    hull_summary: str                   # 外包大三角形摘要报告
    # —— 报告节点输出（终点） ——
    final_report: str                         # LLM 生成的分析报告


# Define stand-alone functions
def chatbot(state: TrafficState) -> dict:
    """
    处理用户输入，提取路名和时间信息，生成CSV文件路径
    使用LLM语义解析增强用户体验
    """
    print("In node: chatbot")
    user_input = state.get("user_input", "")
    
    if not user_input:
        default_path = "data/関越2024_cleaned.csv"
        return {
            "file_path": default_path,
            "file_paths": [default_path],        # 默认情况也返回列表
            "route": "未指定",
            "ts": "未指定"
        }
    
    print(f"🔍 用户输入: {user_input}")
    
    try:
        # 使用增强的LLM文件路径生成功能
        print("🤖 启动LLM语义解析...")
        
        # 首先尝试使用LLM解析
        file_paths = enhanced_generate_file_paths(
            user_input=user_input,
            use_llm=config.is_llm_available(),
            api_key=config.get_openai_api_key()
        )
        
        # 选择第一个文件路径作为主要路径
        primary_file_path = file_paths[0] if file_paths else None
        
        if primary_file_path:
            print(f"✅ LLM解析成功，生成 {len(file_paths)} 个候选文件")
            print(f"📁 选择主要文件: {primary_file_path}")
            
            # 使用LLM解析获取更详细的信息
            parsed_info = call_llm_for_date_parsing(
                user_input, 
                use_real_llm=config.is_llm_available(),
                api_key=config.get_openai_api_key()
            )
            
            route = parsed_info.get('route_name', '未指定')
            year = parsed_info.get('year')
            month = parsed_info.get('month')
            day = parsed_info.get('day')
            
            # 构造时间字符串
            if year and month and day:
                ts = f"{year}年{month}月{day}日"
            elif year and month:
                ts = f"{year}年{month}月"
            elif year:
                ts = f"{year}年"
            else:
                ts = "未指定"
            
            print(f"🎯 解析结果:")
            print(f"   路线: {route}")
            print(f"   时间: {ts}")
            print(f"   置信度: {parsed_info.get('confidence', 0):.2f}")
            
            return {
                "file_path": primary_file_path,
                "file_paths": file_paths,          # 添加多文件路径
                "route": route,
                "ts": ts
            }
        else:
            raise Exception("LLM解析未返回有效文件路径")
            
    except Exception as e:
        print(f"⚠️ LLM解析失败，降级使用传统方法: {str(e)}")
        
        # 降级到传统的正则表达式方法
        extracted_info = extract_route_and_time(user_input)
        route = extracted_info.get('route')
        time = extracted_info.get('time')
        
        print(f"🔧 传统方法提取结果:")
        print(f"   路名: {route}")
        print(f"   时间: {time}")
        
        # 生成文件路径
        file_path = generate_file_path(route, time)
        
        print(f"📁 生成的文件路径: {file_path}")
        
        return {
            "file_path": file_path,
            "file_paths": [file_path],          # 为向后兼容，传统方法也返回列表
            "route": route or "未指定",
            "ts": time or "未指定"
        }


def visualization(state: TrafficState) -> dict:
    """
    读取CSV/Excel文件并进行可视化，生成三角形数据
    支持多文件处理：优先使用file_paths，向后兼容file_path
    """
    print("In node: visualization")
    
    # 获取文件路径（多文件优先）
    file_paths = state.get("file_paths", [])
    file_path = state.get("file_path", "")
    
    # 确定处理模式
    if file_paths and len(file_paths) > 1:
        # 多文件模式
        print(f"📊 多文件模式：处理 {len(file_paths)} 个CSV文件")
        for i, fp in enumerate(file_paths):
            print(f"  文件 {i+1}: {fp}")
        
        try:
            # 调用多文件处理函数
            triangles, fig_path = process_multiple_traffic_data(file_paths, output_dir="output")
            
            print(f"✅ 多文件可视化完成:")
            print(f"  - 生成了 {len(triangles)} 个三角形")
            print(f"  - 图片保存至: {fig_path}")
            
            return {
                "triangles": triangles,       # 包含详细信息的三角形数据列表
                "fig_path": fig_path         # 保存的图片路径
            }
            
        except Exception as e:
            print(f"❌ 多文件可视化处理出错: {str(e)}")
            return {
                "triangles": [],
                "fig_path": ""
            }
    
    elif file_paths and len(file_paths) == 1:
        # 单文件模式（从file_paths获取）
        file_path = file_paths[0]
    
    # 单文件处理（传统模式）
    if not file_path:
        print("❌ 错误：未提供文件路径")
        return {
            "triangles": [],
            "fig_path": ""
        }
    
    try:
        print(f"📄 单文件模式：处理文件 {file_path}")
        
        # 调用传统的单文件处理函数
        triangles, fig_path = process_traffic_data(file_path, output_dir="output")
        
        print(f"✅ 单文件可视化完成:")
        print(f"  - 生成了 {len(triangles)} 个三角形")
        print(f"  - 图片保存至: {fig_path}")
        
        return {
            "triangles": triangles,       # 包含详细信息的三角形数据列表
            "fig_path": fig_path         # 保存的图片路径
        }
        
    except Exception as e:
        print(f"❌ 单文件可视化处理出错: {str(e)}")
        return {
            "triangles": [],
            "fig_path": ""
        }


def cluster(state: TrafficState) -> dict:
    """
    对三角形进行聚类分析
    """
    print("In node: cluster")
    triangles = state.get("triangles", [])
    
    if not triangles:
        print("警告：没有三角形数据用于聚类")
        return {"clusters": []}
    
    print(f"开始对 {len(triangles)} 个三角形进行聚类分析...")
    
    try:
        # 方法1: 使用DBSCAN聚类 (推荐)
        clusters_dbscan = cluster_triangles_dbscan(
            triangles, 
            eps=1.5,        # 邻域半径，可根据数据调整
            min_samples=2   # 最小样本数
        )
        
        # 方法2: 使用自定义时空距离聚类 (备选)
        # clusters_custom = cluster_triangles_custom(
        #     triangles,
        #     distance_threshold=3.0
        # )
        
        # 选择使用DBSCAN结果
        clusters = clusters_dbscan
        
        # 分析聚类结果
        analysis = analyze_clusters(triangles, clusters)
        
        print(f"聚类分析完成:")
        print(f"  - 总聚类数: {analysis['total_clusters']}")
        print(f"  - 总三角形数: {analysis['total_triangles']}")
        
        # 打印每个聚类的统计信息
        for stat in analysis['cluster_stats'][:5]:  # 只显示前5个最大的聚类
            kp_range = stat['kp_range']
            time_range = stat['time_range']
            print(f"  - 聚类 {stat['cluster_id']}: {stat['size']} 个三角形")
            print(f"    KP范围: {kp_range['min']:.1f} - {kp_range['max']:.1f} km")
            print(f"    时间范围: {time_range['min']:.0f} - {time_range['max']:.0f} 分钟")
        
        # 存储聚类分析结果到状态中
        return {
            "clusters": clusters,
            "cluster_analysis": analysis
        }
        
    except Exception as e:
        print(f"聚类分析出错: {str(e)}")
        # 如果聚类失败，返回每个三角形作为单独聚类
        fallback_clusters = [[i] for i in range(len(triangles))]
        return {"clusters": fallback_clusters}


def draw_hulls(state: TrafficState) -> dict:
    """
    为每个聚类绘制外包大三角形
    """
    print("In node: draw_hulls")
    clusters = state.get("clusters", [])
    triangles = state.get("triangles", [])
    fig_path = state.get("fig_path", "")
    
    if not clusters or not triangles:
        print("警告：缺少聚类或三角形数据，无法绘制外包大三角形")
        return {"hulls": []}
    
    print(f"开始为 {len(clusters)} 个聚类绘制外包大三角形...")
    
    try:
        # 1. 计算所有聚类的外包大三角形
        hulls = calculate_all_hull_triangles(triangles, clusters)
        
        if not hulls:
            print("未找到有效的外包大三角形")
            return {"hulls": []}
        
        # 2. 创建带有外包大三角形的可视化图片
        base_name = fig_path.replace('.png', '') if fig_path else 'triangles'
        hull_fig_path = f"{base_name}_with_hulls.png"
        
        # 确保输出目录存在
        import os
        os.makedirs(os.path.dirname(hull_fig_path) if os.path.dirname(hull_fig_path) else '.', exist_ok=True)
        
        # 绘制带外包大三角形的可视化图
        final_fig_path = visualize_triangles_with_hulls(
            triangles=triangles,
            hulls=hulls,
            output_path=hull_fig_path,
            figsize=(16, 12)  # 使用更大的图片尺寸以便清楚显示
        )
        
        # 3. 生成外包大三角形的摘要报告
        hull_summary = create_hull_summary_report(hulls)
        
        print(f"外包大三角形绘制完成:")
        print(f"  - 绘制了 {len(hulls)} 个外包大三角形")
        print(f"  - 覆盖了 {sum(len(cluster) for cluster in clusters)} 个原始三角形")
        print(f"  - 图片保存至: {final_fig_path}")
        
        # 打印每个外包大三角形的信息
        for hull in hulls[:5]:  # 只显示前5个
            kp_start, kp_end = hull['kp_range']
            print(f"  - 聚类 {hull['cluster_id']}: {hull['cluster_size']} 个三角形")
            print(f"    KP范围: {kp_start:.1f} - {kp_end:.1f} km")
            print(f"    覆盖面积: {hull['area']:.1f} 平方单位")
        
        return {
            "hulls": hulls,
            "hull_fig_path": final_fig_path,
            "hull_summary": hull_summary
        }
        
    except Exception as e:
        print(f"绘制外包大三角形出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"hulls": []}


def report(state: TrafficState) -> dict:
    """
    基于三角形分析结果生成简化报告，主要输出大三角形位置信息
    """
    print("In node: report")
    
    # 获取分析结果
    triangles = state.get("triangles", [])
    clusters = state.get("clusters", [])
    hulls = state.get("hulls", [])
    route = state.get("route", "未指定")
    ts = state.get("ts", "未指定")
    
    print("正在生成拥堵热点位置报告...")
    
    # 构建简化报告
    report_sections = []
    
    # 基本信息
    report_sections.append("# 拥堵热点位置分析报告")
    report_sections.append(f"**分析路线**: {route}")
    report_sections.append(f"**分析时间**: {ts}")
    report_sections.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_sections.append("")
    
    # 数据概览
    report_sections.append("## 数据概览")
    report_sections.append(f"- 总拥堵事件: {len(triangles)} 次")
    report_sections.append(f"- 识别聚类: {len(clusters)} 个")
    report_sections.append(f"- 拥堵热点: {len(hulls)} 个")
    report_sections.append("")
    
    # 大三角形位置信息
    if hulls:
        report_sections.append("## 拥堵热点位置详情")
        report_sections.append("以下是识别出的主要拥堵热点区域及其位置信息：")
        report_sections.append("")
        
        # 按聚类大小排序
        sorted_hulls = sorted(hulls, key=lambda x: x['cluster_size'], reverse=True)
        
        for i, hull in enumerate(sorted_hulls):
            cluster_id = hull['cluster_id']
            cluster_size = hull['cluster_size']
            kp_start, kp_end = hull['kp_range']
            time_start, time_end = hull['time_range']
            
            # 转换时间格式
            start_hour, start_min = divmod(int(time_start), 60)
            end_hour, end_min = divmod(int(time_end), 60)
            
            report_sections.extend([
                f"### 热点区域 {i+1} (聚类 {cluster_id})",
                f"- **拥堵频次**: {cluster_size} 次",
                f"- **位置范围**: KP {kp_start:.1f} - {kp_end:.1f} km",
                f"- **时间范围**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
                f"- **路段长度**: {hull['width']:.1f} km",
                f"- **时间跨度**: {hull['height']:.0f} 分钟",
                f"- **影响强度**: {hull['area']:.1f} 平方单位",
                ""
            ])
        
        # 简单总结
        report_sections.append("## 总结")
        max_hotspot = sorted_hulls[0]
        max_kp_start, max_kp_end = max_hotspot['kp_range']
        max_time_start, max_time_end = max_hotspot['time_range']
        max_start_hour, max_start_min = divmod(int(max_time_start), 60)
        max_end_hour, max_end_min = divmod(int(max_time_end), 60)
        
        report_sections.extend([
            f"**最严重拥堵热点**: KP {max_kp_start:.1f} - {max_kp_end:.1f} km",
            f"**最易拥堵时段**: {max_start_hour:02d}:{max_start_min:02d} - {max_end_hour:02d}:{max_end_min:02d}",
            f"**拥堵频次**: {max_hotspot['cluster_size']} 次",
            "",
            "**建议**: 重点关注上述位置和时段，加强交通监控和疏导。"
        ])
    else:
        report_sections.extend([
            "## 分析结果",
            "未识别出明显的拥堵热点区域，表明该路段交通状况相对良好。"
        ])
    
    report_sections.append("")
    report_sections.append("---")
    report_sections.append("*报告由交通拥堵预测系统自动生成*")
    
    final_report = "\n".join(report_sections)
    
    print("拥堵热点位置报告生成完成")
    print(f"报告长度: {len(final_report)} 字符")
    
    # 在控制台输出简要信息
    if hulls:
        print("\n=== 拥堵热点位置摘要 ===")
        for i, hull in enumerate(sorted(hulls, key=lambda x: x['cluster_size'], reverse=True)):
            kp_start, kp_end = hull['kp_range']
            time_start, time_end = hull['time_range']
            start_hour, start_min = divmod(int(time_start), 60)
            end_hour, end_min = divmod(int(time_end), 60)
            print(f"热点 {i+1}: KP {kp_start:.1f}-{kp_end:.1f} km, "
                  f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}, "
                  f"{hull['cluster_size']} 次拥堵")
    
    return {
        "final_report": final_report
    }


agent = CustomAgent(
    state_schema=TrafficState,
    impl=[
        ("chatbot", chatbot),
        ("visualization", visualization),
        ("cluster", cluster),
        ("draw_hulls", draw_hulls),
        ("report", report),
    ],
)

compiled_agent = agent.compile()

# 测试示例
if __name__ == "__main__":
    # 测试不同的用户输入
    test_inputs = [
        "请分析关越高速公路2024年的交通情况",
        "我想看看関越道路在2023年的数据",
        "分析2022年関越高速的渋滞情况",
        "请查看関越线路的最新数据",
    ]
    
    for test_input in test_inputs:
        print(f"\n=== 测试输入: {test_input} ===")
        result = compiled_agent.invoke({"user_input": test_input})
        print("Chatbot结果:", {k: v for k, v in result.items() if k in ['file_path', 'route', 'ts']})
