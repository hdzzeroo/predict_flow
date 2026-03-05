"""このファイルは `langgraph-gen` バージョン 0.0.3 を使用して生成されました。

このファイルは対応するスタブのプレースホルダー実装を提供します。

独自のロジックでプレースホルダー実装を置き換えてください。
"""


from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import pandas as pd
import os

from functions import (extract_route_and_time, generate_file_path, process_traffic_data,
                     cluster_triangles_dbscan, cluster_triangles_custom, analyze_clusters,
                     calculate_all_hull_triangles, visualize_triangles_with_hulls, create_hull_summary_report,
                     enhanced_generate_file_paths, call_llm_for_date_parsing,
                     process_multiple_traffic_data, process_direction_grouped_data, generate_unique_filename,
                     # 新增的方向感知函数
                     process_direction_aware_traffic_data, process_multiple_direction_aware_traffic_data,
                     generate_direction_aware_polygons, visualize_direction_aware_polygons,
                     extract_road_type_from_filename)
from config import config
from stub import CustomAgent
from excel_output_generator import generate_csv_prediction_output

# 新增：导入LLM分析模块
from llm_analyzer import create_batch_analyzer
from data_preparers import OutputFormatter
from visualization_comparison import visualize_all_directions_comparison, convert_hotspot_to_hull


class TrafficState(TypedDict, total=False):
    # —— 用户输入 ——
    user_input: str                     # 原始指令
    # —— 实验配置 ——
    date_matching_strategy: Optional[str]  # 日期匹配策略: "same_date" | "same_weekday"
    # —— Chatbot 节点输出 ——
    file_path: str                      # 主要 CSV 路径（向后兼容）
    file_paths: List[str]              # 多个 CSV 路径列表
    route: str                          # 路段标识（可选）
    ts: str                             # 时间字符串（可选）
    direction: Optional[str]            # 用户指定的方向（"上"/"下"/None）
    target_year: Optional[int]          # 预测目标年份
    target_month: Optional[int]         # 预测目标月份
    target_day: Optional[int]           # 预测目标日期
    # —— 可视化节点输出 ——
    triangles: List[Dict[str, Any]]     # 小三角形数据（向后兼容）
    fig_path: str                       # 图片文件路径（向后兼容，第一个方向）
    fig_paths: List[str]                # 所有方向的图片文件路径列表
    # —— 新增：按方向分组的数据 ——
    direction_data: Dict[str, Dict[str, Any]]  # 按方向分组的所有数据
    # 格式: {
    #   "上": {"triangles": [...], "fig_path": "...", "clusters": [...], "hulls": [...], "analysis": {...}},
    #   "下": {"triangles": [...], "fig_path": "...", "clusters": [...], "hulls": [...], "analysis": {...}}
    # }
    # —— LLM分析节点输出（新增，替代cluster和draw_hulls） ——
    llm_analysis: Dict[str, Dict[str, Any]]  # LLM分析结果
    # 格式: {
    #   "上": {"direction": "上", "hotspots": [...], "summary": {...}},
    #   "下": {"direction": "下", "hotspots": [...], "summary": {...}}
    # }
    # —— 聚类节点输出（已废弃，保留用于向后兼容） ——
    clusters: List[List[int]]           # 三角形索引簇
    cluster_analysis: Dict[str, Any]    # 聚类分析结果
    # —— 大三角形节点输出（已废弃，保留用于向后兼容） ——
    hulls: List[Dict[str, Any]]         # 外包大三角形
    hull_fig_path: str                  # 带外包大三角形的图片路径
    hull_summary: str                   # 外包大三角形摘要报告
    # —— 报告节点输出（终点） ——
    final_report: str                         # LLM 生成的分析报告
    csv_output_path: str                      # CSV输出路径


# スタンドアロン関数を定義
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

        # 获取日期匹配策略
        date_matching_strategy = state.get("date_matching_strategy", "same_date")
        print(f"📅 日期匹配策略: {date_matching_strategy}")

        # 首先尝试使用LLM解析
        file_paths = enhanced_generate_file_paths(
            user_input=user_input,
            use_llm=config.is_llm_available(),
            api_key=config.get_openai_api_key(),
            date_matching_strategy=date_matching_strategy
        )
        
        # 选择第一个文件路径作为主要路径
        primary_file_path = file_paths[0] if file_paths else None
        
        if primary_file_path:
            print(f"✅ LLM parsing successful, generated {len(file_paths)} candidate files")
            print(f"📁 Selected primary file: {primary_file_path}")
            
            # 使用LLM解析获取更详细的信息
            parsed_info = call_llm_for_date_parsing(
                user_input, 
                use_real_llm=config.is_llm_available(),
                api_key=config.get_openai_api_key()
            )
            
            route = parsed_info.get('route_name', 'Not specified')
            direction = parsed_info.get('direction')  # 提取方向信息
            year = parsed_info.get('year')
            month = parsed_info.get('month')
            day = parsed_info.get('day')
            
            # 构造时间字符串
            if year and month and day:
                ts = f"Year {year}, Month {month}, Day {day}"
            elif year and month:
                ts = f"Year {year}, Month {month}"
            elif year:
                ts = f"Year {year}"
            else:
                ts = "Not specified"
            
            print(f"🎯 Parsing result:")
            print(f"   Route: {route}")
            print(f"   Direction: {direction or 'Not specified (will process both directions)'}")
            print(f"   Time: {ts}")
            print(f"   Confidence: {parsed_info.get('confidence', 0):.2f}")

            return {
                "file_path": primary_file_path,
                "file_paths": file_paths,          # 添加多文件路径
                "route": route,
                "direction": direction,            # 添加方向信息
                "ts": ts,
                "target_year": year,               # 添加目标年份
                "target_month": month,             # 添加目标月份
                "target_day": day                  # 添加目标日期
            }
        else:
            raise Exception("LLM解析未返回有效文件路径")
            
    except Exception as e:
        # 不使用 fallback，直接抛出错误让问题暴露
        print(f"❌ LLM parsing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def visualization(state: TrafficState) -> dict:
    """
    读取CSV/Excel文件并进行可视化，生成三角形数据
    支持按方向分组处理：单方向模式 vs 双方向模式
    """
    print("In node: visualization")
    
    # 获取文件路径和方向信息
    file_paths = state.get("file_paths", [])
    file_path = state.get("file_path", "")
    user_direction = state.get("direction")  # 用户指定的方向
    
    # 确定要处理的文件列表
    if file_paths and len(file_paths) > 1:
        # 多文件模式
        files_to_process = file_paths
        print(f"📊 Multi-file mode: processing {len(files_to_process)} CSV files")
    elif file_paths and len(file_paths) == 1:
        # 单文件模式（从file_paths获取）
        files_to_process = file_paths
        print(f"📄 Single-file mode: processing 1 CSV file")
    elif file_path:
        # 传统单文件模式
        files_to_process = [file_path]
        print(f"📄 Legacy single-file mode: processing 1 CSV file")
    else:
        print("❌ Error: No file path provided")
        return {
            "triangles": [],
            "fig_path": "",
            "direction_data": {}
        }
    
    # 显示文件列表
    for i, fp in enumerate(files_to_process):
        print(f"  File {i+1}: {fp}")
    
    try:
        # 根据方向确定处理方式
        if user_direction:
            # 单方向模式：使用方向感知处理
            print(f"🎯 Processing single direction: {user_direction}")
            
            # 提取道路类型（假设所有文件都是同一道路）
            road_type = extract_road_type_from_filename(files_to_process[0]) if files_to_process else "関越道"
            
            if len(files_to_process) == 1:
                # 单文件单方向处理
                triangles, fig_path = process_direction_aware_traffic_data(
                    files_to_process[0], user_direction, "output",
                    state.get("target_year"), state.get("target_month"), state.get("target_day"))
            else:
                # 多文件单方向处理
                triangles, fig_path = process_multiple_direction_aware_traffic_data(
                    files_to_process, user_direction, "output",
                    state.get("target_year"), state.get("target_month"), state.get("target_day"))
            
            # 构建方向数据结构
            direction_data = {
                user_direction: {
                    "triangles": triangles,
                    "fig_path": fig_path,
                    "clusters": [],
                    "hulls": [],
                    "analysis": {}
                }
            }
            
            result = {
                "triangles": triangles,
                "fig_path": fig_path,
                "direction_data": direction_data
            }
        else:
            # 双方向模式：使用原有的方向分组处理函数
            print("🔄 Processing both directions using grouped processing")
            result = process_direction_grouped_data(
                file_paths=files_to_process,
                user_direction=user_direction,
                output_dir="output",
                target_year=state.get("target_year"),
                target_month=state.get("target_month"),
                target_day=state.get("target_day")
            )
        
        # 构建返回结果
        return_data = {
            "triangles": result["triangles"],           # 向后兼容
            "fig_path": result["fig_path"],             # 向后兼容（第一个方向的图片）
            "fig_paths": result.get("fig_paths", [result["fig_path"]]),  # 所有方向的图片路径列表
            "direction_data": result["direction_data"]  # 新增：按方向分组的数据
        }

        print(f"✅ Direction-aware visualization completed:")
        print(f"  - Total triangles: {len(result['triangles'])}")
        print(f"  - Processed directions: {list(result['direction_data'].keys())}")

        # 显示每个方向的详细信息和图片路径
        for direction, data in result['direction_data'].items():
            print(f"  - {direction} direction: {len(data['triangles'])} triangles")
            print(f"    Image: {data.get('fig_path', 'N/A')}")
        
        return return_data
        
    except Exception as e:
        print(f"❌ Direction-aware visualization processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "triangles": [],
            "fig_path": "",
            "direction_data": {}
        }


def cluster(state: TrafficState) -> dict:
    """
    对三角形进行聚类分析
    支持按方向分组的聚类：单方向模式 vs 双方向模式
    """
    print("In node: cluster")
    
    # 获取方向分组数据和传统数据
    direction_data = state.get("direction_data", {})
    triangles = state.get("triangles", [])  # 向后兼容
    
    # 确定处理模式
    if direction_data:
        # 方向分组模式：对每个方向分别进行聚类
        print(f"🧭 Direction-grouped clustering mode: processing {list(direction_data.keys())} directions")
        
        # 更新direction_data中的聚类信息
        updated_direction_data = {}
        all_clusters = []  # 用于向后兼容
        all_analysis = {"total_clusters": 0, "total_triangles": 0, "cluster_stats": []}
        
        for direction, data in direction_data.items():
            dir_triangles = data.get("triangles", [])
            
            if not dir_triangles:
                print(f"⚠️ No triangle data for {direction} direction")
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["clusters"] = []
                updated_direction_data[direction]["cluster_analysis"] = {"total_clusters": 0, "total_triangles": 0, "cluster_stats": []}
                continue
            
            print(f"📊 Clustering {len(dir_triangles)} triangles for {direction} direction...")
            
            try:
                # 使用DBSCAN聚类
                dir_clusters = cluster_triangles_dbscan(
                    dir_triangles,
                    eps=0.6,        # 邻域半径
                    min_samples=2   # 最小样本数
                )
                
                # 分析聚类结果
                dir_analysis = analyze_clusters(dir_triangles, dir_clusters)
                
                print(f"✅ {direction} direction clustering completed:")
                print(f"  - Clusters: {dir_analysis['total_clusters']}")
                print(f"  - Triangles: {dir_analysis['total_triangles']}")
                
                # 显示聚类统计信息
                for stat in dir_analysis['cluster_stats'][:3]:  # 显示前3个聚类
                    kp_range = stat['kp_range']
                    time_range = stat['time_range']
                    print(f"  - Cluster {stat['cluster_id']}: {stat['size']} triangles")
                    print(f"    KP range: {kp_range['min']:.1f} - {kp_range['max']:.1f} km")
                
                # 更新方向数据
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["clusters"] = dir_clusters
                updated_direction_data[direction]["cluster_analysis"] = dir_analysis
                
                # 合并用于向后兼容
                all_clusters.extend(dir_clusters)
                all_analysis["total_clusters"] += dir_analysis["total_clusters"]
                all_analysis["total_triangles"] += dir_analysis["total_triangles"]
                all_analysis["cluster_stats"].extend(dir_analysis["cluster_stats"])
                
            except Exception as e:
                print(f"❌ Clustering error for {direction} direction: {str(e)}")
                # 失败时的回退处理
                fallback_clusters = [[i] for i in range(len(dir_triangles))]
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["clusters"] = fallback_clusters
                updated_direction_data[direction]["cluster_analysis"] = {"total_clusters": len(fallback_clusters), "total_triangles": len(dir_triangles), "cluster_stats": []}
        
        print(f"🎯 Direction-grouped clustering summary:")
        print(f"  - Total directions processed: {len(updated_direction_data)}")
        print(f"  - Total clusters across all directions: {all_analysis['total_clusters']}")
        
        return {
            "direction_data": updated_direction_data,  # 新增：包含聚类信息的方向数据
            "clusters": all_clusters,                  # 向后兼容
            "cluster_analysis": all_analysis           # 向后兼容
        }
    
    else:
        # 传统模式：处理混合数据（向后兼容）
        print("🔄 Traditional clustering mode: processing mixed direction data")
        
        if not triangles:
            print("Warning: No triangle data available for clustering")
            return {"clusters": []}
        
        print(f"Starting clustering analysis for {len(triangles)} triangles...")
        
        try:
            # 使用DBSCAN聚类
            clusters_dbscan = cluster_triangles_dbscan(
                triangles, 
                eps=0.6,        # 邻域半径
                min_samples=2   # 最小样本数
            )
            
            # 分析聚类结果
            analysis = analyze_clusters(triangles, clusters_dbscan)
            
            print(f"Clustering analysis completed:")
            print(f"  - Total clusters: {analysis['total_clusters']}")
            print(f"  - Total triangles: {analysis['total_triangles']}")
            
            # 打印聚类统计信息
            for stat in analysis['cluster_stats'][:5]:
                kp_range = stat['kp_range']
                time_range = stat['time_range']
                print(f"  - Cluster {stat['cluster_id']}: {stat['size']} triangles")
                print(f"    KP range: {kp_range['min']:.1f} - {kp_range['max']:.1f} km")
                print(f"    Time range: {time_range['min']:.0f} - {time_range['max']:.0f} minutes")
            
            return {
                "clusters": clusters_dbscan,
                "cluster_analysis": analysis
            }
            
        except Exception as e:
            print(f"Clustering analysis error: {str(e)}")
            # 如果聚类失败，返回每个三角形作为单独聚类
            fallback_clusters = [[i] for i in range(len(triangles))]
            return {"clusters": fallback_clusters}


def draw_hulls(state: TrafficState) -> dict:
    """
    为每个聚类绘制外包大三角形
    支持按方向分组的热点标识：单方向模式 vs 双方向模式
    """
    print("In node: draw_hulls")
    
    # 获取方向分组数据和传统数据
    direction_data = state.get("direction_data", {})
    clusters = state.get("clusters", [])  # 向后兼容
    triangles = state.get("triangles", [])  # 向后兼容
    fig_path = state.get("fig_path", "")  # 向后兼容
    
    # 确定处理模式
    if direction_data:
        # 方向分组模式：对每个方向分别绘制外包大三角形
        print(f"🧭 Direction-grouped hull drawing mode: processing {list(direction_data.keys())} directions")
        
        # 更新direction_data中的热点信息
        updated_direction_data = {}
        all_hulls = []  # 用于向后兼容
        all_hull_summaries = []  # 收集所有摘要
        
        for direction, data in direction_data.items():
            dir_triangles = data.get("triangles", [])
            dir_clusters = data.get("clusters", [])
            dir_fig_path = data.get("fig_path", "")
            
            if not dir_clusters or not dir_triangles:
                print(f"⚠️ Missing cluster or triangle data for {direction} direction")
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["hulls"] = []
                updated_direction_data[direction]["hull_fig_path"] = ""
                updated_direction_data[direction]["hull_summary"] = ""
                continue
            
            print(f"🎯 Drawing hull triangles for {direction} direction ({len(dir_clusters)} clusters)...")
            
            try:
                # 1. 计算该方向的外包大三角形（传递方向参数）
                dir_hulls = calculate_all_hull_triangles(dir_triangles, dir_clusters, direction)
                
                if not dir_hulls:
                    print(f"⚠️ No valid hull triangles found for {direction} direction")
                    updated_direction_data[direction] = data.copy()
                    updated_direction_data[direction]["hulls"] = []
                    updated_direction_data[direction]["hull_fig_path"] = ""
                    updated_direction_data[direction]["hull_summary"] = ""
                    continue
                
                # 2. 创建该方向的可视化图片
                # 提取文件路径信息用于生成唯一文件名
                direction_files = direction_data.get(direction, {}).get("triangles", [])
                source_files = []
                if direction_files:
                    # 从三角形数据中提取原始文件信息
                    for triangle in direction_files[:1]:  # 只需要第一个三角形的文件信息
                        if 'source_file' in triangle:
                            source_files.append(triangle['source_file'] + '.csv')
                
                dir_hull_fig_name = generate_unique_filename(
                    "hulls", source_files, direction,
                    target_year=state.get("target_year"),
                    target_month=state.get("target_month"),
                    target_day=state.get("target_day")
                )
                dir_hull_fig_path = os.path.join(os.path.dirname(dir_fig_path) if dir_fig_path else "output", dir_hull_fig_name)
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(dir_hull_fig_path) if os.path.dirname(dir_hull_fig_path) else '.', exist_ok=True)
                
                # 绘制该方向的外包大三角形可视化图
                final_dir_fig_path = visualize_triangles_with_hulls(
                    triangles=dir_triangles,
                    hulls=dir_hulls,
                    output_path=dir_hull_fig_path,
                    figsize=(16, 12)
                )
                
                # 3. 生成该方向的摘要报告
                dir_hull_summary = create_hull_summary_report(dir_hulls)
                
                print(f"✅ {direction} direction hull drawing completed:")
                print(f"  - Drew {len(dir_hulls)} hull triangles")
                print(f"  - Covered {sum(len(cluster) for cluster in dir_clusters)} original triangles")
                print(f"  - Image saved to: {final_dir_fig_path}")
                
                # 显示该方向的外包大三角形信息
                for hull in dir_hulls[:3]:  # 显示前3个
                    kp_start, kp_end = hull['kp_range']
                    print(f"  - Cluster {hull['cluster_id']}: {hull['cluster_size']} triangles")
                    print(f"    KP range: {kp_start:.1f} - {kp_end:.1f} km")
                
                # 更新方向数据
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["hulls"] = dir_hulls
                updated_direction_data[direction]["hull_fig_path"] = final_dir_fig_path
                updated_direction_data[direction]["hull_summary"] = dir_hull_summary
                
                # 合并用于向后兼容
                all_hulls.extend(dir_hulls)
                all_hull_summaries.append(f"## {direction}方向\n{dir_hull_summary}")
                
            except Exception as e:
                print(f"❌ Hull drawing error for {direction} direction: {str(e)}")
                import traceback
                traceback.print_exc()
                updated_direction_data[direction] = data.copy()
                updated_direction_data[direction]["hulls"] = []
                updated_direction_data[direction]["hull_fig_path"] = ""
                updated_direction_data[direction]["hull_summary"] = ""
        
        print(f"🎯 Direction-grouped hull drawing summary:")
        print(f"  - Total directions processed: {len(updated_direction_data)}")
        print(f"  - Total hull triangles across all directions: {len(all_hulls)}")
        
        # 合并所有方向的摘要报告
        combined_hull_summary = "\n\n".join(all_hull_summaries) if all_hull_summaries else ""
        
        return {
            "direction_data": updated_direction_data,  # 新增：包含热点信息的方向数据
            "hulls": all_hulls,                        # 向后兼容
            "hull_fig_path": updated_direction_data.get(list(updated_direction_data.keys())[0], {}).get("hull_fig_path", "") if updated_direction_data else "",  # 向后兼容：取第一个方向的图片路径
            "hull_summary": combined_hull_summary      # 向后兼容：合并的摘要报告
        }
    
    else:
        # 传统模式：处理混合数据（向后兼容）
        print("🔄 Traditional hull drawing mode: processing mixed direction data")
        
        if not clusters or not triangles:
            print("Warning: Missing cluster or triangle data, cannot draw hull triangles")
            return {"hulls": []}
        
        print(f"Starting to draw hull triangles for {len(clusters)} clusters...")
        
        try:
            # 1. 计算所有聚类的外包大三角形
            hulls = calculate_all_hull_triangles(triangles, clusters)
            
            if not hulls:
                print("No valid hull triangles found")
                return {"hulls": []}
            
            # 2. 创建带有外包大三角形的可视化图片
            # 提取文件路径信息用于生成唯一文件名
            source_files = []
            if triangles:
                # 从三角形数据中提取原始文件信息
                for triangle in triangles[:1]:  # 只需要第一个三角形的文件信息
                    if 'source_file' in triangle:
                        source_files.append(triangle['source_file'] + '.csv')
            
            hull_fig_name = generate_unique_filename(
                "hulls_traditional", source_files,
                target_year=state.get("target_year"),
                target_month=state.get("target_month"),
                target_day=state.get("target_day")
            )
            hull_fig_path = os.path.join(os.path.dirname(fig_path) if fig_path else "output", hull_fig_name)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(hull_fig_path) if os.path.dirname(hull_fig_path) else '.', exist_ok=True)
            
            # 绘制带外包大三角形的可视化图
            final_fig_path = visualize_triangles_with_hulls(
                triangles=triangles,
                hulls=hulls,
                output_path=hull_fig_path,
                figsize=(16, 12)
            )
            
            # 3. 生成外包大三角形的摘要报告
            hull_summary = create_hull_summary_report(hulls)
            
            print(f"Hull triangle drawing completed:")
            print(f"  - Drew {len(hulls)} hull triangles")
            print(f"  - Covered {sum(len(cluster) for cluster in clusters)} original triangles")
            print(f"  - Image saved to: {final_fig_path}")
            
            # 打印每个外包大三角形的信息
            for hull in hulls[:5]:  # 只显示前5个
                kp_start, kp_end = hull['kp_range']
                print(f"  - Cluster {hull['cluster_id']}: {hull['cluster_size']} triangles")
                print(f"    KP range: {kp_start:.1f} - {kp_end:.1f} km")
                print(f"    Coverage area: {hull['area']:.1f} square units")
            
            return {
                "hulls": hulls,
                "hull_fig_path": final_fig_path,
                "hull_summary": hull_summary
            }
            
        except Exception as e:
            print(f"Hull triangle drawing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"hulls": []}


def analyze_with_llm(state: TrafficState) -> dict:
    """
    使用LLM分析拥堵热点，替代cluster和draw_hulls节点
    支持按方向分组的分析
    """
    print("In node: analyze_with_llm")

    direction_data = state.get("direction_data", {})
    file_paths = state.get("file_paths", [])

    if not direction_data:
        print("❌ Error: No direction data available for LLM analysis")
        return {"llm_analysis": {}}

    print(f"\n{'='*70}")
    print("🤖 Starting LLM-based hotspot analysis")
    print(f"{'='*70}")

    # 获取CSV文件名
    csv_files = [os.path.basename(fp) for fp in file_paths] if file_paths else []

    # 检查LLM是否可用
    api_key = config.get_openai_api_key()
    if not api_key:
        print("⚠️ Warning: No OpenAI API key configured")
        print("⚠️ Will use fallback analysis method")

    # 创建批量分析器
    try:
        batch_analyzer = create_batch_analyzer(
            api_key=api_key if api_key else "",
            model=config.openai_model
        )

        # 分析所有方向
        llm_analysis = batch_analyzer.analyze_all_directions(
            direction_data=direction_data,
            csv_files=csv_files
        )

        # 打印总体摘要
        print(f"\n{'='*70}")
        print("📊 LLM Analysis Summary")
        print(f"{'='*70}")

        total_hotspots = 0
        for direction, analysis in llm_analysis.items():
            hotspot_count = len(analysis.get('hotspots', []))
            total_hotspots += hotspot_count
            confidence = analysis.get('summary', {}).get('analysis_confidence', 0)
            print(f"  {direction}方向: {hotspot_count} hotspots (confidence: {confidence:.2f})")

        print(f"\n  Total hotspots identified: {total_hotspots}")
        print(f"{'='*70}\n")

        # 生成对比可视化图
        print(f"\n🎨 Generating comparison visualizations...")
        try:
            comparison_paths = visualize_all_directions_comparison(
                direction_data=direction_data,
                llm_analysis=llm_analysis,
                output_dir="output",
                road_name=state.get('route', '関越道')
            )

            print(f"✅ Comparison visualizations generated:")
            for direction, path in comparison_paths.items():
                print(f"  {direction}方向: {path}")

        except Exception as e:
            print(f"⚠️ Failed to generate comparison visualizations: {str(e)}")
            comparison_paths = {}

        return {
            "llm_analysis": llm_analysis,
            "comparison_fig_paths": list(comparison_paths.values())  # 新增：对比图路径
        }

    except Exception as e:
        print(f"❌ LLM analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # 返回空结果
        empty_analysis = {}
        for direction in direction_data.keys():
            empty_analysis[direction] = {
                "direction": direction,
                "hotspots": [],
                "summary": {
                    "total_hotspots": 0,
                    "most_severe_hotspot_id": None,
                    "analysis_confidence": 0.0
                }
            }

        return {"llm_analysis": empty_analysis}


def report(state: TrafficState) -> dict:
    """
    基于LLM分析结果生成报告
    支持按方向分组的报告生成：单方向模式 vs 双方向模式
    """
    print("In node: report")

    # 获取LLM分析结果（优先）
    llm_analysis = state.get("llm_analysis", {})

    # 获取方向分组数据和传统数据（向后兼容）
    direction_data = state.get("direction_data", {})
    triangles = state.get("triangles", [])  # 向后兼容
    clusters = state.get("clusters", [])    # 向后兼容
    hulls = state.get("hulls", [])          # 向后兼容
    route = state.get("route", "未指定")
    ts = state.get("ts", "未指定")

    print("Generating traffic congestion hotspot location report based on LLM analysis...")
    
    # 确定处理模式
    if llm_analysis:
        # LLM分析模式：使用LLM识别的热点生成报告
        print(f"🤖 LLM-based report mode: generating report for {list(llm_analysis.keys())} directions")

        report_sections = []

        # 基本信息
        report_sections.append("# Traffic Congestion Hotspot Location Analysis Report")
        report_sections.append(f"**Analysis Route**: {route}")
        report_sections.append(f"**Analysis Time**: {ts}")
        report_sections.append(f"**Analysis Mode**: LLM-based Direction-Grouped Analysis")
        report_sections.append(f"**Analyzed Directions**: {', '.join(llm_analysis.keys())}")
        report_sections.append(f"**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")

        # 整体数据概览
        total_triangles = sum(len(direction_data.get(d, {}).get("triangles", [])) for d in llm_analysis.keys())
        total_hotspots = sum(len(analysis.get("hotspots", [])) for analysis in llm_analysis.values())
        avg_confidence = sum(analysis.get("summary", {}).get("analysis_confidence", 0) for analysis in llm_analysis.values()) / len(llm_analysis) if llm_analysis else 0

        report_sections.append("## Overall Data Overview")
        report_sections.append(f"- Total Congestion Events: {total_triangles}")
        report_sections.append(f"- Total Identified Hotspots (by LLM): {total_hotspots}")
        report_sections.append(f"- Average Analysis Confidence: {avg_confidence:.2f}")
        report_sections.append("")

        # 分方向详细分析（基于LLM结果）
        direction_summaries = []

        for direction, analysis in llm_analysis.items():
            dir_triangles = direction_data.get(direction, {}).get("triangles", [])
            hotspots = analysis.get("hotspots", [])
            summary = analysis.get("summary", {})

            direction_name = f"{direction}り方向" if direction in ["上", "下"] else f"{direction} Direction"

            report_sections.append(f"## {direction_name} Analysis (LLM-based)")

            # 该方向的数据概览
            report_sections.append(f"### Data Overview - {direction_name}")
            report_sections.append(f"- Congestion Events: {len(dir_triangles)}")
            report_sections.append(f"- Identified Hotspots: {len(hotspots)}")
            report_sections.append(f"- Analysis Confidence: {summary.get('analysis_confidence', 0):.2f}")
            report_sections.append("")

            # 该方向的热点详情
            if hotspots:
                report_sections.append(f"### Hotspot Details - {direction_name}")
                report_sections.append(f"The following hotspots were identified by LLM analysis for {direction_name}:")
                report_sections.append("")

                # 按频次排序
                sorted_hotspots = sorted(hotspots, key=lambda x: x['frequency'], reverse=True)

                for i, hotspot in enumerate(sorted_hotspots):
                    hotspot_id = hotspot['hotspot_id']
                    frequency = hotspot['frequency']
                    kp_start, kp_end = hotspot['kp_range']
                    time_start, time_end = hotspot['time_range']
                    severity = hotspot['severity']
                    description = hotspot.get('description', '')

                    # 转换时间格式
                    start_hour, start_min = divmod(int(time_start), 60)
                    end_hour, end_min = divmod(int(time_end), 60)

                    report_sections.extend([
                        f"#### {direction_name} Hotspot {i+1} (ID: {hotspot_id})",
                        f"- **Congestion Frequency**: {frequency} times",
                        f"- **Location Range**: KP {kp_start:.1f} - {kp_end:.1f} km",
                        f"- **Time Range**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
                        f"- **Severity**: {severity}",
                        f"- **Description**: {description}",
                        ""
                    ])

                # 该方向的最严重热点
                max_hotspot = sorted_hotspots[0]
                max_kp_start, max_kp_end = max_hotspot['kp_range']
                max_time_start, max_time_end = max_hotspot['time_range']
                max_start_hour, max_start_min = divmod(int(max_time_start), 60)
                max_end_hour, max_end_min = divmod(int(max_time_end), 60)

                direction_summaries.append({
                    "direction": direction_name,
                    "hotspots": len(hotspots),
                    "events": len(dir_triangles),
                    "max_hotspot": f"KP {max_kp_start:.1f}-{max_kp_end:.1f} km",
                    "max_time": f"{max_start_hour:02d}:{max_start_min:02d}-{max_end_hour:02d}:{max_end_min:02d}",
                    "max_frequency": max_hotspot['frequency'],
                    "confidence": summary.get('analysis_confidence', 0)
                })

            else:
                report_sections.extend([
                    f"### Analysis Results - {direction_name}",
                    f"No obvious congestion hotspots detected for {direction_name} by LLM analysis.",
                    ""
                ])

                direction_summaries.append({
                    "direction": direction_name,
                    "hotspots": 0,
                    "events": len(dir_triangles),
                    "max_hotspot": "None",
                    "max_time": "N/A",
                    "max_frequency": 0,
                    "confidence": summary.get('analysis_confidence', 0)
                })

        # 方向对比分析
        if len(direction_summaries) > 1:
            report_sections.append("## Direction Comparison")
            report_sections.append("Comparison of congestion patterns between different directions:")
            report_sections.append("")

            for summary_info in direction_summaries:
                report_sections.extend([
                    f"**{summary_info['direction']}**:",
                    f"- Hotspots: {summary_info['hotspots']}, Events: {summary_info['events']}",
                    f"- Confidence: {summary_info['confidence']:.2f}",
                    f"- Most Severe: {summary_info['max_hotspot']} ({summary_info['max_frequency']} times)",
                    ""
                ])

        # 综合建议
        report_sections.append("## Summary and Recommendations")

        if direction_summaries:
            # 找出最严重的方向
            most_severe_direction = max(direction_summaries, key=lambda x: x['max_frequency'])

            if most_severe_direction['max_frequency'] > 0:
                report_sections.extend([
                    f"**Most Congestion-Prone Direction**: {most_severe_direction['direction']}",
                    f"**Most Severe Hotspot**: {most_severe_direction['max_hotspot']}",
                    f"**Peak Congestion Time**: {most_severe_direction['max_time']}",
                    f"**Congestion Frequency**: {most_severe_direction['max_frequency']} times",
                    f"**Analysis Confidence**: {most_severe_direction['confidence']:.2f}",
                    "",
                    "**Recommendations (based on LLM analysis)**:",
                    f"1. Focus monitoring and management on {most_severe_direction['direction']}",
                    f"2. Strengthen traffic control at {most_severe_direction['max_hotspot']}",
                    f"3. Consider traffic flow optimization during peak hours {most_severe_direction['max_time']}",
                    "4. Implement direction-specific traffic management strategies"
                ])
            else:
                report_sections.extend([
                    "**Overall Assessment**: No significant congestion hotspots detected by LLM analysis.",
                    "**Recommendation**: Maintain current traffic management practices and continue monitoring."
                ])

        # 在控制台输出分方向摘要
        print("\n=== LLM-based Hotspot Summary ===")
        for summary_info in direction_summaries:
            print(f"{summary_info['direction']}: {summary_info['hotspots']} hotspots, {summary_info['events']} events")
            if summary_info['max_frequency'] > 0:
                print(f"  Most severe: {summary_info['max_hotspot']} ({summary_info['max_frequency']} times)")
                print(f"  Confidence: {summary_info['confidence']:.2f}")

    elif direction_data:
        # 传统模式（向后兼容）：使用聚类和外包大三角形
        print(f"🧭 Direction-grouped report mode: generating report for {list(direction_data.keys())} directions")

        report_sections = []

        # 基本信息
        report_sections.append("# Traffic Congestion Hotspot Location Analysis Report")
        report_sections.append(f"**Analysis Route**: {route}")
        report_sections.append(f"**Analysis Time**: {ts}")
        report_sections.append(f"**Analysis Mode**: Direction-Grouped Analysis")
        report_sections.append(f"**Analyzed Directions**: {', '.join(direction_data.keys())}")
        report_sections.append(f"**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")

        # 整体数据概览
        total_triangles = sum(len(data.get("triangles", [])) for data in direction_data.values())
        total_clusters = sum(len(data.get("clusters", [])) for data in direction_data.values())
        total_hulls = sum(len(data.get("hulls", [])) for data in direction_data.values())

        report_sections.append("## Overall Data Overview")
        report_sections.append(f"- Total Congestion Events: {total_triangles}")
        report_sections.append(f"- Total Identified Clusters: {total_clusters}")
        report_sections.append(f"- Total Congestion Hotspots: {total_hulls}")
        report_sections.append("")
        
        # 分方向详细分析
        direction_summaries = []  # 收集方向摘要用于最终对比
        
        for direction, data in direction_data.items():
            dir_triangles = data.get("triangles", [])
            dir_clusters = data.get("clusters", [])
            dir_hulls = data.get("hulls", [])
            
            direction_name = f"{direction}り方向" if direction in ["上", "下"] else f"{direction} Direction"
            
            report_sections.append(f"## {direction_name} Analysis")
            
            # 该方向的数据概览
            report_sections.append(f"### Data Overview - {direction_name}")
            report_sections.append(f"- Congestion Events: {len(dir_triangles)}")
            report_sections.append(f"- Identified Clusters: {len(dir_clusters)}")
            report_sections.append(f"- Congestion Hotspots: {len(dir_hulls)}")
            report_sections.append("")
            
            # 该方向的热点详情
            if dir_hulls:
                report_sections.append(f"### Hotspot Details - {direction_name}")
                report_sections.append(f"The following are the major congestion hotspot areas identified for {direction_name}:")
                report_sections.append("")
                
                # 按聚类大小排序
                sorted_dir_hulls = sorted(dir_hulls, key=lambda x: x['cluster_size'], reverse=True)
                
                for i, hull in enumerate(sorted_dir_hulls):
                    cluster_id = hull['cluster_id']
                    cluster_size = hull['cluster_size']
                    kp_start, kp_end = hull['kp_range']
                    time_start, time_end = hull['time_range']
                    
                    # 转换时间格式
                    start_hour, start_min = divmod(int(time_start), 60)
                    end_hour, end_min = divmod(int(time_end), 60)
                    
                    report_sections.extend([
                        f"#### {direction_name} Hotspot {i+1} (Cluster {cluster_id})",
                        f"- **Congestion Frequency**: {cluster_size} times",
                        f"- **Location Range**: KP {kp_start:.1f} - {kp_end:.1f} km",
                        f"- **Time Range**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
                        f"- **Section Length**: {hull['width']:.1f} km",
                        f"- **Duration**: {hull['height']:.0f} minutes",
                        f"- **Impact Intensity**: {hull['area']:.1f} square units",
                        ""
                    ])
                
                # 该方向的最严重热点
                max_dir_hotspot = sorted_dir_hulls[0]
                max_kp_start, max_kp_end = max_dir_hotspot['kp_range']
                max_time_start, max_time_end = max_dir_hotspot['time_range']
                max_start_hour, max_start_min = divmod(int(max_time_start), 60)
                max_end_hour, max_end_min = divmod(int(max_time_end), 60)
                
                direction_summaries.append({
                    "direction": direction_name,
                    "hotspots": len(dir_hulls),
                    "events": len(dir_triangles),
                    "max_hotspot": f"KP {max_kp_start:.1f}-{max_kp_end:.1f} km",
                    "max_time": f"{max_start_hour:02d}:{max_start_min:02d}-{max_end_hour:02d}:{end_min:02d}",
                    "max_frequency": max_dir_hotspot['cluster_size']
                })
                
            else:
                report_sections.extend([
                    f"### Analysis Results - {direction_name}",
                    f"No obvious congestion hotspots detected for {direction_name}, indicating relatively good traffic conditions.",
                    ""
                ])
                
                direction_summaries.append({
                    "direction": direction_name,
                    "hotspots": 0,
                    "events": len(dir_triangles),
                    "max_hotspot": "None",
                    "max_time": "N/A",
                    "max_frequency": 0
                })
        
        # 方向对比分析
        if len(direction_summaries) > 1:
            report_sections.append("## Direction Comparison")
            report_sections.append("Comparison of congestion patterns between different directions:")
            report_sections.append("")
            
            for summary in direction_summaries:
                report_sections.extend([
                    f"**{summary['direction']}**:",
                    f"- Hotspots: {summary['hotspots']}, Events: {summary['events']}",
                    f"- Most Severe: {summary['max_hotspot']} ({summary['max_frequency']} times)",
                    ""
                ])
        
        # 综合建议
        report_sections.append("## Summary and Recommendations")
        
        if direction_summaries:
            # 找出最严重的方向
            most_severe_direction = max(direction_summaries, key=lambda x: x['max_frequency'])
            
            if most_severe_direction['max_frequency'] > 0:
                report_sections.extend([
                    f"**Most Congestion-Prone Direction**: {most_severe_direction['direction']}",
                    f"**Most Severe Hotspot**: {most_severe_direction['max_hotspot']}",
                    f"**Peak Congestion Time**: {most_severe_direction['max_time']}",
                    f"**Congestion Frequency**: {most_severe_direction['max_frequency']} times",
                    "",
                    "**Recommendations**:",
                    f"1. Focus monitoring and management on {most_severe_direction['direction']}",
                    f"2. Strengthen traffic control at {most_severe_direction['max_hotspot']}",
                    f"3. Consider traffic flow optimization during peak hours {most_severe_direction['max_time']}",
                    "4. Implement direction-specific traffic management strategies"
                ])
            else:
                report_sections.extend([
                    "**Overall Assessment**: No significant congestion hotspots detected across all directions.",
                    "**Recommendation**: Maintain current traffic management practices and continue monitoring."
                ])
        
        # 在控制台输出分方向摘要
        print("\n=== Direction-Grouped Hotspot Summary ===")
        for summary in direction_summaries:
            print(f"{summary['direction']}: {summary['hotspots']} hotspots, {summary['events']} events")
            if summary['max_frequency'] > 0:
                print(f"  Most severe: {summary['max_hotspot']} ({summary['max_frequency']} times)")
        
    else:
        # 传统模式：处理混合数据（向后兼容）
        print("🔄 Traditional report mode: generating mixed direction report")
        
        report_sections = []
        
        # 基本信息
        report_sections.append("# Traffic Congestion Hotspot Location Analysis Report")
        report_sections.append(f"**Analysis Route**: {route}")
        report_sections.append(f"**Analysis Time**: {ts}")
        report_sections.append(f"**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")
        
        # 数据概览
        report_sections.append("## Data Overview")
        report_sections.append(f"- Total Congestion Events: {len(triangles)}")
        report_sections.append(f"- Identified Clusters: {len(clusters)}")
        report_sections.append(f"- Congestion Hotspots: {len(hulls)}")
        report_sections.append("")
        
        # 大三角形位置信息
        if hulls:
            report_sections.append("## Congestion Hotspot Location Details")
            report_sections.append("The following are the identified major congestion hotspot areas and their location information:")
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
                    f"### Hotspot Area {i+1} (Cluster {cluster_id})",
                    f"- **Congestion Frequency**: {cluster_size} times",
                    f"- **Location Range**: KP {kp_start:.1f} - {kp_end:.1f} km",
                    f"- **Time Range**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
                    f"- **Section Length**: {hull['width']:.1f} km",
                    f"- **Duration**: {hull['height']:.0f} minutes",
                    f"- **Impact Intensity**: {hull['area']:.1f} square units",
                    ""
                ])
            
            # 简单总结
            report_sections.append("## Summary")
            max_hotspot = sorted_hulls[0]
            max_kp_start, max_kp_end = max_hotspot['kp_range']
            max_time_start, max_time_end = max_hotspot['time_range']
            max_start_hour, max_start_min = divmod(int(max_time_start), 60)
            max_end_hour, max_end_min = divmod(int(max_time_end), 60)
            
            report_sections.extend([
                f"**Most Severe Congestion Hotspot**: KP {max_kp_start:.1f} - {max_kp_end:.1f} km",
                f"**Most Congestion-Prone Time Period**: {max_start_hour:02d}:{max_start_min:02d} - {max_end_hour:02d}:{max_end_min:02d}",
                f"**Congestion Frequency**: {max_hotspot['cluster_size']} times",
                "",
                "**Recommendation**: Focus on the above locations and time periods, strengthen traffic monitoring and management."
            ])
        else:
            report_sections.extend([
                "## Analysis Results",
                "No obvious congestion hotspots detected, indicating relatively good traffic conditions on this route section."
            ])
        
        # 在控制台输出简要信息
        if hulls:
            print("\n=== Traffic Congestion Hotspot Location Summary ===")
            for i, hull in enumerate(sorted(hulls, key=lambda x: x['cluster_size'], reverse=True)):
                kp_start, kp_end = hull['kp_range']
                time_start, time_end = hull['time_range']
                start_hour, start_min = divmod(int(time_start), 60)
                end_hour, end_min = divmod(int(time_end), 60)
                print(f"Hotspot {i+1}: KP {kp_start:.1f}-{kp_end:.1f} km, "
                      f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}, "
                      f"{hull['cluster_size']} congestion events")
    
    # 添加报告结尾
    report_sections.append("")
    report_sections.append("---")
    report_sections.append("*Report automatically generated by Traffic Congestion Prediction System*")
    
    final_report = "\n".join(report_sections)
    
    print("Traffic congestion hotspot location report generation completed")
    print(f"Report length: {len(final_report)} characters")
    
    # 生成Excel格式输出
    try:
        # 构建完整的工作流结果用于Excel输出
        excel_workflow_result = {
            "route": route,
            "ts": ts,
            "target_year": state.get("target_year"),      # 添加目标年份
            "target_month": state.get("target_month"),    # 添加目标月份
            "target_day": state.get("target_day"),        # 添加目标日期
            "direction_data": direction_data if direction_data else {},
            "triangles": triangles,  # 向后兼容
            "clusters": clusters,    # 向后兼容
            "hulls": hulls          # 向后兼容
        }
        
        csv_output_path = generate_csv_prediction_output(excel_workflow_result)
        print(f"📊 CSV格式预测结果已生成: {csv_output_path}")
        
        return {
            "final_report": final_report,
            "csv_output_path": csv_output_path
        }
    except Exception as e:
        print(f"⚠️ CSV输出生成失败: {e}")
        # 即使CSV输出失败，也要返回原始报告
        return {
            "final_report": final_report,
            "csv_output_path": None
        }


agent = CustomAgent(
    state_schema=TrafficState,
    impl=[
        ("chatbot", chatbot),
        ("visualization", visualization),
        ("analyze_with_llm", analyze_with_llm),  # 新增：替代cluster和draw_hulls
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
