import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, List, Tuple, Any, Union
from matplotlib.patches import Polygon

from dotenv import load_dotenv
load_dotenv()

# 聚类相关导入 (可选依赖)
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    CLUSTERING_AVAILABLE = True
except ImportError:
    print("警告: sklearn或scipy未安装，聚类功能将使用简化版本")
    CLUSTERING_AVAILABLE = False



# 在functions.py中添加新的增强函数

import glob
import json
from datetime import datetime

def call_openai_api_for_parsing(user_input: str, api_key: str = None) -> Dict[str, Any]:
    """
    使用OpenAI API进行真正的LLM解析
    """
    try:
        import openai
        import json
        
        # 获取API密钥
        api_key_to_use = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key_to_use:
            print("⚠️ 未找到OpenAI API密钥，降级使用本地解析")
            return call_fallback_parsing(user_input)
        
        system_prompt = """你是一个专业的日本交通数据查询解析助手。请从用户输入中精确提取以下信息：

## 任务
从用户的自然语言查询中提取：
1. **道路名称** - 标准化的日本道路名称
2. **年份** - 具体年份（如果未指定则为null）
3. **月份** - 具体月份（1-12）
4. **日期** - 具体日期（1-31），如果没有指定则为null

## 支持的道路名称标准化
- 関越道/関越高速/関越高速道路 → "関越道"
- 東名道/東名高速/東名高速道路 → "東名道"  
- 中央道/中央高速/中央高速道路 → "中央道"
- 東北道/東北高速/東北高速道路 → "東北道"
- 首都高速/首都高 → "首都高速"

## 时间处理
- 日本年号转换：令和X年 = 2018+X年，平成X年 = 1988+X年
- 月份提取：X月/XX月 → 数字
- 日期提取：X日/XX日 → 数字

## 输出格式
严格按照以下JSON格式输出，不要包含任何其他文字：

{
    "route_name": "标准化道路名称",
    "year": 年份数字或null,
    "month": 月份数字(1-12)或null,
    "day": 日期数字(1-31)或null,
    "confidence": 0.0-1.0的置信度
}

## 示例
输入："4月23日関越的交通状况"
输出：{"route_name": "関越道", "year": null, "month": 4, "day": 23, "confidence": 0.95}

输入："2023年東北道5月的数据"
输出：{"route_name": "東北道", "year": 2023, "month": 5, "day": null, "confidence": 0.92}"""

        print("🤖 正在调用OpenAI API...")
        
        # 调用OpenAI API (新版本接口)
        client = openai.OpenAI(api_key=api_key_to_use)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用较便宜的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,  # 降低随机性
            max_tokens=200,   # 限制token数量
            timeout=10        # 设置超时
        )
        
        llm_response = response.choices[0].message.content.strip()
        print(f"🤖 LLM原始响应: {llm_response}")
        
        # 尝试解析JSON响应
        try:
            # 清理响应中可能的markdown格式
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1]
            
            result = json.loads(llm_response)
            
            # 验证结果格式
            required_keys = ["route_name", "year", "month", "day", "confidence"]
            if all(key in result for key in required_keys):
                print("✅ OpenAI API解析成功")
                return result
            else:
                print("⚠️ LLM响应格式不完整，使用降级解析")
                return call_fallback_parsing(user_input)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ LLM响应JSON解析失败: {e}")
            print(f"原始响应: {llm_response}")
            return call_fallback_parsing(user_input)
        
    except ImportError:
        print("⚠️ openai包未安装，请运行: pip install openai")
        return call_fallback_parsing(user_input)
    except Exception as e:
        print(f"⚠️ OpenAI API调用失败: {str(e)}")
        return call_fallback_parsing(user_input)


def call_fallback_parsing(user_input: str) -> Dict[str, Any]:
    """
    降级解析：当LLM不可用时使用的本地正则表达式方法
    """
    import re
    
    print("🔧 使用本地正则表达式解析...")
    
    # 基本的解析逻辑（模拟LLM理解）
    result = {
        "route_name": None,
        "year": None, 
        "month": None,
        "day": None,
        "confidence": 0.7
    }
    
    # 路线识别
    route_keywords = {
        "関越": "関越道", "東名": "東名道", "中央": "中央道", 
        "東北": "東北道", "首都": "首都高速"
    }
    
    for keyword, standard_name in route_keywords.items():
        if keyword in user_input:
            result["route_name"] = standard_name
            result["confidence"] = 0.9
            break
    
    # 月份识别
    month_match = re.search(r'(\d{1,2})月', user_input)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            result["month"] = month
            result["confidence"] = min(result["confidence"] + 0.1, 1.0)
    
    # 日期识别
    day_match = re.search(r'(\d{1,2})日', user_input)
    if day_match:
        day = int(day_match.group(1))
        if 1 <= day <= 31:
            result["day"] = day
            result["confidence"] = min(result["confidence"] + 0.1, 1.0)
    
    # 年份识别
    year_patterns = [r'(\d{4})年?', r'令和(\d+)年', r'平成(\d+)年']
    for pattern in year_patterns:
        match = re.search(pattern, user_input)
        if match:
            if '令和' in pattern:
                result["year"] = 2018 + int(match.group(1))
            elif '平成' in pattern:
                result["year"] = 1988 + int(match.group(1))
            else:
                year = int(match.group(1))
                if 2000 <= year <= 2030:
                    result["year"] = year
            break
    
    return result


def call_llm_for_date_parsing(user_input: str, use_real_llm: bool = True, api_key: str = None) -> Dict[str, Any]:
    """
    使用LLM解析用户输入中的日期和道路信息
    
    Args:
        user_input: 用户输入的查询文本
        use_real_llm: 是否使用真实的LLM API（True）还是降级方法（False）
        api_key: OpenAI API密钥（可选）
    
    Returns:
        解析结果字典
    """
    if use_real_llm:
        return call_openai_api_for_parsing(user_input, api_key)
    else:
        return call_fallback_parsing(user_input)

def enhanced_generate_file_paths(user_input: str, 
                                base_dir: str = "/home/dizhihuang/graduate/predict_workflow/data/processed_data",
                                use_llm: bool = True,
                                api_key: str = None) -> List[str]:
    """
    增强的文件路径生成函数 - 返回多个年份的文件路径
    
    Args:
        user_input: 用户输入的自然语言查询
        base_dir: 数据文件基础目录  
        use_llm: 是否使用LLM进行语义解析
    
    Returns:
        匹配的文件路径列表（通常包含3个年份的文件）
    """
    print(f"📝 解析用户输入: {user_input}")
    
    # 第1步：解析用户输入
    if use_llm:
        print("🤖 使用LLM语义解析...")
        parsed_info = call_llm_for_date_parsing(user_input, use_real_llm=True, api_key=api_key)
    else:
        print("🔧 使用正则表达式解析...")
        parsed_info = call_llm_for_date_parsing(user_input, use_real_llm=False)
    
    print(f"🎯 解析结果: {parsed_info}")
    
    # 第2步：提取信息并设置默认值
    route_name = parsed_info.get("route_name") or "関越道"
    year = parsed_info.get("year")
    month = parsed_info.get("month") or datetime.now().month
    day = parsed_info.get("day")
    
    # 第3步：确定目标年份（总是返回3个年份）
    if year:
        target_years = [year - 1, year - 2, year - 3]
    else:
        current_year = datetime.now().year
        target_years = [current_year - 1, current_year - 2, current_year - 3]
    
    print(f"🗓️  目标年份: {target_years}")
    print(f"🛣️  路线: {route_name}, 📅 月份: {month}, 📆 日期: {day or '整月'}")
    
    # 第4步：构建文件路径并搜索
    all_paths = []
    
    for target_year in target_years:
        if day:
            # 精确匹配特定日期：東北道_2021_04-23.csv
            filename = f"{route_name}_{target_year}_{month:02d}-{day:02d}.csv"
        else:
            # 匹配整个月的第一个文件：東北道_2021_04-*.csv
            pattern = f"{route_name}_{target_year}_{month:02d}-*.csv"
            matching_files = glob.glob(os.path.join(base_dir, pattern))
            if matching_files:
                # 取第一个匹配的文件
                filename = os.path.basename(sorted(matching_files)[0])
            else:
                # 如果没有找到，生成一个默认的文件名
                filename = f"{route_name}_{target_year}_{month:02d}-01.csv"
        
        file_path = os.path.join(base_dir, filename)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            all_paths.append(file_path)
            print(f"✅ 找到文件: {filename}")
        else:
            # 即使文件不存在也添加路径，让调用者知道期望的文件名
            all_paths.append(file_path)
            print(f"⚠️  文件不存在但已生成路径: {filename}")
    
    print(f"📁 共生成 {len(all_paths)} 个文件路径")
    return all_paths



# 为了兼容性，保持原有函数不变，但添加新功能
def generate_file_path_enhanced(route: Optional[str], time: Optional[str], 
                              month: Optional[int] = None, day: Optional[int] = None,
                              return_multiple: bool = False) -> Union[str, List[str]]:
    """
    增强版的原始函数，支持月份日期和多文件返回
    """
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    
    # 标准化路线名称
    route_mapping = {
        "関越": "関越道", "東名": "東名道", "中央": "中央道", 
        "東北": "東北道", "首都": "首都高速"
    }
    
    standard_route = route_mapping.get(route, "関越道") if route else "関越道"
    
    # 处理年份
    if time:
        try:
            year = int(time)
        except:
            year = datetime.now().year
    else:
        year = datetime.now().year
    
    # 处理月份和日期
    if not month:
        month = datetime.now().month
    
    if return_multiple:
        # 返回多个年份的文件
        target_years = [year, year - 1, year - 2]
        all_paths = []
        
        for target_year in target_years:
            if day:
                filename = f"{standard_route}_{target_year}_{month:02d}-{day:02d}.csv"
            else:
                filename = f"{standard_route}_{target_year}_{month:02d}-01.csv"
            
            file_path = os.path.join(base_dir, filename)
            all_paths.append(file_path)
        
        return all_paths
    else:
        # 返回单个文件
        if day:
            filename = f"{standard_route}_{year}_{month:02d}-{day:02d}.csv"
        else:
            filename = f"{standard_route}_{year}_{month:02d}-01.csv"
        
        return os.path.join(base_dir, filename)
    

def extract_route_and_time(user_input: str) -> Dict[str, Optional[str]]:
    """
    从用户输入中提取路名和时间信息
    """
    # 定义路名模式（支持中文路名）
    route_patterns = [
        r'(関越.*?(?:高速|道路|線))',  # 関越高速道路等
        r'(関越.*?(?:高速|道路|線|线路|线))',  # 支持中文"线"
        r'(東名.*?(?:高速|道路|線))',  # 東名高速道路等
        r'(中央.*?(?:高速|道路|線))',  # 中央高速道路等
        r'(首都.*?(?:高速|道路|線))',  # 首都高速道路等
        r'(国道\d+号)',              # 国道XX号
        r'(県道\d+号)',              # 県道XX号
        r'(关越.*?(?:高速|道路|线路|线))',  # 支持简体中文"关越"
    ]
    
    # 定义时间模式
    time_patterns = [
        r'(\d{4})年',                # 2024年
        r'(\d{4})',                  # 2024
        r'(20\d{2})',               # 2021-2099
        r'(令和\d+年)',             # 令和X年
        r'(平成\d+年)',             # 平成X年
    ]
    
    extracted_route = None
    extracted_time = None
    
    # 提取路名
    for pattern in route_patterns:
        match = re.search(pattern, user_input)
        if match:
            extracted_route = match.group(1)
            break
    
    # 如果没有找到特定格式，尝试一些常见的路名关键词
    if not extracted_route:
        route_keywords = ['関越', '東名', '中央', '首都', '高速', '国道', '県道']
        for keyword in route_keywords:
            if keyword in user_input:
                extracted_route = keyword
                break
    
    # 提取时间
    for pattern in time_patterns:
        match = re.search(pattern, user_input)
        if match:
            time_str = match.group(1)
            # 处理年号转换
            if '令和' in time_str:
                year_match = re.search(r'令和(\d+)年', time_str)
                if year_match:
                    reiwa_year = int(year_match.group(1))
                    extracted_time = str(2018 + reiwa_year)  # 令和元年=2019年
            elif '平成' in time_str:
                year_match = re.search(r'平成(\d+)年', time_str)
                if year_match:
                    heisei_year = int(year_match.group(1))
                    extracted_time = str(1988 + heisei_year)  # 平成元年=1989年
            else:
                extracted_time = time_str.replace('年', '')
            break
    
    return {
        'route': extracted_route,
        'time': extracted_time
    }


def generate_file_path(route: Optional[str], time: Optional[str]) -> str:
    """
    根据提取的路名和时间生成对应的文件路径
    """
    base_data_dir = "data"
    
    # 如果有明确的时间，优先使用对应年份的文件
    if time:
        year = time
        
        # 检查data目录下是否有对应的xlsx文件
        potential_files = [
            f"{base_data_dir}/関越{year}.xlsx",
            f"{base_data_dir}/関越{year}_cleaned.xlsx",
        ]
        
        for file_path in potential_files:
            if os.path.exists(file_path):
                # 将xlsx转换为csv路径（假设后续会转换）
                csv_path = file_path.replace('.xlsx', '.csv')
                return csv_path
        
        # 检查meta_data目录下的文件
        meta_data_files = [
            f"{base_data_dir}/meta_data/★{year}_関東支社渋滞データ（01-12）SIC分割【コード変換・BT記入・本社BT】特定更新工事-緊急工事(1～12月分まで).xlsx",
            f"{base_data_dir}/meta_data/★{year}_関東支社渋滞データ（01-12）SIC分割【コード変換・BT記入・本社BT】特定更新工事v3.xlsx",
        ]
        
        for file_path in meta_data_files:
            if os.path.exists(file_path):
                csv_path = file_path.replace('.xlsx', '.csv')
                return csv_path
    
    # 如果有路名信息，根据路名匹配
    if route and '関越' in route:
        # 默认使用最新的関越数据
        if os.path.exists(f"{base_data_dir}/関越2024_cleaned.xlsx"):
            return f"{base_data_dir}/関越2024_cleaned.csv"
        elif os.path.exists(f"{base_data_dir}/関越2024.xlsx"):
            return f"{base_data_dir}/関越2024.csv"
    
    # 默认返回最新的可用文件
    default_files = [
        f"{base_data_dir}/関越2024_cleaned.csv",
        f"{base_data_dir}/関越2024.csv",
        f"{base_data_dir}/meta_data/★2024_関東支社渋滞データ（01-12）SIC分割【コード変換・BT記入・本社BT】特定更新工事-緊急工事(1～12月分まで).csv"
    ]
    
    return default_files[0]  # 返回默认的第一个文件


# === 可视化相关函数 ===

def parse_hhmm_or_int(val):
    """
    支持："06:35:00" / "6:35" / 635 / 905 → datetime.time
    解析时间格式，支持多种输入格式
    """
    if pd.isnull(val):
        return None

    # 1) 已经是 datetime 或 time 对象
    if isinstance(val, (datetime, time)):
        return val.time() if isinstance(val, datetime) else val

    # 2) 字符串 "HH:MM[:SS]"
    if isinstance(val, str):
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(val, fmt).time()
            except ValueError:
                continue
        # 落到纯数字
        val = val.strip()

    # 3) 纯数字（可能是 int/float/数字字符串）
    try:
        val_str = str(int(float(val)))
        if len(val_str) == 3:   # e.g. 635
            h, m = int(val_str[0]), int(val_str[1:])
        elif len(val_str) == 4: # e.g. 1235
            h, m = int(val_str[:2]), int(val_str[2:])
        else:                   # 长度 1~2 → 0~23 点整
            h, m = int(val_str), 0
        return time(h, m)
    except:
        return None


def t2min(t: time) -> int:
    """将time对象转换为分钟数（从00:00开始）"""
    return t.hour * 60 + t.minute


def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    加载Excel/CSV文件并进行基本处理
    """
    print(f"正在加载数据文件: {file_path}")
    
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # 检查xlsx文件是否存在，如果不存在则尝试csv
        if not os.path.exists(file_path):
            csv_path = file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
            if os.path.exists(csv_path):
                print(f"xlsx文件不存在，尝试读取csv文件: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                # 尝试查找原始xlsx文件
                xlsx_path = file_path.replace('.csv', '.xlsx')
                if os.path.exists(xlsx_path):
                    print(f"csv文件不存在，读取原始xlsx文件: {xlsx_path}")
                    df = pd.read_excel(xlsx_path)
                else:
                    raise FileNotFoundError(f"无法找到数据文件: {file_path}")
        else:
            df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，共 {len(df)} 行")
    return df


# def generate_triangles_from_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
#     """
#     从数据生成三角形，返回三角形列表，每个三角形包含详细信息
#     """
#     print("正在生成三角形数据...")
    
#     # 解析时间列
#     df = df.copy()
#     df["発生_t"] = df["発生時刻"].apply(parse_hhmm_or_int)
#     df["ピーク_t"] = df["ピーク時刻"].apply(parse_hhmm_or_int)
    
#     triangles = []
    
#     for idx, row in df.iterrows():
#         if pd.isnull(row["発生_t"]) or pd.isnull(row["ピーク_t"]):
#             continue
        
#         # 左右 KP（横坐标）
#         x_left = row["発生Ｋｐ"]                      # 拥堵起点
#         width = row["ピーク長"]                       # ピーク時の渋滞長 [km]
#         x_right = x_left + width                     # 拥堵终点
#         apex_x = x_left + width / 2                  # 顶点在中点
        
#         # 上下时间（纵坐标, 单位 min）
#         y_base = t2min(row["発生_t"])                # 拥堵开始
#         y_apex = t2min(row["ピーク_t"])              # 拥堵峰值
        
#         # 三角形顶点：[(apex_x, y_apex), (x_left, y_base), (x_right, y_base)]
#         vertices = [(apex_x, y_apex), (x_left, y_base), (x_right, y_base)]
        
#         # 计算三角形的中心点
#         center_x = (apex_x + x_left + x_right) / 3
#         center_y = (y_apex + y_base + y_base) / 3
        
#         # 计算三角形面积
#         area = 0.5 * width * abs(y_apex - y_base)
        
#         triangle_info = {
#             'id': idx,                          # 三角形ID
#             'vertices': vertices,               # 三角形顶点坐标
#             'center': (center_x, center_y),     # 中心点坐标
#             'area': area,                       # 面积
#             'width': width,                     # 宽度（KP跨度）
#             'height': abs(y_apex - y_base),     # 高度（时间跨度）
#             'kp_start': x_left,                 # 起始KP
#             'kp_end': x_right,                  # 结束KP
#             'time_start': y_base,               # 开始时间（分钟）
#             'time_peak': y_apex,                # 峰值时间（分钟）
#             'original_data': {                  # 原始数据
#                 '発生時刻': row["発生時刻"],
#                 'ピーク時刻': row["ピーク時刻"],
#                 '発生Ｋｐ': row["発生Ｋｐ"],
#                 'ピーク長': row["ピーク長"]
#             }
#         }
        
#         triangles.append(triangle_info)
    
#     print(f"生成了 {len(triangles)} 个三角形")
#     return triangles


# def visualize_triangles(triangles: List[Dict[str, Any]], output_path: str = "triangle_visualization.png") -> str:
#     """
#     可视化三角形并保存图片
#     """
#     print(f"正在绘制 {len(triangles)} 个三角形...")
    
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # 绘制每个三角形
#     for triangle in triangles:
#         vertices = triangle['vertices']
#         xs, ys = zip(*vertices)
#         ax.fill(xs, ys, alpha=0.3)  # 使用默认颜色，透明度0.3
    
#     # 设置轴标签和标题
#     ax.set_xlabel("KP [km]")
#     ax.set_ylabel("Time of day [min since 0:00]")
#     ax.set_title("関越道 渋滞イベント（三角形表示）")
#     ax.set_ylim(0, 24*60)
#     ax.invert_yaxis()  # 上方=凌晨，下方=深夜
    
#     # 添加网格
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"图片已保存至: {output_path}")
#     plt.close()  # 关闭图形以释放内存
    
#     return output_path


def process_traffic_data(file_path: str, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], str]:
    """
    完整的交通数据处理流程：加载数据 → 生成三角形 → 可视化
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    df = load_and_process_data(file_path)
    
    # 2. 生成三角形
    triangles = generate_triangles_from_data(df)
    
    # 3. 可视化
    fig_name = f"triangles_{os.path.basename(file_path).split('.')[0]}.png"
    fig_path = os.path.join(output_dir, fig_name)
    visualize_triangles(triangles, fig_path)
    
    return triangles, fig_path


# === 聚类相关函数 ===

def extract_triangle_features(triangles: List[Dict[str, Any]]) -> np.ndarray:
    """
    从三角形数据中提取用于聚类的特征向量
    """
    features = []
    
    for triangle in triangles:
        # 空间特征
        center_x, center_y = triangle['center']
        kp_start = triangle['kp_start']
        kp_end = triangle['kp_end']
        
        # 时间特征
        time_start = triangle['time_start']
        time_peak = triangle['time_peak']
        
        # 形状特征
        area = triangle['area']
        width = triangle['width']
        height = triangle['height']
        
        # 构建特征向量
        feature_vector = [
            center_x,      # KP中心位置
            center_y,      # 时间中心位置
            kp_start,      # 起始KP
            kp_end,        # 结束KP
            time_start,    # 开始时间
            time_peak,     # 峰值时间
            area,          # 面积
            width,         # 宽度
            height,        # 高度（时间跨度）
        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def calculate_spatiotemporal_distance(triangle1: Dict[str, Any], triangle2: Dict[str, Any]) -> float:
    """
    计算两个三角形之间的时空距离
    结合空间距离、时间距离和形状相似性
    """
    # 空间距离（KP维度）
    spatial_dist = abs(triangle1['center'][0] - triangle2['center'][0])
    
    # 时间距离（分钟维度）
    temporal_dist = abs(triangle1['center'][1] - triangle2['center'][1])
    
    # 形状相似性（面积比）
    area1, area2 = triangle1['area'], triangle2['area']
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 1
    shape_dissimilarity = 1 - area_ratio
    
    # 加权组合距离
    # 权重可以根据实际需求调整
    spatial_weight = 1.0    # KP距离权重
    temporal_weight = 0.1   # 时间距离权重（1分钟 ≈ 0.1km的重要性）
    shape_weight = 0.5      # 形状差异权重
    
    combined_distance = (spatial_weight * spatial_dist + 
                        temporal_weight * temporal_dist + 
                        shape_weight * shape_dissimilarity)
    
    return combined_distance


def cluster_triangles_dbscan(triangles: List[Dict[str, Any]], 
                           eps: float = 2.0, 
                           min_samples: int = 3) -> List[List[int]]:
    """
    使用DBSCAN对三角形进行聚类
    
    Args:
        triangles: 三角形数据列表
        eps: DBSCAN的邻域半径
        min_samples: 形成核心点的最小样本数
    
    Returns:
        聚类结果，每个子列表包含属于同一聚类的三角形索引
    """
    if len(triangles) == 0:
        return []
    
    if not CLUSTERING_AVAILABLE:
        print("sklearn不可用，使用自定义聚类方法")
        return cluster_triangles_custom(triangles, distance_threshold=eps*2)
    
    # 提取特征
    features = extract_triangle_features(triangles)
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 执行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(features_scaled)
    
    # 组织聚类结果
    clusters = {}
    noise_points = []
    
    for idx, label in enumerate(cluster_labels):
        if label == -1:  # 噪声点
            noise_points.append(idx)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
    
    # 转换为列表格式
    cluster_list = list(clusters.values())
    
    # 如果有噪声点且数量较少，可以将其作为单独的聚类
    if noise_points and len(noise_points) <= max(min_samples, 3):
        cluster_list.append(noise_points)
    
    print(f"DBSCAN聚类完成:")
    print(f"  - 发现 {len(cluster_list)} 个聚类")
    print(f"  - 噪声点: {len(noise_points) if len(noise_points) > max(min_samples, 3) else 0} 个")
    
    return cluster_list


def cluster_triangles_custom(triangles: List[Dict[str, Any]], 
                           distance_threshold: float = 3.0) -> List[List[int]]:
    """
    使用自定义时空距离的层次聚类
    
    Args:
        triangles: 三角形数据列表
        distance_threshold: 距离阈值
    
    Returns:
        聚类结果
    """
    if len(triangles) == 0:
        return []
    
    n = len(triangles)
    
    # 计算距离矩阵
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_spatiotemporal_distance(triangles[i], triangles[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    # 简单的聚类算法：基于距离阈值
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if not visited[i]:
            cluster = [i]
            visited[i] = True
            
            # 查找距离小于阈值的点
            for j in range(n):
                if not visited[j] and distances[i][j] <= distance_threshold:
                    cluster.append(j)
                    visited[j] = True
            
            if len(cluster) >= 2:  # 至少2个点才形成聚类
                clusters.append(cluster)
    
    print(f"自定义聚类完成:")
    print(f"  - 发现 {len(clusters)} 个聚类")
    
    return clusters


def analyze_clusters(triangles: List[Dict[str, Any]], 
                    clusters: List[List[int]]) -> Dict[str, Any]:
    """
    分析聚类结果，提供统计信息
    """
    if not clusters:
        return {"total_clusters": 0, "cluster_stats": []}
    
    cluster_stats = []
    
    for cluster_idx, triangle_indices in enumerate(clusters):
        cluster_triangles = [triangles[i] for i in triangle_indices]
        
        # 计算聚类统计信息
        kp_positions = [t['center'][0] for t in cluster_triangles]
        time_positions = [t['center'][1] for t in cluster_triangles]
        areas = [t['area'] for t in cluster_triangles]
        
        stats = {
            "cluster_id": cluster_idx,
            "size": len(triangle_indices),
            "triangle_indices": triangle_indices,
            "kp_range": {
                "min": min(kp_positions),
                "max": max(kp_positions),
                "center": np.mean(kp_positions)
            },
            "time_range": {
                "min": min(time_positions),
                "max": max(time_positions),
                "center": np.mean(time_positions)
            },
            "area_stats": {
                "min": min(areas),
                "max": max(areas),
                "mean": np.mean(areas),
                "total": sum(areas)
            }
        }
        
        cluster_stats.append(stats)
    
    # 按聚类大小排序
    cluster_stats.sort(key=lambda x: x["size"], reverse=True)
    
    analysis_result = {
        "total_clusters": len(clusters),
        "total_triangles": sum(len(cluster) for cluster in clusters),
        "cluster_stats": cluster_stats
    }
    
    return analysis_result


# === 外包大三角形相关函数 ===

def calculate_cluster_hull_triangle(triangles: List[Dict[str, Any]], 
                                   cluster_indices: List[int]) -> Dict[str, Any]:
    """
    为一个聚类计算外包大三角形
    
    Args:
        triangles: 所有三角形数据
        cluster_indices: 聚类中三角形的索引列表
    
    Returns:
        大三角形的信息字典
    """
    if not cluster_indices:
        return {}
    
    cluster_triangles = [triangles[i] for i in cluster_indices]
    
    # 收集所有顶点坐标
    all_vertices = []
    for triangle in cluster_triangles:
        all_vertices.extend(triangle['vertices'])
    
    if not all_vertices:
        return {}
    
    # 计算时空边界
    xs = [v[0] for v in all_vertices]  # KP坐标
    ys = [v[1] for v in all_vertices]  # 时间坐标
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 添加一些边距，确保完全覆盖
    kp_margin = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 1.0  # 10%边距，最小1km
    time_margin = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 30  # 10%边距，最小30分钟
    
    # 构造朝右的大三角形的三个顶点
    # 顶点1（右顶点）：KP最大位置，时间在中间 - 三角形的右尖端
    apex_x = max_x + kp_margin
    apex_y = (min_y + max_y) / 2
    
    # 顶点2（左上角）：KP最小位置，时间最早
    left_top_x = min_x - kp_margin
    left_top_y = min_y - time_margin
    
    # 顶点3（左下角）：KP最小位置，时间最晚
    left_bottom_x = min_x - kp_margin
    left_bottom_y = max_y + time_margin
    
    hull_vertices = [(apex_x, apex_y), (left_top_x, left_top_y), (left_bottom_x, left_bottom_y)]
    
    # 计算大三角形的属性
    center_x = (apex_x + left_top_x + left_bottom_x) / 3
    center_y = (apex_y + left_top_y + left_bottom_y) / 3
    
    width = apex_x - left_top_x  # KP方向的宽度
    height = left_bottom_y - left_top_y  # 时间方向的高度
    area = 0.5 * width * height
    
    hull_info = {
        'vertices': hull_vertices,
        'center': (center_x, center_y),
        'area': area,
        'width': width,
        'height': height,
        'kp_range': (left_top_x, apex_x),
        'time_range': (left_top_y, left_bottom_y),
        'cluster_indices': cluster_indices,
        'cluster_size': len(cluster_indices),
        'covered_triangles': cluster_triangles
    }
    
    return hull_info


def calculate_all_hull_triangles(triangles: List[Dict[str, Any]], 
                                clusters: List[List[int]]) -> List[Dict[str, Any]]:
    """
    为所有聚类计算外包大三角形
    
    Args:
        triangles: 所有三角形数据
        clusters: 聚类结果，每个元素是三角形索引列表
    
    Returns:
        所有大三角形的信息列表
    """
    hulls = []
    
    for cluster_idx, cluster_indices in enumerate(clusters):
        if len(cluster_indices) < 2:  # 跳过太小的聚类
            continue
            
        hull = calculate_cluster_hull_triangle(triangles, cluster_indices)
        if hull:
            hull['cluster_id'] = cluster_idx
            hulls.append(hull)
    
    print(f"计算了 {len(hulls)} 个外包大三角形")
    
    return hulls


def visualize_triangles_with_hulls(triangles: List[Dict[str, Any]], 
                                  hulls: List[Dict[str, Any]] = None,
                                  output_path: str = "triangle_visualization_with_hulls.png",
                                  figsize: tuple = (14, 10)) -> str:
    """
    可视化三角形并叠加外包大三角形
    
    Args:
        triangles: 原始三角形数据
        hulls: 外包大三角形数据
        output_path: 输出图片路径
        figsize: 图片尺寸
    
    Returns:
        保存的图片路径
    """
    print(f"正在绘制 {len(triangles)} 个多边形和 {len(hulls) if hulls else 0} 个外包大三角形...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 定义颜色
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # 1. 绘制原始小多边形（较浅的颜色）
    triangle_count = 0
    quad_count = 0
    for i, polygon in enumerate(triangles):
        vertices = polygon['vertices']
        xs, ys = zip(*vertices)
        
        # 根据形状类型选择颜色和透明度
        if polygon.get('shape_type') == 'triangle':
            ax.fill(xs, ys, alpha=0.3, color='lightblue', edgecolor='blue', linewidth=0.5)
            triangle_count += 1
        else:  # quadrilateral
            ax.fill(xs, ys, alpha=0.3, color='lightgreen', edgecolor='green', linewidth=0.5)
            quad_count += 1
    
    # 2. 绘制外包大三角形（较深的颜色，不同颜色区分不同聚类）
    if hulls:
        for i, hull in enumerate(hulls):
            vertices = hull['vertices']
            xs, ys = zip(*vertices)
            
            # 为每个聚类选择不同颜色
            color = colors[i % len(colors)]
            
            # 绘制大三角形边框（较粗）
            ax.plot(xs + (xs[0],), ys + (ys[0],), 
                   color=color, linewidth=3, alpha=0.8,
                   label=f'聚类 {hull["cluster_id"]} ({hull["cluster_size"]}个)')
            
            # 绘制半透明填充
            ax.fill(xs, ys, alpha=0.15, color=color)
            
            # 在大三角形中心标注聚类信息
            center_x, center_y = hull['center']
            ax.text(center_x, center_y, f'C{hull["cluster_id"]}\n({hull["cluster_size"]})', 
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 设置轴标签和标题
    ax.set_xlabel("KP [km]", fontsize=12)
    ax.set_ylabel("Time of day [min since 0:00]", fontsize=12)
    
    title = "拥堵事件多边形分析"
    if triangle_count > 0 and quad_count > 0:
        title += f"（三角形: {triangle_count}, 四边形: {quad_count}）"
    elif triangle_count > 0:
        title += f"（{triangle_count}个三角形）"
    elif quad_count > 0:
        title += f"（{quad_count}个四边形）"
    
    if hulls:
        title += f" - {len(hulls)}个拥堵热点区域"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置y轴范围和反转
    ax.set_ylim(0, 24*60)
    ax.invert_yaxis()  # 上方=凌晨，下方=深夜
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例（如果有外包三角形）
    if hulls and len(hulls) <= 10:  # 只在聚类数不太多时显示图例
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 添加时间刻度标签
    time_ticks = [i*60 for i in range(0, 25, 4)]  # 每4小时一个刻度
    time_labels = [f"{i:02d}:00" for i in range(0, 25, 4)]
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {output_path}")
    plt.close()  # 关闭图形以释放内存
    
    return output_path


def create_hull_summary_report(hulls: List[Dict[str, Any]]) -> str:
    """
    创建外包大三角形的摘要报告
    
    Args:
        hulls: 外包大三角形数据
    
    Returns:
        格式化的报告字符串
    """
    if not hulls:
        return "未发现明显的拥堵热点区域。"
    
    # 按聚类大小排序
    sorted_hulls = sorted(hulls, key=lambda x: x['cluster_size'], reverse=True)
    
    report_lines = [
        "## 拥堵热点区域分析报告",
        f"发现 {len(hulls)} 个主要拥堵热点区域：\n"
    ]
    
    for i, hull in enumerate(sorted_hulls):
        cluster_id = hull['cluster_id']
        cluster_size = hull['cluster_size']
        kp_start, kp_end = hull['kp_range']
        time_start, time_end = hull['time_range']
        
        # 转换时间到小时:分钟格式
        start_hour, start_min = divmod(int(time_start), 60)
        end_hour, end_min = divmod(int(time_end), 60)
        
        report_lines.extend([
            f"### 热点区域 {i+1} (聚类 {cluster_id})",
            f"- **拥堵事件数量**: {cluster_size} 次",
            f"- **影响路段**: KP {kp_start:.1f} - {kp_end:.1f} km",
            f"- **影响时段**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
            f"- **覆盖面积**: {hull['area']:.1f} 平方单位",
            f"- **路段长度**: {hull['width']:.1f} km",
            f"- **持续时长**: {hull['height']:.0f} 分钟\n"
        ])
    
    # 添加总结
    total_events = sum(hull['cluster_size'] for hull in hulls)
    avg_cluster_size = total_events / len(hulls)
    
    report_lines.extend([
        "### 总体分析",
        f"- **总拥堵事件**: {total_events} 次",
        f"- **平均每个热点**: {avg_cluster_size:.1f} 次拥堵",
        f"- **最大热点**: {sorted_hulls[0]['cluster_size']} 次拥堵",
        f"- **热点集中度**: {'高' if avg_cluster_size > 5 else '中等' if avg_cluster_size > 2 else '低'}"
    ])
    
    return "\n".join(report_lines)


def process_multiple_traffic_data(file_paths: List[str], output_dir: str = "output") -> Tuple[List[Dict[str, Any]], str]:
    """
    处理多个CSV文件的完整交通数据处理流程：加载数据 → 生成三角形 → 统一可视化
    
    Args:
        file_paths: CSV文件路径列表
        output_dir: 输出目录
    
    Returns:
        合并的三角形数据列表, 统一可视化图片路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    all_triangles = []
    file_info = []
    
    print(f"📊 开始处理 {len(file_paths)} 个CSV文件...")
    
    for i, file_path in enumerate(file_paths):
        print(f"  处理文件 {i+1}/{len(file_paths)}: {file_path}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"    ⚠️ 文件不存在，跳过: {file_path}")
                continue
            
            # 1. 加载数据
            df = load_and_process_data(file_path)
            
            # 2. 生成三角形
            triangles = generate_triangles_from_data(df)
            
            # 3. 为每个三角形添加文件来源信息
            file_name = os.path.basename(file_path).split('.')[0]
            for triangle in triangles:
                triangle['source_file'] = file_name
                triangle['file_index'] = i
            
            all_triangles.extend(triangles)
            file_info.append({
                'file_path': file_path,
                'file_name': file_name,
                'triangle_count': len(triangles)
            })
            
            print(f"    ✅ 成功处理，生成 {len(triangles)} 个三角形")
            
        except Exception as e:
            print(f"    ❌ 处理失败: {str(e)}")
            continue
    
    if not all_triangles:
        print("⚠️ 没有成功处理任何文件")
        return [], ""
    
    print(f"📈 总计生成 {len(all_triangles)} 个三角形，来自 {len(file_info)} 个文件")
    
    # 4. 统一可视化所有三角形
    fig_name = f"triangles_multi_file_{len(file_info)}files.png"
    fig_path = os.path.join(output_dir, fig_name)
    
    # 使用增强的可视化函数，支持多文件标识
    visualize_triangles_multi_source(all_triangles, fig_path, file_info)
    
    return all_triangles, fig_path


def visualize_triangles_multi_source(triangles: List[Dict[str, Any]], 
                                   output_path: str = "triangle_visualization_multi.png",
                                   file_info: List[Dict[str, Any]] = None) -> str:
    """
    可视化来自多个文件的三角形数据，用不同颜色区分不同文件
    
    Args:
        triangles: 三角形数据列表（包含source_file和file_index字段）
        output_path: 输出图片路径
        file_info: 文件信息列表
    
    Returns:
        保存的图片文件路径
    """
    if not triangles:
        print("没有三角形数据可以可视化")
        return ""
    
    plt.figure(figsize=(16, 12))
    
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 按文件分组绘制
    file_indices = set(triangle.get('file_index', 0) for triangle in triangles)
    
    for file_idx in sorted(file_indices):
        file_triangles = [t for t in triangles if t.get('file_index', 0) == file_idx]
        if not file_triangles:
            continue
            
        color = colors[file_idx % len(colors)]
        file_name = file_triangles[0].get('source_file', f'文件{file_idx+1}')
        
        # 绘制该文件的所有三角形
        for triangle in file_triangles:
            vertices = triangle['vertices']
            # 创建三角形多边形
            poly = Polygon(vertices, alpha=0.6, color=color, edgecolor='black', linewidth=0.5)
            plt.gca().add_patch(poly)
    
    # 设置图形属性
    if triangles:
        all_x = [vertex[0] for triangle in triangles for vertex in triangle['vertices']]
        all_y = [vertex[1] for triangle in triangles for vertex in triangle['vertices']]
        
        margin_x = (max(all_x) - min(all_x)) * 0.05
        margin_y = (max(all_y) - min(all_y)) * 0.05
        
        plt.xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        plt.ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    plt.xlabel('KP (km)', fontsize=12)
    plt.ylabel('时间 (分钟)', fontsize=12)
    plt.title(f'交通拥堵三角形可视化 - 多文件合并\n总计 {len(triangles)} 个三角形', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    if file_info:
        legend_elements = []
        for i, info in enumerate(file_info):
            color = colors[i % len(colors)]
            label = f"{info['file_name']} ({info['triangle_count']}个)"
            legend_elements.append(plt.Rectangle((0,0),1,1, color=color, alpha=0.6, label=label))
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 多文件三角形可视化图已保存至: {output_path}")
    return output_path

def generate_polygons_from_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    从数据生成多边形（三角形或四边形），返回多边形列表，每个多边形包含详细信息
    根据拥堵演化过程生成准确的几何图形
    """
    print("正在生成多边形数据...")
    
    # 解析时间列
    df = df.copy()
    df["発生_t"] = df["発生時刻"].apply(parse_hhmm_or_int)
    df["ピーク_t"] = df["ピーク時刻"].apply(parse_hhmm_or_int)
    
    polygons = []
    
    for idx, row in df.iterrows():
        if pd.isnull(row["発生_t"]) or pd.isnull(row["ピーク_t"]):
            continue
        
        # 提取原始数据
        start_kp = row["発生Ｋｐ"]              # 拥堵起始KP [km]
        start_jam_length = row["発生時渋滞長"]   # 开始时拥堵长度 [km]
        peak_length = row["ピーク長"]           # 峰值时拥堵长度 [km]
        jam_duration = row["渋滞時間"]          # 拥堵持续时间 [分钟]
        
        # 时间坐标（转换为分钟）
        start_time = t2min(row["発生_t"])       # 拥堵开始时间 [分钟]
        peak_time = t2min(row["ピーク_t"])      # 拥堵峰值时间 [分钟]
        end_time = start_time + jam_duration    # 拥堵结束时间 [分钟]
        
        # 定义四个关键顶点
        vertex1 = (start_kp, start_time)                           # 拥堵开始起点
        vertex2 = (start_kp + start_jam_length, start_time)        # 拥堵开始终点
        vertex3 = (start_kp + peak_length, peak_time)              # 拥堵峰值终点
        vertex4 = (start_kp, end_time)                             # 拥堵结束起点
        
        # 根据発生時渋滞長判断是三角形还是四边形
        if start_jam_length == 0:
            # 三角形：起始时无拥堵长度，vertex1和vertex2重合
            vertices = [vertex1, vertex3, vertex4]
            shape_type = "triangle"
        else:
            # 四边形：完整的拥堵演化过程
            vertices = [vertex1, vertex2, vertex3, vertex4]
            shape_type = "quadrilateral"
        
        # 计算中心点（所有顶点的重心）
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)
        
        # 计算面积（使用鞋带公式）
        def calculate_polygon_area(vertices):
            n = len(vertices)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2.0
        
        area = calculate_polygon_area(vertices)
        
        # 计算边界框
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        kp_start = min(xs)
        kp_end = max(xs)
        time_start = min(ys)
        time_end = max(ys)
        
        # 计算宽度和高度
        width = kp_end - kp_start
        height = time_end - time_start
        
        polygon_info = {
            'id': idx,                          # 多边形ID
            'shape_type': shape_type,           # 形状类型：triangle 或 quadrilateral
            'vertices': vertices,               # 多边形顶点坐标
            'center': (center_x, center_y),     # 中心点坐标
            'area': area,                       # 面积
            'width': width,                     # 宽度（KP跨度）
            'height': height,                   # 高度（时间跨度）
            'kp_start': kp_start,               # 起始KP
            'kp_end': kp_end,                   # 结束KP
            'time_start': time_start,           # 开始时间（分钟）
            'time_end': time_end,               # 结束时间（分钟）
            'time_peak': peak_time,             # 峰值时间（分钟）
            'original_data': {                  # 原始数据
                '発生時刻': row["発生時刻"],
                'ピーク時刻': row["ピーク時刻"],
                '発生Ｋｐ': row["発生Ｋｐ"],
                'ピーク長': row["ピーク長"],
                '発生時渋滞長': row["発生時渋滞長"],
                '渋滞時間': row["渋滞時間"]
            }
        }
        
        polygons.append(polygon_info)
    
    # 统计形状类型
    triangle_count = sum(1 for p in polygons if p['shape_type'] == 'triangle')
    quad_count = sum(1 for p in polygons if p['shape_type'] == 'quadrilateral')
    
    print(f"生成了 {len(polygons)} 个多边形")
    print(f"  - 三角形: {triangle_count} 个")
    print(f"  - 四边形: {quad_count} 个")
    
    return polygons


def visualize_polygons(polygons: List[Dict[str, Any]], output_path: str = "polygon_visualization.png") -> str:
    """
    可视化多边形并保存图片，支持三角形和四边形的区分显示
    """
    print(f"正在绘制 {len(polygons)} 个多边形...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 分别绘制三角形和四边形，使用不同颜色
    triangle_count = 0
    quad_count = 0
    
    for polygon in polygons:
        vertices = polygon['vertices']
        xs, ys = zip(*vertices)
        
        if polygon['shape_type'] == 'triangle':
            ax.fill(xs, ys, alpha=0.4, color='lightblue', edgecolor='blue', linewidth=0.5)
            triangle_count += 1
        else:  # quadrilateral
            ax.fill(xs, ys, alpha=0.4, color='lightcoral', edgecolor='red', linewidth=0.5)
            quad_count += 1
    
    # 设置轴标签和标题
    ax.set_xlabel("KP [km]", fontsize=12)
    ax.set_ylabel("Time of day [min since 0:00]", fontsize=12)
    ax.set_title(f"拥堵事件多边形可视化\n三角形: {triangle_count}个, 四边形: {quad_count}个", fontsize=14)
    ax.set_ylim(0, 24*60)
    ax.invert_yaxis()  # 上方=凌晨，下方=深夜
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.4, label=f'三角形 ({triangle_count}个)'),
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.4, label=f'四边形 ({quad_count}个)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加时间刻度标签
    time_ticks = [i*60 for i in range(0, 25, 4)]  # 每4小时一个刻度
    time_labels = [f"{i:02d}:00" for i in range(0, 25, 4)]
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {output_path}")
    plt.close()  # 关闭图形以释放内存
    
    return output_path



# 为了向后兼容，保留原函数名但指向新函数
def generate_triangles_from_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    向后兼容函数：现在生成多边形（三角形或四边形）
    """
    return generate_polygons_from_data(df)


def visualize_triangles(triangles: List[Dict[str, Any]], output_path: str = "triangle_visualization.png") -> str:
    """
    向后兼容函数：现在可视化多边形
    """
    return visualize_polygons(triangles, output_path)