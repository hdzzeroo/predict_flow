import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

def setup_japanese_font():
    """设置matplotlib支持日文字体显示"""
    # 尝试找到系统可用的中日韩字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级排序的字体列表
    preferred_fonts = [
        'Noto Sans CJK JP',     # Google Noto字体（广泛支持）
        'Hiragino Sans',        # macOS日文字体
        'Yu Gothic',            # Windows日文字体
        'Meiryo',               # Windows日文字体
        'IPAexGothic',          # Linux日文字体
        'IPAPGothic',           # Linux日文字体
        'VL PGothic',           # Linux日文字体
        'TakaoGothic',          # Linux日文字体
        'WenQuanYi Micro Hei',  # 文泉驿微黑（支持中文）
        'SimHei',               # 黑体（Windows中文）
        'DejaVu Sans'           # 默认字体（不支持中日文但不会出错）
    ]
    
    # 找到第一个可用的字体
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        print(f"🔤 使用字体: {selected_font}")
        matplotlib.rcParams['font.family'] = [selected_font]
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
    else:
        print("⚠️ 未找到合适的中日韩字体，使用默认字体（可能显示乱码）")
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    
    # 解决坐标轴负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    return selected_font

# 初始化字体设置
setup_japanese_font()
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, List, Tuple, Any, Union
from matplotlib.patches import Polygon

from dotenv import load_dotenv
load_dotenv()

# 全局道路信息缓存
_road_info_cache = {}

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


# === 道路信息加载函数 ===

def load_road_info(road_type: str) -> pd.DataFrame:
    """
    加载道路信息数据
    
    Args:
        road_type: 道路类型 ("関越道" 或 "外環道")
        
    Returns:
        包含道路信息的DataFrame
    """
    global _road_info_cache
    
    if road_type in _road_info_cache:
        return _road_info_cache[road_type]
    
    # 确定文件路径
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data"
    
    if "関越" in road_type or "関越道" in road_type:
        file_path = os.path.join(base_dir, "roadic_kannetsu.csv")
    elif "外環" in road_type or "外環道" in road_type:
        file_path = os.path.join(base_dir, "roadic_gaikan.csv")
    else:
        # 默认尝试加载関越道信息
        file_path = os.path.join(base_dir, "roadic_kannetsu.csv")
        print(f"⚠️ 未识别的道路类型 '{road_type}'，使用関越道信息")
    
    try:
        df = pd.read_csv(file_path)
        _road_info_cache[road_type] = df
        print(f"✅ 已加载道路信息: {road_type} ({len(df)} 个点)")
        return df
    except FileNotFoundError:
        print(f"❌ 道路信息文件未找到: {file_path}")
        return pd.DataFrame()  # 返回空DataFrame
    except Exception as e:
        print(f"❌ 加载道路信息失败: {e}")
        return pd.DataFrame()


def get_road_kp_range(road_type: str, direction: str = None) -> Tuple[float, float]:
    """
    获取指定道路和方向的KP范围
    
    Args:
        road_type: 道路类型
        direction: 方向 ("上", "下", 或 None)
        
    Returns:
        (min_kp, max_kp) 元组
    """
    road_info = load_road_info(road_type)
    
    if road_info.empty:
        # 如果无法加载道路信息，返回默认范围
        print(f"⚠️ 无法获取 {road_type} 的KP信息，使用默认范围")
        return (0, 100)
    
    # 筛选方向
    if direction:
        direction_map = {"上": "up", "下": "down"}
        eng_direction = direction_map.get(direction, direction)
        filtered_df = road_info[road_info['direction'] == eng_direction]
        if filtered_df.empty:
            filtered_df = road_info  # 如果没有匹配的方向，使用全部数据
    else:
        filtered_df = road_info
    
    min_kp = filtered_df['KP'].min()
    max_kp = filtered_df['KP'].max()
    
    return (min_kp, max_kp)


def get_kp_location_name(road_type: str, kp_value: float, direction: str = None) -> str:
    """
    根据KP值获取对应的区间名称
    
    Args:
        road_type: 道路类型
        kp_value: KP值
        direction: 方向
        
    Returns:
        区间名称字符串
    """
    road_info = load_road_info(road_type)
    
    if road_info.empty:
        return f"KP {kp_value:.1f}"
    
    # 筛选方向
    if direction:
        direction_map = {"上": "up", "下": "down"}
        eng_direction = direction_map.get(direction, direction)
        filtered_df = road_info[road_info['direction'] == eng_direction]
        if filtered_df.empty:
            filtered_df = road_info
    else:
        filtered_df = road_info
    
    # 找到最接近的KP点
    closest_idx = (filtered_df['KP'] - kp_value).abs().idxmin()
    closest_row = road_info.loc[closest_idx]
    
    # 根据文件结构返回区间名称
    if 'kukan_name' in closest_row:
        return closest_row['kukan_name']
    else:
        return f"{closest_row['start_name']}~{closest_row['end_name']} (KP{kp_value:.1f})"


def extract_road_type_from_filename(filename: str) -> str:
    """
    从文件名中提取道路类型
    
    Args:
        filename: 文件名或路径
        
    Returns:
        道路类型字符串
    """
    base_name = os.path.basename(filename)
    
    if "関越" in base_name or "関越道" in base_name:
        return "関越道"
    elif "外環" in base_name or "外環道" in base_name:
        return "外環道"
    elif "東名" in base_name:
        return "東名道"
    elif "中央" in base_name:
        return "中央道"
    else:
        return "関越道"  # 默认


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
        
        system_prompt = """You are a professional Japanese traffic data query parsing assistant. Please extract the following information precisely from user input:

## Task
Extract from user's natural language query:
1. **Route Name** - Standardized Japanese route name
2. **Direction** - Traffic direction (上り/下り)
3. **Year** - Specific year (null if not specified)
4. **Month** - Specific month (1-12)
5. **Day** - Specific day (1-31), null if not specified

## Supported Route Name Standardization
- 関越道/関越高速/関越高速道路 → "関越道"
- 東名道/東名高速/東名高速道路 → "東名道"  
- 中央道/中央高速/中央高速道路 → "中央道"
- 東北道/東北高速/東北高速道路 → "東北道"
- 首都高速/首都高 → "首都高速"

## Direction Processing
- 上り/上り線/上り方向 → "上"
- 下り/下り線/下り方向 → "下"
- If no direction specified → null

## Time Processing
- Japanese era conversion: 令和X年 = 2018+X年, 平成X年 = 1988+X年
- Month extraction: X月/XX月 → numbers
- Day extraction: X日/XX日 → numbers

## Output Format
Strictly follow this JSON format, no other text:

{
    "route_name": "standardized route name",
    "direction": "上" or "下" or null,
    "year": year_number or null,
    "month": month_number(1-12) or null,
    "day": day_number(1-31) or null,
    "confidence": confidence_score_0.0_to_1.0
}

## Examples
Input: "4月23日関越道上り的交通状況"
Output: {"route_name": "関越道", "direction": "上", "year": null, "month": 4, "day": 23, "confidence": 0.95}

Input: "2023年東北道下り5月的数据"
Output: {"route_name": "東北道", "direction": "下", "year": 2023, "month": 5, "day": null, "confidence": 0.92}"""

        print("🤖 Calling OpenAI API...")
        
        # 调用OpenAI API (新版本接口)
        client = openai.OpenAI(api_key=api_key_to_use)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用较便宜的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,  # 设置为0，确保输出完全一致
            max_tokens=200,   # 限制token数量
            timeout=10        # 设置超时
        )
        
        llm_response = response.choices[0].message.content.strip()
        print(f"🤖 LLM raw response: {llm_response}")
        
        # 尝试解析JSON响应
        try:
            # 清理响应中可能的markdown格式
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1]
            
            result = json.loads(llm_response)
            
            # 验证结果格式
            required_keys = ["route_name", "direction", "year", "month", "day", "confidence"]
            if all(key in result for key in required_keys):
                print("✅ OpenAI API parsing successful")
                return result
            else:
                print("⚠️ LLM response format incomplete, using fallback parsing")
                return call_fallback_parsing(user_input)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ LLM response JSON parsing failed: {e}")
            print(f"Raw response: {llm_response}")
            return call_fallback_parsing(user_input)
        
    except ImportError:
        print("⚠️ openai package not installed, please run: pip install openai")
        return call_fallback_parsing(user_input)
    except Exception as e:
        print(f"⚠️ OpenAI API call failed: {str(e)}")
        return call_fallback_parsing(user_input)


def call_fallback_parsing(user_input: str) -> Dict[str, Any]:
    """
    降级解析：当LLM不可用时使用的本地正则表达式方法
    """
    import re
    
    print("🔧 Using local regex parsing...")
    
    # 基本的解析逻辑（模拟LLM理解）
    result = {
        "route_name": None,
        "direction": None,
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
    
    # 方向识别
    if "上り" in user_input or "上り線" in user_input or "上り方向" in user_input:
        result["direction"] = "上"
        result["confidence"] = min(result["confidence"] + 0.1, 1.0)
    elif "下り" in user_input or "下り線" in user_input or "下り方向" in user_input:
        result["direction"] = "下"
        result["confidence"] = min(result["confidence"] + 0.1, 1.0)
    
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
    print(f"📝 Parsing user input: {user_input}")
    
    # 第1步：解析用户输入
    if use_llm:
        print("🤖 Using LLM semantic parsing...")
        parsed_info = call_llm_for_date_parsing(user_input, use_real_llm=True, api_key=api_key)
    else:
        print("🔧 Using regex parsing...")
        parsed_info = call_llm_for_date_parsing(user_input, use_real_llm=False)
    
    print(f"🎯 Parsing result: {parsed_info}")
    
    # 第2步：提取信息并设置默认值
    route_name = parsed_info.get("route_name") or "関越道"
    direction = parsed_info.get("direction")  # 不设置默认值，允许为None
    year = parsed_info.get("year")
    month = parsed_info.get("month") or datetime.now().month
    day = parsed_info.get("day")
    
    # 第3步：确定要处理的方向
    if direction is None:
        # 用户没有指定方向，处理上下两个方向
        directions_to_process = ["上", "下"]
        print("🧭 No direction specified, processing both directions: 上り and 下り")
    else:
        # 用户指定了方向，只处理指定方向
        directions_to_process = [direction]
        print(f"🧭 Direction specified: {direction}")
    
    # 第4步：确定目标年份（总是返回3个年份）
    if year == 2025 :
        target_years = [2014, 2019, 2024]
    else:
        current_year = datetime.now().year
        target_years = [current_year - 1, current_year - 2, current_year - 3]
    
    print(f"🗓️  Target years: {target_years}")
    print(f"🛣️  Route: {route_name}, 📅 Month: {month}, 📆 Day: {day or 'entire month'}")
    
    # 第5步：构建文件路径并搜索
    all_paths = []
    
    for direction_item in directions_to_process:
        for target_year in target_years:
            if day:
                # 精确匹配特定日期：東北道_上_2021_04-23.csv
                filename = f"{route_name}_{direction_item}_{target_year}_{month:02d}-{day:02d}.csv"
            else:
                # 匹配整个月的第一个文件：東北道_上_2021_04-*.csv
                pattern = f"{route_name}_{direction_item}_{target_year}_{month:02d}-*.csv"
                matching_files = glob.glob(os.path.join(base_dir, pattern))
                if matching_files:
                    # 取第一个匹配的文件
                    filename = os.path.basename(sorted(matching_files)[0])
                else:
                    # 如果没有找到，生成一个默认的文件名
                    filename = f"{route_name}_{direction_item}_{target_year}_{month:02d}-01.csv"
            
            file_path = os.path.join(base_dir, filename)
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                all_paths.append(file_path)
                print(f"✅ Found file: {filename}")
            else:
                # 即使文件不存在也添加路径，让调用者知道期望的文件名
                all_paths.append(file_path)
                print(f"⚠️  File not found but path generated: {filename}")
    
    print(f"📁 Generated {len(all_paths)} file paths in total")
    return all_paths



# 为了兼容性，保持原有函数不变，但添加新功能
def generate_file_path_enhanced(route: Optional[str], time: Optional[str], 
                              month: Optional[int] = None, day: Optional[int] = None,
                              direction: Optional[str] = None,
                              return_multiple: bool = False) -> Union[str, List[str]]:
    """
    增强版的原始函数，支持月份日期、方向和多文件返回
    """
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    
    # 标准化路线名称
    route_mapping = {
        "関越": "関越道", "東名": "東名道", "中央": "中央道", 
        "東北": "東北道", "首都": "首都高速"
    }
    
    standard_route = route_mapping.get(route, "関越道") if route else "関越道"
    direction = direction or "上"  # 默认使用上り方向
    
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
                filename = f"{standard_route}_{direction}_{target_year}_{month:02d}-{day:02d}.csv"
            else:
                filename = f"{standard_route}_{direction}_{target_year}_{month:02d}-01.csv"
            
            file_path = os.path.join(base_dir, filename)
            all_paths.append(file_path)
        
        return all_paths
    else:
        # 返回单个文件
        if day:
            filename = f"{standard_route}_{direction}_{year}_{month:02d}-{day:02d}.csv"
        else:
            filename = f"{standard_route}_{direction}_{year}_{month:02d}-01.csv"
        
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
    print(f"Loading data file: {file_path}")
    
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # 检查xlsx文件是否存在，如果不存在则尝试csv
        if not os.path.exists(file_path):
            csv_path = file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
            if os.path.exists(csv_path):
                print(f"xlsx file not found, trying to read csv file: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                # 尝试查找原始xlsx文件
                xlsx_path = file_path.replace('.csv', '.xlsx')
                if os.path.exists(xlsx_path):
                    print(f"csv file not found, reading original xlsx file: {xlsx_path}")
                    df = pd.read_excel(xlsx_path)
                else:
                    raise FileNotFoundError(f"Cannot find data file: {file_path}")
        else:
            df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Data loading completed, {len(df)} rows in total")
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


def process_direction_aware_traffic_data(file_path: str, direction: str = None,
                                        output_dir: str = "output", target_year: int = None,
                                        target_month: int = None, target_day: int = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    方向感知的完整交通数据处理流程：加载数据 → 生成方向感知多边形 → 可视化

    Args:
        file_path: 数据文件路径
        direction: 方向 ("上", "下", 或 None)
        output_dir: 输出目录
        target_year: 预测目标年份
        target_month: 预测目标月份
        target_day: 预测目标日期

    Returns:
        (triangles, fig_path) 元组
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    df = load_and_process_data(file_path)

    # 2. 提取道路类型
    road_type = extract_road_type_from_filename(file_path)

    # 3. 生成方向感知的多边形
    triangles = generate_direction_aware_polygons(df, direction, road_type)

    # 4. 生成可视化图片
    fig_filename = generate_unique_filename("triangles_multi", [file_path], direction,
                                           target_year=target_year, target_month=target_month, target_day=target_day)
    fig_path = os.path.join(output_dir, fig_filename)
    
    # 调用方向感知的可视化函数
    final_fig_path = visualize_direction_aware_polygons(triangles, fig_path, road_type, direction)
    
    print(f"✅ 方向感知交通数据处理完成: {len(triangles)} 个多边形, 图片: {final_fig_path}")
    
    return triangles, final_fig_path


def process_multiple_direction_aware_traffic_data(file_paths: List[str], direction: str = None,
                                                 output_dir: str = "output", target_year: int = None,
                                                 target_month: int = None, target_day: int = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    方向感知地处理多个交通数据文件，合并生成统一的可视化

    Args:
        file_paths: 数据文件路径列表
        direction: 方向 ("上", "下", 或 None)
        output_dir: 输出目录
        target_year: 预测目标年份
        target_month: 预测目标月份
        target_day: 预测目标日期

    Returns:
        (所有triangles合并后的列表, 合并后的图片路径)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    all_triangles = []

    # 提取道路类型（假设所有文件都是同一道路）
    road_type = extract_road_type_from_filename(file_paths[0]) if file_paths else "関越道"

    # 1. 处理每个文件
    for file_path in file_paths:
        print(f"\n📁 处理文件: {os.path.basename(file_path)}")

        # 加载数据并生成方向感知的多边形
        df = load_and_process_data(file_path)
        triangles = generate_direction_aware_polygons(df, direction, road_type)

        # 为每个三角形添加来源文件信息
        for triangle in triangles:
            triangle['source_file'] = os.path.basename(file_path).replace('.csv', '')

        all_triangles.extend(triangles)
        print(f"  ✓ 生成了 {len(triangles)} 个方向感知的多边形")

    print(f"\n📊 总计: {len(all_triangles)} 个多边形来自 {len(file_paths)} 个文件")

    # 2. 生成合并的可视化图片
    fig_filename = generate_unique_filename("triangles_multi", file_paths, direction,
                                           target_year=target_year, target_month=target_month, target_day=target_day)
    fig_path = os.path.join(output_dir, fig_filename)
    
    # 调用方向感知的可视化函数
    final_fig_path = visualize_direction_aware_polygons(all_triangles, fig_path, road_type, direction,
                                                       title_suffix=f"({len(file_paths)} files merged)")
    
    print(f"✅ 方向感知多文件交通数据处理完成: {len(all_triangles)} 个多边形, 图片: {final_fig_path}")
    
    return all_triangles, final_fig_path


def process_traffic_data(file_path: str, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], str]:
    """
    向后兼容的完整交通数据处理流程：加载数据 → 生成三角形 → 可视化
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    df = load_and_process_data(file_path)
    
    # 2. 生成三角形
    triangles = generate_triangles_from_data(df)
    
    # 3. 可视化
    fig_name = generate_unique_filename("triangles", [file_path])
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
    spatial_weight = 5.0    # KP距离权重
    temporal_weight = 0.01   # 时间距离权重（1分钟 ≈ 0.1km的重要性）
    shape_weight = 0.5      # 形状差异权重
    
    combined_distance = (spatial_weight * spatial_dist + 
                        temporal_weight * temporal_dist + 
                        shape_weight * shape_dissimilarity)
    
    return combined_distance


def cluster_triangles_dbscan(triangles: List[Dict[str, Any]], 
                           eps: float = 0.01, 
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
        print("sklearn not available, using custom clustering method")
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
    
    print(f"DBSCAN clustering completed:")
    print(f"  - Found {len(cluster_list)} clusters")
    print(f"  - Noise points: {len(noise_points) if len(noise_points) > max(min_samples, 3) else 0}")
    
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
    
    print(f"Custom clustering completed:")
    print(f"  - Found {len(clusters)} clusters")
    
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
                                   cluster_indices: List[int],
                                   direction: str = None) -> Dict[str, Any]:
    """
    为一个聚类计算外包大三角形（方向感知）

    Args:
        triangles: 所有三角形数据
        cluster_indices: 聚类中三角形的索引列表
        direction: 方向 ("上"=朝右, "下"=朝左, None=自动检测或默认朝右)

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

    # 如果没有指定方向，尝试从三角形数据中推断
    if direction is None and cluster_triangles:
        # 从第一个三角形获取方向信息
        if 'direction' in cluster_triangles[0]:
            direction = cluster_triangles[0]['direction']

    # 计算时空边界
    xs = [v[0] for v in all_vertices]  # KP坐标
    ys = [v[1] for v in all_vertices]  # 时间坐标

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 添加一些边距，确保完全覆盖
    kp_margin = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 1.0  # 10%边距，最小1km
    time_margin = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 30  # 10%边距，最小30分钟

    # 根据方向构造大三角形的三个顶点
    if direction == "下":
        # 下行方向：三角形朝左（KP减小方向）←
        # 顶点1（左顶点）：KP最小位置，时间在中间 - 三角形的左尖端
        apex_x = min_x - kp_margin
        apex_y = (min_y + max_y) / 2

        # 顶点2（右上角）：KP最大位置，时间最早
        right_top_x = max_x + kp_margin
        right_top_y = min_y - time_margin

        # 顶点3（右下角）：KP最大位置，时间最晚
        right_bottom_x = max_x + kp_margin
        right_bottom_y = max_y + time_margin

        hull_vertices = [(apex_x, apex_y), (right_top_x, right_top_y), (right_bottom_x, right_bottom_y)]

        # 计算大三角形的属性
        center_x = (apex_x + right_top_x + right_bottom_x) / 3
        center_y = (apex_y + right_top_y + right_bottom_y) / 3

        width = right_top_x - apex_x  # KP方向的宽度
        height = right_bottom_y - right_top_y  # 时间方向的高度
        kp_range = (apex_x, right_top_x)

    else:
        # 上行方向或未指定：三角形朝右（KP增大方向）→
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
        kp_range = (left_top_x, apex_x)

    area = 0.5 * abs(width) * height  # 使用abs确保面积为正

    hull_info = {
        'vertices': hull_vertices,
        'center': (center_x, center_y),
        'area': area,
        'width': abs(width),
        'height': height,
        'kp_range': kp_range,
        'time_range': (min_y - time_margin, max_y + time_margin),
        'cluster_indices': cluster_indices,
        'cluster_size': len(cluster_indices),
        'covered_triangles': cluster_triangles,
        'direction': direction  # 记录方向信息
    }

    return hull_info


def calculate_all_hull_triangles(triangles: List[Dict[str, Any]],
                                clusters: List[List[int]],
                                direction: str = None) -> List[Dict[str, Any]]:
    """
    为所有聚类计算外包大三角形（方向感知）

    Args:
        triangles: 所有三角形数据
        clusters: 聚类结果，每个元素是三角形索引列表
        direction: 方向 ("上"/"下"/None)，影响hull三角形的朝向

    Returns:
        所有大三角形的信息列表
    """
    hulls = []

    for cluster_idx, cluster_indices in enumerate(clusters):
        if len(cluster_indices) < 2:  # 跳过太小的聚类
            continue

        hull = calculate_cluster_hull_triangle(triangles, cluster_indices, direction)
        if hull:
            hull['cluster_id'] = cluster_idx
            hulls.append(hull)

    print(f"Calculated {len(hulls)} hull triangles (direction: {direction or 'auto'})")

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
    print(f"Drawing {len(triangles)} polygons and {len(hulls) if hulls else 0} hull triangles...")
    
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
                   label=f'クラスタ {hull["cluster_id"]} ({hull["cluster_size"]}個)')
            
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
    
    title = "渋滞イベント多角形分析"
    if triangle_count > 0 and quad_count > 0:
        title += f"（三角形: {triangle_count}, 四角形: {quad_count}）"
    elif triangle_count > 0:
        title += f"（{triangle_count}個の三角形）"
    elif quad_count > 0:
        title += f"（{quad_count}個の四角形）"
    
    if hulls:
        title += f" - {len(hulls)}個の渋滞ホットスポット"
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
    print(f"Image saved to: {output_path}")
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
        return "No significant traffic congestion hotspots detected."
    
    # 按聚类大小排序
    sorted_hulls = sorted(hulls, key=lambda x: x['cluster_size'], reverse=True)
    
    report_lines = [
        "## Traffic Congestion Hotspot Analysis Report",
        f"Detected {len(hulls)} major traffic congestion hotspots:\n"
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
            f"### Hotspot Area {i+1} (Cluster {cluster_id})",
            f"- **Congestion Events**: {cluster_size} times",
            f"- **Affected Section**: KP {kp_start:.1f} - {kp_end:.1f} km",
            f"- **Time Period**: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d}",
            f"- **Coverage Area**: {hull['area']:.1f} square units",
            f"- **Section Length**: {hull['width']:.1f} km",
            f"- **Duration**: {hull['height']:.0f} minutes\n"
        ])
    
    # 添加总结
    total_events = sum(hull['cluster_size'] for hull in hulls)
    avg_cluster_size = total_events / len(hulls)
    
    report_lines.extend([
        "### Overall Analysis",
        f"- **Total Congestion Events**: {total_events} times",
        f"- **Average per Hotspot**: {avg_cluster_size:.1f} congestion events",
        f"- **Largest Hotspot**: {sorted_hulls[0]['cluster_size']} congestion events",
        f"- **Hotspot Concentration**: {'High' if avg_cluster_size > 5 else 'Medium' if avg_cluster_size > 2 else 'Low'}"
    ])
    
    return "\n".join(report_lines)


def process_multiple_traffic_data(file_paths: List[str], output_dir: str = "output",
                                 direction: str = None, target_year: int = None,
                                 target_month: int = None, target_day: int = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    处理多个CSV文件的完整交通数据处理流程：加载数据 → 生成三角形 → 统一可视化

    Args:
        file_paths: CSV文件路径列表
        output_dir: 输出目录
        direction: 方向信息 ("上", "下", 或 None)，用于生成唯一文件名
        target_year: 预测目标年份
        target_month: 预测目标月份
        target_day: 预测目标日期

    Returns:
        合并的三角形数据列表, 统一可视化图片路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    all_triangles = []
    file_info = []
    
    print(f"📊 Starting to process {len(file_paths)} CSV files...")
    
    for i, file_path in enumerate(file_paths):
        print(f"  Processing file {i+1}/{len(file_paths)}: {file_path}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"    ⚠️ File not found, skipping: {file_path}")
                continue
            
            # 1. 加载数据
            df = load_and_process_data(file_path)

            # 2. 生成三角形（使用方向感知版本）
            if direction:
                # 提取道路类型
                road_type = extract_road_type_from_filename(file_path)
                triangles = generate_direction_aware_polygons(df, direction, road_type)
            else:
                # 向后兼容：使用传统方法
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
            
            print(f"    ✅ Successfully processed, generated {len(triangles)} triangles")
            
        except Exception as e:
            print(f"    ❌ Processing failed: {str(e)}")
            continue
    
    if not all_triangles:
        print("⚠️ No files were successfully processed")
        return [], ""
    
    print(f"📈 Total generated {len(all_triangles)} triangles from {len(file_info)} files")

    # 4. 统一可视化所有三角形
    fig_name = generate_unique_filename("triangles_multi", file_paths, direction,
                                       target_year=target_year, target_month=target_month, target_day=target_day)
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
        print("No triangle data available for visualization")
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
    plt.ylabel('時間 (分)', fontsize=12)
    plt.title(f'交通渋滞三角形可視化 - 多ファイル統合\n合計 {len(triangles)} 個の三角形', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    if file_info:
        legend_elements = []
        for i, info in enumerate(file_info):
            color = colors[i % len(colors)]
            label = f"{info['file_name']} ({info['triangle_count']}個)"
            legend_elements.append(plt.Rectangle((0,0),1,1, color=color, alpha=0.6, label=label))
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Multi-file triangle visualization saved to: {output_path}")
    return output_path

def generate_direction_aware_polygons(df: pd.DataFrame, direction: str = None, 
                                     road_type: str = "関越道") -> List[Dict[str, Any]]:
    """
    生成方向感知的多边形（三角形或四边形），根据方向正确设置几何形状
    
    Args:
        df: 交通数据DataFrame
        direction: 方向 ("上" 表示KP增大方向, "下" 表示KP减小方向)
        road_type: 道路类型，用于加载道路信息
        
    Returns:
        多边形列表，每个多边形包含详细信息
    """
    print(f"生成方向感知的多边形数据... 方向: {direction or '未指定'}")
    
    # 解析时间列
    df = df.copy()
    df["発生_t"] = df["発生時刻"].apply(parse_hhmm_or_int)
    df["ピーク_t"] = df["ピーク時刻"].apply(parse_hhmm_or_int)
    
    polygons = []
    
    # 加载道路信息获取KP范围
    kp_min, kp_max = get_road_kp_range(road_type, direction)
    
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
        
        # 根据方向决定拥堵扩展方向
        if direction == "上":  # KP增大方向（上り）
            # 拥堵向KP增大的方向扩展
            end_kp_start = start_kp + start_jam_length
            end_kp_peak = start_kp + peak_length
        elif direction == "下":  # KP减小方向（下り）
            # 拥堵向KP减小的方向扩展
            end_kp_start = start_kp - start_jam_length
            end_kp_peak = start_kp - peak_length
        else:
            # 未指定方向，使用默认逻辑（假设向KP增大方向扩展）
            end_kp_start = start_kp + start_jam_length
            end_kp_peak = start_kp + peak_length
        
        # 定义四个关键顶点
        vertex1 = (start_kp, start_time)           # 拥堵开始起点
        vertex2 = (end_kp_start, start_time)       # 拥堵开始终点
        vertex3 = (end_kp_peak, peak_time)         # 拥堵峰值终点
        vertex4 = (start_kp, end_time)             # 拥堵结束起点
        
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
        width = abs(kp_end - kp_start)  # 使用绝对值确保正数
        height = time_end - time_start
        
        # 获取区间名称
        location_name = get_kp_location_name(road_type, start_kp, direction)
        
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
            'direction': direction,             # 方向信息
            'road_type': road_type,             # 道路类型
            'location_name': location_name,     # 区间名称
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
    
    print(f"✅ 生成了 {len(polygons)} 个方向感知的多边形")
    return polygons


def generate_polygons_from_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    向后兼容：从数据生成多边形（三角形或四边形），返回多边形列表，每个多边形包含详细信息
    根据拥堵演化过程生成准确的几何图形
    """
    print("Generating polygon data...")
    
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
    
    print(f"Generated {len(polygons)} polygons")
    print(f"  - Triangles: {triangle_count}")
    print(f"  - Quadrilaterals: {quad_count}")
    
    return polygons


def visualize_polygons(polygons: List[Dict[str, Any]], output_path: str = "polygon_visualization.png") -> str:
    """
    可视化多边形并保存图片，支持三角形和四边形的区分显示
    """
    print(f"Drawing {len(polygons)} polygons...")
    
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
    ax.set_title(f"渋滞イベント多角形可視化\n三角形: {triangle_count}個, 四角形: {quad_count}個", fontsize=14)
    ax.set_ylim(0, 24*60)
    ax.invert_yaxis()  # 上方=凌晨，下方=深夜
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.4, label=f'三角形 ({triangle_count}個)'),
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.4, label=f'四角形 ({quad_count}個)')
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
    print(f"Image saved to: {output_path}")
    plt.close()  # 关闭图形以释放内存
    
    return output_path



# 为了向后兼容，保留原函数名但指向新函数
def generate_triangles_from_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    向后兼容函数：现在生成多边形（三角形或四边形）
    """
    return generate_polygons_from_data(df)


def generate_unique_filename(base_name: str, file_paths: List[str] = None, direction: str = None,
                            extension: str = "png", target_year: int = None, target_month: int = None,
                            target_day: int = None) -> str:
    """
    生成唯一的文件名，包含日期、道路、方向等信息

    Args:
        base_name: 基础名称 (如 "triangles", "hulls" 等)
        file_paths: 文件路径列表，用于提取路线和日期信息
        direction: 方向信息 ("上", "下", 或 None)
        extension: 文件扩展名
        target_year: 预测目标年份
        target_month: 预测目标月份
        target_day: 预测目标日期

    Returns:
        唯一的文件名
    """
    from datetime import datetime

    # 生成时间戳 - 优先使用目标日期
    if target_year and target_month and target_day:
        timestamp = f"{target_year}{target_month:02d}{target_day:02d}_{datetime.now().strftime('%H%M%S')}"
    elif target_year and target_month:
        timestamp = f"{target_year}{target_month:02d}01_{datetime.now().strftime('%H%M%S')}"
    else:
        # 回退到当前日期
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取路线信息
    route = "unknown"
    date_info = ""
    
    if file_paths and len(file_paths) > 0:
        # 从第一个文件路径提取信息
        first_file = os.path.basename(file_paths[0])
        # 例如: 関越道_上_2024_08-01.csv
        parts = first_file.split('_')
        if len(parts) >= 3:
            route = parts[0]  # 関越道
            if len(parts) >= 4:
                date_info = f"_{parts[2]}"  # _2024
                if len(parts) >= 5:
                    month_day = parts[3].split('.')[0]  # 08-01
                    date_info += f"_{month_day}"  # _2024_08-01
    
    # 构建文件名组件
    components = [base_name]
    
    if route != "unknown":
        components.append(route)
    
    if direction:
        components.append(f"{direction}り")
    
    if date_info:
        components.append(date_info.lstrip('_'))
    
    components.append(timestamp)
    
    # 生成最终文件名
    filename = "_".join(components) + f".{extension}"
    
    return filename


def visualize_triangles(triangles: List[Dict[str, Any]], output_path: str = "triangle_visualization.png") -> str:
    """
    向后兼容函数：现在可视化多边形
    """
    return visualize_polygons(triangles, output_path)


def group_files_by_direction(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    按方向分组文件路径
    
    Args:
        file_paths: 文件路径列表
    
    Returns:
        按方向分组的文件路径字典 {"上": [...], "下": [...]}
    """
    direction_groups = {"上": [], "下": []}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if "_上_" in filename:
            direction_groups["上"].append(file_path)
        elif "_下_" in filename:
            direction_groups["下"].append(file_path)
        else:
            # 如果无法识别方向，暂时跳过
            print(f"⚠️ Unable to determine direction for file: {filename}")
    
    return direction_groups


def process_direction_grouped_data(file_paths: List[str], user_direction: Optional[str] = None,
                                 output_dir: str = "output", target_year: int = None,
                                 target_month: int = None, target_day: int = None) -> Dict[str, Any]:
    """
    按方向分组处理交通数据

    Args:
        file_paths: 文件路径列表
        user_direction: 用户指定的方向（"上"/"下"/None）
        output_dir: 输出目录
        target_year: 预测目标年份
        target_month: 预测目标月份
        target_day: 预测目标日期

    Returns:
        包含方向分组数据的字典
    """
    result = {
        "direction_data": {},
        "triangles": [],  # 向后兼容
        "fig_path": ""    # 向后兼容
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if user_direction:
        # 单方向模式：只处理用户指定的方向
        print(f"🧭 Single direction mode: processing {user_direction} direction only")
        
        # 筛选指定方向的文件
        direction_files = [fp for fp in file_paths if f"_{user_direction}_" in os.path.basename(fp)]
        
        if direction_files:
            print(f"📁 Found {len(direction_files)} files for {user_direction} direction")
            
            # 处理指定方向的数据
            triangles, fig_path = process_multiple_traffic_data(direction_files, output_dir, user_direction,
                                                               target_year, target_month, target_day)
            
            # 存储结果
            result["direction_data"][user_direction] = {
                "triangles": triangles,
                "fig_path": fig_path
            }
            
            # 向后兼容
            result["triangles"] = triangles
            result["fig_path"] = fig_path
        else:
            print(f"❌ No files found for {user_direction} direction")
    
    else:
        # 双方向模式：分别处理上下两个方向
        print("🧭 Dual direction mode: processing both 上り and 下り directions")
        
        # 按方向分组文件
        direction_groups = group_files_by_direction(file_paths)
        
        all_triangles = []
        fig_paths = []
        
        for direction, files in direction_groups.items():
            if files:
                print(f"📁 Processing {len(files)} files for {direction} direction")
                
                # 处理该方向的数据
                triangles, fig_path = process_multiple_traffic_data(files, output_dir, direction,
                                                                   target_year, target_month, target_day)
                
                # 存储结果
                result["direction_data"][direction] = {
                    "triangles": triangles,
                    "fig_path": fig_path
                }
                
                # 合并用于向后兼容
                all_triangles.extend(triangles)
                fig_paths.append(fig_path)
            else:
                print(f"⚠️ No files found for {direction} direction")
        
        # 向后兼容：合并所有方向的数据
        result["triangles"] = all_triangles
        result["fig_path"] = fig_paths[0] if fig_paths else ""  # 主图片路径（第一个方向）
        result["fig_paths"] = fig_paths  # 所有方向的图片路径列表
    
    print(f"✅ Direction-grouped processing completed")
    print(f"  - Processed directions: {list(result['direction_data'].keys())}")
    print(f"  - Total triangles: {len(result['triangles'])}")
    
    return result


def visualize_direction_aware_polygons(triangles: List[Dict[str, Any]], output_path: str,
                                     road_type: str = "関越道", direction: str = None,
                                     title_suffix: str = "") -> str:
    """
    可视化方向感知的多边形，使用完整的道路KP范围作为x轴
    
    Args:
        triangles: 多边形数据列表
        output_path: 输出文件路径
        road_type: 道路类型
        direction: 方向信息
        title_suffix: 标题后缀
        
    Returns:
        生成的图片文件路径
    """
    if not triangles:
        print("⚠️ 没有多边形数据可供可视化")
        return output_path
    
    # 获取完整的道路KP范围
    kp_min, kp_max = get_road_kp_range(road_type, direction)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制每个多边形
    triangle_count = 0
    quad_count = 0
    
    for triangle in triangles:
        vertices = triangle['vertices']
        xs, ys = zip(*vertices)
        
        if triangle['shape_type'] == 'triangle':
            plt.fill(xs, ys, alpha=0.4, color='lightblue', edgecolor='blue', linewidth=0.5)
            triangle_count += 1
        else:  # quadrilateral
            plt.fill(xs, ys, alpha=0.4, color='lightcoral', edgecolor='red', linewidth=0.5)
            quad_count += 1
        
        # 可选：在中心标注ID
        center_x, center_y = triangle['center']
        plt.text(center_x, center_y, str(triangle['id']), 
                fontsize=6, ha='center', va='center', alpha=0.7)
    
    # 设置完整的KP范围作为x轴
    plt.xlim(kp_min - 1, kp_max + 1)
    
    # 设置y轴范围（时间）
    all_times = []
    for triangle in triangles:
        all_times.extend([v[1] for v in triangle['vertices']])
    
    if all_times:
        time_min = min(all_times)
        time_max = max(all_times)
        time_margin = (time_max - time_min) * 0.1 if (time_max - time_min) > 0 else 60
        plt.ylim(time_min - time_margin, time_max + time_margin)
    
    # 设置标签和标题
    plt.xlabel('KP位置 (km)', fontsize=12)
    plt.ylabel('時間 (00:00からの分)', fontsize=12)
    
    direction_text = f" ({direction}り)" if direction else ""
    title = f'{road_type}{direction_text} 交通渋滞可視化 {title_suffix}\n三角形: {triangle_count}個, 四角形: {quad_count}個'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.4, label=f'三角形 ({triangle_count}個)'),
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.4, label=f'四角形 ({quad_count}個)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 添加信息文本
    info_text = f'多边形数量: {len(triangles)}\nKP范围: {kp_min:.1f} - {kp_max:.1f} km'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 方向感知可视化已保存: {output_path}")
    return output_path