"""
独立测试chatbot功能
"""

import os
import re
from typing import Dict, Optional


def extract_route_and_time(user_input: str) -> Dict[str, Optional[str]]:
    """
    从用户输入中提取路名和时间信息
    """
    # 定义路名模式（支持中文路名）
    route_patterns = [
        r'(関越.*?(?:高速|道路|線))',  # 関越高速道路等
        r'(東名.*?(?:高速|道路|線))',  # 東名高速道路等
        r'(中央.*?(?:高速|道路|線))',  # 中央高速道路等
        r'(首都.*?(?:高速|道路|線))',  # 首都高速道路等
        r'(国道\d+号)',              # 国道XX号
        r'(県道\d+号)',              # 県道XX号
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
    base_data_dir = "../data"  # 相对于version_1_1目录
    
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


def test_chatbot_logic(user_input: str) -> dict:
    """
    测试chatbot逻辑
    """
    print(f"用户输入: {user_input}")
    
    # 提取路名和时间
    extracted_info = extract_route_and_time(user_input)
    route = extracted_info.get('route')
    time = extracted_info.get('time')
    
    print(f"提取的路名: {route}")
    print(f"提取的时间: {time}")
    
    # 生成文件路径
    file_path = generate_file_path(route, time)
    
    print(f"生成的文件路径: {file_path}")
    
    return {
        "file_path": file_path,
        "route": route or "未指定",
        "ts": time or "未指定"
    }


if __name__ == "__main__":
    # 测试不同的用户输入
    test_inputs = [
        "请分析关越高速公路2024年的交通情况",
        "我想看看関越道路在2023年的数据",
        "分析2022年関越高速的渋滞情况",
        "请查看関越线路的最新数据",
        "查看令和6年的高速公路数据",
        "平成30年東名高速的情况如何",
        "国道1号的交通状况分析",
    ]
    
    print("=== Chatbot 功能测试 ===\n")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"{i}. 测试输入: {test_input}")
        print("-" * 50)
        result = test_chatbot_logic(test_input)
        print(f"结果: {result}")
        print("=" * 60)
        print() 