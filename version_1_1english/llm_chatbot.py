"""
基于LLM的Chatbot节点实现
使用大语言模型的语义理解能力来解析用户输入
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

# 这里可以使用任何LLM API，比如OpenAI、Claude、或本地模型
# 为了演示，我们使用一个模拟的LLM调用函数

def call_llm(system_prompt: str, user_input: str) -> str:
    """
    调用LLM的函数（这里需要根据实际使用的LLM API进行实现）
    可以是OpenAI GPT、Claude、或者本地部署的模型
    """
    # 这里是一个模拟实现，实际使用时需要替换为真实的LLM调用
    # 例如：
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_input}
    #     ]
    # )
    # return response.choices[0].message.content
    
    # 模拟LLM响应 - 在实际使用时删除这部分
    mock_responses = {
        "请分析関越高速公路2024年的交通情况": {
            "route": "関越高速道路",
            "time": "2024",
            "route_type": "高速道路",
            "analysis_type": "交通状况分析",
            "confidence": 0.95
        },
        "我想看看関越道路在令和5年的数据": {
            "route": "関越道路", 
            "time": "令和5年",
            "route_type": "道路",
            "analysis_type": "数据查看",
            "confidence": 0.90
        },
        "分析东名高速2023年拥堵情况": {
            "route": "東名高速道路",
            "time": "2023",
            "route_type": "高速道路", 
            "analysis_type": "拥堵分析",
            "confidence": 0.88
        }
    }
    
    # 简单的模拟逻辑
    for key in mock_responses:
        if any(word in user_input for word in key.split()):
            return json.dumps(mock_responses[key], ensure_ascii=False, indent=2)
    
    # 默认响应
    return json.dumps({
        "route": "関越高速道路",
        "time": "2024", 
        "route_type": "高速道路",
        "analysis_type": "交通分析",
        "confidence": 0.70
    }, ensure_ascii=False, indent=2)


def create_system_prompt() -> str:
    """
    创建用于解析交通数据查询的system prompt
    """
    system_prompt = """
你是一个专业的交通数据分析助手，专门解析用户对日本高速道路和道路交通数据的查询请求。

## 任务说明
从用户的自然语言输入中提取以下信息：
1. **路线名称** - 具体的道路或高速公路名称
2. **时间范围** - 年份、年号或时间段
3. **路线类型** - 高速道路、国道、县道等
4. **分析类型** - 用户想要进行的分析类型

## 支持的路线
- 関越高速道路/関越道/関越高速
- 東名高速道路/東名道/東名高速  
- 中央高速道路/中央道/中央高速
- 首都高速道路/首都高速
- 国道XX号
- 県道XX号

## 时间格式处理
- 西历年份：2024年、2023、2022等
- 日本年号：令和X年、平成X年
- 年号转换：令和元年=2019年，令和5年=2023年，平成30年=2018年

## 输出格式
请严格按照以下JSON格式输出，不要包含任何其他文字：

```json
{
    "route": "提取的路线名称（标准化后）",
    "time": "提取的时间（转换为西历年份）", 
    "route_type": "路线类型（高速道路/国道/県道等）",
    "analysis_type": "分析类型（交通分析/拥堵分析/数据查看等）",
    "confidence": 0.0-1.0的置信度分数
}
```

## 示例
输入："请分析関越高速公路2024年的交通情况"
输出：
```json
{
    "route": "関越高速道路",
    "time": "2024",
    "route_type": "高速道路", 
    "analysis_type": "交通分析",
    "confidence": 0.95
}
```

输入："我想看看関越道路在令和5年的数据"
输出：
```json
{
    "route": "関越道路",
    "time": "2023", 
    "route_type": "道路",
    "analysis_type": "数据查看",
    "confidence": 0.90
}
```

请只返回JSON格式的结果，不要包含任何解释或额外文字。
"""
    return system_prompt


def parse_llm_response(llm_output: str) -> Dict[str, Any]:
    """
    解析LLM的JSON输出
    """
    try:
        # 尝试直接解析JSON
        parsed = json.loads(llm_output)
        return parsed
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return parsed
            except json.JSONDecodeError:
                pass
        
        # 如果还是失败，返回默认值
        print(f"警告：无法解析LLM输出，使用默认值: {llm_output}")
        return {
            "route": "関越高速道路",
            "time": "2024",
            "route_type": "高速道路",
            "analysis_type": "交通分析", 
            "confidence": 0.5
        }


def convert_japanese_year(time_str: str) -> str:
    """
    转换日本年号到西历年份
    """
    if not time_str:
        return "2024"
    
    # 令和年号转换
    if "令和" in time_str:
        import re
        match = re.search(r'令和(\d+)', time_str)
        if match:
            reiwa_year = int(match.group(1))
            return str(2018 + reiwa_year)  # 令和元年=2019年
    
    # 平成年号转换  
    elif "平成" in time_str:
        import re
        match = re.search(r'平成(\d+)', time_str)
        if match:
            heisei_year = int(match.group(1))
            return str(1988 + heisei_year)  # 平成元年=1989年
    
    # 提取4位数字年份
    import re
    year_match = re.search(r'(20\d{2})', time_str)
    if year_match:
        return year_match.group(1)
    
    return time_str


def generate_file_path_smart(route: str, time: str, route_type: str) -> str:
    """
    基于LLM解析结果智能生成文件路径
    """
    base_data_dir = "data"
    
    # 标准化路线名称
    route_mapping = {
        "関越高速道路": "関越",
        "関越道路": "関越", 
        "関越高速": "関越",
        "関越道": "関越",
        "東名高速道路": "東名",
        "東名道": "東名",
        "東名高速": "東名",
        "中央高速道路": "中央", 
        "中央道": "中央",
        "中央高速": "中央",
        "首都高速道路": "首都高速",
        "首都高速": "首都高速"
    }
    
    standard_route = route_mapping.get(route, "関越")
    year = convert_japanese_year(time)
    
    # 优先级文件路径生成
    potential_files = [
        f"{base_data_dir}/{standard_route}{year}_cleaned.csv",
        f"{base_data_dir}/{standard_route}{year}.csv", 
        f"{base_data_dir}/{standard_route}{year}_cleaned.xlsx",
        f"{base_data_dir}/{standard_route}{year}.xlsx",
        f"{base_data_dir}/meta_data/★{year}_関東支社渋滞データ（01-12）SIC分割【コード変換・BT記入・本社BT】特定更新工事-緊急工事(1～12月分まで).csv"
    ]
    
    # 检查文件是否存在
    for file_path in potential_files:
        if os.path.exists(file_path):
            return file_path
    
    # 默认文件
    return f"{base_data_dir}/関越2024_cleaned.csv"


def llm_chatbot(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于LLM的chatbot节点实现
    """
    print("In node: llm_chatbot")
    user_input = state.get("user_input", "")
    
    if not user_input:
        print("用户输入为空，使用默认设置")
        return {
            "file_path": "data/関越2024_cleaned.csv",
            "route": "未指定",
            "ts": "未指定",
            "analysis_type": "交通分析",
            "confidence": 0.5
        }
    
    print(f"用户输入: {user_input}")
    
    try:
        # 1. 创建system prompt
        system_prompt = create_system_prompt()
        
        # 2. 调用LLM
        print("正在调用LLM进行语义解析...")
        llm_output = call_llm(system_prompt, user_input)
        print(f"LLM原始输出: {llm_output}")
        
        # 3. 解析LLM输出
        parsed_result = parse_llm_response(llm_output)
        print(f"解析结果: {parsed_result}")
        
        # 4. 提取信息
        route = parsed_result.get("route", "関越高速道路")
        time = parsed_result.get("time", "2024")
        route_type = parsed_result.get("route_type", "高速道路")
        analysis_type = parsed_result.get("analysis_type", "交通分析")
        confidence = parsed_result.get("confidence", 0.8)
        
        # 5. 生成文件路径
        file_path = generate_file_path_smart(route, time, route_type)
        print(f"生成的文件路径: {file_path}")
        
        # 6. 返回结构化结果
        result = {
            "file_path": file_path,
            "route": route,
            "ts": convert_japanese_year(time),
            "analysis_type": analysis_type,
            "confidence": confidence,
            "route_type": route_type
        }
        
        print(f"Chatbot输出结果: {result}")
        return result
        
    except Exception as e:
        print(f"LLM chatbot处理出错: {str(e)}")
        # 降级到传统方法
        from functions import extract_route_and_time, generate_file_path
        
        extracted_info = extract_route_and_time(user_input)
        route = extracted_info.get('route')
        time = extracted_info.get('time')
        file_path = generate_file_path(route, time)
        
        return {
            "file_path": file_path,
            "route": route or "未指定", 
            "ts": time or "未指定",
            "analysis_type": "交通分析",
            "confidence": 0.6
        }


# 测试函数
def test_llm_chatbot():
    """
    测试LLM chatbot功能
    """
    test_cases = [
        "请分析関越高速公路2024年的交通情况",
        "我想看看関越道路在令和5年的数据", 
        "分析东名高速2023年拥堵情况",
        "查看中央道2022年的渋滞数据",
        "2021年首都高速的交通状况如何？"
    ]
    
    print("=== LLM Chatbot 测试 ===\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"测试案例 {i}:")
        print(f"输入: {test_input}")
        
        # 模拟state
        state = {"user_input": test_input}
        result = llm_chatbot(state)
        
        print(f"输出: {result}")
        print("-" * 50)


if __name__ == "__main__":
    test_llm_chatbot() 