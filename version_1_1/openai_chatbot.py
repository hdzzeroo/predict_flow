"""
基于OpenAI API的Chatbot节点实现
演示如何集成真实的LLM API来解析用户输入
"""

import json
import os
from typing import Dict, Any, Optional

# 真实的OpenAI API集成示例
def call_openai_api(system_prompt: str, user_input: str, api_key: str = None) -> str:
    """
    调用OpenAI GPT API
    
    Args:
        system_prompt: 系统提示词
        user_input: 用户输入
        api_key: OpenAI API密钥
    
    Returns:
        LLM的响应
    """
    try:
        import openai
        
        # 设置API密钥
        if api_key:
            openai.api_key = api_key
        else:
            # 从环境变量获取
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量或传入api_key参数")
        
        # 调用GPT API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 或 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,  # 降低随机性，确保输出格式一致
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        print("警告：openai包未安装，请运行 'pip install openai'")
        return call_fallback_llm(system_prompt, user_input)
    except Exception as e:
        print(f"OpenAI API调用失败: {str(e)}")
        return call_fallback_llm(system_prompt, user_input)


def call_claude_api(system_prompt: str, user_input: str, api_key: str = None) -> str:
    """
    调用Claude API (Anthropic)
    """
    try:
        import anthropic
        
        # 设置API密钥
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
        else:
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # 调用Claude API
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        return response.content[0].text
        
    except ImportError:
        print("警告：anthropic包未安装，请运行 'pip install anthropic'")
        return call_fallback_llm(system_prompt, user_input)
    except Exception as e:
        print(f"Claude API调用失败: {str(e)}")
        return call_fallback_llm(system_prompt, user_input)


def call_local_llm(system_prompt: str, user_input: str, model_path: str = None) -> str:
    """
    调用本地部署的LLM（例如通过Ollama）
    """
    try:
        import requests
        
        # Ollama API端点
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model_path or "llama2",  # 或其他本地模型
            "prompt": f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"本地LLM调用失败: {response.status_code}")
            
    except Exception as e:
        print(f"本地LLM调用失败: {str(e)}")
        return call_fallback_llm(system_prompt, user_input)


def call_fallback_llm(system_prompt: str, user_input: str) -> str:
    """
    降级处理：使用规则基方法
    """
    print("降级到规则基方法...")
    
    # 简单的模式匹配逻辑
    route_keywords = {
        "関越": "関越高速道路",
        "東名": "東名高速道路", 
        "中央": "中央高速道路",
        "首都": "首都高速道路"
    }
    
    route = "関越高速道路"  # 默认
    for keyword, full_name in route_keywords.items():
        if keyword in user_input:
            route = full_name
            break
    
    # 提取年份
    import re
    year_match = re.search(r'(20\d{2})', user_input)
    time = year_match.group(1) if year_match else "2024"
    
    # 分析类型
    if any(word in user_input for word in ["分析", "analyze"]):
        analysis_type = "交通分析"
    elif any(word in user_input for word in ["查看", "看看", "view"]):
        analysis_type = "数据查看"
    elif any(word in user_input for word in ["拥堵", "渋滞", "congestion"]):
        analysis_type = "拥堵分析"
    else:
        analysis_type = "交通分析"
    
    return json.dumps({
        "route": route,
        "time": time,
        "route_type": "高速道路",
        "analysis_type": analysis_type,
        "confidence": 0.7
    }, ensure_ascii=False)


def create_enhanced_system_prompt() -> str:
    """
    创建增强版的system prompt，包含更多示例和约束
    """
    return """
你是一个专业的日本交通数据分析助手。请从用户输入中精确提取交通查询信息。

## 任务目标
解析用户的自然语言查询，提取：
1. 道路名称（标准化）
2. 时间信息（转换为西历年份）
3. 道路类型分类
4. 分析意图识别
5. 解析置信度评估

## 支持的道路网络
### 高速道路
- 関越自动车道：関越高速道路、関越道、関越自动车道、関越高速
- 東名高速道路：東名高速、東名道、東名自动车道
- 中央自动车道：中央道、中央高速道路、中央高速、中央自动车道
- 首都高速道路：首都高速、首都圏中央连络自动车道

### 一般道路  
- 国道系统：国道1号、国道16号等
- 都道府県道：県道XX号

## 时间处理规则
### 西历年份
- 直接数字：2024、2023、2022等
- 带年字：2024年、2023年等

### 日本年号转换
- 令和年号：令和6年→2024年、令和5年→2023年、令和元年→2019年
- 平成年号：平成30年→2018年、平成29年→2017年
- 计算公式：令和X年 = 2018+X年，平成X年 = 1988+X年

## 输出格式要求
严格按照以下JSON Schema输出，不得包含任何解释文字：

```json
{
  "route": "标准化道路名称",
  "time": "西历年份（4位数字）",
  "route_type": "道路类型（高速道路/国道/県道）",
  "analysis_type": "分析类型（交通分析/拥堵分析/数据查看/状况监控）",
  "confidence": 浮点数（0.0-1.0，表示解析置信度）
}
```

## 示例对照表
| 用户输入 | 标准输出 |
|---------|---------|
| "请分析関越高速公路2024年的交通情况" | {"route": "関越高速道路", "time": "2024", "route_type": "高速道路", "analysis_type": "交通分析", "confidence": 0.95} |
| "我想看看関越道路在令和5年的数据" | {"route": "関越高速道路", "time": "2023", "route_type": "高速道路", "analysis_type": "数据查看", "confidence": 0.90} |
| "2022年東名高速的拥堵状况如何？" | {"route": "東名高速道路", "time": "2022", "route_type": "高速道路", "analysis_type": "拥堵分析", "confidence": 0.92} |
| "查看中央道平成30年渋滞数据" | {"route": "中央高速道路", "time": "2018", "route_type": "高速道路", "analysis_type": "数据查看", "confidence": 0.88} |

## 关键约束
1. 输出必须是有效的JSON格式
2. time字段必须是4位西历年份
3. confidence值需要基于解析的确定性程度
4. 如果无法识别具体道路，默认使用"関越高速道路"
5. 如果无法识别年份，默认使用"2024"

请只返回JSON结果，不要添加任何说明文字。
"""


class LLMConfig:
    """LLM配置类"""
    def __init__(self, 
                 provider: str = "openai",  # openai, claude, local, fallback
                 api_key: str = None,
                 model_name: str = None,
                 temperature: float = 0.1):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature


def call_llm_with_config(config: LLMConfig, system_prompt: str, user_input: str) -> str:
    """
    根据配置调用相应的LLM
    """
    if config.provider == "openai":
        return call_openai_api(system_prompt, user_input, config.api_key)
    elif config.provider == "claude":
        return call_claude_api(system_prompt, user_input, config.api_key)
    elif config.provider == "local":
        return call_local_llm(system_prompt, user_input, config.model_name)
    else:
        return call_fallback_llm(system_prompt, user_input)


def enhanced_llm_chatbot(state: Dict[str, Any], llm_config: LLMConfig = None) -> Dict[str, Any]:
    """
    增强版LLM chatbot，支持多种LLM后端
    """
    print("In node: enhanced_llm_chatbot")
    user_input = state.get("user_input", "")
    
    # 默认配置
    if llm_config is None:
        llm_config = LLMConfig(provider="fallback")  # 默认使用降级方法
    
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
    print(f"使用LLM后端: {llm_config.provider}")
    
    try:
        # 1. 创建增强版system prompt
        system_prompt = create_enhanced_system_prompt()
        
        # 2. 调用配置的LLM
        print("正在调用LLM进行语义解析...")
        llm_output = call_llm_with_config(llm_config, system_prompt, user_input)
        print(f"LLM原始输出: {llm_output}")
        
        # 3. 解析LLM输出
        from llm_chatbot import parse_llm_response, generate_file_path_smart, convert_japanese_year
        
        parsed_result = parse_llm_response(llm_output)
        print(f"解析结果: {parsed_result}")
        
        # 4. 提取和验证信息
        route = parsed_result.get("route", "関越高速道路")
        time = parsed_result.get("time", "2024")
        route_type = parsed_result.get("route_type", "高速道路")
        analysis_type = parsed_result.get("analysis_type", "交通分析")
        confidence = parsed_result.get("confidence", 0.8)
        
        # 验证时间格式
        try:
            year_int = int(time)
            if year_int < 2000 or year_int > 2030:
                time = "2024"  # 默认年份
        except:
            time = convert_japanese_year(time)
        
        # 5. 生成文件路径
        file_path = generate_file_path_smart(route, time, route_type)
        print(f"生成的文件路径: {file_path}")
        
        # 6. 返回结构化结果
        result = {
            "file_path": file_path,
            "route": route,
            "ts": time,
            "analysis_type": analysis_type,
            "confidence": confidence,
            "route_type": route_type,
            "llm_provider": llm_config.provider
        }
        
        print(f"Enhanced Chatbot输出结果: {result}")
        return result
        
    except Exception as e:
        print(f"Enhanced LLM chatbot处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 最终降级到传统方法
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
            "confidence": 0.6,
            "llm_provider": "fallback"
        }


# 使用示例和测试
def test_enhanced_chatbot():
    """
    测试增强版chatbot的不同配置
    """
    test_cases = [
        "请分析関越高速公路2024年的交通情况",
        "我想看看関越道路在令和5年的数据", 
        "分析东名高速2023年拥堵情况",
        "查看中央道平成30年的渋滞数据",
        "2021年首都高速的交通状况如何？"
    ]
    
    # 测试不同的LLM配置
    configs = [
        LLMConfig(provider="fallback"),  # 降级方法
        # LLMConfig(provider="openai", api_key="your-openai-key"),  # OpenAI
        # LLMConfig(provider="claude", api_key="your-claude-key"),  # Claude
        # LLMConfig(provider="local", model_name="llama2"),  # 本地模型
    ]
    
    for config in configs:
        print(f"\n=== 测试 {config.provider.upper()} 后端 ===")
        
        for i, test_input in enumerate(test_cases[:2], 1):  # 只测试前两个
            print(f"\n测试案例 {i}:")
            print(f"输入: {test_input}")
            
            state = {"user_input": test_input}
            result = enhanced_llm_chatbot(state, config)
            
            print(f"输出: {result}")
            print("-" * 50)


if __name__ == "__main__":
    test_enhanced_chatbot() 