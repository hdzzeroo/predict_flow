"""
LLM client with vision support for the multi-agent traffic prediction system.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

import base64
import io
import json
from typing import Dict, Any, Optional
import requests
import logging
from PIL import Image

from ..core.data_structures import LLMConfig
from ..core.exceptions import LLMError


class LLMClient:
    """LLM客户端 - 支持视觉输入"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("LLMClient")
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.config.api_key:
            raise LLMError("API密钥未配置")
        if not self.config.model:
            raise LLMError("模型名称未配置")

    def call_with_vision(self, prompt: str, image: Image.Image, **kwargs) -> str:
        """带视觉输入的LLM调用"""
        try:
            # 将图像转换为base64
            image_base64 = self._image_to_base64(image)

            # 准备请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # 构建请求体
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature)
            }

            # 发送请求
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            # 检查响应
            if response.status_code != 200:
                raise LLMError(f"API请求失败: {response.status_code} - {response.text}")

            response_data = response.json()

            # 提取结果
            if 'choices' not in response_data or not response_data['choices']:
                raise LLMError("API响应格式无效：缺少choices")

            result = response_data['choices'][0]['message']['content']

            if not result:
                raise LLMError("LLM返回空内容")

            # 验证响应内容
            self._validate_response(result, prompt)

            self.logger.info(f"LLM调用成功，响应长度: {len(result)}")
            return result

        except requests.exceptions.Timeout:
            raise LLMError(f"LLM API调用超时 (>{self.config.timeout}秒)")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"LLM API网络错误: {e}")
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            raise LLMError(f"LLM调用失败: {e}")

    def call_text_only(self, prompt: str, **kwargs) -> str:
        """纯文本LLM调用"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }

            payload = {
                "model": self.config.model.replace("-vision", ""),  # 移除vision后缀
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature)
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise LLMError(f"API请求失败: {response.status_code} - {response.text}")

            response_data = response.json()
            result = response_data['choices'][0]['message']['content']

            if not result:
                raise LLMError("LLM返回空内容")

            return result

        except Exception as e:
            raise LLMError(f"文本LLM调用失败: {e}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64字符串"""
        try:
            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 将图像保存到内存缓冲区
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)

            # 编码为base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return image_base64

        except Exception as e:
            raise LLMError(f"图像转换失败: {e}")

    def _validate_response(self, response: str, original_prompt: str) -> None:
        """验证LLM响应"""
        if len(response.strip()) < 10:
            raise LLMError("LLM响应过短，可能无效")

        # 检查是否包含预测相关信息
        prediction_keywords = ["预测", "prediction", "三角形", "triangle", "拥堵", "时间", "位置"]
        if not any(keyword in response.lower() for keyword in prediction_keywords):
            self.logger.warning(f"LLM响应可能不包含预测信息: {response[:200]}")

    def parse_prediction_response(self, response: str) -> Dict[str, Any]:
        """解析LLM预测响应"""
        try:
            # 尝试提取JSON格式的响应
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed_data = json.loads(json_str)
                return parsed_data

            # 如果没有找到JSON，尝试解析结构化文本
            return self._parse_structured_text(response)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON解析失败，尝试文本解析: {e}")
            return self._parse_structured_text(response)
        except Exception as e:
            raise LLMError(f"响应解析失败: {e}")

    def _parse_structured_text(self, response: str) -> Dict[str, Any]:
        """解析结构化文本响应"""
        # 这里实现一个简单的文本解析逻辑
        # 实际项目中可能需要更复杂的自然语言处理
        result = {
            "predicted_triangle": {},
            "explanation": response,
            "historical_basis": [],
            "confidence": 0.5  # 默认置信度
        }

        # 尝试提取时间信息
        import re
        time_pattern = r'(\d{1,2}):(\d{2})'
        times = re.findall(time_pattern, response)

        if len(times) >= 2:
            result["predicted_triangle"]["base_start_time"] = f"{times[0][0]}:{times[0][1]}"
            result["predicted_triangle"]["apex_time"] = f"{times[1][0]}:{times[1][1]}"
            if len(times) >= 3:
                result["predicted_triangle"]["base_end_time"] = f"{times[2][0]}:{times[2][1]}"

        # 尝试提取位置信息
        kp_pattern = r'KP\s*(\d+\.?\d*)'
        kps = re.findall(kp_pattern, response, re.IGNORECASE)

        if len(kps) >= 2:
            result["predicted_triangle"]["base_start_kp"] = float(kps[0])
            result["predicted_triangle"]["apex_kp"] = float(kps[1])
            if len(kps) >= 3:
                result["predicted_triangle"]["base_end_kp"] = float(kps[2])

        return result

    def test_connection(self) -> bool:
        """测试LLM连接"""
        try:
            test_response = self.call_text_only("Hello, please respond with 'Connection successful'.")
            return "successful" in test_response.lower()
        except Exception as e:
            self.logger.error(f"LLM连接测试失败: {e}")
            return False


class MockLLMClient(LLMClient):
    """模拟LLM客户端，用于测试"""

    def __init__(self):
        # 创建一个模拟配置
        mock_config = LLMConfig(
            model="mock-gpt-4-vision",
            api_key="mock-api-key",
            max_tokens=2000,
            temperature=0.1
        )
        super().__init__(mock_config)

    def call_with_vision(self, prompt: str, image: Image.Image, **kwargs) -> str:
        """模拟视觉LLM调用"""
        mock_response = """
        基于图像中的历史三角形分布，我分析了过去3年的拥堵模式。观察到在17:30-19:00时段，
        KP25-35区域存在显著的重叠模式。

        ```json
        {
            "predicted_triangle": {
                "apex_time": "18:15",
                "apex_kp": 30.2,
                "base_start_time": "17:30",
                "base_end_time": "19:00",
                "base_start_kp": 25.0,
                "base_end_kp": 35.0,
                "severity": "high",
                "confidence": 0.82
            },
            "explanation": "基于过去3年数据，发现在该时段存在显著的重叠区域，特别是在KP30附近的峰值模式",
            "historical_basis": ["2021年数据显示类似的峰值模式", "2022年确认了相同的时间段", "2023年验证了位置的一致性"]
        }
        ```
        """
        return mock_response

    def call_text_only(self, prompt: str, **kwargs) -> str:
        """模拟文本LLM调用"""
        return "Mock LLM response for text-only query."

    def test_connection(self) -> bool:
        """模拟连接测试"""
        return True