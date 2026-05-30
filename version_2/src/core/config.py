"""
Configuration management for the multi-agent traffic prediction system.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from .data_structures import SystemConfig, TriangleConfig, LLMConfig
from .exceptions import ConfigurationError

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This will load .env file from current directory or parent directories
except ImportError:
    # python-dotenv is optional, system will fall back to regular environment variables
    pass


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('TRAFFIC_PREDICTION_CONFIG')
        self._config: Optional[SystemConfig] = None

    def load_config(self, config_dict: Optional[Dict[str, Any]] = None) -> SystemConfig:
        """加载配置"""
        if config_dict:
            return self._create_config_from_dict(config_dict)

        if self.config_path and os.path.exists(self.config_path):
            return self._load_from_file()

        # 使用默认配置
        return self._create_default_config()

    def _load_from_file(self) -> SystemConfig:
        """从文件加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)

            return self._create_config_from_dict(config_dict)

        except Exception as e:
            raise ConfigurationError(f"无法加载配置文件 {self.config_path}: {e}")

    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """从字典创建配置"""
        try:
            # 创建三角形配置
            triangle_config_dict = config_dict.get('triangle_config', {})
            triangle_config = TriangleConfig(
                min_duration=triangle_config_dict.get('min_duration', 10),
                min_length=triangle_config_dict.get('min_length', 0.5),
                time_resolution=triangle_config_dict.get('time_resolution', 5),
                merge_threshold=triangle_config_dict.get('merge_threshold', 0.8)
            )

            # 创建LLM配置
            llm_config_dict = config_dict.get('llm_config', {})
            llm_config = LLMConfig(
                model=llm_config_dict.get('model', 'gpt-4-vision-preview'),
                api_key=llm_config_dict.get('api_key', os.getenv('OPENAI_API_KEY', '')),
                max_tokens=llm_config_dict.get('max_tokens', 2000),
                temperature=llm_config_dict.get('temperature', 0.1),
                timeout=llm_config_dict.get('timeout', 30)
            )

            # 创建系统配置
            config = SystemConfig(
                data_path=config_dict.get('data_path', './data'),
                output_path=config_dict.get('output_path', './output'),
                triangle_config=triangle_config,
                llm_config=llm_config,
                enable_logging=config_dict.get('enable_logging', True),
                log_level=config_dict.get('log_level', 'INFO')
            )

            self._validate_config(config)
            return config

        except Exception as e:
            raise ConfigurationError(f"配置创建失败: {e}")

    def _create_default_config(self) -> SystemConfig:
        """创建默认配置"""
        current_dir = Path(__file__).parent.parent.parent
        return SystemConfig(
            data_path=str(current_dir / 'data'),
            output_path=str(current_dir / 'output'),
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key=os.getenv('OPENAI_API_KEY', ''))
        )

    def _validate_config(self, config: SystemConfig) -> None:
        """验证配置"""
        if not config.data_path:
            raise ConfigurationError("data_path不能为空")

        if not config.output_path:
            raise ConfigurationError("output_path不能为空")

        # API key validation (allow empty for testing with mock LLM)
        if not config.llm_config.api_key and not self._is_test_mode():
            raise ConfigurationError("LLM API密钥未配置，请在.env文件中设置OPENAI_API_KEY")

        # 创建必要的目录
        os.makedirs(config.data_path, exist_ok=True)
        os.makedirs(config.output_path, exist_ok=True)

    def _is_test_mode(self) -> bool:
        """检查是否为测试模式"""
        return (
            os.getenv('PYTEST_CURRENT_TEST') is not None or  # pytest running
            os.getenv('TESTING') == 'true' or  # explicitly set testing flag
            'test' in os.getenv('PYTHON_ENV', '').lower()  # test environment
        )

    def save_config(self, config: SystemConfig, output_path: str) -> None:
        """保存配置到文件"""
        try:
            config_dict = {
                'data_path': config.data_path,
                'output_path': config.output_path,
                'triangle_config': {
                    'min_duration': config.triangle_config.min_duration,
                    'min_length': config.triangle_config.min_length,
                    'time_resolution': config.triangle_config.time_resolution,
                    'merge_threshold': config.triangle_config.merge_threshold
                },
                'llm_config': {
                    'model': config.llm_config.model,
                    'max_tokens': config.llm_config.max_tokens,
                    'temperature': config.llm_config.temperature,
                    'timeout': config.llm_config.timeout
                },
                'enable_logging': config.enable_logging,
                'log_level': config.log_level
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise ConfigurationError(f"保存配置失败: {e}")


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """获取当前配置"""
    return config_manager.load_config()


def setup_config(config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> SystemConfig:
    """设置配置"""
    global config_manager
    if config_path:
        config_manager = ConfigManager(config_path)
    return config_manager.load_config(config_dict)