"""
Base agent interface and abstract classes for the multi-agent traffic prediction system.
Based on AI_DEVELOPMENT_GUIDE.md requirements.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time
import logging
from ..core.data_structures import AgentResponse, SystemConfig
from ..core.exceptions import AgentError


class AgentInterface(ABC):
    """每个Agent必须实现的接口"""

    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """每个Agent必须实现此方法"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict) -> bool:
        """输入验证必须实现"""
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """状态检查必须实现"""
        pass


class BaseAgent(AgentInterface):
    """Agent基类，提供通用功能"""

    def __init__(self, config: SystemConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = self._setup_logger()
        self._status = "initialized"
        self._error_count = 0
        self._last_execution_time = 0.0

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.agent_name}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.log_level))

        return logger

    def process(self, input_data: Dict) -> Dict:
        """
        处理输入数据的主方法，包含错误处理和监控
        """
        start_time = time.time()

        try:
            # 输入验证
            if not self.validate_input(input_data):
                raise AgentError(f"{self.agent_name}: 输入验证失败")

            self._status = "processing"
            self.logger.info(f"{self.agent_name} 开始处理", extra={"input_keys": list(input_data.keys())})

            # 实际处理逻辑
            result = self._process_internal(input_data)

            # 验证输出
            if not self._validate_output(result):
                raise AgentError(f"{self.agent_name}: 输出验证失败")

            self._status = "completed"
            execution_time = time.time() - start_time
            self._last_execution_time = execution_time

            self.logger.info(f"{self.agent_name} 处理完成", extra={
                "execution_time": execution_time,
                "output_keys": list(result.keys()) if isinstance(result, dict) else "non-dict"
            })

            return AgentResponse(
                success=True,
                data=result,
                execution_time=execution_time
            ).__dict__

        except Exception as e:
            self._error_count += 1
            self._status = "error"
            execution_time = time.time() - start_time

            self.logger.error(f"{self.agent_name} 处理失败", extra={
                "error": str(e),
                "execution_time": execution_time,
                "error_count": self._error_count
            })

            # 根据严格的错误处理原则，不隐藏错误
            raise AgentError(f"{self.agent_name} 处理失败: {e}") from e

    @abstractmethod
    def _process_internal(self, input_data: Dict) -> Dict:
        """子类必须实现的实际处理逻辑"""
        pass

    def _validate_output(self, output_data: Dict) -> bool:
        """验证输出数据，子类可以重写"""
        return output_data is not None and isinstance(output_data, dict)

    def get_status(self) -> Dict:
        """获取Agent状态"""
        return {
            "agent_name": self.agent_name,
            "status": self._status,
            "error_count": self._error_count,
            "last_execution_time": self._last_execution_time,
            "config_loaded": self.config is not None
        }

    def reset_status(self) -> None:
        """重置状态"""
        self._status = "initialized"
        self._error_count = 0
        self._last_execution_time = 0.0
        self.logger.info(f"{self.agent_name} 状态已重置")


def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            self.logger.info(f"{func.__name__} 执行成功", extra={"duration": duration})
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"{func.__name__} 执行失败", extra={
                "duration": duration,
                "error": str(e)
            })
            raise
    return wrapper