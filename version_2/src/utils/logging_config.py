"""
Enhanced logging configuration for the multi-agent traffic prediction system.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON格式化器，用于结构化日志"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加额外字段
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created',
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'stack_info', 'exc_info', 'exc_text']:
                    log_entry[key] = value

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """性能监控日志记录器"""

    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)

    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """记录执行时间"""
        self.logger.info(f"Performance: {operation}", extra={
            'operation': operation,
            'duration_seconds': duration,
            'performance_metric': True,
            **kwargs
        })

    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """记录内存使用"""
        self.logger.info(f"Memory: {operation}", extra={
            'operation': operation,
            'memory_mb': memory_mb,
            'memory_metric': True,
            **kwargs
        })

    def log_api_call(self, api_name: str, duration: float, status: str, **kwargs):
        """记录API调用"""
        self.logger.info(f"API Call: {api_name}", extra={
            'api_name': api_name,
            'duration_seconds': duration,
            'status': status,
            'api_metric': True,
            **kwargs
        })


class SystemLogger:
    """系统日志记录器"""

    def __init__(self, logger_name: str = "system"):
        self.logger = logging.getLogger(logger_name)

    def log_agent_status(self, agent_name: str, status: str, **kwargs):
        """记录Agent状态"""
        self.logger.info(f"Agent Status: {agent_name}", extra={
            'agent_name': agent_name,
            'status': status,
            'agent_status': True,
            **kwargs
        })

    def log_workflow_step(self, step_name: str, status: str, **kwargs):
        """记录工作流程步骤"""
        self.logger.info(f"Workflow: {step_name}", extra={
            'step_name': step_name,
            'status': status,
            'workflow_step': True,
            **kwargs
        })

    def log_data_processing(self, operation: str, record_count: int, **kwargs):
        """记录数据处理"""
        self.logger.info(f"Data Processing: {operation}", extra={
            'operation': operation,
            'record_count': record_count,
            'data_processing': True,
            **kwargs
        })


def setup_logging(config: Dict[str, Any], log_dir: Optional[str] = None) -> Dict[str, logging.Logger]:
    """
    设置完整的日志系统

    Args:
        config: 日志配置
        log_dir: 日志目录

    Returns:
        配置好的日志记录器字典
    """
    # 设置日志目录
    if log_dir is None:
        log_dir = config.get('log_dir', './logs')

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 获取配置参数
    log_level = config.get('log_level', 'INFO')
    enable_file_logging = config.get('enable_file_logging', True)
    enable_json_logging = config.get('enable_json_logging', False)
    max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
    backup_count = config.get('backup_count', 5)

    # 清除现有的处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 设置根日志记录器
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if enable_json_logging:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    loggers = {}

    if enable_file_logging:
        # 主日志文件
        main_file_handler = logging.handlers.RotatingFileHandler(
            log_path / 'traffic_prediction.log',
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        main_file_handler.setLevel(getattr(logging, log_level.upper()))

        if enable_json_logging:
            main_file_formatter = JSONFormatter()
        else:
            main_file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        main_file_handler.setFormatter(main_file_formatter)

        # 为不同组件创建专门的日志记录器
        logger_configs = {
            'system': 'system.log',
            'performance': 'performance.log',
            'agents': 'agents.log',
            'llm': 'llm.log',
            'data': 'data.log',
            'validation': 'validation.log'
        }

        for logger_name, log_file in logger_configs.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, log_level.upper()))

            # 文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))

            if enable_json_logging:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            file_handler.setFormatter(file_formatter)

            logger.addHandler(file_handler)
            logger.addHandler(main_file_handler)  # 也写入主日志文件

            # 防止重复日志
            logger.propagate = False

            loggers[logger_name] = logger

    return loggers


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)


def log_system_info():
    """记录系统信息"""
    logger = get_logger('system')

    try:
        import psutil
        import platform

        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }

        logger.info("System Information", extra=system_info)

    except ImportError:
        logger.warning("psutil not available, skipping detailed system info")
    except Exception as e:
        logger.error(f"Failed to collect system info: {e}")


class LoggingContext:
    """日志上下文管理器"""

    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}", extra={
            'operation': self.operation,
            'status': 'started',
            **self.context
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"Completed: {self.operation}", extra={
                'operation': self.operation,
                'status': 'completed',
                'duration_seconds': duration,
                **self.context
            })
        else:
            self.logger.error(f"Failed: {self.operation}", extra={
                'operation': self.operation,
                'status': 'failed',
                'duration_seconds': duration,
                'error': str(exc_val),
                **self.context
            })

    def add_context(self, **kwargs):
        """添加上下文信息"""
        self.context.update(kwargs)

    def log_progress(self, message: str, **kwargs):
        """记录进度信息"""
        self.logger.info(f"Progress: {self.operation} - {message}", extra={
            'operation': self.operation,
            'progress_message': message,
            **self.context,
            **kwargs
        })


def monitor_memory():
    """监控内存使用"""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    except ImportError:
        return None


def setup_default_logging(log_level: str = "INFO", log_dir: str = "./logs"):
    """设置默认日志配置"""
    config = {
        'log_level': log_level,
        'log_dir': log_dir,
        'enable_file_logging': True,
        'enable_json_logging': False,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }

    return setup_logging(config, log_dir)