#!/usr/bin/env python3
"""
Example usage of the Multi-Agent Traffic Prediction System.
This script demonstrates how to use the system for traffic congestion prediction.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.traffic_prediction_system import TrafficPredictionSystem, create_system
from src.core.data_structures import SystemConfig, TriangleConfig, LLMConfig
from src.utils.logging_config import setup_default_logging, LoggingContext, get_logger
from src.utils.data_loader import load_sample_data
import tempfile
import pandas as pd


def create_sample_data_files(data_dir: Path):
    """创建示例数据文件用于演示"""
    print("创建示例数据文件...")

    # 确保数据目录存在
    data_dir.mkdir(parents=True, exist_ok=True)

    # 获取示例数据
    sample_data = load_sample_data()

    # 创建多个年份的数据文件
    for year in [2021, 2022, 2023]:
        for direction in ['上', '下']:
            filename = f'東北道_{direction}_{year}_05-15.csv'
            file_path = data_dir / filename
            sample_data.to_csv(file_path, index=False)
            print(f"  创建文件: {filename}")

    print(f"示例数据文件已创建在: {data_dir}")


def example_basic_usage():
    """基础使用示例"""
    print("\n" + "="*60)
    print("基础使用示例")
    print("="*60)

    # 设置临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'
        output_dir = Path(temp_dir) / 'output'

        # 创建示例数据
        create_sample_data_files(data_dir)

        # 创建系统配置
        config = SystemConfig(
            data_path=str(data_dir),
            output_path=str(output_dir),
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="demo-key")  # 使用演示密钥
        )

        # 创建系统实例（使用模拟LLM）
        print("\n1. 初始化交通预测系统...")
        system = TrafficPredictionSystem(config=config, use_mock_llm=True)

        # 检查系统状态
        print("\n2. 检查系统状态...")
        status = system.get_system_status()
        print(f"系统状态: {status['system_status']}")

        # 验证配置
        print("\n3. 验证系统配置...")
        validation_result = system.validate_configuration()
        if validation_result['valid']:
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败:")
            for error in validation_result['errors']:
                print(f"  - {error}")

        # 执行预测
        print("\n4. 执行交通拥堵预测...")
        request = {
            'target_date': '2024-05-15',
            'road_name': '東北道',
            'direction': '上'
        }

        try:
            result = system.predict(request)
            print("✅ 预测执行成功!")

            # 显示预测摘要
            summary = result['summary']
            print(f"\n预测摘要:")
            print(f"  预测峰值时间: {summary['prediction_date']}")
            print(f"  预测峰值位置: {summary['prediction_location']}")
            print(f"  影响时间范围: {summary['time_range']}")
            print(f"  影响位置范围: {summary['location_range']}")
            print(f"  严重程度: {summary['severity']}")
            print(f"  置信度: {summary['confidence']:.2f}")
            print(f"  验证状态: {summary['validation_status']}")
            print(f"  建议: {summary['recommendation']}")

        except Exception as e:
            print(f"❌ 预测执行失败: {e}")


def example_quick_predict():
    """快速预测示例"""
    print("\n" + "="*60)
    print("快速预测示例")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'

        # 创建示例数据
        create_sample_data_files(data_dir)

        # 使用便捷函数创建系统
        print("\n1. 使用配置文件创建系统...")

        # 创建配置文件
        config_file = Path(temp_dir) / 'config.yaml'
        config_content = f"""
data_path: {data_dir}
output_path: {temp_dir}/output
triangle_config:
  min_duration: 10
  min_length: 0.5
llm_config:
  model: gpt-4-vision-preview
  api_key: demo-key
log_level: INFO
"""
        config_file.write_text(config_content)

        system = create_system(config_path=str(config_file), use_mock_llm=True)

        # 快速预测
        print("\n2. 执行快速预测...")
        result = system.quick_predict('2024-05-15', '東北道', '上')

        print("快速预测结果:")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  验证通过: {result['validation_passed']}")
        print(f"  建议: {result['recommendation']}")


def example_with_logging():
    """带日志监控的示例"""
    print("\n" + "="*60)
    print("带日志监控的示例")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'
        log_dir = Path(temp_dir) / 'logs'

        # 设置日志
        print("\n1. 设置增强日志系统...")
        setup_default_logging(log_level="INFO", log_dir=str(log_dir))

        logger = get_logger('demo')
        logger.info("演示开始", extra={'demo_type': 'logging_example'})

        # 创建示例数据
        create_sample_data_files(data_dir)

        config = SystemConfig(
            data_path=str(data_dir),
            output_path=str(temp_dir),
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="demo-key")
        )

        # 使用日志上下文管理器
        with LoggingContext(logger, "traffic_prediction_demo") as ctx:
            ctx.add_context(target_date='2024-05-15', road_name='東北道')

            print("\n2. 创建系统并执行预测...")
            system = TrafficPredictionSystem(config=config, use_mock_llm=True)

            ctx.log_progress("系统初始化完成")

            result = system.quick_predict('2024-05-15', '東北道', '上')

            ctx.log_progress("预测执行完成", confidence=result['confidence'])

        logger.info("演示完成", extra={'demo_type': 'logging_example'})

        print(f"\n日志文件已保存在: {log_dir}")
        for log_file in log_dir.glob('*.log'):
            print(f"  - {log_file.name}")


def example_error_handling():
    """错误处理示例"""
    print("\n" + "="*60)
    print("错误处理示例")
    print("="*60)

    # 演示各种错误情况
    print("\n1. 测试无效配置...")
    try:
        invalid_config = SystemConfig(
            data_path="/nonexistent/path",
            output_path="/tmp/output",
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="invalid-key")
        )
        system = TrafficPredictionSystem(config=invalid_config, use_mock_llm=True)
        system.predict({
            'target_date': '2024-05-15',
            'road_name': '不存在的道路'
        })
    except Exception as e:
        print(f"✅ 正确捕获配置错误: {type(e).__name__}: {e}")

    print("\n2. 测试无效请求...")
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        config = SystemConfig(
            data_path=str(data_dir),
            output_path=str(temp_dir),
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="demo-key")
        )
        system = TrafficPredictionSystem(config=config, use_mock_llm=True)

        try:
            system.predict({
                'target_date': 'invalid-date',
                'road_name': '東北道'
            })
        except Exception as e:
            print(f"✅ 正确捕获请求错误: {type(e).__name__}: {e}")


def example_performance_monitoring():
    """性能监控示例"""
    print("\n" + "="*60)
    print("性能监控示例")
    print("="*60)

    from src.utils.logging_config import PerformanceLogger, monitor_memory
    import time

    # 设置性能日志
    perf_logger = PerformanceLogger()

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'
        create_sample_data_files(data_dir)

        config = SystemConfig(
            data_path=str(data_dir),
            output_path=str(temp_dir),
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="demo-key")
        )

        # 监控系统初始化
        start_time = time.time()
        system = TrafficPredictionSystem(config=config, use_mock_llm=True)
        init_time = time.time() - start_time

        perf_logger.log_execution_time("system_initialization", init_time)

        # 监控内存使用
        memory_info = monitor_memory()
        if memory_info:
            perf_logger.log_memory_usage("after_initialization", memory_info['rss_mb'])
            print(f"内存使用: {memory_info['rss_mb']:.1f} MB")

        # 监控预测执行
        start_time = time.time()
        result = system.quick_predict('2024-05-15', '東北道', '上')
        prediction_time = time.time() - start_time

        perf_logger.log_execution_time("prediction_execution", prediction_time,
                                     confidence=result['confidence'])

        print(f"系统初始化时间: {init_time:.2f} 秒")
        print(f"预测执行时间: {prediction_time:.2f} 秒")


def main():
    """主函数"""
    print("多Agent交通预测系统 - 使用示例")
    print("="*60)

    try:
        # 基础使用示例
        example_basic_usage()

        # 快速预测示例
        example_quick_predict()

        # 日志监控示例
        example_with_logging()

        # 错误处理示例
        example_error_handling()

        # 性能监控示例
        example_performance_monitoring()

        print("\n" + "="*60)
        print("所有示例执行完成! ✅")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()