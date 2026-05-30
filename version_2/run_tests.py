#!/usr/bin/env python3
"""
Test runner for the multi-agent traffic prediction system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def run_all_tests():
    """运行所有测试"""
    # 发现并加载测试
    loader = unittest.TestLoader()
    start_dir = src_path / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()

def run_specific_test(test_module):
    """运行特定测试模块"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 运行特定测试
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # 运行所有测试
        success = run_all_tests()

    sys.exit(0 if success else 1)