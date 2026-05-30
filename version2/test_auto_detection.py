#!/usr/bin/env python3
"""
测试自动检测ground truth文件功能
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from test_complete_workflow import auto_detect_ground_truth_file

# 测试用例
test_cases = [
    {
        "name": "関越道下行2025年5月3日",
        "state": {
            "route": "関越道",
            "direction": "下",
            "target_year": 2025,
            "target_month": 5,
            "target_day": 3
        },
        "expected": "/home/dizhihuang/graduate/predict_workflow/data/processed_data/関越道_下_2025_05-03.csv"
    },
    {
        "name": "関越道上行2025年5月4日",
        "state": {
            "route": "関越道",
            "direction": "上",
            "target_year": 2025,
            "target_month": 5,
            "target_day": 4
        },
        "expected": "/home/dizhihuang/graduate/predict_workflow/data/processed_data/関越道_上_2025_05-04.csv"
    },
    {
        "name": "東北道下行2025年5月3日",
        "state": {
            "route": "東北道",
            "direction": "下",
            "target_year": 2025,
            "target_month": 5,
            "target_day": 3
        },
        "expected": "/home/dizhihuang/graduate/predict_workflow/data/processed_data/東北道_下_2025_05-03.csv"
    },
    {
        "name": "不存在的日期",
        "state": {
            "route": "関越道",
            "direction": "下",
            "target_year": 2099,
            "target_month": 12,
            "target_day": 31
        },
        "expected": None
    },
    {
        "name": "缺少信息",
        "state": {
            "route": "関越道",
            "direction": "下",
            "target_year": None,
            "target_month": 5,
            "target_day": 3
        },
        "expected": None
    }
]

print("="*70)
print("测试自动检测ground truth文件功能")
print("="*70)

success_count = 0
total_count = len(test_cases)

for i, test in enumerate(test_cases, 1):
    print(f"\n测试 {i}/{total_count}: {test['name']}")
    print("-" * 50)

    result = auto_detect_ground_truth_file(test['state'])
    expected = test['expected']

    if result == expected:
        print(f"✅ 测试通过")
        success_count += 1
    else:
        print(f"❌ 测试失败")
        print(f"   期望: {expected}")
        print(f"   实际: {result}")

print("\n" + "="*70)
print(f"测试完成: {success_count}/{total_count} 通过")
print("="*70)
