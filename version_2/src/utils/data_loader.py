"""
Data loading utilities for the multi-agent traffic prediction system.
"""

import pandas as pd
import os
from typing import List, Dict, Optional
from pathlib import Path
import glob

from ..core.exceptions import FileNotFoundError, DataValidationError


class DataLoader:
    """数据加载工具类"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

    def find_files_by_pattern(self, pattern: str) -> List[str]:
        """根据模式查找文件"""
        files = list(self.data_path.glob(pattern))
        return [str(f) for f in files]

    def load_csv_files(self, file_patterns: List[str]) -> pd.DataFrame:
        """加载多个CSV文件并合并"""
        all_data = []

        for pattern in file_patterns:
            files = self.find_files_by_pattern(pattern)
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = os.path.basename(file_path)
                    all_data.append(df)
                except Exception as e:
                    raise DataValidationError(f"加载文件失败 {file_path}: {e}")

        if not all_data:
            raise FileNotFoundError(f"未找到匹配的文件: {file_patterns}")

        return pd.concat(all_data, ignore_index=True)

    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """验证CSV文件结构"""
        required_columns = [
            'date', '上下', '原因', '道路番号', '発生時刻',
            'ピーク時刻', 'ピーク長', '発生Ｋｐ', '発生時渋滞長', '渋滞時間'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"CSV文件缺少必要列: {missing_columns}")

        return True


def load_sample_data() -> pd.DataFrame:
    """加载示例数据用于测试"""
    sample_data = {
        'date': ['2023-05-15', '2023-05-15', '2023-05-15'],
        '上下': ['上', '上', '下'],
        '原因': ['交通集中', '交通集中', '事故'],
        '道路番号': ['東北道', '東北道', '東北道'],
        '発生時刻': ['17:30:00', '18:00:00', '17:45:00'],
        'ピーク時刻': ['18:15:00', '18:45:00', '18:30:00'],
        'ピーク長': [8.5, 12.3, 6.2],
        '発生Ｋｐ': [25.4, 30.1, 28.7],
        '発生時渋滞長': [3.2, 4.1, 2.8],
        '渋滞時間': [180, 240, 150]
    }
    return pd.DataFrame(sample_data)