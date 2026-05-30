"""
专家预测数据处理脚本
将 data/prediction_data/ 中的专家预测Excel文件转换为与 processed_data/ 相同的CSV格式

输入: data/prediction_data/*.xlsx (专家预测数据, 107列)
输出: data/processed_prediction_data/{road}_{direction}_{year}_{month}-{day}.csv (10列标准格式)
"""

import pandas as pd
import numpy as np
import os
import re

# ========== 路线名映射: 全称 → 简称 ==========
ROAD_NAME_MAP = {
    '東北自動車道': '東北道',
    '関越自動車道': '関越道',
    '常磐自動車道': '常磐道',
    '東京外環自動車道': '外環道',
    '東関東自動車道': '東関東道',
    '館山自動車道': '館山道',
    '北関東自動車道': '北関東道',
    '上信越自動車道': '上信越道',
    '千葉東金道路': '千葉東金道路',
    '富津館山道路': '富津館山道路',
    '京葉道路': '京葉道路',
    '第三京浜道路': '第三京浜',
    '横浜新道': '横浜新道',
    '横浜横須賀道路': '横横道路',
    '東京湾アクアライン': 'アクアライン',
    '首都圏中央連絡自動車道': '圏央道',
    '仙台南部道路': '仙台南部道路',
    '山形自動車道': '山形道',
    '北陸自動車道': '北陸道',
    '道央自動車道': '道央道',
    '札樽自動車道': '札樽道',
    '道東自動車道': '道東道',
}

# ========== 方向映射 ==========
DIRECTION_MAP = {
    '上り': '上',
    '下り': '下',
    '内回り': '内',
    '外回り': '外',
    '西行き': '西',
}


def parse_time_range(time_str):
    """
    解析时间范围字符串，返回 (発生時刻, 渋滞時間)

    Examples:
        "9:00～13:00"     → ("09:00:00", 240)
        "15:00～翌0:00"   → ("15:00:00", 540)
        "22:00～翌2:00"   → ("22:00:00", 240)
    """
    if pd.isna(time_str):
        return None, None

    time_str = str(time_str).strip()

    # 匹配格式: H:MM～(翌?)H:MM
    match = re.match(r'(\d{1,2}:\d{2})～(翌?)(\d{1,2}:\d{2})', time_str)
    if not match:
        return None, None

    start_str = match.group(1)
    next_day = match.group(2) == '翌'
    end_str = match.group(3)

    start_h, start_m = map(int, start_str.split(':'))
    end_h, end_m = map(int, end_str.split(':'))

    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    if next_day:
        end_min += 1440  # +24 hours

    duration = end_min - start_min

    return f"{start_h:02d}:{start_m:02d}:00", duration


def parse_peak_time(peak_val):
    """
    解析峰值时间，返回 HH:MM:SS 格式

    Examples:
        "10時"   → "10:00:00"
        "翌0時"  → "00:00:00"
        10       → "10:00:00"
        17.0     → "17:00:00"
    """
    if pd.isna(peak_val):
        return None

    peak_str = str(peak_val).strip()

    # 处理整数/浮点数 (如 10, 17.0)
    try:
        h = int(float(peak_str))
        if 0 <= h <= 23:
            return f"{h:02d}:00:00"
    except ValueError:
        pass

    # 处理 "翌X時" 格式
    match = re.match(r'翌?(\d{1,2})時', peak_str)
    if match:
        h = int(match.group(1))
        return f"{h:02d}:00:00"

    return None


# Excel 中小时列的范围: 列 15~38 = 0~23時, 列 39~44 = 翌 0~5時
HOUR_COL_START = 15
HOUR_COL_END = 45  # exclusive


def extract_peak_duration_from_hourly(hourly_row, peak_length):
    """
    从小时拥堵长度序列中提取"峰值持续时间"(分钟)
    = 连续处于峰值长度的小时数 × 60

    Args:
        hourly_row: 一行 Excel 数据的小时部分 (索引 15~44)
        peak_length: 峰值长度 (km)

    Returns:
        峰值持续时间(分钟), 如果没有明显平台期则返回 0
    """
    if peak_length <= 0:
        return 0

    # 容差 0.01: 浮点数比较
    peak_hours = []
    for i, c in enumerate(range(HOUR_COL_START, HOUR_COL_END)):
        if c >= len(hourly_row):
            break
        val = hourly_row.iloc[c] if hasattr(hourly_row, 'iloc') else hourly_row[c]
        if pd.notna(val) and abs(float(val) - peak_length) < 0.01:
            peak_hours.append(i)

    if len(peak_hours) <= 1:
        # 峰值只出现在一个小时 → 三角形(尖顶)
        return 0

    # 连续性检查:找最长连续段
    max_run = 1
    cur_run = 1
    for i in range(1, len(peak_hours)):
        if peak_hours[i] == peak_hours[i-1] + 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    # 平台期 = (连续峰值小时数 - 1) × 60 分钟
    # 例: 连续 3 小时在峰值 → 平台持续 2 小时
    return max(0, (max_run - 1) * 60)


def process_expert_prediction_files(input_dir, output_dir):
    """处理所有专家预测Excel文件"""

    os.makedirs(output_dir, exist_ok=True)

    all_records = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.xlsx'):
            continue
        if fname.startswith('~$'):  # Excel lock file
            continue

        fpath = os.path.join(input_dir, fname)
        print(f"处理文件: {fname}")

        df = pd.read_excel(fpath, header=1)
        # 无 header 读取一份用于访问小时列
        df_raw = pd.read_excel(fpath, header=None, skiprows=3)

        # 清洗列名: 去除换行符
        df.columns = [str(c).replace('\n', '') for c in df.columns]

        # 过滤空行 (标题行残留)
        valid_mask = df['道路名'].notna() & df['月日'].notna()
        df = df[valid_mask]
        df_raw = df_raw[valid_mask.values[:len(df_raw)]] if len(df_raw) >= len(valid_mask) else df_raw

        processed_count = 0
        skipped_count = 0

        for idx, (_, row) in enumerate(df.iterrows()):
            # 路线名转换
            road_full = str(row.get('道路名', '')).strip()
            road_short = ROAD_NAME_MAP.get(road_full, road_full)

            # 方向转换
            direction_full = str(row.get('方向', '')).strip()
            direction = DIRECTION_MAP.get(direction_full, direction_full)
            if direction == 'nan' or not direction:
                skipped_count += 1
                continue

            # 日期
            date_val = row.get('月日')
            if pd.isna(date_val):
                skipped_count += 1
                continue
            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')

            # 时间范围 → 発生時刻 + 渋滞時間
            start_time, duration = parse_time_range(row.get('時間帯'))
            if start_time is None or duration is None:
                skipped_count += 1
                continue

            # 峰值时间
            peak_time = parse_peak_time(row.get('ピーク時間'))
            if peak_time is None:
                skipped_count += 1
                continue

            # 峰值长度 (已经是km单位)
            peak_length = row.get('ピーク時の渋滞長', 0)
            if pd.isna(peak_length):
                peak_length = 0.0
            peak_length = float(peak_length)

            # KP (已经是km单位)
            kp = row.get('KP', 0)
            if pd.isna(kp):
                kp = 0.0
            kp = float(kp)

            # 提取峰值持续时间(分钟) - 从小时级拥堵长度曲线推断平台期
            peak_duration_min = 0
            try:
                if idx < len(df_raw):
                    raw_row = df_raw.iloc[idx]
                    peak_duration_min = extract_peak_duration_from_hourly(raw_row, peak_length)
            except Exception:
                peak_duration_min = 0

            record = {
                'date': date_str,
                '上下': direction,
                '原因': '交通集中',
                '道路番号': road_short,
                '発生時刻': start_time,
                'ピーク時刻': peak_time,
                'ピーク長': peak_length,
                '発生Ｋｐ': kp,
                '発生時渋滞長': 0.0,
                '渋滞時間': int(duration),
                'ピーク持続時間': int(peak_duration_min),
            }
            all_records.append(record)
            processed_count += 1

        print(f"  → 处理: {processed_count} 条, 跳过: {skipped_count} 条")

    if not all_records:
        print("没有数据被处理！")
        return

    df_all = pd.DataFrame(all_records)

    # 列顺序与现有格式一致 + 新增 ピーク持続時間
    column_order = ['date', '上下', '原因', '道路番号', '発生時刻', 'ピーク時刻',
                    'ピーク長', '発生Ｋｐ', '発生時渋滞長', '渋滞時間', 'ピーク持続時間']
    df_all = df_all[column_order]

    # 保存汇总文件
    all_data_path = os.path.join(output_dir, 'all_prediction_data.csv')
    df_all.to_csv(all_data_path, index=False)
    print(f"\n汇总文件已保存: {all_data_path} ({len(df_all)} 条记录)")

    # 按 date/road/direction 分组保存单独CSV
    df_all['date_dt'] = pd.to_datetime(df_all['date'])
    grouped = df_all.groupby(['date_dt', '道路番号', '上下'])

    file_count = 0
    for (date, road, direction), group_df in grouped:
        year = date.year
        month_day = date.strftime('%m-%d')
        filename = f"{road}_{direction}_{year}_{month_day}.csv"
        filepath = os.path.join(output_dir, filename)
        group_df.drop(columns=['date_dt']).to_csv(filepath, index=False)
        file_count += 1

    print(f"已生成 {file_count} 个分日CSV文件 → {output_dir}")

    # 打印统计信息
    print(f"\n===== 统计 =====")
    print(f"总记录数: {len(df_all)}")
    print(f"路线分布:")
    for road, count in df_all['道路番号'].value_counts().items():
        print(f"  {road}: {count}")
    print(f"方向分布:")
    for d, count in df_all['上下'].value_counts().items():
        print(f"  {d}: {count}")
    print(f"日期范围: {df_all['date'].min()} ～ {df_all['date'].max()}")


if __name__ == '__main__':
    # 自动检测工作目录 (兼容本地 macOS 和服务器 Linux)
    _candidates = [
        "/home/dizhihuang/graduate/predict_workflow/data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    ]
    _data_base = next((p for p in _candidates if os.path.isdir(p)), _candidates[-1])
    input_dir = os.path.join(_data_base, "prediction_data")
    output_dir = os.path.join(_data_base, "processed_prediction_data")

    process_expert_prediction_files(input_dir, output_dir)
