#!/usr/bin/env python3
"""
交通速度热图可视化脚本

用于可视化 PEMS-BAY 和 METR-LA 交通数据集
- 横轴: 传感器编号 (按里程桩排序)
- 纵轴: 时间
- 颜色: 速度 (深色=拥堵, 浅色=畅通)
- 红框: 标记拥堵区域 (速度 < 阈值)

特点:
- 按高速公路和行驶方向分组
- 每条路的每个方向单独一张图

使用方法:
    # 可视化单日数据 (按高速公路+方向分组)
    python traffic_heatmap.py --dataset pems-bay --day 10

    # 批量生成所有天的可视化
    python traffic_heatmap.py --dataset pems-bay --batch

    # 批量生成指定范围
    python traffic_heatmap.py --dataset pems-bay --batch --start-day 0 --end-day 30
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import pickle
import urllib.request
import csv
import io


class TrafficHeatmapVisualizer:
    """交通热图可视化器"""

    # 数据集配置
    DATASETS = {
        'pems-bay': {
            'h5_path': 'pems/pems-bay.h5',
            'h5_key': 'speed/block0_values',
            'time_key': 'speed/axis1',
            'sensor_key': 'speed/axis0',
            'meta_path': 'pems/pems-bay-meta.h5',
            'start_date': datetime(2017, 1, 1),
            'location': 'San Francisco Bay Area',
            'has_meta': True
        },
        'metr-la': {
            'h5_path': 'METR/METR-LA.h5',
            'h5_key': 'df/block0_values',
            'time_key': 'df/axis1',
            'sensor_key': 'df/axis0',
            'start_date': datetime(2012, 3, 1),
            'location': 'Los Angeles',
            'has_meta': False,
            'sensor_locations_url': 'https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv',
            'sensor_locations_cache': 'METR/sensor_locations.csv'
        }
    }

    # 方向名称映射
    DIRECTION_NAMES = {
        'N': 'Northbound',
        'S': 'Southbound',
        'E': 'Eastbound',
        'W': 'Westbound',
        'NW': 'Northwestbound',
        'SE': 'Southeastbound'
    }

    def __init__(self, base_path: str = None):
        """初始化"""
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        self.cmap = self._create_colormap()

    def _create_colormap(self):
        """创建单色深浅的颜色映射"""
        colors = [
            '#08306b',  # 深蓝 (拥堵/低速)
            '#2171b5',
            '#4292c6',
            '#6baed6',
            '#9ecae1',
            '#c6dbef',
            '#deebf7',
            '#f7fbff',  # 浅蓝/白 (畅通/高速)
        ]
        return LinearSegmentedColormap.from_list('traffic', colors)

    def _load_metr_la_highway_info(self, sensor_ids: np.ndarray, config: dict) -> dict:
        """
        加载 METR-LA 传感器位置信息并按高速公路聚类

        METR-LA 传感器位于洛杉矶高速公路网络，主要包括:
        - I-5 (南北向主干道)
        - I-10 (东西向主干道)
        - I-110 (南北向)
        - I-405 (南北向)
        - US-101 (西北-东南向)
        - SR-134 (东西向)
        - SR-2 (南北向)

        Returns:
            dict: {highway_direction_name: {'indices': [...], 'sorted_indices': [...]}}
        """
        print("  加载 METR-LA 传感器位置信息...")

        # 尝试从缓存加载
        cache_path = self.base_path / config['sensor_locations_cache']
        if not cache_path.exists():
            print(f"  下载传感器位置数据...")
            try:
                urllib.request.urlretrieve(config['sensor_locations_url'], cache_path)
                print(f"  已保存到: {cache_path}")
            except Exception as e:
                print(f"  下载失败: {e}")
                return self._fallback_metr_la_grouping(sensor_ids)

        # 读取传感器位置
        sensor_locations = {}
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row['sensor_id'])
                sensor_locations[sid] = {
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude'])
                }

        # 创建传感器ID到数据索引的映射
        data_id_to_idx = {int(sid): idx for idx, sid in enumerate(sensor_ids)}

        # 根据地理位置将传感器分配到高速公路
        # 洛杉矶高速公路的大致走向和位置范围
        highway_definitions = [
            # (名称, 方向判断函数, 位置过滤函数)
            ('I-5', 'NS', lambda lat, lon: lon > -118.28 and lon < -118.20 and lat > 34.05),
            ('I-10', 'EW', lambda lat, lon: lat > 34.04 and lat < 34.08 and lon > -118.30),
            ('I-110', 'NS', lambda lat, lon: lon > -118.29 and lon < -118.25 and lat < 34.08),
            ('I-405', 'NS', lambda lat, lon: lon < -118.38),
            ('US-101', 'NW_SE', lambda lat, lon: lat > 34.10 and lon > -118.38 and lon < -118.20),
            ('SR-134', 'EW', lambda lat, lon: lat > 34.14 and lat < 34.17 and lon > -118.30 and lon < -118.18),
            ('SR-2', 'NS', lambda lat, lon: lon > -118.28 and lon < -118.24 and lat > 34.08 and lat < 34.14),
        ]

        # 按高速公路聚类传感器
        highways = {}
        assigned_sensors = set()

        for highway_name, direction_type, filter_func in highway_definitions:
            sensors_on_highway = []

            for sid, loc in sensor_locations.items():
                if sid in assigned_sensors:
                    continue
                if sid not in data_id_to_idx:
                    continue

                if filter_func(loc['lat'], loc['lon']):
                    sensors_on_highway.append({
                        'sensor_id': sid,
                        'data_idx': data_id_to_idx[sid],
                        'lat': loc['lat'],
                        'lon': loc['lon']
                    })
                    assigned_sensors.add(sid)

            if len(sensors_on_highway) < 3:
                continue

            # 根据高速公路走向分组（按方向）
            if direction_type == 'NS':
                # 南北向：按纬度排序，分为北向和南向
                sensors_on_highway.sort(key=lambda x: x['lat'])
                mid = len(sensors_on_highway) // 2

                # 北向 (纬度递增)
                north_sensors = sensors_on_highway[:mid]
                if len(north_sensors) >= 2:
                    name_n = f"{highway_name}_N"
                    highways[name_n] = {
                        'fwy': highway_name,
                        'direction': 'N',
                        'indices': [s['data_idx'] for s in north_sensors],
                        'sorted_indices': [s['data_idx'] for s in north_sensors],
                        'sorted_abs_pm': [s['lat'] for s in north_sensors],
                        'sensor_ids': [s['sensor_id'] for s in north_sensors]
                    }

                # 南向 (纬度递减)
                south_sensors = sensors_on_highway[mid:]
                if len(south_sensors) >= 2:
                    name_s = f"{highway_name}_S"
                    highways[name_s] = {
                        'fwy': highway_name,
                        'direction': 'S',
                        'indices': [s['data_idx'] for s in south_sensors],
                        'sorted_indices': [s['data_idx'] for s in south_sensors],
                        'sorted_abs_pm': [s['lat'] for s in south_sensors],
                        'sensor_ids': [s['sensor_id'] for s in south_sensors]
                    }

            elif direction_type == 'EW':
                # 东西向：按经度排序
                sensors_on_highway.sort(key=lambda x: x['lon'])
                mid = len(sensors_on_highway) // 2

                # 东向 (经度递增)
                east_sensors = sensors_on_highway[:mid]
                if len(east_sensors) >= 2:
                    name_e = f"{highway_name}_E"
                    highways[name_e] = {
                        'fwy': highway_name,
                        'direction': 'E',
                        'indices': [s['data_idx'] for s in east_sensors],
                        'sorted_indices': [s['data_idx'] for s in east_sensors],
                        'sorted_abs_pm': [abs(s['lon']) for s in east_sensors],
                        'sensor_ids': [s['sensor_id'] for s in east_sensors]
                    }

                # 西向 (经度递减)
                west_sensors = sensors_on_highway[mid:]
                if len(west_sensors) >= 2:
                    name_w = f"{highway_name}_W"
                    highways[name_w] = {
                        'fwy': highway_name,
                        'direction': 'W',
                        'indices': [s['data_idx'] for s in west_sensors],
                        'sorted_indices': [s['data_idx'] for s in west_sensors],
                        'sorted_abs_pm': [abs(s['lon']) for s in west_sensors],
                        'sensor_ids': [s['sensor_id'] for s in west_sensors]
                    }

            else:  # NW_SE (如 US-101)
                # 对角线方向：按纬度+经度的组合排序
                sensors_on_highway.sort(key=lambda x: x['lat'] + x['lon'])
                mid = len(sensors_on_highway) // 2

                # 西北向
                nw_sensors = sensors_on_highway[mid:]
                if len(nw_sensors) >= 2:
                    name_nw = f"{highway_name}_NW"
                    highways[name_nw] = {
                        'fwy': highway_name,
                        'direction': 'NW',
                        'indices': [s['data_idx'] for s in nw_sensors],
                        'sorted_indices': [s['data_idx'] for s in nw_sensors],
                        'sorted_abs_pm': [s['lat'] for s in nw_sensors],
                        'sensor_ids': [s['sensor_id'] for s in nw_sensors]
                    }

                # 东南向
                se_sensors = sensors_on_highway[:mid]
                if len(se_sensors) >= 2:
                    name_se = f"{highway_name}_SE"
                    highways[name_se] = {
                        'fwy': highway_name,
                        'direction': 'SE',
                        'indices': [s['data_idx'] for s in se_sensors],
                        'sorted_indices': [s['data_idx'] for s in se_sensors],
                        'sorted_abs_pm': [s['lat'] for s in se_sensors],
                        'sensor_ids': [s['sensor_id'] for s in se_sensors]
                    }

        # 处理未分配的传感器 - 放入 "Other" 组
        unassigned = []
        for sid, loc in sensor_locations.items():
            if sid not in assigned_sensors and sid in data_id_to_idx:
                unassigned.append({
                    'sensor_id': sid,
                    'data_idx': data_id_to_idx[sid],
                    'lat': loc['lat'],
                    'lon': loc['lon']
                })

        if len(unassigned) >= 5:
            unassigned.sort(key=lambda x: x['lon'])
            highways['Other'] = {
                'fwy': 'Other',
                'direction': '',
                'indices': [s['data_idx'] for s in unassigned],
                'sorted_indices': [s['data_idx'] for s in unassigned],
                'sorted_abs_pm': [abs(s['lon']) for s in unassigned],
                'sensor_ids': [s['sensor_id'] for s in unassigned]
            }

        # 打印统计
        total_assigned = sum(len(info['indices']) for info in highways.values())
        print(f"  METR-LA 高速公路分组 (共 {total_assigned}/{len(sensor_ids)} 传感器):")
        for name, info in sorted(highways.items()):
            dir_full = self.DIRECTION_NAMES.get(info['direction'], info['direction'])
            print(f"    {info['fwy']} {dir_full}: {len(info['indices'])} sensors")

        return highways

    def _fallback_metr_la_grouping(self, sensor_ids: np.ndarray) -> dict:
        """METR-LA 备用分组方法（当无法获取位置数据时）"""
        print(f"  使用备用分组方法...")
        n_sensors = len(sensor_ids)
        n_segments = 4
        segment_size = n_sensors // n_segments

        highways = {}
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else n_sensors
            name = f"Segment_{i+1}"
            indices = list(range(start, end))
            highways[name] = {
                'fwy': f'Segment {i+1}',
                'direction': '',
                'indices': indices,
                'sorted_indices': indices,
                'sorted_abs_pm': list(range(len(indices)))
            }
        return highways

    def load_data(self, dataset: str) -> tuple:
        """加载数据集"""
        if dataset not in self.DATASETS:
            raise ValueError(f"未知数据集: {dataset}, 可选: {list(self.DATASETS.keys())}")

        config = self.DATASETS[dataset]
        h5_path = self.base_path / config['h5_path']

        print(f"加载数据: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            data = f[config['h5_key']][:]
            timestamps = f[config['time_key']][:]
            sensor_ids = f[config['sensor_key']][:]

        total_days = data.shape[0] // 288
        print(f"  数据形状: {data.shape} (时间步 x 传感器)")
        print(f"  时间范围: {config['start_date'].strftime('%Y-%m-%d')} 起, 共 {total_days} 天")

        return data, timestamps, sensor_ids, config

    def load_highway_info(self, dataset: str, sensor_ids: np.ndarray) -> dict:
        """
        加载高速公路信息，按高速公路和方向分组

        Returns:
            dict: {highway_direction_name: {'indices': [...], 'abs_pm': [...], 'sorted_indices': [...]}}
        """
        config = self.DATASETS[dataset]

        if not config.get('has_meta', False):
            # METR-LA: 使用地理位置信息进行高速公路聚类
            if dataset == 'metr-la':
                return self._load_metr_la_highway_info(sensor_ids, config)

            # 其他没有元数据的数据集，按固定数量分段
            print(f"  {dataset} 没有元数据，按传感器数量均匀分段")
            n_sensors = len(sensor_ids)
            n_segments = 3 if n_sensors < 250 else 4
            segment_size = n_sensors // n_segments

            highways = {}
            for i in range(n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < n_segments - 1 else n_sensors
                name = f"Segment_{i+1}"
                indices = list(range(start, end))
                highways[name] = {
                    'indices': indices,
                    'sorted_indices': indices,
                    'sorted_abs_pm': list(range(len(indices)))
                }
            return highways

        # 加载 PEMS-BAY 元数据
        meta_path = self.base_path / config['meta_path']
        print(f"  加载元数据: {meta_path}")

        with h5py.File(meta_path, 'r') as f:
            meta = f['meta']
            meta_sensor_ids = meta['axis1'][:]
            block0 = meta['block0_values'][:]  # City, Abs_PM, Lat, Lon, Length
            block1 = meta['block1_values'][:]  # Fwy, District, County, Lanes, User_ID_4
            block2_raw = meta['block2_values'][0]

        # 解析 block2 获取方向信息
        block2_data = pickle.loads(block2_raw.tobytes())
        directions = block2_data[:, 0]  # 第一列是 Dir (N/S/E/W)

        # 创建传感器ID到数据索引的映射
        data_id_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}

        # 按高速公路和方向分组
        highways = {}
        fwy_numbers = block1[:, 0]
        abs_pm_values = block0[:, 1]

        for meta_idx in range(len(meta_sensor_ids)):
            meta_sensor_id = meta_sensor_ids[meta_idx]

            # 检查这个传感器是否在数据中
            if meta_sensor_id not in data_id_to_idx:
                continue

            data_idx = data_id_to_idx[meta_sensor_id]
            fwy = int(fwy_numbers[meta_idx])
            direction = directions[meta_idx]
            abs_pm = abs_pm_values[meta_idx]

            # 创建唯一的名称: Fwy_85_N, Fwy_85_S 等
            highway_name = f"Fwy_{fwy}_{direction}"

            if highway_name not in highways:
                highways[highway_name] = {
                    'fwy': fwy,
                    'direction': direction,
                    'indices': [],
                    'abs_pm': [],
                    'sensor_ids': []
                }

            highways[highway_name]['indices'].append(data_idx)
            highways[highway_name]['abs_pm'].append(abs_pm)
            highways[highway_name]['sensor_ids'].append(meta_sensor_id)

        # 对每条高速公路按里程桩排序
        for highway_name, info in highways.items():
            sorted_order = np.argsort(info['abs_pm'])
            info['sorted_indices'] = [info['indices'][i] for i in sorted_order]
            info['sorted_abs_pm'] = [info['abs_pm'][i] for i in sorted_order]
            info['sorted_sensor_ids'] = [info['sensor_ids'][i] for i in sorted_order]

        # 打印统计
        print(f"  高速公路+方向分组:")
        for name, info in sorted(highways.items(), key=lambda x: (-x[1]['fwy'], x[1]['direction'])):
            dir_full = self.DIRECTION_NAMES.get(info['direction'], info['direction'])
            print(f"    Fwy-{info['fwy']} {dir_full}: {len(info['indices'])} sensors")

        return highways

    def plot_highway_heatmap(self,
                             data_kmh: np.ndarray,
                             time_labels: list,
                             highway_name: str,
                             highway_info: dict,
                             title: str,
                             save_path: str,
                             congestion_threshold_kmh: float = 40,
                             show_congestion_boxes: bool = True,
                             figsize: tuple = (18, 10),
                             dpi: int = 150):
        """绘制单条高速公路单方向的热图"""
        sorted_indices = highway_info['sorted_indices']
        sorted_abs_pm = highway_info['sorted_abs_pm']

        # 提取该高速公路的数据（按里程排序）
        data_highway = data_kmh[:, sorted_indices]

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热图
        im = ax.imshow(data_highway, aspect='auto', cmap=self.cmap,
                       vmin=0, vmax=120, origin='upper')

        # 标记拥堵区域
        if show_congestion_boxes:
            congestion_mask = (data_highway < congestion_threshold_kmh) & (data_highway > 0)
            for t in range(data_highway.shape[0]):
                for s in range(data_highway.shape[1]):
                    if congestion_mask[t, s]:
                        rect = patches.Rectangle(
                            (s - 0.5, t - 0.5), 1, 1,
                            linewidth=0.5,
                            edgecolor='red',
                            facecolor='none',
                            alpha=0.9
                        )
                        ax.add_patch(rect)

        # 设置坐标轴
        n_sensors = len(sorted_indices)

        # 横轴：显示里程桩
        ax.set_xlabel(f'Post Mile (Total {n_sensors} sensors)', fontsize=12)
        sensor_tick_step = max(1, n_sensors // 15)
        sensor_ticks = list(range(0, n_sensors, sensor_tick_step))
        ax.set_xticks(sensor_ticks)
        ax.set_xticklabels([f'{sorted_abs_pm[i]:.1f}' for i in sensor_ticks], fontsize=9)

        # 纵轴：时间
        ax.set_ylabel('Time', fontsize=12)
        n_time = data_highway.shape[0]
        time_tick_step = max(1, n_time // 12)
        time_ticks = list(range(0, n_time, time_tick_step))
        ax.set_yticks(time_ticks)
        ax.set_yticklabels([time_labels[i] for i in time_ticks], fontsize=9)

        ax.set_title(f'{title}\n(Red boxes: Speed < {congestion_threshold_kmh} km/h)', fontsize=14)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Speed (km/h)', fontsize=11)

        # 统计信息
        valid_data = data_highway[data_highway > 0]
        if valid_data.size > 0:
            congestion_ratio = np.sum(valid_data < congestion_threshold_kmh) / valid_data.size * 100
        else:
            congestion_ratio = 0
        missing_ratio = np.sum(data_highway == 0) / data_highway.size * 100

        stats_text = f'Congestion: {congestion_ratio:.2f}% | Missing: {missing_ratio:.2f}%'
        ax.text(0.5, -0.08, stats_text, transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return congestion_ratio

    def visualize_day(self, dataset: str, day_index: int = 0,
                      output_dir: str = None, verbose: bool = True, **kwargs):
        """可视化单日数据（每条高速公路每个方向单独一张图）"""
        data, timestamps, sensor_ids, config = self.load_data(dataset)
        highways = self.load_highway_info(dataset, sensor_ids)

        # 每天288个时间步
        steps_per_day = 288
        start_idx = day_index * steps_per_day
        end_idx = start_idx + steps_per_day

        max_day = data.shape[0] // steps_per_day - 1
        if end_idx > data.shape[0]:
            raise ValueError(f"天数索引超出范围: {day_index}, 最大: {max_day}")

        data_day = data[start_idx:end_idx, :]
        data_kmh = data_day * 1.60934

        # 生成时间标签
        date = config['start_date'] + timedelta(days=day_index)
        time_labels = [(date + timedelta(minutes=5 * i)).strftime('%H:%M')
                       for i in range(steps_per_day)]

        # 保存路径
        if output_dir is None:
            output_dir = self.base_path / 'output'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = date.strftime('%Y%m%d')
        weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][date.weekday()]

        # 为每条高速公路每个方向生成图片
        saved_paths = []
        congestion_ratios = []

        # 按高速公路编号和方向排序
        sorted_highways = sorted(highways.items(),
                                 key=lambda x: (x[1].get('fwy', 0), x[1].get('direction', '')))

        for highway_name, info in sorted_highways:
            save_path = output_dir / f'heatmap_{dataset}_{date_str}_{highway_name}.png'

            # 构建标题
            fwy = info.get('fwy', '')
            direction = info.get('direction', '')
            dir_full = self.DIRECTION_NAMES.get(direction, direction)

            # 根据数据集格式化高速公路名称
            if dataset == 'metr-la':
                fwy_display = fwy  # METR-LA 已经是 "I-5", "US-101" 等格式
            else:
                fwy_display = f'Fwy-{fwy}'  # PEMS-BAY 使用数字，添加前缀

            title = (f'{dataset.upper()} - {fwy_display} {dir_full} ({config["location"]})\n'
                     f'{date.strftime("%Y-%m-%d")} ({weekday})')

            congestion_ratio = self.plot_highway_heatmap(
                data_kmh, time_labels, highway_name, info,
                title, str(save_path), **kwargs
            )

            saved_paths.append(save_path)
            congestion_ratios.append(congestion_ratio)

            if verbose:
                print(f"已保存: {save_path} (拥堵率: {congestion_ratio:.2f}%)")

        return saved_paths, congestion_ratios

    def visualize_batch(self, dataset: str, start_day: int = 0, end_day: int = None,
                        output_dir: str = None, **kwargs):
        """批量生成每天的可视化"""
        data, timestamps, sensor_ids, config = self.load_data(dataset)
        highways = self.load_highway_info(dataset, sensor_ids)

        total_days = data.shape[0] // 288
        num_highways = len(highways)

        if end_day is None:
            end_day = total_days
        end_day = min(end_day, total_days)

        total_images = (end_day - start_day) * num_highways
        print(f"\n批量生成: 第 {start_day} 天 到 第 {end_day - 1} 天")
        print(f"每天 {num_highways} 张图 (高速公路x方向), 共 {total_images} 张图")

        if output_dir is None:
            output_dir = self.base_path / 'output'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sorted_highways = sorted(highways.items(),
                                 key=lambda x: (x[1].get('fwy', 0), x[1].get('direction', '')))

        results = []
        pbar = tqdm(total=total_images, desc="生成热图")

        for day_idx in range(start_day, end_day):
            try:
                steps_per_day = 288
                start_idx = day_idx * steps_per_day
                end_idx = start_idx + steps_per_day
                data_day = data[start_idx:end_idx, :]
                data_kmh = data_day * 1.60934

                date = config['start_date'] + timedelta(days=day_idx)
                time_labels = [(date + timedelta(minutes=5 * i)).strftime('%H:%M')
                               for i in range(steps_per_day)]

                date_str = date.strftime('%Y%m%d')
                weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][date.weekday()]

                for highway_name, info in sorted_highways:
                    save_path = output_dir / f'heatmap_{dataset}_{date_str}_{highway_name}.png'

                    fwy = info.get('fwy', '')
                    direction = info.get('direction', '')
                    dir_full = self.DIRECTION_NAMES.get(direction, direction)

                    # 根据数据集格式化高速公路名称
                    if dataset == 'metr-la':
                        fwy_display = fwy  # METR-LA 已经是 "I-5", "US-101" 等格式
                    else:
                        fwy_display = f'Fwy-{fwy}'  # PEMS-BAY 使用数字，添加前缀

                    title = (f'{dataset.upper()} - {fwy_display} {dir_full} ({config["location"]})\n'
                             f'{date.strftime("%Y-%m-%d")} ({weekday})')

                    congestion_ratio = self.plot_highway_heatmap(
                        data_kmh, time_labels, highway_name, info,
                        title, str(save_path), **kwargs
                    )

                    results.append((day_idx, date_str, highway_name, congestion_ratio))
                    pbar.update(1)

            except Exception as e:
                print(f"\n警告: 第 {day_idx} 天生成失败: {e}")

        pbar.close()
        print(f"\n完成! 共生成 {len(results)} 张图片")

        if results:
            congestion_rates = [r[3] for r in results]
            print(f"拥堵率统计: 最小={min(congestion_rates):.2f}%, "
                  f"最大={max(congestion_rates):.2f}%, "
                  f"平均={np.mean(congestion_rates):.2f}%")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='交通速度热图可视化工具 (按高速公路+方向分组)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化 PEMS-BAY 第10天的数据 (每条高速公路每个方向一张图)
  python traffic_heatmap.py --dataset pems-bay --day 10

  # 批量生成所有天的可视化
  python traffic_heatmap.py --dataset pems-bay --batch

  # 批量生成指定范围 (第0天到第29天)
  python traffic_heatmap.py --dataset pems-bay --batch --start-day 0 --end-day 30

  # 修改拥堵阈值为 30 km/h
  python traffic_heatmap.py --dataset pems-bay --day 10 --threshold 30

  # 不显示拥堵红框
  python traffic_heatmap.py --dataset pems-bay --day 10 --no-boxes
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pems-bay', 'metr-la'],
                        help='数据集名称')
    parser.add_argument('--day', type=int, default=None,
                        help='天数索引 (从0开始)')
    parser.add_argument('--batch', action='store_true',
                        help='批量生成所有天的可视化')
    parser.add_argument('--start-day', type=int, default=0,
                        help='起始天数索引 (用于批量模式)')
    parser.add_argument('--end-day', type=int, default=None,
                        help='结束天数索引 (用于批量模式, 不包含)')
    parser.add_argument('--threshold', type=float, default=40,
                        help='拥堵阈值 (km/h), 默认 40')
    parser.add_argument('--no-boxes', action='store_true',
                        help='不显示拥堵红框')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--base-path', type=str, default=None,
                        help='数据集根目录')
    parser.add_argument('--dpi', type=int, default=150,
                        help='图片分辨率')

    args = parser.parse_args()

    visualizer = TrafficHeatmapVisualizer(base_path=args.base_path)

    kwargs = {
        'congestion_threshold_kmh': args.threshold,
        'show_congestion_boxes': not args.no_boxes,
        'dpi': args.dpi,
    }

    if args.batch:
        visualizer.visualize_batch(
            args.dataset,
            start_day=args.start_day,
            end_day=args.end_day,
            output_dir=args.output_dir,
            **kwargs
        )
    elif args.day is not None:
        visualizer.visualize_day(args.dataset, day_index=args.day,
                                 output_dir=args.output_dir, **kwargs)
    else:
        visualizer.visualize_day(args.dataset, day_index=0,
                                 output_dir=args.output_dir, **kwargs)


if __name__ == '__main__':
    main()
