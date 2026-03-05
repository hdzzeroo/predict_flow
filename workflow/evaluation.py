"""
交通拥堵预测评估模块
实现三种IoU评估指标：
1. 全局多边形IoU
2. 栅格化F1-Score
3. 分层IoU（空间+时间）
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import pandas as pd
import os


class Evaluator:
    """交通拥堵预测评估器"""

    def __init__(
        self,
        road_type: str = "関越道",
        direction: str = "下",
        time_step_minutes: int = 60
    ):
        """
        初始化评估器

        Args:
            road_type: 道路类型 (如 "関越道")
            direction: 方向 ("上" 或 "下")
            time_step_minutes: 时间栅格大小（分钟），默认 60 表示 1小时
        """
        self.road_type = road_type
        self.direction = direction
        self.time_step = time_step_minutes

        # KP区間を読み込む
        self.kp_intervals = self._load_kp_intervals()
        print(f"✓ 已加载 {len(self.kp_intervals)} 个KP区间")

    def _load_kp_intervals(self) -> List[Tuple[float, float]]:
        """
        从道路信息CSV文件加载KP区间

        Returns:
            KP区间列表 [(kp_start, kp_end), ...]
        """
        # CSVファイルパスを確定（サーバー/ローカルの互換性のために複数のベースディレクトリを試す）
        candidate_dirs = [
            "/home/dizhihuang/graduate/predict_workflow/data",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
        ]

        road_file_map = {
            '関越道': "roadic_kannetsu.csv",
            '東北道': "roadic_touhoku.csv",
        }

        if self.road_type not in road_file_map:
            print(f"⚠️ 未识别的道路类型 '{self.road_type}'，使用関越道")
            road_file = road_file_map['関越道']
        else:
            road_file = road_file_map[self.road_type]

        csv_file = None
        for base_dir in candidate_dirs:
            candidate = os.path.join(base_dir, road_file)
            if os.path.exists(candidate):
                csv_file = candidate
                break

        if csv_file is None:
            print(f"❌ 道路信息文件不存在: {road_file} (searched {candidate_dirs})")
            return []

        try:
            # CSVを読み込む
            df = pd.read_csv(csv_file)

            # 过滤方向
            direction_map = {"上": "up", "下": "down"}
            dir_en = direction_map.get(self.direction, "down")
            df_filtered = df[df['direction'] == dir_en].copy()

            # KP値を抽出してソート
            kp_values = sorted(df_filtered['KP'].unique())

            # KP区間を生成：[kp[i], kp[i+1]]
            kp_intervals = []
            for i in range(len(kp_values) - 1):
                kp_intervals.append((kp_values[i], kp_values[i+1]))

            return kp_intervals

        except Exception as e:
            print(f"❌ 加载道路信息失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def evaluate_all_directions(
        self,
        predictions: Dict[str, List[Dict]],
        ground_truth: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        评估所有方向的预测结果

        Args:
            predictions: 预测热点字典 {"上": [...], "下": [...]}
            ground_truth: 真实拥堵字典 {"上": [...], "下": [...]}

        Returns:
            评估结果字典，包含每个方向和平均指标
        """
        results = {}

        # 评估每个方向
        for direction in predictions.keys():
            if direction not in ground_truth:
                print(f"⚠️ 警告：真实数据中没有'{direction}'方向的数据")
                continue

            print(f"\n{'='*50}")
            print(f"评估方向: {direction}")
            print(f"{'='*50}")

            results[direction] = self.evaluate_single_direction(
                predictions[direction],
                ground_truth[direction],
                direction
            )

        # 计算平均指标
        if len(results) > 0:
            results["average"] = self._calculate_average_metrics(results)

        return results

    def evaluate_single_direction(
        self,
        pred_hotspots: List[Dict],
        gt_triangles: List[Dict],
        direction: str
    ) -> Dict[str, Any]:
        """
        评估单个方向的预测结果

        Args:
            pred_hotspots: 预测的热点列表
            gt_triangles: 真实的拥堵三角形列表
            direction: 方向标识

        Returns:
            包含三种指标的评估结果
        """
        # Shapelyポリゴンに変換
        pred_polygons = self._convert_to_polygons(pred_hotspots, "predictions")
        gt_polygons = self._convert_to_polygons(gt_triangles, "ground_truth")

        if not pred_polygons or not gt_polygons:
            print(f"⚠️ 警告：预测或真实数据为空，跳过评估")
            return {
                "polygon_iou": 0.0,
                "grid_metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "grid_iou": 0.0},
                "spatial_iou": 0.0,
                "temporal_iou": 0.0,
                "combined_iou": 0.0,
                "pred_count": len(pred_polygons),
                "gt_count": len(gt_polygons)
            }

        # 指標1: グローバルポリゴンIoU
        polygon_iou = self._evaluate_polygon_iou(pred_polygons, gt_polygons)

        # 指標2: グリッド化F1スコア
        grid_metrics = self._evaluate_grid_f1(pred_polygons, gt_polygons)

        # 指標3: レイヤー別IoU
        spatial_iou = self._evaluate_spatial_iou(pred_polygons, gt_polygons)
        temporal_iou = self._evaluate_temporal_iou(pred_polygons, gt_polygons)
        combined_iou = spatial_iou * temporal_iou

        # 打印结果
        self._print_single_direction_results(
            direction, polygon_iou, grid_metrics,
            spatial_iou, temporal_iou, combined_iou
        )

        return {
            "polygon_iou": polygon_iou,
            "grid_metrics": grid_metrics,
            "spatial_iou": spatial_iou,
            "temporal_iou": temporal_iou,
            "combined_iou": combined_iou,
            "pred_count": len(pred_polygons),
            "gt_count": len(gt_polygons)
        }

    def _convert_to_polygons(
        self,
        data: List[Dict],
        data_type: str
    ) -> List[Polygon]:
        """
        将数据转换为Shapely Polygon对象

        Args:
            data: 数据列表（预测热点或真实三角形）
            data_type: "predictions" 或 "ground_truth"

        Returns:
            Polygon对象列表
        """
        polygons = []

        for i, item in enumerate(data):
            try:
                # 提取顶点坐标
                if data_type == "predictions":
                    # 予測ホットスポット形式：{"vertices": [[kp, time], ...]}
                    if "prediction_shape" in item:
                        vertices = item["prediction_shape"]["vertices"]
                    elif "vertices" in item:
                        vertices = item["vertices"]
                    else:
                        print(f"⚠️ 预测数据{i}缺少vertices字段")
                        continue
                else:
                    # 実データ形式：{"vertices": [...]} または他のフィールドから構築が必要
                    if "vertices" in item:
                        vertices = item["vertices"]
                    else:
                        # kp_start, kp_end, time等のフィールドから構築を試みる
                        vertices = self._construct_vertices_from_bounds(item)
                        if not vertices:
                            print(f"⚠️ 真实数据{i}无法构造vertices")
                            continue

                # 确保顶点格式正确
                if len(vertices) < 3:
                    print(f"⚠️ 数据{i}顶点数不足3个: {len(vertices)}")
                    continue

                # Polygonを作成（頂点がタプルリストであることを確認）
                coords = [(float(v[0]), float(v[1])) for v in vertices]
                poly = Polygon(coords)

                # 验证多边形有效性
                if not poly.is_valid:
                    poly = poly.buffer(0)  # 尝试修复
                    if not poly.is_valid:
                        print(f"⚠️ 数据{i}生成的多边形无效")
                        continue

                if poly.area > 0:
                    polygons.append(poly)
                else:
                    print(f"⚠️ 数据{i}多边形面积为0")

            except Exception as e:
                print(f"⚠️ 处理数据{i}时出错: {e}")
                continue

        print(f"✓ 成功转换{len(polygons)}个多边形 (来自{len(data)}个输入)")
        return polygons

    def _construct_vertices_from_bounds(self, item: Dict) -> List[List[float]]:
        """
        从边界信息构造三角形顶点（用于真实数据）
        """
        try:
            # 尝试多种可能的字段名
            kp_start = item.get('kp_start') or item.get('kp_min') or item.get('x_min')
            kp_end = item.get('kp_end') or item.get('kp_max') or item.get('x_max')
            time_start = item.get('time_start') or item.get('time_min') or item.get('y_min')
            time_end = item.get('time_end') or item.get('time_max') or item.get('y_max')
            time_peak = item.get('time_peak') or item.get('time_center')

            if None in [kp_start, kp_end, time_start, time_end]:
                return None

            # time_peakがない場合、中間点を取る
            if time_peak is None:
                time_peak = (time_start + time_end) / 2

            # 標準三角形を構築：底辺はkp_start、頂点はkp_end
            return [
                [kp_start, time_start],
                [kp_start, time_end],
                [kp_end, time_peak]
            ]
        except:
            return None

    # ========== 指標1: グローバルポリゴンIoU ==========

    def _evaluate_polygon_iou(
        self,
        pred_polygons: List[Polygon],
        gt_polygons: List[Polygon]
    ) -> float:
        """
        计算全局多边形IoU

        IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT)
        """
        try:
            # 合并所有预测多边形
            pred_union = unary_union(pred_polygons)

            # 合并所有真实多边形
            gt_union = unary_union(gt_polygons)

            # 计算交集和并集
            intersection = pred_union.intersection(gt_union)
            union = pred_union.union(gt_union)

            # IoUを計算
            intersection_area = intersection.area
            union_area = union.area

            if union_area == 0:
                return 0.0

            iou = intersection_area / union_area

            # 保存详细信息（用于打印）
            self._polygon_details = {
                "pred_area": pred_union.area,
                "gt_area": gt_union.area,
                "intersection_area": intersection_area,
                "union_area": union_area
            }

            return iou

        except Exception as e:
            print(f"⚠️ 计算多边形IoU时出错: {e}")
            return 0.0

    # ========== 指標2: グリッド化F1スコア ==========

    def _evaluate_grid_f1(
        self,
        pred_polygons: List[Polygon],
        gt_polygons: List[Polygon]
    ) -> Dict[str, float]:
        """
        计算栅格化F1-Score（基于KP区间栅格）

        Returns:
            {"precision": ..., "recall": ..., "f1_score": ..., "grid_iou": ...}
        """
        try:
            # KP区间がロードされているかチェック
            if not self.kp_intervals:
                print(f"⚠️ KP区间为空，无法进行栅格化评估")
                return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "grid_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

            # 1. 确定时间范围
            all_polygons = pred_polygons + gt_polygons
            bounds = self._get_bounds(all_polygons)
            kp_min, kp_max, time_min, time_max = bounds

            # 2. 创建时间栅格（24小时或其他粒度）
            n_time = int(np.ceil((time_max - time_min) / self.time_step))
            n_kp = len(self.kp_intervals)

            if n_kp <= 0 or n_time <= 0:
                print(f"⚠️ 栅格尺寸无效: {n_kp} × {n_time}")
                return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "grid_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

            print(f"  栅格尺寸: {n_kp} KP区间 × {n_time} 时间段 = {n_kp*n_time} 单元格")
            print(f"  KP区间数: {n_kp} (基于道路实际结构)")
            print(f"  时间范围: [{time_min:.0f}, {time_max:.0f}] min (粒度: {self.time_step} min)")

            # 3. 予測ポリゴンをグリッド化（KP区間を使用）
            pred_grid = self._rasterize_polygons_kp_based(
                pred_polygons, time_min, time_max, n_time
            )

            # 4. 実ポリゴンをグリッド化（KP区間を使用）
            gt_grid = self._rasterize_polygons_kp_based(
                gt_polygons, time_min, time_max, n_time
            )

            # 5. 计算混淆矩阵
            tp = np.sum((pred_grid == 1) & (gt_grid == 1))
            fp = np.sum((pred_grid == 1) & (gt_grid == 0))
            fn = np.sum((pred_grid == 0) & (gt_grid == 1))
            tn = np.sum((pred_grid == 0) & (gt_grid == 0))

            # 6. 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            grid_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            # 保存栅格（用于可视化）
            self._grid_data = {
                "pred_grid": pred_grid,
                "gt_grid": gt_grid,
                "time_min": time_min,
                "time_max": time_max,
                "n_time": n_time,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "grid_iou": grid_iou,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn)
            }

        except Exception as e:
            print(f"⚠️ 计算栅格F1时出错: {e}")
            import traceback
            traceback.print_exc()
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "grid_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def _get_bounds(self, polygons: List[Polygon]) -> Tuple[float, float, float, float]:
        """获取所有多边形的边界"""
        all_bounds = [poly.bounds for poly in polygons]  # (minx, miny, maxx, maxy)
        kp_min = min(b[0] for b in all_bounds)
        time_min = min(b[1] for b in all_bounds)
        kp_max = max(b[2] for b in all_bounds)
        time_max = max(b[3] for b in all_bounds)
        return kp_min, kp_max, time_min, time_max

    def _rasterize_polygons_kp_based(
        self,
        polygons: List[Polygon],
        time_min: float,
        time_max: float,
        n_time: int
    ) -> np.ndarray:
        """
        将多边形栅格化（基于KP区间）

        Args:
            polygons: 多边形列表
            time_min: 时间最小值
            time_max: 时间最大值
            n_time: 时间栅格数量

        Returns:
            二值栅格 (n_kp, n_time)，1表示栅格与多边形有重合
        """
        n_kp = len(self.kp_intervals)
        grid = np.zeros((n_kp, n_time), dtype=np.uint8)

        # 合并所有多边形（提高效率）
        union_poly = unary_union(polygons)

        # 各KP区間グリッドを走査
        for i, (kp_start, kp_end) in enumerate(self.kp_intervals):
            # 遍历每个时间栅格
            for j in range(n_time):
                # 计算时间区间
                t_start = time_min + j * self.time_step
                t_end = time_min + (j + 1) * self.time_step

                # グリッド矩形を作成 (shapely.geometry.boxを使用)
                # box(minx, miny, maxx, maxy)
                grid_rect = box(kp_start, t_start, kp_end, t_end)

                # 检查多边形与栅格矩形是否相交
                if union_poly.intersects(grid_rect):
                    grid[i, j] = 1

        return grid

    # ========== 指標3: レイヤー別IoU ==========

    def _evaluate_spatial_iou(
        self,
        pred_polygons: List[Polygon],
        gt_polygons: List[Polygon]
    ) -> float:
        """
        计算空间维度IoU（KP维度）
        """
        try:
            # KP区間を抽出
            pred_intervals = [self._get_kp_interval(p) for p in pred_polygons]
            gt_intervals = [self._get_kp_interval(p) for p in gt_polygons]

            # 1次元区間IoUを計算
            iou = self._interval_iou(pred_intervals, gt_intervals)
            return iou

        except Exception as e:
            print(f"⚠️ 计算空间IoU时出错: {e}")
            return 0.0

    def _evaluate_temporal_iou(
        self,
        pred_polygons: List[Polygon],
        gt_polygons: List[Polygon]
    ) -> float:
        """
        计算时间维度IoU（Time维度）
        """
        try:
            # 提取时间区间
            pred_intervals = [self._get_time_interval(p) for p in pred_polygons]
            gt_intervals = [self._get_time_interval(p) for p in gt_polygons]

            # 1次元区間IoUを計算
            iou = self._interval_iou(pred_intervals, gt_intervals)
            return iou

        except Exception as e:
            print(f"⚠️ 计算时间IoU时出错: {e}")
            return 0.0

    def _get_kp_interval(self, polygon: Polygon) -> Tuple[float, float]:
        """ポリゴンからKP区間を抽出"""
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        return (bounds[0], bounds[2])

    def _get_time_interval(self, polygon: Polygon) -> Tuple[float, float]:
        """从多边形提取时间区间"""
        bounds = polygon.bounds
        return (bounds[1], bounds[3])

    def _interval_iou(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]]
    ) -> float:
        """
        计算1维区间的IoU

        Args:
            intervals1: 区间列表1 [(start, end), ...]
            intervals2: 区间列表2 [(start, end), ...]

        Returns:
            IoU值
        """
        # 合并区间
        union1 = self._merge_intervals(intervals1)
        union2 = self._merge_intervals(intervals2)

        # 计算交集和并集的总长度
        intersection_length = self._intervals_intersection_length(union1, union2)
        union_length = self._intervals_union_length(union1, union2)

        if union_length == 0:
            return 0.0

        return intersection_length / union_length

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        合并重叠的区间

        Args:
            intervals: [(start, end), ...]

        Returns:
            合并后的区间列表
        """
        if not intervals:
            return []

        # 排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        # 合并
        merged = [sorted_intervals[0]]
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 重叠
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _intervals_intersection_length(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]]
    ) -> float:
        """计算两组区间的交集总长度"""
        total_length = 0.0

        for i1 in intervals1:
            for i2 in intervals2:
                # 计算两个区间的交集
                start = max(i1[0], i2[0])
                end = min(i1[1], i2[1])
                if end > start:
                    total_length += (end - start)

        return total_length

    def _intervals_union_length(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]]
    ) -> float:
        """计算两组区间的并集总长度"""
        # 合并所有区间
        all_intervals = intervals1 + intervals2
        merged = self._merge_intervals(all_intervals)

        # 计算总长度
        total_length = sum(end - start for start, end in merged)
        return total_length

    # ========== 辅助函数 ==========

    def _calculate_average_metrics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """计算所有方向的平均指标"""
        directions = [k for k in results.keys() if k != "average"]

        if not directions:
            return {}

        avg = {
            "polygon_iou": np.mean([results[d]["polygon_iou"] for d in directions]),
            "grid_metrics": {
                "precision": np.mean([results[d]["grid_metrics"]["precision"] for d in directions]),
                "recall": np.mean([results[d]["grid_metrics"]["recall"] for d in directions]),
                "f1_score": np.mean([results[d]["grid_metrics"]["f1_score"] for d in directions]),
                "grid_iou": np.mean([results[d]["grid_metrics"]["grid_iou"] for d in directions])
            },
            "spatial_iou": np.mean([results[d]["spatial_iou"] for d in directions]),
            "temporal_iou": np.mean([results[d]["temporal_iou"] for d in directions]),
            "combined_iou": np.mean([results[d]["combined_iou"] for d in directions])
        }

        return avg

    def _print_single_direction_results(
        self,
        direction: str,
        polygon_iou: float,
        grid_metrics: Dict,
        spatial_iou: float,
        temporal_iou: float,
        combined_iou: float
    ):
        """打印单个方向的评估结果"""
        print(f"\n【指标1】全局多边形IoU: {polygon_iou:.4f}")
        if hasattr(self, '_polygon_details'):
            d = self._polygon_details
            print(f"  - 预测区域总面积: {d['pred_area']:.2f} km·min")
            print(f"  - 真实区域总面积: {d['gt_area']:.2f} km·min")
            print(f"  - 交集面积: {d['intersection_area']:.2f} km·min")
            print(f"  - 并集面积: {d['union_area']:.2f} km·min")

        print(f"\n【指标2】栅格化评估 (KP区间: {len(self.kp_intervals)}个 × 时间: {self.time_step}min)")
        print(f"  - Precision: {grid_metrics['precision']:.4f} (预测的{grid_metrics['precision']*100:.1f}%是正确的)")
        print(f"  - Recall: {grid_metrics['recall']:.4f} (真实拥堵的{grid_metrics['recall']*100:.1f}%被预测到)")
        print(f"  - F1-Score: {grid_metrics['f1_score']:.4f}")
        print(f"  - Grid IoU: {grid_metrics['grid_iou']:.4f}")

        print(f"\n【指标3】分层IoU")
        print(f"  - 空间IoU (KP): {spatial_iou:.4f}")
        print(f"  - 时间IoU (Time): {temporal_iou:.4f}")
        print(f"  - 综合IoU (相乘): {combined_iou:.4f}")

    def print_summary_report(self, results: Dict[str, Any]):
        """打印汇总评估报告"""
        print("\n" + "="*60)
        print("                  评估汇总报告")
        print("="*60)

        for direction, metrics in results.items():
            if direction == "average":
                print(f"\n{'─'*60}")
                print(f"【整体平均】")
            else:
                print(f"\n【{direction}行】")

            print(f"  全局多边形IoU:    {metrics['polygon_iou']:.4f}")
            print(f"  栅格F1-Score:      {metrics['grid_metrics']['f1_score']:.4f}")
            print(f"  空间IoU:           {metrics['spatial_iou']:.4f}")
            print(f"  时间IoU:           {metrics['temporal_iou']:.4f}")
            print(f"  综合IoU:           {metrics['combined_iou']:.4f}")

    def save_results_to_json(self, results: Dict[str, Any], output_path: str):
        """評価結果をJSONファイルに保存"""
        # 添加元数据
        output = {
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grid_configuration": {
                "road_type": self.road_type,
                "direction": self.direction,
                "kp_intervals_count": len(self.kp_intervals),
                "time_step_min": self.time_step
            },
            "results": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 评估结果已保存到: {output_path}")
