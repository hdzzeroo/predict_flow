"""
Fallback分析器模块
当LLM不可用时使用简单的规则基算法进行热点分析
"""

from typing import Dict, List, Any
import numpy as np


class FallbackAnalyzer:
    """
    Fallback分析器
    使用简单的距离聚类算法识别热点
    """

    def __init__(
        self,
        kp_threshold: float = 5.0,      # KP距离阈值（km）
        time_threshold: int = 180,       # 时间距离阈值（分钟，3小时）
        min_cluster_size: int = 2        # 最小聚类大小
    ):
        """
        初始化Fallback分析器

        Args:
            kp_threshold: KP空间距离阈值
            time_threshold: 时间距离阈值
            min_cluster_size: 形成热点的最小事件数
        """
        self.kp_threshold = kp_threshold
        self.time_threshold = time_threshold
        self.min_cluster_size = min_cluster_size

    def analyze(
        self,
        triangles: List[Dict[str, Any]],
        direction: str
    ) -> Dict[str, Any]:
        """
        执行fallback分析

        Args:
            triangles: 三角形数据列表
            direction: 方向

        Returns:
            分析结果（与LLM输出格式一致）
        """
        if not triangles:
            return self._empty_result(direction)

        print(f"🔄 Using fallback analyzer for {direction} direction")

        # 执行聚类
        clusters = self._simple_clustering(triangles)

        # 过滤小聚类
        valid_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        if not valid_clusters:
            print(f"  No significant clusters found (min size: {self.min_cluster_size})")
            return self._empty_result(direction)

        # 生成热点
        hotspots = []
        for i, cluster_indices in enumerate(valid_clusters):
            hotspot = self._create_hotspot(
                cluster_indices,
                triangles,
                hotspot_id=i + 1
            )
            hotspots.append(hotspot)

        # 按频次排序
        hotspots.sort(key=lambda x: x['frequency'], reverse=True)

        # 生成摘要
        summary = {
            "total_hotspots": len(hotspots),
            "most_severe_hotspot_id": hotspots[0]['hotspot_id'] if hotspots else None,
            "analysis_confidence": 0.6  # Fallback方法置信度较低
        }

        print(f"  Identified {len(hotspots)} hotspots using fallback method")

        return {
            "direction": direction,
            "hotspots": hotspots,
            "summary": summary
        }

    def _simple_clustering(self, triangles: List[Dict[str, Any]]) -> List[List[int]]:
        """
        简单的距离聚类算法

        Args:
            triangles: 三角形列表

        Returns:
            聚类结果（三角形索引的列表的列表）
        """
        n = len(triangles)
        if n == 0:
            return []

        # 计算所有三角形之间的距离矩阵
        distances = self._calculate_distance_matrix(triangles)

        # 初始化：每个三角形自成一类
        clusters = [[i] for i in range(n)]
        assigned = [False] * n

        # 按peak_kp排序三角形
        sorted_indices = sorted(range(n), key=lambda i: triangles[i].get('peak_kp', 0))

        # 贪心聚类
        for i in sorted_indices:
            if assigned[i]:
                continue

            current_cluster = [i]
            assigned[i] = True

            # 查找相邻的三角形
            for j in sorted_indices:
                if i == j or assigned[j]:
                    continue

                # 检查距离是否足够近
                if distances[i][j] <= 1.0:  # 归一化距离阈值
                    current_cluster.append(j)
                    assigned[j] = True

            if len(current_cluster) >= self.min_cluster_size:
                clusters.append(current_cluster)

        # 返回有效聚类
        return [c for c in clusters if len(c) >= self.min_cluster_size]

    def _calculate_distance_matrix(self, triangles: List[Dict[str, Any]]) -> np.ndarray:
        """
        计算三角形之间的归一化距离矩阵

        Args:
            triangles: 三角形列表

        Returns:
            距离矩阵
        """
        n = len(triangles)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._calculate_distance(triangles[i], triangles[j])
                distances[i][j] = dist
                distances[j][i] = dist

        return distances

    def _calculate_distance(
        self,
        t1: Dict[str, Any],
        t2: Dict[str, Any]
    ) -> float:
        """
        计算两个三角形之间的归一化距离

        Args:
            t1: 三角形1
            t2: 三角形2

        Returns:
            归一化距离（0-1范围）
        """
        # KP距离
        kp1 = t1.get('peak_kp', 0)
        kp2 = t2.get('peak_kp', 0)
        kp_dist = abs(kp1 - kp2) / self.kp_threshold

        # 时间距离
        time1 = t1.get('peak_time', 0)
        time2 = t2.get('peak_time', 0)
        time_dist = abs(time1 - time2) / self.time_threshold

        # 综合距离（加权平均）
        distance = 0.6 * kp_dist + 0.4 * time_dist

        return distance

    def _create_hotspot(
        self,
        cluster_indices: List[int],
        triangles: List[Dict[str, Any]],
        hotspot_id: int
    ) -> Dict[str, Any]:
        """
        从聚类创建热点

        Args:
            cluster_indices: 聚类中的三角形索引
            triangles: 所有三角形
            hotspot_id: 热点ID

        Returns:
            热点数据字典
        """
        cluster_triangles = [triangles[i] for i in cluster_indices]

        # 提取KP和时间范围
        kp_values = []
        time_values = []

        for t in cluster_triangles:
            kp_values.extend([
                t.get('kp_start', 0),
                t.get('kp_end', 0),
                t.get('peak_kp', 0)
            ])
            time_values.extend([
                t.get('time_start', 0),
                t.get('time_end', 0),
                t.get('peak_time', 0)
            ])

        kp_range = [min(kp_values), max(kp_values)]
        time_range = [int(min(time_values)), int(max(time_values))]

        # 评估严重程度
        frequency = len(cluster_indices)
        if frequency >= 7:
            severity = "high"
        elif frequency >= 4:
            severity = "medium"
        else:
            severity = "low"

        # 生成描述
        time_start_hour, time_start_min = divmod(time_range[0], 60)
        time_end_hour, time_end_min = divmod(time_range[1], 60)

        description = (
            f"KP {kp_range[0]:.1f}-{kp_range[1]:.1f}区间, "
            f"{time_start_hour:02d}:{time_start_min:02d}-{time_end_hour:02d}:{time_end_min:02d}时段拥堵"
        )

        return {
            "hotspot_id": hotspot_id,
            "kp_range": [round(kp_range[0], 2), round(kp_range[1], 2)],
            "time_range": time_range,
            "included_triangle_ids": cluster_indices,
            "frequency": frequency,
            "severity": severity,
            "description": description
        }

    def _empty_result(self, direction: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "direction": direction,
            "hotspots": [],
            "summary": {
                "total_hotspots": 0,
                "most_severe_hotspot_id": None,
                "analysis_confidence": 0.0
            }
        }


# 便捷函数
def create_fallback_analyzer(
    kp_threshold: float = 5.0,
    time_threshold: int = 180,
    min_cluster_size: int = 2
) -> FallbackAnalyzer:
    """
    创建fallback分析器的便捷函数

    Args:
        kp_threshold: KP距离阈值
        time_threshold: 时间距离阈值
        min_cluster_size: 最小聚类大小

    Returns:
        FallbackAnalyzer实例
    """
    return FallbackAnalyzer(
        kp_threshold=kp_threshold,
        time_threshold=time_threshold,
        min_cluster_size=min_cluster_size
    )