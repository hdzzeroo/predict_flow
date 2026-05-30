"""
Fallbackアナライザーモジュール
LLMが利用できない場合、シンプルなルールベースアルゴリズムでホットスポット分析
"""

from typing import Dict, List, Any
import numpy as np


class FallbackAnalyzer:
    """
    Fallbackアナライザー
    シンプルな距離クラスタリングアルゴリズムを使用してホットスポットを識別
    """

    def __init__(
        self,
        kp_threshold: float = 5.0,      # KP距離閾値（km）
        time_threshold: int = 180,       # 時間距離閾値（分、3時間）
        min_cluster_size: int = 2        # 最小クラスターサイズ
    ):
        """
        Fallbackアナライザーを初期化

        Args:
            kp_threshold: KP空間距離閾値
            time_threshold: 時間距離閾値
            min_cluster_size: ホットスポットを形成する最小イベント数
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
        fallback分析を実行

        Args:
            triangles: 三角形データリスト
            direction: 方向

        Returns:
            分析結果（LLM出力フォーマットと一致）
        """
        if not triangles:
            return self._empty_result(direction)

        print(f"🔄 Using fallback analyzer for {direction} direction")

        # クラスタリングを実行
        clusters = self._simple_clustering(triangles)

        # 小さいクラスターをフィルタ
        valid_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        if not valid_clusters:
            print(f"  No significant clusters found (min size: {self.min_cluster_size})")
            return self._empty_result(direction)

        # ホットスポットを生成
        hotspots = []
        for i, cluster_indices in enumerate(valid_clusters):
            hotspot = self._create_hotspot(
                cluster_indices,
                triangles,
                hotspot_id=i + 1
            )
            hotspots.append(hotspot)

        # 頻度順にソート
        hotspots.sort(key=lambda x: x['frequency'], reverse=True)

        # サマリーを生成
        summary = {
            "total_hotspots": len(hotspots),
            "most_severe_hotspot_id": hotspots[0]['hotspot_id'] if hotspots else None,
            "analysis_confidence": 0.6  # Fallbackメソッドの信頼度は低い
        }

        print(f"  Identified {len(hotspots)} hotspots using fallback method")

        return {
            "direction": direction,
            "hotspots": hotspots,
            "summary": summary
        }

    def _simple_clustering(self, triangles: List[Dict[str, Any]]) -> List[List[int]]:
        """
        シンプルな距離クラスタリングアルゴリズム

        Args:
            triangles: 三角形リスト

        Returns:
            クラスター結果（三角形インデックスのリストのリスト）
        """
        n = len(triangles)
        if n == 0:
            return []

        # 全三角形間の距離を計算
        distances = self._calculate_distance_matrix(triangles)

        # 初期化：各三角形は独自のクラスター
        clusters = [[i] for i in range(n)]
        assigned = [False] * n

        # peak_kp順に三角形をソート
        sorted_indices = sorted(range(n), key=lambda i: triangles[i].get('peak_kp', 0))

        # 貪欲クラスタリング
        for i in sorted_indices:
            if assigned[i]:
                continue

            current_cluster = [i]
            assigned[i] = True

            # 隣接する三角形を検索
            for j in sorted_indices:
                if i == j or assigned[j]:
                    continue

                # 距離が十分近いかチェック
                if distances[i][j] <= 1.0:  # 归一化距离阈值
                    current_cluster.append(j)
                    assigned[j] = True

            if len(current_cluster) >= self.min_cluster_size:
                clusters.append(current_cluster)

        # 有効なクラスターを返す
        return [c for c in clusters if len(c) >= self.min_cluster_size]

    def _calculate_distance_matrix(self, triangles: List[Dict[str, Any]]) -> np.ndarray:
        """
        三角形間の正規化距離マトリックスを計算

        Args:
            triangles: 三角形リスト

        Returns:
            距離マトリックス
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
        2つの三角形間の正規化距離を計算

        Args:
            t1: 三角形1
            t2: 三角形2

        Returns:
            正規化距離（0-1範囲）
        """
        # KP距離
        kp1 = t1.get('peak_kp', 0)
        kp2 = t2.get('peak_kp', 0)
        kp_dist = abs(kp1 - kp2) / self.kp_threshold

        # 時間距離
        time1 = t1.get('peak_time', 0)
        time2 = t2.get('peak_time', 0)
        time_dist = abs(time1 - time2) / self.time_threshold

        # 総合距離（加重平均）
        distance = 0.6 * kp_dist + 0.4 * time_dist

        return distance

    def _create_hotspot(
        self,
        cluster_indices: List[int],
        triangles: List[Dict[str, Any]],
        hotspot_id: int
    ) -> Dict[str, Any]:
        """
        クラスターからホットスポットを作成

        Args:
            cluster_indices: クラスター内の三角形インデックス
            triangles: 全三角形
            hotspot_id: ホットスポットID

        Returns:
            ホットスポットデータ辞書
        """
        cluster_triangles = [triangles[i] for i in cluster_indices]

        # KPと時間範囲を抽出
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

        # 重大度を評価
        frequency = len(cluster_indices)
        if frequency >= 7:
            severity = "high"
        elif frequency >= 4:
            severity = "medium"
        else:
            severity = "low"

        # 説明を作成
        time_start_hour, time_start_min = divmod(time_range[0], 60)
        time_end_hour, time_end_min = divmod(time_range[1], 60)

        description = (
            f"KP {kp_range[0]:.1f}-{kp_range[1]:.1f}区間, "
            f"{time_start_hour:02d}:{time_start_min:02d}-{time_end_hour:02d}:{time_end_min:02d}時間帯の交通拥挤"
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
        """空の結果を返す"""
        return {
            "direction": direction,
            "hotspots": [],
            "summary": {
                "total_hotspots": 0,
                "most_severe_hotspot_id": None,
                "analysis_confidence": 0.0
            }
        }


# 便利関数
def create_fallback_analyzer(
    kp_threshold: float = 5.0,
    time_threshold: int = 180,
    min_cluster_size: int = 2
) -> FallbackAnalyzer:
    """
    fallbackアナライザーを作成する便利関数

    Args:
        kp_threshold: KP距離閾値
        time_threshold: 時間距離閾値
        min_cluster_size: 最小クラスターサイズ

    Returns:
        FallbackAnalyzerインスタンス
    """
    return FallbackAnalyzer(
        kp_threshold=kp_threshold,
        time_threshold=time_threshold,
        min_cluster_size=min_cluster_size
    )