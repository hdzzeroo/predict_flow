"""
データ準備モジュール
生データをLLMが理解できるフォーマットに変換
"""

from typing import Dict, List, Any, Optional
import pandas as pd


class DataPreparer:
    """データ準備クラス（基底クラス）"""

    @staticmethod
    def clean_vertices(vertices: List[List[float]]) -> List[List[float]]:
        """
        頂点データをクリーン化し、重複した連続頂点を削除

        Args:
            vertices: 頂点リスト、各頂点は [kp, time]

        Returns:
            クリーン化後の頂点リスト
        """
        if not vertices or len(vertices) < 2:
            return vertices

        cleaned = []
        for vertex in vertices:
            # 前の頂点と重複しているかチェック（浮動小数点精度を考慮）
            if not cleaned or \
               abs(vertex[0] - cleaned[-1][0]) > 0.01 or \
               abs(vertex[1] - cleaned[-1][1]) > 0.01:
                cleaned.append(vertex)

        # 先頭と末尾が重複しているかチェック（閉じポリゴンの場合）
        if len(cleaned) > 2:
            if abs(cleaned[0][0] - cleaned[-1][0]) < 0.01 and \
               abs(cleaned[0][1] - cleaned[-1][1]) < 0.01:
                cleaned = cleaned[:-1]

        return cleaned

    @staticmethod
    def normalize_triangle_shapes(triangles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        三角形/四辺形データを正規化：
        - 重複頂点を削除
        - 実際の頂点数量に基づいて shape_type を更新
        - 三角形は3つの異なる点が必要
        - 四辺形は4つの異なる点が必要

        Args:
            triangles: 生三角形データリスト

        Returns:
            正規化後の三角形データリスト
        """
        if not triangles:
            return []

        normalized = []
        stats = {
            'original_triangles': 0,
            'original_quads': 0,
            'cleaned_triangles': 0,
            'cleaned_quads': 0,
            'degraded_to_triangle': 0,
            'invalid_shapes': 0
        }

        for triangle in triangles:
            # 生データをコピー
            normalized_triangle = triangle.copy()

            # 元の shape_type を取得
            original_shape = triangle.get('shape_type', 'triangle')
            if original_shape == 'triangle':
                stats['original_triangles'] += 1
            else:
                stats['original_quads'] += 1

            # 頂点データを取得
            vertices = triangle.get('vertices', [])
            if not vertices:
                # verticesフィールドがない場合、クリーン化をスキップ
                normalized.append(normalized_triangle)
                continue

            # 重複頂点をクリーン化
            cleaned_vertices = DataPreparer.clean_vertices(vertices)
            vertex_count = len(cleaned_vertices)

            # クリーン化後の頂点数量に基づいて形状タイプを判断
            if vertex_count < 3:
                # 無効な形状（3頂点未満）
                print(f"⚠️ Warning: Triangle {triangle.get('id', '?')} has only {vertex_count} vertices after cleaning, skipping")
                stats['invalid_shapes'] += 1
                continue
            elif vertex_count == 3:
                # 三角形
                normalized_triangle['vertices'] = cleaned_vertices
                normalized_triangle['shape_type'] = 'triangle'
                stats['cleaned_triangles'] += 1

                if original_shape == 'quadrilateral':
                    stats['degraded_to_triangle'] += 1

            elif vertex_count == 4:
                # 四辺形
                normalized_triangle['vertices'] = cleaned_vertices
                normalized_triangle['shape_type'] = 'quadrilateral'
                stats['cleaned_quads'] += 1
            else:
                # 頂点数>4、そのまま使用して警告を表示
                print(f"⚠️ Warning: Triangle {triangle.get('id', '?')} has {vertex_count} vertices (>4)")
                normalized_triangle['vertices'] = cleaned_vertices
                normalized.append(normalized_triangle)
                continue

            normalized.append(normalized_triangle)

        # クリーン化統計を出力
        print(f"\n📊 Shape normalization statistics:")
        print(f"  Original: {stats['original_triangles']} triangles, {stats['original_quads']} quadrilaterals")
        print(f"  Cleaned: {stats['cleaned_triangles']} triangles, {stats['cleaned_quads']} quadrilaterals")
        if stats['degraded_to_triangle'] > 0:
            print(f"  ⚠️ Degraded: {stats['degraded_to_triangle']} quadrilaterals → triangles (duplicate vertices removed)")
        if stats['invalid_shapes'] > 0:
            print(f"  ❌ Invalid: {stats['invalid_shapes']} shapes removed (<3 vertices)")

        return normalized

    @staticmethod
    def prepare_triangle_data(triangles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        三角形データを準備し、フォーマットを標準化してIDを追加

        Args:
            triangles: 生三角形データリスト

        Returns:
            標準化後の三角形データ
        """
        if not triangles:
            return []

        prepared = []
        for i, t in enumerate(triangles):
            prepared.append({
                "id": i,
                "kp_start": round(float(t.get('kp_start', 0)), 2),
                "kp_end": round(float(t.get('kp_end', 0)), 2),
                "peak_kp": round(float(t.get('peak_kp', 0)), 2),
                "time_start": int(t.get('time_start', 0)),
                "time_end": int(t.get('time_end', 0)),
                "peak_time": int(t.get('peak_time', 0))
            })

        return prepared

    @staticmethod
    def calculate_data_statistics(triangles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        データ統計情報を計算

        Args:
            triangles: 三角形データリスト

        Returns:
            統計情報辞書
        """
        if not triangles:
            return {
                "total_count": 0,
                "kp_range": [0, 0],
                "time_range": [0, 0],
                "kp_span": 0,
                "time_span": 0
            }

        # 全てのKPと時間値を抽出
        kp_values = []
        time_values = []

        for t in triangles:
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

        # 無効値をフィルタ
        kp_values = [v for v in kp_values if v is not None and v > 0]
        time_values = [v for v in time_values if v is not None and v >= 0]

        if not kp_values or not time_values:
            return {
                "total_count": len(triangles),
                "kp_range": [0, 0],
                "time_range": [0, 0],
                "kp_span": 0,
                "time_span": 0
            }

        kp_min, kp_max = min(kp_values), max(kp_values)
        time_min, time_max = min(time_values), max(time_values)

        return {
            "total_count": len(triangles),
            "kp_range": [round(kp_min, 2), round(kp_max, 2)],
            "time_range": [int(time_min), int(time_max)],
            "kp_span": round(kp_max - kp_min, 2),
            "time_span": int(time_max - time_min)
        }

    @staticmethod
    def prepare_llm_input(
        triangles: List[Dict[str, Any]],
        direction: str,
        csv_files: List[str]
    ) -> Dict[str, Any]:
        """
        完全なLLM入力データを準備

        Args:
            triangles: 三角形データ
            direction: 方向
            csv_files: CSVファイルリスト

        Returns:
            完全なLLM入力データ辞書
        """
        prepared_triangles = DataPreparer.prepare_triangle_data(triangles)
        statistics = DataPreparer.calculate_data_statistics(triangles)

        return {
            "direction": direction,
            "csv_files": csv_files,
            "statistics": statistics,
            "triangles": prepared_triangles
        }


class RawDataLoader:
    """生データローダー"""

    @staticmethod
    def load_csv_summary(csv_path: str) -> Optional[Dict[str, Any]]:
        """
        CSVファイルを読み込んでサマリー情報を生成

        Args:
            csv_path: CSVファイルパス

        Returns:
            CSVデータサマリー（読み込み失敗時はNoneを返す）
        """
        try:
            df = pd.read_csv(csv_path)

            summary = {
                "file_name": csv_path,
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": None,
                "kp_range": None
            }

            # 日付範囲を抽出を試みる
            date_columns = [col for col in df.columns if 'date' in col.lower() or '日付' in col]
            if date_columns:
                try:
                    dates = pd.to_datetime(df[date_columns[0]])
                    summary["date_range"] = [
                        dates.min().strftime('%Y-%m-%d'),
                        dates.max().strftime('%Y-%m-%d')
                    ]
                except:
                    pass

            # KP範囲の抽出を試みる
            kp_columns = [col for col in df.columns if 'kp' in col.lower()]
            if kp_columns:
                try:
                    kp_values = df[kp_columns[0]].dropna()
                    summary["kp_range"] = [
                        float(kp_values.min()),
                        float(kp_values.max())
                    ]
                except:
                    pass

            return summary

        except Exception as e:
            print(f"⚠️ Failed to load CSV {csv_path}: {str(e)}")
            return None

    @staticmethod
    def load_multiple_csv_summaries(csv_paths: List[str]) -> List[Dict[str, Any]]:
        """
        複数のCSVファイルのサマリー情報を読み込み

        Args:
            csv_paths: CSVファイルパスリスト

        Returns:
            CSVサマリー情報リスト
        """
        summaries = []
        for path in csv_paths:
            summary = RawDataLoader.load_csv_summary(path)
            if summary:
                summaries.append(summary)
        return summaries


class OutputFormatter:
    """出力フォーマッター"""

    @staticmethod
    def format_hotspot_for_display(hotspot: Dict[str, Any]) -> str:
        """
        単一ホットスポット情報を表示用にフォーマット

        Args:
            hotspot: ホットスポットデータ辞書

        Returns:
            フォーマット済み文字列
        """
        kp_start, kp_end = hotspot['kp_range']
        time_start, time_end = hotspot['time_range']

        # 時間を時:分フォーマットに変換
        start_hour, start_min = divmod(time_start, 60)
        end_hour, end_min = divmod(time_end, 60)

        return (
            f"Hotspot {hotspot['hotspot_id']}: "
            f"KP {kp_start:.1f}-{kp_end:.1f}km, "
            f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}, "
            f"{hotspot['frequency']} events, "
            f"severity: {hotspot['severity']}"
        )

    @staticmethod
    def format_analysis_summary(analysis_result: Dict[str, Any]) -> str:
        """
        分析結果サマリーをフォーマット

        Args:
            analysis_result: LLM分析結果

        Returns:
            フォーマット済みサマリー文字列
        """
        direction = analysis_result.get('direction', 'Unknown')
        hotspots = analysis_result.get('hotspots', [])
        summary = analysis_result.get('summary', {})

        lines = [
            f"=== {direction}方向分析結果 ===",
            f"識別されたホットスポット数: {summary.get('total_hotspots', 0)} エリア",
            f"分析信頼度: {summary.get('analysis_confidence', 0):.2f}"
        ]

        if hotspots:
            lines.append("\nホットスポット詳細:")
            for hotspot in hotspots:
                lines.append(f"  {OutputFormatter.format_hotspot_for_display(hotspot)}")

        return "\n".join(lines)


# 便利関数
def prepare_direction_data(
    direction_data: Dict[str, Dict[str, Any]],
    file_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    全方向のデータを準備

    Args:
        direction_data: 方向データ辞書
        file_paths: ファイルパスリスト

    Returns:
        準備好的データ辞書
    """
    import os

    prepared = {}
    csv_files = [os.path.basename(fp) for fp in file_paths]

    for direction, data in direction_data.items():
        triangles = data.get('triangles', [])
        prepared[direction] = DataPreparer.prepare_llm_input(
            triangles=triangles,
            direction=direction,
            csv_files=csv_files
        )

    return prepared