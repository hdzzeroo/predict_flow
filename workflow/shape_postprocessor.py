#!/usr/bin/env python3
"""
LLM予測のための形状後処理

2種類の不正な形状を修正：
1. 四角形 → 三角形（先端側の点をマージ）
2. 不規則な台形 → 正規の台形（下端の時間範囲内に上端を配置）

from typing import Dict, List, Any, Tuple, Optional
import copy


def postprocess_prediction_shapes(llm_result: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """
    Post-process LLM prediction shapes to fix malformed shapes.

    Args:
        llm_result: The parsed LLM response containing hotspots
        direction: "上" (upbound) or "下" (downbound)

    Returns:
        Modified llm_result with corrected shapes
    """
    result = copy.deepcopy(llm_result)
    hotspots = result.get('hotspots', [])

    corrections_made = []

    for hotspot in hotspots:
        prediction_shape = hotspot.get('prediction_shape', {})
        vertices = prediction_shape.get('vertices', [])
        shape_type = prediction_shape.get('shape_type', '')
        hotspot_id = hotspot.get('hotspot_id', 'unknown')

        if not vertices:
            continue

        # 頂点を[kp, time]形式に正規化
        normalized_vertices = _normalize_vertices(vertices)

        if len(normalized_vertices) == 4:
            # 四角形かどうかをチェック（三角形に変換が必要）
            if _is_rectangle(normalized_vertices):
                new_vertices = _convert_rectangle_to_triangle(normalized_vertices, direction)
                prediction_shape['vertices'] = new_vertices
                prediction_shape['shape_type'] = 'triangle'
                corrections_made.append({
                    'hotspot_id': hotspot_id,
                    'correction': 'rectangle_to_triangle',
                    'original_vertices': normalized_vertices,
                    'new_vertices': new_vertices
                })
            else:
                # 台形の場合 - 不規則かどうかをチェックし修正
                fixed_vertices, was_fixed = _fix_irregular_trapezoid(normalized_vertices, direction)
                if was_fixed:
                    prediction_shape['vertices'] = fixed_vertices
                    corrections_made.append({
                        'hotspot_id': hotspot_id,
                        'correction': 'irregular_trapezoid_fixed',
                        'original_vertices': normalized_vertices,
                        'new_vertices': fixed_vertices
                    })

        elif len(normalized_vertices) == 3:
            # すでに三角形 - 方向が正しいか確認
            pass  # 必要に応じて方向検証を追加できます

    # 修正ログを結果に保存
    if corrections_made:
        result['_shape_corrections'] = corrections_made
        print(f"   📐 Shape corrections applied: {len(corrections_made)}")
        for c in corrections_made:
            print(f"      - Hotspot {c['hotspot_id']}: {c['correction']}")

    return result


def _normalize_vertices(vertices: List) -> List[List[float]]:
    """
    頂点を[[kp, time], ...]形式に正規化。
    リスト形式と辞書形式の両方を処理。
    """
    normalized = []
    for v in vertices:
        if isinstance(v, dict):
            kp = v.get('kp', v.get('x', 0))
            time = v.get('time_min', v.get('time', v.get('y', 0)))
            normalized.append([float(kp), float(time)])
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            normalized.append([float(v[0]), float(v[1])])
        else:
            normalized.append([float(v[0]) if len(v) > 0 else 0, float(v[1]) if len(v) > 1 else 0])
    return normalized


def _is_rectangle(vertices: List[List[float]]) -> bool:
    """
    4つの頂点が四角形を形成するかチェック。
    四角形は2つの垂直辺（同じKP値）と2つの水平辺（同じ時間値）を持つ。
    """
    if len(vertices) != 4:
        return False

    # 全てのKPと時間値を抽出
    kps = sorted(set(v[0] for v in vertices))
    times = sorted(set(v[1] for v in vertices))

    # 四角形：正確に2つの固有KP値と2つの固有時間値
    # 各角にそれぞれ1つずつ
    if len(kps) == 2 and len(times) == 2:
        # 4つ全ての角が存在することを確認
        expected_corners = {(kps[0], times[0]), (kps[0], times[1]),
                          (kps[1], times[0]), (kps[1], times[1])}
        actual_corners = {(v[0], v[1]) for v in vertices}
        return expected_corners == actual_corners

    return False


def _convert_rectangle_to_triangle(vertices: List[List[float]], direction: str) -> List[List[float]]:
    """
    Convert a rectangle to a triangle by merging the tip-side points.

    For 下 (downbound): tip points LEFT, so merge left two points
    For 上 (upbound): tip points RIGHT, so merge right two points

    Args:
        vertices: 4 vertices of the rectangle
        direction: "上" or "下"

    Returns:
        3 vertices of the triangle
    """
    # 固有KP値を取得（ソート済み）
    kps = sorted(set(v[0] for v in vertices))
    kp_left = kps[0]   # smaller KP
    kp_right = kps[1]  # larger KP

    # KP位置で頂点を分離
    left_vertices = [v for v in vertices if v[0] == kp_left]
    right_vertices = [v for v in vertices if v[0] == kp_right]

    # 時間でソート
    left_vertices = sorted(left_vertices, key=lambda v: v[1])
    right_vertices = sorted(right_vertices, key=lambda v: v[1])

    if direction == "下":
        # 下り：先端は左を指す（小さいKP）
        # 左の2点を1つの先端点にマージ
        # 右の2点を垂直ボトルネック辺として保持
        tip_time = (left_vertices[0][1] + left_vertices[1][1]) / 2
        tip_point = [kp_left, tip_time]

        # 返り値: [bottleneck_early, bottleneck_late, tip]
        return [right_vertices[0], right_vertices[1], tip_point]

    else:  # direction == "上"
        # 上り：先端は右を指す（大きいKP）
        # 右の2点を1つの先端点にマージ
        # 左の2点を垂直ボトルネック辺として保持
        tip_time = (right_vertices[0][1] + right_vertices[1][1]) / 2
        tip_point = [kp_right, tip_time]

        # 返り値: [bottleneck_early, bottleneck_late, tip]
        return [left_vertices[0], left_vertices[1], tip_point]


def _fix_irregular_trapezoid(vertices: List[List[float]], direction: str) -> Tuple[List[List[float]], bool]:
    """
    Fix irregular trapezoid where upper base time range exceeds lower base time range.

    A trapezoid has two vertical edges:
    - Lower base (longer vertical edge): has longer time span
    - Upper base (shorter vertical edge): has shorter time span

    The upper base time projection must fall within lower base time range.
    If not, center the upper base within the lower base range while keeping its length.

    Args:
        vertices: 4 vertices of the trapezoid
        direction: "上" or "下" (affects which side is bottleneck)

    Returns:
        Tuple of (fixed_vertices, was_fixed)
    """
    if len(vertices) != 4:
        return vertices, False

    # 固有KP値を取得
    kps = sorted(set(v[0] for v in vertices))

    if len(kps) != 2:
        # 2つの垂直辺を持つ標準的な台形ではない
        return vertices, False

    kp_left = kps[0]
    kp_right = kps[1]

    # KPで頂点を分離
    left_vertices = sorted([v for v in vertices if v[0] == kp_left], key=lambda v: v[1])
    right_vertices = sorted([v for v in vertices if v[0] == kp_right], key=lambda v: v[1])

    if len(left_vertices) != 2 or len(right_vertices) != 2:
        return vertices, False

    # 各垂直辺の時間スパンを計算
    left_time_span = left_vertices[1][1] - left_vertices[0][1]
    right_time_span = right_vertices[1][1] - right_vertices[0][1]

    # どちらが上端（短い）か下端（長いか）を判定
    if left_time_span >= right_time_span:
        # 左辺は下端（長い）、右辺は上端（短い）
        lower_base_vertices = left_vertices
        upper_base_vertices = right_vertices
        upper_base_kp = kp_right
        lower_base_kp = kp_left
    else:
        # 右辺は下端（長い）、左辺は上端（短い）
        lower_base_vertices = right_vertices
        upper_base_vertices = left_vertices
        upper_base_kp = kp_left
        lower_base_kp = kp_right

    #  시간 범위 가져오기
    lower_t_start = lower_base_vertices[0][1]
    lower_t_end = lower_base_vertices[1][1]
    upper_t_start = upper_base_vertices[0][1]
    upper_t_end = upper_base_vertices[1][1]

    # 上端が下端の範囲を超えているかチェック
    needs_fix = False

    # ケース1：上端が下端より前に開始
    if upper_t_start < lower_t_start:
        needs_fix = True

    # ケース2：上端が下端より後に終了
    if upper_t_end > lower_t_end:
        needs_fix = True

    if not needs_fix:
        return vertices, False

    # 修正：下端の範囲内に上端を中央配置
    upper_span = upper_t_end - upper_t_start
    lower_span = lower_t_end - lower_t_start

    if upper_span >= lower_span:
        # 上端が下端より長いか同じ
        # 少し短くして中央に配置
        upper_span = lower_span * 0.8  # 80% of lower span

    # 新しい上端時間を計算（下端内に中央配置）
    lower_center = (lower_t_start + lower_t_end) / 2
    new_upper_t_start = lower_center - upper_span / 2
    new_upper_t_end = lower_center + upper_span / 2

    # 範囲内であることを確認
    new_upper_t_start = max(new_upper_t_start, lower_t_start)
    new_upper_t_end = min(new_upper_t_end, lower_t_end)

    # 頂点を再構築
    new_upper_base = [[upper_base_kp, new_upper_t_start], [upper_base_kp, new_upper_t_end]]

    # 完全な頂点リストを再構築（反時計回りを維持）
    if upper_base_kp == kp_right:
        # 上端は右、下端は左
        fixed_vertices = [
            [lower_base_kp, lower_t_start],  # bottom-left
            [lower_base_kp, lower_t_end],    # top-left
            [upper_base_kp, new_upper_t_end],  # top-right
            [upper_base_kp, new_upper_t_start] # bottom-right
        ]
    else:
        # 上端は左、下端は右
        fixed_vertices = [
            [upper_base_kp, new_upper_t_start],  # bottom-left
            [upper_base_kp, new_upper_t_end],    # top-left
            [lower_base_kp, lower_t_end],        # top-right
            [lower_base_kp, lower_t_start]       # bottom-right
        ]

    return fixed_vertices, True


def validate_triangle_direction(vertices: List[List[float]], direction: str) -> Tuple[bool, str]:
    """
    Validate that a triangle has correct direction.

    For 上 (upbound): tip should point RIGHT (larger KP)
    For 下 (downbound): tip should point LEFT (smaller KP)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(vertices) != 3:
        return False, f"Expected 3 vertices, got {len(vertices)}"

    # 垂直辺を見つける（同じKPの2点）
    # と先端（異なるKPの単一点）
    kp_counts = {}
    for v in vertices:
        kp = v[0]
        kp_counts[kp] = kp_counts.get(kp, 0) + 1

    bottleneck_kp = None
    tip_kp = None

    for kp, count in kp_counts.items():
        if count == 2:
            bottleneck_kp = kp
        elif count == 1:
            tip_kp = kp

    if bottleneck_kp is None or tip_kp is None:
        return False, "Could not identify bottleneck edge and tip"

    if direction == "上":
        # 上り：先端は右を指す（tip_kp > bottleneck_kp）
        if tip_kp > bottleneck_kp:
            return True, ""
        else:
            return False, f"上り三角形の先端は大きいKPにあるべき。bottleneck={bottleneck_kp}, tip={tip_kp}"
    else:
        # 下り：先端は左を指す（tip_kp < bottleneck_kp）
        if tip_kp < bottleneck_kp:
            return True, ""
        else:
            return False, f"下り三角形の先端は小さいKPにあるべき。bottleneck={bottleneck_kp}, tip={tip_kp}"
