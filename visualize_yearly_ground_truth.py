"""
Visualize Ground Truth congestion triangles for 関越道 上 across three years
(2014, 2019, 2024), mirroring the leftmost panel style of
version2/output/enhanced_eval/enhanced_eval_関越道_上_*.png.
"""

from datetime import datetime, time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Japanese font for titles
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Songti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

DATA_CSV = Path('/Users/huangdizhi/Desktop/projects/predict_workflow/data/processed_data/all_data.csv')
OUT_PNG = Path('/Users/huangdizhi/Desktop/projects/predict_workflow/version2/output/enhanced_eval/ground_truth_0505_関越道_上_2014_2019_2024.png')

ROAD = '関越道'
DIRECTION = '上'
YEARS = [2014, 2019, 2024]
TARGET_MONTH = 5
TARGET_DAY = 5


def parse_hhmm(val):
    """Parse HH:MM[:SS] or integer-like time into datetime.time."""
    if pd.isnull(val):
        return None
    if isinstance(val, time):
        return val
    if isinstance(val, datetime):
        return val.time()
    s = str(val).strip()
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    try:
        n = int(float(s))
        ns = str(n)
        if len(ns) == 3:
            h, m = int(ns[0]), int(ns[1:])
        elif len(ns) == 4:
            h, m = int(ns[:2]), int(ns[2:])
        else:
            h, m = n, 0
        if 0 <= h < 24 and 0 <= m < 60:
            return time(h, m)
    except Exception:
        pass
    return None


def t2min(t):
    return t.hour * 60 + t.minute


def event_to_polygon(row, direction='上'):
    """Convert one row into a shapely Polygon (triangle or quadrilateral).

    Vertices follow the same convention as workflow.functions.generate_polygons:
      v1 = (start_kp, start_time)                         # jam onset at base KP
      v2 = (start_kp +/- start_jam_length, start_time)    # base spread at onset
      v3 = (start_kp +/- peak_length,  peak_time)         # peak extent
      v4 = (start_kp, end_time)                           # jam ends at base KP
    """
    t_start = parse_hhmm(row['発生時刻'])
    t_peak = parse_hhmm(row['ピーク時刻'])
    if t_start is None or t_peak is None:
        return None

    start_kp = row['発生Ｋｐ']
    start_jam_len = row['発生時渋滞長'] or 0.0
    peak_len = row['ピーク長'] or 0.0
    duration = row['渋滞時間'] or 0.0

    if pd.isnull(start_kp) or pd.isnull(peak_len) or pd.isnull(duration):
        return None

    start_time = t2min(t_start)
    peak_time = t2min(t_peak)
    end_time = start_time + float(duration)

    # Keep peak_time within [start_time, end_time] to avoid degenerate shapes
    if peak_time < start_time:
        peak_time = start_time
    if peak_time > end_time:
        peak_time = end_time

    sign = 1.0 if direction == '上' else -1.0
    v1 = (start_kp, start_time)
    v2 = (start_kp + sign * start_jam_len, start_time)
    v3 = (start_kp + sign * peak_len, peak_time)
    v4 = (start_kp, end_time)

    if start_jam_len == 0:
        vertices = [v1, v3, v4]
    else:
        vertices = [v1, v2, v3, v4]

    try:
        poly = Polygon(vertices)
        if not poly.is_valid or poly.area == 0:
            return None
        return poly
    except Exception:
        return None


def build_polygons_for_year(df, year):
    sub = df[(df['道路番号'] == ROAD)
             & (df['上下'] == DIRECTION)
             & (df['year'] == year)
             & (df['month'] == TARGET_MONTH)
             & (df['day'] == TARGET_DAY)]
    polys = []
    for _, row in sub.iterrows():
        p = event_to_polygon(row, DIRECTION)
        if p is not None:
            polys.append(p)
    return polys


def main():
    df = pd.read_csv(DATA_CSV)
    dt = pd.to_datetime(df['date'])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day

    polys_by_year = {y: build_polygons_for_year(df, y) for y in YEARS}

    # Unified axis limits across panels for fair visual comparison.
    # Fall back to full-day / full-highway range when data is sparse.
    all_polys = [p for ps in polys_by_year.values() for p in ps]
    if all_polys:
        bounds = [p.bounds for p in all_polys]
        kp_min = min(b[0] for b in bounds)
        kp_max = max(b[2] for b in bounds)
        t_min = min(b[1] for b in bounds)
        t_max = max(b[3] for b in bounds)
    else:
        kp_min, kp_max = 0, 140
        t_min, t_max = 0, 1440
    kp_pad = max((kp_max - kp_min) * 0.05, 5)
    t_pad = max((t_max - t_min) * 0.05, 30)

    fig, axes = plt.subplots(1, len(YEARS), figsize=(18, 6), sharey=True)

    for ax, year in zip(axes, YEARS):
        polys = polys_by_year[year]
        ax.set_title(f'{ROAD} {DIRECTION} - Ground Truth ({year}-{TARGET_MONTH:02d}-{TARGET_DAY:02d})\n'
                     f'真实拥堵事件: {len(polys)}个',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('KP (km)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, facecolor='lightblue',
                    edgecolor='blue', linewidth=1.5)

        if polys:
            total_area = unary_union(polys).area
        else:
            total_area = 0.0
        stats = f'Events: {len(polys)}\nCoverage: {total_area:.0f} km·min'
        ax.text(0.02, 0.98, stats,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue',
                          alpha=0.8, edgecolor='blue'))

        ax.set_xlim(kp_min - kp_pad, kp_max + kp_pad)
        ax.set_ylim(t_min - t_pad, t_max + t_pad)

    axes[0].set_ylabel('Time (minutes from midnight)', fontsize=11)

    legend = [mpatches.Patch(facecolor='lightblue', edgecolor='blue',
                             alpha=0.6, label='Ground Truth congestion event (真实拥堵)')]
    fig.legend(handles=legend, loc='lower center', ncol=1,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{ROAD} {DIRECTION} — Ground Truth on {TARGET_MONTH:02d}/{TARGET_DAY:02d} (2014 vs 2019 vs 2024)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_PNG}')

    for y in YEARS:
        print(f'  {y}: {len(polys_by_year[y])} polygons')


if __name__ == '__main__':
    main()
