"""
Triangle visualization utilities for the multi-agent traffic prediction system.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image
from typing import List, Dict, Tuple, Optional
import seaborn as sns

from ..core.data_structures import Triangle


class TriangleVisualizer:
    """三角形可视化工具"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_triangle_visualization(self, triangles: List[Triangle], config: Dict = None) -> Image.Image:
        """生成三角形可视化图像"""
        if not triangles:
            return self._create_empty_plot()

        config = config or {}

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 设置坐标轴
        self._setup_axes(ax, triangles)

        # 绘制三角形
        self._draw_triangles(ax, triangles, config)

        # 添加图例和标注
        self._add_legend_and_annotations(ax, triangles)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 保存为图像
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)

        return image

    def _setup_axes(self, ax: plt.Axes, triangles: List[Triangle]) -> None:
        """设置坐标轴"""
        # 计算坐标范围
        x_min = min(t.kp_start for t in triangles) - 2
        x_max = max(t.kp_end for t in triangles) + 2
        y_min = min(t.time_start for t in triangles) - 30
        y_max = max(t.time_peak for t in triangles) + 30

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 设置标签
        ax.set_xlabel('位置 (KP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('时间 (分钟，从0:00开始)', fontsize=12, fontweight='bold')
        ax.set_title('历史拥堵三角形分布', fontsize=14, fontweight='bold', pad=20)

        # 设置时间刻度标签（转换为时:分格式）
        y_ticks = np.arange(int(y_min), int(y_max), 60)  # 每小时一个刻度
        y_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    def _draw_triangles(self, ax: plt.Axes, triangles: List[Triangle], config: Dict) -> None:
        """绘制三角形"""
        # 按年份和方向分组，使用不同颜色
        triangle_groups = self._group_triangles(triangles)
        color_palette = sns.color_palette("Set2", len(triangle_groups))

        for i, (group_name, group_triangles) in enumerate(triangle_groups.items()):
            color = color_palette[i]

            for j, triangle in enumerate(group_triangles):
                self._draw_single_triangle(ax, triangle, color, alpha=0.6)

                # 添加三角形标注
                if config.get('show_labels', True):
                    self._add_triangle_label(ax, triangle, f"{group_name[:4]}-{j+1}")

    def _draw_single_triangle(self, ax: plt.Axes, triangle: Triangle, color, alpha: float = 0.6) -> None:
        """绘制单个三角形"""
        vertices = np.array(triangle.vertices + [triangle.vertices[0]])  # 闭合三角形

        # 绘制边界
        ax.plot(vertices[:, 0], vertices[:, 1],
                color=color, alpha=alpha + 0.2, linewidth=2)

        # 填充三角形
        triangle_patch = patches.Polygon(triangle.vertices,
                                       facecolor=color, alpha=alpha,
                                       edgecolor=color, linewidth=1.5)
        ax.add_patch(triangle_patch)

    def _add_triangle_label(self, ax: plt.Axes, triangle: Triangle, label: str) -> None:
        """添加三角形标注"""
        ax.annotate(label, triangle.center,
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontweight='bold')

    def _group_triangles(self, triangles: List[Triangle]) -> Dict[str, List[Triangle]]:
        """按年份和方向分组三角形"""
        groups = {}

        for triangle in triangles:
            # 从source_event获取年份
            year = triangle.source_event.date.year
            direction = triangle.direction
            key = f"{year}_{direction}"

            if key not in groups:
                groups[key] = []
            groups[key].append(triangle)

        return groups

    def _add_legend_and_annotations(self, ax: plt.Axes, triangles: List[Triangle]) -> None:
        """添加图例和标注"""
        # 添加统计信息文本框
        stats_text = self._generate_stats_text(triangles)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                facecolor='lightblue', alpha=0.8), fontsize=10)

        # 添加方向图例
        directions = list(set(t.direction for t in triangles))
        for i, direction in enumerate(directions):
            ax.text(0.98, 0.98 - i * 0.05, f"方向: {direction}",
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

    def _generate_stats_text(self, triangles: List[Triangle]) -> str:
        """生成统计信息文本"""
        if not triangles:
            return "无数据"

        avg_severity = np.mean([t.severity for t in triangles])
        avg_duration = np.mean([t.duration for t in triangles])

        return (f"三角形数量: {len(triangles)}\n"
               f"平均严重程度: {avg_severity:.2f}\n"
               f"平均持续时间: {avg_duration:.0f}分钟")

    def _create_empty_plot(self) -> Image.Image:
        """创建空白图像"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.text(0.5, 0.5, '无可视化数据', ha='center', va='center',
                fontsize=16, transform=ax.transAxes)
        ax.set_title('历史拥堵三角形分布', fontsize=14, fontweight='bold')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)

        return image

    def save_visualization(self, triangles: List[Triangle], output_path: str, config: Dict = None) -> None:
        """保存可视化图像到文件"""
        image = self.generate_triangle_visualization(triangles, config)
        image.save(output_path, 'PNG', dpi=(self.dpi, self.dpi))

    def create_comparison_visualization(self,
                                     historical_triangles: List[Triangle],
                                     predicted_triangles: List[Triangle] = None) -> Image.Image:
        """创建历史与预测对比可视化"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        all_triangles = historical_triangles.copy()
        if predicted_triangles:
            all_triangles.extend(predicted_triangles)

        if not all_triangles:
            return self._create_empty_plot()

        self._setup_axes(ax, all_triangles)

        # 绘制历史三角形（半透明蓝色）
        for triangle in historical_triangles:
            self._draw_single_triangle(ax, triangle, 'blue', alpha=0.3)

        # 绘制预测三角形（红色高亮）
        if predicted_triangles:
            for i, triangle in enumerate(predicted_triangles):
                self._draw_single_triangle(ax, triangle, 'red', alpha=0.8)
                self._add_triangle_label(ax, triangle, f"预测{i+1}")

        # 添加图例
        if predicted_triangles:
            legend_elements = [
                patches.Patch(color='blue', alpha=0.3, label='历史数据'),
                patches.Patch(color='red', alpha=0.8, label='预测结果')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title('拥堵预测对比图', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)

        return image


def generate_triangle_visualization(triangles: List[Triangle], config: Dict = None) -> Image.Image:
    """便捷函数：生成三角形可视化图像"""
    visualizer = TriangleVisualizer()
    return visualizer.generate_triangle_visualization(triangles, config)


def save_triangle_visualization(triangles: List[Triangle], output_path: str, config: Dict = None) -> None:
    """便捷函数：保存三角形可视化到文件"""
    visualizer = TriangleVisualizer()
    visualizer.save_visualization(triangles, output_path, config)