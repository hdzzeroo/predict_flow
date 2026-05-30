"""
批量评估脚本
读取配置文件，自动运行多个预测任务并评估
"""

import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

from evaluation import Evaluator
from implementation import compiled_agent
from functions import process_direction_aware_traffic_data


class BatchEvaluator:
    """批量评估管理器"""

    def __init__(self, config_path: str):
        """
        初始化批量评估器

        Args:
            config_path: 配置文件路径 (JSON)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ 已加载配置文件: {self.config_path}")
            print(f"  共 {len(config.get('tasks', []))} 个评估任务")
            return config
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            sys.exit(1)

    def run_all_tasks(self):
        """运行所有评估任务"""
        tasks = self.config.get('tasks', [])

        if not tasks:
            print("⚠️ 配置文件中没有任务")
            return

        print(f"\n{'='*70}")
        print(f"开始批量评估 - 共 {len(tasks)} 个任务")
        print(f"{'='*70}\n")

        for i, task in enumerate(tasks, 1):
            print(f"\n{'#'*70}")
            print(f"# 任务 {i}/{len(tasks)}: {task.get('task_id', 'unknown')}")
            print(f"{'#'*70}\n")

            try:
                result = self._run_single_task(task)
                self.results.append({
                    "task_id": task.get('task_id'),
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                print(f"❌ 任务失败: {e}")
                import traceback
                traceback.print_exc()
                self.results.append({
                    "task_id": task.get('task_id'),
                    "status": "failed",
                    "error": str(e)
                })

        # 生成汇总报告
        self._generate_summary_report()

    def _run_single_task(self, task: Dict) -> Dict:
        """
        运行单个评估任务

        Args:
            task: 任务配置
                {
                    "task_id": "...",
                    "route": "関越道",
                    "direction": "上",
                    "target_date": "2025-05-05",
                    "train_years": [2014, 2018, 2024],
                    "ground_truth_file": "data/..."
                }

        Returns:
            评估结果
        """
        task_id = task.get('task_id', 'unknown')
        route = task.get('route')
        direction = task.get('direction')
        target_date = task.get('target_date')
        train_years = task.get('train_years', [2014, 2018, 2024])
        gt_file = task.get('ground_truth_file')

        print(f"📋 任务信息:")
        print(f"  - 路线: {route}")
        print(f"  - 方向: {direction}")
        print(f"  - 目标日期: {target_date}")
        print(f"  - 训练年份: {train_years}")
        print(f"  - 真实数据: {gt_file}")

        # 步骤1: 构造用户输入
        user_input = self._construct_user_input(task)
        print(f"\n🤖 用户输入: {user_input}")

        # 步骤2: 运行预测workflow
        print(f"\n{'─'*70}")
        print("步骤1: 运行预测workflow")
        print(f"{'─'*70}")

        predictions = self._run_prediction(user_input, direction)

        if not predictions:
            raise ValueError("预测workflow未返回结果")

        # 步骤3: 加载真实数据
        print(f"\n{'─'*70}")
        print("步骤2: 加载真实数据")
        print(f"{'─'*70}")

        ground_truth = self._load_ground_truth(gt_file, direction)

        if not ground_truth:
            raise ValueError("真实数据加载失败")

        # 步骤4: 评估
        print(f"\n{'─'*70}")
        print("步骤3: 执行评估")
        print(f"{'─'*70}")

        evaluation_result = self._evaluate(predictions, ground_truth, task_id, route, direction)

        # 步骤5: 保存结果
        self._save_task_result(task_id, evaluation_result)

        return evaluation_result

    def _construct_user_input(self, task: Dict) -> str:
        """构造workflow的用户输入"""
        route = task.get('route', '関越道')
        target_date = task.get('target_date', '2025-05-05')
        direction = task.get('direction', '上')

        # 提取月日
        try:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            month = date_obj.month
            day = date_obj.day
            year = date_obj.year
        except:
            month, day, year = 5, 5, 2025

        return f"预测{route}{direction}行{year}年{month}月{day}日的拥堵情况"

    def _run_prediction(self, user_input: str, direction: str) -> Dict:
        """
        运行预测workflow

        Returns:
            预测结果 {"上": [...], "下": [...]}
        """
        try:
            # 运行workflow
            result = compiled_agent.invoke({
                "user_input": user_input
            })

            # 提取预测结果
            llm_analysis = result.get("llm_analysis", {})

            if not llm_analysis:
                print("⚠️ workflow未返回llm_analysis")
                return {}

            print(f"✓ 预测完成，共 {len(llm_analysis)} 个方向")
            for d, data in llm_analysis.items():
                hotspots = data.get('hotspots', [])
                print(f"  - {d}行: {len(hotspots)} 个热点")

            return llm_analysis

        except Exception as e:
            print(f"❌ 预测workflow执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_ground_truth(self, csv_file: str, direction: str) -> Dict:
        """
        加载真实数据（2025年CSV）

        Returns:
            真实拥堵数据 {"上": [...], "下": [...]}
        """
        try:
            if not os.path.exists(csv_file):
                print(f"❌ 真实数据文件不存在: {csv_file}")
                return {}

            print(f"📂 加载真实数据: {csv_file}")

            # 处理数据，生成三角形
            triangles = self._process_ground_truth_data(csv_file, direction)

            print(f"✓ 真实数据加载完成: {len(triangles)} 个三角形")

            # 返回按方向分组的数据
            return {direction: triangles}

        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _process_ground_truth_data(self, csv_file: str, direction: str) -> List[Dict]:
        """
        处理真实数据CSV，生成三角形列表

        Args:
            csv_file: CSV文件路径
            direction: 方向

        Returns:
            三角形列表 [{"vertices": [...], ...}, ...]
        """
        try:
            # 调用现有的数据处理函数
            triangles_data, _ = process_direction_aware_traffic_data(
                file_path=csv_file,
                direction=direction,
                output_dir="output/evaluation/temp"
            )

            # 转换为标准格式
            triangles = []
            for t in triangles_data:
                vertices = t.get('vertices', [])
                if len(vertices) >= 3:
                    triangles.append({
                        'vertices': vertices,
                        'kp_start': t.get('kp_start'),
                        'kp_end': t.get('kp_end'),
                        'time_start': t.get('time_start'),
                        'time_end': t.get('time_end'),
                        'time_peak': t.get('time_peak'),
                        'area': t.get('area')
                    })

            return triangles

        except Exception as e:
            print(f"⚠️ 处理真实数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _evaluate(
        self,
        predictions: Dict,
        ground_truth: Dict,
        task_id: str,
        route: str = "関越道",
        direction: str = "下"
    ) -> Dict:
        """
        执行评估

        Args:
            predictions: 预测结果
            ground_truth: 真实数据
            task_id: 任务ID
            route: 道路类型
            direction: 方向

        Returns:
            评估结果
        """
        # 转换预测数据格式
        pred_formatted = {}
        for dir_key, data in predictions.items():
            if isinstance(data, dict) and 'hotspots' in data:
                pred_formatted[dir_key] = data['hotspots']
            else:
                pred_formatted[dir_key] = data

        # 创建评估器（使用KP区间栅格）
        evaluator = Evaluator(road_type=route, direction=direction, time_step_minutes=60)

        # 执行评估
        results = evaluator.evaluate_all_directions(pred_formatted, ground_truth)

        # 打印汇总
        evaluator.print_summary_report(results)

        return results

    def _save_task_result(self, task_id: str, result: Dict):
        """保存单个任务的评估结果"""
        output_dir = "output/evaluation"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{task_id}_evaluation.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 任务结果已保存: {output_path}")

    def _generate_summary_report(self):
        """生成所有任务的汇总报告"""
        print(f"\n{'='*70}")
        print("批量评估汇总报告")
        print(f"{'='*70}\n")

        # 统计成功/失败
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        failed_count = len(self.results) - success_count

        print(f"总任务数: {len(self.results)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}\n")

        # 汇总所有成功任务的指标
        if success_count > 0:
            print(f"{'─'*70}")
            print("平均指标:")
            print(f"{'─'*70}\n")

            all_metrics = []
            for r in self.results:
                if r['status'] == 'success' and 'average' in r['result']:
                    all_metrics.append(r['result']['average'])

            if all_metrics:
                avg_polygon_iou = sum(m['polygon_iou'] for m in all_metrics) / len(all_metrics)
                avg_f1 = sum(m['grid_metrics']['f1_score'] for m in all_metrics) / len(all_metrics)
                avg_spatial = sum(m['spatial_iou'] for m in all_metrics) / len(all_metrics)
                avg_temporal = sum(m['temporal_iou'] for m in all_metrics) / len(all_metrics)

                print(f"  全局多边形IoU:    {avg_polygon_iou:.4f}")
                print(f"  栅格F1-Score:      {avg_f1:.4f}")
                print(f"  空间IoU:           {avg_spatial:.4f}")
                print(f"  时间IoU:           {avg_temporal:.4f}")

        # 逐任务详细结果
        print(f"\n{'─'*70}")
        print("逐任务结果:")
        print(f"{'─'*70}\n")

        for r in self.results:
            task_id = r['task_id']
            status = r['status']

            if status == 'success':
                result = r['result']
                avg = result.get('average', {})
                print(f"✓ {task_id}")
                print(f"    Polygon IoU: {avg.get('polygon_iou', 0):.4f} | "
                      f"F1: {avg.get('grid_metrics', {}).get('f1_score', 0):.4f} | "
                      f"Spatial: {avg.get('spatial_iou', 0):.4f} | "
                      f"Temporal: {avg.get('temporal_iou', 0):.4f}")
            else:
                print(f"✗ {task_id}: {r.get('error', 'unknown error')}")

        # 保存汇总结果
        output_path = "output/evaluation/batch_summary.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        summary = {
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(self.results),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 汇总报告已保存: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量评估预测结果')
    parser.add_argument(
        '--config',
        type=str,
        default='evaluation_config.json',
        help='配置文件路径 (默认: evaluation_config.json)'
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print(f"\n请创建配置文件，示例格式:")
        print("""
{
  "tasks": [
    {
      "task_id": "sekietsu_2025_05_05_down",
      "route": "関越道",
      "direction": "下",
      "target_date": "2025-05-05",
      "train_years": [2014, 2018, 2024],
      "ground_truth_file": "data/processed_data/関越道_下_2025_05-05.csv"
    }
  ]
}
        """)
        sys.exit(1)

    # 运行批量评估
    batch_evaluator = BatchEvaluator(args.config)
    batch_evaluator.run_all_tasks()

    print(f"\n{'='*70}")
    print("批量评估完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
