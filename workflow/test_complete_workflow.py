#!/usr/bin/env python3
"""
評価付きの完全なワークフローテストスクリプト
chatbotノードからreportノードまでのエンドツーエンドテスト、オプションで評価付き
"""

import os
import sys
import json
import argparse
sys.path.append(os.path.dirname(__file__))

# 全てのノード関数をインポート
from implementation import chatbot, visualization, analyze_with_llm, report
from config import config
from evaluation import Evaluator
from functions import process_direction_aware_traffic_data


def auto_detect_ground_truth_file(state: dict) -> dict:
    """
    chatbotノードから抽出された情報に基づいてground truthファイルを自動検出（双方向サポート）

    重要ロジック:
    - 履歴データ（2014, 2019, 2024など）を使用してパターン学習
    - 予測対象年（2025など）の交通拥挤状況を予測
    - Ground Truthは予測対象年の実データであるべき

    例：ユーザーが"2025/5/4関越"と入力
      -> 学習データ: 2014, 2019, 2024年5月4日
      -> 予測対象: 2025年5月4日
      -> Ground Truth: 2025年5月4日の実データ

    Args:
        state: workflow stateにはroute, target_year, target_month, target_day, direction_dataが含まれる

    Returns:
        dict: {"上": file_path, "下": file_path} または {}（検出失敗時）
    """
    # 必要情報を抽出
    route = state.get('route', '')
    target_year = state.get('target_year')  # これは予測対象年（2025など）
    month = state.get('target_month')
    day = state.get('target_day')

    # 基本情報をチェック
    if not all([route, target_year, month, day]):
        print(f"⚠️  ground truthを自動検出できません: 必要な情報が不足")
        print(f"   Route: {route}, Year: {target_year}, Month: {month}, Day: {day}")
        return {}

    # 多方向データがあるかチェック
    direction_data = state.get('direction_data', {})

    # direction_dataがない場合、単一directionを試す
    if not direction_data:
        single_direction = state.get('direction', '')
        if single_direction:
            direction_data = {single_direction: {}}
        else:
            print(f"⚠️  方向情報を検出できません")
            return {}

    # 完全パスを構築
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    detected_files = {}

    print(f"   {len(direction_data)} 方向を検出: {list(direction_data.keys())}")
    print(f"   🎯 予測対象年: {target_year}（該年の実データをGround Truthとして使用）")

    # 各方向のground truthファイルを検出
    for direction in direction_data.keys():
        # 予測対象年を使用してGround Truthファイル名を構築
        # 例: 関越道_上_2025_05-04.csv
        filename = f"{route}_{direction}_{target_year}_{month:02d}-{day:02d}.csv"
        full_path = os.path.join(base_dir, filename)

        # ファイルが存在するかチェック
        if os.path.exists(full_path):
            detected_files[direction] = full_path
            print(f"   ✅ [{direction}方向] 検出: {filename}")
        else:
            print(f"   ℹ️  [{direction}方向] 未検出: {filename}")

    if detected_files:
        print(f"✅ {len(detected_files)} 方向のground truthファイルを正常に検出")
    else:
        print(f"❌ ground truthファイルが検出されませんでした")

    return detected_files


def load_ground_truth(ground_truth_file: str, direction: str) -> dict:
    """
    実データをground truthとして読み込み

    Args:
        ground_truth_file: 実データCSVファイルパス
        direction: 方向 ("上" または "下")

    Returns:
        フォーマット済みground truthデータ {"上": [...], "下": [...]}
    """
    if not os.path.exists(ground_truth_file):
        print(f"⚠️  Ground truthファイルが存在しません: {ground_truth_file}")
        return {}

    print(f"\n📂 Ground Truthデータを読み込み: {ground_truth_file}")

    try:
        # 既存の関数を使用して実データを処理
        triangles_data, _ = process_direction_aware_traffic_data(
            file_path=ground_truth_file,
            direction=direction,
            output_dir="output/temp"
        )

        # 標準フォーマットに変換
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

        print(f"✓ Ground Truth読み込み完了: {len(triangles)} 件の交通拥挤イベント")
        return {direction: triangles}

    except Exception as e:
        print(f"❌ Ground Truthの読み込みに失敗: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_evaluation(predictions: dict, ground_truth: dict, route: str, direction: str):
    """
    評価を実行

    Args:
        predictions: 予測結果 {"上": [...], "下": [...]}
        ground_truth: 実データ {"上": [...], "下": [...]}
        route: 道路名
        direction: 方向
    """
    print("\n" + "="*70)
    print("📊 予測結果の評価を開始")
    print("="*70)

    # 转换预测数据格式
    pred_formatted = {}
    for dir_key, data in predictions.items():
        if isinstance(data, dict) and 'hotspots' in data:
            pred_formatted[dir_key] = data['hotspots']
        else:
            pred_formatted[dir_key] = data

    # 创建评估器
    evaluator = Evaluator(
        road_type=route,
        direction=direction,
        time_step_minutes=60
    )

    # 执行评估
    results = evaluator.evaluate_all_directions(pred_formatted, ground_truth)

    # 打印汇总报告
    evaluator.print_summary_report(results)

    # 保存评估结果
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output = os.path.join(output_dir, f"evaluation_{route}_{direction}_{timestamp}.json")
    evaluator.save_results_to_json(results, eval_output)

    return results


def test_complete_workflow(user_input: str, ground_truth_file: str = None):
    """
    Test the complete workflow with optional evaluation

    Args:
        user_input: ユーザー入力クエリ
        ground_truth_file: オプションのground truthファイルパス（評価用）
    """
    print("🚀 Starting complete workflow test")
    print("=" * 80)
    print(f"📝 User input: {user_input}")
    print("-" * 80)

    # Initialize state
    state = {"user_input": user_input}

    try:
        # ===============================
        # Step 1: Chatbot node
        # ===============================
        print("\n1️⃣ Chatbot node - Parse user input")
        print("-" * 40)

        chatbot_result = chatbot(state)
        state.update(chatbot_result)

        print("✅ Chatbot node completed")
        print(f"   Main file path: {state.get('file_path', 'None')}")
        print(f"   Multiple file paths: {len(state.get('file_paths', []))} files")
        if state.get('file_paths'):
            for i, fp in enumerate(state.get('file_paths', []), 1):
                print(f"      {i}. {fp}")
        print(f"   Route: {state.get('route', 'None')}")
        print(f"   Time: {state.get('ts', 'None')}")

        # ===============================
        # Step 2: Visualization node
        # ===============================
        print("\n2️⃣ Visualization node - Generate triangle visualization")
        print("-" * 40)

        viz_result = visualization(state)
        state.update(viz_result)

        triangles = state.get('triangles', [])
        fig_path = state.get('fig_path', '')

        print("✅ Visualization node completed")
        print(f"   Generated triangles count: {len(triangles)}")
        print(f"   Image path: {fig_path}")
        print(f"   Image file exists: {os.path.exists(fig_path) if fig_path else False}")

        if triangles:
            # Analyze triangle sources
            source_stats = {}
            for triangle in triangles:
                source = triangle.get('source_file', 'unknown')
                source_stats[source] = source_stats.get(source, 0) + 1

            if len(source_stats) > 1:
                print("   📊 Multi-file source distribution:")
                for source, count in source_stats.items():
                    print(f"      {source}: {count} triangles")

        if not triangles:
            print("❌ No triangle data generated, cannot continue")
            return False

        # ===============================
        # ground truthファイルの自動検出（Visualizationの後、direction_dataが必要）
        # ===============================
        if ground_truth_file is None:
            print("\n🔍 ground truthファイルを自動検出...")
            print("-" * 40)
            auto_gt_file = auto_detect_ground_truth_file(state)
            if auto_gt_file:
                ground_truth_file = auto_gt_file
                print(f"   検出したファイルを使用して評価を実行")
            else:
                print(f"   ground truthファイルが検出されず、評価をスキップ")

        # ===============================
        # Step 3: Analyze with LLM node (clusterとdraw_hullsの代替)
        # ===============================
        print("\n3️⃣ Analyze with LLM node - LLM-based hotspot analysis")
        print("-" * 40)

        llm_result = analyze_with_llm(state)
        state.update(llm_result)

        llm_analysis = state.get('llm_analysis', {})

        print("✅ LLM analysis node completed")

        # 统计各方向的热点数量
        total_hotspots = 0
        for direction, analysis in llm_analysis.items():
            hotspots = analysis.get('hotspots', [])
            total_hotspots += len(hotspots)
            summary = analysis.get('summary', {})
            confidence = summary.get('analysis_confidence', 0)

            print(f"   {direction} direction: {len(hotspots)} hotspots (confidence: {confidence:.2f})")

            # 显示前3个热点
            for i, hotspot in enumerate(hotspots[:3]):
                kp_range = hotspot['kp_range']
                time_range = hotspot['time_range']
                print(f"      Hotspot {i+1}: KP {kp_range[0]:.1f}-{kp_range[1]:.1f}km, "
                      f"frequency: {hotspot['frequency']}, severity: {hotspot['severity']}")

        print(f"   Total hotspots across all directions: {total_hotspots}")

        # ===============================
        # Step 4: Report node
        # ===============================
        print("\n4️⃣ Report node - Generate final report")
        print("-" * 40)

        report_result = report(state)
        state.update(report_result)

        final_report = state.get('final_report', '')

        print("✅ Report node completed")
        print(f"   Report length: {len(final_report)} characters")

        # Show first few lines of the report
        if final_report:
            report_lines = final_report.split('\n')
            print("   📋 Report preview:")
            for line in report_lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"      {line}")
            if len(report_lines) > 10:
                print("      ...")

        # ===============================
        # Workflow completion summary
        # ===============================
        print("\n🎉 Complete workflow executed successfully!")
        print("=" * 80)

        print("📊 Final result statistics:")
        print(f"   Original file count: {len(state.get('file_paths', []))}")
        print(f"   Generated triangles count: {len(state.get('triangles', []))}")
        print(f"   LLM-identified hotspots count: {total_hotspots}")

        # 显示CSV输出路径
        csv_output_path = state.get('csv_output_path', '')
        if csv_output_path:
            print(f"   CSV prediction output: {csv_output_path}")

        print("\n📁 Generated files:")
        if fig_path and os.path.exists(fig_path):
            print(f"   🖼️ Triangle visualization: {fig_path}")

        # 显示所有图片路径
        fig_paths = state.get('fig_paths', [])
        if fig_paths and len(fig_paths) > 1:
            for i, path in enumerate(fig_paths[1:], 2):
                if os.path.exists(path):
                    print(f"   🖼️ Triangle visualization {i}: {path}")

        if csv_output_path and os.path.exists(csv_output_path):
            print(f"   📄 CSV prediction: {csv_output_path}")

        # Save final state
        save_workflow_state(state, user_input)

        # ===============================
        # Step 5: Evaluation (Optional)
        # ===============================
        if ground_truth_file:
            route = state.get('route', '関越道')

            # dict（多方向）かstr（単方向）かを判断
            if isinstance(ground_truth_file, dict):
                # 多方向評価
                print("\n" + "="*70)
                print("📊 多方向評価を開始")
                print("="*70)

                all_eval_results = {}

                for direction, gt_file in ground_truth_file.items():
                    print(f"\n▶️  评估 [{direction}] 方向")
                    print("-" * 40)

                    # 加载该方向的ground truth
                    ground_truth = load_ground_truth(gt_file, direction)

                    if ground_truth and llm_analysis:
                        # 运行评估
                        eval_results = run_evaluation(
                            predictions=llm_analysis,
                            ground_truth=ground_truth,
                            route=route,
                            direction=direction
                        )

                        all_eval_results[direction] = eval_results
                    else:
                        print(f"   ⚠️  [{direction}] 无法评估：数据为空")

                # 全方向の評価サマリーを表示
                if all_eval_results:
                    print("\n" + "="*70)
                    print("📈 多方向評価サマリー")
                    print("="*70)

                    for direction, eval_results in all_eval_results.items():
                        print(f"\n{'='*70}")
                        print(f"🔹 [{direction}] 方向の評価結果")
                        print(f"{'='*70}")

                        for dir_key, metrics in eval_results.items():
                            if dir_key == "average":
                                print(f"\n【全体平均】")
                            else:
                                print(f"\n【{dir_key}行】")

                            print(f"  ポリゴンIoU:      {metrics['polygon_iou']:.4f}")
                            print(f"  F1-Score:       {metrics['grid_metrics']['f1_score']:.4f}")
                            print(f"  Precision:      {metrics['grid_metrics']['precision']:.4f}")
                            print(f"  Recall:         {metrics['grid_metrics']['recall']:.4f}")
                else:
                    print("\n⚠️  全方向の評価に失敗")

            else:
                # 単方向評価（旧ロジックとの互換性）
                direction = state.get('direction', '下')

                # ground truthを読み込み
                ground_truth = load_ground_truth(ground_truth_file, direction)

                if ground_truth and llm_analysis:
                    # 評価を実行
                    eval_results = run_evaluation(
                        predictions=llm_analysis,
                        ground_truth=ground_truth,
                        route=route,
                        direction=direction
                    )

                    # 評価サマリーを表示
                    print("\n" + "="*70)
                    print("📈 Evaluation Summary")
                    print("="*70)

                    for dir_key, metrics in eval_results.items():
                        if dir_key == "average":
                            print(f"\n【全体平均】")
                        else:
                            print(f"\n【{dir_key}行】")

                        print(f"  ポリゴンIoU:      {metrics['polygon_iou']:.4f}")
                        print(f"  F1-Score:       {metrics['grid_metrics']['f1_score']:.4f}")
                        print(f"  Precision:      {metrics['grid_metrics']['precision']:.4f}")
                        print(f"  Recall:         {metrics['grid_metrics']['recall']:.4f}")
                else:
                    print("\n⚠️  評価を実行できません：Ground truthまたは予測結果が空です")
        else:
            print("\n💡 Ground Truthファイルが提供されず、評価をスキップ")
            print("   ヒント：プログラムはground truthファイルを自動検出します")

        return True

    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_workflow_state(state, user_input):
    """Save workflow state to file (完全な三角形のデータとLLM分析結果を含む)"""
    try:
        # 出力ディレクトリを作成
        os.makedirs("output", exist_ok=True)

        # 保存用の状態を準備（シリアライズ不可能なコンテンツを除外）
        save_state = {}
        for key, value in state.items():
            if key == 'triangles':
                # 🔥 完全な三角形の情報を保存（最初の10個に制限しない）
                save_state[key] = [
                    {
                        'id': t.get('id'),
                        'shape_type': t.get('shape_type', 'triangle'),
                        'vertices': t.get('vertices', []),  # 完全な頂点座標を含む
                        'center': t.get('center'),
                        'area': t.get('area'),
                        'width': t.get('width'),
                        'height': t.get('height'),
                        'kp_start': t.get('kp_start'),
                        'kp_end': t.get('kp_end'),
                        'kp_range': [t.get('kp_start'), t.get('kp_end')],
                        'time_start': t.get('time_start'),
                        'time_end': t.get('time_end'),
                        'time_peak': t.get('time_peak'),
                        'source_file': t.get('source_file', 'unknown')
                    }
                    for t in value  # 全三角形を保存、切り詰めなし
                ]
            elif key == 'direction_data':
                # 🔥 方向ごとにグループ化された完全データを保存
                save_state[key] = {}
                for direction, data in value.items():
                    save_state[key][direction] = {
                        'triangles': [
                            {
                                'id': t.get('id'),
                                'shape_type': t.get('shape_type', 'triangle'),
                                'vertices': t.get('vertices', []),
                                'center': t.get('center'),
                                'area': t.get('area'),
                                'kp_start': t.get('kp_start'),
                                'kp_end': t.get('kp_end'),
                                'time_start': t.get('time_start'),
                                'time_end': t.get('time_end'),
                                'time_peak': t.get('time_peak'),
                                'source_file': t.get('source_file', 'unknown')
                            }
                            for t in data.get('triangles', [])
                        ],
                        'fig_path': data.get('fig_path', ''),
                        'triangle_count': len(data.get('triangles', []))
                    }
            elif key == 'llm_analysis':
                # 🔥 LLM分析結果を保存
                save_state[key] = value  # LLM分析結果はすでにシリアライズ可能なdict
            elif key == 'hulls':
                # Save convex hull triangle information
                save_state[key] = [
                    {
                        'cluster_id': h.get('cluster_id'),
                        'cluster_size': h.get('cluster_size'),
                        'kp_range': h.get('kp_range'),
                        'area': h.get('area')
                    }
                    for h in value
                ]
            elif isinstance(value, (str, int, float, bool, list)) and key != 'cluster_analysis':
                save_state[key] = value

        # メタデータを追加
        llm_analysis = state.get('llm_analysis', {})
        total_hotspots = sum(len(analysis.get('hotspots', [])) for analysis in llm_analysis.values())

        save_state['_metadata'] = {
            'user_input': user_input,
            'workflow_version': 'version2',
            'execution_time': pd.Timestamp.now().isoformat(),
            'total_triangles': len(state.get('triangles', [])),
            'total_clusters': len(state.get('clusters', [])),
            'total_hulls': len(state.get('hulls', [])),
            'total_hotspots': total_hotspots,  # 新規：LLMが識別したホットスポット総数
            'directions_analyzed': list(state.get('direction_data', {}).keys())  # 新規：分析した方向
        }

        # Save to JSON file
        output_file = "output/workflow_state.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_state, f, ensure_ascii=False, indent=2)

        print(f"   📄 Workflow state saved: {output_file}")
        print(f"      - Total triangles saved: {len(state.get('triangles', []))}")
        print(f"      - Directions saved: {list(state.get('direction_data', {}).keys())}")
        print(f"      - LLM hotspots saved: {total_hotspots}")

    except Exception as e:
        print(f"   ⚠️ State saving failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_predefined_tests(ground_truth_file: str = None):
    """Run predefined test cases"""
    print("\n🧪 Running predefined test cases")

    test_cases = [
        "April 23rd Kan-Etsu Expressway traffic conditions",
        "Analyze Kan-Etsu Expressway 2024 data",
        "Tohoku Expressway spring traffic congestion situation",
        "Please analyze 2023 Chuo Expressway congestion patterns"
    ]

    success_count = 0

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 Test case {i}/{len(test_cases)}")
        print("=" * 80)

        success = test_complete_workflow(test_input, ground_truth_file)

        if success:
            success_count += 1
            print(f"✅ Test case {i} succeeded")
        else:
            print(f"❌ Test case {i} failed")

        if i < len(test_cases):
            print(f"\n{'='*80}")
            print("Waiting for next test case...")
            input("Press Enter to continue...")

    # Final statistics
    print(f"\n🏁 Predefined tests completed!")
    print(f"   Success: {success_count}/{len(test_cases)} test cases")
    print(f"   Success rate: {success_count/len(test_cases)*100:.1f}%")

    if success_count == len(test_cases):
        print("🎉 All predefined test cases executed successfully!")
    else:
        print("⚠️ Some predefined test cases failed, please check error messages.")


def run_interactive_test(ground_truth_file: str = None):
    """Run interactive test"""
    print("\n💬 Interactive test mode")
    print("You can input custom queries to test chatbot and complete workflow")
    print("Enter 'quit' or 'exit' to exit")
    print("-" * 60)

    test_count = 0
    success_count = 0

    while True:
        try:
            # Get user input
            print(f"\n📝 Please enter your query (test #{test_count + 1}):")
            user_input = input(">>> ").strip()

            # Check exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Check empty input
            if not user_input:
                print("⚠️ Please enter valid query content")
                continue

            test_count += 1
            print(f"\n🚀 Starting test with your input: {user_input}")
            print("=" * 80)

            # Execute test
            success = test_complete_workflow(user_input, ground_truth_file)

            if success:
                success_count += 1
                print(f"✅ Your test succeeded!")
            else:
                print(f"❌ Your test failed")

            # Ask whether to continue
            print(f"\n{'='*80}")
            continue_test = input("Continue testing? (y/n or Enter to continue): ").strip().lower()
            if continue_test in ['n', 'no']:
                break

        except KeyboardInterrupt:
            print("\n\n⏹️ User interrupted test")
            break
        except Exception as e:
            print(f"\n❌ Interactive test error: {str(e)}")
            continue

    # Interactive test statistics
    if test_count > 0:
        print(f"\n🏁 Interactive test completed!")
        print(f"   Total tests: {test_count}")
        print(f"   Successful tests: {success_count}")
        print(f"   Success rate: {success_count/test_count*100:.1f}%")
    else:
        print("\n📝 No tests were conducted")


def main():
    """Main function - provide interactive and predefined test selection"""
    parser = argparse.ArgumentParser(
        description='Complete workflow test with optional evaluation'
    )
    parser.add_argument(
        '--gt',
        type=str,
        default=None,
        help='Ground truth CSV file path (optional, for evaluation)'
    )

    args = parser.parse_args()

    print("🏗️ Traffic Congestion Analysis Workflow - Complete Test")
    print("Configuration Status:")
    print(f"   LLM Available: {config.is_llm_available()}")
    print(f"   OpenAI API Key: {'Set' if config.get_openai_api_key() else 'Not Set'}")

    if args.gt:
        print(f"   Ground Truth: {args.gt}")
        if not os.path.exists(args.gt):
            print(f"   ⚠️  Warning: Ground truth file does not exist!")
    else:
        print(f"   Ground Truth: 自動検出（解析された日付と道路情報に基づく）")

    print("")

    # Display test options
    print("Please select test mode:")
    print("1. 💬 Interactive Test (Manual Query Input)")
    print("2. 🧪 Predefined Test (Using Built-in Test Cases)")
    print("3. 🔄 Run Both Modes")
    print("0. ❌ Exit")

    while True:
        try:
            choice = input("\nPlease select (1/2/3/0): ").strip()

            if choice == '1':
                run_interactive_test(args.gt)
                break
            elif choice == '2':
                run_predefined_tests(args.gt)
                break
            elif choice == '3':
                print("\n🔄 First run interactive test, then run predefined test")
                run_interactive_test(args.gt)

                print("\n" + "="*80)
                print("Interactive test completed, now starting predefined test...")
                input("Press Enter to continue to predefined test...")

                run_predefined_tests(args.gt)
                break
            elif choice == '0':
                print("👋 Exit test")
                break
            else:
                print("⚠️ Invalid selection, please enter 1, 2, 3 or 0")

        except KeyboardInterrupt:
            print("\n\n👋 Exit test")
            break
        except Exception as e:
            print(f"❌ Input processing error: {str(e)}")
            continue


if __name__ == "__main__":
    # Add missing import
    import pandas as pd

    main()
