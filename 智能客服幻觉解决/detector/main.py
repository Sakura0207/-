"""入口脚本 — 运行完整检测 + 评估流程"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.engine import DetectionEngine
from detector.reporter import Reporter


def main():
    parser = argparse.ArgumentParser(
        description="客服回复幻觉检测工具 v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python detector/main.py                          # Mock模式
  python detector/main.py --api-key sk-xxx         # DeepSeek模式
  python detector/main.py --use-mock false         # DeepSeek模式（从环境变量读取API Key）
  python detector/main.py --replies data.json --ground-truth gt.json
        """,
    )
    parser.add_argument(
        "--replies",
        default=str(ROOT / "task4_replies.json"),
        help="回复数据文件路径 (默认: task4_replies.json)",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(ROOT / "task4_ground_truth.json"),
        help="人工标注文件路径 (默认: task4_ground_truth.json)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results"),
        help="输出目录 (默认: results/)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API Key (可选，不传则用Mock或环境变量DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--use-mock",
        default=True,
        type=lambda x: x.lower() not in ("false", "0", "no"),
        help="使用Mock模式 (默认: true, 设为false则用DeepSeek API)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  客服回复幻觉检测工具 v1.0")
    print("=" * 60)

    # ── 阶段1: 检测 ──
    print(f"\n▶ 加载数据: {args.replies}")
    mode = "Mock" if args.use_mock else "DeepSeek"
    print(f"▶ 检测模式: {mode}")
    engine = DetectionEngine(use_mock=args.use_mock, api_key=args.api_key)
    results = engine.detect_all(args.replies)
    print(f"▶ 检测完成: {len(results)} 条\n")

    # ── 阶段2: 输出 ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "detection_results.json"
    report_path = output_dir / "evaluation_report.json"
    misanalysis_path = output_dir / "misanalysis.md"

    gt_path = Path(args.ground_truth)

    if gt_path.exists():
        print("▶ 评估中 (vs ground_truth.json)...")
        reporter = Reporter(results, str(gt_path))
        report = reporter.export_results(str(results_path), str(report_path))
        reporter.generate_misanalysis(str(misanalysis_path))

        print(f"\n{'=' * 60}")
        print("  评估结果")
        print(f"{'=' * 60}")
        print(f"  精确率 (Precision):\t{report.precision:.1%}")
        print(f"  召回率 (Recall):\t{report.recall:.1%}")
        print(f"  F1 分数:\t\t{report.f1_score:.1%}")
        print(f"  准确率 (Accuracy):\t{report.accuracy:.1%}")
        print(f"  TP={report.confusion_matrix.true_positive}  "
              f"FP={report.confusion_matrix.false_positive}  "
              f"TN={report.confusion_matrix.true_negative}  "
              f"FN={report.confusion_matrix.false_negative}")

        if report.false_negatives:
            print(f"\n⚠️  漏检 {len(report.false_negatives)} 条:")
            for fn in report.false_negatives:
                print(f"  - {fn['id']}: {fn.get('possible_cause', '')}")
        if report.false_positives:
            print(f"\n⚠️  误报 {len(report.false_positives)} 条:")
            for fp in report.false_positives:
                print(f"  - {fp['id']}: {fp.get('possible_cause', '')}")
    else:
        # 仅输出检测结果
        output = [r.to_dict() for r in results]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print("▶ 无 ground_truth.json，跳过评估")

    # ── 打印检测摘要 ──
    hallucination_count = sum(1 for r in results if r.is_hallucination)
    normal_count = len(results) - hallucination_count

    print(f"\n{'=' * 60}")
    print("  检测结果摘要")
    print(f"{'=' * 60}")
    print(f"  总计: {len(results)} 条 | "
          f"❌ 幻觉: {hallucination_count} 条 | "
          f"✅ 正常: {normal_count} 条")
    print()

    for r in results:
        status = "❌" if r.is_hallucination else "✅"
        print(f"  {r.id}: {status}  {r.primary_type or '正常':　<6s}  [{r.detected_by}]")

    print(f"\n📁 结果已保存至: {output_dir.resolve()}")
    print(f"  检测结果: {results_path}")
    if gt_path.exists():
        print(f"  评估报告: {report_path}")
        print(f"  误判分析: {misanalysis_path}")

    return results


if __name__ == "__main__":
    main()
