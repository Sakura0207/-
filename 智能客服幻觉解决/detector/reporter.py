"""报告生成器 — 评估检测结果 vs Ground Truth"""

import json
import re
from pathlib import Path
from .models import (
    DetectionResult,
    GroundTruthItem,
    EvaluationReport,
    ConfusionMatrix,
    load_ground_truth,
    dataclass_to_dict,
)


class Reporter:
    """评估检测结果与人工标注的对比"""

    def __init__(self, results: list[DetectionResult], gt_path: str):
        self.results: dict[str, DetectionResult] = {r.id: r for r in results}
        self.ground_truth: dict[str, GroundTruthItem] = {
            gt.id: gt for gt in load_ground_truth(gt_path)
        }

    def evaluate(self) -> EvaluationReport:
        report = EvaluationReport(total=len(self.results))

        tp, fp, tn, fn = 0, 0, 0, 0
        false_negatives: list[dict] = []
        false_positives: list[dict] = []

        for rid, result in self.results.items():
            gt = self.ground_truth.get(rid)
            if gt is None:
                continue

            pred = result.is_hallucination
            actual = gt.is_hallucination

            if pred and actual:
                tp += 1
            elif pred and not actual:
                fp += 1
                false_positives.append({
                    "id": rid,
                    "system_reply": result.system_reply,
                    "detected_type": result.primary_type,
                    "detected_by": result.detected_by,
                    "possible_cause": "规则或LLM误判",
                })
            elif not pred and actual:
                fn += 1
                false_negatives.append({
                    "id": rid,
                    "system_reply": result.system_reply,
                    "knowledge_base": result.knowledge_base,
                    "ground_truth_type": gt.hallucination_type,
                    "ground_truth_detail": gt.detail,
                    "possible_cause": "规则未捕获 + LLM未识别",
                })
            else:
                tn += 1

        report.confusion_matrix = ConfusionMatrix(
            true_positive=tp, false_positive=fp,
            true_negative=tn, false_negative=fn,
        )
        report.ground_truth_positive = tp + fn
        report.detected_positive = tp + fp
        report.false_negatives = false_negatives
        report.false_positives = false_positives

        total = tp + fp + tn + fn
        report.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        report.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        report.accuracy = (tp + tn) / total if total > 0 else 0.0
        report.f1_score = (
            2 * report.precision * report.recall / (report.precision + report.recall)
            if (report.precision + report.recall) > 0
            else 0.0
        )

        # 按类型统计
        type_breakdown: dict[str, dict] = {}
        for rid, result in self.results.items():
            gt = self.ground_truth.get(rid)
            if gt and gt.is_hallucination and gt.hallucination_type:
                t = gt.hallucination_type
                if t not in type_breakdown:
                    type_breakdown[t] = {"total": 0, "detected": 0, "missed": 0}
                type_breakdown[t]["total"] += 1
                if result.is_hallucination:
                    type_breakdown[t]["detected"] += 1
                else:
                    type_breakdown[t]["missed"] += 1

        report.per_type_breakdown = type_breakdown
        return report

    def export_results(self, results_path: str, report_path: str) -> EvaluationReport:
        """导出检测结果 JSON 和评估报告 JSON"""
        output = [r.to_dict() for r in self.results.values()]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        report = self.evaluate()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                dataclass_to_dict(report),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return report

    def generate_misanalysis(self, output_path: str):
        """生成误判原因分析 Markdown"""
        report = self.evaluate()
        lines: list[str] = []
        _w = lines.append

        _w("# 误判原因分析\n")
        _w("> 基于工具检测结果与人工标注（ground_truth.json）的对比分析\n")
        _w("## 概览\n")
        _w(f"- 总样本: {report.total}")
        _w(
            f"- 正确: {report.confusion_matrix.true_positive + report.confusion_matrix.true_negative}"
        )
        _w(
            f"- 误判: {report.confusion_matrix.false_positive + report.confusion_matrix.false_negative}"
        )
        _w(f"  - 漏检 (FN): {report.confusion_matrix.false_negative}")
        _w(f"  - 误报 (FP): {report.confusion_matrix.false_positive}")
        _w(f"- 精确率: {report.precision:.1%}")
        _w(f"- 召回率: {report.recall:.1%}")
        _w(f"- F1 分数: {report.f1_score:.1%}\n")

        _w("## 漏检分析 (False Negatives)\n")
        if report.false_negatives:
            for fn in report.false_negatives:
                rid = fn["id"]
                result = self.results.get(rid)
                gt = self.ground_truth.get(rid)
                if not result or not gt:
                    continue

                _w(f"### {rid}")
                _w(f"- **用户问题**: {result.user_question}")
                _w(f"- **客服回复**: {result.system_reply}")
                _w(f"- **知识库**: {result.knowledge_base}")
                _w(f"- **人工标注**: {gt.hallucination_type} — {gt.detail}")
                _w(f"- **检测结果**: 漏检（未判定为幻觉）")
                _w(f"- **原因分析**:")
                for rr in result.rule_results:
                    if not rr.hit:
                        _w(
                            f"  - `{rr.detector_name}`: 未命中"
                            f"（置信度={rr.confidence}）"
                        )
                _w(f"  - **根因**: {fn.get('possible_cause', '未知')}")
                _w("")
        else:
            _w("✅ 无漏检\n")

        _w("## 误报分析 (False Positives)\n")
        if report.false_positives:
            for fp in report.false_positives:
                rid = fp["id"]
                result = self.results.get(rid)
                gt = self.ground_truth.get(rid)
                if not result or not gt:
                    continue

                _w(f"### {rid}")
                _w(f"- **用户问题**: {result.user_question}")
                _w(f"- **客服回复**: {result.system_reply}")
                _w(f"- **知识库**: {result.knowledge_base}")
                _w(f"- **检测类型**: {result.primary_type}")
                _w(f"- **命中规则**: {[r.detector_name for r in result.rule_results if r.hit]}")
                _w(f"- **原因分析**: {fp.get('possible_cause', '未知')}")
                _w("")
        else:
            _w("✅ 无误报\n")

        _w("## 系统性误判模式\n")
        _w("根据以上分析，归纳出的系统性误判模式：\n")

        # 分析常见的漏检模式
        fn_patterns: dict[str, int] = {}
        for fn_item in report.false_negatives:
            rid = fn_item["id"]
            gt = self.ground_truth.get(rid)
            if gt and gt.hallucination_type:
                fn_patterns[gt.hallucination_type] = (
                    fn_patterns.get(gt.hallucination_type, 0) + 1
                )

        if fn_patterns:
            _w("### 易漏检的幻觉类型\n")
            for htype, count in sorted(fn_patterns.items(), key=lambda x: -x[1]):
                _w(f"- **{htype}**: {count} 条漏检")
            _w("")

        # 分析规则引擎覆盖不足
        _w("### 规则引擎覆盖盲区\n")
        _w("1. **语义级幻觉**: 规则引擎依赖关键词匹配，对以下情况不敏感：")
        _w("   - 信息遗漏（KB 有额外信息，回复未提及）")
        _w("   - 安全误导（KB 含警告，回复忽略）")
        _w("   - 部分正确部分错误（如 h04 发票政策部分对部分错）")
        _w("2. **隐含矛盾**: 回复的陈述逻辑上矛盾但无直接关键词冲突")
        _w("3. **数值模式不完整**: 非标准格式的数值无法被预定义模式捕获")
        _w("")

        _w("### 改进建议\n")
        _w("1. **扩充数值模式库**: 覆盖更多数值表述格式")
        _w("2. **引入语义相似度**: 对规则引擎的 medium 结果用向量相似度做二次过滤")
        _w('3. **LLM Judge 优化**: 对信息遗漏类 case，在 prompt 中增加「检查是否有遗漏信息」的明确指令')
        _w("4. **混合阈值调优**: 对不同类型使用不同的置信度阈值")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
