"""检测引擎 — 编排规则检测器 + LLM Judge + 结果融合"""

from .models import (
    DetectionResult,
    RuleResult,
    LLMJudgeResult,
    Severity,
    load_replies,
)
from .rules.num_compare import NumCompareDetector
from .rules.negation_flip import NegationFlipDetector
from .rules.capability_check import CapabilityCheckDetector
from .rules.statement_check import StatementConsistencyDetector
from .llm_judge import create_judge


class DetectionEngine:
    """检测引擎：初始化所有检测器，逐条检测"""

    def __init__(self, use_mock: bool = True, api_key: str | None = None):
        self.detectors = [
            NumCompareDetector(),
            NegationFlipDetector(),
            CapabilityCheckDetector(),
            StatementConsistencyDetector(),
        ]
        self.judge = create_judge(use_mock=use_mock, api_key=api_key)

    def detect_one(self, item: dict) -> DetectionResult:
        reply = item["system_reply"]
        kb = item["knowledge_base"]

        # ── 阶段1: 规则引擎 ──
        rule_results: list[RuleResult] = []
        for detector in self.detectors:
            result = detector.detect(reply, kb)
            rule_results.append(result)

        # ── 阶段2: LLM Judge ──
        llm_result = self.judge.judge(item)

        # ── 阶段3: 结果融合 ──
        final = self._fuse(rule_results, llm_result, item)
        final.rule_results = rule_results
        final.llm_result = llm_result
        return final

    def _fuse(
        self,
        rule_results: list[RuleResult],
        llm_result: LLMJudgeResult,
        item: dict,
    ) -> DetectionResult:
        result = DetectionResult(
            id=item["id"],
            user_question=item["user_question"],
            system_reply=item["system_reply"],
            knowledge_base=item["knowledge_base"],
        )

        high_hits = [r for r in rule_results if r.hit and r.confidence == "high"]
        med_hits = [r for r in rule_results if r.hit and r.confidence == "medium"]

        # 规则高置信度命中 → 直接采纳
        if high_hits:
            best = high_hits[0]
            result.is_hallucination = True
            result.primary_type = best.hallucination_type
            result.sub_type = best.sub_type
            result.evidence = [e for r in high_hits for e in r.evidence]
            result.detected_by = "+".join(r.detector_name for r in high_hits)

            # 从 LLM 结果补充严重程度和问题句子
            if llm_result.is_hallucination:
                result.severity = llm_result.severity
                result.problematic_sentences = llm_result.problematic_sentences
            else:
                result.severity = Severity.CRITICAL

            return result

        # LLM Judge 检测到幻觉
        if llm_result.is_hallucination:
            result.is_hallucination = True
            result.primary_type = llm_result.primary_type
            result.sub_type = llm_result.sub_type
            result.severity = llm_result.severity or Severity.MEDIUM
            result.problematic_sentences = llm_result.problematic_sentences

            # 从 analysis 中提取 evidence
            for dim_name, dim_data in llm_result.analysis.items():
                if isinstance(dim_data, dict) and dim_data.get("exists"):
                    result.evidence.extend(dim_data.get("evidence", []))

            result.detected_by = "llm_judge"
            if med_hits:
                result.detected_by += "+" + "+".join(
                    r.detector_name for r in med_hits
                )
            return result

        # 规则中等置信度命中
        if med_hits:
            best = med_hits[0]
            result.is_hallucination = True
            result.primary_type = best.hallucination_type
            result.sub_type = best.sub_type
            result.evidence = [e for r in med_hits for e in r.evidence]
            result.detected_by = "+".join(r.detector_name for r in med_hits)
            result.severity = Severity.MEDIUM
            return result

        # 全部未命中 → 正常
        result.is_hallucination = False
        result.detected_by = "none"
        return result

    def detect_all(self, file_path: str) -> list[DetectionResult]:
        """批量检测文件中的所有回复"""
        items = load_replies(file_path)
        results: list[DetectionResult] = []
        for item in items:
            result = self.detect_one(item)
            results.append(result)
        return results
