"""能力越界检测器 — 系统不具备某能力，回复却假装已执行或已查询

工作原理：
1. 检测 KB 中是否有"不具备/未接入"等否定词
2. 从否定上下文中提取被否定的能力关键词
3. 检查回复中是否包含这些能力关键词对应的操作措辞
"""

import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType


# KB 中标识"系统无此能力"的关键词
KB_NO_CAPABILITY_MARKERS: list[str] = [
    "未接入", "不具备", "无此功能", "无接口", "未对接",
    "需人工", "无法自动", "不能自动",
]

# 回复中表明"系统已执行操作"的关键措辞
REPLY_ACTION_MARKERS: list[str] = [
    "已帮您", "我帮您查", "已为您", "已升级", "已修改", "已发",
    "帮您查了", "我来查", "我查了一下", "正在查",
    "查了一下", "已经查",
]

# 能力动词 — 如果 KB 否定某能力且回复使用这些动词，标记为越界
CAPABILITY_VERBS: list[str] = [
    "查", "升级", "修改", "改", "发", "送", "换",
]


class CapabilityCheckDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "capability_check"

    def _extract_negated_capability(self, kb: str) -> str | None:
        """从 KB 中提取被否定的能力名称"""
        # 查找否定词后面的能力描述
        patterns = [
            r"不[^，。]{0,4}具备?\s*([^，。]{2,15}功能)",
            r"未接入\s*([^，。]{2,15}(?:接口|系统))",
            r"无[^，。]{0,5}此功能[^，。]{0,5}([^，。]{2,15})",
            r"不具备?\s*([^，。]{2,15}(?:能力|功能))",
        ]
        for pat in patterns:
            m = re.search(pat, kb)
            if m and m.lastindex:
                return m.group(m.lastindex).strip()
        return None

    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)

        # 1) KB 中是否有能力否定标记
        has_no_cap = any(kw in knowledge_base for kw in KB_NO_CAPABILITY_MARKERS)
        if not has_no_cap:
            return result

        # 2) 提取被否定的能力名（用于后续精确匹配）
        negated_capability = self._extract_negated_capability(knowledge_base)

        # 3) 检查回复中是否有操作性承诺
        action_claims = [kw for kw in REPLY_ACTION_MARKERS if kw in reply]

        # 4) 额外检查：如果提取到了被否定的能力名，
        #    检查回复中是否包含该能力相关的动词
        cap_verb_matches = []
        if negated_capability:
            # 从能力名中提取关键词（包括2字子串）
            cap_keywords: set[str] = set()
            cap_tokens = re.findall(r'[\w一-鿿]+', negated_capability)
            for t in cap_tokens:
                if len(t) >= 2:
                    cap_keywords.add(t)
                if len(t) > 4:
                    for i in range(len(t) - 1):
                        cap_keywords.add(t[i:i+2])

            for verb in CAPABILITY_VERBS:
                if verb in reply:
                    for m in re.finditer(re.escape(verb), reply):
                        start = max(0, m.start() - 15)
                        end = min(len(reply), m.end() + 15)
                        window = reply[start:end]
                        if any(kw in window for kw in cap_keywords):
                            cap_verb_matches.append(verb)
                            break

        if not action_claims and not cap_verb_matches:
            return result

        # 5) 判断子类型
        query_markers = ["查", "物流", "退款", "进度"]
        operation_markers = ["修改", "改", "升级", "发到", "发送", "发券", "发", "送"]

        all_evidence = []

        if action_claims:
            all_evidence.append(f"回复含操作性承诺: {', '.join(action_claims)}")

        if cap_verb_matches:
            all_evidence.append(
                f"回复含能力动词「{', '.join(cap_verb_matches)}」"
                f"（KB否定能力: {negated_capability or '未知'}）"
            )

        sub_type = SubType.CI_A  # 默认查询冒充
        if any(kw in reply for kw in operation_markers):
            sub_type = SubType.CI_B

        result.hit = True
        result.hallucination_type = HallucinationType.CAPABILITY_OVERREACH
        result.sub_type = sub_type
        result.evidence = [
            f"KB声明系统不具备此能力: 「{knowledge_base.strip()}」"
        ] + all_evidence
        result.confidence = "high"

        return result
