"""否定翻转检测器 — KB 中否定某事物，回复却以肯定形式提及

基于通用的否定模式，不从数据集特例出发。
"""

import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType


def extract_keywords(text: str) -> list[str]:
    """从文本中提取关键词

    对2-3字的词保留完整；对>=4字的词，额外提取2字子串
    如"线下门店" → ["线下门店", "线下", "下门", "门店"]
    如"退货地址" → ["退货地址", "退货", "货地", "地址"]
    如"品牌关联关系" → ["品牌关联关系", "品牌", "牌关", "关联", "联关", "关系"]
    """
    tokens = re.findall(r'[\w一-鿿]+', text)
    keywords: list[str] = []
    for t in tokens:
        if len(t) >= 2:
            keywords.append(t)
        # 对>=4字的词，提取其中的2字子串
        if len(t) >= 4:
            for i in range(len(t) - 1):
                keywords.append(t[i:i+2])
    return keywords


# 通用否定模式: (正则, 幻觉类型, 子类型)
# 每个模式用最后一个捕获组标识"被否定的主体"
NEGATION_PATTERNS: list[tuple[str, str, str]] = [
    # ── 政策/优惠否定 ──
    (r"无\s*(学生[^，。]{1,10}(?:优惠|折扣|政策))",
                                             HallucinationType.FABRICATION,      SubType.IF_A),
    (r"无\s*(满\d+[^，。]{1,10}(?:活动|优惠))",
                                             HallucinationType.FABRICATION,      SubType.IF_A),
    (r"不支持\s*(纸质[^，。]{1,6})",         HallucinationType.FACTUAL_CONFLICT, SubType.FC_C),

    # ── 渠道/实体否定 ──
    (r"(?:无|纯线上)[^，。]{0,20}((?:线下|实体|体验)[^，。]{1,10})",
                                             HallucinationType.FABRICATION,      SubType.IF_B),
    (r"不支持\s*(货到[^，。]{1,6})",         HallucinationType.FACTUAL_CONFLICT, SubType.FC_C),

    # ── 属性否定 ──
    # "未标注/未包含某功能属性" → 捕获属性名
    (r"(?:未标注|未包含|未提及)[^，。]*?([^，。]{2,15}(?:功能|接口|材质|版本|类型|技术))",
                                             HallucinationType.FACTUAL_CONFLICT, SubType.FC_B),
    # "不是真皮" → 捕获材质名
    (r"不[是为][^，。]*?([^，。]{2,10}(?:皮|革|质地|面料))",
                                             HallucinationType.FACTUAL_CONFLICT, SubType.FC_B),

    # ── 品牌关联否定 ──
    (r"未[提涉][^，。]*?((?:品牌|关联|关系)[^，。]{1,10}(?:关联|关系|旗下|子品牌))",
                                             HallucinationType.FABRICATION,      SubType.IF_C),

    # ── 能力否定 ──
    (r"不[具备支持][^，。]*?(?:此功能|该能力|工单|操作|修改)",
                                             HallucinationType.CAPABILITY_OVERREACH, SubType.CI_A),
    (r"未(?:接入|对接)[^，。]*?(?:接口|系统)",
                                             HallucinationType.CAPABILITY_OVERREACH, SubType.CI_A),
    (r"不\S*具备[^，。]*?功能",              HallucinationType.CAPABILITY_OVERREACH, SubType.CI_A),

    # ── 信息限制否定 ──
    # "不可口头告知退货地址" → 捕获"退货地址"
    (r"(?:不可|禁止|不得|不应)[^，。]{0,15}(?:口头|直接)[^，。]{0,15}?"
     r"((?:退货|送货|配送|收货)[^，。]{0,6}(?:地址|方式|信息))",
                                             HallucinationType.FABRICATION,      SubType.IF_B),
]


class NegationFlipDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "negation_flip"

    def _find_negated_subjects(self, kb: str) -> list[tuple[str, str, str]]:
        """从 KB 中找出被否定的关键信息"""
        subjects: list[tuple[str, str, str]] = []
        for pattern, h_type, sub_type in NEGATION_PATTERNS:
            match = re.search(pattern, kb)
            if match:
                # 使用最后一个捕获组作为主体
                if match.lastindex and match.group(match.lastindex):
                    subject = match.group(match.lastindex).strip()
                else:
                    subject = match.group(0).strip()
                subjects.append((subject, h_type, sub_type))
        return subjects

    def _reply_asserts_positive(self, reply: str, subject: str) -> bool:
        """检查 reply 是否以肯定形式提到了这个 subject

        使用多级匹配：先尝试完整词匹配，再尝试2字子串匹配。
        要求至少有一个2字关键词匹配且附近无否定。
        """
        keywords = extract_keywords(subject)
        # 只保留2字及以上的关键词
        keywords = [k for k in keywords if len(k) >= 2]

        if not keywords:
            return False

        # 计算有多少关键词出现在回复中
        matched = [(w, re.search(re.escape(w), reply)) for w in keywords]
        matched_words = [w for w, m in matched if m is not None]

        if not matched_words:
            return False

        # 对匹配到的词，检查附近(前后20字)是否有否定词
        for word in matched_words:
            for m in re.finditer(re.escape(word), reply):
                start = max(0, m.start() - 20)
                end = min(len(reply), m.end() + 20)
                window = reply[start:end]
                # 检查否定词
                negations = [neg for neg in ["不", "无", "没", "不支持", "不具备", "未标注"]
                           if neg in window]
                if negations:
                    # 排除中性搭配
                    neutral = ["不好", "不错", "不仅", "不断", "不用", "不客气", "不方便"]
                    if not any(np in window for np in neutral):
                        return False

        return True

    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)

        negated_items = self._find_negated_subjects(knowledge_base)
        for subject, h_type, sub_type in negated_items:
            if self._reply_asserts_positive(reply, subject):
                result.hit = True
                result.hallucination_type = h_type
                result.sub_type = sub_type
                result.evidence.append(
                    f"KB否定「{subject}」, 回复却以肯定形式提及"
                )
                result.confidence = "high"

        return result
