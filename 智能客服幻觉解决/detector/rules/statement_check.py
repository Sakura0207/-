"""声明一致性检测器 — 提取回复中的断言并与知识库对照"""

import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType


# 声明模式: (正则, 类别名) — 从回复中提取关键断言
STATEMENT_PATTERNS: list[tuple[str, str]] = [
    # 属性/材质
    (r"(?:是|采用|使用|属于)([^，。]{1,20}?(?:材质|面料|材料|接口|设计))", "属性"),
    (r"([^，。]{2,20}(?:材质|面料|材料|接口)[：:为]?([^，。]{1,15}))", "属性"),

    # 功能声明
    (r"支持([^，。]{2,20}(?:功能|支付|连接|充电|公交))", "功能"),
    (r"(?:可以|能够)用[于来]([^，。]{2,20})", "功能"),

    # 政策/服务
    (r"支持([^，。]{2,20}(?:退货|保修|发票|配送))", "政策"),
    (r"提供([^，。]{2,20}(?:包装|服务|保障))", "政策"),
    (r"(?:包邮|免运费|承担运费)", "政策"),

    # 关联关系
    (r"(?:是|属于)([^，。]{2,20}(?:旗下|子品牌|关联|分公司))", "关联"),
]


# KB 中的矛盾标记词
KB_CONTRADICT_MARKERS: list[str] = [
    "不", "无", "没", "不支持", "不具备", "未标注", "未提及",
    "纯线上", "不存在", "暂不", "没有",
]

class StatementConsistencyDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "statement_check"

    def _extract_statements(self, reply: str) -> list[tuple[str, str]]:
        """从回复中提取断言式陈述"""
        statements: list[tuple[str, str]] = []
        for pattern, category in STATEMENT_PATTERNS:
            for match in re.finditer(pattern, reply):
                statements.append((match.group(0).strip(), category))
        return statements

    def _kb_contradicts(self, statement: str, kb: str) -> bool:
        """判断 KB 是否与这个陈述矛盾"""
        # 提取陈述中的有意义的词（>=2 的中文/英文词）
        words = re.findall(r'[\w一-鿿]{2,}', statement)
        if not words:
            return False

        # 找 KB 中包含这些关键词的句子
        relevant_sentences: list[str] = []
        for sent in re.split(r'[。！？;；]', kb):
            sent = sent.strip()
            if not sent:
                continue
            # 至少匹配 2 个词才认为相关
            match_count = sum(1 for w in words if w in sent)
            if match_count >= 2:
                relevant_sentences.append(sent)

        if not relevant_sentences:
            return False

        # 检查相关句子中是否有否定词
        for sent in relevant_sentences:
            has_negation = any(m in sent for m in KB_CONTRADICT_MARKERS)
            if has_negation:
                return True

        return False

    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)

        statements = self._extract_statements(reply)
        for stmt, category in statements:
            if self._kb_contradicts(stmt, knowledge_base):
                result.hit = True
                result.hallucination_type = HallucinationType.FACTUAL_CONFLICT
                result.sub_type = SubType.FC_B
                result.evidence.append(f"声明「{stmt}」与KB矛盾")
                result.confidence = "medium"

        return result
