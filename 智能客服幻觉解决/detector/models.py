"""数据模型 — 使用 dataclasses 替代 pydantic，零外部依赖"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── 枚举常量 ──────────────────────────────────────────

class HallucinationType:
    FACTUAL_CONFLICT = "事实矛盾"
    FABRICATION = "信息编造"
    CAPABILITY_OVERREACH = "能力冒充"
    COMPLETENESS_ISSUE = "信息遗漏"


class SubType:
    FC_A = "FC-A"  # 数值/版本错误
    FC_B = "FC-B"  # 属性描述错误
    FC_C = "FC-C"  # 政策规则错误
    IF_A = "IF-A"  # 优惠/政策编造
    IF_B = "IF-B"  # 实体/渠道编造
    IF_C = "IF-C"  # 关联关系编造
    CI_A = "CI-A"  # 查询冒充
    CI_B = "CI-B"  # 操作冒充
    IC_A = "IC-A"  # 条件遗漏
    IC_B = "IC-B"  # 安全提示遗漏


class Severity:
    CRITICAL = "严重"
    MEDIUM = "中等"
    LOW = "轻微"


# ─── 数据模型 ──────────────────────────────────────────

@dataclass
class RuleResult:
    detector_name: str
    hit: bool = False
    hallucination_type: Optional[str] = None
    sub_type: Optional[str] = None
    evidence: list[str] = field(default_factory=list)
    confidence: str = "low"  # high / medium / low


@dataclass
class LLMJudgeResult:
    is_hallucination: bool = False
    primary_type: Optional[str] = None
    sub_type: Optional[str] = None
    severity: Optional[str] = None
    problematic_sentences: list[str] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    raw_response: Optional[str] = None


@dataclass
class DetectionResult:
    id: str
    user_question: str
    system_reply: str
    knowledge_base: str
    is_hallucination: bool = False
    primary_type: Optional[str] = None
    sub_type: Optional[str] = None
    severity: Optional[str] = None
    problematic_sentences: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    detected_by: str = ""
    rule_results: list[RuleResult] = field(default_factory=list)
    llm_result: Optional[LLMJudgeResult] = None

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class GroundTruthItem:
    id: str
    is_hallucination: bool
    hallucination_type: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0


@dataclass
class EvaluationReport:
    total: int = 0
    ground_truth_positive: int = 0
    detected_positive: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    false_negatives: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    per_type_breakdown: dict[str, dict] = field(default_factory=dict)


# ─── 工具函数 ──────────────────────────────────────────

def load_replies(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(path: str | Path) -> list[GroundTruthItem]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [GroundTruthItem(**item) for item in data]


def dataclass_to_dict(obj):
    """将 dataclass 递归转为 dict（用于 JSON 序列化）"""
    return json.loads(json.dumps(obj, default=lambda o: asdict(o), ensure_ascii=False))
