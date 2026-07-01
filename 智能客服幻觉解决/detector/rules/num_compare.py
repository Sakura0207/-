"""数值比对检测器 — 检测回复与知识库中数字信息的不一致

支持阿拉伯数字和中文数字（两、二、三、半等）。
"""

import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType

# 中文数字 → 阿拉伯数字映射
CHINESE_NUM_MAP: dict[str, str] = {
    "零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
    "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
    "九": "9", "十": "10", "半": "0.5",
}


def normalize_number(text: str) -> str:
    """将文本中的中文数字替换为阿拉伯数字"""
    result = text
    for cn, ar in CHINESE_NUM_MAP.items():
        result = result.replace(cn, ar)
    return result


# 数值指标模式（归一化后匹配）
# 注意：在匹配前会把中文数字替换为阿拉伯数字
KEY_METRICS: dict[str, str] = {
    "蓝牙版本":         r"蓝牙\s*(\d+\.?\d*)",
    "保修期":           r"保修[期]*[：:为]*\s*(\d+)\s*[个月年]",
    "无理由退货天数":   r"(\d+)\s*天\s*无理由",
    "发货时间":         r"(\d+)\s*小时[内发]",
    "延迟":             r"延迟[低至约]*\s*(\d+)\s*ms",
    "到货时间":         r"(\d+)[-~至]*(\d*)\s*天\s*(?:到货|到|送达)",
    "折扣额":           r"满\s*(\d+)\s*[减送]",
    "折扣率":           r"(\d+)\s*折",
    "响应时间":         r"(\d+)\s*小时[内联系]",
    "使用时间":         r"(\d+)\s*[个月年]",
}


class NumCompareDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "num_compare"

    def _normalize_and_find(self, pattern: str, text: str) -> list[str]:
        """先归一化中文数字，再匹配数值"""
        normalized = normalize_number(text)
        return re.findall(pattern, normalized)

    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)

        mismatches: list[str] = []
        for metric_name, pattern in KEY_METRICS.items():
            reply_vals = self._normalize_and_find(pattern, reply)
            kb_vals = self._normalize_and_find(pattern, knowledge_base)

            # 只在两边都有值时才比对
            if not reply_vals or not kb_vals:
                continue

            # 扁平化处理
            def flatten(vals):
                result_list = []
                for v in vals:
                    if isinstance(v, tuple):
                        result_list.extend(x for x in v if x)
                    else:
                        result_list.append(v)
                return tuple(result_list)

            reply_flat = flatten(reply_vals)
            kb_flat = flatten(kb_vals)

            if reply_flat and kb_flat and reply_flat != kb_flat:
                mismatches.append(
                    f"{metric_name}: 回复={reply_flat}, 知识库={kb_flat}"
                )

        if mismatches:
            result.hit = True
            result.hallucination_type = HallucinationType.FACTUAL_CONFLICT
            result.sub_type = SubType.FC_A
            result.evidence = mismatches
            result.confidence = "high"

        return result
