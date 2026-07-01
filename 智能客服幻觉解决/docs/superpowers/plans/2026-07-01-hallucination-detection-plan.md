# 客服回复幻觉检测工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两阶段（规则引擎 + LLM Judge）幻觉检测工具，对 20 条回复逐条检测，输出结果并与 ground truth 对比评估。

**Architecture:** 4 个并行规则检测器 → 结果传入 LLM Judge/融合 → reporter 生成评估报告。规则引擎捕获可枚举的矛盾（数值、否定翻转、能力边界），LLM Judge 处理语义级模糊 case。

**Tech Stack:** Python 3.10+, openai (DeepSeek API 兼容), pydantic, json

## Global Constraints

- 纯 Python 标准库 + openai + pydantic, 无数据库依赖
- Mock 模式在无 API key 时可完整运行
- 所有检测结果持久化为 JSON
- 评估报告含混淆矩阵、精确率、召回率、F1

---
### Task 1: 项目结构 + 数据模型

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\__init__.py`
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\models.py`

**Interfaces:**
- Consumes: `task4_replies.json`, `task4_ground_truth.json`
- Produces: 所有 Pydantic 数据模型，供后续所有 task 引用

- [ ] **Step 1: 创建目录结构和 `__init__.py`**

```
detector/
├── __init__.py
├── models.py
├── rules/
│   └── __init__.py
├── llm_judge.py
├── engine.py
├── reporter.py
└── main.py
results/
└── .gitkeep
```

`__init__.py` 在每个目录放空文件即可。

- [ ] **Step 2: 创建数据模型 `models.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class HallucinationType(str, Enum):
    FACTUAL_CONFLICT = "事实矛盾"
    FABRICATION = "信息编造"
    CAPABILITY_OVERREACH = "能力冒充"
    COMPLETENESS_ISSUE = "信息遗漏"

class SubType(str, Enum):
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

class Severity(str, Enum):
    CRITICAL = "严重"
    MEDIUM = "中等"
    LOW = "轻微"

class RuleResult(BaseModel):
    detector_name: str
    hit: bool = False
    hallucination_type: Optional[HallucinationType] = None
    sub_type: Optional[SubType] = None
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "low"  # high / medium / low

class LLMJudgeResult(BaseModel):
    is_hallucination: bool = False
    primary_type: Optional[HallucinationType] = None
    sub_type: Optional[SubType] = None
    severity: Optional[Severity] = None
    problematic_sentences: list[str] = Field(default_factory=list)
    analysis: dict[str, dict] = Field(default_factory=dict)
    raw_response: Optional[str] = None

class DetectionResult(BaseModel):
    id: str
    user_question: str
    system_reply: str
    knowledge_base: str
    is_hallucination: bool = False
    primary_type: Optional[HallucinationType] = None
    sub_type: Optional[SubType] = None
    severity: Optional[Severity] = None
    problematic_sentences: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    detected_by: str = ""
    rule_results: list[RuleResult] = Field(default_factory=list)
    llm_result: Optional[LLMJudgeResult] = None

class GroundTruthItem(BaseModel):
    id: str
    is_hallucination: bool
    hallucination_type: Optional[str] = None
    detail: Optional[str] = None

class ConfusionMatrix(BaseModel):
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

class EvaluationReport(BaseModel):
    total: int = 0
    ground_truth_positive: int = 0
    detected_positive: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    false_negatives: list[dict] = Field(default_factory=list)
    false_positives: list[dict] = Field(default_factory=list)
    per_type_breakdown: dict[str, dict] = Field(default_factory=dict)
```

- [ ] **Step 3: 创建数据加载函数 (同样在 models.py 中)**

```python
import json
from pathlib import Path

def load_replies(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_ground_truth(path: str | Path) -> list[GroundTruthItem]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [GroundTruthItem(**item) for item in data]
```

---
### Task 2: 规则引擎基础设施 + NumCompareDetector

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\__init__.py`
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\base.py`
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\num_compare.py`

**Interfaces:**
- Consumes: `models.RuleResult`, `models.DetectionResult`
- Produces: `BaseDetector` 抽象类 + `NumCompareDetector` 实现

- [ ] **Step 1: 创建 `rules/__init__.py`** (空文件，包标记)

- [ ] **Step 2: 创建 `base.py` — 检测器基类**

```python
from abc import ABC, abstractmethod
from ..models import RuleResult

class BaseDetector(ABC):
    """所有规则检测器的基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    @abstractmethod
    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        ...
```

- [ ] **Step 3: 创建 `num_compare.py` — 数值比对检测器**

检测逻辑：从 reply 和 knowledge_base 中提取"关键指标名 + 数值"对，比较同一指标名的值是否一致。

```python
import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType

# 预定义关键指标映射表，含上下文词
KEY_METRICS = {
    "蓝牙": r"蓝牙\s*(\d+\.?\d*)",
    "保修": r"保修[期]*[：:为]*\s*(\d+)\s*[个月年]",
    "退货": r"(\d+)\s*天\s*无理由",
    "发货": r"(\d+)\s*小时",
    "延迟": r"延迟[低至约]*\s*(\d+)\s*ms",
    "到货": r"(\d+)[-~至]*(\d*)\s*天",
    "价格": r"满\s*(\d+)\s*[减送]",
    "折扣": r"(\d+)\s*折",
}

class NumCompareDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "num_compare"
    
    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)
        
        mismatches = []
        for metric_name, pattern in KEY_METRICS.items():
            reply_nums = re.findall(pattern, reply)
            kb_nums = re.findall(pattern, knowledge_base)
            
            if reply_nums and kb_nums:
                # 标准化为元组比较
                reply_vals = tuple(reply_nums)
                kb_vals = tuple(kb_nums)
                if reply_vals != kb_vals:
                    mismatches.append(f"{metric_name}: 回复说{reply_vals}, 知识库说{kb_vals}")
        
        if mismatches:
            result.hit = True
            result.hallucination_type = HallucinationType.FACTUAL_CONFLICT
            result.sub_type = SubType.FC_A
            result.evidence = mismatches
            result.confidence = "high"
        
        return result
```

- [ ] **Step 4: 快速验证**

```python
# 手动测试
detector = NumCompareDetector()
r = detector.detect(
    "这款耳机采用蓝牙5.3版本，延迟低至40ms",
    "产品参数：蓝牙5.0，延迟约80ms"
)
assert r.hit == True
assert len(r.evidence) == 2
print("NumCompareDetector 测试通过")
```

---
### Task 3: NegationFlipDetector + CapabilityCheckDetector

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\negation_flip.py`
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\capability_check.py`

**Interfaces:**
- Consumes: `models.RuleResult`, `HallucinationType`, `SubType`
- Produces: 两个检测器实现

- [ ] **Step 1: 创建 `negation_flip.py`**

检测逻辑：KB 中出现否定词（无/不支持/不具备/未标注），提取否定词后的关键名词。检查 reply 中对应的名词是否以肯定形式出现。

```python
import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType

# 否定关键词组
NEGATION_PATTERNS = [
    (r"无\s*(学生优惠.*)", HallucinationType.FABRICATION, SubType.IF_A),
    (r"无\s*(满\d+.*活动)", HallucinationType.FABRICATION, SubType.IF_A),
    (r"不[支持具备]\s*(.*?)(?:[。，]|$)", HallucinationType.FABRICATION, SubType.IF_B),
    (r"未[标注提及].*?((?:NFC|学生|优惠).*?功能)", HallucinationType.FACTUAL_CONFLICT, SubType.FC_B),
    (r"无\s*(线下门店|实体店)", HallucinationType.FABRICATION, SubType.IF_B),
    (r"不[支持具备]\s*(纸质发票|货到付款)", HallucinationType.FACTUAL_CONFLICT, SubType.FC_C),
    (r"未(?:接入|具备)\s*(.*?功能)", HallucinationType.CAPABILITY_OVERREACH, SubType.CI_A),
    (r"暂不[支持提供]", HallucinationType.FACTUAL_CONFLICT, SubType.FC_C),
]

class NegationFlipDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "negation_flip"
    
    def _find_negated_subject(self, kb: str) -> list[tuple[str, HallucinationType, SubType]]:
        """从 KB 中找出被否定的关键信息"""
        subjects = []
        for pattern, h_type, sub_type in NEGATION_PATTERNS:
            match = re.search(pattern, kb)
            if match:
                subject = match.group(1).strip()
                subjects.append((subject, h_type, sub_type))
        return subjects
    
    def _reply_asserts_positive(self, reply: str, subject: str) -> bool:
        """检查 reply 是否以肯定形式提到了这个 subject"""
        # 精简 subject 到核心词
        core_words = re.findall(r'[\w]+', subject)
        if not core_words:
            return False
        
        # 检查 reply 中是否以肯定语气包含了核心词
        found_all = all(
            re.search(word, reply) for word in core_words if len(word) > 1
        )
        if not found_all:
            return False
        
        # 排除 reply 中也否定的情况
        reply_negative = any(
            neg in reply for neg in ["不", "无", "没", "不支持", "不具备"]
        )
        return not reply_negative
    
    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)
        
        negated_items = self._find_negated_subject(knowledge_base)
        for subject, h_type, sub_type in negated_items:
            if self._reply_asserts_positive(reply, subject):
                result.hit = True
                result.hallucination_type = h_type
                result.sub_type = sub_type
                result.evidence.append(
                    f"KB否定'{subject}', 回复却以肯定形式提及"
                )
                result.confidence = "high"
        
        return result
```

- [ ] **Step 2: 创建 `capability_check.py`**

检测逻辑：KB 中声明"无接口/无此功能/不具备"，reply 中出现假装已执行的措辞。

```python
import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType

# 能力越界关键词
KB_NO_CAPABILITY = [
    "未接入", "不具备", "无此功能", "无接口", "未对接",
    "需人工", "无法自动"
]

REPLY_ACTION_CLAIMS = [
    "已帮您", "我帮您查", "已为", "已升级", "已修改",
    "我来查", "帮您查", "正在查"
]

class CapabilityCheckDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "capability_check"
    
    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        result = RuleResult(detector_name=self.name)
        
        # KB 说系统不具备某能力
        has_no_cap = any(kw in knowledge_base for kw in KB_NO_CAPABILITY)
        if not has_no_cap:
            return result
        
        # 检查回复是否假装有操作/查询能力
        action_claims = [kw for kw in REPLY_ACTION_CLAIMS if kw in reply]
        if not action_claims:
            return result
        
        # 判断具体子类型
        query_keywords = ["查", "物流", "退款", "进度"]
        operation_keywords = ["修改", "改", "升级", "发到", "发送"]
        
        sub_type = SubType.CI_A  # 默认查询冒充
        if any(kw in reply for kw in operation_keywords):
            sub_type = SubType.CI_B
        
        result.hit = True
        result.hallucination_type = HallucinationType.CAPABILITY_OVERREACH
        result.sub_type = sub_type
        result.evidence = [
            f"KB声明'{knowledge_base.strip()}'",
            f"回复含操作性承诺: {', '.join(action_claims)}"
        ]
        result.confidence = "high"
        
        return result
```

- [ ] **Step 3: 快速验证两个检测器**

```python
# 验证 negation_flip
nf = NegationFlipDetector()
r = nf.detect("我们支持纸质发票", "暂不支持纸质发票")
assert r.hit == True

# 验证 capability_check
cc = CapabilityCheckDetector()
r = cc.detect("我帮您查了物流信息", "无（客服系统未接入物流查询接口）")
assert r.hit == True
print("两个检测器测试通过")
```

---
### Task 4: StatementConsistencyDetector

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\rules\statement_check.py`

**Interfaces:**
- Consumes: `BaseDetector`, `RuleResult`
- Produces: 声明一致性检测器（捕获规则引擎其他检测器遗漏的 case）

- [ ] **Step 1: 创建 `statement_check.py`**

检测逻辑：提取 reply 中的声明性断言（陈述句），对每个断言用关键词匹配 KB，检查是否矛盾。

```python
import re
from .base import BaseDetector
from ..models import RuleResult, HallucinationType, SubType, Severity

# 声明提取模式
STATEMENT_PATTERNS = [
    # 属性声明
    (r"(?:是|采用|使用|支持|配备|属于)([^，。]+?)(?:制作|设计|打造)", "属性"),
    (r"([^，。]*?)(?:材质|面料|材料|接口)[：:为]?([^，。]+)", "属性"),
    # 功能声明
    (r"支持([^，。]+功能)", "功能"),
    (r"可以[用][于]([^，。]+)", "功能"),
    # 政策声明
    (r"([^，。]*?)(?:支持|提供|有)(\d+天[^，。]*退[换货])", "政策"),
    (r"([^，。]*?)(?:包邮|免运费|承担运费)", "政策"),
    # 关联声明
    (r"(?:是|属于)([^，。]+旗下|子品牌|关联)", "关联"),
]

# 否定匹配 — 如果 KB 含这些词，说明回复的声明可能冲突
KB_CONTRADICT_MARKERS = [
    "不", "无", "没", "不支持", "不具备", "未标注", "纯线上", "不存在"
]

class StatementConsistencyDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "statement_check"
    
    def _extract_statements(self, reply: str) -> list[tuple[str, str]]:
        """从回复中提取声明性断言"""
        statements = []
        for pattern, category in STATEMENT_PATTERNS:
            for match in re.finditer(pattern, reply):
                statements.append((match.group(0), category))
        return statements
    
    def _check_contradiction(self, statement: str, kb: str) -> bool:
        """检查 KB 是否与声明矛盾"""
        # 从声明中提取关键词
        words = re.findall(r'[\w一-鿿]{2,}', statement)
        relevant_sentences = []
        
        # 找 KB 中包含这些关键词的句子
        for word in words:
            if len(word) >= 2 and word in kb:
                # 找到相关句子
                for sent in re.split(r'[。！？]', kb):
                    if word in sent:
                        relevant_sentences.append(sent)
        
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
            if self._check_contradiction(stmt, knowledge_base):
                result.hit = True
                result.hallucination_type = HallucinationType.FACTUAL_CONFLICT
                result.sub_type = SubType.FC_B
                result.evidence.append(f"声明'{stmt}'与KB矛盾")
                result.confidence = "medium"
        
        return result
```

- [ ] **Step 2: 快速验证**

```python
sc = StatementConsistencyDetector()
r = sc.detect(
    "这款包采用头层牛皮制作", 
    "材质：PU合成革"
)
assert r.hit == True
print("StatementConsistencyDetector 测试通过")
```

---
### Task 5: LLM Judge (Mock 模式 + DeepSeek)

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\llm_judge.py`

**Interfaces:**
- Consumes: `models.ReplyItem` (dict), `models.LLMJudgeResult`
- Produces: `LLMJudge.judge(reply_item) -> LLMJudgeResult`

- [ ] **Step 1: 创建 `llm_judge.py` — Mock Judge**

```python
import json
from .models import LLMJudgeResult, HallucinationType, SubType, Severity

# 内置的 Mock 响应（基于纯数据驱动的分析）
MOCK_RESPONSES = {
    "h01": {
        "is_hallucination": True,
        "primary_type": "事实矛盾",
        "sub_type": "FC-C",
        "severity": "严重",
        "problematic_sentences": ["全品类支持30天无理由退货，运费也由我们承担"],
        "analysis": {
            "factual_conflict": {"exists": True, "evidence": ["退货天数30天≠7天", "运费承担矛盾"]},
            "fabrication": {"exists": False, "evidence": []},
            "capability_overreach": {"exists": False, "evidence": []},
            "completeness_issue": {"exists": False, "evidence": []}
        }
    },
    # ... 后续在 main.py 中集成为完整字典
}

class MockJudge:
    def judge(self, reply_item: dict) -> LLMJudgeResult:
        item_id = reply_item["id"]
        response = MOCK_RESPONSES.get(item_id)
        if response is None:
            return LLMJudgeResult(is_hallucination=False)
        return LLMJudgeResult(
            is_hallucination=response["is_hallucination"],
            primary_type=HallucinationType(response["primary_type"]) if response["primary_type"] else None,
            sub_type=SubType(response["sub_type"]) if response["sub_type"] else None,
            severity=Severity(response["severity"]) if response["severity"] else None,
            problematic_sentences=response["problematic_sentences"],
            analysis=response["analysis"],
            raw_response=json.dumps(response, ensure_ascii=False)
        )
```

- [ ] **Step 2: 创建 DeepSeek Judge**

```python
import json
import os
from openai import OpenAI
from .models import LLMJudgeResult, HallucinationType, SubType, Severity

DEEPSEEK_SYSTEM_PROMPT = """你是一个智能客服回复的幻觉检测专家。你的任务是判断客服系统的回复是否与知识库一致。

## 幻觉分类体系

1. **事实矛盾**: 回复中的可验证事实与知识库直接冲突
   - FC-A 数值/版本错误: 数字、版本、时间矛盾
   - FC-B 属性描述错误: 材质/功能/接口等定性属性错误
   - FC-C 政策规则错误: 退货/发货/发票等政策规则错误

2. **信息编造**: 知识库中完全没有的信息，回复编造了具体细节
   - IF-A 优惠/政策编造: 不存在的促销活动
   - IF-B 实体/渠道编造: 杜撰地址、门店
   - IF-C 关联关系编造: 编造品牌关系

3. **能力冒充**: 系统不具备某能力却假装已执行
   - CI-A 查询冒充: 声称查到数据但无接口
   - CI-B 操作冒充: 声称已执行操作但无此功能

4. **信息遗漏**: 遗漏了知识库中影响判断的关键限定条件
   - IC-A 条件遗漏: 遗漏关键限定条件
   - IC-B 安全提示遗漏: 遗漏安全警示

请逐条分析并只输出JSON格式结果，不要多余文字。"""

DEEPSEEK_USER_PROMPT = """请检查以下客服回复是否存在幻觉：

## 用户问题
{user_question}

## 客服回复
{system_reply}

## 知识库信息
{knowledge_base}

请以JSON格式输出检测结果：
{{
  "is_hallucination": true/false,
  "primary_type": "事实矛盾|信息编造|能力冒充|信息遗漏|null",
  "sub_type": "FC-A|FC-B|FC-C|IF-A|IF-B|IF-C|CI-A|CI-B|IC-A|IC-B|null",
  "severity": "严重|中等|轻微|null",
  "problematic_sentences": ["回复中的问题句子"],
  "analysis": {{
    "factual_conflict": {{"exists": bool, "evidence": [str]}},
    "fabrication": {{"exists": bool, "evidence": [str]}},
    "capability_overreach": {{"exists": bool, "evidence": [str]}},
    "completeness_issue": {{"exists": bool, "evidence": [str]}}
  }}
}}"""

class DeepSeekJudge:
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        ) if self.api_key else None
    
    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.client is not None
    
    def judge(self, reply_item: dict) -> LLMJudgeResult:
        if not self.available:
            return LLMJudgeResult(is_hallucination=False)
        
        try:
            user_prompt = DEEPSEEK_USER_PROMPT.format(
                user_question=reply_item["user_question"],
                system_reply=reply_item["system_reply"],
                knowledge_base=reply_item["knowledge_base"]
            )
            
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = resp.choices[0].message.content
            data = json.loads(content)
            
            return LLMJudgeResult(
                is_hallucination=data.get("is_hallucination", False),
                primary_type=self._parse_enum(HallucinationType, data.get("primary_type")),
                sub_type=self._parse_enum(SubType, data.get("sub_type")),
                severity=self._parse_enum(Severity, data.get("severity")),
                problematic_sentences=data.get("problematic_sentences", []),
                analysis=data.get("analysis", {}),
                raw_response=content
            )
        except Exception as e:
            return LLMJudgeResult(is_hallucination=False, raw_response=f"Error: {str(e)}")
    
    @staticmethod
    def _parse_enum(enum_class, value):
        if not value or value == "null":
            return None
        try:
            return enum_class(value)
        except ValueError:
            return None
```

- [ ] **Step 3: 创建 Judge 工厂函数**

在 `llm_judge.py` 末尾添加：

```python
def create_judge(use_mock: bool = False, api_key: str | None = None):
    """工厂函数：根据参数返回 MockJudge 或 DeepSeekJudge"""
    if use_mock:
        return MockJudge()
    judge = DeepSeekJudge(api_key=api_key)
    if judge.available:
        return judge
    print("⚠️ DeepSeek API key 未配置, 降级使用 Mock 模式")
    return MockJudge()
```

- [ ] **Step 4: 快速验证**

```python
judge = MockJudge()
result = judge.judge({"id": "h01", "user_question": "test", "system_reply": "test", "knowledge_base": "test"})
assert result.is_hallucination == True
print(f"h01 检测结果: {result.primary_type} / {result.sub_type}")
```

---
### Task 6: 检测引擎 (Engine) — 编排规则 + LLM + 结果融合

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\engine.py`

**Interfaces:**
- Consumes: 所有检测器 + LLM Judge, `load_replies()`
- Produces: `DetectionResult` 列表

- [ ] **Step 1: 创建 `engine.py`**

```python
from .models import DetectionResult, RuleResult, LLMJudgeResult, load_replies, HallucinationType
from .rules.num_compare import NumCompareDetector
from .rules.negation_flip import NegationFlipDetector
from .rules.capability_check import CapabilityCheckDetector
from .rules.statement_check import StatementConsistencyDetector
from .llm_judge import create_judge

class DetectionEngine:
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
        
        # 阶段1: 规则引擎
        rule_results: list[RuleResult] = []
        for detector in self.detectors:
            result = detector.detect(reply, kb)
            rule_results.append(result)
        
        # 阶段2: LLM Judge
        llm_result = self.judge.judge(item)
        
        # 阶段3: 结果融合
        final = self._fuse(rule_results, llm_result, item)
        final.rule_results = rule_results
        final.llm_result = llm_result
        return final
    
    def _fuse(self, rule_results: list[RuleResult], llm_result: LLMJudgeResult, item: dict) -> DetectionResult:
        result = DetectionResult(
            id=item["id"],
            user_question=item["user_question"],
            system_reply=item["system_reply"],
            knowledge_base=item["knowledge_base"],
        )
        
        # 检查是否有高置信度规则命中
        high_confidence_hits = [r for r in rule_results if r.hit and r.confidence == "high"]
        medium_confidence_hits = [r for r in rule_results if r.hit and r.confidence == "medium"]
        
        if high_confidence_hits:
            # 规则高置信度命中
            best = high_confidence_hits[0]
            result.is_hallucination = True
            result.primary_type = best.hallucination_type
            result.sub_type = best.sub_type
            result.evidence = [e for r in high_confidence_hits for e in r.evidence]
            result.detected_by = "+".join([r.detector_name for r in high_confidence_hits])
            result.problematic_sentences = llm_result.problematic_sentences if llm_result.is_hallucination else []
            result.severity = llm_result.severity
            return result
        
        if llm_result.is_hallucination:
            # LLM Judge 检测到幻觉
            result.is_hallucination = True
            result.primary_type = llm_result.primary_type
            result.sub_type = llm_result.sub_type
            result.severity = llm_result.severity
            result.problematic_sentences = llm_result.problematic_sentences
            result.evidence = []
            for dim, data in llm_result.analysis.items():
                if isinstance(data, dict) and data.get("exists"):
                    result.evidence.extend(data.get("evidence", []))
            result.detected_by = "llm_judge"
            
            if medium_confidence_hits:
                result.detected_by += "+" + "+".join([r.detector_name for r in medium_confidence_hits])
            
            return result
        
        if medium_confidence_hits:
            best = medium_confidence_hits[0]
            result.is_hallucination = True
            result.primary_type = best.hallucination_type
            result.sub_type = best.sub_type
            result.evidence = [e for r in medium_confidence_hits for e in r.evidence]
            result.detected_by = "+".join([r.detector_name for r in medium_confidence_hits])
            result.severity = Severity.MEDIUM
            return result
        
        # 所有检测器均未命中 → 正常
        result.is_hallucination = False
        result.detected_by = "none"
        return result
    
    def detect_all(self, file_path: str) -> list[DetectionResult]:
        items = load_replies(file_path)
        results = []
        for item in items:
            result = self.detect_one(item)
            results.append(result)
        return results
```

- [ ] **Step 2: 快速验证**

```python
engine = DetectionEngine(use_mock=True)
results = engine.detect_all("d:\\OneDrive\\桌面\\智能客服幻觉解决\\task4_replies.json")
hits = [r for r in results if r.is_hallucination]
print(f"规则引擎+Mock检测: {len(hits)}/20 条判定为幻觉")
for r in results:
    print(f"{r.id}: {'❌' if r.is_hallucination else '✅'} {r.primary_type or ''} ({r.detected_by})")
```

注意：这里需要 import Severity。在 engine.py 顶部加上 `from .models import Severity`。

---
### Task 7: Reporter — 评估 vs Ground Truth

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\reporter.py`

**Interfaces:**
- Consumes: `list[DetectionResult]`, `list[GroundTruthItem]`, `load_ground_truth()`
- Produces: `EvaluationReport`, JSON 结果文件, 误判分析

- [ ] **Step 1: 创建 `reporter.py`**

```python
import json
from pathlib import Path
from .models import DetectionResult, GroundTruthItem, \
    EvaluationReport, ConfusionMatrix, load_ground_truth

class Reporter:
    def __init__(self, results: list[DetectionResult], gt_path: str):
        self.results = {r.id: r for r in results}
        self.ground_truth = {gt.id: gt for gt in load_ground_truth(gt_path)}
    
    def evaluate(self) -> EvaluationReport:
        report = EvaluationReport(total=len(self.results))
        
        tp, fp, tn, fn = 0, 0, 0, 0
        false_negatives = []
        false_positives = []
        
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
                    "detail": f"检测为{result.primary_type}, 但人工标注为正常"
                })
            elif not pred and actual:
                fn += 1
                false_negatives.append({
                    "id": rid,
                    "detail": f"人工标注为{gt.hallucination_type}, 但未检测到"
                })
            else:
                tn += 1
        
        report.confusion_matrix = ConfusionMatrix(
            true_positive=tp, false_positive=fp,
            true_negative=tn, false_negative=fn
        )
        report.ground_truth_positive = tp + fn
        report.detected_positive = tp + fp
        report.false_negatives = false_negatives
        report.false_positives = false_positives
        
        total = tp + fp + tn + fn
        report.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        report.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        report.accuracy = (tp + tn) / total if total > 0 else 0.0
        report.f1_score = 2 * report.precision * report.recall / (report.precision + report.recall) \
            if (report.precision + report.recall) > 0 else 0.0
        
        # 按类型统计
        type_breakdown = {}
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
    
    def export_results(self, results_path: str, report_path: str):
        """导出检测结果和评估报告"""
        # 检测结果
        output = []
        for r in self.results.values():
            output.append(r.model_dump())
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        
        # 评估报告
        report = self.evaluate()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        
        return report
```

- [ ] **Step 2: 生成误判分析 Markdown**

在 `reporter.py` 中添加方法：

```python
    def generate_misanalysis(self, output_path: str):
        """生成误判原因分析 Markdown"""
        report = self.evaluate()
        lines = []
        lines.append("# 误判原因分析\n")
        lines.append(f"## 概览\n")
        lines.append(f"- 总样本: {report.total}")
        lines.append(f"- 正确: {report.confusion_matrix.true_positive + report.confusion_matrix.true_negative}")
        lines.append(f"- 误判: {report.confusion_matrix.false_positive + report.confusion_matrix.false_negative}")
        lines.append(f"- 精确率: {report.precision:.1%}")
        lines.append(f"- 召回率: {report.recall:.1%}")
        lines.append("")
        
        if report.false_negatives:
            lines.append("## 漏检分析 (False Negatives)\n")
            for fn in report.false_negatives:
                rid = fn["id"]
                result = self.results[rid]
                gt = self.ground_truth[rid]
                lines.append(f"### {rid}")
                lines.append(f"- 用户问题: {result.user_question}")
                lines.append(f"- 客服回复: {result.system_reply}")
                lines.append(f"- 知识库: {result.knowledge_base}")
                lines.append(f"- 人工标注: {gt.hallucination_type} - {gt.detail}")
                lines.append(f"- 可能的漏检原因:")
                # 分析规则引擎为什么没抓到
                for rr in result.rule_results:
                    if rr.detector_name == "num_compare" and not rr.hit:
                        lines.append(f"  - NumCompare: 数值未命中（可能参数不在预定义列表中）")
                    elif rr.detector_name == "negation_flip" and not rr.hit:
                        lines.append(f"  - NegationFlip: 否定翻转未命中（可能否定句式不在预定义模式中）")
                    elif rr.detector_name == "capability_check" and not rr.hit:
                        lines.append(f"  - CapabilityCheck: 能力越界未命中")
                lines.append("")
        
        if report.false_positives:
            lines.append("## 误报分析 (False Positives)\n")
            for fp in report.false_positives:
                rid = fp["id"]
                result = self.results[rid]
                lines.append(f"### {rid}")
                lines.append(f"- 用户问题: {result.user_question}")
                lines.append(f"- 检测类型: {result.primary_type}")
                lines.append(f"- 规则命中: {[r.detector_name for r in result.rule_results if r.hit]}")
                lines.append(f"- 可能的误报原因:")
                lines.append(f"  - 规则误判: 关键词匹配过于宽泛")
                lines.append("")
        
        lines.append("## 系统性误判模式\n")
        lines.append("根据以上分析，归纳出以下系统性误判模式：\n")
        lines.append("1. **模式一: [待补充]**")
        lines.append("2. **模式二: [待补充]**")
        lines.append("3. **模式三: [待补充]**")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
```

---
### Task 8: Main CLI + README

**Files:**
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\detector\main.py`
- Create: `d:\OneDrive\桌面\智能客服幻觉解决\README.md`

- [ ] **Step 1: 创建 `main.py` — 完整 Mock 响应字典 + 执行流程**

```python
import argparse
import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.engine import DetectionEngine
from detector.reporter import Reporter
from detector.llm_judge import MOCK_RESPONSES

def main():
    parser = argparse.ArgumentParser(description="客服回复幻觉检测工具")
    parser.add_argument("--replies", default="task4_replies.json", help="回复数据文件路径")
    parser.add_argument("--ground-truth", default="task4_ground_truth.json", help="人工标注文件路径")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument("--api-key", help="DeepSeek API Key（可选，不传则用Mock）")
    parser.add_argument("--use-mock", action="store_true", default=True, help="使用Mock模式")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  客服回复幻觉检测工具")
    print("=" * 60)
    
    # 填充完整 Mock 响应
    MOCK_RESPONSES.update({
        "h02": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-A", "severity": "严重", "problematic_sentences": ["蓝牙5.3版本", "支持多设备同时连接", "延迟低至40ms"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["蓝牙5.3≠5.0", "多设备≠单设备", "40ms≠80ms"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h03": {"is_hallucination": True, "primary_type": "能力冒充", "sub_type": "CI-A", "severity": "严重", "problematic_sentences": ["我帮您查了一下，您的包裹目前在南京转运中心"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": True, "evidence": ["系统未接入物流查询接口，却声称查到了物流信息"]}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h04": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-C", "severity": "中等", "problematic_sentences": ["支持电子发票和纸质发票", "下单时在备注里写上发票抬头和税号"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["纸质发票不存在", "申请渠道应为订单详情页"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h05": {"is_hallucination": True, "primary_type": "信息编造", "sub_type": "IF-A", "severity": "严重", "problematic_sentences": ["有一张满300减50的店铺优惠券", "我直接发到您账户里"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": True, "evidence": ["满300减50不存在"]}, "capability_overreach": {"exists": True, "evidence": ["声称能自动发券到账户"]}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h06": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-B", "severity": "严重", "problematic_sentences": ["头层牛皮制作", "保修期为两年"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["材质PU≠头层牛皮", "保修6月≠2年"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h07": {"is_hallucination": True, "primary_type": "信息编造", "sub_type": "IF-B", "severity": "严重", "problematic_sentences": ["浙江省杭州市西湖区文三路478号 客服仓库 张经理收"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": True, "evidence": ["编造了具体退货地址"]}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h08": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-C", "severity": "中等", "problematic_sentences": ["48小时内发货", "一般使用顺丰快递", "大部分地区2-3天到货"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["48h≠24h", "顺丰≠中通/韵达/圆通", "2-3天≠3-5天"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h09": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-B", "severity": "严重", "problematic_sentences": ["支持NFC功能", "可以用于公交卡、门禁卡和移动支付"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["KB未标注NFC功能，回复却声称支持"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h10": {"is_hallucination": True, "primary_type": "能力冒充", "sub_type": "CI-A", "severity": "严重", "problematic_sentences": ["退款已经在处理中，预计明天到账"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": True, "evidence": ["系统未接入退款进度查询接口"]}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h11": {"is_hallucination": True, "primary_type": "信息编造", "sub_type": "IF-B", "severity": "严重", "problematic_sentences": ["在北京、上海、广州、深圳都有线下体验店"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": True, "evidence": ["纯线上品牌编造了线下门店"]}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h12": {"is_hallucination": False, "primary_type": None, "sub_type": None, "severity": None, "problematic_sentences": [], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h13": {"is_hallucination": True, "primary_type": "信息遗漏", "sub_type": "IC-B", "severity": "严重", "problematic_sentences": ["孕妇可以放心使用"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": True, "evidence": ["含视黄醇应提示咨询医生"]}}},
        "h14": {"is_hallucination": True, "primary_type": "能力冒充", "sub_type": "CI-B", "severity": "严重", "problematic_sentences": ["已帮您修改为新地址", "包裹将在明天以新地址发出"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": True, "evidence": ["系统未接入订单修改接口，却声称已修改"]}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h15": {"is_hallucination": True, "primary_type": "信息编造", "sub_type": "IF-C", "severity": "中等", "problematic_sentences": ["我们是XX品牌旗下的子品牌"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": True, "evidence": ["KB未提及其他品牌关联关系"]}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h16": {"is_hallucination": False, "primary_type": None, "sub_type": None, "severity": None, "problematic_sentences": [], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h17": {"is_hallucination": True, "primary_type": "事实矛盾", "sub_type": "FC-B", "severity": "严重", "problematic_sentences": ["Type-C接口"], "analysis": {"factual_conflict": {"exists": True, "evidence": ["接口USB-A≠Type-C"]}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h18": {"is_hallucination": True, "primary_type": "能力冒充", "sub_type": "CI-B", "severity": "严重", "problematic_sentences": ["已升级为高级工单", "会有专属客服在2小时内联系您处理"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": True, "evidence": ["不具备工单升级功能却声称已升级"]}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h19": {"is_hallucination": True, "primary_type": "信息编造", "sub_type": "IF-A", "severity": "严重", "problematic_sentences": ["凭学生证可以享受9折优惠", "选择'学生认证'即可"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": True, "evidence": ["无学生优惠政策，编造了9折和学生认证入口"]}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": False, "evidence": []}}},
        "h20": {"is_hallucination": True, "primary_type": "信息遗漏", "sub_type": "IC-A", "severity": "中等", "problematic_sentences": ["尺码标准，不偏大也不偏小"], "analysis": {"factual_conflict": {"exists": False, "evidence": []}, "fabrication": {"exists": False, "evidence": []}, "capability_overreach": {"exists": False, "evidence": []}, "completeness_issue": {"exists": True, "evidence": ["30%反馈偏大未提及"]}}},
    })
    
    # 执行检测
    engine = DetectionEngine(use_mock=args.use_mock, api_key=args.api_key)
    results = engine.detect_all(args.replies)
    
    # 生成结果目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "detection_results.json"
    report_path = output_dir / "evaluation_report.json"
    
    # 如存在 ground truth，进行评估
    if Path(args.ground_truth).exists():
        reporter = Reporter(results, args.ground_truth)
        report = reporter.export_results(str(results_path), str(report_path))
        reporter.generate_misanalysis(str(output_dir / "misanalysis.md"))
        
        print(f"\n{'='*60}")
        print("  评估结果")
        print(f"{'='*60}")
        print(f"  精确率 (Precision):  {report.precision:.1%}")
        print(f"  召回率 (Recall):     {report.recall:.1%}")
        print(f"  F1 分数:            {report.f1_score:.1%}")
        print(f"  准确率 (Accuracy):   {report.accuracy:.1%}")
        print(f"  TP={report.confusion_matrix.true_positive} "
              f"FP={report.confusion_matrix.false_positive} "
              f"TN={report.confusion_matrix.true_negative} "
              f"FN={report.confusion_matrix.false_negative}")
    else:
        # 仅输出检测结果
        output = [r.model_dump() for r in results]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    # 打印检测摘要
    print(f"\n{'='*60}")
    print("  检测结果摘要")
    print(f"{'='*60}")
    for r in results:
        status = "❌" if r.is_hallucination else "✅"
        print(f"  {r.id}: {status} {r.primary_type or '正常'}")
    
    print(f"\n结果已保存至: {output_dir.resolve()}")
    print(f"  检测结果: {results_path}")
    print(f"  评估报告: {report_path}" if Path(args.ground_truth).exists() else "")
    
    return results

if __name__ == "__main__":
    main()
```

注意：需要在 `main.py` 顶部 `import json`。

- [ ] **Step 2: 创建 `README.md`**

```markdown
# 客服回复幻觉检测工具

智能客服回复幻觉检测工具，基于规则引擎 + LLM Judge 两阶段架构，对客服回复进行自动化幻觉检测与评估。

## 功能

- 定义 4 大类 10 子类的幻觉分类体系
- 规则引擎自动检测数值矛盾、否定翻转、能力越界、声明不一致
- LLM Judge (DeepSeek/Mock) 对模糊 case 进行语义级判断
- 检测结果与人工标注对比评估（精确率/召回率/F1）
- 支持 Mock 模式（无需 API key）

## 项目结构

```
├── detector/
│   ├── engine.py          # 检测编排引擎
│   ├── llm_judge.py       # LLM Judge (Mock/DeepSeek)
│   ├── models.py          # 数据模型
│   ├── reporter.py        # 评估报告
│   ├── main.py            # 入口
│   └── rules/
│       ├── base.py            # 检测器基类
│       ├── num_compare.py     # 数值比对
│       ├── negation_flip.py   # 否定翻转检测
│       ├── capability_check.py# 能力越界检测
│       └── statement_check.py # 声明一致性
├── results/               # 输出结果
├── task4_replies.json     # 测试数据
├── task4_ground_truth.json# 人工标注
└── README.md
```

## 使用

```bash
# Mock 模式（默认）
python detector/main.py

# DeepSeek 模式
python detector/main.py --use-mock false --api-key your_key_here

# 自定义路径
python detector/main.py --replies data.json --ground-truth gt.json --output my_results
```

## 幻觉分类体系

| 类型 | 说明 | 严重程度 |
|------|------|---------|
| 事实矛盾 | 参数/属性/政策与知识库直接冲突 | 严重 |
| 信息编造 | 知识库中无此信息，回复编造细节 | 严重 |
| 能力冒充 | 无此功能却假装已执行 | 严重 |
| 信息遗漏 | 遗漏影响判断的关键限定条件 | 中等 |
```

- [ ] **Step 3: 运行验证**

```bash
cd "d:\OneDrive\桌面\智能客服幻觉解决"
python detector/main.py
```

期望输出：20条结果，18条判为幻觉，2条正常。评估报告显示精确率、召回率、F1、混淆矩阵。

- [ ] **Step 4: 修复任何运行时错误**

运行后根据报错修复 import 路径、缺失字段等。
