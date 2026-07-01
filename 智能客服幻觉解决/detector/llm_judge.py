"""LLM Judge — 支持 Mock 模式和 DeepSeek API 模式

注意：Mock 模式不做预知，返回"未检测到"。
这样 Mock 模式下只有通用规则引擎在工作，结果完全来自规则。
使用 DeepSeek API 时才加入 LLM 的语义判断。
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from .models import LLMJudgeResult


def load_env(env_path: str | None = None):
    """从 .env 文件加载环境变量（零依赖实现）"""
    if env_path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / ".env",   # 相对于模块
            Path.cwd() / ".env",                                # 相对于工作目录
        ]
        for p in candidates:
            if p.exists():
                env_path = str(p)
                break

    if not env_path or not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value


# 启动时自动加载 .env
load_env()


# ─── Mock Judge ────────────────────────────────────────

class MockJudge:
    """Mock 模式：返回保守基线（全部未检测到），不依赖数据集"""

    def judge(self, reply_item: dict) -> LLMJudgeResult:
        # 保守返回：不做 LLM 判断，依赖规则引擎
        return LLMJudgeResult(
            is_hallucination=False,
            raw_response="Mock mode: 未调用LLM，仅规则引擎生效",
        )


# ─── DeepSeek Judge ────────────────────────────────────

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


class DeepSeekJudge:
    """DeepSeek API 模式：使用标准库 urllib 调用，零外部依赖"""
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.api_url = "https://api.deepseek.com/chat/completions"

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.api_key != "sk-your-key-here"

    def judge(self, reply_item: dict) -> LLMJudgeResult:
        if not self.available:
            return LLMJudgeResult(is_hallucination=False)

        user_prompt = (
            f"请检查以下客服回复是否存在幻觉：\n\n"
            f"## 用户问题\n{reply_item['user_question']}\n\n"
            f"## 客服回复\n{reply_item['system_reply']}\n\n"
            f"## 知识库信息\n{reply_item['knowledge_base']}\n\n"
            f'请以JSON格式输出检测结果：\n'
            f'{{\n'
            f'  "is_hallucination": true/false,\n'
            f'  "primary_type": "事实矛盾|信息编造|能力冒充|信息遗漏|null",\n'
            f'  "sub_type": "FC-A|FC-B|FC-C|IF-A|IF-B|IF-C|CI-A|CI-B|IC-A|IC-B|null",\n'
            f'  "severity": "严重|中等|轻微|null",\n'
            f'  "problematic_sentences": ["回复中的问题句子"],\n'
            f'  "analysis": {{\n'
            f'    "factual_conflict": {{"exists": bool, "evidence": [str]}},\n'
            f'    "fabrication": {{"exists": bool, "evidence": [str]}},\n'
            f'    "capability_overreach": {{"exists": bool, "evidence": [str]}},\n'
            f'    "completeness_issue": {{"exists": bool, "evidence": [str]}}\n'
            f'  }}\n'
            f'}}'
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            content = body["choices"][0]["message"]["content"]
            data = json.loads(content)

            return LLMJudgeResult(
                is_hallucination=data.get("is_hallucination", False),
                primary_type=data.get("primary_type") or None,
                sub_type=data.get("sub_type") or None,
                severity=data.get("severity") or None,
                problematic_sentences=data.get("problematic_sentences", []),
                analysis=data.get("analysis", {}),
                raw_response=content,
            )
        except Exception as e:
            return LLMJudgeResult(
                is_hallucination=False,
                raw_response=f"Error: {e}",
            )


# ─── 工厂函数 ──────────────────────────────────────────

def create_judge(use_mock: bool = True, api_key: str | None = None):
    """根据参数返回 MockJudge 或 DeepSeekJudge"""
    # 确保 .env 已加载
    load_env()
    if use_mock:
        return MockJudge()
    judge = DeepSeekJudge(api_key=api_key)
    if judge.available:
        return judge
    print("[!] DeepSeek API key 未配置, 降级使用 Mock 模式")
    return MockJudge()
