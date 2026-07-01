# 客服回复幻觉检测工具 — 设计文档

## 1. 背景

智能客服系统偶尔会生成与知识库不符的"幻觉"回复——编造不存在的优惠政策、给出错误的地址、杜撰产品参数。需开发一个自动化检测工具，批量检测 20 条测试回复中的幻觉。

## 2. 数据源

- `task4_replies.json`: 20 条客服对话（user_question + system_reply + knowledge_base）
- `task4_ground_truth.json`: 人工标注结果（仅用于最终评估，不参与分类体系设计和检测开发）

## 3. 幻觉分类体系（纯数据驱动）

基于 reply vs knowledge_base 的差异模式分析，定义 4 大类、10 子类：

### 3.1 事实矛盾 (Factual Conflict) — 严重

| 子类 | 说明 | 检测信号 |
|------|------|---------|
| FC-A 数值/版本错误 | 数字、版本、时间信息直接矛盾 | 数值不一致 |
| FC-B 属性描述错误 | 材质/功能/接口等定性属性矛盾 | 属性词冲突 |
| FC-C 政策规则错误 | 退货/发货/发票规则与政策不符 | 政策声明冲突 |

### 3.2 信息编造 (Information Fabrication) — 严重

| 子类 | 说明 | 检测信号 |
|------|------|---------|
| IF-A 优惠/政策编造 | 不存在促销或政策 | KB 无此信息 |
| IF-B 实体/渠道编造 | 杜撰地址/门店/快递 | KB 无此信息 |
| IF-C 关联关系编造 | 编造品牌关系 | KB 无此信息 |

### 3.3 能力冒充 (Capability Impersonation) — 严重

| 子类 | 说明 | 检测信号 |
|------|------|---------|
| CI-A 查询冒充 | 无查询接口却声称查到数据 | KB:"无接口" + reply:"我帮您查了" |
| CI-B 操作冒充 | 无操作能力却声称已执行 | KB:"无功能" + reply:"已帮您" |

### 3.4 信息遗漏 (Information Completeness) — 中等

| 子类 | 说明 | 检测信号 |
|------|------|---------|
| IC-A 条件遗漏 | 遗漏关键限定条件 | reply 结论在 KB 中有前提条件 |
| IC-B 安全提示遗漏 | 遗漏安全警示 | KB 含健康/安全警告 |

## 4. 检测架构：两阶段流水线

```
回复数据 → 规则引擎 (4个检测器) → LLM Judge (DeepSeek/Mock) → 结果融合 → 报告输出
```

### 4.1 阶段一：规则引擎 (Rule Engine)

4 个并行检测器：

| 检测器 | 方法 | 捕获类型 |
|--------|------|---------|
| `NumCompareDetector` | 正则提取 reply/KB 中数字对，检查 key 同值不同 | FC-A |
| `NegationFlipDetector` | KB 含"无/不支持"时 reply 对应位置是肯定 | FC-B/FC-C/IF-A |
| `CapabilityCheckDetector` | KB 含"未接入/无此功能"时 reply 含"我帮您/已为您" | CI-A/CI-B |
| `StatementConsistencyDetector` | 提取回复声明 + KB 语义比对 | 全覆盖补充 |

输出: `{"hit": bool, "type": str, "evidence": [str], "confidence": "high"|"medium"}`

### 4.2 阶段二：LLM Judge

- 模型：DeepSeek API（兼容 OpenAI SDK 格式）
- 调用频次：每个 case 调用 1 次（共 20 次）
- 结构化输出 Schema：

```json
{
  "is_hallucination": true/false,
  "primary_type": "事实矛盾|信息编造|能力冒充|信息遗漏",
  "sub_type": "FC-A|...",
  "severity": "严重|中等|轻微",
  "problematic_sentences": ["..."],
  "analysis": {
    "factual_conflict": {"exists": bool, "evidence": ["..."]},
    "fabrication": {"exists": bool, "evidence": ["..."]},
    "capability_overreach": {"exists": bool, "evidence": ["..."]},
    "completeness_issue": {"exists": bool, "evidence": ["..."]}
  }
}
```

- Mock 模式：内置预置响应，API key 不可用时自动降级

### 4.3 阶段三：结果融合

| 规则引擎 | LLM Judge | 最终判定 |
|---------|-----------|---------|
| 命中 (high) | 确认 | 采纳，置信度最高 |
| 命中 (high) | 否认 | 保留规则结果，注明分歧 |
| 未命中 | 检测到 | 采纳 LLM 结果 |
| 未命中 | 未检测到 | 正常 |

## 5. 验证方案

用 ground_truth.json 计算：

- **精确率**: 检测为幻觉且实际是幻觉 / 检测为幻觉总数
- **召回率**: 检测为幻觉且实际是幻觉 / 实际幻觉总数
- **F1 分数**: 2 × (精确率 × 召回率) / (精确率 + 召回率)
- **混淆矩阵**: 各类别漏检/误报明细

## 6. 输出产物

1. `detector/`: 检测引擎 Python 包
   - `engine.py` — 主引擎
   - `rules/` — 规则检测器
   - `llm_judge.py` — LLM Judge
   - `reporter.py` — 报告生成
2. `results/`: 检测结果
   - `detection_results.json` — 逐条检测结果
   - `evaluation_report.json` — 检出率评估
   - `misanalysis.md` — 误判原因分析
3. `README.md`: 项目说明

## 7. 技术栈

- Python 3.10+
- 依赖: openai (DeepSeek API 兼容), pydantic
- 零外部数据库依赖
