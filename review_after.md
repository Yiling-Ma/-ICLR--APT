# 修改后复审

按 `review_before.md` 的同一 benchmark/empirical 标准，不因为自己修改过稿就自动加分；TODO 空白仍不扣分。

| 初审问题 | 状态 | 实际改变与剩余问题 |
|---|---|---|
| 直接 Apt-seq 文献缺失 | 写作已修 | 补 Delley2018，分开测量技术、一般评估原则与临床APT任务；不再暗示首次配对 |
| 低F1来源/对齐 | 部分修复 | 新回归完整cell键对齐PASS；保留反例与全类别；neural cell-ID provenance、oracle解码仍缺 |
| 基线/训练随机性 | 未实验解决 | 未把cappedMLP冒充full-budgetMLP；三seeds缺口保留；不可声称superiority |
| composition增量 | 新实验完成 | direct subtype MAE0.01421，soft0.01478，差值0.00057 CI[-0.00092,0.00211]；没有cell-output优势证据；lineage负结果仍保留 |
| HCE/contrastive复现 | 部分实质修复 | HCE subtree解码通过；补正例质量公式；修正同lineage负例weight实际1.0；当前resolved_config不能单独证明原训练轮数 |
| stacking inner隔离 | 已澄清而未重训 | 明确meta inner交叉依赖与outer-test泄漏不同；没有宣称补做三层nested |
| LODO选择性呈现 | 呈现已修、数据缺口未补 | 明列Normal unavailable，删除五组平均/相对pooled降幅；未声称六组验证完成 |
| 冗余/估计目标 | 已修 | 归档disease cell ranking、legacy bridge、equal-cell大表及head/tail；完整per-class变大字号，负对照不删 |

## 评分

仍为 **5/10，borderline leaning reject**，主观范围约4--6，不是录用概率。表达与可信度改善，但最重要的训练随机性、完整baseline及跨队列可靠性并未由改写解决。新direct对照限制了cell-to-patient路径的必要性，而不是制造positive finding。可接受性的提升主要是减少过度主张和协议歧义，不能据此承诺accept。

## 三个最大剩余风险

1. **可通过写作解决（已明显改善）**：benchmark价值应是可复用的临床APT任务与可检验诊断，不是重新发现patient split重要；避免多条主线和算法优越性期待。
2. **需实验解决**：full-budget MLP与关键神经多seed、cell-ID/probability schema完整核查、true-lineage oracle、重复shuffle及更宽RNA敏感性。现有低F1尚不能排除模型/预处理/标签构造的影响。
3. **需新数据解决**：40人单APT队列与未测acquisition来源限制模态泛化及疾病结论；需要有批次记录、疾病跨批次支持的独立APT数据。不能由COMBAT/OneK1K RNA scaling替代。
