# 修订计划逐项核查与本轮修改

依据：作者提供的 APT_Bench_ICLR_Revision_Plan_CN.docx；核查起点 69bb119。本文区分已有产物、本轮文本修复和新的实验建议。没有把计划中的建议参数或预期结果当作已完成实验，也没有承诺 ICLR 分数。

## E0–E7 对照

| 项目 | 当前证据 | 尚未满足计划的部分 | 本轮处理 |
|---|---|---|---|
| E0 输入/配方 | Concern 3 的 main_protocol_audit.json、协议 CSV 和 keyed replay；已核查部分 cell/feature/label 映射及训练标准化 | supplied apt_expression 的完整上游变换未恢复；统一 raw-count MLP 的单因素 development 实验未完成 | 修正正文过宽的 preprocessing 隔离表述；未将来源未知误写为已发生泄漏 |
| E1 统一主榜 | 六模型 reference-recipe 主表已有 full-budget 三 seed MLP；各历史模型实际 fit/selection 数据见 Table 11 | 不是统一 24 人 selection / 32 人 refit、统一输入和 SB selection 的新榜单 | 保留原主表为参考方案比较，不提前改表名或拼接新旧成绩 |
| E2 强 RNA | 2,000/5,000-HVG、三 seed capped probe；配对增益 0.0180/0.0165 | full-cell-budget RNA、参数容量对照、训练 PCA 对照未建立 | §5.3 与 Discussion 明确输入宽度验证不等于容量/训练充分性验证 |
| E3 条件 shuffle | hvg2000/completion.json：三 training seeds、每 seed 三个 patient 和 patient×lineage realization；完整配对差值 | 现有结果是 capped E3b；APT-only oracle 条件 shuffle 的 E3a 不是现有 prior，也不是 RNA+APT shuffle；统一新配方尚缺 | 明确 E3a/E3b 不可互相替代；保留总体 unresolved，不为跨零追加训练 |
| E4 部署解码 | 同一 full-budget MLP D0/D4 保存分数；27×27 错误分析和 oracle 不变量已完成 | 本轮未建立匹配的 D1/D2/D3/D5 结果；旧 Soft-cascade 不能视为同一 MLP 的解码消融 | 保留现有诊断，不把 oracle 提升写成可部署收益或 routing 归因 |
| E5 泛化 | 固定五折 OOF；历史探索分析已归档 | 没有核实独立 APT 队列或可用 acquisition batch；计划要求的新配方完整内部复验未完成 | 明确现有 folds 已被查看，不能重新命名为 untouched test；不把旧残缺 LODO 恢复为验证 |
| E6 学习曲线 | 历史 scaling 支持报告存在 | 不是 E1/E2 新配方的统一资源曲线；固定总细胞仍需 label-distribution 控制 | 不恢复到核心叙事，不以历史数字冒充新预算控制 |
| E7 标签 | 固定 ontology 和全部 subtype 支持及混淆已核查；RNA-derived 标签限定已在稿件 | 不看 APT 结果的领域专家 marker 复核、独立测量/语义 taxonomy 尚无对应产物 | 不凭名称改 CD4 Monocyte-4，不删除 Erythrocyte，不按测试混淆合并类别 |

## 已有证据的位置

- analysis/generated/main_protocol_audit.json、main_training_protocols.csv 和 CONCERN3_RESOLUTION_ZH.md：主榜实际历史配方及不可恢复字段。
- analysis/CELL_IDENTITY_QUESTIONS_V1.md：已有 exploratory follow-up 规则。它不等于新计划的统一实验冻结文件。
- outputs/cell_identity_questions_v1/hvg2000/completion.json、paired_contrasts.csv：目标条件打乱的重复数与结果；tables/rna_width_repeats.tex：两种宽度。
- analysis/hierarchy_failures/PROTOCOL.md、REPORT_ZH.md、results/：完整三 seed D0/D4 错误分解，支持量与完整矩阵。普通/oracle 是同一分数，不是独立训练的 parent 内分类器。

本次是当前仓库、报告、配置和现有聚合产物核查；没有重新遍历远端所有历史目录，未发现对应产物不等于证明远端从未运行过类似实验。任何后续复用仍须核实 run ID、细胞集合、变换、预算和监督。

## 当前可完成的论文修复

1. 将未经逐探针验证的 target-specific nucleic-acid aptamers 改为 aptamer features。
2. 将可核实的 benchmark-fitted transforms 与来源未完全恢复的 supplied expression 区分，避免推断 training-only StandardScaler 能替上游处理担保。
3. 明确 capped RNA 的 5,000-HVG 验证不能代替 full-budget/容量控制。
4. 明确 APT-only oracle prior、APT-only 条件 shuffle 和可部署 predicted-lineage decoding 是不同问题。
5. 补充既有 folds 的适应性分析背景及 OOF 训练集重叠对区间解释的限制。

## 尚未执行的新实验与依赖

计划的 E0/E1/E2/E3/E4 最低完整方案没有在本轮完成。新增训练 0、推理 0；没有运行中的新作业。不能将这轮润色称为“完成统一实验方案”或“达到 6 分”。

下一阶段应先恢复原始 APT 与 supplied expression 的生成关系并核实特征/条码，再在 development 上完成 E0、记录可行配置和单 fit 资源。之后另存冻结的新协议及全部候选、seed、控制、停止规则，执行 E1/E2，最后衔接 E3/E4。新计划的建议种子不应追认为旧实验的种子；不应仅因旧区间下界接近零而替换原停止规则。

E1/E2 新结果完成之前，不替换当前 Table 1/2。若后续只做 saved-score D1/D2，必须标明旧配方、独立 coarse/fine 模型或共享 heads 的真实情况，不冒称统一训练实验。E5 外部样本和 E7 领域复核需真实输入；本轮不修改原有临床/发布 TODO。
