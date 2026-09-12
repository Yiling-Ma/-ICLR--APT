# 配对 RNA、疾病置换措辞与摘要范围：版本核查

核查基线：Git commit 19393a6，当前 iclr2027_conference.tex 及其实际编译输入。

## 结论：当前 concern 可关闭；无需新增训练

评论中的 +0.022、+0.010、[0.002, 0.017] 对应旧 single-seed pilot，不是当前正文的目标实验。当前 §5.3 / Table 2 已报告三个 training seeds 的配对差值、重复 shuffle 和更宽 RNA 设置。本次进一步将三个 seed 及 patient-shuffle 的完整联合区间写进正文，避免快速阅读时误认为仍是单次拟合。

来源：outputs/cell_identity_questions_v1/hvg2000/completion.json 记录 seeds 20260912、20260913、20260914，每 seed 三次 shuffle；paired_contrasts.csv 保存目标实验差值。tables/paired_increment_main.tex 与 tables/rna_width_repeats.tex 保留目标对照和宽度敏感性，main/identity_questions_appendix.tex 明确逐 seed 计算 F1 后平均及患者/seed/shuffle 的配对重采样。

当前 MLP fine 结果：

- 2,000 HVGs，配对减 RNA：+0.0180，目标实验区间 [0.0092, 0.0297]。
- 配对减 patient shuffle：+0.0088，[0.0014, 0.0145]。
- 配对减 patient-by-lineage shuffle：+0.0064，[-0.00018, 0.01180]，总体 pairing specificity 仍 unresolved。
- 5,000 HVGs，配对减 RNA：+0.0165，[0.0093, 0.0289]；不能将此结果说成已验证更宽 RNA 下的条件 shuffle。

这是探索性分析：设计曾参考 pilot，三个 seeds 仅提供有限训练不确定性证据，不能追认成事前注册实验。更宽 RNA 结果只支持增量不局限于已测的 2,000-HVG 配置，不排除其他 RNA 表征/模型的影响。本次新增训练与推理均为 0。

## 两处历史文字

“This rejects a random patient-label explanation under the fixed protocol”仅存在于 archive/manuscript_before_findings_focus.tex 和 archive/manuscript_before_reviewer_revision.tex 等历史快照，未编入当前论文。疾病分类与 composition 支线已整体归档，因此不将旧段落迁回当前 B.5（现在是 Transformer oracle replay）。归档快照保持历史原貌，不将旧确认性措辞作为当前结论。

摘要的 0.298--0.335 / 0.126--0.145 范围同样只在历史快照中；当前摘要直接报告 oracle/prior 与配对 RNA 两项发现，没有把部分模型范围写成所有参考模型范围。当前主表已有六个模型，不能按旧评论新增“五个模型”的限定。

## 本次修改与验证

仅增强 §5.3 的重复实验与区间解释；保留现有 Table 2、宽 RNA 结果、严格条件 shuffle 的负下界、benchmark 定位和全部 TODO。核查当前编译文本不包含旧置换拒绝句和旧摘要范围，并重新编译、检查引用和相关页面。无需 foundation model 或新一轮显著性驱动训练。
