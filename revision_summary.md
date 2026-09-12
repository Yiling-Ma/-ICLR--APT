# 本轮修改与验收说明

执行用户 Markdown 的先审稿、后修改、再复审流程。基线版本78e2676，隔离分支codex/evidence-led-revision。TODO未填写不作为评分缺陷，未编造填写。

## 已完成

- 补直接Apt-seq文献，三段引言区分已有测量技术、通用评估原则与APT资源贡献。
- 明确APT-only只描述推理；Soft-cascade有额外disease训练监督，HCE是adaptation且coarse用subtree scores。
- 远程核实contrastive损失与五折resolved配置：温度0.1、无正例anchor跳过、同lineage负例权重1而非额外强调、warmup5/ramp10；补准确公式。
- 执行E3-direct-v1，使用vllab11 CPU（经vllab7），不占GPU；提供patient-median APT ridge-to-simplex对照。subtype MAE0.01421，对soft0.01478的配对差异未解决；完整保留lineage负结果、全部类别和TV恒等式。
- 疾病结果留附录，去掉多模型cell-level排名；保留QC条件差异、centering、patient fingerprint等必要限制。
- LODO明确六组中的Normal artifact未验证；不造数、不把五癌症平均与pooled5fold相减。
- 归档旧version comparison、head/tail、equal-cell Monte Carlo大表、冗长compact-panel文字；原代码、表、日志、预测未删除。原稿快照在archive/manuscript_before_reviewer_revision.tex，CSV提供逐项索引。
- 完整27-subtype表删除冗余显示列以增大字号，不删除任何subtype；保留Plasma/Memory B/CTL反例。

## 验证与边界

新实验qa验证361792个唯一细胞、40患者、5个8人fold、inner与outer分离、simplex约束、旧soft MAE重建；本地check_reviewer_revision.py检查TV与MAE恒等式、32类三方法完整性及关键控制保留。主表与paired probe严格区分SB/CW和训练预算。

E1、E2、重复shuffle、更宽RNA representation没有完成；见experiment_plan.md，不能将建议写成Results。源码静态检查不能替代冻结checkpoint provenance。病例数与clinical confounding不是TODO空白问题，而是现有证据边界。

新实验原始patient-level NPZ留在远程独立目录，仓库仅含aggregate结果。数据、模型、旧runs均未终止或删除。

文献与官方指南来源见review_before.md及analysis/reviewer_revision_code_audit.md。复审见review_after.md，未自动上调主观评分。

最终 PDF 为24页，主文6页。已渲染检查全篇，修复附录强制分页导致的大块空白，检查跨页27类表的表头和续页；LaTeX日志无overfull、undefined reference或warning。本地数值保留检查、Python编译检查和git diff --check通过。

后续执行更新：E2 true-lineage oracle 已完成15个checkpoint，普通预测精确重建，
数值与原主表SB fine一致。结果新增至正文及附录；该控制使用真实lineage，
不是可部署性能。E1多seed全预算MLP与E3重复模态控制已启动，未提前写结果。
新增诊断并清理末页孤立段落后，PDF为23页，主文仍6页，编译与视觉检查通过。
