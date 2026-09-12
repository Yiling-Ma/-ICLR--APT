# Concern 4 核查与关闭记录

## 结论：resolved（预测失败结构的有限目标）

历史“没有 true-lineage oracle”批评已失效。现有 full-budget MLP、matched-budget APT/RNA 和 Transformer replay 已有 oracle；本次保留全部先验对照，不恢复 chance-adjusted gap 主线。

新增分析仅使用 full-budget Plain MLP 的 3 个训练 seed（20260912、20260913、20260914）各 5 个 fold 的保存分数。每个 seed 均覆盖全部 361,792 个 OOF 细胞、40 位患者、27 个 subtype。没有取交集、筛掉难例或混用 capped-budget 结果。新增训练 0，新增推理 0。

## 核实结果

- 患者平衡的普通预测：正确 33.4%、跨 parent 错误 49.4%、同 parent 内错误 17.3%。parent 来自预测 subtype 的固定映射，不是 coarse head。
- oracle 的总体修正质量为 30.8%，普通正确质量加修正质量得到 oracle 正确率 64.2%。这些不是 Macro-F1 的加性分解。
- 每位患者先计算“修正跨 parent 错误 / 全部跨 parent 错误”，再平均患者和 seed，得到 63.2%；不是两个总体平均数相除。无跨 parent 错误的患者不参与该条件比率；本批次各 scope 的 eligible 患者均有跨 parent 错误。
- 按 seed 平均、患者平衡的非对角质量排序，oracle 剩余最大两项为 CD8 Tem-1 → CD4 + Tcm-1（6.64%）及反方向（2.76%），三个 seed 中分别均排第一、第二。两类均覆盖全部 40 位患者，细胞数分别 26,714、70,653。
- 第三项为 CD16 Monocyte-1 → CD14 Monocyte-1（1.99%，逐 seed 排名 3/3/4）；两类均覆盖 40 位患者。完整矩阵与全部类别同时保留，不仅展示代表性类别。

上述混淆并非只出现在稀有类别，但不构成已控制 abundance 的因果结论。低分与剩余混淆无法区分标签、测量、预处理、预算或模型能力。oracle 的跨 parent 零块由规则保证，不代表模型学会 lineage，更不是信息上限。

## 来源与复现

`analysis/generated/hierarchy_failures.json` 包含 15 个源 NPZ 的绝对路径和 SHA-256、逐 seed 核查结果、矩阵与指标。源位于 vllab11 的 `/ssd3/mayiling/apt_agent_runtime/remaining_v1/full_mlp/`。远端运行 `analysis/hierarchy_failures/run.py`，复用该运行环境的 `patient_cell_scaling` 元数据、固定 ontology 与 fold 记录；stdout 保存为上述 JSON。原始 cell/patient IDs 不在公开聚合输出中重复导出。

本地执行 `python analysis/hierarchy_failures/render.py` 生成表图；执行 `python -m unittest discover -s analysis/hierarchy_failures -p test_analysis.py` 验证输出。需要 NumPy、Matplotlib。

`results/` 文件：

- `error_decomposition_by_seed.csv` 与 `error_decomposition_summary.csv`：总体和各 lineage 的患者等权错误比例及条件修正率；Other 有 38 位 eligible 患者，其余为 40。
- `confusion_matrices.npz`：每种解码的三套 27×27 SB 矩阵及 row-normalized 矩阵，另含固定类别、parent、seed 顺序。SB 矩阵先将每位患者按全部测试细胞数归一化再求和，总质量为 40。
- `ordinary_confusion.pdf` / `oracle_confusion.pdf`：先平均三 seed 的 SB 矩阵，再按行归一化，统一 0–1 色标。热图不是 seed-specific F1 均值的替代计算。
- `per_subtype_by_seed.csv` / `per_subtype_summary.csv`：全部 27 类的支持量、患者覆盖、SB-derived per-class F1 与错误去向。
- `all_confusion_edges.csv`：全部 seed、普通/oracle、27×27 有向边；`top_oracle_edges.csv` 为协议规定的前 3 个残余方向，包含支持量、seed 范围与排名。seed 范围不是置信区间。

## 验证与修改边界

远端逐 fold 检查完整测试 cell ID 集合、唯一性、patient/label 映射、分数列与 parent、有限概率及概率和、已存 ordinary/oracle 混淆矩阵完全一致。逐 seed 验证正确预测保持正确、同 parent 错误预测不变、oracle 无跨 parent 输出、三类比例和为一，以及正确质量加修正质量等于 oracle 正确质量。

重算普通/oracle SB-F1 为 0.140982 / 0.385211（逐 seed F1 后平均），与现有 0.141 / 0.385 一致；从训练患者计数重建 expected-CM prior 为 0.179835，与现有 0.180 一致。未新增显著性检验或 bootstrap。

正文只补少量错误结构观察，保留 matched APT/RNA Figure 2；附录补全错误定义、分母、全 subtype 表与两张完整矩阵。历史 Soft-cascade CW 表保持独立标注。既有 TODO 与 split audit 定位不变。
