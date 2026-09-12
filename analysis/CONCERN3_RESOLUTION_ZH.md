# Concern 3：主模型覆盖与训练协议透明性

## 结论

状态：**resolved（参考训练方案比较的范围内）**。新增训练 **0**。
完整预算 Plain MLP 已在本次工作前进入主表，历史“MLP 缺失”批评不再适用。
本轮未重做实验，而是追溯六个模型的实际产物，补齐协议总表，澄清选择指标与拟合集的真实差异。
没有将结果改写为算法、routing、HCE 或 contrastive learning 的独立贡献。
Concern 1 的两项核心实证发现和 Concern 2 的附录描述性 split audit 保持原有定位。

## 产物核查

- 从 Table 1 对应的 `benchmark_reweighted_metrics.csv` 和 `remaining_full_mlp_v1/summary.csv` 反向追溯。
- LR/XGBoost：读取各任务 pooled OOF CSV，按 cell ID 核对全部测试细胞、患者、标签和 outer fold。
- 三个 Transformer：读取逐 fold split、训练日志、原始 best checkpoint、final checkpoint、配置和带 cell IDs 的 oracle replay。
- 逐一校验 final checkpoint 哈希与 replay sidecar 一致；final 与 best 的全部模型张量相等；best epoch/score 与训练日志中首次验证最大值一致。
- 神经历史 pooled CSV 与 replay 的患者、真实标签和普通预测逐行一致，再由 replay IDs 核对完整测试集。没有取交集或删除无法匹配的行。
- MLP：读取 30 个最终预测 NPZ、30 份 JSON 选模记录和 30 个 checkpoint，涵盖两个任务、五 fold、三个训练 seed。
- 每个 MLP run 的两个 decay 候选均保存完整 50-epoch 验证历史；选定 decay/epoch 符合验证分数规则。
- 核对 MLP checkpoint 的 feature IDs，并用其 outer-development cells 重新计算 StandardScaler，匹配保存的 mean/scale。这里只计算统计量，不训练模型。
- 六个模型均对应相同 361,792 个 keyed 测试细胞、40 位患者、固定 5/27 类。重算的 SB 点估计与 Table 1 的来源值一致。

生成 `generated/main_protocol_audit.json`（详细配置、来源路径及 SHA-256）和
`generated/main_training_protocols.csv`（95 条 model/task/fold/stage/seed 记录）。
95 是记录行数，不是新增拟合次数：MLP inner selection 与 outer refit 分列，Transformer 联合预测 coarse/fine。

## 实际协议差异

| 项目 | LR / XGBoost | Plain MLP | 三个 Transformer |
|---|---|---|---|
| 最终拟合患者 | 32 | 32，重新初始化后 outer refit | 24，直接使用验证选中的 checkpoint |
| 用于选模的患者 | 无独立验证阶段的保留固定方案 | 24 训练 + 8 验证 | 24 训练 + 8 验证 |
| 训练细胞 | 全部分配细胞，无 cap | 两阶段均无 cap | 全部分配细胞，无 cap |
| 验证选择指标 | surviving source 无验证选模 | 对应任务 SB pooled Macro-F1 | fine CW pooled Macro-F1，不是 SB |
| 类别权重 | LR balanced；XGB 无样本权重 | 无权重 CE | 无 weighted sampler；目标函数不同 |
| 训练随机性 | 一套保留预测；原始 XGB CLI seed 未恢复 | 20260912 / 20260913 / 20260914 | 各 fold 保存 seed 42 |
| 额外 disease 监督 | 无 | 无 | 仅 Soft-cascade 用于 contrastive pairing |
| 主表区间 | 固定拟合的 patient bootstrap | joint seed/patient bootstrap | 固定拟合的 patient bootstrap |

精确细胞数（fold 0--4）：

| Fold | 32-patient development | 24-patient inner/Transformer train | 8-patient validation | 8-patient test |
|---|---:|---:|---:|---:|
| 0 | 294812 | 226431 | 68381 | 66980 |
| 1 | 293411 | 207172 | 86239 | 68381 |
| 2 | 275553 | 206704 | 68849 | 86239 |
| 3 | 292943 | 221600 | 71343 | 68849 |
| 4 | 290449 | 223469 | 66980 | 71343 |

MLP 和 Transformer 验证患者是下一 outer fold 的患者；训练患者排除验证及测试患者。
MLP 选模后把验证患者纳入 development refit，但测试患者始终不进入拟合。
classical 没有这个选模阶段；不能把同一 outer fold 理解成相同训练数据量。

## 配置、指标和预算

**输入。** 主表均使用已有 `apt_expression` 的 293 个 APT 特征，不是 capped probe 的 raw-count + log1p。
当前矩阵、元数据、注释、fold 和映射路径及哈希记录在 JSON；不能把当前文件哈希当作原始数据采集时的版本证明。
历史代码用训练细胞的 float32 mean/std，std 小于 1e-6 时置为 1。
MLP 用各阶段独立的 StandardScaler，原始 checkpoint 中的 outer scaler 已实际核验。
原始 classical scaler/estimator 快照未恢复；神经 feature order 由完整 replay 支持，但没有另存的原始 feature-ID manifest。

**LR/XGBoost。** surviving fit source 指定 LR LBFGS、max_iter=500、class_weight=balanced；
类别权重按训练集 N/(K N_k) 计算，不是患者均衡。LR C 未显式设置，不能将当前依赖默认值冒充原始保存配置。
XGBoost source 指定 400 trees、depth 6、lr .1、subsample .8、colsample_bytree .8、hist、mlogloss，无显式样本权重。
这描述的是保留实现；原始 estimator/CLI、完整候选搜索历史和依赖锁定环境 unavailable。
因此不声称实际历史总搜索次数为 1。XGB source default seed=42，不等于独立核实原始 invocation；LR LBFGS 不使用 RNG seed。

**MLP。** 256/128 ReLU、dropout .1、AdamW lr .001、batch 1024、无疾病监督。
每 task/fold/seed 两个 weight decay（1e-5、1e-3）各运行 50 epoch，无提前终止；按验证 SB 分数选择 decay 与 epoch，严格最大值保留最早 tie。
再重新初始化、重拟合 scaler，在全部 32 development patients 上训练所选 epoch 数。
训练 seed 也控制 minibatch 顺序；不存在另一个 capped sampling seed。

**Transformer。** backbone 256 / 4 layers / 8 heads / FF512 / dropout .1；AdamW lr .0001、decay .00001、batch512、clip1、AMP。
15 个 best checkpoint 均保存 maximum_epochs=50。实际 epoch 数分别为：
Flat 19/21/26/19/43；HCE 50/32/26/42/50；Soft-cascade 32/36/39/23/41。
patience=10 不表示实际训练时长相等。Flat/HCE 验证 F1 使用 observed true/predicted label union，Soft-cascade 明确固定全部 27 类。
因此组件表原来“checkpoint rule Same”过于宽泛，本轮已拆开写明。

部分早期 best checkpoint 配置包含 5-epoch head-calibration 预算；它们不能简单当作最终预测配方。
当前 final 模型张量逐项等于 calibration 前的 best 模型，final replay 配置的 calibration_epochs 为 0。
日志也表明曾有 calibration 执行，不能声称全部历史计算只包含表中最终选择阶段。
完整丢弃配置、总计算及历史研究决策记录 unavailable；未从残留配置推断实际跑过多少候选。
这不影响最终产物身份核查，但限制“等调参预算”的主张。

**统计。** 所有主表评分先把每位患者完整混淆矩阵除以该患者测试细胞数，再汇总计算固定类别集 Macro-F1；零分母为 0。
不是 mean within-patient F1，也没有池化不同 seed 的预测来计算 MLP 均值。
历史 P 区间：2,000 次患者抽样，bootstrap seed=42。MLP SP：每次有放回抽 3 个 seed 和 40 位患者，先算各 seed 分数再平均；2,000 次，bootstrap seed=20260912。
训练 seed 与 bootstrap seed 是不同角色，即使整数偶然相同。区间不包含重新选模，不是模型间配对显著性检验。

## 修改与关闭依据

1. §4 引用新的六模型协议 Table 11；附录 B.9 解释实际阶段、输入和产物映射。原组件 Table 3 和 probe Table 4 保留。
2. Table 3 不再把验证类别集合规则写成 Same；方法部分明确 CW 与 SB，删除无保留对照支持的采样优劣归因。
3. 结果保持 reference-recipe comparison：不声称 MLP/Transformer 等效、hierarchy 无效、routing 或 contrastive 单独有效。
4. 新的 `render_main_protocol_audit.py` 是当前六行 Table 1 的数字更新入口；旧 `revise_benchmark_table.py` 标为 pre-MLP 历史阶段，避免误覆盖完整 MLP。
5. 旧 split-audit 记录中的“最大 epoch 未恢复”是之前仅看 resolved_config 的结论；本次读取五个 patient-disjoint best checkpoints 后已恢复为 50。历史 cell-split 的完整训练记录仍不在此次六模型核查范围。

当前核查没有发现需要改变测试集、重新选择 checkpoint、重算不同指标或重训的实质性错误。
这不是声称已恢复所有历史决策：无法恢复的字段已显式保留，且不用于支持 fully matched fairness 或组件因果结论。

## 复现与验证

`audit_main_protocols.py` 在 vllab11 已有运行环境执行，读取本项目已知 checkpoint；不运行 forward/backward，不启动训练。
随后本地运行 `render_main_protocol_audit.py` 和三个专项测试，核验主表 12 个点估计、95 条记录、分区数量、seed 和 checkpoint/选模边界。
三个 Concern 3 专项测试及三个 Concern 2 专项测试均通过。最终 PDF 已编译，
Table 11、Table 3 和主文引用已核查；未发现未定义引用或表格越界。
不修改临床/数据获取 TODO，不新增模型，不为显著性追加 seed。
