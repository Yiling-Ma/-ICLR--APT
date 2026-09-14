# 层级信号补实验状态

## 冻结范围

以 PROTOCOL.md 为准。已有 v2 四模型错误分析不重训；APT-only
patient x lineage shuffle 先做 MLP 三个打乱版本、每个三个训练 seed；
另做相同参数量的 conditional MLP。主阶段共 20 个 fold/recipe 搜索，
80 个内部候选拟合、60 个最终 refit 预测。复用真实 MLP 的既有结果。
Flat 打乱仅当 MLP 总体主要差值的 95% CI 下界大于零时，按冻结规则
追加复核，最多另有 60 个内部候选拟合和 45 个最终 refit。
可选的更充分 RNA 连接训练未启动，不将旧 capped RNA 当作新实验。

## 已完成

- 五项单元测试通过：条件概率及梯度；分区内整向量打乱可复现且保持组；参数量一致；普通训练循环与 v2 在相同合成输入/seed 下一轮训练权重及预测完全一致；固定 ontology 评分与条件先验矩阵质量正确。
- 新版 MLP、Flat、HCE、Cascade 的全部 OOF 三 seed 重分析完成。
- 八个普通/oracle F1 均值与 v2 scores.csv 一致，容差 1e-12。
- 四个模型的 oracle 最大剩余有向混淆均为 CD8 Tem-1 到 CD4 + Tcm-1；前者有 26,714 个细胞且覆盖全部 40 位患者。完整矩阵保留，不把这个例子解读为排除了支持量或标签因素。
- 完整患者级比例、逐类 F1、普通/oracle 混淆矩阵与图已生成，见 reanalysis/。
- 保存各 seed 全矩阵；热图是 seed 平均可视化，不能由其替代 seed-specific F1 均值。

总体描述性结果（先患者比例、再患者及 seed 均值）：

| 模型 | 跨 parent 占全部错误 | oracle 修复跨 parent 错误 | parent 正确时 fine 跨 parent |
|---|---:|---:|---:|
| MLP | 74.37% | 63.58% | 17.56% |
| Flat | 73.45% | 64.21% | 14.44% |
| HCE | 74.02% | 64.18% | 15.38% |
| Cascade | 73.82% | 64.02% | 15.94% |

条件比例排除分母为零的患者，并在 CSV 报 eligible patient 数。
HCE 的 parent 输出是修正后的 subtree mass，不是独立 coarse-head softmax。
这些比例不是 Macro-F1 的加性分解，也不证明测量上限、标签噪声或可解决机制。

## 正在运行

2026-09-14 14:22 UTC：MLP 阶段 60/60 最终预测完成，独立 source audit
PASS；配对汇总完成。真实 APT oracle 为 0.37557，条件打乱 oracle 为
0.21601，主要差值 +0.15956，95% CI [0.13233, 0.17223]。
Conditional MLP 的可部署分数为 0.13783（普通 MLP 0.13705），
差值 +0.00078，区间 [-0.00565, 0.00559]，尚未建立实际预测优势。
其 oracle 为 0.38901，相对普通 MLP oracle 增量 +0.01344，
区间 [0.00567, 0.02244]。这是特权诊断改善，不是部署收益。
预先规定的 Flat gate 因总体 cell-signal 下界大于零而开启；
GPU 0/1/2 已开始 Flat 的三次打乱复核，尚未完成。所有 lineage
结果保存在 summary_mlp/，不得把总体正差值扩大为每个 lineage 都成立。
论文整合待 Flat 阶段及完整复核后进行。

主机 vllab15，目录 `/ssd2/mayiling/apt_hierarchy_signal`，使用本地磁盘。
GPU 0/1/2 对应打乱 101/211/307，GPU 3 为 conditional MLP。
启动时 GPU 空闲，本地磁盘剩余约 185 GB；本实验不写 home 大型产物。
初始 launcher PID 894188，worker_0.log 至 worker_3.log 保存进度。
finish.sh 等待完整阶段结束，随后独立核查 source indices/scalers/labels/选择历史，
生成配对结果；只有冻结的 MLP gate 通过才运行 Flat 复核。
finish.log 的 ALL_HIERARCHY_SIGNAL_COMPLETE 是计算流水线完成标记，
不代表论文已经更新。训练进行中不得将 validation 分数当作结论。
后处理 PID 902414。已建立每十分钟静默检查的任务监控，仅在阶段完成、
触发/未触发预定 Flat 复核、失败或全部完成时通知。

## 复核与论文更新入口

远端通过 `ssh mayiling@vllab7.ucmerced.edu` 跳转，或本地 ProxyJump 到 vllab15。
不改变 run_signal.py 或 PROTOCOL.md；每个运行记录代码哈希且拒绝不一致续跑。
汇总需要对应 `{model}_source_audit.json` PASS，缺 fold/seed 时失败而非取交集。
summary_mlp/、可选 summary_flat/ 中保存总体和五 lineage 的配对差值。
全部完成后先审查审计、原始记录、分母、区间及失败日志，再更新正文/附录。
未完成实验不得预填结果，不因区间跨零追加 seed。
