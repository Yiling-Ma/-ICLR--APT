# 统一主表与消融实验 v2

本轮已实现并启动，不是只修改论文措辞。历史主表和两项已完成诊断保留，不能在新结果齐全前替换。

## 冻结设计

- 六行主表：LR、XGBoost、MLP、Flat、HCE、无 disease contrastive 的 Cascade。
- 独立 2×2 表：Flat、Flat+contrastive、Cascade、Cascade+contrastive。前后两张表复用相同 Flat/Cascade 产物，不重复训练。
- 层级诊断：同一保存概率计算普通 D0、预测 parent 限制 D1、概率一致组合 D2、真实 parent D4；D4 始终 privileged。
- 同一 raw count/log1p/训练分区 StandardScaler；同一 24/8 内部划分、32 人 refit、8 人 outer test；所有训练细胞，无 class/patient 权重。
- 选择 seed17；随机模型 refit17/29/43。LR 确定性，不伪造重复 seed。
- LR/XGB 两个独立任务；神经网络 fine CE + .5 coarse CE，fine SB-F1 选型，coarse 是同 checkpoint 辅助成绩。这是明确的任务合同差异，不能称 coarse 独立优化完全对等。
- Transformer 共用现有 tokenizer/backbone 的 128维、2层、4头、FF256 配置。该新预算配置不是历史 256维/4层的原样重放。各模型有四组 lr/decay 内部搜索；上限50epoch、patience8、选择epoch后从头refit。
- 整套 cascade 含 global/expert/compatibility 项；C-A 不单独解释为 routing。所有 Transformer wrapper 保留共同 projection，以及 Flat 中不用于 forward 的 cascade expert 参数，因此成本和有效参数量不能混为一谈。

## 实现与产物

PROTOCOL.md 在新 outer 结果生成前冻结。run.py 保存输入哈希、各阶段人数/细胞数、scaler、选择历史、checkpoint、完整 cell/patient IDs、真实标签和概率。summarize.py 缺任何模型/seed/fold 即失败，不输出残缺主表；验证完整 OOF 集合，逐 fit SB-F1 后平均，生成配对差值及 2×2 交互；5000次共同 patient/seed resampling。

运行：vllab11，CUDA_VISIBLE_DEVICES=4，单 worker，CPU线程4。
目录：/ssd3/mayiling/apt_agent_runtime/unified_v2
日志：worker.log；结果：results/；代码快照：code/。
启动时已确认主进程及 MLP fold0 进程存在。其后进度须以实时日志/sidecar为准，不能把启动当作完成。

三个训练host测试通过：所有神经配置输出形状及梯度、contrastive有限值/无正样本处理、固定27类SB计分。无需重写临床/发布TODO。

## 尚待完成

训练、完整 OOF 验证、汇总及三张新结果表仍待完成。没有预填胜负或承诺性能。未实施 E0 因素网格、强RNA/E2、E3额外shuffle、独立队列等不同任务。当前冻结 raw-count 定义不基于 outer 分数选取。
CPU/GPU训练时间只作为分项复现记录，不直接在主表排名。完成后仍需检查收敛警告、全部失败运行和论文表图，不能仅以脚本退出零宣布科学结论成立。
