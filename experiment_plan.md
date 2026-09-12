# 本轮最小实验与验收

2026-09-12；在运行前冻结下述 E3-direct-v1。现有 E1/E2 不假装已完成。

## E3-direct-v1：patient median APT 到 reference composition

- 问题：soft cell-output composition 是否优于不做 cell typing 的直接患者回归？
- 输入：benchmark 同一细胞集的 293 维已提供 APT expression，每患者逐 feature median；不擅自猜测或更改原始 expression normalization。
- 监督：仅开发患者的 RNA-derived composition，coarse/fine 分开；不用 disease 标签训练。
- 划分：原 5 outer folds；outer 开发集内用其余 4 frozen folds 调 ridge alpha。
- 模型：训练折 StandardScaler + multioutput Ridge，alpha 固定 {0.1,1,10,100,1000}；投影到概率 simplex 后按均值 MAE 选最优，平局按列表顺序；outer-test 不参与选择。
- 对照：training-patient mean、同 patient IDs 的 frozen nested soft compositions。全 5/27 类，不删 rare classes。
- 统计：seed 20260912，2000 次 disease-stratified paired patient bootstrap；报告 MAE、TV、direct-vs-soft 配对差异及所有类 bias/MAE；TV=K*MAE/2，不作为独立证据。只反映固定拟合条件下的不确定性。
- 输出只含 aggregate CSV/JSON；不上传 raw profiles、患者级数值或标签。验收包括唯一 cell 键、40 patients、5*8 folds、simplex、inner/outer 分离和已存在 MAE 重建。
- 命令：`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/revision_composition_control.py --root /home/mayiling/projs/apt_agent --output /ssd3/mayiling/apt_agent_runtime/revision_composition_control_v1`。CPU，读取 expression 后数百个小规模 ridge fits；无 GPU 需求，预计内存 1--4GB，时间由数据读盘决定。

## E1：尚未执行
## 执行状态更新（2026-09-12）

下方“尚未执行”段落保留为上一轮审查时的状态，不再代表当前状态。
已实现并在 vllab11 启动 E1、E2、E3；最新冻结方案见
`analysis/REMAINING_EXPERIMENTS_V1.md`。新增 full-budget MLP 使用 AdamW
weight decay 内层选择，取代旧计划中尚未实现的 alpha runner。
E1 已产生首个完整 outer-fold artifact；E2 首个 Flat fold 已通过普通预测
逐项完全重建；E3 已开始三 seed 重复。5000-HVG 敏感性排在2000-HVG后执行。
全部完成前不修改论文数值或称其已完成。

E3 已完成并通过 QA；aggregate 在 `outputs/revision_composition_control_v1/`。远程运行额外使用 `--reference-summary /tmp/summary.csv`，该文件来自已有 composition recovery summary，仅用于验证旧 soft MAE 重建，不参与拟合或选择。

当前没有可验证的 full-budget MLP 三 seed 主表 artifact。需独立 runner、固定 3 seeds (20260912,20260913,20260914)、完整开发患者训练、inner SB-Macro-F1 选 alpha/early stop、保存 ID 与全 ontology 概率。禁止把 capped MLP 或 distillation student 当作该主表 baseline。
已有 `cell_JEPA/apt_jepa/scripts/train_matched_ft_transformer.py` 只支持 flat_ce/hce，不应虚构 MLP CLI。尚无可运行完整 E1 命令；阻塞是缺专用同协议 runner 与冻结试验预算，不是已证明 GPU 不足。预计最低 5 folds*3 seeds*2 tasks=30 outer fits，另有 inner tuning；墙钟时间需一个非 test 试跑测量，不能可靠预报。

## E2：尚未执行

本地主表导出只有 hard predictions/summary；完整 OOF logits、类列顺序和 cell IDs 未验证齐全。需要先导出冻结 checkpoint 的 361792*27 probability 和显式 ID，再按 true parent children mask argmax。不能从 CM 或只取 lineage-correct 子集构造 oracle。
可运行检查入口：`python analysis/revise_benchmark_table.py --help`；这不是 oracle 计算命令。完整 oracle runner 待 score schema 确认后实现，约 39MB float32 scores/model，加 ID 和 coarse 数据。LODO 的 Normal 数值缺 artifact 时明确 unavailable，不补造。

## E3-modality：已有单 seed，不重写成重复验证

`python analysis/validate_paired_information.py --help` 可检查 prepare/run/aggregate 接口。当前脚本 SEED 固定，不能声称命令支持多 seeds；需独立版本增加 seed/shuffle-repeat 参数和不覆盖旧结果的输出目录。更宽 RNA representation 也需 inner/outer HVG 独立重新生成，不是拼接测试折基因。当前结果仅作单 seed 探索性诊断。
