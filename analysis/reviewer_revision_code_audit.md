# 代码与证据核查

2026-09-12；当前源码静态核查与 artifact 核查分开，不把可读源码自动等同原训练执行。

| 检查 | 结论 | 证据/边界 |
|---|---|---|
| APT/annotation/patient 对齐 | keyed merge；本轮 direct experiment 一对一验证 PASS | `revision_composition_control.py` 对全部 361792 cells 强制 one_to_one；不外推至没有 cell ID 的神经 pooled exports |
| 主表神经 barcode 顺序 | 未独立验证 | `benchmark_reweighted_audit.json` 明确记录 pooled neural CSV 缺 cell IDs；患者/fold/标签计数一致不是逐 barcode 一致 |
| 原 APT preprocessing | 不声称 raw/log 已全体统一 | `patient_cell_scaling.load_data` 读取提供的 expression，再训练集 z-score；当前 paired probe 另用 raw counts log1p，协议不同 |
| RNA HVG | 当前 prepare_fold_rna.R 使用训练行选择2000 genes | barcode match 到 RDS rownames，inner/outer 独立；旧 `paired_rna_oracle/alignment_audit` 不能代替新协议全矩阵重审 |
| HCE decoding | 当前实现通过 | train_matched_ft_transformer.py decode 使用 hierarchical_scores coarse subtree，fine leaf argmax；这是 adaptation |
| Soft routing consistency | 不保证 coarse argmax=parent(fine argmax) | 独立输出头，附录明确；没有理论保证 |
| Contrastive formula | 原文 additional same-lineage emphasis 不正确，已修 | 远程 cross_disease_supcon.py 用正例 exp 总质量/全有效pairs exp总质量；temperature0.1，valid anchors only；无有效anchor返回0 |
| Contrastive配置 | 五折 resolved snapshots一致 | fold0..4 contrast temperature0.1，negative weight1，warmup5，ramp10，target0.2；故没有额外同lineage负例加权 |
| 原训练轮数 provenance | 仍不完整 | 五折当前 resolved_config train.epochs=0，可能为重评估快照；不能据此断言原来没训练，也不能证明原50epochs训练。保留原协议与实际checkpoint provenance区别 |
| 缺训练类与概率列 | 固定 LabelEncoder，padding使用model.classes_；加载校验不足以证实所有列名 | downstream build_contexts 只检查 probability列数与和；没有重新逐列核对 metadata class_names；oracle前应强化 schema |
| Nested bridge | outer-test隔离；meta inner存在交叉依赖 | tune_c复用inner-OOF特征，生成部分meta-train特征的base fit可能用过meta-validation患者subtype；不是outer-test泄漏 |
| Disease permutation | 不是简单在最终预测上洗标签 | permutation_worker调用evaluate_feature重跑LR与C选择；base不用disease，可固定；但原disease-stratified folds条件下无条件置换exchangeability未建立 |
| 新direct baseline | PASS | all40patients、5*8folds、独立inner标准化/拟合/选择、simplex、旧softMAE重建；small ridge不需要GPU |

本轮没有重新训练主表神经模型，没有改 ontology/folds，没有重写原预测。新实验私有 patient-level NPZ 留在远端独立输出目录，仅 aggregate CSV/JSON 入仓库。

## 文献核实

- [Delley et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5811598/)：Apt-seq 已联合测量 aptamer 和 transcriptome；新增直接引用，不使用首次配对测量主张。
- [Luecken et al. 2025](https://www.nature.com/articles/s41587-025-02694-w)：Open Problems 的 common-task 组织框架；不把其一般框架当 APT-Bench 独创。
- [ICLR 2027 author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines) 与 [reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) 已查阅；主文限制9页，当前主文在限制内；不要求新算法。

本轮核实直接相关原始文献，并非穷尽全领域新颖性检索；不声称已核实整篇所有参考文献内容。
