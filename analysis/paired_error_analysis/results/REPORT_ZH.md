# RNA / RNA+APT 配对错误分析



本次仅增加 analysis，未重新训练模型，未修改论文或 PDF。

## 如何理解结果

APT 的增量伴随大量双向改判，不是只纠正错误而不损害原有判断。部分混淆方向在更宽 RNA 设置下不稳定，因此当前证据更适合描述异质性的纠错结构，而不是给某个 subtype 或 probe 下确定性结论。

## 范围与审计

- 覆盖 2,000 / 5,000 HVG、SGD / MLP、3 个训练 seeds、5 个患者互斥 folds。

- 每个模型设置覆盖 361,792 个唯一细胞、40 位患者、27 个 subtype；重复 seeds 不当作新增样本。

- 检查 barcode、真实标签、预测类别顺序、OOF 覆盖和错误流量守恒。

- 提取 613 个特征列；marker 缺失：[]。

- 细胞级预测、纠错标记、RNA markers、APT 与 QC 使用同一行索引，留在服务器 private 目录；GitHub 只包含聚合结果。

## 主要混淆变化

以下以 2,000-HVG MLP 为主。正差值表示该方向的错分减少，负值表示增加。分母为每位患者该真实 subtype 的细胞数，再对患者及 seeds 等权；不是 Macro-F1。

按效应排序为事后展示，完整 27×27 结果包含在 CSV 中，不能把点式 CI 当作多重检验后的发现。

### 错分减少最多的方向

| true_subtype | predicted_subtype | delta_rna_minus_paired | ci_low | ci_high | rescued_cells_mean | harmed_cells_mean |
| --- | --- | --- | --- | --- | --- | --- |
| platelet | CD8 Tem-2 | 0.0714 | 0.0081 | 0.1347 | 68.0000 | 49.3333 |
| CD14 Monocyte-2 | CD8 Tem-1 | 0.0368 | 0.0131 | 0.0595 | 278.0000 | 154.6667 |
| GCB | CD4 + Tcm-2 | 0.0301 | -0.0005 | 0.0928 | 9.6667 | 20.3333 |
| CD4 Monocyte-4 | CD8 Tem-2 | 0.0287 | 0.0041 | 0.0698 | 13.6667 | 16.3333 |
| Naive T | CD4 + Tcm-1 | 0.0261 | -0.0190 | 0.0922 | 425.3333 | 224.3333 |
| CD8 Tem-1 | CD4 + Tcm-1 | 0.0246 | 0.0119 | 0.0404 | 1394.0000 | 972.6667 |
| CD8 + CTL-2 | CD4 + Tcm-1 | 0.0178 | -0.0516 | 0.0827 | 0.3333 | 1.0000 |
| CD14 Monocyte-3 | CD4 + Tcm-2 | 0.0174 | -0.0071 | 0.0546 | 8.3333 | 12.0000 |

### 错分增加最多的方向

| true_subtype | predicted_subtype | delta_rna_minus_paired | ci_low | ci_high | rescued_cells_mean | harmed_cells_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Erythrocyte | CD8 Tem-2 | -0.1208 | -0.4307 | 0.1667 | 0.0000 | 1.0000 |
| Memory B-2 | Memory B-1 | -0.0515 | -0.1517 | 0.0511 | 311.6667 | 106.0000 |
| GCB | Memory B-1 | -0.0302 | -0.0847 | 0.0140 | 312.6667 | 195.6667 |
| CD14 Monocyte-3 | CD14 Monocyte-1 | -0.0265 | -0.0940 | 0.0463 | 195.6667 | 227.6667 |
| Tcm | CD4 + Tcm-2 | -0.0181 | -0.0568 | 0.0186 | 6.0000 | 8.6667 |
| CD8 + CTL-1 | CD4 + Tcm-1 | -0.0180 | -0.0650 | 0.0307 | 26.3333 | 26.0000 |
| Naive T | CD8 Tem-1 | -0.0146 | -0.0678 | 0.0138 | 35.3333 | 28.6667 |
| Naive T | CD16 NK | -0.0144 | -0.0464 | 0.0012 | 16.3333 | 10.3333 |

错分净变化不等于纠错数减新增错误数：还包含“错误 A 转成错误 B”。CSV 单独保留 rerouted_in / rerouted_out 并验证守恒。

### 直接被纠正的细胞最多来自哪些混淆

下表按直接纠错的平均细胞数排序，与上面的患者等权错分率排序不同；同时报告同方向新增错误，避免把双向改判都算成收益。

| true_subtype | predicted_subtype | rescued_cells_mean | harmed_cells_mean | rescue_patients_union | delta_rna_minus_paired |
| --- | --- | --- | --- | --- | --- |
| CD16 NK | CD56 NK | 1537.3333 | 984.0000 | 40 | 0.0055 |
| CD4 + Tcm-1 | CD16 NK | 1514.3333 | 1622.0000 | 38 | -0.0061 |
| CD16 NK | CD4 + Tcm-1 | 1399.6667 | 1335.0000 | 39 | 0.0038 |
| CD8 Tem-1 | CD4 + Tcm-1 | 1394.0000 | 972.6667 | 37 | 0.0246 |
| CD4 + Tcm-1 | Naive T | 1352.3333 | 943.3333 | 40 | 0.0027 |
| CD4 + Tcm-1 | CD8 Tem-1 | 1306.0000 | 1318.3333 | 30 | -0.0027 |
| CD8 Tem-1 | CD16 NK | 922.3333 | 804.6667 | 36 | 0.0131 |
| CD16 NK | CD8 Tem-1 | 893.0000 | 785.6667 | 31 | -0.0006 |
| CD56 NK | CD16 NK | 853.3333 | 652.3333 | 36 | -0.0135 |
| CD56 NK | CD8 + CTL-2 | 751.3333 | 862.3333 | 19 | 0.0013 |

## 纠错与新增错误总量（每个 seed 的细胞计数取均值）

| width | model | rescued | harmed | rescue_fraction_of_rna_errors | harm_fraction_of_rna_correct |
| --- | --- | --- | --- | --- | --- |
| 2000 | mlp | 31192.3333 | 25158.6667 | 0.2904 | 0.0989 |
| 2000 | sgd | 39751.0000 | 35654.0000 | 0.2420 | 0.1805 |
| 5000 | mlp | 31914.0000 | 25544.0000 | 0.2999 | 0.1000 |
| 5000 | sgd | 38239.3333 | 31499.6667 | 0.2617 | 0.1461 |

这里的比例是 cell-weighted 描述量，不等同于原有报告的 patient-balanced 指标；两种比例分母不同，不能直接相减。

## 更宽 RNA 表征复核

| true_subtype | predicted_subtype | delta_rna_minus_paired_2000 | delta_rna_minus_paired_5000 | ci_low_5000 | ci_high_5000 |
| --- | --- | --- | --- | --- | --- |
| platelet | CD8 Tem-2 | 0.0714 | 0.0167 | -0.0548 | 0.1060 |
| CD14 Monocyte-2 | CD8 Tem-1 | 0.0368 | 0.0117 | -0.0607 | 0.0767 |
| GCB | CD4 + Tcm-2 | 0.0301 | -0.0014 | -0.0219 | 0.0176 |
| CD4 Monocyte-4 | CD8 Tem-2 | 0.0287 | -0.0269 | -0.0693 | -0.0060 |
| Naive T | CD4 + Tcm-1 | 0.0261 | 0.0208 | -0.0180 | 0.0758 |
| CD8 Tem-1 | CD4 + Tcm-1 | 0.0246 | 0.0068 | -0.0068 | 0.0195 |
| CD8 + CTL-2 | CD4 + Tcm-1 | 0.0178 | 0.0114 | -0.0235 | 0.0461 |
| CD14 Monocyte-3 | CD4 + Tcm-2 | 0.0174 | -0.0101 | -0.0429 | 0.0083 |

## RNA / APT / QC 对应

纠错对照：同一患者 × 真实 subtype × RNA 错分标签内，比较被纠正与仍然错分的细胞。新增错误对照：同一患者 × 真实 subtype 内，比较被损害与保持正确的细胞。每组至少 5 个细胞；先在患者内等权汇总 strata，再对患者与 seeds 等权。

所有特征以及符合覆盖门槛的逐混淆边结果均公开于 feature_associations.csv；以下仅展示绝对差值较大的上下文关联，不作显著性筛选。

### rescue / RNA:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| RNA:S100A8 | 0.0400 | 0.0158 | 0.0648 | 40 | 40 |
| RNA:IL7R | -0.0385 | -0.0718 | -0.0086 | 40 | 40 |
| RNA:GZMB | 0.0317 | 0.0120 | 0.0555 | 40 | 40 |
| RNA:FCGR3A | 0.0260 | 0.0016 | 0.0529 | 40 | 40 |

### rescue / APT_raw:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| APT_raw:APT-280 | -0.0542 | -0.0812 | -0.0265 | 40 | 40 |
| APT_raw:APT-161 | -0.0474 | -0.0684 | -0.0280 | 40 | 40 |
| APT_raw:APT-149 | -0.0440 | -0.0666 | -0.0227 | 40 | 40 |
| APT_raw:APT-21 | -0.0427 | -0.0639 | -0.0224 | 40 | 40 |

### rescue / APT_normalized:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| APT_normalized:APT-34 | -0.0580 | -0.0935 | -0.0237 | 40 | 40 |
| APT_normalized:APT-15 | -0.0567 | -0.1044 | -0.0102 | 40 | 40 |
| APT_normalized:APT-182 | -0.0552 | -0.0853 | -0.0261 | 40 | 40 |
| APT_normalized:APT-50 | -0.0537 | -0.0847 | -0.0246 | 40 | 40 |

### rescue / QC:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| QC:RNA_mito_percent | -0.0432 | -0.0706 | -0.0174 | 40 | 40 |
| QC:log1p_APT_total | -0.0365 | -0.0572 | -0.0171 | 40 | 40 |
| QC:log1p_APT_detected | -0.0117 | -0.0185 | -0.0052 | 40 | 40 |
| QC:log1p_RNA_total | 0.0070 | -0.0035 | 0.0176 | 40 | 40 |

### harm / RNA:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| RNA:FCGR3A | -0.0658 | -0.0964 | -0.0354 | 40 | 40 |
| RNA:IL7R | 0.0557 | 0.0342 | 0.0799 | 40 | 40 |
| RNA:GZMB | -0.0512 | -0.0787 | -0.0255 | 40 | 40 |
| RNA:FCER1A | -0.0499 | -0.0655 | -0.0351 | 40 | 40 |

### harm / APT_raw:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| APT_raw:APT-13 | 0.0426 | 0.0176 | 0.0657 | 40 | 40 |
| APT_raw:APT-280 | 0.0401 | 0.0124 | 0.0713 | 40 | 40 |
| APT_raw:APT-190 | 0.0392 | 0.0174 | 0.0616 | 40 | 40 |
| APT_raw:APT-270 | 0.0374 | 0.0180 | 0.0575 | 40 | 40 |

### harm / APT_normalized:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| APT_normalized:APT-270 | 0.0457 | 0.0105 | 0.0814 | 40 | 40 |
| APT_normalized:APT-80 | -0.0415 | -0.0749 | -0.0074 | 40 | 40 |
| APT_normalized:APT-272 | 0.0391 | 0.0056 | 0.0716 | 40 | 40 |
| APT_normalized:APT-124 | 0.0377 | 0.0051 | 0.0670 | 40 | 40 |

### harm / QC:

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| QC:log1p_RNA_total | -0.0575 | -0.0727 | -0.0443 | 40 | 40 |
| QC:log1p_RNA_detected | -0.0390 | -0.0499 | -0.0297 | 40 | 40 |
| QC:RNA_mito_percent | -0.0303 | -0.0710 | 0.0114 | 40 | 40 |
| QC:log1p_APT_total | 0.0206 | 0.0012 | 0.0404 | 40 | 40 |

## 一个逐混淆方向的关联示例

CD8 Tem-1 被 RNA 错分为 CD4 + Tcm-1 的细胞中，将被纠正者与仍然错分者在患者内比较。这个例子事后选自直接纠错较多的方向，不是独立验证集，也不能说明这些特征驱动了纠错。

| feature | difference | ci_low | ci_high | eligible_patients_min | eligible_patients_max |
| --- | --- | --- | --- | --- | --- |
| QC:log1p_RNA_detected | 0.0201 | 0.0051 | 0.0352 | 30 | 30 |
| RNA:CCR7 | -0.0454 | -0.0941 | 0.0034 | 30 | 30 |
| RNA:GNLY | 0.1404 | 0.0603 | 0.2297 | 30 | 30 |
| RNA:NKG7 | 0.0835 | 0.0048 | 0.1616 | 30 | 30 |
| RNA:GZMB | 0.0299 | -0.0056 | 0.0679 | 30 | 30 |
| QC:log1p_APT_total | 0.0447 | 0.0114 | 0.0900 | 30 | 30 |
| APT_normalized:APT-164 | -0.1032 | -0.2086 | -0.0069 | 30 | 30 |
| APT_normalized:APT-172 | -0.0694 | -0.1170 | -0.0241 | 30 | 30 |

该例的 APT 总量方向与全局分层平均不完全一致，进一步说明不能用一个整体 marker/QC profile 解释全部 subtype。

RNA/APT 表达差值单位为各自的 log1p-normalized 或 log1p-count 单位；QC:RNA_mito_percent 是百分比值之差，例如 -0.043 表示 -0.043 个百分点，不是 -4.3%。

## 必须保留的限制

- 这是事后、基于预测结果筛选细胞的关联分析，不是 APT 特征因果贡献，也不是 SHAP/消融。RNA markers 不能独立验证 RNA 来源的标签。

- 本轮未进行每个 probe 的干预，也不能证明纠错来源于生物信号而非技术因素。未提供 probe 靶标映射，因此只报告 APT 编号。

- APT raw 与 library-normalized 视图用于观察测量总量敏感性；它们不能替代 acquisition metadata。QC 差异也不能确诊 doublet 或低质量。

- 点式 95% CI 未做全特征/全混淆边多重校正。5-cell 分层阈值会排除稀疏 strata；覆盖详见 feature_matching_coverage.csv。

- 纠错的具体细胞配对价值仍应结合 patient×lineage shuffle 对照解释；本分析不把该对照的不确定结果改写成确定机制。

- Marker 面板仅用于 PBMC 上下文，不是 27-subtype 的完整判定规则。[来源：Seurat 官方教程](https://satijalab.org/seurat/articles/pbmc3k_tutorial.html)。

## 文件

- all_confusion_edges.csv：全部方向、原始错分、纠错、新增错误、错误转移、患者覆盖和联合区间。

- subtype_seed_counts.csv：所有 subtype、所有 seeds 的原始计数。

- feature_associations.csv：全局分层及支持充分的混淆边对应 RNA/APT/QC 关联。

- feature_matching_coverage.csv / audit.json：分层保留率、barcode 和 QC 审计。

- confusion_change_*.png / bidirectional_error_changes.png / conditional_feature_associations.png：可视化。
