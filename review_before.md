# 修改前审查

版本：`78e2676`，2026-09-12；源文件为 `/tmp/apt-review-current`，输入是当前约 25 页版本，不是清单中的 38 页版本。未找到适用 AGENTS.md。此次在 `codex/evidence-led-revision` 隔离分支修改。TODO 的未填写不计分、不补造。

## 摘要与优点

该稿已转为 patient-disjoint APT cell-typing benchmark。优点是固定患者划分、明确的 SB 指标、保留 classical/Transformer 参照、配对 RNA 与 shuffled/noise 控制，以及对临床混杂的克制。不能要求 benchmark 必须提出新算法或达到 SOTA。

## 主要问题

| 问题 | 论文位置 | 现有证据 | 已证实问题/待核查疑点 | 对结论的影响 | 最小修复 |
|---|---|---|---|---|---|
| 模态资源价值缺少直接文献定位 | Related Work | 未引用 Delley 2018 Apt-seq | 已证实遗漏；不能声称配对测量首次 | 新颖性边界模糊 | 加直接文献，区分测量技术、临床参考任务和一般评估原则 |
| 低 F1 来源没有充分隔离 | Results/数据附录 | keyed merge；旧 alignment audit PASS；neural pooled export 缺 cell IDs | 并非已发现错配；神经导出无法独立 barcode 复核 | 不能把低 F1 等同 assay 上限 | 报告局部 PASS 与缺口；保留全部 subtype 及反例 |
| baseline 与随机训练覆盖不足 | 主表、reference 方法 | full-budget MLP 缺失；无三 training seeds；Soft-cascade 独有 disease supervision | 已证实 | 不支持架构优越性或 mechanism attribution | 明示 inference/training 区别；E1 独立协议 |
| composition 只突出 subtype 改善 | 主文 secondary context | subtype MAE 改善，lineage 改善 CI 跨零；无 direct median-APT 回归 | 已证实 | 聚合结果不能推出 subtype 对 disease 增量 | 同时报 lineage、TV；补低成本 direct regression |
| stacking 内层不是完全重新 cross-fit | bridge downstream tune_c | 缓存 inner-OOF 特征后在其上调 C；其他训练特征的 base model 可见该 meta-validation fold 的 subtype | 已证实代码依赖，不是 outer-test 泄漏 | inner selection 的隔离表述过强 | 区分 meta 内层交叉依赖与 outer-test 隔离 |
| LODO 缺第六组而比较跨协议 pooled 数值 | LOD 表与正文 | 仅五癌症数值，Normal 文字 near-chance | 当前 artifact 未证实第六组数值 | 不能作完整六组汇总或定量泛化降幅 | 明列 Normal unavailable、移除平均和跨协议降幅 |
| 冗余诊断与陈旧指针 | disease、composition、split、compact | appendix table 仍称 main text；均衡采样近似仍整表；版本变更表插在 composition | 已证实 | 影响聚焦与复现阅读 | 归档可重现旧块，负结果与必要控制保留 |

## 必须回答的问题

1. 新资源价值是匿名 aptamer 向量与 RNA operational labels 在患者外推下的可比较任务，不是 patient split 原理或配对测量的发明；何种新模型能利用这个接口，需要更完整 baseline 支持。
2. HCE 当前 `decode()` 使用 `hierarchical_scores` 的 coarse subtree scores，静态检查通过，不应凭担忧制造 bug。仍须区分当前代码与每个冻结 checkpoint 的执行 provenance。
3. APT-only 是推理契约，不等于只使用 subtype/lineage 训练监督。
4. 疾病置换会重做 LR 调参，base model 不用 disease 因而不必重训；但固定的原 disease-stratified folds 使无条件 exchangeability 不可直接假定。不能凭置换小 p 值证明生物特异性。
5. 对 RNA HVG 新协议与旧 alignment audit 必须区分版本；源代码中的 keyed alignment 不替代所有导出 artifact 的审计。

## 次要问题

摘要略有防御性重复；主文模型公式占据过多 reference 篇幅；per-class 表未显式 CW；数据 TODO 列表出现在 paired probe 下；单 split sampler 仍称 ablation。

## 主观评分

修改前倾向 **5/10，borderline leaning reject**，判断范围约 4--6。不是录用概率，也不是校准预测。依据是 empirically useful but incomplete characterization、训练随机性与 alignment 可验证性，不因为 TODO 空白、没有新算法或没有 SOTA 扣分。

## 已核实指南

[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)：提交主文 9 页；讨论/终稿 10 页，附录另计；AI use statement 必需。页面另有终稿措辞不一致，以明确 submission formatting 段为准。
[Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)：强调严谨、开放、简洁，不要求每篇都竞争既有 leaderboard。
