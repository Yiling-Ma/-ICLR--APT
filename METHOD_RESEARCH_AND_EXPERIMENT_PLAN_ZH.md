# APT 方法创新与性能改进研究报告

## 研究判断

最值得优先研究的方向是：利用训练阶段的配对 RNA，为 APT 模型提供经过跨患者验证、适配预测粒度的监督，在部署时仍然只输入 APT。核心问题应当具体到“RNA 教师能够识别的细胞差异，哪些能够由 APT 学会并推广到新患者”。这与血液 APT 检测的应用目标直接相连，也能利用已经完成的配对实验。

建议将候选方法暂时描述为 **按可迁移性调整的层级 RNA-to-APT 蒸馏**。这只是开发描述，不代表已经证明算法新颖或性能提高。普通蒸馏、层级损失、可靠性门控、跨患者对比学习均有明确先例；真正需要验证的增量是：以未见患者上的迁移效用为依据，对 lineage 内的细分类监督进行投影与收缩，是否优于现有选择性蒸馏。

检索范围以 2025–2026 年研究为主，并补充决定创新边界的早期文献。资料核验截至 2026-09-11。以下区分正式论文、预印本、已有实验和新提出的研究假设；检索不是穷尽证明，也不能保证 ICLR 接收。

## 一、从现有方法与结果出发

当前正文和实现片段显示，APT 编码器把 293 个特征转换为 token，经过 Transformer 后生成共享细胞表示。Soft-cascade 将全局 subtype logits、lineage 专家分数以及 coarse 路由先验相加。Flat 和 HCE 基线使用相同编码器；仅 soft-cascade 使用疾病标签构造跨疾病对比学习。因而当前对照匹配了 backbone，但没有完全匹配训练目标。换一个名字并不会改变这一证据边界。

当前配对验证文件给出以下结果。这里统一使用该实验自己的 subject-balanced pooled Macro-F1，不与主表的 cell-weighted 指标混用。

| 输入 | MLP Coarse F1 | MLP Fine F1 | SGD Fine F1 |
|---|---:|---:|---:|
| APT | 0.310 | 0.094 | 0.090 |
| RNA | 0.871 | 0.584 | 0.431 |
| RNA + APT | 0.873 | 0.606 | 0.458 |

MLP 的 RNA+APT 相对 RNA fine 增量为 0.0223，配对患者 bootstrap 区间为 [0.0129, 0.0372]；相对患者内打乱 APT 配对的增量为 0.0099 [0.0016, 0.0170]。这些结果支持尝试配对监督，但不说明 APT 蒸馏后能够达到 RNA 的性能，也不是 RNA 标签独立可靠性的验证。

本次核验的依据：

- [配对验证汇总](/tmp/apt-review-current/outputs/paired_information_validation/summary.csv)。
- [完成状态](/tmp/apt-review-current/outputs/paired_information_validation/completion.json)：PASS，100 个 outer fits、200 个 inner fits、40 位测试患者，配对患者对齐通过。
- [配对验证实现](/tmp/apt-review-current/analysis/validate_paired_information.py)：每个训练子集独立进行 RNA 特征选择，训练细胞每患者最多 1,000 个；小型 MLP/SGD，40 epochs，两项正则候选。
- [当前方法定义](/tmp/apt-review-current/main/method.tex) 与 [Flat/HCE 实现](/tmp/apt-review-current/cell_JEPA/apt_jepa/scripts/train_matched_ft_transformer.py)。本地只包含部分模型源码，本报告不声称完成了远程训练代码全量审计。

还有一处需要后续论文维护时修正：当前 [Results](/tmp/apt-review-current/main/results.tex) 对这些更新后的数值仍附带“fixed HVG representation selected during data development”的旧描述，而新版脚本已定义折内 HVG。应依据实际结果 manifest 统一文字；RNA-derived labels 的循环性限制仍然成立。本文不据此把 RNA 称为无偏的“生物学真值”。

## 二、最近研究提供了什么启发

### 1. HCE 与 HACE：层级结构本身已经不是新的空白

Cultrera di Montesano 等的 HCE 于 2026-01-30 发表在 Nature Computational Science，直接把细胞本体结构纳入损失，并在跨研究测试中评估泛化。其关键场景包括不同注释粒度；APT 中所有 fine 标签都是固定叶子，不能把原论文的收益预期直接套过来。[1]

同一研究方向在 2026-05 的 HACE 预印本中进一步结合概率向祖先汇总与祖先标签平滑。这使“加层级 CE/层级平滑”更适合作为强基线，而不是重新命名成主要创新。[2]

对 APT 的启发是保留 hierarchy 作为监督的组织方式，但将新问题放在 **不同层级的监督是否能跨模态、跨患者迁移**。

### 2. MMoCHi：多模态层级细胞分类已有直接近邻

MMoCHi 的正式论文发表于 2025 年 Cell Reports Methods。它依据 marker 识别高置信度训练细胞，再组织层级随机森林分类器；官方工具也支持不同单细胞模态。[3]

这意味着“用 RNA 和表面测量做层级细胞注释”本身不足以构成新贡献。更有区别的任务设定是：配对 RNA 只在训练中出现，外层测试患者的预测完全依赖 APT。适配 MMoCHi 时必须说明输入约束，不能把需要测试 RNA 的系统与 APT-only 部署模型作为同条件比较。

### 3. C²KD：跨模态的软标签也可能不适合直接复制

CVPR 2024 的 C²KD 明确研究模态差异、软标签错配以及教师知识如何适配学生，并引入选择性蒸馏与代理模型。[4] 它是本项目不能绕开的直接算法对照。

RNA 能精确区分两个 subtype，并不意味着 APT 中存在同样的判别依据。直接令 APT 拟合全部 RNA logits，可能得到正则化收益，也可能强迫学生拟合无法观测的差异。这个结果必须通过普通 KD 与选择性 KD 对照测量。

### 4. BioKD：最接近的最新可靠性门控先例

2026-08-06 的 BioKD 预印本已经使用额外生理模态作为训练期教师，部署仅保留视频学生；门控结合教师置信度、历史一致性与分歧，并有 subject-wise 评估。[5]

因此，“跨受试者 + 可靠性门控 + 特权模态”不能直接作为 APT 的独占创新。候选方法必须证明：训练患者内的一致性不足以判断新患者上的收益，而且 coarse 与 sibling-subtype 的迁移效用需要分别估计。BioKD 尚为预印本，作为强近邻比较，不应写成已获顶会认可的结论。

### 5. FreqKD：迁移强度应随信息内容改变

2026-06 的 FreqKD 预印本针对 RGB-to-infrared 蒸馏，依据频段之间的模态一致性差异采用不同监督强度。[6] 其具体频率设计不能直接用于 APT，但启发很明确：教师知识不应被视为同样可迁移的一整块。

APT 中可以研究的对应结构是 broad lineage 与各 lineage 内部的 subtype 差异。二者只是待检验的类比，不应预设“coarse 全部可迁移、fine 全部不可迁移”。

### 6. BioX-Bridge：轻量跨模态迁移已经进入 ICLR 2026

BioX-Bridge 是 ICLR 2026 正式论文，研究通过轻量桥接网络在不同 biosignal 模态与模型之间传递信息，并选择适当的表示对齐位置。[7]

对当前工程的启发是：没有必要立刻引入巨型 RNA foundation model。先用已有小型 RNA 教师和缓存软标签验证机制，可以降低训练与存储成本。BioX-Bridge 的模型与任务不同，不能直接声称可在 APT 上得到同样收益。

### 7. SAFAARI 与 MrVI：跨样本差异不能被简单等同于噪声

2026 年 SAFAARI 结合监督对比学习和对抗域适应做单细胞整合与注释，说明域不变表示与对比学习已有充分相关工作。[8] 其使用目标域数据的适配设定，与完全未见患者的归纳预测需要严格区分。

2025 年 Nature Methods 的 MrVI 则从样本层级异质性出发，研究不同细胞状态中的样本差异。[9] 对 APT 而言，这提醒我们：患者特异信号可能同时包含生物与技术成分。当前没有足够 acquisition metadata，不应把一个对抗损失命名为“去除技术混杂”并当作已证实机制。

### 8. TabM：性能提升可能先来自更合适的 backbone

ICLR 2025 的 TabM 用参数高效的 MLP 集成改善表格学习，提供正式论文与可用 PyTorch 实现。[10] 它是优先级很高的现代性能基线，也可作为蒸馏学生。

对于 293 维连续 APT 输入，建议先比较 TabM、MLP、现有 Transformer 与 XGBoost。使用 TabM 不属于本项目的算法创新；如果新损失只有在更强 backbone 上提升，必须在同一 backbone 内比较增量。

### 9. 单细胞蒸馏与表征学习的其他先例

2025 年 scKAN 已将单细胞教师知识迁移到轻量学生；2022 年 Concerto 已使用自蒸馏与对比学习处理单细胞表征；2024 年 scTab 已系统研究大规模单细胞注释。[11–13] 它们的任务不完全等同于 RNA-to-APT，但足以否定“首次把蒸馏用于单细胞”的宽泛表述。

DKD 则在 CVPR 2022 就已分解目标类与非目标类蒸馏。[14] 若新方法使用按类别分解的 KL，需要明确相对于这一类方法增加了什么，而不是把损失重新拆写视为数学创新。

## 三、三条可以实施的路线

| 路线 | 主要解决的问题 | 工程量 | 创新性判断 | 优先级 |
|---|---|---|---|---|
| 强 backbone 与同预算训练 | 当前编码器是否适合 293 维表格输入 | 较低 | 属于必要基线，不能单独支撑新方法 | 立即作为地基 |
| 按跨患者迁移效用调整 RNA 蒸馏 | 如何利用 RNA 而不盲目复制模态特异信息 | 中等至较高 | 有研究空间，需与 C²KD/BioKD 等区分 | 核心候选 |
| 将细胞类型与类型内状态聚合到患者 | composition 之外是否还保留疾病相关状态 | 中等 | 与已有多实例/样本模型接近，40 患者限制严重 | 第二阶段 |

不建议同时加入新的 Transformer、多个专家、对抗损失、对比损失、图神经网络与患者注意力池化。这样的系统可能提高调参自由度，却使有效机制和实验归因更难判断。

## 四、核心候选方法的具体定义

### 4.1 预测与信息使用边界

每个训练细胞具有 APT 向量 x、配对 RNA 向量 r、subtype 标签 y、父 lineage m(y) 和患者标识 p。外层测试只输入 x，患者标识用于聚合与评估；测试 RNA 不进入预测、特征选择、门控或模型更新。

RNA 教师 T 输出 q_T(y|r)，APT 学生 S 输出 p_S(y|x)。第一版仅使用 RNA 教师，不立即使用 RNA+APT 教师，以便清晰测量监督来源。后续可将相同 RNA+APT 教师用于所有 KD 对照，检查是否更有效。

APT 学生可以保留当前 soft-cascade 的全局与路由结构。但新方法必须同时放到 Flat/MLP 或 TabM 学生上，证明收益不是来自旧架构或不同参数量。第一轮新方法比较中的所有模型统一不使用疾病标签与跨疾病对比项。

### 4.2 将教师监督映射到 APT 可学习的分布

训练一个容量受限的预测器 g(x)，让它仅从 APT 预测 RNA 教师的软分布。记输出为 q_P(y|x)。它可以是正则化小型 MLP，先不引入复杂生成模型。

拟合目标采用患者等权的软标签交叉熵：

\[
\mathcal L_{proj}=\frac1{|\mathcal P|}\sum_{p\in\mathcal P}
\frac1{n_p}\sum_{i:p_i=p}H(q_{T,i},g(x_i)).
\]

g 的输出只有在该患者未参与其训练、上游教师训练和预处理拟合时，才用于声称“跨患者投影”。它是对“APT 能预测的教师输出”的有限模型近似，不是已识别的生物学共同子空间。跨患者分布变化仍可能使它失败。

这里存在一个必须正面面对的理论限制。对于固定数据分布和无限容量，最小化前向 KL 的普通蒸馏本来就以条件均值为最优解：

\[
\arg\min_{p(\cdot|x)}\mathbb E_{r|x}
\operatorname{KL}(q_T(\cdot|r)\|p(\cdot|x))
=\mathbb E[q_T(\cdot|r)\mid x].
\]

因此，加 g 并没有创造额外信息或新的总体最优解。候选贡献只能来自有限数据下的正则化、跨患者目标构建与按粒度收缩，必须与普通 KD、teacher assistant 和直接部署 g 的结果比较。若 g 自身就达到全部收益，就应使用更简单的模型。

### 4.3 在 lineage 内调整细分类监督

令 K_m 为 lineage m 下的 subtype 集合，从完整细分类分布得到：

\[
q_T^L(m)=\sum_{k\in K_m}q_T(k),\qquad
q_T(k|m)=q_T(k)/q_T^L(m).
\]

对 q_P 作同样处理。所有分布在归一化前设置小的正数下限，避免空父节点和数值溢出。定义候选训练目标：

\[
\widetilde q(k)=q_T^L(m(k))
\left[a_{m(k)}q_T(k|m(k))+
(1-a_{m(k)})q_P(k|m(k))\right].
\]

它保留教师 coarse 概率质量，同时在各 lineage 内，在原始 RNA 细分类目标与 APT 预测目标之间收缩。a_m 较大表示更直接地使用 RNA 细分类监督；较小表示更多使用平滑后的 APT 投影。不能先规定哪个 lineage 必须有更大的 a_m。

进一步用层级 KL 的链式分解定义蒸馏损失：

\[
\mathcal L_{KD}=\lambda_L\operatorname{KL}(q_T^L\|p_S^L)
+\lambda_F\sum_m q_T^L(m)
\operatorname{KL}(\widetilde q(\cdot|m)\|p_S(\cdot|m)).
\]

这里 p_S^L 由学生 fine 概率按父节点汇总，保证该损失内使用一致的分布；当前独立 coarse head 可以继续接受真实 coarse 标签监督。不要把“汇总得到的 coarse 预测”和“独立 coarse head 预测”混为同一指标。

当 a_m=1 且两层权重相同时，该分解退化为完整 fine 分布的普通 KL，不能宣称这条分解公式本身是创新。真正待检验的成分是 a_m 的估计方式、投影目标的构建以及跨患者评价。

### 4.4 用未见患者上的收益选择迁移强度

建议先只使用一个全局 a∈{0,0.5,1}，并包含 λ_F=0 的候选。若能重复观察到 lineage 间稳定差异，再引入向全局 a 收缩的五个 a_m。不要一开始为 27 个 subtype 分别自由调参，40 个患者不足以稳健支持这种选择。

系数选择依据内层未见患者的 supervised 预测损失与预先确定的 fine Macro-F1 主指标；教师自信程度、学生与教师一致程度仅作为比较基线。训练损失更小不等于测试 F1 更高，二者需分别报告。

最终目标为真实 coarse/fine 标签监督加上上述 KD，整体以患者等权平均。是否额外采用类别权重应成为独立因子，并让所有对应基线使用相同权重。这个目标是 Macro-F1 的可优化代理，不等价于直接最大化 Macro-F1。

这一方案的差异化假设是：**迁移效用取决于接收模态、预测粒度以及跨患者稳定性，不能仅由教师置信度决定。** 此假设目前尚未在 APT 上验证。

### 4.5 实现时的嵌套规则

1. 固定外层测试患者 E，全部训练、教师缓存、HVG、投影器和选择均限制在开发患者 D 内。
2. 内层模型选择时，验证患者 V 必须从整条上游训练链排除。不能先用全部 D 做教师 OOF，然后把其中预测随意当作不含 V 信息的内层训练输入。
3. 对需要生成目标的患者组 G，在 D\G 中构建教师与投影器。可将该集合再按患者拆为 A、B：教师只在 A 拟合，在 B 上生成 RNA 软目标，投影器用 B 的 APT 拟合这些目标，再对 G 生成 q_P。
4. G 的 RNA 可以供不见 G 的教师生成训练监督 q_T；G 不参加教师、投影器或相关预处理拟合。轮换 G 后获得开发集的诚实目标缓存。每个缓存记录源患者集合和训练配置哈希。
5. 调参时在 D\V 中重复步骤 3–4，再训练 APT 学生并在 V 上评分。选定配置后，使用全部 D 内生成的目标训练最终学生，在 E 上只输入 APT。
6. 对所有 KD 对照使用相同的教师训练数据、缓存策略、学生训练步数与超参数搜索预算。普通 KD 可另列传统 in-sample teacher 作为敏感性对照，但不能偷偷给予不同数据量。

严格嵌套比简单缓存更费计算。可以复用不依赖学生配置的 teacher/projection 缓存来降低成本，但不能复用包含当前验证患者信息的上游缓存。第一轮可先采用诚实的单次 A/B/V 开发划分验证可行性，不能把它当作最终五折结果。

## 五、最小实验组合与成功标准

### 5.1 先建立公平的性能基线

至少比较 XGBoost、MLP、TabM 与当前 Transformer 的 supervised-only 版本。固定患者划分、实际训练细胞、APT 预处理、标签空间和选择预算。可以同时保留“同训练细胞/步数”与“同 wall-clock”两种效率分析，不能把它们混称为同预算。

当前小型 probe 的 APT fine 0.094 不是新的 KD 主对手。正式方法表必须在新学生的相同训练设置下重跑 supervised-only 基线，主表的 0.152 也只能作为历史参考。

### 5.2 核心方法表

| 组别 | 监督配置 | 需要回答的问题 |
|---|---|---|
| A | APT supervised-only | 学生本身能达到什么水平？ |
| B | 同学生 + 普通 RNA logits KD | 配对教师是否已经足够？ |
| C | 同学生 + teacher-confidence KD | 收益是否仅来自教师置信度？ |
| D | 同学生 + C²KD 适配 | 是否超过已有跨模态选择机制？ |
| E | 同学生 + BioKD 可靠性门控适配 | 是否超过最新可靠性控制？ |
| F | 同学生 + 仅投影软目标 | g 是否足以解释全部收益？ |
| G | 同学生 + 全局收缩 | 是否只是简单集成/平滑？ |
| H | 同学生 + 按 lineage 迁移效用收缩 | 粒度适配是否增加价值？ |
| I | 直接部署投影器 g | 额外学生训练是否必要？ |

C²KD/BioKD 的原始任务与本任务不同，适配必须逐项说明保留的机制、不能照搬的部分，以及搜索预算。不能把自行写的 entropy gate 标成“复现 BioKD”。若代码或依赖无法获取，明确标为机制适配并披露限制。

优先在一个学生 backbone 上完成 A/B/C/F/G 的开发比较，排除明显无效方案，再对冻结的候选完成五外折和至少三个独立训练 seeds。最终方法论证应补 D/E/H/I，并至少在第二个学生 backbone 上复现核心增量。

### 5.3 指标和推断

主指标使用 subject-balanced pooled Fine Macro-F1；coarse、root-excluded hierarchical F1、Subtype Tree Distance 与 calibration 为预先指定的次指标。报告各 subtype 的收益与 donor support，避免平均改进掩盖稀有类别恶化。

所有比较使用相同患者、相同 seed 和采样 manifest 的 paired difference。联合重采样匹配的训练 seeds 与 held-out patients；跨方法共用重采样索引。报告联合 95% 区间及 bootstrap 中 Δ>0 的比例，后者不能直接解释为贝叶斯后验概率。

主要比较限定为候选方法对 supervised-only、普通 KD、最强近邻 KD，必要时对统一比较族做 Holm 校正。开发后新增的假设标记为探索性。重复查看旧外层测试集会引入适应性选择；最终候选应尽可能增加一个未用于设计的外部配对数据集。

可以把相对 supervised-only 的 absolute fine F1 +0.02、且相对普通 KD/近邻仍有可复现增量，作为讨论用的工程目标。这是新设的开发门槛，不是性能预测、既有预注册阈值或接收标准。是否继续应结合区间、训练成本及跨数据集重复性决定。

### 5.4 直接检验机制的分析

- 教师正确/错误与学生正确/错误的四象限分析，检查权重是否错误压制“教师正确、学生尚未学会”的困难样本。
- 对各 lineage 比较 RNA 教师置信度、投影器跨患者误差、选择的 a_m 与实际 KD 增量，检验效用估计是否优于置信度。
- 比较相同教师的普通 KD、层级 KL 等权、全局收缩和分 lineage 收缩，排除只因加参数或换损失尺度而提升。
- 保留 CE、标准 label smoothing、相同类别内打乱 teacher 软目标的控制。类别内打乱保留类别软分布，用来检验细胞级配对信息的额外作用。
- 比较 g 的容量与正则强度；若投影结果接近类别先验，则其收益应解释为平滑，不能解释为迁移细胞特异知识。

## 六、用模拟区分“教师可靠”和“知识可迁移”

这一模拟比单纯再增加一个正则项更有助于建立算法机制。使用共享变量 s、RNA 特有变量 u、患者扰动 b_p 构建配对数据，分别控制标签对 s 与 u 的依赖。简单形式为 x_APT=A s+b_p+ε_x，r_RNA=B s+C u+ε_r；coarse 主要由 s 决定，fine 可由 s 与 u 的不同组合决定。

至少预定义四个环境：

| 环境 | RNA 教师 | APT 中标签信息 | 要检验的机制 |
|---|---|---|---|
| 可迁移 | 高质量 | 信息充足 | 新方法应保留普通 KD 的收益 |
| 正确但不可迁移 | 高质量 | fine 依赖 RNA-only u | 高置信度不应自动触发强迁移 |
| 错误且自信 | 受人为标签/shortcut 扰动 | APT 仍有有效信号 | 方法是否继承教师错误？ |
| 跨患者失效 | 训练中 shortcut 有效 | 测试患者 shortcut 改变 | 患者内一致性是否误导权重？ |

不可迁移环境中不能要求任何方法恢复缺失的 u。应通过生成过程或大规模独立 Monte Carlo 基准估计 APT 可达到的风险，并报告该估计自身的误差。不要只挑选使普通 KD 失败的参数；预先覆盖共享/私有信息、患者异质性和噪声的网格，报告完整结果。

普通 KD 在总体极限并不必然因 RNA-only 信息而变差，新方法优势如果存在，预期主要体现在有限数据、模型限制或分布偏移下。这一点应写进模拟设计，避免人为构造一个必胜证明。

## 七、怎样使用 COMBAT 和 OneK1K

APT cohort 用作主要应用：训练使用配对 RNA+APT，测试只使用 APT。若 COMBAT 的当前可用数据确实包含同细胞配对 RNA/ADT，可按相同原则训练 RNA 教师与 ADT 学生，作为第二种测量技术的验证。应先审计 paired cell IDs、donor IDs、可用 ADT 特征和训练标签来源，不能只因为有 COMBAT RNA scaling 结果就认为完成了配对准备。

OneK1K 如果当前只有 RNA，就不能充当真实 RNA-to-protein 迁移的外部验证。它可以用于“全 RNA 教师 → 限定 gene panel 学生”的附加迁移实验，但必须命名为 RNA panel restriction，并让 panel 选择只依赖训练患者。优先级低于一个干净的 COMBAT 配对实验。

跨数据集验证可分别使用各数据集自己的冻结 ontology；无需强行把所有细胞类型映射到 APT 的 27 类。此时验证的是算法适用性，不能声称 APT 训练好的权重能够直接迁移到 ADT。

## 八、如何回到患者表征与疾病任务

类型预测改善以后，先验证 subtype composition 恢复是否改善，而不是直接追求 40 人疾病分类的小幅分数变化。现有独立的 composition recovery 评估可以继续使用，并保留训练患者平均 composition 的常量基线。

如需增加类型内状态，可对患者 p、subtype k 计算 soft composition π_pk 和概率加权的低维细胞状态 h_pk。疾病模型使用 π 与 h 的低维汇总，并与 cell count、QC、lineage composition 进行嵌套增量比较。每个 disease outer fold 内，整条 RNA 教师、投影器和细胞学生链都必须排除该 fold，训练患者特征还需进一步 cross-fit。

这是一项后续实验。40 名患者不适合优先训练大型端到端疾病 MIL 或几十个自由 attention heads；现有 near-perfect disease score 也不能作为新算法的主要成功指标。配对教师或对抗训练都无法在缺少真实 batch/run/date 信息时证明解决了技术混杂。

## 九、工程计划与停止规则

先完成数据/指标和缓存审计，检查真实 APT/RNA 配对、各 fold 的 HVG、训练样本重用、教师置信度以及每类 donor support。使用已有 SGD/小 MLP 作为 RNA 教师，缓存概率即可开始 logits KD，不必同步常驻两个大型模型。

第二步建立 TabM 与当前学生的 supervised-only 基线，再做普通 KD 和全局投影收缩。只有当开发患者上的结果显示可重复的额外收益，才升级到分 lineage 收缩和完整近邻复现。所有候选与失败运行均保留 manifest，避免结果只留下获胜配置。

第三步固定候选，在 APT 和可用的 COMBAT 配对数据上完成五折、多 seeds、paired uncertainty 与训练成本报告。教师训练、HVG、缓存生成、学生训练应分别计时；“推理成本不增加”不等于“总训练成本不增加”。

在没有实测吞吐和 GPU 空闲状态前，不给出确定完成时刻。可用一个代表性 fold 的计时估算：总时间约为不可复用的教师/投影预处理成本，加学生配置数×fold 数×seed 数×单次训练耗时，再除以实际可并行的工作数；还要考虑 I/O 与内存限制。缓存和检查点继续使用远程本地 SSD，避免写满共享 home。

如果 TabM supervised-only 已解释全部增益，应报告更强基线。如果普通 KD 达到候选方法的相同水平，采用简单 KD 并收紧创新主张。如果投影器与门控只改善 calibration，而不改善 fine F1，可以转向明确的可靠性贡献，但必须单独评价，不能把它包装为分类性能提升。

## 十、论文叙事和可用措辞

建议的研究问题是：**训练阶段获得的 transcriptomic supervision，能在多大程度上转化为新患者上的 APT-only 细胞识别能力？** 这样可以保留血液检测的目标，又把方法核心落实到可比较的学习问题。

在实验完成前可以写：

> We investigate whether transcriptomic supervision can improve APT-only cell typing in unseen patients, and whether the utility of this supervision depends on annotation granularity and cross-patient transferability.

若最终证据支持，再写成结果陈述：

> We introduce a distillation procedure that adjusts within-lineage supervision using transfer utility estimated on held-out development patients. The deployed student uses only aptamer measurements.

后面必须跟真实的对照增量、联合区间、外部重复性以及计算成本。不要提前写成“首次”“解决 domain shift”“消除混杂”“恢复所有 RNA 信息”或“临床疾病预测”。

候选标题可采用：**Learning APT-Only Cell Typing from Paired Transcriptomes across Patients**。若方法确实超过近邻，可再在标题中加入具体机制；若结果主要支持 benchmark，则继续使用 APT-Bench 名称。是否加入 DropCascade 应由组件证据决定，不必让旧名称束缚新的研究问题。

“一滴血”的措辞需要真实采样体积、细胞回收和实验流程证据。本算法研究能支持的直接结论是 blood-derived APT profiling 和 APT-only inference，不能由软件分数推断采样体积或临床使用能力。

## 参考文献与核验范围

下列为支持本报告判断的原始论文或作者官方实现。报告中的方法公式、实验设计和优先级排序属于研究建议，并非这些来源已经验证的 APT 结果。

1. Cultrera di Montesano, S. et al. **Improving atlas-scale single-cell annotation models with hierarchical cross-entropy loss.** Nature Computational Science, 2026，在线发表 2026-01-30。[正式论文](https://www.nature.com/articles/s43588-025-00945-z)。核验正文；支持 hierarchy/OOD 的先例及任务差异。
2. Chan, A., D'Ascenzo, D., Cultrera di Montesano, S. **When Labels Have Structure: Improving Image Classification with Hierarchy-Aware Cross-Entropy.** arXiv, 2026-05-07。[预印本](https://arxiv.org/abs/2605.06274)。核验摘要；不声称正式录用。
3. Caron, D. P. et al. **Multimodal hierarchical classification of CITE-seq data delineates immune cell states across lineages and tissues.** Cell Reports Methods 5, 100938, 2025。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11840950/)；[官方文档](https://mmochi.readthedocs.io/en/stable/index.html)。论文检索内容与官方文档交叉核验，PMC 全文直开受到验证页限制。
4. Huo, F. et al. **C²KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation.** CVPR 2024。[原始论文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)。核验摘要及方法概述；为选择性 KD 的直接近邻。
5. Hou, B. et al. **BioKD: Selective Physiology-to-Video Knowledge Distillation via Reliability Gate for Emotion Recognition.** arXiv, 2026-08-06。[摘要及版本](https://arxiv.org/abs/2608.06023)；[全文](https://arxiv.org/html/2608.06023v1)。核验门控、渐进蒸馏与 subject-wise 实验；仍为预印本。
6. Thaker, K. et al. **FreqKD: Frequency-Decoupled Cross-Modal Knowledge Distillation for Infrared Object Detection.** arXiv, 2026-06-10。[预印本](https://arxiv.org/abs/2606.11572)。核验摘要；用于设计启发，不将其视觉结果外推至 APT。
7. Li, C., Liu, Y., Denison, T., Zhu, T. **BioX-Bridge: Model Bridging for Unsupervised Cross-Modal Knowledge Transfer across Biosignals.** ICLR 2026。[正式会议论文页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b4ec8a0ae0373d694dc8529ca523e9a9-Abstract-Conference.html)。核验正式会议记录及摘要。
8. Aminzadeh, F. et al. **SAFAARI: Contrastive Adversarial Open-set Domain Adaptation for Single-cell Integration & Annotation.** Genomics, Proteomics & Bioinformatics, 2026。[原始论文 DOI](https://doi.org/10.1093/gpbjnl/qzag008)；[摘要记录](https://pubmed.ncbi.nlm.nih.gov/41615425/)。核验摘要；不声称逐式复现其方法。
9. Boyeau, P. et al. **Deep generative modeling of sample-level heterogeneity in single-cell genomics.** Nature Methods, 2025-10-13。[原始论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12615264/)。核验正文；支持样本效应与细胞状态分开建模的动机。
10. Gorishniy, Y., Kotelnikov, A., Babenko, A. **TabM: Advancing tabular deep learning with parameter-efficient ensembling.** ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)；[作者实现](https://github.com/yandex-research/tabm)。核验论文摘要和实现说明；不预设 APT 上的优越性。
11. He et al. **scKAN: interpretable single-cell analysis for cell-type-specific gene discovery and drug repurposing via Kolmogorov-Arnold networks.** Genome Biology 26, 300, 2025。[正式论文](https://link.springer.com/article/10.1186/s13059-025-03779-0)。核验蒸馏框架的原文检索片段。
12. **Contrastive learning enables rapid mapping to multimodal single-cell atlas of multimillion scale.** Nature Machine Intelligence, 2022。[正式论文](https://www.nature.com/articles/s42256-022-00518-z)。核验 Concerto 的自蒸馏/多模态摘要；作为早期先例。
13. **scTab: Scaling cross-tissue single-cell annotation models.** Nature Communications, 2024。[正式论文](https://www.nature.com/articles/s41467-024-51059-5)。核验方法与数据预处理说明；属于 RNA 注释相关先例。
14. Zhao, B. et al. **Decoupled Knowledge Distillation.** CVPR 2022。[正式论文页面](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)。核验目标类/非目标类分解；不把本报告的 KL 重写称为独立数学创新。

检索也检查了 2026 年 9 月新近公开的相关条目；未因为发布时间更近就优先推荐不匹配的任务。尚未确认一个与本文完整设定完全一致的现有算法，但不能据此作首创声明。进入投稿前，应围绕最终实际实现再次进行近邻检索。
