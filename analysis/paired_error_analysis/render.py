"""Render public aggregate diagnostics; never read private cell-level data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        values=[]
        for c in columns:
            v=row[c]
            values.append(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
    out=parser.parse_args().directory
    edges=pd.read_csv(out/'all_confusion_edges.csv')
    features=pd.read_csv(out/'feature_associations.csv')
    counts=pd.read_csv(out/'subtype_seed_counts.csv')
    audit=json.loads((out/'audit.json').read_text())
    off=edges[edges.true_subtype!=edges.predicted_subtype].copy()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    vmax=max(abs(off.delta_rna_minus_paired).max(),.01)
    for (width,model),df in edges.groupby(['width','model']):
        mat=df.pivot(index='true_subtype',columns='predicted_subtype',values='delta_rna_minus_paired')
        matrix=mat.to_numpy().copy(); np.fill_diagonal(matrix,0)
        fig,ax=plt.subplots(figsize=(13,11),layout='constrained')
        im=ax.imshow(matrix,cmap='RdBu',vmin=-vmax,vmax=vmax)
        ax.set_xticks(range(len(mat)),mat.columns,rotation=75,ha='right',fontsize=9)
        ax.set_yticks(range(len(mat)),mat.index,fontsize=9)
        ax.set_xlabel('Predicted subtype'); ax.set_ylabel('True subtype')
        ax.set_title(f'{model.upper()}, {width:,} RNA HVGs: change after adding APT\n'
                     'Blue: fewer errors; red: more errors; diagonal suppressed')
        fig.colorbar(im,ax=ax,label='RNA error rate minus RNA+APT error rate\n(patient/true-subtype normalized)')
        fig.savefig(out/f'confusion_change_{width}_{model}.png',dpi=170); plt.close(fig)
    main_edges=off[(off.width==2000)&(off.model=='mlp')].copy()
    supported=main_edges[(main_edges.true_class_patients>=5)&
                         (main_edges[['rna_cells_mean','paired_cells_mean']].max(axis=1)>=20)]
    ordered=supported.sort_values('delta_rna_minus_paired',ascending=False)
    selected=pd.concat([ordered.head(8),ordered.tail(8)]).drop_duplicates(['true_subtype','predicted_subtype'])
    selected=selected.sort_values('delta_rna_minus_paired')
    fig,ax=plt.subplots(figsize=(12,9),layout='constrained')
    labels=selected.true_subtype+' -> '+selected.predicted_subtype
    for n,(_,row) in enumerate(selected.iterrows()):
        color='#176c9e' if row.delta_rna_minus_paired>=0 else '#b84c35'
        ax.plot([row.ci_low,row.ci_high],[n,n],color=color,lw=2)
        ax.scatter(row.delta_rna_minus_paired,n,color=color,s=35)
    ax.set_yticks(range(len(selected)),labels,fontsize=10); ax.axvline(0,color='.4',ls='--')
    ax.set_xlabel('Reduction in directed confusion rate (positive = fewer errors)')
    ax.set_title('MLP, 2,000 HVGs: strongest reductions and increases\n'
                 'Post-hoc ranking; paired seed/patient 95% intervals are pointwise')
    fig.savefig(out/'bidirectional_error_changes.png',dpi=180); plt.close(fig)
    globalf=features[(features.width==2000)&(features.scope=='all_eligible_strata')]
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for i,kind in enumerate(['rescue','harm']):
        for j,prefix in enumerate(['RNA:','APT_normalized:','QC:']):
            df=globalf[(globalf.contrast==kind)&globalf.feature.str.startswith(prefix)].copy()
            df=df.loc[df.difference.abs().nlargest(6).index].sort_values('difference')
            ax=axes[i,j]
            for n,(_,row) in enumerate(df.iterrows()):
                ax.plot([row.ci_low,row.ci_high],[n,n],color='#176c9e',lw=2)
                ax.scatter(row.difference,n,color='#176c9e',s=30)
            ax.set_yticks(range(len(df)),df.feature.str.replace(prefix,'',regex=False),fontsize=9)
            ax.axvline(0,color='.5',ls='--'); ax.set_title(f'{kind}: {prefix.rstrip(":")}')
            ax.set_xlabel('Conditional mean difference (feature-specific units)')
    fig.suptitle('Outcome-associated features, not causal explanations or feature importance\n'
                 'Matched within patient and subtype; rescue also matches the RNA error',fontsize=13)
    fig.savefig(out/'conditional_feature_associations.png',dpi=170); plt.close(fig)
    text=['# RNA / RNA+APT 配对错误分析','',
          '本次仅增加 analysis，未重新训练模型，未修改论文或 PDF。',
          '## 如何理解结果',
          'APT 的增量伴随大量双向改判，不是只纠正错误而不损害原有判断。部分混淆方向在更宽 RNA 设置下不稳定，因此当前证据更适合描述异质性的纠错结构，而不是给某个 subtype 或 probe 下确定性结论。',
          '## 范围与审计',
          '- 覆盖 2,000 / 5,000 HVG、SGD / MLP、3 个训练 seeds、5 个患者互斥 folds。',
          '- 每个模型设置覆盖 361,792 个唯一细胞、40 位患者、27 个 subtype；重复 seeds 不当作新增样本。',
          '- 检查 barcode、真实标签、预测类别顺序、OOF 覆盖和错误流量守恒。',
          f'- 提取 {audit["features"]} 个特征列；marker 缺失：{audit["missing_markers"]}。',
          '- 细胞级预测、纠错标记、RNA markers、APT 与 QC 使用同一行索引，留在服务器 private 目录；GitHub 只包含聚合结果。',
          '## 主要混淆变化',
          '以下以 2,000-HVG MLP 为主。正差值表示该方向的错分减少，负值表示增加。分母为每位患者该真实 subtype 的细胞数，再对患者及 seeds 等权；不是 Macro-F1。',
          '按效应排序为事后展示，完整 27×27 结果包含在 CSV 中，不能把点式 CI 当作多重检验后的发现。',
          '### 错分减少最多的方向',
          table(ordered.head(8),['true_subtype','predicted_subtype','delta_rna_minus_paired','ci_low','ci_high','rescued_cells_mean','harmed_cells_mean']),
          '### 错分增加最多的方向',
          table(ordered.tail(8).sort_values('delta_rna_minus_paired'),['true_subtype','predicted_subtype','delta_rna_minus_paired','ci_low','ci_high','rescued_cells_mean','harmed_cells_mean']),
          '错分净变化不等于纠错数减新增错误数：还包含“错误 A 转成错误 B”。CSV 单独保留 rerouted_in / rerouted_out 并验证守恒。',
          '### 直接被纠正的细胞最多来自哪些混淆',
          '下表按直接纠错的平均细胞数排序，与上面的患者等权错分率排序不同；同时报告同方向新增错误，避免把双向改判都算成收益。',
          table(main_edges.nlargest(10,'rescued_cells_mean'),['true_subtype','predicted_subtype','rescued_cells_mean','harmed_cells_mean','rescue_patients_union','delta_rna_minus_paired']),
          '## 纠错与新增错误总量（每个 seed 的细胞计数取均值）']
    totals=counts.groupby(['width','model','seed'])[['cells','rna_correct','paired_correct','rescued','harmed']].sum()
    avg=totals.groupby(['width','model']).mean().reset_index()
    avg['rescue_fraction_of_rna_errors']=avg.rescued/(avg.cells-avg.rna_correct)
    avg['harm_fraction_of_rna_correct']=avg.harmed/avg.rna_correct
    text += [table(avg,['width','model','rescued','harmed','rescue_fraction_of_rna_errors','harm_fraction_of_rna_correct']),
             '这里的比例是 cell-weighted 描述量，不等同于原有报告的 patient-balanced 指标；两种比例分母不同，不能直接相减。',
             '## 更宽 RNA 表征复核']
    rep=main_edges.merge(off[(off.width==5000)&(off.model=='mlp')],on=['true_subtype','predicted_subtype'],suffixes=('_2000','_5000'))
    keys=ordered.head(8)[['true_subtype','predicted_subtype']]
    text += [table(keys.merge(rep,on=['true_subtype','predicted_subtype']),['true_subtype','predicted_subtype','delta_rna_minus_paired_2000','delta_rna_minus_paired_5000','ci_low_5000','ci_high_5000']),
             '## RNA / APT / QC 对应',
             '纠错对照：同一患者 × 真实 subtype × RNA 错分标签内，比较被纠正与仍然错分的细胞。新增错误对照：同一患者 × 真实 subtype 内，比较被损害与保持正确的细胞。每组至少 5 个细胞；先在患者内等权汇总 strata，再对患者与 seeds 等权。',
             '所有特征以及符合覆盖门槛的逐混淆边结果均公开于 feature_associations.csv；以下仅展示绝对差值较大的上下文关联，不作显著性筛选。']
    for kind in ['rescue','harm']:
        for prefix in ['RNA:','APT_raw:','APT_normalized:','QC:']:
            df=globalf[(globalf.contrast==kind)&globalf.feature.str.startswith(prefix)]
            df=df.loc[df.difference.abs().nlargest(4).index]
            text += [f'### {kind} / {prefix}',table(df,['feature','difference','ci_low','ci_high','eligible_patients_min','eligible_patients_max'])]
    example=features[(features.width==2000)&(features.true_subtype=='CD8 Tem-1')&
                     (features.rna_wrong_subtype=='CD4 + Tcm-1')]
    requested=['RNA:GNLY','RNA:NKG7','RNA:GZMB','RNA:CCR7','APT_normalized:APT-164',
               'APT_normalized:APT-172','QC:log1p_RNA_detected','QC:log1p_APT_total']
    text += ['## 一个逐混淆方向的关联示例',
        'CD8 Tem-1 被 RNA 错分为 CD4 + Tcm-1 的细胞中，将被纠正者与仍然错分者在患者内比较。这个例子事后选自直接纠错较多的方向，不是独立验证集，也不能说明这些特征驱动了纠错。',
        table(example[example.feature.isin(requested)],['feature','difference','ci_low','ci_high','eligible_patients_min','eligible_patients_max']),
        '该例的 APT 总量方向与全局分层平均不完全一致，进一步说明不能用一个整体 marker/QC profile 解释全部 subtype。',
        'RNA/APT 表达差值单位为各自的 log1p-normalized 或 log1p-count 单位；QC:RNA_mito_percent 是百分比值之差，例如 -0.043 表示 -0.043 个百分点，不是 -4.3%。']
    text += ['## 必须保留的限制',
        '- 这是事后、基于预测结果筛选细胞的关联分析，不是 APT 特征因果贡献，也不是 SHAP/消融。RNA markers 不能独立验证 RNA 来源的标签。',
        '- 本轮未进行每个 probe 的干预，也不能证明纠错来源于生物信号而非技术因素。未提供 probe 靶标映射，因此只报告 APT 编号。',
        '- APT raw 与 library-normalized 视图用于观察测量总量敏感性；它们不能替代 acquisition metadata。QC 差异也不能确诊 doublet 或低质量。',
        '- 点式 95% CI 未做全特征/全混淆边多重校正。5-cell 分层阈值会排除稀疏 strata；覆盖详见 feature_matching_coverage.csv。',
        '- 纠错的具体细胞配对价值仍应结合 patient×lineage shuffle 对照解释；本分析不把该对照的不确定结果改写成确定机制。',
        '- Marker 面板仅用于 PBMC 上下文，不是 27-subtype 的完整判定规则。[来源：Seurat 官方教程](https://satijalab.org/seurat/articles/pbmc3k_tutorial.html)。',
        '## 文件',
        '- all_confusion_edges.csv：全部方向、原始错分、纠错、新增错误、错误转移、患者覆盖和联合区间。',
        '- subtype_seed_counts.csv：所有 subtype、所有 seeds 的原始计数。',
        '- feature_associations.csv：全局分层及支持充分的混淆边对应 RNA/APT/QC 关联。',
        '- feature_matching_coverage.csv / audit.json：分层保留率、barcode 和 QC 审计。',
        '- confusion_change_*.png / bidirectional_error_changes.png / conditional_feature_associations.png：可视化。']
    (out/'REPORT_ZH.md').write_text('\n\n'.join(text)+'\n')


if __name__=='__main__': main()
