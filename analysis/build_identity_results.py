"""Render completed aggregate evidence without accessing private cell arrays."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cell_identity_questions_v1/hvg2000'
scores=pd.read_csv(OUT/'scope_scores.csv')
effects=pd.read_csv(OUT/'paired_contrasts.csv')
lineages=['all','B','Myeloid','NK','Other','T']
fig,axes=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
for ax,contrast,title in zip(axes,['apt_oracle-minus-prior_expected_cm','rna_apt-minus-lineage_shuffle'],
    ['APT oracle minus training prior','Paired minus patient-by-lineage shuffle']):
    for j,(model,color) in enumerate([('sgd','#2868a8'),('mlp','#be6329')]):
        sub=effects[(effects.model==model)&(effects.metric=='sb_f1')&(effects.comparison==contrast)].set_index('scope').loc[lineages]
        yy=np.arange(6)+(j-.5)*.22
        ax.hlines(yy,sub.low,sub.high,color=color,lw=1.3)
        ax.scatter(sub['mean'],yy,s=20,color=color,label=model.upper(),zorder=3)
    ax.axvline(0,color='.4',lw=.8,ls='--');ax.set_yticks(np.arange(6),['All',*lineages[1:]])
    ax.invert_yaxis();ax.set_xlabel('Difference in SB-Macro-F1');ax.set_title(title,fontsize=10)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
axes[0].legend(frameon=False,fontsize=9)
fig.savefig(OUT/'matched_identity_controls.pdf');fig.savefig(OUT/'matched_identity_controls.png',dpi=220)

rows=[]
for model in ['sgd','mlp']:
    for scope in lineages:
        sub=scores[(scores.model==model)&(scores.scope==scope)].set_index('method')['mean']
        rows.append(f"{model.upper()} & {'All' if scope=='all' else scope} & "+' & '.join(f'{sub[m]:.3f}' for m in ['apt','apt_oracle','rna_oracle','prior_oracle','prior_expected_cm'])+r' \\')
table=r'''\begin{table}[H]
\centering\small
\begin{tabular}{llccccc}
\toprule
Model & Scope & APT & APT oracle & RNA oracle & Prior MAP & Prior expected CM \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Matched-budget hierarchy diagnostics: mean over three training seeds,
2,000 training-selected RNA HVGs, identical capped training subsets and patient
folds. All-cell rows normalize each patient's complete confusion matrix;
lineage rows instead normalize within that lineage and score its fixed
children. Their scores cannot be averaged to recover all-cell SB-Macro-F1. Oracle and prior controls receive true lineage. Expected-CM F1 is not
expected finite-sample F1. Other has 38 eligible patients; other scopes have 40.
These are descriptive point estimates, not paired APT--RNA superiority tests.}
\label{tab:matched_identity}
\end{table}
'''
(ROOT/'tables/matched_identity.tex').write_text(table)
rows=[]
for width in [2000,5000]:
    df=pd.read_csv(ROOT/f'outputs/remaining_modality_hvg{width}_v1/summary.csv')
    for task in ['coarse','fine']:
        for model in ['sgd','mlp']:
            sub=df[(df.task==task)&(df.model==model)].set_index('comparison')
            delta=sub.loc['rna_apt-minus-rna']
            vals=' & '.join(f"{sub.loc[m,'estimate']:.3f}" for m in ['rna','rna_apt','rna_noise','rna_shuffled_apt'])
            rows.append(f"{width:,} & {task.title()} & {model.upper()} & {vals} & {delta.estimate:+.3f} [{delta.low:.3f}, {delta.high:.3f}]"+r' \\')
table=r'''\begin{table}[H]
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllccccc}
\toprule
HVGs & Task & Model & RNA & Paired & Noise & Patient shuffle & Paired minus RNA \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{RNA-width reference sensitivity: three training seeds and five patient folds. Point estimates average seed-specific SB-Macro-F1;
paired differences have joint seed/patient 95\% intervals. Patient shuffle
uses one draw per seed in this grid; the targeted 2,000-HVG experiment uses
three draws per seed for each shuffle type. All normalization and HVG selection
are training-partition-only. Noise adds 293 dimensions.}
\label{tab:rna_width_repeats}
\end{table}
'''
(ROOT/'tables/rna_width_repeats.tex').write_text(table)
