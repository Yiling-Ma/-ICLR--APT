"""Rebuild manuscript displays from completed aggregate results; no fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cell_identity_questions_v1/hvg2000'
scopes=['B','Myeloid','NK','Other','T']
scores=pd.read_csv(OUT/'scope_scores.csv')
ontology=pd.read_csv(ROOT/'outputs/oof_composition_bridge/subtype_to_lineage_mapping.csv')
assert ontology.fine_subtype.is_unique and len(ontology)==27
child_counts=ontology.groupby('coarse_lineage').size()
scope_labels=[f'{scope} ({child_counts[scope]})' for scope in scopes]
effects=pd.read_csv(OUT/'paired_contrasts.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.4,3.3),layout='constrained',sharey=True)
methods=[('prior_expected_cm','Conditional prior','#8b6a45','s'),
         ('apt_oracle','APT oracle','#167c83','o'),
         ('rna_oracle','RNA oracle','#b44f36','D')]
for ax,model in zip(axes,['sgd','mlp']):
    for n,scope in enumerate(scopes):
        vals=scores[(scores.model==model)&(scores.scope==scope)].set_index('method')['mean']
        ax.plot([vals[m] for m,_,_,_ in methods],[n]*3,color='.8',lw=1,zorder=1)
        for m,label,color,marker in methods:
            ax.scatter(vals[m],n,color=color,marker=marker,s=43,
                       label=label if n==0 else None,zorder=3)
    ax.set_yticks(np.arange(5),scope_labels); ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xticks(np.arange(0,1.01,.2)); ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Within-lineage SB-Macro-F1'); ax.set_title(model.upper(),fontsize=12)
    ax.spines[['top','right']].set_visible(False)
axes[1].tick_params(labelleft=True)
# Explicit limits avoid sharey/invert_yaxis toggling the shared axis twice.
axes[0].set_ylim(4.5,-.5)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False,fontsize=10)
fig.savefig(ROOT/'figures/matched_lineage_identity.pdf')
fig.savefig(ROOT/'figures/matched_lineage_identity.png',dpi=220)
plt.close(fig)

rows=[]
labels=[('rna','RNA'),('rna_noise','RNA + noise'),
        ('patient_shuffle','Patient shuffle'),
        ('lineage_shuffle',r'Patient $\times$ lineage shuffle')]
def entry(v,precision=4):
    return (r'\makecell{' + f"${v['mean']:+.4f}$" + r'\\' +
            f"$[{v['low']:.{precision}f}, {v['high']:.{precision}f}]$" + '}')
for control,label in labels:
    vals=[]
    for model in ['sgd','mlp']:
        v=effects[(effects.model==model)&(effects.scope=='all')&
                  (effects.metric=='sb_f1')&
                  (effects.comparison=='rna_apt-minus-'+control)]
        assert len(v)==1
        vals.append(entry(v.iloc[0],5 if control=='lineage_shuffle' else 4))
    rows.append('2,000 & '+label+' & '+' & '.join(vals)+r' \\')
df=pd.read_csv(ROOT/'outputs/remaining_modality_hvg5000_v1/summary.csv')
vals=[]
for model in ['sgd','mlp']:
    v=df[(df.model==model)&(df.task=='fine')&(df.comparison=='rna_apt-minus-rna')]
    assert len(v)==1
    r=v.iloc[0].to_dict();r['mean']=r['estimate']; vals.append(entry(r))
rows.append(r'\midrule'+'\n'+'5,000 & RNA (width sensitivity) & '+' & '.join(vals)+r' \\')
text=r'''\begin{table}[H]
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcc}
\toprule
RNA HVGs & Comparator to paired RNA+APT & SGD $\Delta$ [95\% CI] & MLP $\Delta$ [95\% CI] \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Overall fine SB-Macro-F1 differences, positive favoring paired
RNA+APT. Shuffle comparators append shuffled APT to RNA.
All 2,000-HVG contrasts use the same targeted experiment: three
training seeds and three realizations per seed for each shuffle type.
The 5,000-HVG row is the separate matched-width RNA comparison, not a
conditional-shuffle test. Scores are computed per fit before averaging;
2,000 paired seed/patient bootstrap replicates additionally resample shuffle
realizations where applicable. Intervals are pointwise and exploratory.
Both patient-by-lineage-shuffle intervals include zero.
Differences are calculated before rounding; displayed marginal scores
need not subtract to the displayed difference.}
\label{tab:paired_increment_main}
\end{table}
'''
(ROOT/'tables/paired_increment_main.tex').write_text(text)
gain2=effects[(effects.model=='mlp')&(effects.scope=='all')&
              (effects.metric=='sb_f1')&(effects.comparison=='rna_apt-minus-rna')]['mean'].item()
gain5=df[(df.model=='mlp')&(df.task=='fine')&
         (df.comparison=='rna_apt-minus-rna')]['estimate'].item()
(ROOT/'tables/paired_effect_values.tex').write_text(
    '% Generated from unrounded paired contrasts by build_findings_focus.py.\n'+
    r'\newcommand{\pairedMLPGainTwoK}{'+f'{gain2:.4f}'+'}\n'+
    r'\newcommand{\pairedMLPGainFiveK}{'+f'{gain5:.4f}'+'}\n')
print('Built matched-lineage figure and source-aligned paired contrast table.')
