"""Build paper tables only from complete, audited v2 summaries."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).with_name('results')
NAMES={'lr':'Logistic Regression','xgb':'XGBoost','mlp':'Plain MLP','flat':'Flat Transformer',
       'hce':'HCE Transformer','cascade':'Soft-cascade (no disease)',
       'flat_contrast':'Flat + contrastive','cascade_contrast':'Cascade + contrastive'}
MAIN=['lr','xgb','mlp','flat','hce','cascade']


def table(name, columns, header, rows, caption, label, placement='t'):
    content=[r'\begin{table}['+placement+']',r'\centering\small',
             r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    content += [row+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}']
    (ROOT/'tables'/name).write_text('\n'.join(content)+'\n')


def main():
    for name in ['artifact_audit.json','summary_validation.json','hce_decoding_correction.json']:
        assert json.loads((DATA/name).read_text())['status']=='PASS',name
    scores={(r['model'],r['decoding']):r for r in csv.DictReader((DATA/'scores.csv').open())}
    contrasts={r['comparison']:r for r in csv.DictReader((DATA/'paired_contrasts.csv').open())}
    def fmt(r, digits=3):
        return f"{float(r['mean']):.{digits}f} [{float(r['low']):.{digits}f}, {float(r['high']):.{digits}f}]"
    def point(m,d):return f"{float(scores[m,d]['mean']):.3f}"
    table('main_result.tex','llcc',r'Model & CI & Fine SB-F1 $\uparrow$ & Coarse SB-F1 $\uparrow$',
        [' & '.join([NAMES[m],'P' if m=='lr' else 'SP',fmt(scores[m,'D0']),fmt(scores[m,'coarse'])]) for m in MAIN],
        r'Unified v2 APT-only reference comparison on the same 361{,}792 OOF cells and 40 patients. All models use raw-count log1p inputs, training-only standardization, the same inner-patient split, and fresh refits on all 32 development patients. No main-table model uses disease supervision. Values are means across three training seeds (LR: one deterministic fit) with 5{,}000-resample pointwise 95\% intervals: P resamples patients; SP jointly resamples patients and seeds. LR/XGBoost select the two tasks separately; neural coarse scores are auxiliary outputs of fine-selected fits. HCE coarse scores use subtree probability mass. Model-specific search spaces and the selection-conditioned uncertainty target are given in Appendix~\ref{sec:appendix:unified_v2}.',
        'tab:main','H')
    keys=[None,'flat_contrast minus flat','cascade minus flat','cascade_contrast minus cascade']
    models=['flat','flat_contrast','cascade','cascade_contrast']
    refs=['---','B$-$A','C$-$A','D$-$C']
    rows=[]
    for letter,m,key,ref in zip('ABCD',models,keys,refs):
        rows.append(' & '.join([letter+': '+('Cascade' if m.startswith('cascade') else 'Flat'),'Yes' if m.endswith('contrast') else 'No',
            fmt(scores[m,'D0']),ref if key is None else ref+': '+fmt(contrasts[key],4)]))
    table('unified_ablation.tex','llcc',r'Recipe & Disease & Fine SB-F1 & Paired difference [95\% CI]',rows,
        r'Unified v2 $2\times2$ comparison of the whole cascade structure and cross-disease contrastive supervision. A and C reuse the main-table fits; B and D add the same contrastive term. Shared encoder, inputs and search rules are held fixed, but each recipe selects its own validation optimum. C$-$A is not a routing-only effect. The interaction (D$-$C)$-$(B$-$A) is '+fmt(contrasts['contrastive interaction (D-C)-(B-A)'],4)+r'. Intervals are paired, pointwise and exploratory, not multiplicity-adjusted tests.',
        'tab:unified_ablation','t')
    table('unified_decoding.tex','lcccc',r'Model & Ordinary D0 & Hard D1 & Consistent D2 & Oracle D4',
        [' & '.join([NAMES[m]]+[point(m,d) for d in ['D0','D1','D2','D4']]) for m in NAMES],
        r'Frozen-score decoding diagnostics for all unified v2 recipes (fine SB-F1, seed means). D1 restricts candidates to the predicted parent; D2 combines predicted parent probabilities with within-parent subtype probabilities. Both are deployable post-processing rules using the same frozen scores. D4 uses the true parent and is privileged, not a deployable score or an information upper bound. LR/XGBoost use separately selected parent models; neural references use their auxiliary parent outputs, with subtree probabilities for HCE. No retraining or test-based tuning of the decoding rules is performed. Full scores, seed values and paired intervals are released in the v2 CSV summaries.',
        'tab:unified_decoding','H')
    rows=[]
    for f in range(5):
        a=json.loads((DATA/f'mlp_f{f}'/'audit.json').read_text())
        rows.append(' & '.join([str(f+1)]+[f"{len(a['patients'][s])} / {a['cells'][s]:,}" for s in ['inner','validation','refit','test']]))
    table('unified_fold_counts.tex','ccccc','Fold & Inner fit & Validation & Outer refit & Test',rows,
        r'Exact unified v2 patient / cell counts, reconstructed from input IDs and frozen partitions and checked against every recipe. All methods share these counts. Validation is the next cyclic outer fold; remaining patients form the inner fit. Outer refit combines inner fit and validation. Counts include all annotated benchmark cells, without a cap.',
        'tab:unified_fold_counts','H')
    rows=[]
    for m in NAMES:
        sels=[]; refit=0
        for f in range(5):
            for task in (['fine','coarse'] if m in ['lr','xgb'] else ['joint']):
                s=json.loads((DATA/f'{m}_f{f}'/f'{task}_selection.json').read_text()); sels.append(s)
                for seed in ([17] if m=='lr' else [17,29,43]):
                    refit+=json.loads((DATA/f'{m}_f{f}'/f'{task}_s{seed}.json').read_text())['seconds']
        ep=[s['selected']['epochs'] for s in sels if 'epochs' in s['selected']]
        rows.append(' & '.join([NAMES[m],str(sum(len(s['trials']) for s in sels)),
            f'{min(ep)}--{max(ep)}' if ep else '---',f"{sum(s['seconds'] for s in sels)/3600:.2f}",f'{refit/3600:.2f}']))
    table('unified_compute.tex','lcccc','Recipe & Candidate fits & Refit epochs & Search h & Refit h',rows,
        r'Realized v2 search/refit records summed across folds (and across tasks for LR/XGBoost). Neural search uses four candidates per fold; LR uses four and XGBoost twelve per task/fold. Epoch ranges are selected stopping epochs, not the epochs actually consumed by all search trials. Times are accumulated timed-block wall hours, not elapsed project time or GPU-hours: parallel jobs overlap, hosts differ, and input/scaler preparation is outside these blocks. Neural runs used RTX 6000 Ada GPUs; classical runs used CPU. This is a reproducibility record, not a cross-hardware efficiency ranking. HCE correction inference is excluded.',
        'tab:unified_compute','H')


if __name__=='__main__':main()
