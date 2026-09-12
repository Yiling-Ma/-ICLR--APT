"""Render descriptive full-ontology diagnostics from audited aggregate scores."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/hierarchy_failures/results'


def save_csv(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def tex(s): return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=json.loads((ROOT/'analysis/generated/hierarchy_failures.json').read_text())
    labels=a['display_classes'];lineages=a['display_lineages'];k=len(labels)
    save_csv('error_decomposition_by_seed.csv',a['summary'])
    save_csv('per_subtype_by_seed.csv',a['per_class'])
    edges=[];means={};edge_ranks={}
    for mode in ['ordinary','oracle']:
        blocks=[r for r in a['matrices'] if r['mode']==mode]
        for b in blocks:
            m=np.asarray(b['sb_cm']); row=np.asarray(b['row_normalized'])
            ranked=sorted([(i,j) for i in range(k) for j in range(k) if i!=j],key=lambda ij:-m[ij])
            edge_ranks[(mode,b['seed'])]={ij:r+1 for r,ij in enumerate(ranked)}
            for i in range(k):
                for j in range(k):
                    edges.append(dict(mode=mode,seed=b['seed'],truth=labels[i],prediction=labels[j],
                        sb_cm_mass=m[i,j],all_cell_fraction=m[i,j]/40,row_fraction=row[i,j],
                        zero_truth_row=i in b['zero_rows']))
        mean=np.mean([b['sb_cm'] for b in blocks],axis=0);means[mode]=mean
        den=mean.sum(1,keepdims=True)
        visual=np.divide(mean,den,out=np.zeros_like(mean),where=den>0)
        fig,ax=plt.subplots(figsize=(8.2,8.0),layout='constrained')
        masked=np.ma.masked_where(np.broadcast_to(den==0,visual.shape),visual)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#bdbdbd')
        im=ax.imshow(masked,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks(np.arange(k),labels,rotation=90,fontsize=8.5)
        ax.set_yticks(np.arange(k),labels,fontsize=8.5)
        bounds=[i-.5 for i in range(1,k) if lineages[i]!=lineages[i-1]]
        for v in bounds:
            ax.axhline(v,color='white',lw=1);ax.axvline(v,color='white',lw=1)
        ax.set_xlabel('Predicted subtype',fontsize=11);ax.set_ylabel('True subtype',fontsize=11)
        ax.set_title('Full-budget APT MLP: '+('ordinary decoding' if mode=='ordinary' else 'true-lineage oracle'),fontsize=12)
        bar=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);bar.set_label('Row fraction after seed-mean SB pooling',fontsize=9)
        fig.savefig(OUT/f'{mode}_confusion.pdf');fig.savefig(OUT/f'{mode}_confusion.png',dpi=160)
        plt.close(fig)
    save_csv('all_confusion_edges.csv',edges)
    np.savez_compressed(OUT/'confusion_matrices.npz',classes=np.asarray(labels),lineages=np.asarray(lineages),
        seeds=np.asarray(a['seeds']),
        ordinary_sb=np.asarray([b['sb_cm'] for b in a['matrices'] if b['mode']=='ordinary']),
        oracle_sb=np.asarray([b['sb_cm'] for b in a['matrices'] if b['mode']=='oracle']),
        ordinary_row=np.asarray([b['row_normalized'] for b in a['matrices'] if b['mode']=='ordinary']),
        oracle_row=np.asarray([b['row_normalized'] for b in a['matrices'] if b['mode']=='oracle']))
    classes=[]
    for i,name in enumerate(labels):
        rows=[r for r in a['per_class'] if r['subtype']==name]
        vals={mode:[r['sb_f1'] for r in rows if r['mode']==mode] for mode in ['ordinary','oracle']}
        dest=max([j for j in range(k) if j!=i],key=lambda j:means['oracle'][i,j])
        classes.append(dict(subtype=name,lineage=lineages[i],cell_count=rows[0]['cell_count'],
            patient_coverage=rows[0]['patient_coverage'],
            ordinary_sb_f1=np.mean(vals['ordinary']),oracle_sb_f1=np.mean(vals['oracle']),
            ordinary_seed_min=min(vals['ordinary']),ordinary_seed_max=max(vals['ordinary']),
            oracle_seed_min=min(vals['oracle']),oracle_seed_max=max(vals['oracle']),
            largest_oracle_error=labels[dest],oracle_error_row_fraction=means['oracle'][i,dest]/means['oracle'][i].sum()))
    save_csv('per_subtype_summary.csv',classes)
    ranked=sorted([(i,j) for i in range(k) for j in range(k) if i!=j],key=lambda ij:-means['oracle'][ij])
    pairs=[]
    for rank,(i,j) in enumerate(ranked[:3],1):
        mass=[np.asarray(b['sb_cm'])[i,j]/40 for b in a['matrices'] if b['mode']=='oracle']
        pairs.append(dict(rank=rank,truth=labels[i],prediction=labels[j],
            all_cell_fraction=np.mean(mass),seed_min=min(mass),seed_max=max(mass),
            seed_ranks=[edge_ranks[('oracle',s)][(i,j)] for s in a['seeds']],
            true_cells=classes[i]['cell_count'],true_patients=classes[i]['patient_coverage'],
            destination_cells=classes[j]['cell_count'],destination_patients=classes[j]['patient_coverage']))
    save_csv('top_oracle_edges.csv',pairs)
    scopes=list(dict.fromkeys(r['scope'] for r in a['summary']));scope_rows=[]
    for scope in scopes:
        rs=[r for r in a['summary'] if r['scope']==scope]
        row=dict(scope=scope,patients=rs[0]['eligible_patients'],cells=rs[0]['cells'])
        for key in ['correct','cross_error','within_error','repair','oracle_correct','cross_repair_fraction']:
            row[key]=float(np.mean([r[key] for r in rs]));row[key+'_min']=min(r[key] for r in rs);row[key+'_max']=max(r[key] for r in rs)
        row['cross_ratio_patients_min']=min(r['cross_ratio_eligible_patients'] for r in rs)
        row['cross_ratio_patients_max']=max(r['cross_ratio_eligible_patients'] for r in rs)
        scope_rows.append(row)
    save_csv('error_decomposition_summary.csv',scope_rows)
    lines=[r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Scope & Patients & Correct & Cross-parent & Within-parent & Oracle correct & Cross repaired \\',r'\midrule']
    for r in scope_rows:
        lines.append(tex(r['scope'])+f" & {r['patients']} & "+' & '.join(f"{100*r[key]:.1f}" for key in ['correct','cross_error','within_error','oracle_correct','cross_repair_fraction'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Full-budget MLP error proportions (percent), averaged within patient and then across three seeds. The first three rates partition all cells in the scope and sum to 100\% before rounding. Cross-parent uses the parent of the predicted subtype, not the independent coarse head. Cross repaired is the within-patient fraction of cross-parent errors corrected by oracle, averaged over patients with such errors (all listed patients in each seed); it is not a ratio of the displayed averages. These are error rates, not additive contributions to Macro-F1.}',r'\label{tab:mlp_error_partition}',r'\end{table}']
    (ROOT/'tables/mlp_error_partition.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrr r l}',r'\toprule',
        r'Subtype & Cells & Patients & Ordinary & Oracle & Largest oracle error (row \%) \\',r'\midrule']
    last=None
    for r in classes:
        if last is not None and r['lineage']!=last:lines.append(r'\midrule')
        last=r['lineage']
        lines.append(tex(r['subtype'])+f" & {r['cell_count']:,} & {r['patient_coverage']} & {r['ordinary_sb_f1']:.3f} & {r['oracle_sb_f1']:.3f} & "+tex(r['largest_oracle_error'])+f" ({100*r['oracle_error_row_fraction']:.1f})"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{All 27 subtypes for the same full-budget MLP and three seeds. Ordinary/oracle entries average seed-specific per-class F1 derived from subject-normalized pooled confusion matrices. These are not the historical Soft-cascade CW values in Table~\ref{tab:subtype_per_class}. Destination percentages row-normalize the seed-mean oracle SB matrix; they are visual summaries, not per-patient conditional averages. Full matrices, per-seed F1 and every directed edge are retained.}',r'\label{tab:mlp_subtype_errors}',r'\end{table}']
    (ROOT/'tables/mlp_subtype_errors.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scopes=scope_rows,top_edges=pairs),indent=2))


if __name__=='__main__':main()
