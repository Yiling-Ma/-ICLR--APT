"""Check the documented order against the production scorer and joint loop."""
import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
source=ast.parse((ROOT/'analysis/cell_identity_questions.py').read_text())
functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and
           n.name in ['scoped_score','bootstrap_scores']]
env={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),'<production scorers>','exec'),env)
score=env['scoped_score']; bootstrap=env['bootstrap_scores']


class BootstrapContractTests(unittest.TestCase):
    def test_vectorized_patient_draw_equals_explicit_resampling(self):
        rng=np.random.default_rng(7)
        cm=rng.integers(0,20,(40,3,3)); cm[0]=0
        indices=rng.integers(0,40,(20,40))
        counts=np.stack([np.bincount(i,minlength=40) for i in indices])
        fast=bootstrap(cm,counts,np.array([0,1,2]))['sb_f1']
        slow=np.array([score(cm[i],np.array([0,1,2])) for i in indices])
        np.testing.assert_allclose(fast,slow)

    def test_f1_is_not_computed_after_seed_pooling(self):
        cm=np.array([[[90,0],[0,10]],[[0,90],[10,0]]])
        separate=np.mean([score(c[None],np.arange(2)) for c in cm])
        merged=score(cm.sum(0)[None],np.arange(2))
        self.assertAlmostEqual(separate,.5)
        self.assertGreater(abs(separate-merged),.09)

    def test_outside_lineage_predictions_remain_false_negatives(self):
        cm=np.array([[[3,0,7],[0,4,6],[0,0,0]]])
        value=score(cm,np.array([0,1]))
        self.assertLess(value,1)
        self.assertAlmostEqual(value,((6/13)+(8/14))/2)

    def test_three_seed_three_shuffle_loop(self):
        rng=np.random.default_rng(42); B=25
        hb=rng.random((3,B)); lb=rng.random((3,3,B))
        ss=rng.integers(0,3,(B,3)); rr=rng.integers(0,3,(B,3,3))
        d=np.arange(B); vector=np.zeros(B)
        # Same indexing and averaging order as aggregate() in production.
        for slot in range(3):
            null=np.mean([lb[ss[:,slot],rr[:,slot,r],d] for r in range(3)],axis=0)
            vector+=(hb[ss[:,slot],d]-null)/3
        direct=[np.mean([hb[ss[b,j],b]-np.mean([lb[ss[b,j],rr[b,j,r],b]
                  for r in range(3)]) for j in range(3)]) for b in range(B)]
        np.testing.assert_allclose(vector,direct)

    def test_display_values_and_child_counts(self):
        df=pd.read_csv(ROOT/'outputs/remaining_modality_hvg5000_v1/summary.csv')
        value=df[(df.model=='mlp')&(df.task=='fine')&
                 (df.comparison=='rna_apt-minus-rna')]['estimate'].item()
        macros=(ROOT/'tables/paired_effect_values.tex').read_text()
        self.assertIn(r'\newcommand{\pairedMLPGainFiveK}{'+f'{value:.4f}'+'}',macros)
        self.assertIn(f'{value:+.4f}',(ROOT/'tables/rna_width_repeats.tex').read_text())
        for file in ['iclr2027_conference.tex','main/intro.tex','main/results.tex']:
            self.assertIn(r'\pairedMLPGainFiveK{}',(ROOT/file).read_text())
        mapping=pd.read_csv(ROOT/'outputs/oof_composition_bridge/subtype_to_lineage_mapping.csv')
        self.assertTrue(mapping.fine_subtype.is_unique)
        self.assertEqual(mapping.groupby('coarse_lineage').size().to_dict(),
                         {'B':5,'Myeloid':8,'NK':2,'Other':2,'T':10})

    def test_saved_primary_contrast_and_unresolved_interval(self):
        df=pd.read_csv(ROOT/'outputs/cell_identity_questions_v1/hvg2000/paired_contrasts.csv')
        df=df[(df.model=='mlp')&(df.scope=='all')&(df.metric=='sb_f1')].set_index('comparison')
        self.assertAlmostEqual(df.loc['rna_apt-minus-patient_shuffle','mean'],.008777389157539858)
        v=df.loc['rna_apt-minus-lineage_shuffle']
        self.assertLess(v.low,0); self.assertGreater(v.high,0)
        table=(ROOT/'tables/paired_increment_main.tex').read_text()
        self.assertIn('-0.00018',table)
        self.assertIn('+0.0088',table)


if __name__=='__main__': unittest.main()
