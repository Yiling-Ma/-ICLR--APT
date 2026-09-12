import unittest
import numpy as np
from run import error_counts, joint, stratified_features


class AnalysisTests(unittest.TestCase):
    def test_rescue_harm_and_rerouting(self):
        y=np.array([0,0,0,0,1]); r=np.array([1,0,1,2,1]); a=np.array([0,1,2,1,1])
        c=error_counts(np.zeros(5,int),y,r,a,1,3)
        self.assertEqual(c['rescued'][0,0,1],1)
        self.assertEqual(c['harmed'][0,0,1],1)
        self.assertEqual(c['rerouted_out'][0,0,1],1)
        self.assertEqual(c['rerouted_in'][0,0,1],1)

    def test_patient_not_cell_weighted(self):
        p=np.array([0]*100+[1]); y=np.zeros(101,int)
        r=np.array([1]*100+[0]); a=np.zeros(101,int)
        c=error_counts(p,y,r,a,2,2)
        rate=c['rescued'][:,0,1]/c['rna'][:,0,:].sum(1)
        point,_,_=joint(np.stack([rate]*3)[:,:,None])
        self.assertAlmostEqual(point[0],.5)

    def test_feature_matching(self):
        # Two different RNA mistakes cannot be pooled into one comparison.
        p=np.zeros(20,int); y=np.zeros(20,int)
        r=np.array([1]*10+[2]*10); a=np.array([0]*5+[1]*5+[0]*10)
        x=np.array([3]*5+[1]*5+[100]*10)[:,None]
        means,_,audit=stratified_features(x,p,y,r,a,1,3,'rescue')
        self.assertAlmostEqual(means[0,0],2)
        self.assertEqual(audit['eligible_strata'],1)
        self.assertEqual(audit['retained_positive_cells'],5)


if __name__=='__main__': unittest.main()
