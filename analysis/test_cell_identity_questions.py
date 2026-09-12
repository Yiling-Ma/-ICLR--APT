import unittest
import numpy as np
from cell_identity_questions import permutation,prior_predictions,scoped_cm,scoped_score,bootstrap_scores,expected_prior_cm

class Controls(unittest.TestCase):
    def test_conditional_permutation(self):
        ids=np.array(['a']*6+['b']*6); lin=np.tile([0,0,0,1,1,1],2)
        p,a=permutation(ids,lin,42,True)
        np.testing.assert_array_equal(ids,ids[p]);np.testing.assert_array_equal(lin,lin[p])
        np.testing.assert_array_equal(np.sort(p),np.arange(12))
        self.assertTrue(0<=a['fixed_fraction']<=1)
    def test_prior_uses_training_only(self):
        parent=np.array([0,0,1,1])
        np.testing.assert_array_equal(prior_predictions(np.array([1,1,2]),np.array([0,1,2,3]),parent),[1,1,2,2])
    def test_outside_prediction_retains_false_negative(self):
        cm=scoped_cm(np.array([0,0]),np.array([0,2]),np.array(['a','a']),np.array([True,True]),3)
        self.assertAlmostEqual(scoped_score(cm,np.array([0])),2/3)
        b=bootstrap_scores(cm,np.array([[1],[2]]),np.array([0]))
        np.testing.assert_allclose(b['sb_f1'],2/3)
        np.testing.assert_allclose(b['subject_accuracy'],.5)
    def test_empty_scope(self):
        b=bootstrap_scores(np.zeros((2,3,3)),np.array([[1,1]]),np.array([0]))
        self.assertTrue(np.isnan(b['sb_f1'][0]))
    def test_expected_prior_preserves_mass_and_lineage(self):
        parent=np.array([0,0,1,1]); truth=np.array([0,1,2,3])
        cm=expected_prior_cm(np.array([0,0,1]),truth,parent,np.array(['a']*4),np.ones(4,bool))
        np.testing.assert_allclose(cm.sum(2),[[1,1,1,1]])
        np.testing.assert_allclose(cm[0,0],[2/3,1/3,0,0])
        np.testing.assert_allclose(cm[0,2],[0,0,.5,.5])
    def test_patient_normalization_not_cell_weighting(self):
        cm=np.array([[[100,0],[0,0]],[[0,0],[1,0]]],dtype=float)
        self.assertAlmostEqual(scoped_score(cm,np.array([0,1])),1/3)

if __name__=='__main__':unittest.main()
