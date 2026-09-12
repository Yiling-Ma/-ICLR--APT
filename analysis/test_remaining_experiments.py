import unittest
import numpy as np
import torch
from run_full_budget_mlp import oracle, matrices, score, fit, predict


class ProtocolTests(unittest.TestCase):
    def test_oracle_preserves_correct_and_parent(self):
        p=np.array([[.1,.2,.7],[.8,.1,.1],[.1,.8,.1]])
        truth=np.array([0,2,1]); parent=np.array([0,0,1])
        pred=oracle(p,truth,parent)
        np.testing.assert_array_equal(pred,[1,2,1])
        np.testing.assert_array_equal(parent[pred],parent[truth])

    def test_subject_balance(self):
        y=np.array([0,1,0,1]); pred=np.array([0,1,1,0])
        ids=np.array(['a','a','b','b'])
        _,a=matrices(y,pred,ids,2)
        _,b=matrices(np.r_[y,y[:2]],np.r_[pred,pred[:2]],np.r_[ids,ids[:2]],2)
        self.assertAlmostEqual(score(a),score(b))

    def test_training_probabilities_and_missing_class(self):
        torch.set_num_threads(1)
        x=np.random.default_rng(4).normal(size=(20,4)).astype('float32')
        y=np.arange(20)%2
        m,_,ep,h=fit(x,y,3,42,.001,2,torch.device('cpu'),(x,y,np.repeat(['a','b'],10)))
        p=predict(m,x,torch.device('cpu'))
        self.assertEqual(p.shape,(20,3))
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        self.assertTrue(1<=ep<=2)
        self.assertEqual(len(h),2)


if __name__=='__main__': unittest.main()
