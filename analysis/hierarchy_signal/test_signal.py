import unittest
import numpy as np
import torch
from run_signal import conditional_logp, permutation, base, train, predict


class SignalTests(unittest.TestCase):
    def test_conditional_loss_and_joint(self):
        parent=torch.arange(27)%5
        f=torch.randn(11,27,requires_grad=True); c=torch.randn(11,5,requires_grad=True)
        y=torch.arange(11)%27; r=conditional_logp(f,parent)
        for m in range(5):
            torch.testing.assert_close(r[:,parent==m].exp().sum(1),torch.ones(11))
        joint=r.exp()*c.softmax(1)[:,parent]
        torch.testing.assert_close(joint.sum(1),torch.ones(11))
        loss=-r.gather(1,y[:,None]).mean()+.5*torch.nn.functional.cross_entropy(c,parent[y])
        loss.backward(); self.assertTrue(torch.isfinite(f.grad).all())
        # Other-lineage leaf logits have exactly zero conditional-loss gradient.
        self.assertTrue(torch.all(f.grad[parent[None,:]!=parent[y,None]]==0))

    def test_permutation_scope_and_repeatability(self):
        ids=np.array(['a']*6+['b']*6); coarse=np.tile([0,0,1,1,2,2],2)
        ix=np.array([0,1,2,3,6,7,8,9]); a=permutation(ix,ids,coarse,101,0,0)
        np.testing.assert_array_equal(a,permutation(ix,ids,coarse,101,0,0))
        np.testing.assert_array_equal(ids[a],ids[ix]); np.testing.assert_array_equal(coarse[a],coarse[ix])
        np.testing.assert_array_equal(np.sort(a),ix)

    def test_parameter_identity(self):
        parent=np.arange(27)%5
        ordinary=base.Network('mlp',parent); conditional=base.Network('mlp',parent)
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()),sum(p.numel() for p in conditional.parameters()))

    def test_ordinary_loop_matches_v2(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(5); parent=np.arange(27)%5
        x=rng.normal(size=(32,293)).astype(np.float32); y=np.arange(32)%27; c=parent[y]
        cfg=dict(lr=.001,decay=1e-5)
        old,_=base.train('mlp',x,y,c,c,parent,cfg,17,1,'cpu')
        new,_=train(x,y,c,parent,cfg,17,1,'cpu',False,'mlp')
        for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        op,oq=base.predict(old,x,'cpu'); np_,nq=predict(new,x,parent,'cpu',False)
        np.testing.assert_array_equal(op,np_); np.testing.assert_array_equal(oq,nq)

    def test_scope_prior_and_metric(self):
        from summarize_signal import cm_by_patient, f1
        y=np.array([0,0,1,1]); ids=np.array(['a','a','b','b']); parent=np.arange(27)%5
        cm=cm_by_patient(y,y,ids,-1,parent,['a','b'])
        self.assertAlmostEqual(float(f1(cm.sum(0),np.arange(27))),2/27)
        prior=np.ones((4,27))
        for m in range(5): prior[:,parent==m]/=(parent==m).sum()
        expected=cm_by_patient(y,None,ids,-1,parent,['a','b'],prior)
        np.testing.assert_allclose(expected.sum((1,2)),1)


if __name__=='__main__': unittest.main()
