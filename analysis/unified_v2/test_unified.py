"""Run on the training host, using its existing apt_jepa installation."""
import unittest
import numpy as np
import torch
from run import Network, contrast, matrices, score


class UnifiedTest(unittest.TestCase):
    def test_heads_and_gradients(self):
        parent=np.array([i%5 for i in range(27)])
        for name in ['mlp','flat','hce','cascade','flat_contrast','cascade_contrast']:
            model=Network(name,parent)
            f,c,z=model(torch.zeros(4,293))
            self.assertEqual(tuple(f.shape),(4,27)); self.assertEqual(tuple(c.shape),(4,5))
            (f.sum()+c.sum()).backward()
            self.assertTrue(torch.isfinite(f).all())

    def test_contrast(self):
        z=torch.randn(4,8,requires_grad=True)
        value=contrast(z,torch.tensor([0,0,1,1]),torch.tensor([0,1,0,1]))
        self.assertTrue(torch.isfinite(value)); value.backward()
        self.assertEqual(float(contrast(z,torch.arange(4),torch.arange(4))),0.)

    def test_fixed_ontology(self):
        _,cm=matrices(np.array([0,0]),np.array([0,0]),np.array(['a','b']),27)
        self.assertAlmostEqual(score(cm),1/27)


if __name__=='__main__':
    torch.set_num_threads(2); unittest.main()
