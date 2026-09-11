"""Checks for reweighting without changing the frozen prediction targets."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from revise_benchmark_table import summarize


class ReweightedMetricsTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(sample_id=['a']*6+['b']*2,
            y_true=['0','0','0','1','1','1','0','1'],
            y_pred=['0','0','0','1','1','1','1','0']))

    def score(self, frame):
        return summarize({'coarse': {'DropCascade': frame}}, n_boot=50).set_index('metric')

    def test_matches_sample_weighted_f1(self):
        rows = self.score(self.frame)
        weights = 1/self.frame.groupby('sample_id').sample_id.transform('size')
        expected = f1_score(self.frame.y_true, self.frame.y_pred,
                            sample_weight=weights, average='macro', labels=['0','1'])
        self.assertAlmostEqual(rows.loc['subject_balanced','estimate'], expected)
        self.assertAlmostEqual(rows.loc['cell_weighted','estimate'], .75)
        self.assertAlmostEqual(expected, .5)

    def test_patient_cell_replication_invariance(self):
        larger = pd.concat([self.frame, self.frame[self.frame.sample_id == 'a']]*2)
        self.assertAlmostEqual(self.score(larger).loc['subject_balanced','estimate'], .5)

    def test_target_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            summarize({'coarse': {'DropCascade': self.frame, 'bad': self.frame.iloc[:-1]}}, n_boot=10)

    def test_deterministic(self):
        np.testing.assert_array_equal(self.score(self.frame).lower, self.score(self.frame).lower)


if __name__ == '__main__':
    unittest.main()
