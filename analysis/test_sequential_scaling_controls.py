import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sequential_scaling_controls import sample, aggregate, configuration, jobs, name
from class_matched_scaling import allocate_quotas


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.d = np.repeat(np.arange(32), 80)
        self.y = np.tile(np.arange(80) % 4, 32)
        self.y[(self.d >= 8) & (self.y == 3)] = 4
        rng = np.random.default_rng(12)
        self.priority = rng.random(len(self.y))
        self.dp = dict(zip(range(32), rng.random(32)))
        self.q = allocate_quotas(np.bincount(self.y[self.d < 8], minlength=5), 320)

    def test_invariants(self):
        for p in (8,32):
            for condition in "ABCD":
                ix = sample(self.y,self.d,list(range(p)),self.q,self.priority,self.dp,condition)
                self.assertEqual(len(ix),320)
                self.assertEqual(len(set(ix)),320)
                self.assertTrue(np.all(self.d[ix] < p))
                if condition != "A":
                    self.assertEqual(set(self.y[ix]), {0,1,2,3})
                if condition in "CD":
                    np.testing.assert_array_equal(np.bincount(self.y[ix],minlength=5),self.q)
                again = sample(self.y,self.d,list(range(p)),self.q,self.priority,self.dp,condition)
                np.testing.assert_array_equal(ix,again)

    def test_new_class_only_unmatched(self):
        ix = sample(self.y,self.d,list(range(32)),self.q,self.priority,self.dp,"A")
        self.assertIn(4,self.y[ix])

    def test_balancing(self):
        ix = sample(self.y,self.d,list(range(8)),self.q,self.priority,self.dp,"D")
        for k in range(4):
            counts = np.bincount(self.d[ix][self.y[ix] == k],minlength=8)
            self.assertLessEqual(np.ptp(counts),1)

    def test_insufficient_capacity(self):
        with self.assertRaises(ValueError):
            sample(self.y,self.d,[0],self.q,self.priority,self.dp,"C")

    def test_paired_aggregation_and_incomplete_rejection(self):
        args = SimpleNamespace(dataset="apt",models="logistic_regression",seeds=1,total=100,bootstrap=20)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                aggregate(args,output)
            (output/"runs").mkdir()
            for j in jobs(args):
                cm = np.array([[[4,1],[1,4]]]) if j["p"] == 8 else np.array([[[5,0],[0,5]]])
                stem = output/"runs"/name(j)
                np.savez(stem.with_suffix(".npz"),patient_ids=[str(j["fold"])],confusions=cm)
                stem.with_suffix(".json").write_text(json.dumps(dict(config=configuration(args))))
            aggregate(args,output)
            result = json.loads((output/"summary.json").read_text())
            for row in result["results"]:
                for effect in row["conditions"].values():
                    self.assertAlmostEqual(effect["estimate"],.2)
                    self.assertAlmostEqual(effect["lower"],.2)
                for contrast in row["contrasts"].values():
                    self.assertAlmostEqual(contrast["estimate"],0.)


if __name__ == "__main__":
    unittest.main()
