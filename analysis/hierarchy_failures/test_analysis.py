"""Consistency checks for aggregate saved-score diagnostics, without training."""
import csv
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = json.loads((ROOT / 'analysis/generated/hierarchy_failures.json').read_text())


class DiagnosticsTest(unittest.TestCase):
    def test_complete_scope_and_sources(self):
        self.assertEqual((A['cells'], A['patients']), (361792, 40))
        self.assertEqual(len(A['sources']), 15)
        self.assertEqual(len(A['matrices']), 6)
        self.assertEqual(len(A['per_class']), 162)
        self.assertEqual((A['new_training'], A['new_inference']), (0, 0))
        for check in A['checks'].values():
            for key, value in check.items():
                if key != 'prior_expected_sb_f1':
                    self.assertIs(value, True)
            self.assertAlmostEqual(check['prior_expected_sb_f1'], .1798346603007621)

    def test_error_partition(self):
        for row in A['summary']:
            self.assertAlmostEqual(sum(row[k] for k in ['correct', 'cross_error', 'within_error']), 1)
            self.assertAlmostEqual(row['correct'] + row['repair'], row['oracle_correct'])
            self.assertEqual(row['eligible_patients'], 38 if row['scope'] == 'Other' else 40)

    def test_matrices_and_per_class_scores(self):
        parents = np.asarray(A['display_lineages'])
        for block in A['matrices']:
            cm = np.asarray(block['sb_cm'])
            self.assertEqual(cm.shape, (27, 27))
            self.assertAlmostEqual(cm.sum(), 40)
            den = cm.sum(0) + cm.sum(1)
            f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(27), where=den > 0)
            self.assertAlmostEqual(f1.mean(), block['macro_f1'])
            np.testing.assert_allclose(np.asarray(block['row_normalized']).sum(1), 1)
            if block['mode'] == 'oracle':
                self.assertTrue((cm[parents[:, None] != parents[None, :]] == 0).all())
            rows = [r for r in A['per_class'] if r['seed'] == block['seed'] and r['mode'] == block['mode']]
            self.assertEqual([r['subtype'] for r in rows], A['display_classes'])
            self.assertEqual(sum(r['cell_count'] for r in rows), 361792)
            np.testing.assert_allclose(f1, [r['sb_f1'] for r in rows])

    def test_saved_exports(self):
        directory = ROOT / 'analysis/hierarchy_failures/results'
        with (directory / 'all_confusion_edges.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6 * 27 * 27)
        with np.load(directory / 'confusion_matrices.npz', allow_pickle=False) as data:
            self.assertEqual(data['ordinary_sb'].shape, (3, 27, 27))
            for mode in ['ordinary', 'oracle']:
                np.testing.assert_allclose(data[mode + '_sb'], [b['sb_cm'] for b in A['matrices'] if b['mode'] == mode])


if __name__ == '__main__':
    unittest.main()
