import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HistoricalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((ROOT / 'analysis/generated/historical_split_audit.json').read_text())

    def test_four_unrounded_contrasts(self):
        self.assertEqual(len(self.audit['contrasts']), 4)
        table = (ROOT / 'tables/protocol_sensitivity.tex').read_text()
        self.assertNotIn('Inflation', table)
        for item in self.audit['contrasts']:
            self.assertAlmostEqual(item['difference'], item['cell'] - item['patient'], places=14)
            self.assertIn(f"${item['difference']:+.3f}$", table)

    def test_unmatched_allocations(self):
        splits = self.audit['splits']
        cell = splits['cell_both_models']
        self.assertEqual([cell['parts'][s]['cells'] for s in ['train', 'val', 'test']],
                         [253254, 54269, 54269])
        self.assertEqual(set(cell['patient_intersections'].values()), {40})
        self.assertEqual(set(cell['cell_intersections'].values()), {0})
        for model, expected in [('xgb_patient', [32, 0, 8]), ('soft_patient', [24, 8, 8])]:
            for fold in splits[model]:
                self.assertEqual([fold['parts'][s]['patients'] for s in ['train', 'val', 'test']], expected)
                self.assertFalse(any(fold['patient_intersections'].values()))
            self.assertEqual(sum(f['parts']['test']['cells'] for f in splits[model]), 361792)

    def test_provenance_limits_and_no_training(self):
        self.assertEqual(self.audit['new_training_runs'], 0)
        for headers in self.audit['neural_prediction_headers'].values():
            self.assertNotIn('cell_id', headers)
        for log in self.audit['soft_training_logs'].values():
            self.assertEqual(log['resolved_epoch_budget'], 0)
            self.assertGreater(log['epochs_logged'], 0)
        for source in self.audit['sources'].values():
            self.assertEqual(len(source['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
