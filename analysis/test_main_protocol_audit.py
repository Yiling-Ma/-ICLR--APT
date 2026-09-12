import collections
import json
import unittest
from pathlib import Path
from render_main_protocol_audit import build

ROOT=Path(__file__).resolve().parents[1]


class MainProtocolAuditTests(unittest.TestCase):
    def test_main_points_and_manifest(self):
        a=build()
        self.assertEqual(set(r['model'] for r in a['records']),
            {'logistic_regression','xgboost','plain_mlp','flat_ce','hce','soft_cascade'})
        self.assertEqual(a['new_training'],0)

    def test_partition_counts_and_seeds(self):
        a=json.loads((ROOT/'analysis/generated/main_protocol_audit.json').read_text())
        expected_dev=[294812,293411,275553,292943,290449]
        expected_train=[226431,207172,206704,221600,223469]
        for r in a['records']:
            self.assertEqual(r['test']['patient_count'],8)
            outer=r['stage'] in ['outer_refit','final_fit']
            self.assertEqual(r['patient_count'],32 if outer else 24)
            self.assertEqual(r['cell_count'],(expected_dev if outer else expected_train)[r['fold']])
        mlp=[r for r in a['records'] if r['model']=='plain_mlp']
        self.assertEqual(collections.Counter(r['stage'] for r in mlp),{'inner_selection':30,'outer_refit':30})
        self.assertEqual({r['seed'] for r in mlp},{20260912,20260913,20260914})
        self.assertTrue(all(r['evaluated_configs']==2 for r in mlp))

    def test_historical_selection_boundary(self):
        a=json.loads((ROOT/'analysis/generated/main_protocol_audit.json').read_text())
        neural=[r for r in a['records'] if r['stage']=='selected_checkpoint']
        self.assertEqual(len(neural),15)
        self.assertTrue(all(r['declared_max_epochs']==50 for r in neural))
        self.assertTrue(all('CW' in r['selection'] for r in neural))
        self.assertTrue(all(r['final_replay_calibration_budget']==0 for r in neural))
        self.assertTrue(all(r['final_equals_best_tensors'] for r in neural))


if __name__=='__main__':
    unittest.main()
