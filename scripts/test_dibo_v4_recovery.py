"""Recovery must refuse ambiguous physical-battle state before starting jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from resume_dibo_v4_background import validate_saved_state


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for arm in ('cem','dibo'):
            (self.root/f'{arm}.json').write_text(json.dumps({'rounds':{'0':{'selected':[], 'evaluated':[]}}}))
            (self.root/f'{arm}_round0_battles_0.json').write_text('{}')

    def tearDown(self):self.temp.cleanup()

    def test_receipt_fingerprint_detects_changed_evidence(self):
        before=validate_saved_state(self.root)
        (self.root/'cem_round0_battles_0.json').write_text('{"changed":true}')
        self.assertNotEqual(before,validate_saved_state(self.root))

    def test_incomplete_receipt_blocks_recovery(self):
        (self.root/'dibo_round1_battles_0.incomplete.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Incomplete battle'):
            validate_saved_state(self.root)

    def test_selected_unfinished_batch_blocks_recovery(self):
        (self.root/'dibo.json').write_text(json.dumps({'rounds':{'1':{'selected':[{}]}}}))
        with self.assertRaisesRegex(RuntimeError,'pending battles'):
            validate_saved_state(self.root)

    def test_model_only_interruption_is_safe(self):
        (self.root/'dibo.json').write_text(json.dumps({'rounds':{'1':{'training':{}}}}))
        self.assertEqual(len(validate_saved_state(self.root)),2)


if __name__=='__main__':unittest.main(verbosity=2)
