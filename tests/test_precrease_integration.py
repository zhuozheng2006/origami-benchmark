"""Verify public artifact grading preserves identity and failure semantics."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from scripts.apply_precrease_grading import apply_candidate

ROOT=Path(__file__).resolve().parents[1]

class PrecreaseIntegrationTests(unittest.TestCase):
    def private(self):
        return json.loads((ROOT/'data/precrease-hard-v4/private/tasks.json').read_text())['H102']

    def fixture(self, root):
        run=Path(root)/'run'
        shutil.copytree(ROOT/'examples/reference-evidence-v4/reference-H102-01',run)
        return run

    def test_real_reference_scores_without_promoting_oracle_or_erasing_original(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.fixture(d);original=json.loads((run/'report.json').read_text())
            result=apply_candidate(run,'H102',self.private())
            self.assertTrue(result['task_success'])
            self.assertFalse(result['score_eligible'])
            self.assertEqual(result['run_id'],original['run_id'])
            self.assertEqual(json.loads((run/'report-legacy.json').read_text()),original)
            with self.assertRaises(FileExistsError):apply_candidate(run,'H102',self.private())

    def test_environment_status_cannot_be_upgraded_to_success(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.fixture(d);original=json.loads((run/'report.json').read_text())
            original.update(status='environment_error',failure_stage='supervisor_timeout')
            original.pop('environment_error',None)
            (run/'report.json').write_text(json.dumps(original))
            result=apply_candidate(run,'H102',self.private())
            self.assertFalse(result['task_success'])
            self.assertTrue(result['environment_error'])
            self.assertEqual(result['status'],'environment_error')

    def test_missing_physical_evidence_is_unscorable_not_a_model_failure(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.fixture(d);(run/'metrics.json').write_text('[]')
            result=apply_candidate(run,'H102',self.private())
            self.assertFalse(result['task_success'])
            self.assertEqual(result['status'],'environment_error')
            self.assertEqual(result['failure_category'],'grading_evidence_error')
