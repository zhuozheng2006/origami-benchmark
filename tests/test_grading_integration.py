import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.apply_structural_grading import apply_candidate


class GradingIntegrationTests(unittest.TestCase):
    def test_opt_in_preserves_legacy_and_cannot_reapply_or_promote_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            shutil.copytree(ROOT/'examples/reference-evidence-v2/followup/final-O01',run)
            original = json.loads((run/'report.json').read_text())
            result = apply_candidate(run, 'O01')
            self.assertEqual(json.loads((run/'report-legacy.json').read_text()), original)
            self.assertTrue(result['task_success'])
            self.assertFalse(result['score_eligible'])
            self.assertEqual(result['grading_contract'], 'structural-v3')
            with self.assertRaises(FileExistsError):
                apply_candidate(run,'O01')

    def test_entrypoints_reject_wrong_track_before_model_or_simulation(self):
        import subprocess
        for script,args in [('evaluate_public.py',['--task','D001','--reference','--output','unused']),
                            ('run_suite.py',['--suite','basics','--reference','--output','unused'])]:
            p=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args,'--grading-contract','structural-v3'],
                             capture_output=True,text=True)
            # evaluate verifies manifest first; freeze is refreshed before release tests.
            self.assertNotEqual(p.returncode,0)
            self.assertIn('layer',p.stderr)
