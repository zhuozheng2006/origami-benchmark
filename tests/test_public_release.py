import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rescore',ROOT/'scripts/rescore_evidence.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PublicReleaseTests(unittest.TestCase):
    def test_failure_and_undo_keep_history_and_cost(self):
        from origami_bench.diagram.attempt_ledger import AttemptLedger, BudgetExhausted
        with tempfile.TemporaryDirectory() as temp:
            ledger=AttemptLedger(Path(temp)/'attempts.sqlite3',2,100)
            try:
                for op,status in [('fold_line','rejected'),('undo','accepted')]:
                    i=ledger.begin()
                    ledger.response(i,dict(action={'op':op},input_tokens=10,output_tokens=5,seconds=1))
                    ledger.finish(i,dict(result={'status':status},before='same',after='same'))
                self.assertEqual(ledger.budget()['attempts_remaining'],0)
                self.assertEqual(ledger.budget()['known_costs']['output_tokens'],10)
                self.assertEqual(len(ledger.public_history()),2)
                self.assertEqual(ledger.public_history()[0]['result']['status'],'rejected')
                with self.assertRaises(BudgetExhausted):ledger.begin()
                with self.assertRaises(sqlite3.IntegrityError):ledger.db.execute('DELETE FROM journal')
            finally:
                ledger.close()

    def test_bundled_reference_paths_pass(self):
        for task in ['O01','O02','O03']:
            with self.subTest(task=task):
                result=mod.rescore(ROOT/'examples/reference-evidence'/task,task)
                self.assertTrue(result['task_success'])

    def test_missing_finish_and_unknown_physics_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'run'
            shutil.copytree(ROOT/'examples/reference-evidence/O01',dest)
            events=json.loads((dest/'events.json').read_text())
            (dest/'events.json').write_text(json.dumps(events[:-1]))
            self.assertFalse(mod.rescore(dest,'O01')['task_success'])
            (dest/'events.json').write_text(json.dumps(events))
            metrics=json.loads((dest/'metrics.json').read_text())
            metrics[0].pop('clear',None)
            (dest/'metrics.json').write_text(json.dumps(metrics))
            result=mod.rescore(dest,'O01')
            self.assertFalse(result['physical_pass'])
            self.assertFalse(result['task_success'])

    def test_public_task_assets_resolve(self):
        tasks=json.loads((ROOT/'data/release-v1/private/tasks.json').read_text())
        for task in tasks.values():
            for path in task['pages']+[task[k] for k in ('goal','initial_goal','checkpoints')]:
                self.assertTrue((ROOT/path).is_file(),path)
            self.assertTrue((ROOT/'tasks/crease_meshes'/f"{task['mesh']}.json").is_file())


if __name__=='__main__':
    unittest.main()
