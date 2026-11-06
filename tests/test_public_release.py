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
    def test_all_public_v2_reference_evidence_regrades(self):
        evidence=json.loads((ROOT/'audit/fresh-reference-validation.json').read_text())
        registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
        # Historical v2 evidence stays immutable; newly accepted tracks must
        # carry their own complete qualification, not rewrite old experiments.
        fresh=json.loads((ROOT/'data/precrease-hard-v4/qualification-summary.json').read_text())['tasks']
        new_ids={k for k,v in fresh.items() if v.get('scoring_acceptance_pass') is True}
        self.assertFalse(set(evidence['evidence_by_accepted_task']) & new_ids)
        self.assertEqual(set(evidence['evidence_by_accepted_task']) | new_ids,
                         {k for k,v in registry.items() if v['status']=='accepted'})
        for run in evidence['completed_runs']:
            score=mod.rescore(ROOT/run['evidence'],run['task_id'])
            self.assertEqual(score['task_success'],run['task_success'],run['evidence'])
        for tid in new_ids:
            for names,expected in [('positive_runs',True),('negative_runs',False)]:
                for name in fresh[tid][names]:
                    score=mod.rescore(ROOT/'examples/reference-evidence-v4'/name,tid)
                    self.assertIs(score['task_success'],expected,(tid,name))
                    self.assertTrue(score['physical_pass'],(tid,name))

    def test_suite_startup_failures_remain_in_reports(self):
        import tempfile,runpy,sys,io
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'suite'
            argv=['run_suite.py','--suite','basics','--agent','missing:create','--output',str(output)]
            with patch.object(sys,'argv',argv),patch('subprocess.run',return_value=SimpleNamespace(returncode=1)),patch('sys.stdout',io.StringIO()):
                runpy.run_path(str(ROOT/'scripts/run_suite.py'),run_name='__main__')
            reports=[json.loads(p.read_text()) for p in output.glob('*/report.json')]
            self.assertEqual(len(reports),7)
            self.assertTrue(all(r['status']=='environment_error' and r['score_eligible'] and not r['task_success'] and r['physical_pass'] is None for r in reports))
            self.assertEqual(len((output/'suite-attempts.jsonl').read_text().splitlines()),7)

    def test_summary_does_not_pool_protocols_or_development(self):
        import tempfile,subprocess,sys
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,(suite,split,eligible) in enumerate([('small','test',True),('small','development',True),('layers','test',True),('ultra','test',False)]):
                path=Path(tmp)/str(i);path.mkdir();paths.append(str(path))
                from origami_bench.diagram.model_identity import normalize_identity
                identity=normalize_identity(dict(model='test',provider='synthetic',output_cap=8192),'examples.openai_compatible:create')
                (path/'report.json').write_text(json.dumps(dict(task_id=str(i),suite=suite,split=split,score_eligible=eligible,agent_kind='external_agent:test',task_success=True,release_manifest_sha256='frozen',model_identity=identity,run_id=str(i),repeat_id=1,release_track='synthetic',policy_sha256='policy')))
            result=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/summarize_public.py'),*paths]))
            self.assertEqual(len(result['groups']),3)
            self.assertEqual(result['excluded_oracle_diagnostic_or_replay'],1)

    def test_step_layout_selection_never_uses_task_identity(self):
        from origami_bench.diagram import line_fold_agent_v9, single_panel_policy
        from PIL import Image
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'diagram.png'
            Image.new('RGB',(400,400)).save(path)
            self.assertIs(single_panel_policy.select_policy({'step_count':1},path),single_panel_policy)
            Image.new('RGB',(800,400)).save(path)
            self.assertIs(single_panel_policy.select_policy({'step_count':1},path),line_fold_agent_v9)
            Image.new('RGB',(1200,400)).save(path)
            self.assertIs(single_panel_policy.select_policy({'step_count':2},path),line_fold_agent_v9)
        registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
        for tid,specification in registry.items():
            if specification['episode']!='step':continue
            data=ROOT/'data'/specification['dataset']
            task=next(x for x in json.loads((data/'public/index.json').read_text())['tasks'] if x['id']==tid)
            selected=single_panel_policy.select_policy(task,data/'public'/task['image'])
            self.assertIn(selected,(line_fold_agent_v9,single_panel_policy))

    def test_expanded_registry_keeps_failed_tasks_out(self):
        registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
        for tid in ['D006','J001','S001','INT-001','INT-01']:
            self.assertNotIn(tid,registry)
        self.assertEqual(sum(v['episode']=='step' for v in registry.values()),121)
        self.assertEqual(registry['D05']['status'],'needs_repair')
        self.assertEqual(registry['H03']['status'],'needs_repair')
        self.assertTrue(all(v['status']!='accepted' for v in registry.values() if v['track']=='diagnostic'))

    def test_expanded_assets_and_original_freezes(self):
        from origami_bench.diagram.episode import check_freeze
        registry=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
        for ds in {v['dataset'] for v in registry.values() if v['episode']=='step'}:
            check_freeze(ROOT/'data'/ds)
        for tid,specification in registry.items():
            data=ROOT/'data'/specification['dataset']
            task=json.loads((data/'private/tasks.json').read_text())[tid]
            self.assertTrue(any((ROOT/'tasks'/folder/f"{task['mesh']}.json").is_file() for folder in ['crease_meshes','multifold_meshes']),tid)
            if specification['episode']=='step':
                pub=next(t for t in json.loads((data/'public/index.json').read_text())['tasks'] if t['id']==tid)
                self.assertTrue((data/'public'/pub['image']).is_file(),tid)
                self.assertTrue((data/'private'/task['targets']).is_file(),tid)
            else:
                self.assertTrue((ROOT/task['goal']).is_file(),tid)
                if not specification.get('requires_source_download'):
                    for page in task['pages']:self.assertTrue((ROOT/page).is_file(),tid)

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
