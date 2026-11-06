import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.model_identity import normalize_identity, RecordingAgent, validate_identity
from origami_bench.diagram.public_summary import summarize_runs


def identity(model='A', **parameters):
    return normalize_identity(dict(model=model, provider='synthetic-provider', output_cap=8192,
                                   parameters=parameters), 'examples.openai_compatible:create')


def report(model='A', **extra):
    result = dict(task_id='SYNTHETIC', suite='synthetic', split='test', release_track='synthetic',
        score_eligible=True, task_success=True, status='passed_scoped', run_id=str(uuid.uuid4()),
        repeat_id=1, grading_contract='legacy-public-v2', release_manifest_sha256='synthetic-freeze', policy_sha256='synthetic-policy',
        agent_kind='external_agent:examples.openai_compatible:create', model_identity=identity(model))
    result.update(extra)
    return result


class IdentityTests(unittest.TestCase):
    def summarize(self, reports, plan=None, duplicate_path=False):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, row in enumerate(reports):
                p = Path(tmp)/str(i)
                p.mkdir()
                (p/'report.json').write_text(json.dumps(row))
                paths.append(p)
            if duplicate_path: paths.append(paths[0])
            plans = []
            if plan:
                p = Path(tmp)/'plan.json'
                p.write_text(json.dumps(plan))
                plans.append(p)
            return summarize_runs(paths, plans)

    def test_same_adapter_different_models_stay_separate(self):
        r = self.summarize([report('A'), report('B')])
        self.assertEqual(len(r['groups']), 2)
        self.assertFalse(r['comparison_ready'])  # no coverage plan supplied

    def test_parameters_and_grading_stay_separate(self):
        r = self.summarize([report(model_identity=identity('A',temperature=0)),
                            report(model_identity=identity('A',temperature=1)),
                            report(model_identity=identity('A',temperature=0),grading_contract='candidate')])
        self.assertEqual(len(r['groups']), 3)

    def test_legacy_missing_identity_excluded(self):
        r = self.summarize([report(model_identity=None)])
        self.assertEqual(r['groups'], [])
        self.assertEqual(r['issues'][0]['reason'], 'missing_or_invalid_model_identity')

    def test_duplicate_run_ids_and_directories_do_not_inflate(self):
        a = report()
        r = self.summarize([a, dict(a)], duplicate_path=True)
        self.assertEqual(r['groups'][0]['runs'], 1)
        self.assertEqual({x['reason'] for x in r['issues']}, {'duplicate_run_id','duplicate_directory'})

    def test_coverage_missing_and_startup_failure_visible(self):
        a = report(task_success=False,status='environment_error')
        expected = [{k:a[k] for k in ('run_id','task_id','repeat_id','release_manifest_sha256','policy_sha256','grading_contract')}]
        expected[0]['model_identity_sha256'] = a['model_identity']['identity_sha256']
        absent = dict(expected[0], run_id=str(uuid.uuid4()), task_id='MISSING')
        r = self.summarize([a], dict(runs=expected+[absent]))
        self.assertFalse(r['comparison_ready'])
        self.assertEqual(len(r['coverage']['missing']), 1)
        self.assertEqual(r['groups'][0]['environment_errors'], 1)
        self.assertIsNone(r['groups'][0]['conditional_success_rate'])
        r = self.summarize([a], dict(runs=expected))
        self.assertTrue(r['comparison_ready'])
        self.assertEqual(r['groups'][0]['end_to_end_success_rate'], 0)

    def test_environment_failure_stays_in_model_denominator(self):
        r=self.summarize([report(returned_models=['A']), report(task_success=False,status='environment_error',returned_models=[])])
        self.assertEqual(len(r['groups']),1)
        self.assertEqual(r['groups'][0]['end_to_end_success_rate'],.5)
        self.assertEqual(r['groups'][0]['environment_errors'],1)

    def test_returned_identity_change_blocks_comparison(self):
        r=self.summarize([report(returned_models=['A-v1']),report(returned_models=['A-v2'])])
        self.assertEqual(len(r['groups']),1)
        self.assertEqual(r['issues'][0]['reason'],'inconsistent_returned_models')

    def test_declared_identity_mismatch_is_not_scored(self):
        a = report()
        plan = dict(runs=[dict(run_id=a['run_id'],task_id=a['task_id'],repeat_id=1,
                              release_manifest_sha256='synthetic-freeze',model_identity_sha256='wrong')])
        r = self.summarize([a], plan)
        self.assertEqual(r['groups'], [])
        self.assertEqual(len(r['coverage']['unscorable']), 1)

    def test_credentials_not_copied_from_metadata(self):
        value = normalize_identity(dict(model='A',endpoint='https://provider.invalid/v1?api_key=SECRET',
            output_cap=8192,api_key='SECRET',headers={'Authorization':'SECRET'}), 'examples.openai_compatible:create')
        self.assertNotIn('SECRET', json.dumps(value))
        self.assertEqual(value['provider'], 'https://provider.invalid')
        self.assertTrue(validate_identity(value))
        with self.assertRaises(ValueError):
            normalize_identity(dict(model='A',provider='P',output_cap=8192,parameters={'api_key':'SECRET'}), 'examples.openai_compatible:create')

    def test_returned_model_mismatch_is_recorded(self):
        a = SimpleNamespace(request_public=lambda text,images:dict(action={'op':'finish'},returned_model='different'))
        wrapper = RecordingAgent(a)
        wrapper.request_public('public', [])
        r = self.summarize([report(returned_models=sorted(wrapper.returned_models))])
        self.assertTrue(r['groups'][0]['returned_model_mismatch'])

    def test_suite_preregisters_repeats_and_retains_startup_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'suite'
            argv=['run_suite.py','--suite','basics','--agent','missing:create','--repeats','2','--output',str(dest)]
            with patch.object(sys,'argv',argv),patch('subprocess.run',return_value=SimpleNamespace(returncode=1)),patch('sys.stdout',io.StringIO()):
                runpy.run_path(str(ROOT/'scripts/run_suite.py'),run_name='__main__')
            plan=json.loads((dest/'experiment-plan.json').read_text())
            self.assertEqual(len(plan['runs']),14)
            self.assertEqual({r['repeat_id'] for r in plan['runs']},{1,2})
            rows=[json.loads(p.read_text()) for p in dest.glob('*/report.json')]
            self.assertEqual({r['run_id'] for r in rows},{r['run_id'] for r in plan['runs']})
            self.assertTrue(all(r['status']=='environment_error' and r['physical_pass'] is None for r in rows))


if __name__ == '__main__': unittest.main()
