"""Offline controls for qualification accounting; never starts a native worker."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import qualify_hard_candidates as q


class FakeBackend:
    def __init__(self):
        self.material = [[0, 0, 0], [.1, 0, 0], [0, .1, 0]]
        self.configuration = {'physics': 'fixed'}
        self.vertices = np.array(self.material, float)
        self.faces = [[0, 1, 2]]
        self.frames = [self.vertices.copy()]
        self.closed = False
    def state(self):
        return dict(vertices=self.vertices.tolist(), faces=self.faces, steps=len(self.frames)-1,
                    hash='state-'+str(len(self.frames)), time=0)
    def save(self):
        raise AssertionError('Fake finish writes evidence itself')
    def close(self):
        self.closed = True


class FakeEpisode:
    reject_at = None
    throw_at = None
    physical = True
    break_finish = False
    drift_mm = 0
    def __init__(self, task, private, output, factory, decisions, fixture_mode):
        self.output = Path(output)
        self.output.mkdir(parents=True)
        self.backend = factory(private['mesh'])
        self.task = task
        self.initialization = []
        self.initial_state = self.backend.state()
        self.controller = type('Controller', (), {'records': []})()
        self.finished = False
        self.calls = []
    def step(self, action):
        self.calls.append(copy.deepcopy(action))
        if self.throw_at == len(self.calls):
            raise RuntimeError('Simulation transport lost')
        if self.reject_at == len(self.calls):
            return dict(status='rejected', category='simulator_rejection',
                        detail='ActionRejected: principal strain reached 1% limit', rollback_verified=True)
        if action['op'] == 'finish':
            self.finished = True
        else:
            record = dict(action=action, details=dict(axis_point=[0, 0, 0],
                          axis_direction=[0, 0, 1], degrees=90))
            # Actual quarter-turn implemented independently of IdealTrace.
            p = self.backend.vertices
            self.backend.vertices = np.column_stack((-p[:, 1], p[:, 0], p[:, 2]))
            self.backend.vertices[0, 2] += self.drift_mm / 1000
            self.backend.frames.append(self.backend.vertices.copy())
            record['frame'] = len(self.backend.frames)-1
            self.controller.records.append(record)
        return dict(status='accepted')
    def finish(self, kind):
        report = dict(physical_pass=self.physical, task_success=True, memory_pass=True,
                      unsigned_memory_pass=True, band_memory_pass=True)
        (self.output/'report.json').write_text(json.dumps(report))
        if self.break_finish:
            raise RuntimeError('Final artifact collection incomplete')
        np.savez_compressed(self.output/'trajectory.npz', vertices=self.backend.frames)
        (self.output/'metrics.json').write_text(json.dumps([dict(clear=True, clearance={'clear': True},
            stretch=0.001, compression=0.001, grip_error=getattr(self, 'metric_grip_error', 0))]))
        return report
    def close(self):
        self.backend.close()


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/'run'
        self.task = {'id': 'C001', 'title': 'Candidate', 'instruction': 'Complete printed step 1', 'page_count': 1}
        self.private = {'mesh': 'test_mesh', 'pages': [], 'reference_actions': [{'op': 'rotate_paper', 'degrees': 90}]}
        self.backends = []
        self.patch = patch.object(q, 'signature', lambda config: config)
        self.patch.start()
        self.addCleanup(self.patch.stop)
    def backend(self, mesh, **settings):
        self.assertEqual(settings, q.SETTINGS)
        b = FakeBackend()
        self.backends.append(b)
        return b
    def run_case(self, klass=FakeEpisode, **kwargs):
        return q.qualify(self.task, self.private, self.output,
                         expected_physics={'physics': 'fixed'}, episode_factory=klass,
                         backend_factory=self.backend, **kwargs)
    def test_complete_reference_creates_goal_and_checkpoints_but_no_score(self):
        r = self.run_case()
        self.assertTrue(r['candidate_reference_pass'])
        self.assertTrue(r['annotated_plan_complete'])
        self.assertFalse(r['score_eligible'])
        self.assertFalse(r['task_success'])
        self.assertEqual(r['attempts'], 2)
        self.assertTrue((self.output/'candidate-goal.npz').exists())
        checkpoints = np.load(self.output/'candidate-checkpoints.npz')['vertices']
        self.assertEqual(checkpoints.shape, (1, 3, 3))
        self.assertTrue(self.backends[0].closed)
    def test_rejection_keeps_raw_cost_and_valid_prefix_never_a_goal(self):
        self.private['reference_actions'] *= 2
        r = self.run_case(type('RejectSecond', (FakeEpisode,), {'reject_at': 2}))
        self.assertFalse(r['candidate_reference_pass'])
        self.assertTrue(r['retained_prefix_physical_pass'])
        self.assertFalse(r['annotated_plan_complete'])
        self.assertEqual(r['accepted_reference_actions'], 1)
        attempts = json.loads((self.output/'reference-attempts.json').read_text())
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[-1]['result']['rollback_verified'])
        self.assertIn('1% limit', attempts[-1]['result']['detail'])
        self.assertFalse((self.output/'candidate-goal.npz').exists())
        self.assertFalse((self.output/'annotated-goal.npz').exists())
        self.assertTrue((self.output/'candidate-checkpoints-prefix.npz').exists())
    def test_budget_counts_finish_and_cannot_mark_unfinished_plan_passed(self):
        r = self.run_case(max_decisions=1)
        self.assertTrue(r['reference_actions_complete'])
        self.assertFalse(r['finished'])
        self.assertFalse(r['annotated_plan_complete'])
        self.assertEqual(r['remaining_decisions'], 0)
        self.assertFalse((self.output/'candidate-goal.npz').exists())
    def test_transport_exception_is_recorded_and_backend_closed(self):
        r = self.run_case(type('TransportFailure', (FakeEpisode,), {'throw_at': 1}))
        self.assertEqual(r['status'], 'reference_environment_failure')
        attempts = json.loads((self.output/'reference-attempts.json').read_text())
        self.assertEqual(attempts[0]['status'], 'environment_error')
        self.assertFalse(r['physical_pass'])
        self.assertTrue(self.backends[0].closed)
    def test_partial_success_report_from_failed_finish_is_not_qualified(self):
        r = self.run_case(type('ArtifactFailure', (FakeEpisode,), {'break_finish': True}))
        self.assertEqual(r['status'], 'reference_environment_failure')
        self.assertFalse(json.loads((self.output/'report.json').read_text())['task_success'])
        self.assertFalse((self.output/'candidate-goal.npz').exists())
    def test_independent_geometry_failure_is_not_physical_or_model_failure(self):
        r = self.run_case(type('GeometricDrift', (FakeEpisode,), {'drift_mm': 4}))
        self.assertTrue(r['annotated_plan_complete'])
        self.assertTrue(r['physical_pass'])
        self.assertFalse(r['geometry_pass'])
        self.assertEqual(r['status'], 'candidate_reference_failed_checks')
        self.assertFalse(r['candidate_reference_pass'])
    def test_signature_mismatch_stops_before_preparation_and_closes_allocated_backend(self):
        r = q.qualify(self.task, self.private, self.output, expected_physics={'physics': 'other'},
                      episode_factory=FakeEpisode, backend_factory=self.backend)
        self.assertEqual(r['status'], 'reference_environment_failure')
        self.assertEqual(r['attempts'], 0)
        self.assertIn('signature mismatch', r['environment_error']['detail'])
        self.assertTrue(self.backends[0].closed)
    def test_nonfinite_metric_cannot_pass_despite_legacy_boolean(self):
        r = self.run_case(type('InvalidMetric', (FakeEpisode,), {'metric_grip_error': float('nan')}))
        self.assertTrue(r['annotated_plan_complete'])
        self.assertTrue(r['retained_prefix_physical_pass'])
        self.assertFalse(r['strict_physical_evidence_pass'])
        self.assertFalse(r['candidate_reference_pass'])
    def test_preparation_capability_rejection_is_separate_from_transport_loss(self):
        def preparation_failure(task, private, output, factory, decisions, fixture_mode):
            Path(output).mkdir(parents=True)
            factory(private['mesh'])
            (Path(output)/'preparation-error.json').write_text('{}')
            raise q.SkillUnsupported('Current crease crosses mesh faces; remeshing unavailable')
        self.private['preparation'] = [{'op': 'fold_line'}]
        r = self.run_case(preparation_failure)
        self.assertEqual(r['status'], 'candidate_preparation_rejected')
        self.assertIsNone(r['environment_error'])
        self.assertEqual(r['failure']['category'], 'preparation_capability_rejection')
        self.assertFalse(r['candidate_reference_pass'])
        self.assertTrue((self.output/'preparation-attempts.json').exists())
        self.assertTrue(self.backends[0].closed)
    def test_existing_attempts_are_immutable(self):
        self.run_case()
        original = (self.output/'report.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_case()
        self.assertEqual((self.output/'report.json').read_bytes(), original)
    def test_input_paths_resolve_explicitly_and_finish_cannot_hide_in_reference(self):
        base = Path(self.tmp.name)
        (base/'page.png').write_bytes(b'local source bytes')
        pub, priv = base/'public.json', base/'private.json'
        pub.write_text(json.dumps({'tasks': [self.task]}))
        private = dict(self.private, pages=['page.png'])
        priv.write_text(json.dumps({'C001': private}))
        _, resolved = q.load_candidate(pub, priv, 'C001', base)
        self.assertEqual(resolved['pages'], [str((base/'page.png').resolve())])
        private['reference_actions'] = [{'op': 'finish'}]
        priv.write_text(json.dumps({'C001': private}))
        with self.assertRaisesRegex(ValueError, 'finish'):
            q.load_candidate(pub, priv, 'C001', base)


if __name__ == '__main__':
    unittest.main()
