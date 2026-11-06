"""Synthetic evidence I/O tests; these fixtures are not physical simulations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.score_precrease_candidates import rescore


class PrecreaseRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'; self.run.mkdir()
        self.v = np.array([[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.hinges = [[0, 1, 2, 3]]
        np.savez(self.root / 'source-goal.npz', vertices=self.v, rest=self.v, faces=self.faces)
        np.savez(self.root / 'source-initial.npz', vertices=self.v, rest=self.v, faces=self.faces)
        np.savez(self.run / 'trajectory.npz', vertices=[self.v, self.v], faces=self.faces)
        self.json('final-state.json', dict(vertices=self.v.tolist(), faces=self.faces.tolist()))
        self.json('events.json', [dict(action={'op':'finish'}, result={'status':'accepted'})])
        self.json('run-config.json', dict(initial_frame=0))
        self.json('metrics.json', [dict(clear=True, clearance=dict(clear=True), stretch=0., compression=0., grip_error=0.)])
        self.history = [dict(op='initial', hinges=self.hinges, angles=[0.]),
                        dict(op='fold', hinges=self.hinges, angles=[np.pi]),
                        dict(op='unfold', hinges=self.hinges, angles=[0.])]
        self.json('material-history.json', self.history)
        self.spec = dict(grading_contract='precrease-artifact-v4', goal='source-goal.npz',
                         initial_goal='source-initial.npz', precrease_memory=dict(
                            hinge_stencils=self.hinges, required_signed_hinges={'0':[1]},
                            prepared_signed_hinges={}, task_added_signed_hinges={'0':[1]},
                            required_band_creases=[dict(direction=[0, 1],offset_m=-.1,
                                                       hinge_ids=[0],weights=[1.],signs=[1])]))

    def tearDown(self): self.temp.cleanup()

    def json(self, name, value): (self.run / name).write_text(json.dumps(value))

    def score(self): return rescore(self.run, 'synthetic', self.spec, self.root)

    def test_external_goal_used_not_run_candidate_goal(self):
        np.savez(self.run / 'candidate-goal.npz', vertices=self.v * 100, rest=self.v, faces=self.faces)
        self.assertTrue(self.score()['task_success'])
        np.savez(self.root / 'source-goal.npz', vertices=self.v * 2, rest=self.v, faces=self.faces)
        self.assertFalse(self.score()['task_success'])

    def test_finish_flat_without_memory_rejected(self):
        self.json('material-history.json', self.history[:1])
        result = self.score()
        self.assertTrue(result['shape']['passed']); self.assertFalse(result['memory_pass'])
        self.assertFalse(result['task_success'])

    def test_material_topology_mismatch_rejected(self):
        bad = copy.deepcopy(self.history)
        for row in bad: row['hinges'][0][0] = 2
        self.json('material-history.json', bad)
        self.assertFalse(self.score()['task_success'])
        self.json('material-history.json', self.history)
        np.savez(self.run / 'trajectory.npz', vertices=[self.v, self.v], faces=self.faces[::-1])
        result = self.score(); self.assertFalse(result['task_success']); self.assertIn('topology', result['evidence_error'])

    def test_unknown_or_missing_physical_evidence_fails_closed(self):
        self.json('metrics.json', [])
        self.assertFalse(self.score()['physical_pass']); self.assertTrue(self.score()['environment_error'])
        (self.run / 'metrics.json').unlink()
        result = self.score(); self.assertFalse(result['task_success']); self.assertFalse(result['input_valid'])
        self.assertTrue(result['environment_error'])
        self.assertEqual(result['failure_category'], 'grading_evidence_error')

    def test_physical_failure_distinguished_from_unknown_evidence(self):
        metric=dict(clear=True,clearance=dict(clear=True),stretch=.02,compression=0.,grip_error=0.)
        self.json('metrics.json',[metric]); result=self.score()
        self.assertFalse(result['physical_pass']);self.assertFalse(result['environment_error'])
        for key, value in [('stretch',float('nan')),('clear',None)]:
            changed=dict(metric);changed[key]=value;self.json('metrics.json',[changed]);result=self.score()
            self.assertFalse(result['task_success']);self.assertTrue(result['environment_error'])

    def test_report_environment_status_cannot_be_overridden_by_good_prefix(self):
        for report in [dict(status='environment_error', failure_stage='supervisor_timeout'),
                       dict(status='reference_environment_failure'),
                       dict(status='environment_error', environment_error=False),
                       dict(failure=dict(category='environment_error'))]:
            self.json('report.json', report)
            result = self.score()
            self.assertTrue(result['shape']['passed']); self.assertTrue(result['physical_pass'])
            self.assertTrue(result['environment_error']); self.assertFalse(result['task_success'])

    def test_final_state_disagrees_and_nan_rejected(self):
        self.json('final-state.json',dict(vertices=(self.v+1).tolist(),faces=self.faces.tolist()))
        self.assertFalse(self.score()['input_valid'])
        bad=self.v.copy();bad[0,0]=float('nan')
        np.savez(self.run / 'trajectory.npz',vertices=[self.v,bad],faces=self.faces)
        self.assertFalse(self.score()['input_valid'])


if __name__ == '__main__': unittest.main()
