import copy
import unittest

import numpy as np

from origami_bench.diagram.precrease_artifact_grading import signed_precrease_check, grade_precrease_artifact


class PrecreaseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]])
        self.hinges = [[0, 1, 2, 3], [1, 2, 3, 0]]
        self.initial = dict(op='initial', hinges=self.hinges, angles=[0., 0.])
        self.fold = dict(op='rotate_grips', hinges=self.hinges, angles=[np.pi, 0.])
        self.flat = dict(op='rotate_grips', hinges=self.hinges, angles=[0., 0.])
        self.history = [self.initial, self.fold, self.flat]
        self.spec = dict(required_signed_hinges={'0': [1]}, prepared_signed_hinges={},
                         hinge_stencils=self.hinges,
                         task_added_signed_hinges={'0': [1]},
                         required_band_creases=[dict(direction=[0., 1.], offset_m=-.1,
                            hinge_ids=[0], weights=[1.], signs=[1])])

    def memory(self, history=None, spec=None):
        return signed_precrease_check(self.history if history is None else history,
                                     self.spec if spec is None else spec, self.rest)

    def grade(self, **kwargs):
        args = dict(physical_pass=True, memory_pass=True, finished=True,
                    initial_state_pass=True, environment_error=False)
        args.update(kwargs)
        return grade_precrease_artifact(self.rest, self.rest, **args)

    def test_directed_coverage_and_order_independence(self):
        self.assertTrue(self.memory()['passed'])
        self.assertTrue(self.memory(list(reversed(self.history)))['passed'])

    def test_empty_preparation_cannot_earn_task(self):
        self.assertFalse(self.memory([self.initial])['passed'])
        self.assertFalse(self.memory([self.initial] * 100)['passed'])
        spec = copy.deepcopy(self.spec); spec['task_added_signed_hinges'] = {}
        self.assertFalse(self.memory(spec=spec)['input_valid'])
        spec['task_added_signed_hinges'] = {'0': [1]}; spec['prepared_signed_hinges'] = {'0': [1]}
        self.assertFalse(self.memory(spec=spec)['input_valid'])

    def test_wrong_sign_and_both_sign_saturation_fail(self):
        wrong = copy.deepcopy(self.fold); wrong['angles'][0] = -np.pi
        r = self.memory([self.initial, wrong, self.flat])
        self.assertFalse(r['passed']); self.assertEqual(len(r['missing_hinges']), 1)
        r = self.memory(self.history + [wrong])
        self.assertFalse(r['passed']); self.assertEqual(len(r['missing_hinges']), 0)
        self.assertEqual(len(r['opposite_sign_hinges']), 1)
        self.assertEqual(len(r['opposite_sign_bands']), 1)

    def test_source_explicit_both_signs_allowed(self):
        spec = copy.deepcopy(self.spec)
        spec['required_signed_hinges']['0'] = [-1, 1]
        spec['task_added_signed_hinges']['0'] = [-1, 1]
        spec['required_band_creases'][0]['signs'] = [-1, 1]
        wrong = copy.deepcopy(self.fold); wrong['angles'][0] = -np.pi
        self.assertTrue(self.memory(self.history + [wrong], spec)['passed'])

    def test_extra_crease_line_is_rejected(self):
        extra = copy.deepcopy(self.flat); extra['angles'][1] = np.pi
        result = self.memory(self.history + [extra])
        self.assertFalse(result['passed']); self.assertTrue(result['unexpected_bands'])

    def test_undo_removes_retained_crease(self):
        self.assertFalse(self.memory([self.initial, self.fold, dict(op='undo')])['passed'])

    def test_bad_evidence_fails_closed(self):
        for history in [[], [{'op': 'initial'}], [dict(self.fold, angles=[float('nan'), 0])],
                        [dict(self.fold, hinges=[[100, 1, 2, 3], [1, 2, 3, 0]])],
                        [self.fold, dict(self.flat, hinges=list(reversed(self.hinges)))]]:
            self.assertFalse(self.memory(history)['input_valid'])

    def test_bad_spec_fails_closed(self):
        for value in [[], [0], [1, 1], [True]]:
            spec = copy.deepcopy(self.spec); spec['required_signed_hinges']['0'] = value
            self.assertFalse(self.memory(spec=spec)['input_valid'])
        spec = copy.deepcopy(self.spec); spec['required_band_creases'][0]['weights'] = [float('nan')]
        self.assertFalse(self.memory(spec=spec)['input_valid'])
        spec = copy.deepcopy(self.spec); spec['hinge_stencils'] = list(reversed(self.hinges))
        self.assertFalse(self.memory(spec=spec)['input_valid'])

    def test_proper_yaw_xyz_placement_allowed(self):
        angle = .8; yaw = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        translated = self.rest.copy(); translated[:, :2] = translated[:, :2] @ yaw
        translated += [1., -3., 2.]
        result = grade_precrease_artifact(translated, self.rest, physical_pass=True,
                 memory_pass=True, finished=True, initial_state_pass=True)
        self.assertTrue(result['task_success'])
        self.assertFalse(result['process_order_scored'])

    def test_wrong_face_reflection_and_scale_fail(self):
        for modification in ['flip', 'scale']:
            changed = self.rest.copy()
            if modification == 'flip': changed[:, 0] *= -1
            else: changed *= 1.5
            result = grade_precrease_artifact(changed, self.rest, physical_pass=True,
                memory_pass=True, finished=True, initial_state_pass=True)
            self.assertFalse(result['task_success'])

    def test_all_gates_explicit_not_unknown(self):
        self.assertTrue(self.grade()['task_success'])
        for name in ['physical_pass', 'memory_pass', 'finished', 'initial_state_pass']:
            for value in [False, None, 1]:
                self.assertFalse(self.grade(**{name: value})['task_success'])
        for value in [True, None, 0]:
            self.assertFalse(self.grade(environment_error=value)['task_success'])


if __name__ == '__main__':
    unittest.main()
