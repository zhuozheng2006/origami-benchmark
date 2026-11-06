import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.structural_grading_v3 import compare_placement, ordered_motion_checkpoints, combine_success, physical_evidence_pass
from scripts.rescore_structural_v3 import rescore


class StructuralGradingTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.array([[0, 0, 0], [.10, 0, .01], [0, .06, .015], [.04, .03, .04], [.08, .06, -.02]])
        first = self.initial.copy(); first[1, 2] += .025
        second = first.copy(); second[3, 2] += .04
        self.targets = np.array([first, second])
        self.path = np.array([self.initial, (self.initial+first)/2, first, second])

    def test_positive_reference_and_extra_samples(self):
        self.assertTrue(ordered_motion_checkpoints(self.path, self.targets, self.initial)['passed'])
        self.assertTrue(ordered_motion_checkpoints(np.repeat(self.path, 3, axis=0), self.targets, self.initial)['passed'])

    def test_global_yaw_and_xyz_placement_and_height_drift(self):
        r = np.array([[0, 1], [-1, 0]])
        moved = self.path.copy(); moved[:, :, :2] = moved[:, :, :2] @ r
        moved += np.array([.2, -.1, .006])
        moved[:, :, 2] += np.arange(len(moved))[:, None]*.002
        self.assertTrue(compare_placement(moved[-1], self.targets[-1])['passed'])
        self.assertTrue(ordered_motion_checkpoints(moved, self.targets, self.initial)['passed'])

    def test_duplicate_and_nearby_targets_invalid_even_with_padded_frames(self):
        for perturbation in [0, .0001]:
            targets = np.array([self.targets[0], self.targets[0].copy()])
            targets[1, 1, 2] += perturbation
            result = ordered_motion_checkpoints(np.repeat(self.path, 100, axis=0), targets, self.initial)
            self.assertFalse(result['passed']); self.assertFalse(result['contract_valid'])
            self.assertEqual(result['reason'], 'ambiguous_checkpoint_transition')

    def test_static_and_skipped_and_reversed_paths_fail(self):
        paths = [np.repeat(self.initial[None], 100, axis=0),
                 np.array([self.initial, self.targets[-1]]),
                 np.array([self.initial, self.targets[1], self.targets[0]])]
        for path in paths:
            self.assertFalse(ordered_motion_checkpoints(path, self.targets, self.initial)['passed'])

    def test_rotation_checkpoint_requires_actual_orientation_change(self):
        target = self.initial.copy(); target[:, :2] = target[:, :2] @ np.array([[0, 1], [-1, 0]])
        self.assertTrue(ordered_motion_checkpoints(np.array([self.initial, target]), target[None], self.initial)['passed'])
        bad = ordered_motion_checkpoints(np.repeat(self.initial[None], 100, axis=0), target[None], self.initial)
        self.assertTrue(bad['contract_valid']); self.assertFalse(bad['passed'])

    def test_reflection_flip_scale_and_wrong_shape_fail(self):
        variants = []
        for axis in [0, 2]:
            v = self.targets[-1].copy(); v[:, axis] *= -1; variants.append(v)
        v = self.targets[-1].copy(); v[:, [0, 2]] *= -1; variants.append(v)  # actual 180-degree turnover
        variants.append(self.targets[-1] * 1.2)
        v = self.targets[-1].copy(); v[2] += [.01, 0, .02]; variants.append(v)
        for v in variants:
            self.assertFalse(compare_placement(v, self.targets[-1])['passed'])

    def test_empty_nan_correspondence_and_degenerate_yaw_fail_closed(self):
        for bad in [np.empty((0, 3)), np.full((5, 3), np.nan), np.zeros((5, 3))]:
            with self.assertRaises(ValueError): compare_placement(bad, self.initial)
        with self.assertRaises(ValueError): ordered_motion_checkpoints([], self.targets, self.initial)
        with self.assertRaises(ValueError): ordered_motion_checkpoints(self.path, [], self.initial)
        with self.assertRaises(ValueError): ordered_motion_checkpoints(self.path, self.targets, self.initial, -1)

    def test_all_evidence_gates_remain_required(self):
        shape, path = dict(passed=True), dict(passed=True, contract_valid=True)
        self.assertTrue(combine_success(shape, path, True, True, True, False))
        for evidence in [(False, True, True, False), (True, None, True, False),
                         (True, False, True, False), (True, True, False, False),
                         (True, True, True, True)]:
            self.assertFalse(combine_success(shape, path, *evidence))

    def test_numeric_physical_evidence_is_finite_complete_and_strict(self):
        good = dict(clear=True, clearance=dict(clear=True), stretch=0.001, compression=0.001, grip_error=.0001)
        self.assertTrue(physical_evidence_pass([good]))
        for bad in [float('nan'), float('inf'), None, -1, True, .01]:
            self.assertFalse(physical_evidence_pass([{**good, 'compression': bad}]))
        self.assertFalse(physical_evidence_pass([]))
        self.assertFalse(physical_evidence_pass([{**good, 'clear': None}]))

    def test_real_layer_references_still_pass(self):
        paths = [('O01', 'followup/final-O01'), ('O02', 'sweep/O02'),
                 ('O03', 'sweep/O03'), ('L01', 'followup/L01-longpath')]
        for task, path in paths:
            with self.subTest(task=task):
                self.assertTrue(rescore(ROOT/'examples/reference-evidence-v2'/path, task)['task_success'])

    def test_archived_o04_separates_shape_from_required_process(self):
        evidence = ROOT/'examples/grading-evidence-v3'
        gpt = rescore(evidence/'model-gpt-6-astra-O04-01', 'O04')
        self.assertTrue(gpt['structure_success']); self.assertFalse(gpt['task_success'])
        self.assertEqual(gpt['diagram_path']['matched'], 4)
        self.assertEqual(gpt['diagram_path']['required'], 8)
        for run in ['reference-O04-v17-01', 'reference-O04-v17-02', 'model-gemini-4-argon-O04-02']:
            self.assertTrue(rescore(evidence/run, 'O04')['task_success'])
        failed = rescore(evidence/'model-claude-opus-5-5-O04-01', 'O04')
        self.assertTrue(failed['environment_error']); self.assertFalse(failed['task_success'])


if __name__ == '__main__':
    unittest.main()
