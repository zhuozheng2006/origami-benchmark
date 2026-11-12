"""Offline scheduling tests; archived geometry and fake execution are not physics reruns."""
import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from origami_bench.diagram.layer_controller import motion
from origami_bench.diagram.layer_controller_v16 import LayerControllerV16, compile_stack_fold_v15
from origami_bench.diagram.skills import SkillUnsupported

ROOT = Path(__file__).resolve().parents[1]

class RecordingBackend:
    """Commands advance counters only; no physical validity assertion."""
    def __init__(self, vertices, faces):
        self.paper = dict(vertices=vertices.tolist(), faces=faces.tolist(),
                          hash='fake', steps=0, time=0.)
    def state(self):
        return copy.deepcopy(self.paper)
    def step(self, command):
        frames = command.get('frames', 0)
        self.paper['steps'] += frames
        self.paper['time'] += frames / 60
        return self.state()

class PacketFoldFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / 'examples/reference-evidence-v2/sweep/O03'
        cls.records = json.loads((folder / 'controller-records.json').read_text())
        with np.load(folder / 'trajectory.npz') as archive:
            cls.vertices, cls.faces = archive['vertices'], archive['faces']

    def command(self, degrees=180, radius=.075, **kwargs):
        return motion([0], np.zeros(3), np.array([1., 0., 0.]), degrees,
                      np.array([[0., radius, 0.]]), **kwargs)

    def test_default_schedule_preserved(self):
        self.assertEqual(self.command()['frames'], 360)

    def test_configurable_loading_schedule(self):
        for step in (.5, .45, .4, .388, .3):
            for degrees in (-180, -90, 22.5, 90, 180):
                with self.subTest(step=step, degrees=degrees):
                    self.assertEqual(self.command(degrees, angle_step_deg=step)['frames'],
                                     math.ceil(abs(degrees) / step))

    def test_scale_bound_dominates_both_schedules(self):
        for step in (.5, .388):
            self.assertEqual(self.command(radius=.12, angle_step_deg=step)['frames'], 472)

    def test_invalid_options_rejected_before_execution(self):
        for step in (0, -.1, float('nan'), float('inf'), True, '.388', .6):
            with self.subTest(step=step):
                with self.assertRaises(ValueError): self.command(angle_step_deg=step)
                with self.assertRaises(ValueError): LayerControllerV16(None, packet_angle_step_deg=step)

    def test_budget_still_enforced(self):
        with self.assertRaises(SkillUnsupported): self.command(angle_step_deg=.25)
        with self.assertRaises(SkillUnsupported): self.command(radius=.2)

    def compile_record(self, index, layers=None, **kwargs):
        record = self.records[index]
        frame = self.records[index - 1]['frame'] if index else 0
        action = copy.deepcopy(record['action'])
        if layers is not None: action['layers'] = layers
        return compile_stack_fold_v15(self.vertices[frame], self.faces, action,
                                      record['details']['fixture_mode'], **kwargs)

    def test_default_archived_fold_frame_schedules_preserved(self):
        for i, record in enumerate(self.records):
            if record['action']['op'] != 'fold_line': continue
            low, info = self.compile_record(i)
            self.assertEqual(low[-1]['frames'], record['low_level_actions'][-1]['frames'])
            self.assertFalse(info['loading_schedule']['experimental'])

    def test_equivalent_material_selection_has_identical_commands(self):
        all_low, all_info = self.compile_record(7, packet_angle_step_deg=.388)
        for side in ('top', 'bottom'):
            low, info = self.compile_record(7, {'side': side, 'count': 2}, packet_angle_step_deg=.388)
            self.assertEqual(low, all_low)
            self.assertEqual(info['loading_schedule'], all_info['loading_schedule'])
            self.assertEqual(info['selected_faces'], all_info['selected_faces'])
        self.assertEqual(all_low[-1]['frames'], 464)
        self.assertAlmostEqual(all_info['loading_schedule']['nominal_duration_s'], 464 / 60)

    def test_archived_adaptive_step_is_not_fixed_464_request(self):
        low, info = self.compile_record(6, packet_angle_step_deg=.388)
        self.assertEqual(low[-1]['frames'], 360)
        self.assertEqual(self.records[6]['frame'] - self.records[5]['frame'], 464)
        self.assertEqual(info['loading_schedule']['selected_hinge_components'], 1)

    def test_partial_unfold_inherits_schedule_and_records_actual_counters(self):
        backend = RecordingBackend(self.vertices[2624], self.faces)
        ctrl = LayerControllerV16(backend, 'thickness_panel', packet_angle_step_deg=.388)
        ctrl.execute(self.records[7]['action'])
        # Changing a later configuration must not change an existing fold's reverse policy.
        ctrl.packet_angle_step_deg = None
        ctrl.execute(dict(op='unfold', angle_deg=90))
        schedule = ctrl.records[-1]['details']['loading_schedule']
        self.assertEqual(schedule['requested_frames'], 232)
        self.assertEqual(schedule['actual_integrator_segments'], 232)
        self.assertAlmostEqual(schedule['actual_duration_s'], 232 / 60)
        self.assertEqual(ctrl.fold_stack[0]['degrees'], -90)
        ctrl.execute(dict(op='unfold'))
        self.assertEqual(ctrl.records[-1]['low_level_actions'][-1]['frames'], 232)
        self.assertFalse(ctrl.fold_stack)

if __name__ == '__main__': unittest.main()
