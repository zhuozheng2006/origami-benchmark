"""Planning-only fixture tests; the fake backend never simulates physics."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from origami_bench.diagram import layer_controller_v16 as controller

class PlanningComplete(Exception):pass

class CaptureBackend:
    def __init__(self,vertices):
        self.paper=dict(vertices=vertices.tolist(),faces=[[0,1,2],[3,4,5]],hash='planning-only-unchanged')
        self.commands=[]
    def state(self):return copy.deepcopy(self.paper)
    def step(self,command):
        self.commands.append(copy.deepcopy(command))
        if command['op'] in ('rotate_grips','rotate_soft_grips'):raise PlanningComplete()
        return self.state()

def fixture(angle=0,components=2,seam=True,degenerate_pair=False):
    v=np.array([[-.030,-.015,0],[-.0275,.004,0],[-.028,.025,0],
                [-.030,-.014,.0008],[-.0275,.005,.0008],[-.028,.026,.0008],
                [.03,-.02,0],[.03,.02,0]])
    if degenerate_pair:v[0,0]=v[2,0]=-.010
    a=np.deg2rad(angle);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    v=v@r.T;axis=r@np.array([0.,1.,0.]);center=[0.,0.,.0001]
    moving=list(range(6)) if components==2 else list(range(3))
    groups=[list(range(3)),list(range(3,6))] if components==2 else [list(range(3))]
    info=dict(axis_direction=axis.tolist(),axis_point=center,selected_planar_components=components,
              has_multiple_surface_orientations=components>1,crosses_folded_seam=seam,
              moving_vertices=moving,fixed_vertices=[6,7],moving_panel_groups=groups,
              free_crease_band_halfwidth_m=.001)
    cmd=dict(op='rotate_grips',vertices=moving,axis_point=center,axis_direction=axis.tolist(),degrees=-180,frames=360)
    low=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=moving+[6,7]),cmd]
    return v,low,info

class DirectionalFixtureTests(unittest.TestCase):
    def plan(self,mode,**kwargs):
        v,low,info=fixture(**kwargs);backend=CaptureBackend(v)
        with patch.object(controller,'compile_stack_fold_v15',return_value=(low,info)):
            with self.assertRaises(PlanningComplete):controller.LayerControllerV16(backend,mode).execute(dict(op='fold_line'))
        return [c for c in backend.commands if c['op']!='undo'],info

    def test_legacy_cross_seam_commands_unchanged(self):
        commands,info=self.plan('guided_directional_10000')
        v,low,_=fixture();rotate=dict(low[-1],op='rotate_soft_grips')
        expected=[dict(op='release_soft_grips'),dict(op='release'),dict(op='set_grips',vertices=[6,7]),
                  dict(op='attach_soft_grips',vertices=list(range(6)),stiffness=1000.,normal=[0.,0.,1.]),
                  dict(op='attach_soft_grips',vertices=[1],stiffness=10000.),rotate]
        self.assertEqual(commands,expected)
        self.assertNotIn('experimental_fixture_mode',info)
        self.assertNotIn('paired_base_anchor_vertices',info)

    def test_all_axes_only_changes_activation_not_force_or_anchor(self):
        old,old_info=self.plan('guided_directional_10000',components=1,seam=False)
        new,info=self.plan('guided_all_directional_10000',components=1,seam=False)
        self.assertEqual(old[-1]['op'],'rotate_grips')
        self.assertEqual(new[-1]['op'],'rotate_soft_grips')
        self.assertEqual(new[-3]['stiffness'],1000.)
        self.assertEqual(new[-2],dict(op='attach_soft_grips',vertices=[1],stiffness=10000.))
        self.assertEqual(info['experimental_fixture_mode'],'guided_all_directional_10000')
        self.assertNotIn('paired_base_anchor_vertices',info)
        existing,_=self.plan('guided_directional_10000')
        always,_=self.plan('guided_all_directional_10000')
        self.assertEqual(existing,always)

    def test_pair_only_adds_one_anchor_within_same_base_panel(self):
        old,_=self.plan('guided_directional_10000')
        paired,info=self.plan('guided_pair_directional_10000')
        expected=copy.deepcopy(old);expected[-2]['vertices']=[1,2]
        self.assertEqual(paired,expected)
        self.assertEqual(info['paired_base_anchor_vertices'],[1,2])
        self.assertAlmostEqual(info['paired_anchor_axis_separation_m'],.021)
        self.assertNotIn(3,paired[-2]['vertices'])

    def test_pair_does_not_enable_directional_branch_on_legacy_inactive_axis(self):
        old,_=self.plan('guided_directional_10000',components=1,seam=False)
        pair,info=self.plan('guided_pair_directional_10000',components=1,seam=False)
        self.assertEqual(old,pair)
        self.assertNotIn('paired_base_anchor_vertices',info)

    def test_pair_preserves_other_legacy_panel_anchor(self):
        old,_=self.plan('guided_directional_10000',seam=False)
        pair,_=self.plan('guided_pair_directional_10000',seam=False)
        self.assertEqual(old[-2]['vertices'],[1,4])
        self.assertEqual(pair[-2]['vertices'],[1,2,4])

    def test_planar_rotations_keep_material_anchor_choice_and_spacing(self):
        for degrees in (0,17,45,90,137,180,270):
            with self.subTest(degrees=degrees):
                pair,info=self.plan('guided_pair_directional_10000',angle=degrees)
                self.assertEqual(pair[-2]['vertices'],[1,2])
                self.assertAlmostEqual(info['paired_anchor_axis_separation_m'],.021)
                all_axes,_=self.plan('guided_all_directional_10000',angle=degrees,components=1,seam=False)
                self.assertEqual(all_axes[-1]['op'],'rotate_soft_grips')
                self.assertEqual(all_axes[-2]['vertices'],[1])

    def test_pair_without_separation_refuses_before_backend_mutation(self):
        v,low,info=fixture(degenerate_pair=True);backend=CaptureBackend(v)
        with patch.object(controller,'compile_stack_fold_v15',return_value=(low,info)):
            with self.assertRaisesRegex(controller.SkillUnsupported,'separated far anchors'):
                controller.LayerControllerV16(backend,'guided_pair_directional_10000').execute(dict(op='fold_line'))
        self.assertEqual(backend.commands,[])

    def test_unknown_and_combined_modes_are_rejected(self):
        for mode in ('guided_all_directional_500','guided_pair_directional_10000_extra','guided_all_pair_directional_10000','unknown'):
            backend=CaptureBackend(fixture()[0])
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'Unknown fixture mode'):
                controller.LayerControllerV16(backend,mode).execute(dict(op='fold_line'))
            self.assertEqual(backend.commands,[])

if __name__=='__main__':unittest.main()
