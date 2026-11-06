"""Meaningful positive/negative controls; no physics or model invocation."""
import unittest
import numpy as np
from origami_bench.diagram.grouped_precrease_grading import grade_grouped_precrease


class GroupedPrecreaseContractTests(unittest.TestCase):
    def setUp(self):
        self.initial=np.array([[0.,0.,0.],[.15,0.,0.],[.13,.12,0.],[.01,.10,0.],[.04,.07,0.]])
        self.a=self.initial.copy();self.a[0,2]=.025
        self.b=self.initial.copy();self.b[1,2]=.035
        self.c=self.initial.copy();self.c[2,2]=.045
        self.groups=[dict(id='paired_arrows',targets=[self.a,self.b]),dict(id='next_panel',targets=[self.c])]
        self.frames=np.array([self.initial,self.a,self.b,self.c])

    def grade(self,frames=None,groups=None,**kwargs):
        params=dict(physical_pass=True,memory_pass=True,finished=True,environment_error=False)
        params.update(kwargs)
        return grade_grouped_precrease(self.frames if frames is None else frames,0,self.initial,self.c,
                                       self.groups if groups is None else groups,**params)

    def test_reference_and_within_group_reverse_order_pass(self):
        self.assertTrue(self.grade()['task_success'])
        reversed_order=self.grade(np.array([self.initial,self.b,self.a,self.c]))
        self.assertTrue(reversed_order['task_success'])
        self.assertEqual([m['target'] for m in reversed_order['diagram_path']['matches'][:2]],[1,0])

    def test_actual_decomposition_and_extra_nonmatching_states_pass(self):
        frames=np.array([self.initial,(self.initial+self.a)/2,self.a,self.initial,
                         (self.initial+self.b)/2,self.b,self.initial,self.c])
        self.assertTrue(self.grade(frames)['task_success'])

    def test_fixed_yaw_and_per_frame_global_xyz_placement_pass(self):
        angle=.38;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        frames=self.frames.copy();frames[:,:,:2]=frames[:,:,:2]@r
        frames+=np.array([[.1,-.02,.04],[-.3,.04,.1],[.03,-.01,-.2],[.2,.15,.06]])[:,None,:]
        self.assertTrue(self.grade(frames)['task_success'])

    def test_reversed_stage_or_skipped_target_fails(self):
        wrong_order=np.array([self.initial,self.c,self.a,self.b,self.c+.0])
        # Returning to c later is a valid decomposition, so do not reject it.
        self.assertTrue(self.grade(wrong_order)['task_success'])
        no_return=np.array([self.initial,self.c,self.a,self.b])
        result=self.grade(no_return)
        self.assertFalse(result['diagram_path']['passed'])
        self.assertEqual(result['diagram_path']['first_incomplete_group'],1)
        self.assertFalse(self.grade(np.array([self.initial,self.a,self.c]))['task_success'])

    def test_static_padding_cannot_complete_coverage(self):
        frames=np.array([self.initial]+[self.a]*30+[self.c])
        result=self.grade(frames)
        self.assertFalse(result['task_success'])
        self.assertEqual(result['diagram_path']['matched'],1)

    def test_duplicate_or_near_duplicate_targets_fail_contract(self):
        for target in [self.a,self.a+np.array([0,0,.0001])]:
            result=self.grade(groups=[dict(targets=[self.a,target]),dict(targets=[self.c])])
            self.assertFalse(result['task_success'])
            self.assertFalse(result['diagram_path']['contract_valid'])
        result=self.grade(groups=[dict(targets=[self.a]),dict(targets=[self.a]),dict(targets=[self.c])])
        self.assertEqual(result['diagram_path']['reason'],'overlapping_target_acceptance_regions')

    def test_prepared_state_is_not_an_extra_motion_target(self):
        result=self.grade(groups=[dict(targets=[self.initial]),dict(targets=[self.c])])
        self.assertFalse(result['diagram_path']['contract_valid'])
        self.assertEqual(result['diagram_path']['reason'],'target_duplicates_prepared_state')

    def test_final_structure_is_separate_from_process(self):
        wrong=self.c.copy();wrong[3,0]+=.03
        result=self.grade(np.array([*self.frames,wrong]))
        self.assertTrue(result['diagram_path']['passed'])
        self.assertFalse(result['structure_success'])
        self.assertFalse(result['task_success'])

    def test_wrong_side_reflection_and_scale_do_not_pass(self):
        wrong_side=self.c.copy();wrong_side[:,2]*=-1
        self.assertFalse(self.grade(np.array([self.initial,self.a,self.b,wrong_side]))['task_success'])
        mirror=self.frames.copy();mirror[:,:,0]*=-1
        self.assertFalse(self.grade(mirror)['task_success'])
        self.assertFalse(self.grade(self.frames*1.2)['task_success'])

    def test_unknown_physics_signed_memory_or_missing_finish_fails(self):
        for changes in [dict(physical_pass=None),dict(memory_pass=False),dict(memory_pass=None),
                        dict(finished=False),dict(environment_error=True),dict(environment_error=None)]:
            result=self.grade(**changes)
            self.assertTrue(result['structure_success'])
            self.assertTrue(result['diagram_path']['passed'])
            self.assertFalse(result['task_success'])

    def test_nan_empty_malformed_evidence_fail_closed(self):
        nan=self.frames.copy();nan[2,0,0]=np.nan
        for frames in [nan,[],np.zeros((2,3)),self.frames[:,:2,:]]:
            result=self.grade(frames)
            self.assertFalse(result['task_success'])
            self.assertFalse(result['input_valid'])
        for groups in [[],[{}],[dict(targets=[])],[dict(targets=[self.a[:2]])]]:
            self.assertFalse(self.grade(groups=groups)['task_success'])


if __name__=='__main__':unittest.main()
