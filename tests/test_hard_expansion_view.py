"""Static viewer export controls: no physical worker, API, or browser required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_hard_expansion_view as view


class HardViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.artifact, self.runs, self.public = [self.base/x for x in ['artifact', 'runs', 'public']]
        for p in [self.artifact, self.runs, self.public]: p.mkdir()
        self.catalog = self.artifact/'candidate-catalog.json'
        self.catalog.write_text(json.dumps({'candidates':[dict(id='C1', title='Candidate', source=None,
            source_group='family', scope='real_source_precreasing_subtask', printed_steps='1–3')]}))
    def create_run(self, name, negative=None, score=None, finalized=True, sequence_pass=True, run_id=None):
        d=self.runs/name;d.mkdir()
        report=dict(task_id='C1', run_id=run_id or name, candidate_reference_pass=sequence_pass,
                    physical_pass=True, accepted_reference_actions=1, expected_reference_actions=1,
                    finished=True, annotated_plan_complete=True, task_success=False)
        (d/'report.json').write_text(json.dumps(report))
        (d/'candidate-public.json').write_text(json.dumps({'id':'C1'}))
        (d/'candidate-private.json').write_text(json.dumps(
            {'control_kind':negative,'expected_task_score':False} if negative else {}))
        (d/'reference-attempts.json').write_text(json.dumps([dict(attempt_id=1, action={'op':'fold_line'},
             before_frame=0, after_frame=123, result={'status':'accepted'})]))
        if finalized:
            (d/'qualification-complete.json').write_text(json.dumps(dict(run_id=report['run_id'],report_sha256=view.sha(d/'report.json'))))
        if score is not None:
            (d/'precrease-grading.json').write_text(json.dumps(dict(task_id='C1', task_success=score,
               physical_pass=True, finished=True, environment_error=False, memory_pass=score,
               grading_contract='precrease-artifact-v4', grading_version='version1', memory_diagnostics={})))
        return d
    def build(self):
        return view.build(self.catalog,self.artifact,self.runs,self.public,self.public/'hard-v4',self.base/'missing.sqlite3')
    def approve_summary(self):
        p=self.artifact/'precrease-scoring';p.mkdir(exist_ok=True)
        (p/'summary.json').write_text(json.dumps({'tasks':{'C1':{'scoring_acceptance_pass':True}}}))
    def full_controls(self):
        self.create_run('reference-C1-01',score=True)
        self.create_run('alternate-C1-02',score=True)
        self.create_run('negative-C1-empty','negative_missing_all_new_creases',False)
        self.create_run('negative-C1-sign','negative_wrong_sign',False)
    def test_negative_sequence_success_never_counts_reference_or_usable_task(self):
        self.create_run('negative-C1-empty','negative_missing_all_new_creases')
        result=self.build();run=result['tasks'][0]['runs'][0]
        self.assertTrue(run['physical_sequence_pass'])
        self.assertTrue(run['recorded_candidate_reference_pass'])
        self.assertFalse(run['candidate_reference_pass'])
        self.assertIsNone(run['task_success'])
        self.assertEqual(result['counts']['passed_reference_runs'],0)
        self.assertEqual(result['counts']['scoring_validated_candidates'],0)
        self.assertIn('负对照',run['reason'])
    def test_two_positive_physical_runs_still_pending_without_scores_and_controls(self):
        self.create_run('reference-C1-01');self.create_run('alternate-C1-02');self.approve_summary()
        result=self.build();task=result['tasks'][0]
        self.assertEqual(result['counts']['passed_reference_runs'],2)
        self.assertEqual(task['status_type'],'pending')
        self.assertFalse(task['candidate_scoring_validated'])
        self.assertIn('待评分验收',task['status_label'])
    def test_actual_task_scores_and_both_negative_controls_required_with_summary(self):
        self.full_controls()
        self.assertFalse(self.build()['tasks'][0]['candidate_scoring_validated'])
        self.approve_summary();result=self.build()
        self.assertTrue(result['tasks'][0]['candidate_scoring_validated'])
        self.assertEqual(result['counts']['passed_reference_runs'],2)
        self.assertEqual(result['counts']['correct_negative_controls'],2)
        self.assertEqual(result['counts']['scoring_validated_candidates'],1)
        self.assertEqual(result['counts']['failed_finalized_runs'],0)
    def test_incorrect_negative_score_blocks_acceptance(self):
        self.full_controls();self.approve_summary()
        p=self.runs/'negative-C1-sign/precrease-grading.json';x=json.loads(p.read_text());x['task_success']=True;p.write_text(json.dumps(x))
        result=self.build()
        self.assertFalse(result['tasks'][0]['candidate_scoring_validated'])
        self.assertEqual(result['counts']['incorrect_negative_controls'],1)
        self.assertEqual(result['counts']['passed_reference_runs'],2)
    def test_missing_marker_and_duplicate_positive_run_ids_block_acceptance(self):
        self.create_run('reference-C1-01',score=True,run_id='same')
        self.create_run('alternate-C1-02',score=True,run_id='same')
        self.create_run('negative-C1-empty','negative_missing_all_new_creases',False)
        d=self.create_run('negative-C1-sign','negative_wrong_sign',False,finalized=False)
        self.approve_summary()
        result=self.build();self.assertFalse(result['tasks'][0]['candidate_scoring_validated'])
        run=next(r for r in result['tasks'][0]['runs'] if r['id']==d.name)
        self.assertIsNone(run['task_success']);self.assertFalse(run['report_finalized'])
    def test_different_grading_contracts_cannot_be_mixed(self):
        self.full_controls();self.approve_summary()
        p=self.runs/'alternate-C1-02/precrease-grading.json';x=json.loads(p.read_text());x['grading_contract']='other';p.write_text(json.dumps(x))
        task=self.build()['tasks'][0]
        self.assertFalse(task['candidate_scoring_validated'])
        self.assertFalse(task['scoring_acceptance']['consistent_grading_contract'])
    def test_sampling_preserves_all_endpoints_and_does_not_change_original_evidence(self):
        d=self.create_run('reference-C1-01',score=True)
        v=np.zeros((1000,3,3));v[:,1,0]=.1;v[:,2,1]=.1
        np.savez_compressed(d/'trajectory.npz',vertices=v,faces=np.array([[0,1,2]]))
        original={p.name:p.read_bytes() for p in d.iterdir()}
        result=self.build();run=result['tasks'][0]['runs'][0]
        trace=json.loads((self.public/'hard-v4'/run['trajectory']).read_text())
        self.assertIn(123,trace['frame_numbers']);self.assertLessEqual(len(trace['vertices']),102)
        self.assertEqual(trace['total_frames'],1000)
        self.build()
        self.assertEqual(original,{p.name:p.read_bytes() for p in d.iterdir()})


if __name__=='__main__': unittest.main()
