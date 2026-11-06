import json
import sqlite3
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import run_frozen_pilot as pilot
from origami_bench.diagram.cost_book import CostBook


class FrozenPilotTests(unittest.TestCase):
    def test_failed_response_identity_counts_missing_and_corrupt_records(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for call in (1,2,3):
                (folder/f'call-{call:05d}-request.json').write_text('{}')
            (folder/'call-00001-response.json').write_text(json.dumps({'response':{'model':'actual-model'}}))
            (folder/'call-00002-response.json').write_text('{broken')
            value=pilot.response_identity_evidence(folder)
            self.assertEqual(value['returned_models'],['actual-model'])
            self.assertEqual(value['returned_model_missing_calls'],2)
            self.assertEqual(len(value['response_identity_issues']),2)

    def prepare(self, folder):
        ledger=folder/'shared.sqlite3'
        book=CostBook(ledger,150,140);book.db.close()
        cfg=dict(tasks=['O01','O04'],models=[dict(slug='modela',model='example',
                 cpu_group=4,endpoint='https://aicodelink.top/v1')],stage_cap_cny=30,
                 operating_cap_cny=140,cumulative_cap_cny=150,physics_signature={'test':True})
        path=folder/'config.json';path.write_text(json.dumps(cfg))
        return ledger,path,cfg

    def test_predeclares_unique_runs_diagnostics_and_binds_shared_ledger(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,_=self.prepare(folder);out=folder/'study'
            pilot.freeze(config,out,ledger)
            plan=json.loads((out/'modela/experiment-plan.json').read_text())
            self.assertEqual([r['task_id'] for r in plan['runs']],['O01','O04'])
            self.assertEqual(len({r['run_id'] for r in plan['runs']}),2)
            frozen=json.loads((out/'study-freeze.json').read_text())
            self.assertEqual(frozen['ledger_path'],str(ledger.resolve()))
            self.assertEqual(frozen['ledger_accounting_sha256'],pilot.audit(ledger)['accounting_sha256'])
            # A changed ledger, even with otherwise valid plans, must fail before
            # any credential read or start marker: no blank replacement budget.
            book=CostBook(ledger,150,140);book.reserve('other','other',1);book.db.close()
            with self.assertRaisesRegex(ValueError,'Shared ledger'):
                pilot.execute(out,folder/'never-read-secrets.json',ledger)
            self.assertFalse((out/'execution-start.json').exists())

    def test_new_accounting_evidence_blocks_execution_with_unchanged_call_rows(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,_=self.prepare(folder);out=folder/'study'
            pilot.freeze(config,out,ledger)
            before=pilot.audit(ledger)
            # Appended evidence is invisible to the legacy calls-only digest.
            with sqlite3.connect(ledger) as db:
                db.execute('CREATE TABLE provider_evidence (evidence_id TEXT PRIMARY KEY, data TEXT)')
                db.execute("INSERT INTO provider_evidence VALUES ('new-row','verified provider charge')")
            self.assertEqual(pilot.audit(ledger)['calls_sha256'],before['calls_sha256'])
            with self.assertRaisesRegex(ValueError,'accounting evidence changed'):
                pilot.execute(out,folder/'never-read-secrets.json',ledger)
            self.assertFalse((out/'execution-start.json').exists())

    def test_old_frozen_study_is_preserved_and_requires_fresh_freeze(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,_=self.prepare(folder);out=folder/'study'
            pilot.freeze(config,out,ledger)
            path=out/'study-freeze.json';value=json.loads(path.read_text())
            value.pop('ledger_accounting_sha256');value['schema']='frozen-pilot-v1'
            path.write_text(json.dumps(value));old=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'Legacy frozen study'):
                pilot.execute(out,folder/'never-read-secrets.json',ledger)
            self.assertEqual(path.read_bytes(),old)
            self.assertFalse((out/'execution-start.json').exists())

    def test_changed_budget_limits_fail_before_credentials(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,_=self.prepare(folder);out=folder/'study'
            pilot.freeze(config,out,ledger)
            with sqlite3.connect(ledger) as db:
                db.execute('CREATE TABLE budget_policy (id INTEGER PRIMARY KEY, operating_ceiling REAL)')
                db.execute('INSERT INTO budget_policy VALUES (1,139)')
            with self.assertRaisesRegex(ValueError,'budget ceilings changed'):
                pilot.execute(out,folder/'never-read-secrets.json',ledger)
            self.assertFalse((out/'execution-start.json').exists())

    def test_invalid_caps_or_missing_ledger_never_create_study(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,cfg=self.prepare(folder)
            for stage in (float('nan'),151,True,-1):
                cfg['stage_cap_cny']=stage;config.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):pilot.freeze(config,folder/'study',ledger)
                self.assertFalse((folder/'study').exists())
            cfg['stage_cap_cny']=30;config.write_text(json.dumps(cfg))
            with self.assertRaises(Exception):pilot.freeze(config,folder/'study',folder/'missing.sqlite3')
            self.assertFalse((folder/'missing.sqlite3').exists())

    def test_precrease_suite_and_independent_repeat_are_declared_before_execution(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'verify'):
            folder=Path(d);ledger,config,cfg=self.prepare(folder)
            cfg.update(tasks=['H102','H103'],grading_contract='precrease-artifact-v4',repeat_id=2)
            config.write_text(json.dumps(cfg));out=folder/'study'
            pilot.freeze(config,out,ledger)
            plan=json.loads((out/'modela/experiment-plan.json').read_text())
            self.assertEqual(plan['suite'],'precrease-hard')
            self.assertEqual([r['repeat_id'] for r in plan['runs']],[2,2])
            for invalid in (0,-1,True,1.5):
                cfg['repeat_id']=invalid;config.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):pilot.freeze(config,folder/'bad-study',ledger)
                self.assertFalse((folder/'bad-study').exists())


if __name__=='__main__':unittest.main()
